//! v69-fit: initialization, qualification, fixed fits and endpoint evaluation.
//!
//!   v69-fit init              --artifacts DIR [--run gen-001] [--seed-file seed/master_seed.hex]
//!   v69-fit make-intervention --artifacts DIR
//!   v69-fit qualify           --artifacts DIR [--arms A,B,C]
//!   v69-fit fit    --arm A    --artifacts DIR
//!   v69-fit eval   --arm A    --artifacts DIR
//!
//! CUDA FP32 only. Dataset files are opened exclusively through the
//! role-restricted `Access` (learner / evaluator); the withheld dataset parts
//! are not reachable from this binary.

use anyhow::{Context, Result, bail, ensure};
use burn::backend::cuda::CudaDevice;
use burn::prelude::*;
use recur64_v69::access::{Access, Role};
use recur64_v69::custody::Custody;
use recur64_v69::features::Features;
use recur64_v69::metrics::PredRow;
use recur64_v69::provenance::{sha256_hex, source_id};
use recur64_v69::streams::{MasterSeed, STREAM_INTERVENTION, STREAM_MODEL_INIT};
use recur64_v69_model::data::{Loaded, load_split, manifest_hash};
use recur64_v69_model::init::{self, InitHeader, InitTensor};
use recur64_v69_model::inventory::{dump_values, inventory, load_values};
use recur64_v69_model::model::{Arm, Batch, Model};
use recur64_v69_model::qualify::{G, GA, QualCtx, qualify_arm};
use recur64_v69_model::train::{ACCUM, MICRO, Micro, TOTAL_UPDATES, Trainer, load_model_for_eval, sample_order, save_checkpoint};
use serde_json::json;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Git head that produced the accepted gen-001 data (recorded in the data-phase
/// evidence under docs/v69): the DATA PRODUCER, distinct from the consumer source id.
const DATA_PRODUCER_GIT_HEAD: &str = "36a81508b456ede1cb682f2f03fe678fd08db70f";

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

struct Env {
    custody: Custody,
    run: PathBuf,
    seed_rel: PathBuf,
}

impl Env {
    fn from(args: &[String]) -> Result<Self> {
        let custody = Custody::new(Path::new(&arg(args, "--artifacts").context("--artifacts")?))?;
        Ok(Self {
            custody,
            run: PathBuf::from(arg(args, "--run").unwrap_or_else(|| "gen-001".into())),
            seed_rel: PathBuf::from(arg(args, "--seed-file").unwrap_or_else(|| "seed/master_seed.hex".into())),
        })
    }
    fn access(&self, role: Role) -> Result<Access> {
        Access::new(&self.custody, role, &self.run, &self.seed_rel)
    }
    fn seed(&self, a: &Access) -> Result<MasterSeed> {
        MasterSeed::from_hex(String::from_utf8(a.read(&self.seed_rel)?)?.trim())
    }
}

fn cuda_device() -> Result<CudaDevice> {
    let d = CudaDevice::default();
    // Visible failure if the CUDA runtime/device is unavailable: never substitute CPU.
    G::sync(&d).map_err(|e| anyhow::anyhow!("CUDA device unavailable (no CPU fallback): {e:?}"))?;
    Ok(d)
}

fn read_init(a: &Access) -> Result<(String, Vec<(Vec<usize>, Vec<f32>)>, String)> {
    let bytes = a.read(Path::new("init/canonical_init.bin"))?;
    let file_sha = sha256_hex(&bytes);
    let (_, vals) = init::from_bytes(&bytes)?;
    let th = init::tensors_hash(&vals);
    Ok((file_sha, vals, th))
}

// ---------------------------------------------------------------- init

fn cmd_init(args: &[String]) -> Result<()> {
    let env = Env::from(args)?;
    let a = env.access(Role::Learner)?;
    let seed = env.seed(&a)?;
    let dev = cuda_device()?;
    let m = Model::<G>::new(&dev);
    let inv = inventory::<G, _>(&m);
    let total: usize = inv.iter().map(|p| p.numel).sum();
    ensure!(total <= 4_000_000, "parameter ceiling exceeded: {total}");
    let vals = init::generate(&seed, STREAM_MODEL_INIT, &inv);
    let header = InitHeader {
        version: 1,
        stream_label: STREAM_MODEL_INIT.to_string(),
        seed_fingerprint: seed.fingerprint(),
        tensors: inv.iter().map(|p| InitTensor { path: p.path.clone(), leaf: p.leaf.clone(), shape: p.shape.clone() }).collect(),
    };
    let bytes = init::to_bytes(&header, &vals);
    let target = Path::new("init/canonical_init.bin");
    ensure!(!a.custody().resolve(target)?.exists(), "canonical init exists; never regenerated");
    a.write(target, &bytes)?;
    let mut by_comp: BTreeMap<String, usize> = BTreeMap::new();
    for p in &inv {
        *by_comp.entry(p.component.clone()).or_default() += p.numel;
    }
    let info = json!({
        "file_sha256": sha256_hex(&bytes),
        "tensors_sha256": init::tensors_hash(&vals),
        "stream": format!("{STREAM_MODEL_INIT}/0"),
        "seed_fingerprint": seed.fingerprint(),
        "total_parameters": total,
        "ceiling": 4_000_000,
        "by_component": by_comp,
        "decay_parameters": inv.iter().filter(|p| p.decay).map(|p| p.numel).sum::<usize>(),
        "nodecay_parameters": inv.iter().filter(|p| !p.decay).map(|p| p.numel).sum::<usize>(),
        "tensor_inventory": inv,
    });
    a.write(Path::new("spec/param_inventory.json"), serde_json::to_string_pretty(&info)?.as_bytes())?;
    println!("parameters={total} tensors={} file_sha256={}", inv.len(), sha256_hex(&bytes));
    println!("{}", serde_json::to_string_pretty(&by_comp)?);
    Ok(())
}

// ---------------------------------------------------------------- intervention

fn cmd_make_intervention(args: &[String]) -> Result<()> {
    let env = Env::from(args)?;
    let a = env.access(Role::Evaluator)?;
    let seed = env.seed(&a)?;
    let mut out: BTreeMap<&str, BTreeMap<String, String>> = BTreeMap::new();
    let mut cells_report = BTreeMap::new();
    for part in ["fit", "val"] {
        let l = load_split(&a, &env.run, part)?;
        // Labels are never touched here: the map depends only on (family, budget, id).
        let mut cells: BTreeMap<(String, u8), Vec<String>> = BTreeMap::new();
        for (r, fam) in l.rows.iter().zip(&l.fams) {
            cells.entry((fam.name().to_string(), r.budget)).or_default().push(r.id.clone());
        }
        let mut map = BTreeMap::new();
        for ((fam, b), mut ids) in cells {
            ids.sort();
            let mut rng = seed.stream(&format!("{STREAM_INTERVENTION}/{part}/{fam}/{b}"), 0);
            for i in (1..ids.len()).rev() {
                ids.swap(i, rng.below(i as u64 + 1) as usize);
            }
            let n = ids.len();
            ensure!(n >= 2, "cell too small to derange");
            for i in 0..n {
                map.insert(ids[i].clone(), ids[(i + 1) % n].clone()); // cyclic shift of a shuffled order
            }
            cells_report.insert(format!("{part}/{fam}/n{b}"), n);
        }
        ensure!(map.len() == l.rows.len(), "map must cover every row");
        let donors: std::collections::HashSet<&String> = map.values().collect();
        ensure!(donors.len() == map.len(), "donor map must be a bijection");
        ensure!(map.iter().all(|(k, v)| k != v), "fixed point in derangement");
        out.insert(part, map);
    }
    let body = json!({"derangement": "cyclic shift of a seeded shuffle within (family, budget) cells, per partition; label-independent", "stream": format!("{STREAM_INTERVENTION}/<partition>/<family>/<budget>"), "seed_fingerprint": seed.fingerprint(), "cells": cells_report, "fit": out["fit"], "val": out["val"]});
    let bytes = serde_json::to_string_pretty(&body)?.into_bytes();
    let target = Path::new("intervention/map.json");
    ensure!(!a.custody().resolve(target)?.exists(), "intervention map already frozen");
    a.write(target, &bytes)?;
    a.write(Path::new("intervention/map.sha256"), sha256_hex(&bytes).as_bytes())?;
    println!("intervention map sha256 {}", sha256_hex(&bytes));
    Ok(())
}

// ---------------------------------------------------------------- qualify

fn cmd_qualify(args: &[String]) -> Result<()> {
    let env = Env::from(args)?;
    let a = env.access(Role::Learner)?;
    let seed = env.seed(&a)?;
    let _dev = cuda_device()?;
    let fit = load_split(&a, &env.run, "fit")?;
    // fixed fitting-only panel: 16 rows at stride 48 of the id-sorted fit rows
    let mut order: Vec<usize> = (0..fit.rows.len()).collect();
    order.sort_by(|&x, &y| fit.rows[x].id.cmp(&fit.rows[y].id));
    let pick: Vec<usize> = (0..16).map(|k| order[k * 48]).collect();
    let (_file_sha, sci_vals, sci_hash) = read_init(&a)?;
    let ctx = QualCtx {
        rows: pick.iter().map(|&i| fit.rows[i].clone()).collect(),
        feats: pick.iter().map(|&i| fit.feats[i].clone()).collect(),
        seed,
        scientific_init_values: sci_vals,
        scientific_init_hash: sci_hash,
    };
    let arms: Vec<Arm> = arg(args, "--arms").unwrap_or_else(|| "A,B,C".into()).split(',').map(Arm::parse).collect::<Result<_>>()?;
    let mut all = Vec::new();
    let mut ok = true;
    for arm in arms {
        let ckpt = a.output_path(Path::new(&format!("qual/{}/ckpt/x", arm.name())))?.parent().unwrap().to_path_buf();
        let _ = std::fs::remove_dir_all(&ckpt);
        let rep = qualify_arm(&ctx, arm, &ckpt);
        ok &= rep["qualified"].as_bool().unwrap_or(false);
        a.write(Path::new(&format!("qual/qual_report_{}.json", arm.name())), serde_json::to_string_pretty(&rep)?.as_bytes())?;
        all.push(rep);
    }
    let summary = json!({
        "all_qualified": ok,
        "panel_ids": ctx.rows.iter().map(|r| r.id.clone()).collect::<Vec<_>>(),
        "source": source_id(),
        "weights": "disposable (qualification_init stream); never used to initialize scientific fits",
        "arms": all,
    });
    a.write(Path::new("qual/qual_summary.json"), serde_json::to_string_pretty(&summary)?.as_bytes())?;
    println!("QUALIFIED={ok}");
    if !ok {
        std::process::exit(5);
    }
    Ok(())
}

// ---------------------------------------------------------------- fit

fn provenance(a: &Access, env: &Env, fit: &Loaded, init_file_sha: &str, init_tensors: &str, arm: Arm) -> Result<serde_json::Value> {
    let spec = a.read(Path::new("spec/MODEL_SPEC.md"))?;
    let frozen = a.read(Path::new("spec/frozen.json"))?;
    Ok(json!({
        "consumer_source": source_id(),
        "data_producer_git_head": DATA_PRODUCER_GIT_HEAD,
        "dataset_dir": env.run.to_string_lossy(),
        "fit_jsonl_sha256": fit.sha256,
        "manifest_fit_hash": manifest_hash(a, &env.run, "data/fit.jsonl")?,
        "canonical_init_file_sha256": init_file_sha,
        "canonical_init_tensors_sha256": init_tensors,
        "spec_sha256": sha256_hex(&spec),
        "frozen_manifest_sha256": sha256_hex(&frozen),
        "seed_streams": {"train_order": "train_order/<epoch>, epochs concatenated", "model_init": "model_init/0"},
        "arm": arm.name(),
        "schedule": arm.schedule().iter().map(|o| format!("{o:?}")).collect::<Vec<_>>(),
        "recipe": {"updates": TOTAL_UPDATES, "microbatch": MICRO, "accumulation": ACCUM, "peak_lr": 5e-4, "final_lr": 5e-5, "warmup": 20, "wd": 1e-4, "clip": 1.0, "betas": [0.9, 0.999], "eps": 1e-8},
    }))
}

fn cmd_fit(args: &[String]) -> Result<()> {
    let env = Env::from(args)?;
    let arm = Arm::parse(&arg(args, "--arm").context("--arm")?)?;
    let a = env.access(Role::Learner)?;
    let seed = env.seed(&a)?;
    let dev = cuda_device()?;
    let out_dir = format!("fits/{}", arm.name());
    ensure!(!a.custody().resolve(Path::new(&out_dir))?.join("final").exists(), "fit for arm {} already exists; no reruns", arm.name());
    let fit = load_split(&a, &env.run, "fit")?;
    ensure!(fit.rows.len() == 768, "unexpected fit size {}", fit.rows.len());
    let (init_file_sha, vals, init_tensors) = read_init(&a)?;
    // qualification must have passed: refuse scientific fitting otherwise
    let q: serde_json::Value = serde_json::from_slice(&a.read(Path::new("qual/qual_summary.json"))?)?;
    ensure!(q["all_qualified"].as_bool() == Some(true), "qualification not passed: refusing scientific fit");
    let model = load_values::<GA, _>(Model::<GA>::new(&dev), &vals, &dev);
    let loaded_hash = init::tensors_hash(&dump_values::<GA, _>(&model));
    ensure!(loaded_hash == init_tensors, "loaded model does not match canonical init");
    let prov = provenance(&a, &env, &fit, &init_file_sha, &init_tensors, arm)?;
    let mut tr = Trainer::<GA>::new(model, arm, &dev);
    let order = sample_order(&seed, fit.rows.len(), TOTAL_UPDATES * MICRO * ACCUM);
    a.write(Path::new(&format!("{out_dir}/status.json")), json!({"status": "running", "arm": arm.name(), "provenance": prov}).to_string().as_bytes())?;
    let mut trace = Vec::new();
    let mut exposure = vec![0u32; fit.rows.len()];
    let t_all = std::time::Instant::now();
    for u in 0..TOTAL_UPDATES {
        let idx = &order[u * MICRO * ACCUM..(u + 1) * MICRO * ACCUM];
        for &i in idx {
            exposure[i] += 1;
        }
        let micros: Vec<Micro> = (0..ACCUM)
            .map(|m| Micro { feats: vec![&fit.feats[idx[2 * m]], &fit.feats[idx[2 * m + 1]]], labels: vec![fit.rows[idx[2 * m]].label, fit.rows[idx[2 * m + 1]].label] })
            .collect();
        let st = tr.step(&micros);
        println!("{}", serde_json::to_string(&st)?);
        if !(st.loss.is_finite() && st.grad_norm_pre_clip.is_finite()) {
            a.write(Path::new(&format!("{out_dir}/status.json")), json!({"status": "failed_nonfinite", "update": u}).to_string().as_bytes())?;
            bail!("non-finite loss/gradient at update {u}: stopping (INCOMPLETE, no restart)");
        }
        trace.push(st);
    }
    let wall = t_all.elapsed().as_secs_f64();
    let ckdir = a.output_path(Path::new(&format!("{out_dir}/final/x")))?.parent().unwrap().to_path_buf();
    save_checkpoint(&ckdir, &tr, prov.clone())?;
    let clipped = trace.iter().filter(|s| s.clipped).count();
    let summary = json!({
        "status": "complete", "arm": arm.name(), "completed_updates": tr.update, "wall_secs": wall,
        "clip_ops": tr.clip_calls, "clipped_updates": clipped, "clip_frequency": clipped as f64 / TOTAL_UPDATES as f64,
        "exposure": {"min": exposure.iter().min(), "max": exposure.iter().max(), "mean": exposure.iter().sum::<u32>() as f64 / exposure.len() as f64, "total_examples_seen": exposure.iter().sum::<u32>()},
        "first_loss": trace.first().map(|s| s.loss), "last_loss": trace.last().map(|s| s.loss),
        "mean_update_ms": trace.iter().map(|s| s.millis).sum::<f64>() / trace.len() as f64,
        "provenance": prov, "trace": trace,
    });
    a.write(Path::new(&format!("{out_dir}/trace.json")), serde_json::to_string(&summary)?.as_bytes())?;
    a.write(Path::new(&format!("{out_dir}/status.json")), json!({"status": "complete", "completed_updates": tr.update}).to_string().as_bytes())?;
    println!("FIT COMPLETE arm={} updates={} wall={wall:.1}s", arm.name(), tr.update);
    Ok(())
}

// ---------------------------------------------------------------- eval

fn predict(model: &Model<G>, feats: &[&Features], arm: Arm, dev: &CudaDevice) -> (Vec<Vec<f32>>, f64) {
    let mut out: Vec<Vec<f32>> = Vec::with_capacity(feats.len());
    let mut ms = 0.0;
    let mut nb = 0;
    for chunk in feats.chunks(16) {
        let t0 = std::time::Instant::now();
        let b = Batch::<G>::from_features(chunk, dev);
        let outs = model.forward(&b, arm);
        let per: Vec<Vec<f32>> = outs.into_iter().map(|o| o.into_data().to_vec::<f32>().unwrap()).collect();
        let _ = G::sync(dev);
        ms += t0.elapsed().as_secs_f64() * 1e3;
        nb += 1;
        for i in 0..chunk.len() {
            out.push(per.iter().map(|r| r[i]).collect());
        }
    }
    (out, ms / nb.max(1) as f64)
}

fn cmd_eval(args: &[String]) -> Result<()> {
    let env = Env::from(args)?;
    let arm = Arm::parse(&arg(args, "--arm").context("--arm")?)?;
    let a = env.access(Role::Evaluator)?;
    let dev = cuda_device()?;
    let ck = a.input_path(Path::new(&format!("fits/{}/final/meta.json", arm.name())))?;
    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(&ck)?)?;
    ensure!(meta["completed_updates"].as_u64() == Some(TOTAL_UPDATES as u64), "fit incomplete: refusing endpoint evaluation");
    ensure!(meta["arm"].as_str() == Some(arm.name()), "checkpoint arm mismatch");
    let model = load_model_for_eval::<G>(ck.parent().unwrap(), &dev)?;
    // intervention map (frozen before results) and its hash
    let map_bytes = a.read(Path::new("intervention/map.json"))?;
    let frozen_hash = String::from_utf8(a.read(Path::new("intervention/map.sha256"))?)?;
    ensure!(sha256_hex(&map_bytes) == frozen_hash.trim(), "intervention map hash mismatch");
    let map: serde_json::Value = serde_json::from_slice(&map_bytes)?;
    let mut summary = serde_json::Map::new();
    let mut files = BTreeMap::new();
    let mut latency = BTreeMap::new();
    for part in ["fit", "val"] {
        let l = load_split(&a, &env.run, part)?;
        let idx_of: BTreeMap<&str, usize> = l.rows.iter().enumerate().map(|(i, r)| (r.id.as_str(), i)).collect();
        let refs: Vec<&Features> = l.feats.iter().collect();
        let (real, lat) = predict(&model, &refs, arm, &dev);
        latency.insert(part, lat);
        let mut rows: Vec<PredRow> = l.rows.iter().zip(&real).map(|(r, z)| PredRow { id: r.id.clone(), mode: "real".into(), logits: z.clone(), donor_id: None }).collect();
        // derangement: recipient keeps its budget; board (and board-derived rule features) from the donor
        let mut dfeats: Vec<Features> = Vec::new();
        let mut donors: Vec<String> = Vec::new();
        for r in &l.rows {
            let did = map[part][&r.id].as_str().with_context(|| format!("no donor for {}", r.id))?.to_string();
            let di = *idx_of.get(did.as_str()).context("donor not in partition")?;
            let mut f = l.feats[di].clone();
            ensure!(f.budget == r.budget, "donor outside budget cell");
            f.budget = r.budget; // legitimate task budget preserved (identical within a cell)
            dfeats.push(f);
            donors.push(did);
        }
        let drefs: Vec<&Features> = dfeats.iter().collect();
        let (der, _) = predict(&model, &drefs, arm, &dev);
        for ((r, z), d) in l.rows.iter().zip(der).zip(donors) {
            rows.push(PredRow { id: r.id.clone(), mode: "derange".into(), logits: z, donor_id: Some(d) });
        }
        let efeats: Vec<Features> = l.feats.iter().map(|f| f.erased()).collect();
        let erefs: Vec<&Features> = efeats.iter().collect();
        let (ers, _) = predict(&model, &erefs, arm, &dev);
        for (r, z) in l.rows.iter().zip(ers) {
            rows.push(PredRow { id: r.id.clone(), mode: "erase".into(), logits: z, donor_id: None });
        }
        let text: String = rows.iter().map(|r| serde_json::to_string(r).unwrap() + "\n").collect();
        let rel = format!("eval/{}/predictions_{part}.jsonl", arm.name());
        a.write(Path::new(&rel), text.as_bytes())?;
        files.insert(rel, sha256_hex(text.as_bytes()));
        // in-process (f32) summary for the independent-aggregator cross-check
        let (mut tp, mut fnn, mut tn, mut fp, mut bce) = (0f64, 0f64, 0f64, 0f64, 0f64);
        for (r, z) in l.rows.iter().zip(&real) {
            let zf = *z.last().unwrap();
            let pos = zf > 0.0;
            match (r.label, pos) {
                (true, true) => tp += 1.0,
                (true, false) => fnn += 1.0,
                (false, false) => tn += 1.0,
                (false, true) => fp += 1.0,
            }
            let y = if r.label { 1.0f32 } else { 0.0 };
            bce += (zf.max(0.0) - zf * y + (-zf.abs()).exp().ln_1p()) as f64;
        }
        let n = l.rows.len() as f64;
        summary.insert(part.to_string(), json!({"bal_acc": 0.5 * (tp / (tp + fnn).max(1.0) + tn / (tn + fp).max(1.0)), "acc": (tp + tn) / n, "final_bce": bce / n}));
    }
    let prov = json!({
        "consumer_source": source_id(),
        "data_producer_git_head": DATA_PRODUCER_GIT_HEAD,
        "arm": arm.name(),
        "checkpoint_meta": meta,
        "intervention_map_sha256": frozen_hash.trim(),
        "prediction_files": files,
        "inference_latency_ms_per_batch16": latency,
        "device": "CUDA FP32",
    });
    a.write(Path::new(&format!("eval/{}/inprocess_summary.json", arm.name())), serde_json::to_string_pretty(&serde_json::Value::Object(summary))?.as_bytes())?;
    a.write(Path::new(&format!("eval/{}/eval_provenance.json", arm.name())), serde_json::to_string_pretty(&prov)?.as_bytes())?;
    println!("EVAL COMPLETE arm={}", arm.name());
    Ok(())
}

fn main() -> Result<()> {
    // Burn record (de)serialization recurses deeply; the default 1 MiB Windows main-thread
    // stack overflows. Run on a dedicated large-stack thread.
    std::thread::Builder::new().stack_size(512 * 1024 * 1024).spawn(real_main)?.join().map_err(|_| anyhow::anyhow!("worker thread panicked"))?
}

fn real_main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("init") => cmd_init(&args[2..]),
        Some("make-intervention") => cmd_make_intervention(&args[2..]),
        Some("qualify") => cmd_qualify(&args[2..]),
        Some("fit") => cmd_fit(&args[2..]),
        Some("eval") => cmd_eval(&args[2..]),
        Some("diag") => cmd_diag(&args[2..]),
        _ => bail!("usage: v69-fit <init|make-intervention|qualify|fit|eval> ..."),
    }
}

// ---------------------------------------------------------------- diagnostic (post-hoc, inference only)

/// Across-example signal in the head input: ||std over examples||_2 / ||mean over examples||_2
/// of the mean-pooled encoder output and of the mean-pooled final workspace, plus the logit std.
/// Compares the canonical initialization with a fitted endpoint. No training, no selection.
fn cmd_diag(args: &[String]) -> Result<()> {
    let env = Env::from(args)?;
    let a = env.access(Role::Evaluator)?;
    let dev = cuda_device()?;
    let val = load_split(&a, &env.run, "val")?;
    let refs: Vec<&Features> = val.feats.iter().collect();
    let stats = |model: &Model<G>, arm: Arm| -> serde_json::Value {
        let (mut ep, mut wp, mut lg) = (Vec::new(), Vec::new(), Vec::new());
        for chunk in refs.chunks(16) {
            let b = Batch::<G>::from_features(chunk, &dev);
            let (e, w) = model.pooled_features(&b, arm);
            let z = model.forward(&b, arm).pop().unwrap();
            ep.extend(e.into_data().to_vec::<f32>().unwrap().chunks(192).map(|c| c.to_vec()));
            wp.extend(w.into_data().to_vec::<f32>().unwrap().chunks(192).map(|c| c.to_vec()));
            lg.extend(z.into_data().to_vec::<f32>().unwrap());
        }
        let ratio = |rows: &Vec<Vec<f32>>| -> f64 {
            let n = rows.len() as f64;
            let (mut sm, mut ss) = (0f64, 0f64);
            for j in 0..192 {
                let m = rows.iter().map(|r| r[j] as f64).sum::<f64>() / n;
                let v = rows.iter().map(|r| (r[j] as f64 - m).powi(2)).sum::<f64>() / n;
                sm += m * m;
                ss += v;
            }
            (ss / sm.max(1e-30)).sqrt()
        };
        let m = lg.iter().map(|x| *x as f64).sum::<f64>() / lg.len() as f64;
        let sd = (lg.iter().map(|x| (*x as f64 - m).powi(2)).sum::<f64>() / lg.len() as f64).sqrt();
        json!({"encoder_pool_std_over_mean_norm": ratio(&ep), "workspace_pool_std_over_mean_norm": ratio(&wp), "logit_mean": m, "logit_std_over_examples": sd})
    };
    let (_, vals, _) = read_init(&a)?;
    let mut out = serde_json::Map::new();
    for arm in Arm::ALL {
        let init_model = load_values::<G, _>(Model::<G>::new(&dev), &vals, &dev);
        let ck = a.input_path(Path::new(&format!("fits/{}/final/meta.json", arm.name())))?;
        let fitted = load_model_for_eval::<G>(ck.parent().unwrap(), &dev)?;
        out.insert(arm.name().to_string(), json!({"canonical_init": stats(&init_model, arm), "fitted_update_600": stats(&fitted, arm)}));
    }
    let text = serde_json::to_string_pretty(&serde_json::Value::Object(out))?;
    a.write(Path::new("eval/diag_signal.json"), text.as_bytes())?;
    println!("{text}");
    Ok(())
}
