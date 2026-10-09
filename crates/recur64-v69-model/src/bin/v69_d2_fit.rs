//! v69-d2-fit: D2 fitting-set scaling diagnostic (CUDA only).
//!   v69-d2-fit qualify --artifacts DIR
//!   v69-d2-fit fit --model A|M --size 64|256|768 --artifacts DIR
//! Learner role: seed, data/fit.jsonl and D2 inputs only; no validation rows or withheld dataset parts.

use anyhow::{Context, Result, bail, ensure};
use burn::backend::cuda::CudaDevice;
use burn::module::AutodiffModule;
use burn::prelude::*;
use recur64_v69::access::{Access, Role};
use recur64_v69::custody::Custody;
use recur64_v69::d1::{FrozenD1, verify_group};
use recur64_v69::features::{Features, ModelRow, featurize, read_rows};
use recur64_v69::provenance::{sha256_hex, source_id};
use recur64_v69::streams::MasterSeed;
use recur64_v69_model::d1::*;
use recur64_v69_model::d1_qualify::{D1QualCtx, qualify_d1};
use recur64_v69_model::init;
use recur64_v69_model::inventory::{dump_values, inventory, load_values};
use recur64_v69_model::model::{Arm, Batch, Model};
use recur64_v69_model::qualify::{G, GA};
use serde_json::json;
use std::path::{Path, PathBuf};

const UPDATES: usize = 2400;
const SNAPSHOTS: [usize; 11] = [0, 50, 100, 200, 400, 600, 800, 1000, 1600, 2000, 2400];

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
        Ok(Self {
            custody: Custody::new(Path::new(&arg(args, "--artifacts").context("--artifacts")?))?,
            run: PathBuf::from(arg(args, "--run").unwrap_or_else(|| "gen-001".into())),
            seed_rel: PathBuf::from(arg(args, "--seed-file").unwrap_or_else(|| "seed/master_seed.hex".into())),
        })
    }
    fn access(&self) -> Result<Access> {
        Access::new(&self.custody, Role::Learner, &self.run, &self.seed_rel)
    }
}

fn cuda_device() -> Result<CudaDevice> {
    let d = CudaDevice::default();
    G::sync(&d).map_err(|e| anyhow::anyhow!("CUDA device unavailable (no CPU fallback): {e:?}"))?;
    Ok(d)
}

fn load_rows(a: &Access, rel: &str) -> Result<(Vec<ModelRow>, Vec<Features>, String)> {
    let bytes = a.read(Path::new(rel))?;
    let rows = read_rows(std::str::from_utf8(&bytes)?)?;
    let feats = rows.iter().map(|r| featurize(&r.fen, r.budget).map(|x| x.0)).collect::<Result<Vec<_>>>()?;
    Ok((rows, feats, sha256_hex(&bytes)))
}

fn read_values(a: &Access, rel: &str) -> Result<(String, Vec<(Vec<usize>, Vec<f32>)>)> {
    let bytes = a.read(Path::new(rel))?;
    let (_, vals) = init::from_bytes(&bytes)?;
    Ok((sha256_hex(&bytes), vals))
}

fn check_config(a: &Access) -> Result<String> {
    let bytes = a.read(Path::new("d2/config.json"))?;
    let c: serde_json::Value = serde_json::from_slice(&bytes)?;
    let n = &c["neural"];
    ensure!(n["updates"] == UPDATES && n["microbatch"] == D1_MICRO && n["accumulation_steps"] == D1_ACCUM, "config updates/microbatch mismatch");
    ensure!(n["peak_lr"].as_f64() == Some(recur64_v69_model::train::PEAK_LR) && n["final_lr"].as_f64() == Some(recur64_v69_model::train::FINAL_LR), "config lr mismatch");
    ensure!(n["warmup_updates"] == recur64_v69_model::train::WARMUP && n["weight_decay"].as_f64() == Some(recur64_v69_model::train::WEIGHT_DECAY) && n["grad_clip_global_norm"].as_f64() == Some(1.0), "config optimizer mismatch");
    ensure!(n["snapshot_updates"] == json!(SNAPSHOTS), "config snapshots mismatch");
    ensure!(c["sizes"] == json!([64, 256, 768]), "config sizes mismatch");
    Ok(sha256_hex(&bytes))
}

fn read_frozen(a: &Access) -> Result<FrozenD1> {
    Ok(serde_json::from_slice(&a.read(Path::new("d2/frozen_d2.json"))?)?)
}

// ---------------------------------------------------------------- qualify

fn cmd_qualify(args: &[String]) -> Result<()> {
    let env = Env::from(args)?;
    let a = env.access()?;
    let seed = MasterSeed::from_hex(String::from_utf8(a.read(&env.seed_rel)?)?.trim())?;
    let _dev = cuda_device()?;
    let (rows, feats, _) = load_rows(&a, "d2/subsets/s64_rows.jsonl")?;
    ensure!(rows.len() == 64, "s64 rows");
    let (_, e1_vals) = read_values(&a, "init/canonical_init.bin")?;
    let ctx = D1QualCtx { rows: rows[..32].to_vec(), feats: feats[..32].to_vec(), seed, e1_init_values: e1_vals };
    let mut rep = qualify_d1(&ctx);
    // D2-specific checks: scheduler boundaries, example-stream indexing
    let mut extra = Vec::new();
    let lr = |u: usize| lr_d1(u);
    let close = |x: f64, y: f64| (x - y).abs() <= 1e-12;
    let sched_ok = close(lr(0), 5e-4 / 20.0) && close(lr(19), 5e-4) && close(lr(20), 5e-4) && close(lr(1999), 5e-5) && (2000..2400).all(|u| lr(u) == 5e-5) && (21..2400).all(|u| lr(u) <= lr(u - 1) + 1e-15);
    extra.push(json!({"name": "D2:scheduler_boundaries", "pass": sched_ok, "detail": {"lr0": lr(0), "lr19": lr(19), "lr20": lr(20), "lr1999": lr(1999), "lr2000": lr(2000), "lr2399": lr(2399)}}));
    let mut order_ok = true;
    let mut detail = Vec::new();
    for n in [64usize, 256, 768] {
        let ob = a.read(Path::new(&format!("d2/subsets/s{n}_order.json")))?;
        let oj: serde_json::Value = serde_json::from_slice(&ob)?;
        let order: Vec<usize> = oj["order"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let shape = order.len() == UPDATES * 16 && oj["size"] == n && order.iter().all(|&i| i < n);
        let mut perms = true;
        for ep in 0..(order.len() / n) {
            let mut seen = vec![false; n];
            for &i in &order[ep * n..(ep + 1) * n] {
                perms &= !seen[i];
                seen[i] = true;
            }
        }
        let mut distinct_in_update = true;
        for u in 0..UPDATES {
            let s: std::collections::HashSet<usize> = order[u * 16..(u + 1) * 16].iter().copied().collect();
            distinct_in_update &= n < 16 || s.len() == 16 || (u * 16) / n != ((u + 1) * 16 - 1) / n; // duplicates only possible across an epoch boundary
        }
        order_ok &= shape && perms && distinct_in_update;
        detail.push(json!({"size": n, "shape": shape, "epochs_are_permutations": perms, "within_update_distinct_or_boundary": distinct_in_update}));
    }
    extra.push(json!({"name": "D2:example_stream_indexing", "pass": order_ok, "detail": detail}));
    let all = rep["qualified"].as_bool().unwrap_or(false) && sched_ok && order_ok;
    rep["qualified"] = json!(all);
    rep["d2_extra_checks"] = json!(extra);
    let full = json!({"source": source_id(), "result": rep});
    a.write(Path::new("d2/qual_report.json"), serde_json::to_string_pretty(&full)?.as_bytes())?;
    println!("D2 QUALIFIED={all}");
    if !all {
        std::process::exit(5);
    }
    Ok(())
}

// ---------------------------------------------------------------- fit

fn variation_ratio(rows: &[Vec<f32>]) -> f64 {
    let n = rows.len() as f64;
    let (mut sm, mut ss) = (0f64, 0f64);
    for j in 0..rows[0].len() {
        let m = rows.iter().map(|r| r[j] as f64).sum::<f64>() / n;
        ss += rows.iter().map(|r| (r[j] as f64 - m).powi(2)).sum::<f64>() / n;
        sm += m * m;
    }
    (ss / sm.max(1e-30)).sqrt()
}

#[allow(clippy::too_many_arguments)]
fn fit_model<M>(env: &Env, a: &Access, name: &str, n: usize, model: M, init_file_sha: &str, init_tensors: &str, dev: &CudaDevice, extra: &dyn Fn(&M, &[Features]) -> serde_json::Value) -> Result<()>
where
    M: DiagModel<GA> + AutodiffModule<GA>,
    M::InnerModule: DiagModel<G>,
{
    let out = format!("d2/fits/{name}/s{n}");
    ensure!(!a.custody().resolve(Path::new(&out))?.join("provenance.json").exists(), "D2-{name} N={n} already run; no reruns");
    let frozen = read_frozen(a)?;
    let nv = verify_group(a, &env.run, &frozen, "learner")?;
    eprintln!("[d2 fit {name} N={n}] {nv} frozen hashes verified");
    let cfg_sha = check_config(a)?;
    let (rows, feats, rows_sha) = load_rows(a, &format!("d2/subsets/s{n}_rows.jsonl"))?;
    ensure!(rows.len() == n, "subset size");
    let ob = a.read(Path::new(&format!("d2/subsets/s{n}_order.json")))?;
    let order_sha = sha256_hex(&ob);
    let oj: serde_json::Value = serde_json::from_slice(&ob)?;
    let order: Vec<usize> = oj["order"].as_array().context("order")?.iter().map(|v| v.as_u64().unwrap() as usize).collect();
    ensure!(order.len() == UPDATES * 16 && oj["size"] == n, "order shape");
    let labels: Vec<bool> = rows.iter().map(|r| r.label).collect();
    let theta0 = dump_values::<GA, _>(&model);
    let inv = inventory::<GA, _>(&model);
    ensure!(init::tensors_hash(&theta0) == init_tensors, "loaded model does not match the expected init tensors");
    let mut tr = DTrainer::<GA, M>::new(model, dev);
    let mut preds: Vec<serde_json::Value> = Vec::new();
    let mut moves: Vec<Movement> = Vec::new();
    let mut variation: Vec<serde_json::Value> = Vec::new();
    let mut inproc = serde_json::Map::new();
    let mut inf_ms: Vec<f64> = Vec::new();
    let snapshot = |tr: &DTrainer<GA, M>, u: usize, preds: &mut Vec<serde_json::Value>, moves: &mut Vec<Movement>, variation: &mut Vec<serde_json::Value>, inf_ms: &mut Vec<f64>| {
        let t0 = std::time::Instant::now();
        let z = panel_logits::<GA, M>(&tr.model, &feats, dev);
        let _ = G::sync(dev);
        inf_ms.push(t0.elapsed().as_secs_f64() * 1e3);
        let (mut correct, mut bce) = (0usize, 0f64);
        for ((r, l), zz) in rows.iter().zip(&labels).zip(&z) {
            preds.push(json!({"id": r.id, "update": u, "logit": *zz as f64}));
            correct += ((*zz > 0.0) == *l) as usize;
            let (y, zf) = (*l as u8 as f64, *zz as f64);
            bce += zf.max(0.0) - zf * y + (-zf.abs()).exp().ln_1p();
        }
        moves.push(movement::<GA, _>(&tr.model, &theta0, &inv, u));
        let ev = extra(&tr.model, &feats);
        if !ev.is_null() {
            variation.push(json!({"update": u, "variation": ev}));
        }
        json!({"correct": correct, "bce": bce / n as f64})
    };
    inproc.insert("0".into(), snapshot(&tr, 0, &mut preds, &mut moves, &mut variation, &mut inf_ms));
    a.write(Path::new(&format!("{out}/status.json")), json!({"status": "running"}).to_string().as_bytes())?;
    let mut trace = Vec::new();
    let mut exposure = vec![0u32; n];
    let t_all = std::time::Instant::now();
    for u in 0..UPDATES {
        let idx = &order[u * 16..(u + 1) * 16];
        for &i in idx {
            exposure[i] += 1;
        }
        let micros: Vec<(Vec<&Features>, Vec<bool>)> = (0..D1_ACCUM).map(|m| (vec![&feats[idx[2 * m]], &feats[idx[2 * m + 1]]], vec![labels[idx[2 * m]], labels[idx[2 * m + 1]]])).collect();
        let st = tr.step(&micros);
        if u % 100 == 0 || u + 1 == UPDATES {
            println!("{}", serde_json::to_string(&st)?);
        }
        if !(st.loss.is_finite() && st.grad_norm_pre_clip.is_finite()) {
            a.write(Path::new(&format!("{out}/status.json")), json!({"status": "failed_nonfinite", "update": u}).to_string().as_bytes())?;
            bail!("non-finite loss/gradient at update {u}: INCOMPLETE (no restart)");
        }
        trace.push(st);
        if SNAPSHOTS.contains(&(u + 1)) {
            let s = snapshot(&tr, u + 1, &mut preds, &mut moves, &mut variation, &mut inf_ms);
            println!("snapshot update {}: {s}", u + 1);
            inproc.insert((u + 1).to_string(), s);
        }
    }
    let wall = t_all.elapsed().as_secs_f64();
    let ck = a.output_path(Path::new(&format!("{out}/final/x")))?.parent().unwrap().to_path_buf();
    save_ckpt(&ck, &tr)?;
    let mut ckhash = serde_json::Map::new();
    for f in ["model.mpk", "opt_decay.mpk", "opt_nodecay.mpk", "meta.json"] {
        ckhash.insert(f.into(), json!(sha256_hex(&std::fs::read(ck.join(f))?)));
    }
    let ptext: String = preds.iter().map(|p| serde_json::to_string(p).unwrap() + "\n").collect();
    a.write(Path::new(&format!("{out}/predictions.jsonl")), ptext.as_bytes())?;
    // exposures in sorted-id order (rows are sorted by id, so index order == sorted-id order)
    let clipped = trace.iter().filter(|s| s.clipped).count();
    let tj = json!({"trace": trace, "exposure_by_sorted_id": exposure, "wall_secs": wall, "mean_update_ms": trace.iter().map(|s| s.millis).sum::<f64>() / trace.len() as f64, "clipped_updates": clipped, "clip_frequency": clipped as f64 / UPDATES as f64, "snapshot_inference_ms": inf_ms, "in_process_summary": inproc});
    a.write(Path::new(&format!("{out}/trace.json")), serde_json::to_string(&tj)?.as_bytes())?;
    a.write(Path::new(&format!("{out}/movement.json")), serde_json::to_string_pretty(&moves)?.as_bytes())?;
    if !variation.is_empty() {
        a.write(Path::new(&format!("{out}/variation.json")), serde_json::to_string_pretty(&variation)?.as_bytes())?;
    }
    let mut prov = json!({
        "model": name, "size": n,
        "consumer_source": source_id(),
        "contract_sha256": sha256_hex(&a.read(Path::new("d2/D2_CONTRACT.md"))?),
        "config_sha256": cfg_sha,
        "frozen_d2_sha256": sha256_hex(&a.read(Path::new("d2/frozen_d2.json"))?),
        "subset_rows_sha256": rows_sha, "train_order_sha256": order_sha,
        "init_file_sha256": init_file_sha, "init_tensors_sha256": init_tensors,
        "completed_updates": tr.update, "clip_ops": tr.clip_calls,
        "checkpoint_file_sha256": ckhash,
        "predictions_sha256": sha256_hex(ptext.as_bytes()),
        "precision": "f32 storage/accumulation; matmul inputs possibly TF32 (not strict FP32)",
        "data_producer_git_head": "36a81508b456ede1cb682f2f03fe678fd08db70f",
    });
    let id = sha256_hex(prov.to_string().as_bytes());
    prov["provenance_id"] = json!(id);
    a.write(Path::new(&format!("{out}/provenance.json")), serde_json::to_string_pretty(&prov)?.as_bytes())?;
    a.write(Path::new(&format!("{out}/status.json")), json!({"status": "complete", "completed_updates": tr.update}).to_string().as_bytes())?;
    println!("D2 FIT COMPLETE model={name} N={n} updates={} wall={wall:.1}s", tr.update);
    Ok(())
}

fn cmd_fit(args: &[String]) -> Result<()> {
    let env = Env::from(args)?;
    let a = env.access()?;
    let which = arg(args, "--model").context("--model A|M")?;
    let n: usize = arg(args, "--size").context("--size")?.parse()?;
    ensure!([64, 256, 768].contains(&n), "size must be 64, 256 or 768");
    let dev = cuda_device()?;
    match which.as_str() {
        "A" => {
            let (sha, vals) = read_values(&a, "init/canonical_init.bin")?;
            let model = load_values::<GA, _>(Model::<GA>::new(&dev), &vals, &dev);
            // pooled encoder / workspace variation (D1 diagnostic definition) on the subset rows
            let var = |m: &Model<GA>, feats: &[Features]| -> serde_json::Value {
                let inner = m.valid();
                let (mut ep, mut wp) = (Vec::new(), Vec::new());
                for chunk in feats.chunks(16) {
                    let refs: Vec<&Features> = chunk.iter().collect();
                    let b = Batch::<G>::from_features(&refs, &dev);
                    let (e, w) = inner.pooled_features(&b, Arm::A);
                    ep.extend(e.into_data().to_vec::<f32>().unwrap().chunks(192).map(|c| c.to_vec()));
                    wp.extend(w.into_data().to_vec::<f32>().unwrap().chunks(192).map(|c| c.to_vec()));
                }
                json!({"encoder_pool_std_over_mean_norm": variation_ratio(&ep), "workspace_pool_std_over_mean_norm": variation_ratio(&wp)})
            };
            fit_model(&env, &a, "A", n, model, &sha, &init::tensors_hash(&vals), &dev, &var)
        }
        "M" => {
            let (sha, vals) = read_values(&a, "d1/mlp_init.bin")?;
            let model = load_values::<GA, _>(Mlp::<GA>::new(&dev), &vals, &dev);
            fit_model(&env, &a, "M", n, model, &sha, &init::tensors_hash(&vals), &dev, &|_, _| serde_json::Value::Null)
        }
        _ => bail!("--model must be A or M"),
    }
}

fn main() -> Result<()> {
    std::thread::Builder::new().stack_size(512 * 1024 * 1024).spawn(real_main)?.join().map_err(|_| anyhow::anyhow!("worker thread panicked"))?
}

fn real_main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("qualify") => cmd_qualify(&args[2..]),
        Some("fit") => cmd_fit(&args[2..]),
        _ => bail!("usage: v69-d2-fit <qualify|fit> --artifacts DIR"),
    }
}
