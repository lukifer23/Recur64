//! v69-d1-fit: D1 neural diagnostics (CUDA only).
//!   v69-d1-fit init-mlp --artifacts DIR
//!   v69-d1-fit qualify  --artifacts DIR
//!   v69-d1-fit fit --model A|M --artifacts DIR
//! Learner role only: seed, data/fit.jsonl and D1 inputs; no validation rows or withheld dataset parts.

use anyhow::{Context, Result, bail, ensure};
use burn::backend::cuda::CudaDevice;
use burn::module::AutodiffModule;
use burn::prelude::*;
use recur64_v69::access::{Access, Role};
use recur64_v69::custody::Custody;
use recur64_v69::d1::{STREAM_D1_MLP_INIT, read_frozen, verify_group};
use recur64_v69::features::{Features, ModelRow, featurize, read_rows};
use recur64_v69::provenance::{sha256_hex, source_id};
use recur64_v69::streams::MasterSeed;
use recur64_v69_model::d1::*;
use recur64_v69_model::d1_qualify::{D1QualCtx, qualify_d1};
use recur64_v69_model::init::{self, InitHeader, InitTensor};
use recur64_v69_model::inventory::{dump_values, inventory, load_values};
use recur64_v69_model::model::Model;
use recur64_v69_model::qualify::{G, GA};
use serde_json::json;
use std::path::{Path, PathBuf};

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
    // Visible failure if CUDA is unavailable: never substitute CPU.
    G::sync(&d).map_err(|e| anyhow::anyhow!("CUDA device unavailable (no CPU fallback): {e:?}"))?;
    Ok(d)
}

fn load_panel(a: &Access) -> Result<(Vec<ModelRow>, Vec<Features>, String)> {
    let bytes = a.read(Path::new("d1/panel/panel_rows.jsonl"))?;
    let rows = read_rows(std::str::from_utf8(&bytes)?)?;
    ensure!(rows.len() == 32, "panel must have 32 rows");
    let feats = rows.iter().map(|r| featurize(&r.fen, r.budget).map(|x| x.0)).collect::<Result<Vec<_>>>()?;
    Ok((rows, feats, sha256_hex(&bytes)))
}

fn read_values(a: &Access, rel: &str) -> Result<(String, Vec<(Vec<usize>, Vec<f32>)>)> {
    let bytes = a.read(Path::new(rel))?;
    let (_, vals) = init::from_bytes(&bytes)?;
    Ok((sha256_hex(&bytes), vals))
}

// ---------------------------------------------------------------- init-mlp

fn cmd_init_mlp(args: &[String]) -> Result<()> {
    let env = Env::from(args)?;
    let a = env.access()?;
    let seed = MasterSeed::from_hex(String::from_utf8(a.read(&env.seed_rel)?)?.trim())?;
    let dev = cuda_device()?;
    let inv = inventory::<G, _>(&Mlp::<G>::new(&dev));
    let total: usize = inv.iter().map(|p| p.numel).sum();
    ensure!(total == 116_225, "unexpected MLP size {total}");
    let vals = init::generate(&seed, STREAM_D1_MLP_INIT, &inv);
    let header = InitHeader { version: 1, stream_label: STREAM_D1_MLP_INIT.into(), seed_fingerprint: seed.fingerprint(), tensors: inv.iter().map(|p| InitTensor { path: p.path.clone(), leaf: p.leaf.clone(), shape: p.shape.clone() }).collect() };
    let bytes = init::to_bytes(&header, &vals);
    let target = Path::new("d1/mlp_init.bin");
    ensure!(!a.custody().resolve(target)?.exists(), "MLP init exists; never regenerated");
    a.write(target, &bytes)?;
    a.write(Path::new("d1/mlp_init_info.json"), serde_json::to_string_pretty(&json!({"file_sha256": sha256_hex(&bytes), "tensors_sha256": init::tensors_hash(&vals), "stream": format!("{STREAM_D1_MLP_INIT}/0"), "parameters": total, "inventory": inv}))?.as_bytes())?;
    println!("mlp parameters={total} file_sha256={}", sha256_hex(&bytes));
    Ok(())
}

// ---------------------------------------------------------------- qualify

fn cmd_qualify(args: &[String]) -> Result<()> {
    let env = Env::from(args)?;
    let a = env.access()?;
    let seed = MasterSeed::from_hex(String::from_utf8(a.read(&env.seed_rel)?)?.trim())?;
    let _dev = cuda_device()?;
    let (rows, feats, _) = load_panel(&a)?;
    let (_, e1_vals) = read_values(&a, "init/canonical_init.bin")?;
    let ctx = D1QualCtx { rows, feats, seed, e1_init_values: e1_vals };
    let rep = qualify_d1(&ctx);
    let ok = rep["qualified"].as_bool().unwrap_or(false);
    let full = json!({"source": source_id(), "result": rep});
    a.write(Path::new("d1/qual_report.json"), serde_json::to_string_pretty(&full)?.as_bytes())?;
    println!("D1 QUALIFIED={ok}");
    if !ok {
        std::process::exit(5);
    }
    Ok(())
}

// ---------------------------------------------------------------- fit

fn check_config(a: &Access) -> Result<String> {
    let bytes = a.read(Path::new("d1/config.json"))?;
    let c: serde_json::Value = serde_json::from_slice(&bytes)?;
    let n = &c["neural"];
    ensure!(n["updates"] == D1_UPDATES && n["microbatch"] == D1_MICRO && n["accumulation_steps"] == D1_ACCUM, "config updates/microbatch mismatch");
    ensure!(n["peak_lr"].as_f64() == Some(recur64_v69_model::train::PEAK_LR) && n["final_lr"].as_f64() == Some(recur64_v69_model::train::FINAL_LR), "config lr mismatch");
    ensure!(n["warmup_updates"] == recur64_v69_model::train::WARMUP && n["weight_decay"].as_f64() == Some(recur64_v69_model::train::WEIGHT_DECAY) && n["grad_clip_global_norm"].as_f64() == Some(1.0), "config optimizer mismatch");
    ensure!(n["measure_updates"] == json!(MEASURE_UPDATES), "config measure updates mismatch");
    ensure!(c["mlp"]["input_width"] == MLP_IN, "config mlp width mismatch");
    Ok(sha256_hex(&bytes))
}

fn fit_model<M>(env: &Env, a: &Access, name: &str, model: M, init_file_sha: &str, init_tensors: &str, dev: &CudaDevice) -> Result<()>
where
    M: DiagModel<GA> + AutodiffModule<GA>,
    M::InnerModule: DiagModel<G>,
{
    let out = format!("d1/fits/{name}");
    ensure!(!a.custody().resolve(Path::new(&out))?.join("provenance.json").exists(), "D1-{name} already run; no reruns");
    let frozen = read_frozen(a)?;
    let nverified = verify_group(a, &env.run, &frozen, "learner")?;
    eprintln!("[d1 fit {name}] {nverified} frozen hashes verified");
    let cfg_sha = check_config(a)?;
    let (rows, feats, panel_sha) = load_panel(a)?;
    let ob = a.read(Path::new("d1/panel/train_order.json"))?;
    let order_sha = sha256_hex(&ob);
    let order_json: serde_json::Value = serde_json::from_slice(&ob)?;
    let order: Vec<usize> = order_json["order"].as_array().context("order")?.iter().map(|v| v.as_u64().unwrap() as usize).collect();
    ensure!(order.len() == D1_UPDATES * 16 && order_json["panel_size"] == 32, "order shape");
    let labels: Vec<bool> = rows.iter().map(|r| r.label).collect();
    let theta0 = dump_values::<GA, _>(&model);
    let inv = inventory::<GA, _>(&model);
    ensure!(init::tensors_hash(&theta0) == init_tensors, "loaded model does not match the frozen init tensors");
    let mut tr = DTrainer::<GA, M>::new(model, dev);
    let mut preds: Vec<serde_json::Value> = Vec::new();
    let mut moves: Vec<Movement> = Vec::new();
    let mut inproc = serde_json::Map::new();
    let snapshot = |tr: &DTrainer<GA, M>, u: usize, preds: &mut Vec<serde_json::Value>, moves: &mut Vec<Movement>| {
        let z = panel_logits::<GA, M>(&tr.model, &feats, dev);
        let mut correct = 0;
        let mut bce = 0f64;
        for ((r, l), zz) in rows.iter().zip(&labels).zip(&z) {
            preds.push(json!({"id": r.id, "update": u, "logit": *zz as f64}));
            correct += ((*zz > 0.0) == *l) as usize;
            let y = *l as u8 as f64;
            let zf = *zz as f64;
            bce += zf.max(0.0) - zf * y + (-zf.abs()).exp().ln_1p();
        }
        moves.push(movement::<GA, _>(&tr.model, &theta0, &inv, u));
        json!({"correct": correct, "bce": bce / 32.0})
    };
    inproc.insert("0".into(), snapshot(&tr, 0, &mut preds, &mut moves));
    a.write(Path::new(&format!("{out}/status.json")), json!({"status": "running"}).to_string().as_bytes())?;
    let mut trace = Vec::new();
    let mut exposure = vec![0u32; 32];
    let t_all = std::time::Instant::now();
    for u in 0..D1_UPDATES {
        let idx = &order[u * 16..(u + 1) * 16];
        for &i in idx {
            exposure[i] += 1;
        }
        let micros: Vec<(Vec<&Features>, Vec<bool>)> = (0..D1_ACCUM).map(|m| (vec![&feats[idx[2 * m]], &feats[idx[2 * m + 1]]], vec![labels[idx[2 * m]], labels[idx[2 * m + 1]]])).collect();
        let st = tr.step(&micros);
        if u % 50 == 0 || u + 1 == D1_UPDATES {
            println!("{}", serde_json::to_string(&st)?);
        }
        if !(st.loss.is_finite() && st.grad_norm_pre_clip.is_finite()) {
            a.write(Path::new(&format!("{out}/status.json")), json!({"status": "failed_nonfinite", "update": u}).to_string().as_bytes())?;
            bail!("non-finite loss/gradient at update {u}: INCOMPLETE (no restart)");
        }
        trace.push(st);
        if MEASURE_UPDATES.contains(&(u + 1)) {
            let s = snapshot(&tr, u + 1, &mut preds, &mut moves);
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
    a.write(Path::new(&format!("{out}/panel_predictions.jsonl")), ptext.as_bytes())?;
    let clipped = trace.iter().filter(|s| s.clipped).count();
    let tj = json!({"trace": trace, "exposure": exposure, "wall_secs": wall, "clipped_updates": clipped, "clip_frequency": clipped as f64 / D1_UPDATES as f64, "in_process_panel_summary": inproc});
    a.write(Path::new(&format!("{out}/trace.json")), serde_json::to_string(&tj)?.as_bytes())?;
    a.write(Path::new(&format!("{out}/movement.json")), serde_json::to_string_pretty(&moves)?.as_bytes())?;
    let frozen_sha = sha256_hex(&a.read(Path::new("d1/frozen_d1.json"))?);
    let mut prov = json!({
        "model": name,
        "consumer_source": source_id(),
        "contract_sha256": sha256_hex(&a.read(Path::new("d1/D1_CONTRACT.md"))?),
        "config_sha256": cfg_sha,
        "frozen_d1_sha256": frozen_sha,
        "e1_supplementary_manifest_sha256": sha256_hex(&a.read(Path::new("d1/e1_supplementary_manifest.json"))?),
        "panel_rows_sha256": panel_sha,
        "train_order_sha256": order_sha,
        "init_file_sha256": init_file_sha,
        "init_tensors_sha256": init_tensors,
        "completed_updates": tr.update,
        "clip_ops": tr.clip_calls,
        "checkpoint_file_sha256": ckhash,
        "panel_predictions_sha256": sha256_hex(ptext.as_bytes()),
        "precision": "f32 storage/accumulation; matmul inputs possibly TF32 (not strict FP32)",
        "data_producer_git_head": "36a81508b456ede1cb682f2f03fe678fd08db70f",
    });
    let id = sha256_hex(prov.to_string().as_bytes());
    prov["provenance_id"] = json!(id);
    a.write(Path::new(&format!("{out}/provenance.json")), serde_json::to_string_pretty(&prov)?.as_bytes())?;
    a.write(Path::new(&format!("{out}/status.json")), json!({"status": "complete", "completed_updates": tr.update}).to_string().as_bytes())?;
    println!("D1 FIT COMPLETE model={name} updates={} wall={wall:.1}s", tr.update);
    Ok(())
}

fn cmd_fit(args: &[String]) -> Result<()> {
    let env = Env::from(args)?;
    let a = env.access()?;
    let which = arg(args, "--model").context("--model A|M")?;
    let dev = cuda_device()?;
    match which.as_str() {
        "A" => {
            let (sha, vals) = read_values(&a, "init/canonical_init.bin")?;
            let model = load_values::<GA, _>(Model::<GA>::new(&dev), &vals, &dev);
            fit_model(&env, &a, "A", model, &sha, &init::tensors_hash(&vals), &dev)
        }
        "M" => {
            let (sha, vals) = read_values(&a, "d1/mlp_init.bin")?;
            let model = load_values::<GA, _>(Mlp::<GA>::new(&dev), &vals, &dev);
            fit_model(&env, &a, "M", model, &sha, &init::tensors_hash(&vals), &dev)
        }
        _ => bail!("--model must be A or M"),
    }
}

fn main() -> Result<()> {
    // Burn record (de)serialization recurses deeply: run on a large-stack worker thread.
    std::thread::Builder::new().stack_size(512 * 1024 * 1024).spawn(real_main)?.join().map_err(|_| anyhow::anyhow!("worker thread panicked"))?
}

fn real_main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("init-mlp") => cmd_init_mlp(&args[2..]),
        Some("qualify") => cmd_qualify(&args[2..]),
        Some("fit") => cmd_fit(&args[2..]),
        _ => bail!("usage: v69-d1-fit <init-mlp|qualify|fit> --artifacts DIR"),
    }
}
