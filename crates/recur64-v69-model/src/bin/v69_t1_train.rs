//! v69-t1-train: one pre-registered T1 training run (CUDA only).
//!   v69-t1-train --artifacts DIR --model A|M --k 250|1000|2000 --aug off|d8 --lr 0.001 --epochs 30 [--max-updates N]
//! Role: T1 trainer (train/val rows and canonical inits only; never test rows, metadata or the held-out partition).

use anyhow::{Context, Result, bail, ensure};
use burn::backend::cuda::CudaDevice;
use burn::module::AutodiffModule;
use burn::prelude::*;
use recur64_v69::access::{Access, Role};
use recur64_v69::custody::Custody;
use recur64_v69::d1::{FrozenD1, verify_group};
use recur64_v69::features::{Features, ModelRow, featurize, read_rows};
use recur64_v69::provenance::{sha256_hex, source_id};
use recur64_v69::streams::{MasterSeed, keyed_u64};
use recur64_v69_model::d1::*;
use recur64_v69_model::init;
use recur64_v69_model::inventory::{dump_values, inventory, load_values};
use recur64_v69_model::model::Model;
use recur64_v69_model::qualify::{G, GA};
use serde_json::json;
use std::path::{Path, PathBuf};

const BATCH: usize = 64;
const WARMUP: usize = 200;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

struct Env {
    custody: Custody,
    run: PathBuf,
    seed_rel: PathBuf,
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

/// T1 schedule: linear warmup (200 updates) then cosine to 0.1*peak at the final update.
pub fn lr_t1(u: usize, total: usize, peak: f64) -> f64 {
    if u < WARMUP {
        peak * (u + 1) as f64 / WARMUP as f64
    } else {
        let p = ((u - WARMUP) as f64 / (total.saturating_sub(1 + WARMUP)).max(1) as f64).min(1.0);
        0.1 * peak + 0.45 * peak * (1.0 + (std::f64::consts::PI * p).cos())
    }
}

fn permutation(seed: &MasterSeed, label: &str, epoch: u64, n: usize) -> Vec<usize> {
    let mut p: Vec<usize> = (0..n).collect();
    let mut rng = seed.stream(label, epoch);
    for i in (1..n).rev() {
        p.swap(i, rng.below(i as u64 + 1) as usize);
    }
    p
}

#[allow(clippy::too_many_arguments)]
fn run<M>(env: &Env, a: &Access, seed: &MasterSeed, run_id: &str, model: M, init_sha: &str, init_tensors: &str, dev: &CudaDevice, k: usize, aug: bool, peak: f64, epochs: usize, max_updates: Option<usize>) -> Result<()>
where
    M: DiagModel<GA> + AutodiffModule<GA>,
    M::InnerModule: DiagModel<G>,
{
    let out = format!("t1/train_runs/{run_id}");
    ensure!(!a.custody().resolve(Path::new(&out))?.join("provenance.json").exists(), "run {run_id} already exists; no reruns");
    let frozen: FrozenD1 = serde_json::from_slice(&a.read(Path::new("t1/frozen_train.json"))?)?;
    let nv = verify_group(a, &env.run, &frozen, "learner")?;
    eprintln!("[t1 train {run_id}] {nv} frozen hashes verified");
    let (rows, feats, train_sha) = load_rows(a, &format!("t1/rows/train_s{k}.jsonl"))?;
    let (vrows, vfeats, val_sha) = load_rows(a, "t1/rows/val.jsonl")?;
    let n = rows.len();
    let labels: Vec<bool> = rows.iter().map(|r| r.label).collect();
    let ups_per_epoch = n / BATCH;
    let total = max_updates.unwrap_or(epochs * ups_per_epoch);
    let theta0 = dump_values::<GA, _>(&model);
    ensure!(init::tensors_hash(&theta0) == init_tensors, "loaded model does not match the expected init tensors");
    let inv = inventory::<GA, _>(&model);
    let mut tr = DTrainer::<GA, M>::new(model, dev);
    let snaps: Vec<usize> = [total / 4, total / 2, 3 * total / 4, total].to_vec();
    let order_label = format!("t1_order/{run_id}");
    let aug_label = format!("t1_aug/{run_id}");
    let mut preds: Vec<serde_json::Value> = Vec::new();
    let mut inproc = serde_json::Map::new();
    let mut trace: Vec<serde_json::Value> = Vec::new();
    let mut win_loss = 0.0;
    let mut win_n = 0usize;
    let mut cache: Option<(u64, Vec<usize>)> = None;
    let t_all = std::time::Instant::now();
    a.write(Path::new(&format!("{out}/status.json")), json!({"status": "running", "total_updates": total}).to_string().as_bytes())?;
    let mut clipped_total = 0usize;
    for u in 0..total {
        let epoch = (u / ups_per_epoch) as u64;
        if cache.as_ref().map(|c| c.0) != Some(epoch) {
            cache = Some((epoch, permutation(seed, &order_label, epoch, n)));
        }
        let perm = &cache.as_ref().unwrap().1;
        let base = (u % ups_per_epoch) * BATCH;
        let mut fs: Vec<Features> = Vec::with_capacity(BATCH);
        let mut ls: Vec<bool> = Vec::with_capacity(BATCH);
        for j in 0..BATCH {
            let i = perm[base + j];
            let f = if aug {
                let t = (keyed_u64(seed, &aug_label, &((u * BATCH + j) as u64).to_le_bytes()) % 8) as usize;
                feats[i].d8(t)
            } else {
                feats[i].clone()
            };
            fs.push(f);
            ls.push(labels[i]);
        }
        let micro = vec![(fs.iter().collect::<Vec<&Features>>(), ls)];
        let lr = lr_t1(u, total, peak);
        let st = tr.step_lr(&micro, lr);
        if !(st.loss.is_finite() && st.grad_norm_pre_clip.is_finite()) {
            a.write(Path::new(&format!("{out}/status.json")), json!({"status": "failed_nonfinite", "update": u}).to_string().as_bytes())?;
            bail!("non-finite loss/gradient at update {u}: INCOMPLETE (no restart)");
        }
        win_loss += st.loss;
        win_n += 1;
        clipped_total += st.clipped as usize;
        if (u + 1) % 100 == 0 || u + 1 == total {
            trace.push(json!({"update": u + 1, "lr": lr, "train_loss_mean_last_window": win_loss / win_n as f64, "grad_norm": st.grad_norm_pre_clip, "ms_per_update": st.millis}));
            println!("{}", trace.last().unwrap());
            win_loss = 0.0;
            win_n = 0;
        }
        if snaps.contains(&(u + 1)) {
            let z = panel_logits::<GA, M>(&tr.model, &vfeats, dev);
            let (mut correct, mut bce) = (0usize, 0f64);
            let (mut tp, mut tn, mut np, mut nn) = (0f64, 0f64, 0f64, 0f64);
            for ((r, zz), _) in vrows.iter().zip(&z).zip(0..) {
                preds.push(json!({"id": r.id, "update": u + 1, "logit": *zz as f64}));
                let (y, zf) = (r.label as u8 as f64, *zz as f64);
                correct += ((*zz > 0.0) == r.label) as usize;
                bce += zf.max(0.0) - zf * y + (-zf.abs()).exp().ln_1p();
                if r.label { np += 1.0; tp += (*zz > 0.0) as u8 as f64 } else { nn += 1.0; tn += (*zz <= 0.0) as u8 as f64 }
            }
            let s = json!({"val_acc": correct as f64 / vrows.len() as f64, "val_bal_acc": 0.5 * (tp / np + tn / nn), "val_bce": bce / vrows.len() as f64});
            println!("snapshot update {}: {s}", u + 1);
            inproc.insert((u + 1).to_string(), s);
            let part: String = preds.iter().map(|p| serde_json::to_string(p).unwrap() + "\n").collect();
            a.write(Path::new(&format!("{out}/val_predictions_partial.jsonl")), part.as_bytes())?;
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
    a.write(Path::new(&format!("{out}/val_predictions.jsonl")), ptext.as_bytes())?;
    let mv = movement::<GA, _>(&tr.model, &theta0, &inv, total);
    a.write(Path::new(&format!("{out}/trace.json")), serde_json::to_string(&json!({"trace": trace, "wall_secs": wall, "clip_frequency": clipped_total as f64 / total as f64, "in_process_val": inproc, "movement": mv}))?.as_bytes())?;
    let mut prov = json!({
        "run_id": run_id, "model": run_id.split('_').next(), "k": k, "n_train": n, "aug": if aug { "d8" } else { "off" }, "peak_lr": peak, "epochs": epochs, "batch": BATCH, "updates": total, "completed_updates": tr.update,
        "consumer_source": source_id(),
        "frozen_train_sha256": sha256_hex(&a.read(Path::new("t1/frozen_train.json"))?),
        "contract_sha256": sha256_hex(&a.read(Path::new("t1/T1_CONTRACT.md"))?), "config_sha256": sha256_hex(&a.read(Path::new("t1/config.json"))?),
        "train_rows_sha256": train_sha, "val_rows_sha256": val_sha, "init_file_sha256": init_sha, "init_tensors_sha256": init_tensors,
        "checkpoint_file_sha256": ckhash, "val_predictions_sha256": sha256_hex(ptext.as_bytes()), "wall_secs": wall,
        "precision": "f32 storage/accumulation; matmul inputs possibly TF32 (not strict FP32)",
    });
    let id = sha256_hex(prov.to_string().as_bytes());
    prov["provenance_id"] = json!(id);
    a.write(Path::new(&format!("{out}/provenance.json")), serde_json::to_string_pretty(&prov)?.as_bytes())?;
    a.write(Path::new(&format!("{out}/status.json")), json!({"status": "complete", "completed_updates": tr.update}).to_string().as_bytes())?;
    println!("T1 RUN COMPLETE {run_id} updates={} wall={wall:.1}s", tr.update);
    Ok(())
}

fn real_main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let env = Env {
        custody: Custody::new(Path::new(&arg(&args, "--artifacts").context("--artifacts")?))?,
        run: PathBuf::from("gen-001"),
        seed_rel: PathBuf::from("t1/seed/t1_master_seed.hex"),
    };
    let a = Access::new(&env.custody, Role::T1Trainer, &env.run, &env.seed_rel)?;
    let seed = MasterSeed::from_hex(String::from_utf8(a.read(&env.seed_rel)?)?.trim())?;
    let which = arg(&args, "--model").context("--model")?;
    let k: usize = arg(&args, "--k").context("--k")?.parse()?;
    ensure!([250, 1000, 2000].contains(&k), "k must be 250, 1000 or 2000");
    let aug = match arg(&args, "--aug").context("--aug")?.as_str() {
        "d8" => true,
        "off" => false,
        o => bail!("--aug must be off or d8, got {o}"),
    };
    let lr: f64 = arg(&args, "--lr").context("--lr")?.parse()?;
    let epochs: usize = arg(&args, "--epochs").context("--epochs")?.parse()?;
    let max_updates: Option<usize> = arg(&args, "--max-updates").map(|s| s.parse()).transpose()?;
    let run_id = format!("{which}_k{k}_{}_lr{lr:e}{}", if aug { "d8" } else { "off" }, if max_updates.is_some() { "_smoke" } else { "" });
    let dev = cuda_device()?;
    match which.as_str() {
        "A" => {
            let (sha, vals) = read_values(&a, "init/canonical_init.bin")?;
            let model = load_values::<GA, _>(Model::<GA>::new(&dev), &vals, &dev);
            run(&env, &a, &seed, &run_id, model, &sha, &init::tensors_hash(&vals), &dev, k, aug, lr, epochs, max_updates)
        }
        "M" => {
            let (sha, vals) = read_values(&a, "d1/mlp_init.bin")?;
            let model = load_values::<GA, _>(Mlp::<GA>::new(&dev), &vals, &dev);
            run(&env, &a, &seed, &run_id, model, &sha, &init::tensors_hash(&vals), &dev, k, aug, lr, epochs, max_updates)
        }
        _ => bail!("--model must be A or M"),
    }
}

fn main() -> Result<()> {
    std::thread::Builder::new().stack_size(512 * 1024 * 1024).spawn(real_main)?.join().map_err(|_| anyhow::anyhow!("worker thread panicked"))?
}
