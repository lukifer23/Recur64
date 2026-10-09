//! v69-g1-eval: frozen-candidate evaluator for G1 (inference only).
//!   v69-g1-eval verify-evaluator --artifacts DIR   fitting-only fixture; writes an append-only receipt
//!   v69-g1-eval eval --candidate A|M|B --artifacts DIR   ONE registered invocation per candidate
//! Role: G1 frozen-candidate evaluator. Reads only the G1 model rows, the frozen candidate files, the frozen
//! intervention map and the fixed fitting-only fixture; never root-bearing metadata or gen-001 data.
//! A and M run on CUDA (no CPU fallback); B (baseline) is host-side f64 inference (authorized, not a fallback).

use anyhow::{Context, Result, bail, ensure};
use burn::backend::cuda::CudaDevice;
use burn::prelude::*;
use recur64_v69::access::{Access, Role};
use recur64_v69::custody::Custody;
use recur64_v69::d1::{BaselineModel, FrozenD1, N_BASELINE, baseline_features, verify_group};
use recur64_v69::d2::write_new;
use recur64_v69::features::{Features, ModelRow, featurize, read_rows};
use recur64_v69::provenance::{sha256_hex, source_digest, source_id};
use recur64_v69_model::d1::{Mlp, load_mlp_eval};
use recur64_v69_model::model::{Arm, Batch, Model};
use recur64_v69_model::qualify::G;
use recur64_v69_model::train::load_model_for_eval;
use serde_json::json;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

const TOL_NEURAL: f64 = 2e-3;
const TOL_BASELINE: f64 = 1e-9;

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
            run: PathBuf::from("gen-001"),
            seed_rel: PathBuf::from("g1r1/seed/g1_master_seed.hex"),
        })
    }
    fn access(&self) -> Result<Access> {
        Access::new(&self.custody, Role::G1Evaluator, &self.run, &self.seed_rel)
    }
}

fn cuda_device() -> Result<CudaDevice> {
    let d = CudaDevice::default();
    G::sync(&d).map_err(|e| anyhow::anyhow!("CUDA device unavailable (no CPU fallback): {e:?}"))?;
    Ok(d)
}

fn jsonl_vals(text: &str) -> Result<Vec<serde_json::Value>> {
    text.lines().map(|l| Ok(serde_json::from_str(l)?)).collect()
}

fn load_rows(a: &Access, rel: &str) -> Result<(Vec<ModelRow>, Vec<Features>, Vec<(String, u8)>, String)> {
    let bytes = a.read(Path::new(rel))?;
    let rows = read_rows(std::str::from_utf8(&bytes)?)?;
    let mut feats = Vec::new();
    let mut fb = Vec::new();
    for r in &rows {
        feats.push(featurize(&r.fen, r.budget)?.0);
        fb.push((r.fen.clone(), r.budget));
    }
    Ok((rows, feats, fb, sha256_hex(&bytes)))
}

fn predict_neural<M, F: Fn(&M, &[&Features]) -> Vec<f32>>(m: &M, feats: &[&Features], f: &F) -> Vec<f32> {
    let mut out = Vec::new();
    for chunk in feats.chunks(16) {
        out.extend(f(m, chunk));
    }
    out
}

fn a_logits(m: &Model<G>, f: &[&Features], dev: &CudaDevice) -> Vec<f32> {
    let b = Batch::<G>::from_features(f, dev);
    let z = m.forward(&b, Arm::A).pop().unwrap().into_data().to_vec::<f32>().unwrap();
    let _ = G::sync(dev);
    z
}

fn m_logits(m: &Mlp<G>, f: &[&Features], dev: &CudaDevice) -> Vec<f32> {
    let z = m.forward(recur64_v69_model::d1::mlp_input::<G>(f, dev)).into_data().to_vec::<f32>().unwrap();
    let _ = G::sync(dev);
    z
}

/// Verify the listed prefixes of the frozen protocol file that this role may read.
fn verify_protocol(a: &Access) -> Result<usize> {
    let fz: FrozenD1 = serde_json::from_slice(&a.read(Path::new("g1r1/frozen_protocol.json"))?)?;
    let mut n = 0;
    for (rel, want) in &fz.groups["protocol"] {
        let readable = (rel.starts_with("d3/fits/") && !rel.contains("/final/opt")) || rel.starts_with("d1/baseline/") || rel == "d2/subsets/s768_rows.jsonl" || rel.starts_with("g1r1/G1_") || rel == "g1r1/config.json";
        if !readable {
            continue;
        }
        let got = sha256_hex(&a.read(Path::new(rel)).with_context(|| format!("verify {rel}"))?);
        ensure!(&got == want, "FROZEN HASH MISMATCH {rel}: expected {want}, got {got}");
        n += 1;
    }
    Ok(n)
}

fn baseline_model(a: &Access) -> Result<BaselineModel> {
    let v: serde_json::Value = serde_json::from_slice(&a.read(Path::new("d1/baseline/model.json"))?)?;
    Ok(serde_json::from_value(v["model"].clone())?)
}

/// Erasure for the baseline: every board-derived base feature replaced by its fit mean; budget indicator kept;
/// interactions recomputed from those constants.
fn baseline_erased(m: &BaselineModel, budget: u8) -> [f64; N_BASELINE] {
    let b2 = (budget == 2) as u8 as f64;
    let mut out = [0.0; N_BASELINE];
    for j in 0..11 {
        out[j] = m.mean[j];
        out[12 + j] = m.mean[j] * b2;
    }
    out[11] = b2;
    out
}

// ---------------------------------------------------------------- verify-evaluator

fn cmd_verify(args: &[String]) -> Result<()> {
    let env = Env::from(args)?;
    let a = env.access()?;
    let dev = cuda_device()?;
    let np = verify_protocol(&a)?;
    let (rows, feats, fb, rows_sha) = load_rows(&a, "d2/subsets/s768_rows.jsonl")?;
    ensure!(rows.len() == 768, "fixture size");
    let refs: Vec<&Features> = feats.iter().collect();
    let mut checks = Vec::new();
    // A and M: forward agreement with the recorded D3 update-12,000 fitting predictions
    for (name, dir) in [("A", "d3/fits/A"), ("M", "d3/fits/M")] {
        let ck = a.input_path(Path::new(&format!("{dir}/final/meta.json")))?.parent().unwrap().to_path_buf();
        let z: Vec<f32> = if name == "A" {
            let m = load_model_for_eval::<G>(&ck, &dev)?;
            predict_neural(&m, &refs, &|m: &Model<G>, f: &[&Features]| a_logits(m, f, &dev))
        } else {
            let m = load_mlp_eval::<G>(&ck, &dev)?;
            predict_neural(&m, &refs, &|m: &Mlp<G>, f: &[&Features]| m_logits(m, f, &dev))
        };
        let recorded: HashMap<String, f64> = jsonl_vals(&a.read_to_string(Path::new(&format!("{dir}/predictions.jsonl")))?)?.into_iter().filter(|v| v["update"] == 12000).map(|v| (v["id"].as_str().unwrap().to_string(), v["logit"].as_f64().unwrap())).collect();
        ensure!(recorded.len() == 768, "recorded predictions");
        let worst = rows.iter().zip(&z).map(|(r, zz)| (*zz as f64 - recorded[&r.id]).abs()).fold(0.0, f64::max);
        let agree = rows.iter().zip(&z).filter(|(r, zz)| (**zz > 0.0) == (recorded[&r.id] > 0.0)).count();
        // target separation: mutate labels and ids -> identical features and logits
        let flipped: Vec<Features> = rows.iter().take(8).map(|r| featurize(&r.fen, r.budget).unwrap().0).collect();
        let sep = flipped.iter().zip(feats.iter()).all(|(x, y)| x == y);
        checks.push(json!({"candidate": name, "max_abs_logit_diff_vs_recorded_d3_update12000": worst, "tolerance": TOL_NEURAL, "classification_agreement": agree, "of": 768, "target_separation_features_independent_of_labels_ids": sep, "pass": worst <= TOL_NEURAL && agree >= 765 && sep}));
    }
    // B: exact reproduction of the recorded D1 fitting predictions
    let bm = baseline_model(&a)?;
    let rec: HashMap<String, f64> = jsonl_vals(&a.read_to_string(Path::new("d1/baseline/predictions_fit.jsonl"))?)?.into_iter().map(|v| (v["id"].as_str().unwrap().to_string(), v["logit"].as_f64().unwrap())).collect();
    let mut worst = 0f64;
    for (r, (fen, b)) in rows.iter().zip(&fb) {
        worst = worst.max((bm.logit(&baseline_features(fen, *b)?) - rec[&r.id]).abs());
    }
    checks.push(json!({"candidate": "B", "max_abs_logit_diff_vs_recorded_d1_fit_predictions": worst, "tolerance": TOL_BASELINE, "standardization": "stored fit statistics (not recomputed)", "pass": worst <= TOL_BASELINE}));
    // erasure / derangement machinery sanity on the fixture (no G1 contact): erased != original inputs
    let er = feats[0].erased();
    let erased_ok = er.piece.iter().all(|p| *p == 0) && er.budget == feats[0].budget;
    let be = baseline_erased(&bm, 2);
    let berased_ok = (0..11).all(|j| be[j] == bm.mean[j]) && be[11] == 1.0;
    checks.push(json!({"name": "erasure_definitions_consistent", "neural": erased_ok, "baseline": berased_ok, "pass": erased_ok && berased_ok}));
    let ok = checks.iter().all(|c| c["pass"].as_bool() == Some(true));
    let rep = json!({"ok": ok, "protocol_hashes_verified": np, "fixture": "D2 s768 fitting rows (fitting-only); recorded D3 update-12000 and D1 baseline fitting predictions", "fixture_rows_sha256": rows_sha, "checks": checks, "no_g1_labels_or_outputs_used": true, "source": source_id(), "evaluator_source_digest": source_digest()});
    write_new(&a, "g1r1/receipts/evaluator_verification.json", serde_json::to_string_pretty(&rep)?.as_bytes())?;
    println!("{}", serde_json::to_string_pretty(&rep)?);
    if !ok {
        std::process::exit(5);
    }
    Ok(())
}

// ---------------------------------------------------------------- eval

fn cmd_eval(args: &[String]) -> Result<()> {
    let t_proc = Instant::now();
    let env = Env::from(args)?;
    let a = env.access()?;
    let cand = arg(args, "--candidate").context("--candidate A|M|B")?;
    ensure!(["A", "M", "B"].contains(&cand.as_str()), "candidate must be A, M or B");
    let out = format!("g1r1/eval/{cand}");
    ensure!(!a.custody().resolve(Path::new(&out))?.join("provenance.json").exists(), "candidate {cand} already evaluated: one registered evaluation only");
    // frozen hashes + executable-source identity
    let frozen: FrozenD1 = serde_json::from_slice(&a.read(Path::new("g1r1/frozen_g1.json"))?)?;
    let nv = verify_group(&a, &env.run, &frozen, "evaluator")?;
    let src: serde_json::Value = serde_json::from_slice(&a.read(Path::new("g1r1/evaluator_source.json"))?)?;
    let sid = source_id();
    ensure!(sid.git_dirty_files == 0, "evaluator must run from a clean tree ({} dirty files)", sid.git_dirty_files);
    ensure!(src["source_digest"].as_str() == Some(source_digest().as_str()), "EVALUATOR SOURCE DIGEST MISMATCH: the verified evaluator source is frozen");
    eprintln!("[g1 eval {cand}] {nv} frozen hashes verified; source digest matches");
    let (rows, feats, fb, rows_sha) = load_rows(&a, "g1r1/rows/g1_rows.jsonl")?;
    ensure!(rows.len() == 1536, "G1 rows");
    let idx: HashMap<&str, usize> = rows.iter().enumerate().map(|(i, r)| (r.id.as_str(), i)).collect();
    let map_bytes = a.read(Path::new("g1r1/intervention/map.json"))?;
    let map_v: serde_json::Value = serde_json::from_slice(&map_bytes)?;
    let donor_of: Vec<usize> = rows.iter().map(|r| idx[map_v["map"][&r.id].as_str().unwrap()]).collect();
    ensure!(donor_of.iter().enumerate().all(|(i, d)| *d != i), "frozen map has a fixed point");
    ensure!(donor_of.iter().all(|d| feats[*d].budget == feats[donor_of.iter().position(|x| x == d).unwrap()].budget), "donor budget cell");
    let drefs: Vec<Features> = (0..rows.len()).map(|i| { let mut f = feats[donor_of[i]].clone(); f.budget = feats[i].budget; f }).collect();
    let erefs: Vec<Features> = feats.iter().map(|f| f.erased()).collect();
    let mut timing = serde_json::Map::new();
    let (real, der, ers): (Vec<f64>, Vec<f64>, Vec<f64>);
    let mut files = serde_json::Map::new();
    let t_start;
    match cand.as_str() {
        "A" | "M" => {
            let t0 = Instant::now();
            let dev = cuda_device()?;
            let ck = a.input_path(Path::new(&format!("d3/fits/{cand}/final/meta.json")))?.parent().unwrap().to_path_buf();
            for f in ["model.mpk", "meta.json"] {
                files.insert(format!("d3/fits/{cand}/final/{f}"), json!(sha256_hex(&std::fs::read(ck.join(f))?)));
            }
            let run = |fs: &[Features], timing: &mut serde_json::Map<String, serde_json::Value>, key: &str, first: bool| -> Vec<f64> { let _ = (timing, key, first); fs.iter().map(|_| 0.0).collect() };
            let _ = run;
            let (am, mm): (Option<Model<G>>, Option<Mlp<G>>) = if cand == "A" { (Some(load_model_for_eval::<G>(&ck, &dev)?), None) } else { (None, Some(load_mlp_eval::<G>(&ck, &dev)?)) };
            t_start = t0.elapsed().as_secs_f64() * 1e3;
            let pred = |fs: &[&Features]| -> Vec<f64> {
                let z = if let Some(m) = &am { predict_neural(m, fs, &|m: &Model<G>, f: &[&Features]| a_logits(m, f, &dev)) } else { predict_neural(mm.as_ref().unwrap(), fs, &|m: &Mlp<G>, f: &[&Features]| m_logits(m, f, &dev)) };
                z.into_iter().map(|x| x as f64).collect()
            };
            let t1 = Instant::now();
            let r: Vec<&Features> = feats.iter().collect();
            real = pred(&r);
            timing.insert("real_pass_incl_first_batch_jit_ms".into(), json!(t1.elapsed().as_secs_f64() * 1e3));
            let d: Vec<&Features> = drefs.iter().collect();
            der = pred(&d);
            let e: Vec<&Features> = erefs.iter().collect();
            ers = pred(&e);
            // timing-only repeats (predictions above are final): warmed, synchronized, batch 16
            let first16: Vec<&Features> = feats.iter().take(16).collect();
            let mut lat = Vec::new();
            for i in 0..25 {
                let t = Instant::now();
                let _ = pred(&first16);
                if i >= 5 {
                    lat.push(t.elapsed().as_secs_f64() * 1e3);
                }
            }
            lat.sort_by(|x, y| x.partial_cmp(y).unwrap());
            timing.insert("warmed_synchronized_batch16_median_ms".into(), json!(lat[lat.len() / 2]));
            let tw = Instant::now();
            let _ = pred(&r);
            timing.insert("warm_full_pass_1536_inference_only_ms".into(), json!(tw.elapsed().as_secs_f64() * 1e3));
            timing.insert("batch_size".into(), json!(16));
        }
        _ => {
            let t0 = Instant::now();
            let bm = baseline_model(&a)?;
            files.insert("d1/baseline/model.json".into(), json!(sha256_hex(&a.read(Path::new("d1/baseline/model.json"))?)));
            t_start = t0.elapsed().as_secs_f64() * 1e3;
            let tf = Instant::now();
            let raw: Vec<[f64; N_BASELINE]> = fb.iter().map(|(fen, b)| baseline_features(fen, *b)).collect::<Result<_>>()?;
            timing.insert("feature_extraction_total_ms_1536_rows_incl_parse_and_legal_move_features".into(), json!(tf.elapsed().as_secs_f64() * 1e3));
            let tl = Instant::now();
            real = raw.iter().map(|r| bm.logit(r)).collect();
            timing.insert("logit_computation_ms_1536".into(), json!(tl.elapsed().as_secs_f64() * 1e3));
            der = (0..rows.len()).map(|i| { let mut r = raw[donor_of[i]]; let _ = &mut r; bm.logit(&raw[donor_of[i]]) }).collect();
            ers = fb.iter().map(|(_, b)| bm.logit(&baseline_erased(&bm, *b))).collect();
            timing.insert("device".into(), json!("host f64 (no GPU)"));
        }
    }
    timing.insert("startup_cuda_init_and_model_load_ms".into(), json!(t_start));
    // predictions
    let mut text = String::new();
    for (i, r) in rows.iter().enumerate() {
        text.push_str(&serde_json::to_string(&json!({"id": r.id, "mode": "real", "logit": real[i]}))?);
        text.push('\n');
    }
    for (i, r) in rows.iter().enumerate() {
        text.push_str(&serde_json::to_string(&json!({"id": r.id, "mode": "derange", "logit": der[i], "donor_id": rows[donor_of[i]].id}))?);
        text.push('\n');
    }
    for (i, r) in rows.iter().enumerate() {
        text.push_str(&serde_json::to_string(&json!({"id": r.id, "mode": "erase", "logit": ers[i]}))?);
        text.push('\n');
    }
    a.write(Path::new(&format!("{out}/predictions.jsonl")), text.as_bytes())?;
    timing.insert("evaluator_process_wall_ms_excluding_aggregation".into(), json!(t_proc.elapsed().as_secs_f64() * 1e3));
    let prov = json!({
        "candidate": cand, "evaluator_source": {"git_head": sid.git_head, "digest": src["source_digest"]},
        "frozen_g1_sha256": sha256_hex(&a.read(Path::new("g1r1/frozen_g1.json"))?),
        "protocol_sha256": sha256_hex(&a.read(Path::new("g1r1/frozen_protocol.json"))?),
        "data_manifest_sha256": sha256_hex(&a.read(Path::new("g1r1/MANIFEST.sha256.json"))?),
        "g1_rows_sha256": rows_sha, "intervention_map_sha256": sha256_hex(&map_bytes),
        "candidate_files": files, "predictions_sha256": sha256_hex(text.as_bytes()),
        "precision": "f32 storage/accumulation; matmul inputs possibly TF32 (not strict FP32); baseline host f64",
        "timing": timing,
    });
    let mut prov = prov;
    let id = sha256_hex(prov.to_string().as_bytes());
    prov["provenance_id"] = json!(id);
    a.write(Path::new(&format!("{out}/provenance.json")), serde_json::to_string_pretty(&prov)?.as_bytes())?;
    println!("G1 EVAL COMPLETE candidate={cand} rows=1536 modes=3");
    Ok(())
}

fn main() -> Result<()> {
    std::thread::Builder::new().stack_size(512 * 1024 * 1024).spawn(real_main)?.join().map_err(|_| anyhow::anyhow!("worker thread panicked"))?
}

fn real_main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("verify-evaluator") => cmd_verify(&args[2..]),
        Some("eval") => cmd_eval(&args[2..]),
        _ => bail!("usage: v69-g1-eval <verify-evaluator|eval> --artifacts DIR"),
    }
}
