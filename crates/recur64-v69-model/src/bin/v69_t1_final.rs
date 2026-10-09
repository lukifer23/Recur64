//! v69-t1-final: frozen-candidate evaluator for T1 (inference only).
//!   v69-t1-final verify-evaluator --artifacts DIR     validation-only fixture; writes an append-only receipt
//!   v69-t1-final eval --candidate A|M|B|B2 --artifacts DIR   ONE registered invocation per candidate
//! Role: T1 frozen-candidate evaluator (test rows, G1 rows, frozen candidate files, frozen intervention maps).
//! A and M run on CUDA (no CPU fallback); B and B2 (baselines) are host-side f64 inference.

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
        Ok(Self { custody: Custody::new(Path::new(&arg(args, "--artifacts").context("--artifacts")?))?, run: PathBuf::from("gen-001"), seed_rel: PathBuf::from("t1/seed/t1_master_seed.hex") })
    }
    fn access(&self) -> Result<Access> {
        Access::new(&self.custody, Role::T1Evaluator, &self.run, &self.seed_rel)
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

struct Data {
    rows: Vec<ModelRow>,
    feats: Vec<Features>,
    fb: Vec<(String, u8)>,
    sha: String,
}

fn load_rows(a: &Access, rel: &str) -> Result<Data> {
    let bytes = a.read(Path::new(rel))?;
    let rows = read_rows(std::str::from_utf8(&bytes)?)?;
    let mut feats = Vec::new();
    let mut fb = Vec::new();
    for r in &rows {
        feats.push(featurize(&r.fen, r.budget)?.0);
        fb.push((r.fen.clone(), r.budget));
    }
    Ok(Data { rows, feats, fb, sha: sha256_hex(&bytes) })
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

enum Net {
    A(Model<G>),
    M(Mlp<G>),
}

impl Net {
    fn logits(&self, fs: &[&Features], dev: &CudaDevice) -> Vec<f64> {
        let mut out = Vec::new();
        for chunk in fs.chunks(16) {
            let z = match self {
                Net::A(m) => a_logits(m, chunk, dev),
                Net::M(m) => m_logits(m, chunk, dev),
            };
            out.extend(z.into_iter().map(|x| x as f64));
        }
        out
    }
}

fn baseline_from(a: &Access, rel: &str) -> Result<BaselineModel> {
    let v: serde_json::Value = serde_json::from_slice(&a.read(Path::new(rel))?)?;
    Ok(serde_json::from_value(v["model"].clone())?)
}

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

fn selection(a: &Access) -> Result<serde_json::Value> {
    Ok(serde_json::from_slice(&a.read(Path::new("t1/report/selection.json"))?)?)
}

fn ckpt_dir(a: &Access, run_id: &str) -> Result<PathBuf> {
    Ok(a.input_path(Path::new(&format!("t1/train_runs/{run_id}/final/meta.json")))?.parent().unwrap().to_path_buf())
}

fn load_net(a: &Access, cand: &str, run_id: &str, dev: &CudaDevice) -> Result<Net> {
    let ck = ckpt_dir(a, run_id)?;
    Ok(match cand {
        "A" => Net::A(load_model_for_eval::<G>(&ck, dev)?),
        _ => Net::M(load_mlp_eval::<G>(&ck, dev)?),
    })
}

// ---------------------------------------------------------------- verify-evaluator

fn cmd_verify(args: &[String]) -> Result<()> {
    let env = Env::from(args)?;
    let a = env.access()?;
    let dev = cuda_device()?;
    let sel = selection(&a)?;
    let val = load_rows(&a, "t1/rows/val.jsonl")?;
    let refs: Vec<&Features> = val.feats.iter().collect();
    let mut checks = Vec::new();
    for cand in ["A", "M"] {
        let run_id = sel[cand]["run_id"].as_str().context("selected run")?;
        let net = load_net(&a, cand, run_id, &dev)?;
        let z = net.logits(&refs, &dev);
        let recorded: HashMap<String, f64> = jsonl_vals(&a.read_to_string(Path::new(&format!("t1/train_runs/{run_id}/val_predictions.jsonl")))?)?
            .into_iter()
            .filter(|v| v["update"] == sel[cand]["final_update"])
            .map(|v| (v["id"].as_str().unwrap().to_string(), v["logit"].as_f64().unwrap()))
            .collect();
        ensure!(recorded.len() == val.rows.len(), "recorded predictions for {cand}");
        let worst = val.rows.iter().zip(&z).map(|(r, zz)| (zz - recorded[&r.id]).abs()).fold(0.0, f64::max);
        let agree = val.rows.iter().zip(&z).filter(|(r, zz)| (**zz > 0.0) == (recorded[&r.id] > 0.0)).count();
        // target separation: features do not depend on label or id
        let again: Vec<Features> = val.rows.iter().take(8).map(|r| featurize(&r.fen, r.budget).unwrap().0).collect();
        let sep = again.iter().zip(val.feats.iter()).all(|(x, y)| x == y);
        // TTA machinery: identity transform equals the ordinary input
        let ident = val.feats.iter().take(8).all(|f| f.d8(0) == *f);
        checks.push(json!({"candidate": cand, "run_id": run_id, "max_abs_logit_diff_vs_recorded_val_final": worst, "tolerance": TOL_NEURAL, "classification_agreement": agree, "of": val.rows.len(), "target_separation": sep, "d8_identity": ident, "pass": worst <= TOL_NEURAL && agree + 6 >= val.rows.len() && sep && ident}));
    }
    for (cand, rel, pred) in [("B", "d1/baseline/model.json", None), ("B2", "t1/baseline_refit/model.json", Some("t1/baseline_refit/predictions_val.jsonl"))] {
        let Some(pred) = pred else {
            // B has no T1 validation predictions; check determinism + erasure definition consistency
            let bm = baseline_from(&a, rel)?;
            let x = baseline_features(&val.fb[0].0, val.fb[0].1)?;
            let ok = (bm.logit(&x) - bm.logit(&baseline_features(&val.fb[0].0, val.fb[0].1)?)).abs() == 0.0;
            let be = baseline_erased(&bm, 2);
            let eok = (0..11).all(|j| be[j] == bm.mean[j]) && be[11] == 1.0;
            checks.push(json!({"candidate": cand, "deterministic": ok, "erasure_definition_consistent": eok, "pass": ok && eok}));
            continue;
        };
        let bm = baseline_from(&a, rel)?;
        let rec: HashMap<String, f64> = jsonl_vals(&a.read_to_string(Path::new(pred))?)?.into_iter().map(|v| (v["id"].as_str().unwrap().to_string(), v["logit"].as_f64().unwrap())).collect();
        let mut worst = 0f64;
        for (r, (fen, b)) in val.rows.iter().zip(&val.fb) {
            worst = worst.max((bm.logit(&baseline_features(fen, *b)?) - rec[&r.id]).abs());
        }
        checks.push(json!({"candidate": cand, "max_abs_logit_diff_vs_recorded_refit_val": worst, "tolerance": TOL_BASELINE, "pass": worst <= TOL_BASELINE}));
    }
    let er = val.feats[0].erased();
    checks.push(json!({"name": "neural_erasure_definition", "pass": er.piece.iter().all(|p| *p == 0) && er.budget == val.feats[0].budget}));
    let ok = checks.iter().all(|c| c["pass"].as_bool() == Some(true));
    let rep = json!({"ok": ok, "fixture": "T1 validation rows (no test/G1 labels or outputs touched)", "val_rows_sha256": val.sha, "checks": checks, "source": source_id(), "evaluator_source_digest": source_digest()});
    write_new(&a, "t1/receipts/evaluator_verification.json", serde_json::to_string_pretty(&rep)?.as_bytes())?;
    println!("{}", serde_json::to_string_pretty(&rep)?);
    if !ok {
        std::process::exit(5);
    }
    Ok(())
}

// ---------------------------------------------------------------- eval

struct Mode {
    real: Vec<f64>,
    der: Vec<f64>,
    ers: Vec<f64>,
    tta: Option<Vec<f64>>,
}

fn cmd_eval(args: &[String]) -> Result<()> {
    let t_proc = Instant::now();
    let env = Env::from(args)?;
    let a = env.access()?;
    let cand = arg(args, "--candidate").context("--candidate A|M|B|B2")?;
    ensure!(["A", "M", "B", "B2"].contains(&cand.as_str()), "candidate must be A, M, B or B2");
    let out = format!("t1/final/{cand}");
    ensure!(!a.custody().resolve(Path::new(&out))?.join("provenance.json").exists(), "candidate {cand} already evaluated: one registered evaluation only");
    let frozen: FrozenD1 = serde_json::from_slice(&a.read(Path::new("t1/frozen_final.json"))?)?;
    let nv = verify_group(&a, &env.run, &frozen, "evaluator")?;
    let src: serde_json::Value = serde_json::from_slice(&a.read(Path::new("t1/evaluator_source.json"))?)?;
    let sid = source_id();
    ensure!(sid.git_dirty_files == 0, "evaluator must run from a clean tree ({} dirty files)", sid.git_dirty_files);
    ensure!(src["source_digest"].as_str() == Some(source_digest().as_str()), "EVALUATOR SOURCE DIGEST MISMATCH: the verified evaluator source is frozen");
    eprintln!("[t1 final {cand}] {nv} frozen hashes verified; source digest matches");
    let sel = selection(&a)?;
    let mut sets: Vec<(&str, Data, HashMap<String, String>)> = Vec::new();
    for (name, rows_rel, map_rel, n) in [("test", "t1/rows/test.jsonl", "t1/intervention/test_map.json", 3072usize), ("g1", "g1r1/rows/g1_rows.jsonl", "g1r1/intervention/map.json", 1536usize)] {
        let d = load_rows(&a, rows_rel)?;
        ensure!(d.rows.len() == n, "{name} rows");
        let mv: serde_json::Value = serde_json::from_slice(&a.read(Path::new(map_rel))?)?;
        let map: HashMap<String, String> = mv["map"].as_object().context("map")?.iter().map(|(k, v)| (k.clone(), v.as_str().unwrap().to_string())).collect();
        sets.push((name, d, map));
    }
    let mut timing = serde_json::Map::new();
    let mut files = serde_json::Map::new();
    let mut results: Vec<(&str, Mode, Vec<usize>)> = Vec::new();
    let t_start;
    let neural = cand == "A" || cand == "M";
    let mut run_id_used = String::new();
    if neural {
        let t0 = Instant::now();
        let dev = cuda_device()?;
        run_id_used = sel[cand.as_str()]["run_id"].as_str().context("selected run")?.to_string();
        let ck = ckpt_dir(&a, &run_id_used)?;
        for f in ["model.mpk", "meta.json"] {
            files.insert(format!("t1/train_runs/{run_id_used}/final/{f}"), json!(sha256_hex(&std::fs::read(ck.join(f))?)));
        }
        let net = load_net(&a, &cand, &run_id_used, &dev)?;
        t_start = t0.elapsed().as_secs_f64() * 1e3;
        for (name, d, map) in &sets {
            let idx: HashMap<&str, usize> = d.rows.iter().enumerate().map(|(i, r)| (r.id.as_str(), i)).collect();
            let donor_of: Vec<usize> = d.rows.iter().map(|r| idx[map[&r.id].as_str()]).collect();
            ensure!(donor_of.iter().enumerate().all(|(i, dd)| *dd != i), "frozen map has a fixed point");
            let drefs: Vec<Features> = (0..d.rows.len()).map(|i| { let mut f = d.feats[donor_of[i]].clone(); f.budget = d.feats[i].budget; f }).collect();
            let erefs: Vec<Features> = d.feats.iter().map(|f| f.erased()).collect();
            let t1 = Instant::now();
            let r: Vec<&Features> = d.feats.iter().collect();
            let real = net.logits(&r, &dev);
            timing.insert(format!("{name}_real_pass_incl_first_batch_jit_ms"), json!(t1.elapsed().as_secs_f64() * 1e3));
            let dr: Vec<&Features> = drefs.iter().collect();
            let der = net.logits(&dr, &dev);
            let er: Vec<&Features> = erefs.iter().collect();
            let ers = net.logits(&er, &dev);
            // report-only 8-fold board-symmetry test-time augmentation (mean logit)
            let mut sum = vec![0.0f64; d.rows.len()];
            for t in 0..8 {
                let tf: Vec<Features> = d.feats.iter().map(|f| f.d8(t)).collect();
                let tr: Vec<&Features> = tf.iter().collect();
                for (s, z) in sum.iter_mut().zip(net.logits(&tr, &dev)) {
                    *s += z / 8.0;
                }
            }
            results.push((name, Mode { real, der, ers, tta: Some(sum) }, donor_of));
        }
        let first16: Vec<&Features> = sets[0].1.feats.iter().take(16).collect();
        let mut lat = Vec::new();
        for i in 0..25 {
            let t = Instant::now();
            let _ = net.logits(&first16, &dev);
            if i >= 5 {
                lat.push(t.elapsed().as_secs_f64() * 1e3);
            }
        }
        lat.sort_by(|x, y| x.partial_cmp(y).unwrap());
        timing.insert("warmed_synchronized_batch16_median_ms".into(), json!(lat[lat.len() / 2]));
        let all: Vec<&Features> = sets[0].1.feats.iter().collect();
        let tw = Instant::now();
        let _ = net.logits(&all, &dev);
        timing.insert("warm_full_test_pass_inference_only_ms".into(), json!(tw.elapsed().as_secs_f64() * 1e3));
        timing.insert("batch_size".into(), json!(16));
    } else {
        let t0 = Instant::now();
        let rel = if cand == "B" { "d1/baseline/model.json" } else { "t1/baseline_refit/model.json" };
        let bm = baseline_from(&a, rel)?;
        files.insert(rel.into(), json!(sha256_hex(&a.read(Path::new(rel))?)));
        t_start = t0.elapsed().as_secs_f64() * 1e3;
        for (name, d, map) in &sets {
            let idx: HashMap<&str, usize> = d.rows.iter().enumerate().map(|(i, r)| (r.id.as_str(), i)).collect();
            let donor_of: Vec<usize> = d.rows.iter().map(|r| idx[map[&r.id].as_str()]).collect();
            let tf = Instant::now();
            let raw: Vec<[f64; N_BASELINE]> = d.fb.iter().map(|(fen, b)| baseline_features(fen, *b)).collect::<Result<_>>()?;
            timing.insert(format!("{name}_feature_extraction_total_ms"), json!(tf.elapsed().as_secs_f64() * 1e3));
            let real: Vec<f64> = raw.iter().map(|r| bm.logit(r)).collect();
            let der: Vec<f64> = (0..d.rows.len()).map(|i| bm.logit(&raw[donor_of[i]])).collect();
            let ers: Vec<f64> = d.fb.iter().map(|(_, b)| bm.logit(&baseline_erased(&bm, *b))).collect();
            results.push((name, Mode { real, der, ers, tta: None }, donor_of));
        }
        timing.insert("device".into(), json!("host f64 (no GPU)"));
    }
    timing.insert("startup_cuda_init_and_model_load_ms".into(), json!(t_start));
    let mut preds_sha = serde_json::Map::new();
    for (name, m, donor_of) in &results {
        let d = &sets.iter().find(|s| s.0 == *name).unwrap().1;
        let mut text = String::new();
        let mut emit = |mode: &str, z: &Vec<f64>, donor: bool| {
            for (i, r) in d.rows.iter().enumerate() {
                let mut v = json!({"id": r.id, "mode": mode, "logit": z[i]});
                if donor {
                    v["donor_id"] = json!(d.rows[donor_of[i]].id);
                }
                text.push_str(&serde_json::to_string(&v).unwrap());
                text.push('\n');
            }
        };
        emit("real", &m.real, false);
        emit("derange", &m.der, true);
        emit("erase", &m.ers, false);
        if let Some(t) = &m.tta {
            emit("tta8", t, false);
        }
        a.write(Path::new(&format!("{out}/predictions_{name}.jsonl")), text.as_bytes())?;
        preds_sha.insert(name.to_string(), json!(sha256_hex(text.as_bytes())));
    }
    timing.insert("evaluator_process_wall_ms_excluding_aggregation".into(), json!(t_proc.elapsed().as_secs_f64() * 1e3));
    let mut data_sha = serde_json::Map::new();
    for (name, d, _) in &sets {
        data_sha.insert(name.to_string(), json!(d.sha));
    }
    let mut prov = json!({
        "candidate": cand, "selected_run_id": run_id_used, "evaluator_source": {"git_head": sid.git_head, "digest": src["source_digest"]},
        "frozen_final_sha256": sha256_hex(&a.read(Path::new("t1/frozen_final.json"))?),
        "selection_sha256": sha256_hex(&a.read(Path::new("t1/report/selection.json"))?),
        "rows_sha256": data_sha, "candidate_files": files, "predictions_sha256": preds_sha,
        "precision": "f32 storage/accumulation; matmul inputs possibly TF32 (not strict FP32); baselines host f64",
        "timing": timing,
    });
    let id = sha256_hex(prov.to_string().as_bytes());
    prov["provenance_id"] = json!(id);
    a.write(Path::new(&format!("{out}/provenance.json")), serde_json::to_string_pretty(&prov)?.as_bytes())?;
    println!("T1 EVAL COMPLETE candidate={cand}");
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
        _ => bail!("usage: v69-t1-final <verify-evaluator|eval> --artifacts DIR"),
    }
}
