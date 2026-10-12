//! v69-t3: held-out material-family transfer (T3). CUDA only for training/inference of the MLP; baselines host f64.
//!   v69-t3 run --artifacts DIR --fold KQQvK|KQRvK|KRRvK --recipe S|D    one registered run (train on the other two families, evaluate on the held-out family)
//!   v69-t3 aggregate --artifacts DIR                                    metrics, paired cluster bootstrap, decision flags
//! Reads only T2 train/val rows and metadata, the D1 MLP init, the seed and the frozen D1 baseline. Refuses any path containing `test`, `g1` or `sealed`.

use anyhow::{Context, Result, bail, ensure};
use burn::backend::cuda::CudaDevice;
use burn::prelude::*;
use recur64_v69::d1::{BaselineModel, N_BASELINE, baseline_features, fit_baseline};
use recur64_v69::d1_metrics::auroc;
use recur64_v69::features::{Features, featurize, read_rows};
use recur64_v69::provenance::{sha256_hex, source_id};
use recur64_v69::streams::{MasterSeed, keyed_u64};
use recur64_v69_model::d1::*;
use recur64_v69_model::init;
use recur64_v69_model::inventory::{dump_values, load_values};
use recur64_v69_model::qualify::{G, GA};
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

const BATCH: usize = 64;
const WARMUP: usize = 200;
const FAMILIES: [&str; 3] = ["KQQvK", "KQRvK", "KRRvK"];
const RESAMPLES: usize = 5000;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

struct Store(PathBuf);

impl Store {
    fn guard(rel: &str) -> Result<()> {
        let l = rel.to_ascii_lowercase();
        ensure!(!["test", "g1", "sealed"].iter().any(|w| l.contains(w)), "T3 refuses to open {rel}");
        Ok(())
    }
    fn read(&self, rel: &str) -> Result<Vec<u8>> {
        Self::guard(rel)?;
        std::fs::read(self.0.join(rel)).with_context(|| format!("read {rel}"))
    }
    fn text(&self, rel: &str) -> Result<String> {
        Ok(String::from_utf8(self.read(rel)?)?)
    }
    fn write_new(&self, rel: &str, bytes: &[u8]) -> Result<()> {
        Self::guard(rel)?;
        let p = self.0.join(rel);
        ensure!(!p.exists(), "{rel} already exists (append-only)");
        std::fs::create_dir_all(p.parent().unwrap())?;
        std::fs::write(p, bytes)?;
        Ok(())
    }
}

fn cuda_device() -> Result<CudaDevice> {
    let d = CudaDevice::default();
    G::sync(&d).map_err(|e| anyhow::anyhow!("CUDA device unavailable (no CPU fallback): {e:?}"))?;
    Ok(d)
}

struct Row {
    id: String,
    fen: String,
    budget: u8,
    label: bool,
    family: String,
    group: String,
}

fn load_part(s: &Store, rows_rel: &str, meta_rel: &str) -> Result<Vec<Row>> {
    let rows = read_rows(&s.text(rows_rel)?)?;
    let mut meta: HashMap<String, (String, String)> = HashMap::new();
    for line in s.text(meta_rel)?.lines().filter(|l| !l.trim().is_empty()) {
        let v: Value = serde_json::from_str(line)?;
        meta.insert(v["id"].as_str().context("id")?.to_string(), (v["family"].as_str().context("family")?.to_string(), v["group_id"].as_str().context("group_id")?.to_string()));
    }
    rows.into_iter()
        .map(|r| {
            let (family, group) = meta.get(&r.id).with_context(|| format!("no metadata for {}", r.id))?.clone();
            Ok(Row { id: r.id, fen: r.fen, budget: r.budget, label: r.label, family, group })
        })
        .collect()
}

fn lr_sched(u: usize, total: usize, peak: f64) -> f64 {
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

#[derive(Default, Clone)]
struct Counts {
    tn: f64,
    fp: f64,
    fnn: f64,
    tp: f64,
    bce: f64,
    n: f64,
}

impl Counts {
    fn add(&mut self, z: f64, y: bool) {
        match (y, z > 0.0) {
            (true, true) => self.tp += 1.0,
            (true, false) => self.fnn += 1.0,
            (false, false) => self.tn += 1.0,
            (false, true) => self.fp += 1.0,
        }
        self.bce += z.max(0.0) - z * (y as u8 as f64) + (-z.abs()).exp().ln_1p();
        self.n += 1.0;
    }
    fn merge(&mut self, o: &Counts) {
        self.tn += o.tn;
        self.fp += o.fp;
        self.fnn += o.fnn;
        self.tp += o.tp;
        self.bce += o.bce;
        self.n += o.n;
    }
    fn ba(&self) -> f64 {
        0.5 * (self.tp / (self.tp + self.fnn).max(1.0) + self.tn / (self.tn + self.fp).max(1.0))
    }
    fn mean_bce(&self) -> f64 {
        self.bce / self.n.max(1.0)
    }
}

fn baseline_logits(m: &BaselineModel, rows: &[&Row]) -> Result<Vec<f64>> {
    rows.iter().map(|r| Ok(m.logit(&baseline_features(&r.fen, r.budget)?))).collect()
}

fn cmd_run(args: &[String]) -> Result<()> {
    let s = Store(PathBuf::from(arg(args, "--artifacts").context("--artifacts")?));
    let fold = arg(args, "--fold").context("--fold")?;
    ensure!(FAMILIES.contains(&fold.as_str()), "fold must be one of {FAMILIES:?}");
    let recipe = arg(args, "--recipe").context("--recipe S|D")?;
    let (mode, peak, epochs) = match recipe.as_str() {
        "S" => ("sym", 3e-3f64, 30usize),
        "D" => ("d8", 3e-3f64, 60usize),
        _ => bail!("recipe must be S or D"),
    };
    let max_updates: Option<usize> = arg(args, "--max-updates").map(|x| x.parse()).transpose()?;
    let run_id = format!("{fold}_{recipe}{}", if max_updates.is_some() { "_smoke" } else { "" });
    let out = format!("t3/fold_runs/{run_id}");
    ensure!(!s.0.join(&out).join("provenance.json").exists(), "run {run_id} already exists; no reruns");
    let sid = source_id();
    ensure!(sid.git_dirty_files == 0, "T3 runs require a clean tree ({} dirty files)", sid.git_dirty_files);
    let seed = MasterSeed::from_hex(s.text("t2/seed/t2_master_seed.hex")?.trim())?;
    let train_all = load_part(&s, "t2/rows/train.jsonl", "t2/meta/train.meta.jsonl")?;
    let val_all = load_part(&s, "t2/rows/val.jsonl", "t2/meta/val.meta.jsonl")?;
    let train: Vec<&Row> = train_all.iter().filter(|r| r.family != fold).collect();
    let held: Vec<&Row> = train_all.iter().filter(|r| r.family == fold).collect();
    let val_in: Vec<&Row> = val_all.iter().filter(|r| r.family != fold).collect();
    let val_out: Vec<&Row> = val_all.iter().filter(|r| r.family == fold).collect();
    ensure!(train.len() == 48_000 && held.len() == 24_000 && val_in.len() == 2_048 && val_out.len() == 1_024, "unexpected partition sizes {} {} {} {}", train.len(), held.len(), val_in.len(), val_out.len());
    ensure!(train.iter().all(|r| r.family != fold) && held.iter().all(|r| r.family == fold), "family isolation");
    let feat = |rs: &[&Row]| -> Result<Vec<Features>> { rs.iter().map(|r| featurize(&r.fen, r.budget).map(|x| x.0)).collect() };
    let (tf, hf, vif, vof) = (feat(&train)?, feat(&held)?, feat(&val_in)?, feat(&val_out)?);
    let labels: Vec<bool> = train.iter().map(|r| r.label).collect();
    eprintln!("[t3 {run_id}] train {} held-out {} ({fold}); mode {mode} lr {peak} epochs {epochs}", train.len(), held.len());

    let dev = cuda_device()?;
    let init_bytes = s.read("d1/mlp_init.bin")?;
    let (_, vals) = init::from_bytes(&init_bytes)?;
    let model = load_values::<GA, _>(Mlp::<GA>::new(&dev), &vals, &dev);
    ensure!(init::tensors_hash(&dump_values::<GA, _>(&model)) == init::tensors_hash(&vals), "init mismatch");
    let mut tr = DTrainer::<GA, Mlp<GA>>::new(model, &dev);
    let n = train.len();
    let ups = n / BATCH;
    let total = max_updates.unwrap_or(epochs * ups);
    let order_label = format!("t3_order/{run_id}");
    let aug_label = format!("t3_aug/{run_id}");
    let mut cache: Option<(u64, Vec<usize>)> = None;
    let mut trace: Vec<Value> = Vec::new();
    let (mut win, mut win_n) = (0.0, 0usize);
    let t_all = std::time::Instant::now();
    for u in 0..total {
        let epoch = (u / ups) as u64;
        if cache.as_ref().map(|c| c.0) != Some(epoch) {
            cache = Some((epoch, permutation(&seed, &order_label, epoch, n)));
        }
        let perm = &cache.as_ref().unwrap().1;
        let base = (u % ups) * BATCH;
        let mut fs: Vec<Features> = Vec::with_capacity(BATCH);
        let mut ls: Vec<bool> = Vec::with_capacity(BATCH);
        for j in 0..BATCH {
            let i = perm[base + j];
            fs.push(if mode == "d8" { tf[i].d8((keyed_u64(&seed, &aug_label, &((u * BATCH + j) as u64).to_le_bytes()) % 8) as usize) } else { tf[i].clone() });
            ls.push(labels[i]);
        }
        let micro = vec![(fs.iter().collect::<Vec<&Features>>(), ls)];
        let lr = lr_sched(u, total, peak);
        let st = if mode == "sym" { tr.step_lr_sym(&micro, lr) } else { tr.step_lr(&micro, lr) };
        if !(st.loss.is_finite() && st.grad_norm_pre_clip.is_finite()) {
            bail!("non-finite loss/gradient at update {u}: INCOMPLETE (no restart)");
        }
        win += st.loss;
        win_n += 1;
        if (u + 1) % 500 == 0 || u + 1 == total {
            trace.push(json!({"update": u + 1, "lr": lr, "train_loss_mean_last_window": win / win_n as f64, "grad_norm": st.grad_norm_pre_clip}));
            println!("{}", trace.last().unwrap());
            win = 0.0;
            win_n = 0;
        }
    }
    let wall = t_all.elapsed().as_secs_f64();
    // registered inference: sym -> mean logit over the 8 board symmetries; d8 -> ordinary logit
    let infer = |fs: &[Features]| -> Vec<f64> {
        if mode == "sym" {
            let mut sum = vec![0f64; fs.len()];
            for t in 0..8 {
                let views: Vec<Features> = fs.iter().map(|f| f.d8(t)).collect();
                for (a, x) in sum.iter_mut().zip(panel_logits::<GA, Mlp<GA>>(&tr.model, &views, &dev)) {
                    *a += x as f64 / 8.0;
                }
            }
            sum
        } else {
            panel_logits::<GA, Mlp<GA>>(&tr.model, fs, &dev).into_iter().map(|x| x as f64).collect()
        }
    };
    let zh = infer(&hf);
    let (zvi, zvo) = (infer(&vif), infer(&vof));
    // baselines: B2F refit on this fold's training rows; B = frozen D1 baseline
    let xf: Vec<[f64; N_BASELINE]> = train.iter().map(|r| baseline_features(&r.fen, r.budget)).collect::<Result<_>>()?;
    let b2f = fit_baseline(&xf, &labels);
    let zb2f = baseline_logits(&b2f, &held)?;
    let bv: Value = serde_json::from_slice(&s.read("d1/baseline/model.json")?)?;
    let b: BaselineModel = serde_json::from_value(bv["model"].clone())?;
    let zb = baseline_logits(&b, &held)?;
    let diag = |zs: &[f64], rs: &[&Row]| {
        let mut c = Counts::default();
        for (z, r) in zs.iter().zip(rs) {
            c.add(*z, r.label);
        }
        json!({"n": rs.len(), "balanced_accuracy": c.ba(), "bce": c.mean_bce()})
    };
    let mut ptext = String::new();
    for (i, r) in held.iter().enumerate() {
        ptext += &(json!({"id": r.id, "group": r.group, "budget": r.budget, "label": r.label, "z_model": zh[i], "z_b2f": zb2f[i], "z_b": zb[i]}).to_string() + "\n");
    }
    s.write_new(&format!("{out}/predictions_heldout.jsonl"), ptext.as_bytes())?;
    let ck = s.0.join(&out).join("final");
    std::fs::create_dir_all(&ck)?;
    save_ckpt(&ck, &tr)?;
    let mut ckhash = serde_json::Map::new();
    for f in ["model.mpk", "meta.json"] {
        ckhash.insert(f.into(), json!(sha256_hex(&std::fs::read(ck.join(f))?)));
    }
    let diagnostics = json!({"val_trained_families": diag(&zvi, &val_in), "val_heldout_family": diag(&zvo, &val_out)});
    s.write_new(&format!("{out}/trace.json"), serde_json::to_string(&json!({"trace": trace, "wall_secs": wall, "diagnostics_report_only": diagnostics}))?.as_bytes())?;
    let mut prov = json!({
        "run_id": run_id, "fold": fold, "recipe": recipe, "mode": mode, "peak_lr": peak, "epochs": epochs, "batch": BATCH, "updates": total, "completed_updates": tr.update,
        "n_train": n, "n_heldout": held.len(), "consumer_source": source_id(),
        "train_rows_sha256": sha256_hex(&s.read("t2/rows/train.jsonl")?), "train_meta_sha256": sha256_hex(&s.read("t2/meta/train.meta.jsonl")?),
        "init_file_sha256": sha256_hex(&init_bytes), "checkpoint_file_sha256": ckhash, "predictions_heldout_sha256": sha256_hex(ptext.as_bytes()),
        "contract_sha256": sha256_hex(&std::fs::read(s.0.join("../../docs/v69/T3_CONTRACT.md")).unwrap_or_default()),
        "wall_secs": wall, "precision": "f32 storage/accumulation; matmul inputs possibly TF32 (not strict FP32); baselines host f64",
    });
    let id = sha256_hex(prov.to_string().as_bytes());
    prov["provenance_id"] = json!(id);
    s.write_new(&format!("{out}/provenance.json"), serde_json::to_string_pretty(&prov)?.as_bytes())?;
    println!("T3 RUN COMPLETE {run_id} updates={} wall={wall:.1}s diagnostics={diagnostics}", tr.update);
    Ok(())
}

// ------------------------------------------------------------------ aggregation

struct Pred {
    group: usize,
    budget: u8,
    label: bool,
    zs: [f64; 3], // model, b2f, b
}

fn pct(v: &mut [f64], q: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * q).round() as usize]
}

fn cmd_aggregate(args: &[String]) -> Result<()> {
    let s = Store(PathBuf::from(arg(args, "--artifacts").context("--artifacts")?));
    let sid = source_id();
    ensure!(sid.git_dirty_files == 0, "aggregation requires a clean tree");
    let seed = MasterSeed::from_hex(s.text("t2/seed/t2_master_seed.hex")?.trim())?;
    let t2: Value = serde_json::from_str(&std::fs::read_to_string(s.0.join("../../docs/v69/evidence/t2/t2_report.json")).context("T2 report")?)?;
    let mut report = serde_json::Map::new();
    let mut summary: BTreeMap<String, Vec<bool>> = BTreeMap::new();
    for fold in FAMILIES {
        // in-family reference from the spent T2 test report (equal-sized cells): BA = mean over the four family cells' accuracies
        let refba = |cand: &str| -> f64 {
            let cells = &t2["test"]["candidates"][cand]["real"]["per_cell"];
            let v: Vec<f64> = cells.as_object().unwrap().iter().filter(|(k, _)| k.starts_with(fold)).map(|(_, c)| c["acc"].as_f64().unwrap()).collect();
            v.iter().sum::<f64>() / v.len() as f64
        };
        for recipe in ["S", "D"] {
            let run_id = format!("{fold}_{recipe}");
            let text = s.text(&format!("t3/fold_runs/{run_id}/predictions_heldout.jsonl"))?;
            let prov: Value = serde_json::from_str(&s.text(&format!("t3/fold_runs/{run_id}/provenance.json"))?)?;
            ensure!(sha256_hex(text.as_bytes()) == prov["predictions_heldout_sha256"].as_str().unwrap(), "{run_id}: prediction hash mismatch");
            let mut gidx: HashMap<String, usize> = HashMap::new();
            let mut preds: Vec<Pred> = Vec::new();
            for line in text.lines() {
                let v: Value = serde_json::from_str(line)?;
                let g = v["group"].as_str().unwrap().to_string();
                let next = gidx.len();
                let gi = *gidx.entry(g).or_insert(next);
                preds.push(Pred { group: gi, budget: v["budget"].as_u64().unwrap() as u8, label: v["label"].as_bool().unwrap(), zs: [v["z_model"].as_f64().unwrap(), v["z_b2f"].as_f64().unwrap(), v["z_b"].as_f64().unwrap()] });
            }
            ensure!(preds.len() == 24_000, "{run_id}: {} predictions", preds.len());
            let ng = gidx.len();
            let mut cnt = vec![Counts::default(); 3];
            let mut pg: Vec<Vec<Counts>> = vec![vec![Counts::default(); ng]; 3];
            let mut cells: BTreeMap<String, Counts> = BTreeMap::new();
            for p in &preds {
                for k in 0..3 {
                    cnt[k].add(p.zs[k], p.label);
                    pg[k][p.group].add(p.zs[k], p.label);
                }
                cells.entry(format!("n{}/{}", p.budget, if p.label { "pos" } else { "neg" })).or_default().add(p.zs[0], p.label);
            }
            // derangement control: cyclic shift of a keyed shuffle within budget cells (label-independent); prediction of the donor board, scored against recipient label
            let mut der = Counts::default();
            for b in [1u8, 2u8] {
                let mut idx: Vec<usize> = (0..preds.len()).filter(|&i| preds[i].budget == b).collect();
                idx.sort_by_key(|&i| keyed_u64(&seed, &format!("t3_derange/{run_id}"), &(i as u64).to_le_bytes()));
                for j in 0..idx.len() {
                    der.add(preds[idx[(j + 1) % idx.len()]].zs[0], preds[idx[j]].label);
                }
            }
            let zs: Vec<f64> = preds.iter().map(|p| p.zs[0]).collect();
            let ys: Vec<bool> = preds.iter().map(|p| p.label).collect();
            let (mut dba, mut dbce, mut dbr_b) = (Vec::new(), Vec::new(), Vec::new());
            for r in 0..RESAMPLES {
                let mut rng = seed.stream(&format!("t3_boot/{fold}"), r as u64);
                let (mut m, mut b2, mut b) = (Counts::default(), Counts::default(), Counts::default());
                for _ in 0..ng {
                    let g = rng.below(ng as u64) as usize;
                    m.merge(&pg[0][g]);
                    b2.merge(&pg[1][g]);
                    b.merge(&pg[2][g]);
                }
                dba.push(m.ba() - b2.ba());
                dbce.push(m.mean_bce() - b2.mean_bce());
                dbr_b.push(m.ba() - b.ba());
            }
            let (ba_gain, bce_diff) = (cnt[0].ba() - cnt[1].ba(), cnt[0].mean_bce() - cnt[1].mean_bce());
            let (lo, hi) = (pct(&mut dba.clone(), 0.025), pct(&mut dba, 0.975));
            let (blo, bhi) = (pct(&mut dbce.clone(), 0.025), pct(&mut dbce, 0.975));
            let (clo, chi) = (pct(&mut dbr_b.clone(), 0.025), pct(&mut dbr_b, 0.975));
            let drop_pp = (cnt[0].ba() - der.ba()) * 100.0;
            let transfer = cnt[0].ba() >= 0.75 && cnt[0].mean_bce() <= 0.55 && ba_gain * 100.0 >= 5.0 && lo > 0.0 && bce_diff <= 0.0 && drop_pp >= 15.0;
            summary.entry(recipe.to_string()).or_default().push(transfer);
            let cell_json: BTreeMap<String, Value> = cells.iter().map(|(k, c)| (k.clone(), json!({"accuracy": (c.tp + c.tn) / c.n, "mean_bce": c.mean_bce(), "n": c.n}))).collect();
            report.insert(
                run_id.clone(),
                json!({
                    "fold": fold, "recipe": recipe, "n_groups": ng,
                    "model": {"balanced_accuracy": cnt[0].ba(), "bce": cnt[0].mean_bce(), "auroc": auroc(&zs, &ys), "confusion": {"tn": cnt[0].tn, "fp": cnt[0].fp, "fn": cnt[0].fnn, "tp": cnt[0].tp}, "per_cell": cell_json},
                    "b2f": {"balanced_accuracy": cnt[1].ba(), "bce": cnt[1].mean_bce()},
                    "b_frozen_d1_report_only": {"balanced_accuracy": cnt[2].ba(), "bce": cnt[2].mean_bce(), "model_minus_b_ba_ci95": [clo, chi]},
                    "paired_vs_b2f": {"ba_gain_pp": ba_gain * 100.0, "ba_ci95_pp": [lo * 100.0, hi * 100.0], "bce_diff": bce_diff, "bce_ci95": [blo, bhi]},
                    "derangement": {"balanced_accuracy_vs_recipient_labels": der.ba(), "drop_pp": drop_pp},
                    "in_family_reference_t2_test_report_only": {"candidate": if recipe == "S" { "M" } else { "P" }, "family_accuracy": refba(if recipe == "S" { "M" } else { "P" })},
                    "transfer_criterion_met": transfer,
                    "diagnostics_val_report_only": serde_json::from_str::<Value>(&s.text(&format!("t3/fold_runs/{run_id}/trace.json"))?)?["diagnostics_report_only"].clone(),
                }),
            );
        }
    }
    let verdict: serde_json::Map<String, Value> = summary.iter().map(|(r, v)| (r.clone(), json!({"folds_passing": v.iter().filter(|x| **x).count(), "of": v.len(), "claim": if v.iter().all(|x| *x) { "family-general" } else if v.iter().any(|x| *x) { "partially family-general" } else { "family-specific" }}))).collect();
    let rep = json!({"source": sid, "resamples": RESAMPLES, "unit": "connected group_id (T2 train metadata)", "runs": report, "verdict": verdict});
    s.write_new("t3/report/t3_report.json", serde_json::to_string_pretty(&rep)?.as_bytes())?;
    println!("{}", serde_json::to_string_pretty(&rep["verdict"])?);
    Ok(())
}

fn real_main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("run") => cmd_run(&args[2..]),
        Some("aggregate") => cmd_aggregate(&args[2..]),
        _ => bail!("usage: v69-t3 <run|aggregate> --artifacts DIR [--fold F --recipe S|D]"),
    }
}

fn main() -> Result<()> {
    let _ = Path::new("");
    std::thread::Builder::new().stack_size(512 * 1024 * 1024).spawn(real_main)?.join().map_err(|_| anyhow::anyhow!("worker thread panicked"))?
}
