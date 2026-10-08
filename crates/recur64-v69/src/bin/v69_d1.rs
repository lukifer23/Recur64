//! v69-d1: data-side tooling for the D1 diagnostic (host only, no GPU).
//!   v69-d1 verify-e1 --artifacts DIR        verify the supplementary E1 manifest + E1 frozen deps
//!   v69-d1 panel     --artifacts DIR        select the 32-example panel + frozen example stream
//!   v69-d1 freeze    --artifacts DIR        write d1/frozen_d1.json (expected hashes)
//!   v69-d1 baseline  --artifacts DIR        shallow feature baseline (fit once, evaluate once)
//!   v69-d1 aggregate --artifacts DIR        independent metrics + decision
//! Every open goes through the role-restricted Access.

use anyhow::{Context, Result, bail, ensure};
use recur64_v69::access::{Access, Role};
use recur64_v69::custody::Custody;
use recur64_v69::d1::*;
use recur64_v69::d1_metrics::{D1Pred, summarize};
use recur64_v69::dataset::Example;
use recur64_v69::provenance::{sha256_hex, source_id};
use recur64_v69::streams::MasterSeed;
use serde_json::json;
use std::collections::{BTreeMap, HashMap};
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
    fn access(&self, role: Role) -> Result<Access> {
        Access::new(&self.custody, role, &self.run, &self.seed_rel)
    }
    fn seed(&self, a: &Access) -> Result<MasterSeed> {
        MasterSeed::from_hex(String::from_utf8(a.read(&self.seed_rel)?)?.trim())
    }
}

fn jsonl<T: serde::de::DeserializeOwned>(text: &str) -> Result<Vec<T>> {
    text.lines().map(|l| Ok(serde_json::from_str(l)?)).collect()
}

fn write_jsonl<T: serde::Serialize>(a: &Access, rel: &str, rows: &[T]) -> Result<String> {
    let text: String = rows.iter().map(|r| serde_json::to_string(r).unwrap() + "\n").collect();
    a.write(Path::new(rel), text.as_bytes())?;
    Ok(sha256_hex(text.as_bytes()))
}

// ------------------------------------------------------------------ verify-e1

fn cmd_verify_e1(args: &[String]) -> Result<()> {
    let env = Env::from(args)?;
    let a = env.access(Role::DataAudit)?;
    let man: serde_json::Value = serde_json::from_slice(&a.read(Path::new("d1/e1_supplementary_manifest.json"))?)?;
    let mut bad = Vec::new();
    let mut n = 0;
    for (rel, v) in man["files"].as_object().context("files")? {
        let got = sha256_hex(&a.read(Path::new(rel))?);
        n += 1;
        if Some(got.as_str()) != v["sha256"].as_str() {
            bad.push(format!("{rel}: {got}"));
        }
    }
    // E1 frozen dependency hashes (spec/frozen.json) against the actual files
    let fz: serde_json::Value = serde_json::from_slice(&a.read(Path::new("spec/frozen.json"))?)?;
    let checks = [
        ("model_spec_sha256", "spec/MODEL_SPEC.md"),
        ("contract_sha256", "spec/CONTRACT.md"),
        ("canonical_init_file_sha256", "init/canonical_init.bin"),
        ("parameter_inventory_sha256", "spec/param_inventory.json"),
        ("intervention_map_sha256", "intervention/map.json"),
        ("qual_summary_sha256", "qual/qual_summary.json"),
    ];
    for (k, rel) in checks {
        let got = sha256_hex(&a.read(Path::new(rel))?);
        if fz[k].as_str() != Some(got.as_str()) {
            bad.push(format!("frozen {k} -> {rel}: {got}"));
        }
        n += 1;
    }
    // KNOWN E1 DEVIATION (found by this verification): the pre-fit audit receipt that frozen.json hashed was
    // later overwritten in place by the post-campaign audit re-run. The pre-fit copy is preserved (committed
    // evidence, copied to d1/e1_preserved/). Enforce the frozen hash against the preserved copy, and require
    // the live file to equal the post-run copy that replaced it.
    let pre = sha256_hex(&a.read(Path::new("d1/e1_preserved/audit_receipt_v2_prerun.json"))?);
    if fz["audit_v2_receipt_sha256"].as_str() != Some(pre.as_str()) {
        bad.push(format!("frozen audit_v2_receipt_sha256 vs preserved pre-run copy: {pre}"));
    }
    let live = sha256_hex(&a.read(Path::new("audit/gen-001_audit_receipt_v2.json"))?);
    let post = sha256_hex(&a.read(Path::new("audit/gen-001_audit_receipt_v2_postrun.json"))?);
    if live != post {
        bad.push("live audit receipt differs from its preserved post-run copy".into());
    }
    let dm = sha256_hex(&a.read(&env.run.join("MANIFEST.sha256.json"))?);
    if fz["dataset_manifest_sha256"].as_str() != Some(dm.as_str()) {
        bad.push("dataset manifest".into());
    }
    let ok = bad.is_empty();
    let rep = json!({"ok": ok, "files_checked": n, "mismatches": bad, "known_deviation": "E1 audit receipt audit/gen-001_audit_receipt_v2.json was overwritten in place by the post-campaign audit re-run; frozen hash verified against preserved pre-run copy (d1/e1_preserved), live file verified equal to the post-run copy. All other E1 frozen dependencies matched exactly.", "source": source_id()});
    a.write(Path::new("d1/e1_verification.json"), serde_json::to_string_pretty(&rep)?.as_bytes())?;
    println!("{}", serde_json::to_string_pretty(&rep)?);
    if !ok {
        std::process::exit(4);
    }
    Ok(())
}

// ------------------------------------------------------------------ panel

fn cmd_panel(args: &[String]) -> Result<()> {
    let env = Env::from(args)?;
    let a = env.access(Role::D1Panel)?;
    let seed = env.seed(&a)?;
    ensure!(!a.custody().resolve(Path::new("d1/panel/panel_rows.jsonl"))?.exists(), "panel already frozen");
    let manifest: BTreeMap<String, String> = serde_json::from_slice(&a.read(&env.run.join("MANIFEST.sha256.json"))?)?;
    let meta_bytes = a.read(&env.run.join("meta/fit.meta.jsonl"))?;
    ensure!(manifest.get("meta/fit.meta.jsonl") == Some(&sha256_hex(&meta_bytes)), "fit metadata hash mismatch");
    let fit_meta: Vec<Example> = jsonl(std::str::from_utf8(&meta_bytes)?)?;
    ensure!(fit_meta.iter().all(|e| e.partition == recur64_v69::dataset::Partition::Fit), "non-fit rows in fit metadata");
    let (panel, sel) = select_panel(&seed, &fit_meta)?;
    #[derive(serde::Serialize)]
    struct Row<'a> {
        id: &'a str,
        fen: &'a str,
        budget: u8,
        label: bool,
    }
    let rows: Vec<Row> = panel.iter().map(|e| Row { id: &e.id, fen: &e.fen, budget: e.budget, label: e.label }).collect();
    let h_rows = write_jsonl(&a, "d1/panel/panel_rows.jsonl", &rows)?;
    let h_meta = write_jsonl(&a, "d1/panel/panel_meta.jsonl", &panel)?;
    let order = d1_order(&seed, panel.len());
    let order_json = json!({"updates": D1_UPDATES, "batch": D1_BATCH, "panel_size": panel.len(), "stream": format!("{STREAM_D1_ORDER}/<epoch>"), "index_space": "panel rows sorted by id (panel_rows.jsonl order)", "order": order});
    let ob = serde_json::to_vec(&order_json)?;
    a.write(Path::new("d1/panel/train_order.json"), &ob)?;
    let mut exposure = vec![0usize; panel.len()];
    for i in &order {
        exposure[*i] += 1;
    }
    a.write(Path::new("d1/panel/panel_selection.json"), serde_json::to_string_pretty(&json!({"selection": sel, "panel_rows_sha256": h_rows, "panel_meta_sha256": h_meta, "train_order_sha256": sha256_hex(&ob), "exposure_min_max": [exposure.iter().min(), exposure.iter().max()]}))?.as_bytes())?;
    println!("panel 32: {} pos / {} neg, distinct groups {}, rows sha {}, order sha {}", sel.positives, sel.negatives, sel.groups.iter().collect::<std::collections::HashSet<_>>().len(), &h_rows[..12], &sha256_hex(&ob)[..12]);
    println!("per-cell {:?}; extra strata {:?}", sel.per_cell, sel.extra_strata);
    Ok(())
}

// ------------------------------------------------------------------ freeze

fn cmd_freeze(args: &[String]) -> Result<()> {
    let env = Env::from(args)?;
    let a = env.access(Role::DataAudit)?;
    let seed = env.seed(&a)?;
    let target = Path::new("d1/frozen_d1.json");
    ensure!(!a.custody().resolve(target)?.exists(), "D1 already frozen");
    let h = |rel: &str| -> Result<String> {
        Ok(match rel.strip_prefix("dataset:") {
            Some(d) => sha256_hex(&a.read(&env.run.join(d))?),
            None => sha256_hex(&a.read(Path::new(rel))?),
        })
    };
    let mut groups: BTreeMap<String, BTreeMap<String, String>> = BTreeMap::new();
    let e1: Vec<String> = ["A", "B", "C"].iter().flat_map(|x| ["model.mpk", "opt_decay.mpk", "opt_nodecay.mpk", "meta.json"].map(|f| format!("fits/{x}/final/{f}"))).collect();
    let learner: Vec<String> = ["d1/config.json", "d1/D1_CONTRACT.md", "d1/panel/panel_rows.jsonl", "d1/panel/train_order.json", "d1/mlp_init.bin", "init/canonical_init.bin", "spec/frozen.json", "d1/e1_supplementary_manifest.json", "d1/e1_verification.json", "dataset:MANIFEST.sha256.json", "dataset:data/fit.jsonl"].iter().map(|s| s.to_string()).chain(e1).collect();
    let baseline: Vec<String> = ["d1/config.json", "d1/D1_CONTRACT.md", "spec/frozen.json", "dataset:MANIFEST.sha256.json", "dataset:data/fit.jsonl", "dataset:data/val.jsonl"].iter().map(|s| s.to_string()).collect();
    let aggregator: Vec<String> = ["d1/config.json", "d1/D1_CONTRACT.md", "d1/panel/panel_rows.jsonl", "d1/panel/panel_meta.jsonl", "d1/panel/train_order.json", "dataset:MANIFEST.sha256.json", "dataset:meta/fit.meta.jsonl", "dataset:meta/val.meta.jsonl"].iter().map(|s| s.to_string()).collect();
    for (name, list) in [("learner", learner), ("baseline", baseline), ("aggregator", aggregator)] {
        let mut m = BTreeMap::new();
        for rel in list {
            m.insert(rel.clone(), h(&rel).with_context(|| format!("hash {rel}"))?);
        }
        groups.insert(name.to_string(), m);
    }
    let fz = FrozenD1 { created_utc: chrono_like_now(), run: env.run.to_string_lossy().into_owned(), seed_fingerprint: seed.fingerprint(), groups };
    let text = serde_json::to_string_pretty(&fz)?;
    a.write(target, text.as_bytes())?;
    println!("frozen_d1.json sha256 {}", sha256_hex(text.as_bytes()));
    Ok(())
}

fn chrono_like_now() -> String {
    let d = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap();
    format!("unix:{}", d.as_secs())
}

// ------------------------------------------------------------------ baseline

fn cmd_baseline(args: &[String]) -> Result<()> {
    let env = Env::from(args)?;
    let a = env.access(Role::Evaluator)?;
    let frozen = read_frozen(&a)?;
    let n = verify_group(&a, &env.run, &frozen, "baseline")?;
    eprintln!("[d1 baseline] {n} frozen hashes verified");
    ensure!(!a.custody().resolve(Path::new("d1/baseline/model.json"))?.exists(), "baseline already fitted (single evaluation)");
    let fit = load_rows_verified(&a, &env.run, "fit")?;
    let val = load_rows_verified(&a, &env.run, "val")?;
    let xf: Vec<[f64; N_BASELINE]> = fit.iter().map(|r| baseline_features(&r.fen, r.budget)).collect::<Result<_>>()?;
    let yf: Vec<bool> = fit.iter().map(|r| r.label).collect();
    let xv: Vec<[f64; N_BASELINE]> = val.iter().map(|r| baseline_features(&r.fen, r.budget)).collect::<Result<_>>()?;
    let model = fit_baseline(&xf, &yf);
    let mk = |rows: &Vec<recur64_v69::features::ModelRow>, x: &Vec<[f64; N_BASELINE]>| -> Vec<D1Pred> { rows.iter().zip(x).map(|(r, f)| D1Pred { id: r.id.clone(), update: 0, logit: model.logit(f) }).collect() };
    let hf = write_jsonl(&a, "d1/baseline/predictions_fit.jsonl", &mk(&fit, &xf))?;
    let hv = write_jsonl(&a, "d1/baseline/predictions_val.jsonl", &mk(&val, &xv))?;
    let names: Vec<String> = BASE_FEATURES.iter().map(|s| s.to_string()).chain(std::iter::once("budget_is_2".to_string())).chain(BASE_FEATURES.iter().map(|s| format!("{s}*budget_is_2"))).collect();
    let mj = json!({"features": names, "model": model, "source": source_id(), "fit_rows": fit.len(), "val_rows": val.len(), "predictions_fit_sha256": hf, "predictions_val_sha256": hv, "convention": "objective = mean BCE + (l2/2)*sum(w^2), intercept excluded; full-batch GD from zero, 1000 steps, lr 0.05, l2 0.01; standardized with fit statistics; threshold logit 0"});
    a.write(Path::new("d1/baseline/model.json"), serde_json::to_string_pretty(&mj)?.as_bytes())?;
    println!("baseline fitted: objective {:.4} -> {:.4}; predictions fit {} val {}", model.objective_first_step, model.final_train_objective, &hf[..12], &hv[..12]);
    Ok(())
}

// ------------------------------------------------------------------ aggregate

fn read_preds(a: &Access, rel: &str) -> Result<(Vec<D1Pred>, String)> {
    let b = a.read(Path::new(rel))?;
    Ok((jsonl(std::str::from_utf8(&b)?)?, sha256_hex(&b)))
}

fn cmd_aggregate(args: &[String]) -> Result<()> {
    let env = Env::from(args)?;
    let a = env.access(Role::MetricAggregator)?;
    let seed = env.seed(&a)?;
    let frozen = read_frozen(&a)?;
    verify_group(&a, &env.run, &frozen, "aggregator")?;
    let panel_meta: Vec<Example> = jsonl(&a.read_to_string(Path::new("d1/panel/panel_meta.jsonl"))?)?;
    let pm: HashMap<&str, &Example> = panel_meta.iter().map(|e| (e.id.as_str(), e)).collect();
    ensure!(pm.len() == 32, "panel size");
    let mut report = serde_json::Map::new();
    let mut memorized: BTreeMap<String, bool> = BTreeMap::new();
    for model in ["A", "M"] {
        let prov: serde_json::Value = serde_json::from_slice(&a.read(Path::new(&format!("d1/fits/{model}/provenance.json")))?)
            .with_context(|| format!("D1-{model} provenance missing (incomplete run?)"))?;
        ensure!(prov["completed_updates"].as_u64() == Some(D1_UPDATES as u64), "D1-{model} incomplete: not a scientific result");
        let rel = format!("d1/fits/{model}/panel_predictions.jsonl");
        let (preds, sha) = read_preds(&a, &rel)?;
        ensure!(prov["panel_predictions_sha256"].as_str() == Some(sha.as_str()), "prediction file hash differs from provenance");
        let mut by_update: BTreeMap<u32, Vec<(String, f64)>> = BTreeMap::new();
        for p in preds {
            by_update.entry(p.update).or_default().push((p.id, p.logit));
        }
        ensure!(by_update.keys().copied().collect::<Vec<_>>() == vec![0, 100, 500, 1000, 2000], "unexpected measurement updates");
        let mut snaps = serde_json::Map::new();
        for (u, rows) in &by_update {
            let s = summarize(rows, &pm, None)?;
            if *u == 2000 {
                memorized.insert(model.to_string(), s.n_correct == 32 && s.bce <= 0.05);
            }
            snaps.insert(u.to_string(), serde_json::to_value(&s)?);
        }
        report.insert(format!("D1-{model}"), json!({"provenance_ref": prov["provenance_id"], "prediction_sha256": sha, "snapshots": snaps}));
    }
    // baseline
    let fm: Vec<Example> = jsonl(&a.read_to_string(&env.run.join("meta/fit.meta.jsonl"))?)?;
    let vm: Vec<Example> = jsonl(&a.read_to_string(&env.run.join("meta/val.meta.jsonl"))?)?;
    let fmm: HashMap<&str, &Example> = fm.iter().map(|e| (e.id.as_str(), e)).collect();
    let vmm: HashMap<&str, &Example> = vm.iter().map(|e| (e.id.as_str(), e)).collect();
    let bmeta: serde_json::Value = serde_json::from_slice(&a.read(Path::new("d1/baseline/model.json"))?)?;
    let mut base = serde_json::Map::new();
    for (name, rel, mm) in [("fit", "d1/baseline/predictions_fit.jsonl", &fmm), ("val", "d1/baseline/predictions_val.jsonl", &vmm)] {
        let (preds, sha) = read_preds(&a, rel)?;
        ensure!(bmeta[format!("predictions_{name}_sha256")].as_str() == Some(sha.as_str()), "baseline {name} predictions hash mismatch");
        let rows: Vec<(String, f64)> = preds.into_iter().map(|p| (p.id, p.logit)).collect();
        let s = summarize(&rows, mm, Some((&seed, name, 2000)))?;
        base.insert(name.to_string(), serde_json::to_value(&s)?);
    }
    report.insert("baseline".into(), serde_json::Value::Object(base));
    let (ma, mm_) = (memorized.get("A").copied().unwrap_or(false), memorized.get("M").copied().unwrap_or(false));
    let decision = match (ma, mm_) {
        (true, true) => "BOTH_MEMORIZE: basic tiny-panel trainability established for both; larger-set learning remains open.",
        (false, true) => "MLP_ONLY_MEMORIZES: prioritize current-architecture / optimization-path diagnostics (D1-A failed to memorize while the direct-board MLP did).",
        (false, false) => "NEITHER_MEMORIZES: prioritize shared input/target/loss/update mechanics and training dynamics; do not conclude the task is unlearnable.",
        (true, false) => "D1A_ONLY_MEMORIZES: retain the current architecture as a candidate; investigate why the MLP control was insufficient.",
    };
    report.insert("memorization_criterion".into(), json!("32/32 correct and BCE <= 0.05 at update 2000"));
    report.insert("memorized".into(), json!(memorized));
    report.insert("decision".into(), json!(decision));
    report.insert("label".into(), json!("D1 is retrospective/diagnostic. Gen-001 validation is already-exposed development evidence; baseline validation numbers are NOT fresh confirmation."));
    report.insert("source".into(), serde_json::to_value(source_id())?);
    let text = serde_json::to_string_pretty(&serde_json::Value::Object(report))?;
    a.write(Path::new("d1/report/d1_report.json"), text.as_bytes())?;
    println!("{decision}");
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("verify-e1") => cmd_verify_e1(&args[2..]),
        Some("panel") => cmd_panel(&args[2..]),
        Some("freeze") => cmd_freeze(&args[2..]),
        Some("baseline") => cmd_baseline(&args[2..]),
        Some("aggregate") => cmd_aggregate(&args[2..]),
        _ => bail!("usage: v69-d1 <verify-e1|panel|freeze|baseline|aggregate> --artifacts DIR"),
    }
}
