//! v69-d3: data-side tooling for the D3 additional-update diagnostic (host only).
//!   v69-d3 order | preserve --label start|end | freeze | aggregate
//! Every open goes through the role-restricted Access; receipts are append-only.

use anyhow::{Context, Result, bail, ensure};
use recur64_v69::access::{Access, Role};
use recur64_v69::custody::Custody;
use recur64_v69::d1::{FrozenD1, verify_group};
use recur64_v69::d1_metrics::{D1Pred, Summary, summarize};
use recur64_v69::d2::*;
use recur64_v69::dataset::Example;
use recur64_v69::provenance::{sha256_hex, source_id};
use recur64_v69::streams::MasterSeed;
use serde_json::json;
use std::collections::{BTreeMap, HashMap, HashSet};
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

fn order_of(bytes: &[u8]) -> Result<Vec<usize>> {
    let v: serde_json::Value = serde_json::from_slice(bytes)?;
    Ok(v["order"].as_array().context("order")?.iter().map(|x| x.as_u64().unwrap() as usize).collect())
}

/// Independent re-statement of the D3 learning-rate rule (contract section 5).
fn lr_formula(u: usize) -> f64 {
    let (peak, fin) = (5e-4f64, 5e-5f64);
    if u < 20 {
        peak * (u + 1) as f64 / 20.0
    } else if u <= 1999 {
        fin + 0.5 * (peak - fin) * (1.0 + (std::f64::consts::PI * (u - 20) as f64 / 1979.0).cos())
    } else {
        fin
    }
}

// ------------------------------------------------------------------ order

fn cmd_order(args: &[String]) -> Result<()> {
    let env = Env::from(args)?;
    let a = env.access(Role::D1Panel)?;
    let seed = env.seed(&a)?;
    ensure!(!a.custody().resolve(Path::new("d3/order_s768_12000.json"))?.exists(), "D3 order already frozen");
    let d2 = order_of(&a.read(Path::new("d2/subsets/s768_order.json"))?)?;
    let o = d3_order(&seed, 768, 768, D3_UPDATES);
    ensure!(o.len() == D3_UPDATES * D2_BATCH, "length");
    ensure!(o[..D2_UPDATES * D2_BATCH] == d2[..], "D3 prefix does not match the frozen D2 stream");
    let mut expo = vec![0u32; 768];
    for i in &o {
        expo[*i] += 1;
    }
    ensure!(expo.iter().all(|e| *e == 250), "every example must receive exactly 250 exposures");
    let bytes = serde_json::to_vec(&json!({"size": 768, "updates": D3_UPDATES, "batch": D2_BATCH, "stream": "d2_train_order/768/<epoch> (extended, no new stream)", "index_space": "subset rows sorted by id (d2/subsets/s768_rows.jsonl order)", "order": o}))?;
    write_new(&a, "d3/order_s768_12000.json", &bytes)?;
    println!("D3 order: {} samples, prefix == D2 ({} samples), exposures all 250, sha256 {}", o.len(), d2.len(), sha256_hex(&bytes));
    Ok(())
}

// ------------------------------------------------------------------ preserve

fn cmd_preserve(args: &[String]) -> Result<()> {
    let env = Env::from(args)?;
    let a = env.access(Role::DataAudit)?;
    let label = arg(args, "--label").context("--label start|end")?;
    let mut mismatches: Vec<String> = Vec::new();
    let mut n = 0;
    for (mf, what) in [("d1/e1_supplementary_manifest.json", "E1"), ("d2/d1_supplementary_manifest.json", "D1"), ("d3/d2_supplementary_manifest.json", "D2")] {
        let man: serde_json::Value = serde_json::from_slice(&a.read(Path::new(mf))?)?;
        for (rel, v) in man["files"].as_object().context("files")? {
            n += 1;
            if Some(sha256_hex(&a.read(Path::new(rel))?).as_str()) != v["sha256"].as_str() {
                mismatches.push(format!("{what}: {rel}"));
            }
        }
    }
    let gm: BTreeMap<String, String> = serde_json::from_slice(&a.read(&env.run.join("MANIFEST.sha256.json"))?)?;
    for (rel, want) in &gm {
        n += 1;
        if &sha256_hex(&a.read(&env.run.join(rel))?) != want {
            mismatches.push(format!("gen-001: {rel}"));
        }
    }
    let ok = mismatches.is_empty();
    let rep = json!({"label": label, "ok": ok, "files_checked": n, "mismatches": mismatches,
        "documented_e1_deviation": "E1 audit receipt audit/gen-001_audit_receipt_v2.json was overwritten in place by the E1 post-campaign audit; pre-run copy preserved (d1/e1_preserved, committed evidence). D3 never writes to audit/; receipts are append-only.",
        "source": source_id()});
    write_new(&a, &format!("d3/receipts/preservation_{label}.json"), serde_json::to_string_pretty(&rep)?.as_bytes())?;
    println!("{}", serde_json::to_string_pretty(&rep)?);
    if !ok {
        std::process::exit(4);
    }
    Ok(())
}

// ------------------------------------------------------------------ freeze

fn cmd_freeze(args: &[String]) -> Result<()> {
    let env = Env::from(args)?;
    let a = env.access(Role::DataAudit)?;
    let seed = env.seed(&a)?;
    ensure!(!a.custody().resolve(Path::new("d3/frozen_d3.json"))?.exists(), "D3 already frozen");
    let h = |rel: &str| -> Result<String> {
        Ok(match rel.strip_prefix("dataset:") {
            Some(d) => sha256_hex(&a.read(&env.run.join(d))?),
            None => sha256_hex(&a.read(Path::new(rel))?),
        })
    };
    let ck = |root: &str, items: &[&str]| -> Vec<String> { items.iter().flat_map(|x| ["model.mpk", "opt_decay.mpk", "opt_nodecay.mpk", "meta.json"].map(|f| format!("{root}/{x}/final/{f}"))).collect() };
    let mut prior: Vec<String> = ck("fits", &["A", "B", "C"]);
    prior.extend(ck("d1/fits", &["A", "M"]));
    for m in ["A", "M"] {
        for n in D2_SIZES {
            for f in ["model.mpk", "opt_decay.mpk", "opt_nodecay.mpk", "meta.json"] {
                prior.push(format!("d2/fits/{m}/s{n}/final/{f}"));
            }
        }
    }
    let base: Vec<String> = ["d3/config.json", "d3/D3_CONTRACT.md", "d3/order_s768_12000.json", "d3/receipts/preservation_start.json", "d3/d2_supplementary_manifest.json", "d2/d1_supplementary_manifest.json", "d1/e1_supplementary_manifest.json", "spec/frozen.json", "d1/frozen_d1.json", "d2/frozen_d2.json", "d2/subsets/s768_rows.jsonl", "d2/subsets/s768_order.json", "dataset:MANIFEST.sha256.json"].iter().map(|s| s.to_string()).collect();
    let learner: Vec<String> = base.iter().cloned().chain(["init/canonical_init.bin".to_string(), "d1/mlp_init.bin".to_string(), "dataset:data/fit.jsonl".to_string()]).chain(prior).collect();
    let mut agg = base.clone();
    for f in ["d2/subsets/s768_meta.jsonl", "d2/subsets/s256_meta.jsonl", "d1/baseline/predictions_fit.jsonl", "dataset:meta/fit.meta.jsonl", "d2/fits/A/s768/predictions.jsonl", "d2/fits/M/s768/predictions.jsonl", "d2/fits/A/s768/provenance.json", "d2/fits/M/s768/provenance.json", "d2/fits/A/s768/trace.json", "d2/fits/M/s768/trace.json"] {
        agg.push(f.to_string());
    }
    let mut groups: BTreeMap<String, BTreeMap<String, String>> = BTreeMap::new();
    for (name, list) in [("learner", learner), ("aggregator", agg)] {
        let mut m = BTreeMap::new();
        for rel in list {
            m.insert(rel.clone(), h(&rel).with_context(|| format!("hash {rel}"))?);
        }
        groups.insert(name.to_string(), m);
    }
    let fz = FrozenD1 { created_utc: format!("unix:{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs()), run: env.run.to_string_lossy().into_owned(), seed_fingerprint: seed.fingerprint(), groups };
    let text = serde_json::to_string_pretty(&fz)?;
    a.write(Path::new("d3/frozen_d3.json"), text.as_bytes())?;
    println!("frozen_d3.json sha256 {}", sha256_hex(text.as_bytes()));
    Ok(())
}

// ------------------------------------------------------------------ aggregate

fn meets(s: &Summary) -> bool {
    s.bal_acc >= 0.60 && s.bce <= 0.65
}

fn cmd_aggregate(args: &[String]) -> Result<()> {
    let env = Env::from(args)?;
    let a = env.access(Role::MetricAggregator)?;
    let frozen: FrozenD1 = serde_json::from_slice(&a.read(Path::new("d3/frozen_d3.json"))?)?;
    let nv = verify_group(&a, &env.run, &frozen, "aggregator")?;
    eprintln!("[d3 aggregate] {nv} frozen hashes verified");
    let meta: Vec<Example> = jsonl(&a.read_to_string(Path::new("d2/subsets/s768_meta.jsonl"))?)?;
    let meta256: Vec<Example> = jsonl(&a.read_to_string(Path::new("d2/subsets/s256_meta.jsonl"))?)?;
    ensure!(meta.len() == 768, "s768 meta");
    let mm: HashMap<&str, &Example> = meta.iter().map(|e| (e.id.as_str(), e)).collect();
    let order = order_of(&a.read(Path::new("d3/order_s768_12000.json"))?)?;
    let d2order = order_of(&a.read(Path::new("d2/subsets/s768_order.json"))?)?;
    ensure!(order[..d2order.len()] == d2order[..], "D3 order prefix differs from D2");
    let mut expect = vec![0u32; 768];
    for i in &order {
        expect[*i] += 1;
    }
    ensure!(expect.iter().all(|e| *e == 250), "expected exposure is 250 for every example");
    // group structure: 256 -> 768 expansion
    let g256: HashSet<&str> = meta256.iter().map(|e| e.group_id.as_str()).collect();
    let ids256: HashSet<&str> = meta256.iter().map(|e| e.id.as_str()).collect();
    let new_ex: Vec<&Example> = meta.iter().filter(|e| !ids256.contains(e.id.as_str())).collect();
    let in_existing = new_ex.iter().filter(|e| g256.contains(e.group_id.as_str())).count();
    let g768: HashSet<&str> = meta.iter().map(|e| e.group_id.as_str()).collect();
    let expansion = json!({"groups_at_256": g256.len(), "groups_at_768": g768.len(), "examples_added": new_ex.len(), "added_within_existing_256_groups": in_existing, "added_in_new_groups": new_ex.len() - in_existing});
    // baseline fitting predictions
    let bp: Vec<D1Pred> = jsonl(&a.read_to_string(Path::new("d1/baseline/predictions_fit.jsonl"))?)?;
    let base: HashMap<&str, f64> = bp.iter().map(|p| (p.id.as_str(), p.logit)).collect();
    ensure!(base.len() == 768 && meta.iter().all(|e| base.contains_key(e.id.as_str())), "baseline fitting predictions must cover the 768 fitting ids");
    let d2lr_a: serde_json::Value = serde_json::from_slice(&a.read(Path::new("d2/fits/A/s768/trace.json"))?)?;
    let d2_lrs: Vec<f64> = d2lr_a["trace"].as_array().unwrap().iter().map(|x| x["lr"].as_f64().unwrap()).collect();
    let mut report = serde_json::Map::new();
    let mut finals: BTreeMap<String, (bool, bool)> = BTreeMap::new();
    for model in ["A", "M"] {
        let dir = format!("d3/fits/{model}");
        let prov: serde_json::Value = serde_json::from_slice(&a.read(Path::new(&format!("{dir}/provenance.json")))?).with_context(|| format!("D3-{model} provenance missing (INCOMPLETE)"))?;
        ensure!(prov["completed_updates"].as_u64() == Some(D3_UPDATES as u64), "D3-{model} incomplete");
        let pb = a.read(Path::new(&format!("{dir}/predictions.jsonl")))?;
        ensure!(prov["predictions_sha256"].as_str() == Some(sha256_hex(&pb).as_str()), "predictions hash differs from provenance");
        let preds: Vec<D1Pred> = jsonl(std::str::from_utf8(&pb)?)?;
        let mut by_u: BTreeMap<u32, Vec<(String, f64)>> = BTreeMap::new();
        for p in preds {
            by_u.entry(p.update).or_default().push((p.id, p.logit));
        }
        ensure!(by_u.keys().map(|u| *u as usize).collect::<Vec<_>>() == D3_SNAPSHOTS.to_vec(), "unexpected snapshot set");
        let trace: serde_json::Value = serde_json::from_slice(&a.read(Path::new(&format!("{dir}/trace.json")))?)?;
        let expo: Vec<u32> = trace["exposure_by_sorted_id"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        ensure!(expo == expect, "recorded exposures differ from the frozen stream (250 each)");
        let tr = trace["trace"].as_array().unwrap();
        ensure!(tr.len() == D3_UPDATES, "trace length");
        // learning-rate independent check
        let mut lr_ok = true;
        for (u, st) in tr.iter().enumerate() {
            lr_ok &= (st["lr"].as_f64().unwrap() - lr_formula(u)).abs() <= 1e-15;
            if u < D2_UPDATES {
                lr_ok &= (st["lr"].as_f64().unwrap() - d2_lrs[u]).abs() <= 1e-15;
            }
        }
        ensure!(lr_ok, "learning-rate sequence differs from the contract/D2");
        // D2 prefix comparison
        let d2pb = a.read(Path::new(&format!("d2/fits/{model}/s768/predictions.jsonl")))?;
        let d2p: Vec<D1Pred> = jsonl(std::str::from_utf8(&d2pb)?)?;
        let mut d2by: BTreeMap<u32, HashMap<String, f64>> = BTreeMap::new();
        for p in d2p {
            d2by.entry(p.update).or_default().insert(p.id, p.logit);
        }
        let mut snaps = serde_json::Map::new();
        let mut summaries: BTreeMap<u32, Summary> = BTreeMap::new();
        let mut prefix = Vec::new();
        for (u, rows) in &by_u {
            let s = summarize(rows, &mm, None)?;
            let ip = &trace["in_process_summary"][u.to_string()];
            let dif = (ip["bce"].as_f64().unwrap() - s.bce).abs().max((ip["correct"].as_f64().unwrap() - s.n_correct as f64).abs());
            ensure!(dif < 1e-4, "independent recomputation differs from in-process summary by {dif}");
            if let Some(d2m) = d2by.get(u) {
                let d2rows: Vec<(String, f64)> = d2m.iter().map(|(k, v)| (k.clone(), *v)).collect();
                let s2 = summarize(&d2rows, &mm, None)?;
                let diffs: Vec<f64> = rows.iter().map(|(id, z)| (z - d2m[id]).abs()).collect();
                prefix.push(json!({"update": u, "d3": {"bal_acc": s.bal_acc, "bce": s.bce, "auroc": s.auroc}, "d2": {"bal_acc": s2.bal_acc, "bce": s2.bce, "auroc": s2.auroc}, "logit_abs_diff_mean": diffs.iter().sum::<f64>() / diffs.len() as f64, "logit_abs_diff_max": diffs.iter().cloned().fold(0.0, f64::max)}));
            }
            let mut v = serde_json::to_value(&s)?;
            v["strong_fit"] = json!(s.bal_acc >= 0.95 && s.bce <= 0.05);
            v["exact_memorization"] = json!(s.acc == 1.0 && s.bce <= 0.05);
            v["meets_partial_criterion"] = json!(meets(&s));
            snaps.insert(u.to_string(), v);
            summaries.insert(*u, s);
        }
        // plateau-escape analysis: first snapshot meeting BA>=0.60 & BCE<=0.65, confirmed at the next scheduled snapshot
        let us: Vec<u32> = summaries.keys().copied().collect();
        let mut confirmed: Option<u32> = None;
        let mut temporary: Vec<u32> = Vec::new();
        for (i, u) in us.iter().enumerate() {
            if meets(&summaries[u]) {
                let nxt = us.get(i + 1).map(|n| meets(&summaries[n]));
                if nxt == Some(true) && confirmed.is_none() {
                    confirmed = Some(*u);
                } else if nxt != Some(true) && confirmed.is_none() && i + 1 < us.len() {
                    temporary.push(*u);
                }
            }
        }
        let fin = &summaries[&12000];
        finals.insert(model.to_string(), (fin.bal_acc >= 0.95 && fin.bce <= 0.05, fin.acc == 1.0 && fin.bce <= 0.05));
        // group / error analysis at the final endpoint
        let frows = &by_u[&12000];
        let mut per_group: BTreeMap<&str, (usize, usize, f64)> = BTreeMap::new();
        for (id, z) in frows {
            let e = mm[id.as_str()];
            let g = per_group.entry(e.group_id.as_str()).or_default();
            g.0 += 1;
            g.1 += ((*z > 0.0) != e.label) as usize;
            g.2 += if e.label { *z } else { -*z };
        }
        let mut gl: Vec<(&str, (usize, usize, f64))> = per_group.into_iter().collect();
        gl.sort_by(|x, y| y.1.1.cmp(&x.1.1).then(x.0.cmp(y.0)));
        let groups_with_errors = gl.iter().filter(|g| g.1.1 > 0).count();
        let multi: Vec<&(&str, (usize, usize, f64))> = gl.iter().filter(|g| g.1.0 > 1).collect();
        let group_json = json!({"groups": gl.len(), "groups_with_any_error": groups_with_errors, "groups_with_multiple_examples": multi.len(), "multi_example_groups_with_error": multi.iter().filter(|g| g.1.1 > 0).count(),
            "top_error_groups": gl.iter().take(10).map(|g| json!({"group": g.0, "n": g.1.0, "errors": g.1.1, "mean_signed_margin": g.1.2 / g.1.0 as f64})).collect::<Vec<_>>()});
        // complementarity with the D1 shallow baseline's FITTING predictions (threshold logit 0; no ensemble)
        let (mut both_ok, mut both_bad, mut n_only, mut b_only, mut agree) = (0, 0, 0, 0, 0);
        for (id, z) in frows {
            let e = mm[id.as_str()];
            let (nc, bc) = ((*z > 0.0) == e.label, (base[id.as_str()] > 0.0) == e.label);
            both_ok += (nc && bc) as usize;
            both_bad += (!nc && !bc) as usize;
            n_only += (nc && !bc) as usize;
            b_only += (!nc && bc) as usize;
            agree += ((*z > 0.0) == (base[id.as_str()] > 0.0)) as usize;
        }
        let bal_base = {
            let mut c = [0usize; 4];
            for e in &meta {
                let pos = base[e.id.as_str()] > 0.0;
                c[(e.label as usize) * 2 + pos as usize] += 1;
            }
            0.5 * (c[3] as f64 / (c[2] + c[3]) as f64 + c[0] as f64 / (c[0] + c[1]) as f64)
        };
        let compl = json!({"agreement": agree, "both_correct": both_ok, "both_wrong": both_bad, "neural_only_correct": n_only, "baseline_only_correct": b_only, "total": frows.len(), "baseline_fit_balanced_accuracy": bal_base, "note": "descriptive only: baseline not retrained, no ensemble, no thresholds chosen, no validation"});
        // dynamics windows
        let mut dyn_ = serde_json::Map::new();
        for (name, a0, b0) in [("0_100", 0usize, 100usize), ("100_600", 100, 600), ("600_1200", 600, 1200), ("1200_2000", 1200, 2000), ("2000_2400", 2000, 2400), ("2400_3000", 2400, 3000), ("3000_4000", 3000, 4000), ("4000_6000", 4000, 6000), ("6000_8000", 6000, 8000), ("8000_10000", 8000, 10000), ("10000_12000", 10000, 12000)] {
            let s = &tr[a0..b0];
            dyn_.insert(name.into(), json!({"mean_loss": s.iter().map(|x| x["loss"].as_f64().unwrap()).sum::<f64>() / s.len() as f64, "mean_grad_norm": s.iter().map(|x| x["grad_norm_pre_clip"].as_f64().unwrap()).sum::<f64>() / s.len() as f64, "clip_fraction": s.iter().filter(|x| x["clipped"].as_bool().unwrap()).count() as f64 / s.len() as f64}));
        }
        report.insert(format!("D3-{model}"), json!({
            "provenance_id": prov["provenance_id"], "predictions_sha256": prov["predictions_sha256"],
            "snapshots": snaps, "d2_prefix_comparison": prefix,
            "plateau_escape": {"criterion": "first snapshot with BA >= 0.60 and BCE <= 0.65, confirmed (also met) at the next scheduled snapshot", "confirmed_first_snapshot": confirmed, "temporary_improvements_before_confirmation": temporary},
            "group_error_analysis_at_12000": group_json, "baseline_complementarity_at_12000": compl, "dynamics_windows": dyn_,
            "wall_secs": trace["wall_secs"], "mean_update_ms": trace["mean_update_ms"], "clip_frequency": trace["clip_frequency"],
            "learning_rate_sequence_checked": lr_ok, "exposure_checked_all_250": true,
        }));
    }
    report.insert("expansion_256_to_768_group_structure".into(), expansion);
    let fa = finals["A"];
    let fm = finals["M"];
    report.insert("final_endpoints".into(), json!({"A": {"strong_fit": fa.0, "exact": fa.1}, "M": {"strong_fit": fm.0, "exact": fm.1}}));
    report.insert("label".into(), json!("D3 is fitting-only: no validation/sealed use; descriptive criteria, not generalization claims."));
    report.insert("source".into(), serde_json::to_value(source_id())?);
    let text = serde_json::to_string_pretty(&serde_json::Value::Object(report))?;
    a.write(Path::new("d3/report/d3_report.json"), text.as_bytes())?;
    println!("A strong_fit={} exact={} | M strong_fit={} exact={}", fa.0, fa.1, fm.0, fm.1);
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("order") => cmd_order(&args[2..]),
        Some("preserve") => cmd_preserve(&args[2..]),
        Some("freeze") => cmd_freeze(&args[2..]),
        Some("aggregate") => cmd_aggregate(&args[2..]),
        _ => bail!("usage: v69-d3 <order|preserve|freeze|aggregate> --artifacts DIR"),
    }
}
