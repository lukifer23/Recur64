//! v69-d2: data-side tooling for the D2 fitting-set scaling diagnostic (host only).
//!   v69-d2 subsets   nested fitting subsets + example streams
//!   v69-d2 preserve  --label start|end   append-only preservation receipt (E1, D1, gen-001)
//!   v69-d2 freeze    expected hashes (frozen_d2.json)
//!   v69-d2 aggregate independent metrics, comparisons, decision
//! Every open goes through the role-restricted Access.

use anyhow::{Context, Result, bail, ensure};
use recur64_v69::access::{Access, Role};
use recur64_v69::custody::Custody;
use recur64_v69::d1::{FrozenD1, verify_group};
use recur64_v69::d1_metrics::{D1Pred, summarize};
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

fn read_frozen_d2(a: &Access) -> Result<FrozenD1> {
    Ok(serde_json::from_slice(&a.read(Path::new("d2/frozen_d2.json"))?)?)
}

// ------------------------------------------------------------------ subsets

fn cmd_subsets(args: &[String]) -> Result<()> {
    let env = Env::from(args)?;
    let a = env.access(Role::D1Panel)?;
    let seed = env.seed(&a)?;
    ensure!(!a.custody().resolve(Path::new("d2/subsets/subset_summary.json"))?.exists(), "subsets already frozen");
    let manifest: BTreeMap<String, String> = serde_json::from_slice(&a.read(&env.run.join("MANIFEST.sha256.json"))?)?;
    let mb = a.read(&env.run.join("meta/fit.meta.jsonl"))?;
    ensure!(manifest.get("meta/fit.meta.jsonl") == Some(&sha256_hex(&mb)), "fit metadata hash mismatch");
    let fit_meta: Vec<Example> = jsonl(std::str::from_utf8(&mb)?)?;
    // D1 panel ids (hash-checked against the D1 frozen file's panel rows hash is done at launch; here ids only)
    let panel_text = a.read_to_string(Path::new("d1/panel/panel_rows.jsonl"))?;
    let panel_ids: HashSet<String> = panel_text.lines().map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap()["id"].as_str().unwrap().to_string()).collect();
    let subsets = select_nested(&seed, &fit_meta, &panel_ids)?;
    #[derive(serde::Serialize)]
    struct Row<'a> {
        id: &'a str,
        fen: &'a str,
        budget: u8,
        label: bool,
    }
    let mut summaries = Vec::new();
    let mut hashes = BTreeMap::new();
    for (n, rows, sum) in &subsets {
        let mr: Vec<Row> = rows.iter().map(|e| Row { id: &e.id, fen: &e.fen, budget: e.budget, label: e.label }).collect();
        let t1: String = mr.iter().map(|r| serde_json::to_string(r).unwrap() + "\n").collect();
        let t2: String = rows.iter().map(|r| serde_json::to_string(r).unwrap() + "\n").collect();
        let order = d2_order(&seed, rows.len(), *n);
        let oj = serde_json::to_vec(&json!({"size": n, "updates": D2_UPDATES, "batch": D2_BATCH, "stream": format!("{LABEL_ORDER}/{n}/<epoch>"), "index_space": "subset rows sorted by id (s<N>_rows.jsonl order)", "order": order}))?;
        a.write(Path::new(&format!("d2/subsets/s{n}_rows.jsonl")), t1.as_bytes())?;
        a.write(Path::new(&format!("d2/subsets/s{n}_meta.jsonl")), t2.as_bytes())?;
        a.write(Path::new(&format!("d2/subsets/s{n}_order.json")), &oj)?;
        let mut exposure = vec![0usize; rows.len()];
        for i in &order {
            exposure[*i] += 1;
        }
        hashes.insert(format!("s{n}"), json!({"rows_sha256": sha256_hex(t1.as_bytes()), "meta_sha256": sha256_hex(t2.as_bytes()), "order_sha256": sha256_hex(&oj), "exposure_min_max": [exposure.iter().min(), exposure.iter().max()]}));
        summaries.push(sum.clone());
        println!("N={n}: groups {} (repeated memberships {}), roots {}, pos/neg {}/{}, extra-pair strata {:?}, contains D1 panel {}", sum.distinct_groups, sum.repeated_group_memberships, sum.distinct_roots, sum.positives, sum.negatives, sum.extra_pair_strata, sum.contains_d1_panel);
    }
    a.write(Path::new("d2/subsets/subset_summary.json"), serde_json::to_string_pretty(&json!({"summaries": summaries, "hashes": hashes, "seed_fingerprint": seed.fingerprint(), "source": source_id()}))?.as_bytes())?;
    Ok(())
}

// ------------------------------------------------------------------ preserve (append-only receipt)

fn cmd_preserve(args: &[String]) -> Result<()> {
    let env = Env::from(args)?;
    let a = env.access(Role::DataAudit)?;
    let label = arg(args, "--label").context("--label start|end")?;
    let mut mismatches: Vec<String> = Vec::new();
    let mut n = 0;
    for (mf, what) in [("d1/e1_supplementary_manifest.json", "E1"), ("d2/d1_supplementary_manifest.json", "D1")] {
        let man: serde_json::Value = serde_json::from_slice(&a.read(Path::new(mf))?)?;
        for (rel, v) in man["files"].as_object().context("files")? {
            n += 1;
            let got = sha256_hex(&a.read(Path::new(rel))?);
            if Some(got.as_str()) != v["sha256"].as_str() {
                // d1/ files that D2 is allowed to change: none. Anything differing is reported.
                mismatches.push(format!("{what}: {rel}"));
            }
        }
    }
    // gen-001 vs its own manifest
    let gm: BTreeMap<String, String> = serde_json::from_slice(&a.read(&env.run.join("MANIFEST.sha256.json"))?)?;
    for (rel, want) in &gm {
        n += 1;
        if &sha256_hex(&a.read(&env.run.join(rel))?) != want {
            mismatches.push(format!("gen-001: {rel}"));
        }
    }
    let ok = mismatches.is_empty();
    let rep = json!({"label": label, "ok": ok, "files_checked": n, "mismatches": mismatches,
        "documented_e1_deviation": "E1 audit receipt audit/gen-001_audit_receipt_v2.json was overwritten in place by the E1 post-campaign audit; the pre-run copy is preserved (d1/e1_preserved and committed evidence). D2 never writes to audit/ and receipts here are append-only.",
        "source": source_id()});
    let rel = format!("d2/receipts/preservation_{label}.json");
    write_new(&a, &rel, serde_json::to_string_pretty(&rep)?.as_bytes())?;
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
    let target = "d2/frozen_d2.json";
    ensure!(!a.custody().resolve(Path::new(target))?.exists(), "D2 already frozen");
    let h = |rel: &str| -> Result<String> {
        Ok(match rel.strip_prefix("dataset:") {
            Some(d) => sha256_hex(&a.read(&env.run.join(d))?),
            None => sha256_hex(&a.read(Path::new(rel))?),
        })
    };
    let e1: Vec<String> = ["A", "B", "C"].iter().flat_map(|x| ["model.mpk", "opt_decay.mpk", "opt_nodecay.mpk", "meta.json"].map(|f| format!("fits/{x}/final/{f}"))).collect();
    let d1f: Vec<String> = ["A", "M"].iter().flat_map(|x| ["model.mpk", "opt_decay.mpk", "opt_nodecay.mpk", "meta.json"].map(|f| format!("d1/fits/{x}/final/{f}"))).collect();
    let mut sub: Vec<String> = Vec::new();
    for n in D2_SIZES {
        sub.push(format!("d2/subsets/s{n}_rows.jsonl"));
        sub.push(format!("d2/subsets/s{n}_order.json"));
    }
    let base: Vec<String> = ["d2/config.json", "d2/D2_CONTRACT.md", "d2/receipts/preservation_start.json", "d2/d1_supplementary_manifest.json", "d1/e1_supplementary_manifest.json", "spec/frozen.json", "d1/frozen_d1.json", "dataset:MANIFEST.sha256.json"].iter().map(|s| s.to_string()).collect();
    let learner: Vec<String> = base.iter().cloned().chain(sub.clone()).chain(["init/canonical_init.bin".to_string(), "d1/mlp_init.bin".to_string(), "dataset:data/fit.jsonl".to_string()]).chain(e1).chain(d1f).collect();
    let mut agg: Vec<String> = base.clone();
    agg.extend(sub);
    for n in D2_SIZES {
        agg.push(format!("d2/subsets/s{n}_meta.jsonl"));
    }
    agg.push("dataset:meta/fit.meta.jsonl".into());
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
    a.write(Path::new(target), text.as_bytes())?;
    println!("frozen_d2.json sha256 {}", sha256_hex(text.as_bytes()));
    Ok(())
}

// ------------------------------------------------------------------ aggregate

fn strong(s: &recur64_v69::d1_metrics::Summary) -> bool {
    s.bal_acc >= 0.95 && s.bce <= 0.05
}

fn exact(s: &recur64_v69::d1_metrics::Summary) -> bool {
    s.acc == 1.0 && s.bce <= 0.05
}

fn cmd_aggregate(args: &[String]) -> Result<()> {
    let env = Env::from(args)?;
    let a = env.access(Role::MetricAggregator)?;
    let frozen = read_frozen_d2(&a)?;
    let nv = verify_group(&a, &env.run, &frozen, "aggregator")?;
    eprintln!("[d2 aggregate] {nv} frozen hashes verified");
    let mut report = serde_json::Map::new();
    let mut table: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    let mut endpoint: BTreeMap<(String, usize), (bool, bool)> = BTreeMap::new();
    let mut at: HashMap<(String, usize, usize), (f64, f64, f64)> = HashMap::new(); // (model,size,update)->(acc,bce,auroc)
    for &n in &D2_SIZES {
        let meta: Vec<Example> = jsonl(&a.read_to_string(Path::new(&format!("d2/subsets/s{n}_meta.jsonl")))?)?;
        let mm: HashMap<&str, &Example> = meta.iter().map(|e| (e.id.as_str(), e)).collect();
        ensure!(mm.len() == n, "subset size");
        let ordb = a.read(Path::new(&format!("d2/subsets/s{n}_order.json")))?;
        let oj: serde_json::Value = serde_json::from_slice(&ordb)?;
        let order: Vec<usize> = oj["order"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let mut exp_expected = vec![0u32; n];
        for i in &order {
            exp_expected[*i] += 1;
        }
        let mut ids_sorted: Vec<&str> = mm.keys().copied().collect();
        ids_sorted.sort();
        for model in ["A", "M"] {
            let dir = format!("d2/fits/{model}/s{n}");
            let prov: serde_json::Value = serde_json::from_slice(&a.read(Path::new(&format!("{dir}/provenance.json")))?).with_context(|| format!("D2-{model} N={n} provenance missing (INCOMPLETE)"))?;
            ensure!(prov["completed_updates"].as_u64() == Some(D2_UPDATES as u64), "D2-{model} N={n} incomplete");
            let pb = a.read(Path::new(&format!("{dir}/predictions.jsonl")))?;
            ensure!(prov["predictions_sha256"].as_str() == Some(sha256_hex(&pb).as_str()), "predictions hash differs from provenance");
            let preds: Vec<D1Pred> = jsonl(std::str::from_utf8(&pb)?)?;
            let mut by_u: BTreeMap<u32, Vec<(String, f64)>> = BTreeMap::new();
            for p in preds {
                by_u.entry(p.update).or_default().push((p.id, p.logit));
            }
            ensure!(by_u.keys().map(|u| *u as usize).collect::<Vec<_>>() == D2_SNAPSHOTS.to_vec(), "unexpected snapshot updates");
            let trace: serde_json::Value = serde_json::from_slice(&a.read(Path::new(&format!("{dir}/trace.json")))?)?;
            let expo: Vec<u32> = trace["exposure_by_sorted_id"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
            ensure!(expo == exp_expected, "recorded exposures differ from the frozen order");
            let mut snaps = serde_json::Map::new();
            for (u, rows) in &by_u {
                let s = summarize(rows, &mm, None)?;
                at.insert((model.to_string(), n, *u as usize), (s.acc, s.bce, s.auroc));
                if *u as usize == D2_UPDATES {
                    endpoint.insert((model.to_string(), n), (strong(&s), exact(&s)));
                }
                // cross-check vs in-process summary
                let ip = &trace["in_process_summary"][u.to_string()];
                let dif = (ip["bce"].as_f64().unwrap() - s.bce).abs().max((ip["correct"].as_f64().unwrap() - s.n_correct as f64).abs());
                ensure!(dif < 1e-4, "independent recomputation differs from in-process summary by {dif} (N={n} {model} u={u})");
                let mut v = serde_json::to_value(&s)?;
                v["strong_fit"] = json!(strong(&s));
                v["exact_memorization"] = json!(exact(&s));
                snaps.insert(u.to_string(), v);
            }
            let tr = trace["trace"].as_array().unwrap();
            let win = |a0: usize, b0: usize| -> serde_json::Value {
                let s = &tr[a0..b0];
                json!({"mean_loss": s.iter().map(|x| x["loss"].as_f64().unwrap()).sum::<f64>() / s.len() as f64, "mean_grad_norm": s.iter().map(|x| x["grad_norm_pre_clip"].as_f64().unwrap()).sum::<f64>() / s.len() as f64, "clip_fraction": s.iter().filter(|x| x["clipped"].as_bool().unwrap()).count() as f64 / s.len() as f64})
            };
            let dyn_ = json!({"updates_0_100": win(0, 100), "updates_100_600": win(100, 600), "updates_600_1200": win(600, 1200), "updates_1200_2000": win(1200, 2000), "updates_2000_2200": win(2000, 2200), "updates_2200_2400": win(2200, 2400)});
            table.insert(format!("{model}/N{n}"), json!({"provenance_id": prov["provenance_id"], "predictions_sha256": prov["predictions_sha256"], "snapshots": snaps, "dynamics": dyn_, "wall_secs": trace["wall_secs"], "mean_update_ms": trace["mean_update_ms"], "exposure_min_max": [expo.iter().min(), expo.iter().max()], "movement_file": format!("{dir}/movement.json"), "variation_file": if model == "A" { json!(format!("{dir}/variation.json")) } else { json!(null) }}));
        }
    }
    report.insert("runs".into(), json!(table));
    // comparisons
    let cmp_equal_update: Vec<serde_json::Value> = ["A", "M"].iter().flat_map(|m| D2_SIZES.iter().map(move |n| (m, n))).map(|(m, n)| { let t = at[&(m.to_string(), *n, 600)]; json!({"model": m, "N": n, "update": 600, "acc": t.0, "bce": t.1, "auroc": t.2}) }).collect();
    let expo_map = [(64usize, 50usize, 200usize), (256, 200, 800), (768, 600, 2400)];
    let mut cmp_expo = Vec::new();
    for m in ["A", "M"] {
        for (n, u125, u50) in expo_map {
            for (exposure, u) in [(12.5, u125), (50.0, u50)] {
                let t = at[&(m.to_string(), n, u)];
                cmp_expo.push(json!({"model": m, "N": n, "avg_exposure_per_example": exposure, "update": u, "acc": t.0, "bce": t.1, "auroc": t.2, "note": "learning rates at equal-exposure checkpoints differ (schedule confound); no pure causal attribution"}));
            }
        }
    }
    report.insert("equal_update_600".into(), json!(cmp_equal_update));
    report.insert("equal_average_exposure".into(), json!(cmp_expo));
    let ep: Vec<serde_json::Value> = endpoint.iter().map(|((m, n), (s, e))| json!({"model": m, "N": n, "strong_fit_at_2400": s, "exact_memorization_at_2400": e})).collect();
    report.insert("endpoints_at_2400".into(), json!(ep));
    let a768 = endpoint.get(&("A".to_string(), 768)).map(|x| x.0).unwrap_or(false);
    let m768 = endpoint.get(&("M".to_string(), 768)).map(|x| x.0).unwrap_or(false);
    let decision = match (a768, m768) {
        (true, true) => "BOTH_STRONG_FIT_768: larger-set trainability established under D2; propose a separately frozen generalization experiment.",
        (false, true) => "M_ONLY_FITS_768: prioritize current-architecture/optimization-path investigation; M is a useful simple control, not a proven chess model.",
        (true, false) => "A_ONLY_FITS_768: retain A as a candidate; recurrence still has no demonstrated benefit.",
        (false, false) => "NEITHER_FITS_768: report the degradation bracket, exposure comparisons and training dynamics; do not enlarge model or dataset yet.",
    };
    report.insert("decision".into(), json!(decision));
    report.insert("strong_fit_definition".into(), json!("balanced accuracy >= 95% and BCE <= 0.05 (fitting diagnostic, not a generalization gate)"));
    report.insert("label".into(), json!("D2 is fitting-only. No validation or sealed use. The shallow baseline's exposed-validation result remains retrospective."));
    report.insert("source".into(), serde_json::to_value(source_id())?);
    let text = serde_json::to_string_pretty(&serde_json::Value::Object(report))?;
    a.write(Path::new("d2/report/d2_report.json"), text.as_bytes())?;
    println!("{decision}");
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("subsets") => cmd_subsets(&args[2..]),
        Some("preserve") => cmd_preserve(&args[2..]),
        Some("freeze") => cmd_freeze(&args[2..]),
        Some("aggregate") => cmd_aggregate(&args[2..]),
        _ => bail!("usage: v69-d2 <subsets|preserve|freeze|aggregate> --artifacts DIR"),
    }
}
