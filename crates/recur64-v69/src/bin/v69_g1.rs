//! v69-g1: data-side tooling for the G1 fresh generalization evaluation (host only).
//!   protocol-freeze | preserve --label | generate | audit | intervention | freeze | aggregate
//! Every open goes through the role-restricted Access; receipts are append-only.

use anyhow::{Context, Result, bail, ensure};
use cozy_chess::Board;
use recur64_v69::access::{Access, Role};
use recur64_v69::canon::{canonical_key, key_hex, key_id};
use recur64_v69::custody::Custody;
use recur64_v69::d1::{FrozenD1, verify_group};
use recur64_v69::d2::write_new;
use recur64_v69::dataset::{Example, RootOutcome, RootRec, analyze_root, build_groups};
use recur64_v69::features::{featurize, read_rows};
use recur64_v69::g1::*;
use recur64_v69::generate::{Family, Reject, sample_root};
use recur64_v69::oracle::{Oracle, Verdict};
use recur64_v69::provenance::{sha256_hex, source_digest, source_id};
use recur64_v69::reference;
use recur64_v69::streams::MasterSeed;
use serde::Serialize;
use serde_json::json;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

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
            seed_rel: PathBuf::from("g1/seed/g1_master_seed.hex"),
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

fn text_of<T: Serialize>(rows: &[T]) -> String {
    rows.iter().map(|r| serde_json::to_string(r).unwrap() + "\n").collect()
}

#[derive(Serialize, Default, Clone)]
struct Fam {
    attempted: u64,
    illegal_overlap: u64,
    illegal_adjacent_kings: u64,
    illegal_defender_in_check: u64,
    illegal_other: u64,
    duplicate: u64,
    analyzed: u64,
    accepted_m2: u64,
    accepted_m3: u64,
    rejected_m1: u64,
    rejected_no_mate_within_3: u64,
    unresolved_node_limit: u64,
}

const FROZEN_PROTOCOL: &str = "g1/frozen_protocol.json";
const FROZEN_G1: &str = "g1/frozen_g1.json";

// ------------------------------------------------------------------ protocol-freeze

fn cmd_protocol_freeze(args: &[String]) -> Result<()> {
    let env = Env::from(args)?;
    let a = env.access(Role::DataAudit)?;
    ensure!(!a.custody().resolve(Path::new(FROZEN_PROTOCOL))?.exists(), "protocol already frozen");
    ensure!(!a.custody().resolve(&env.seed_rel)?.exists(), "protocol must be frozen BEFORE the G1 seed exists");
    let mut files: Vec<String> = ["g1/G1_CONTRACT.md", "g1/config.json", "g1/d3_supplementary_manifest.json", "g1/receipts/preservation_start.json", "d1/e1_supplementary_manifest.json", "d2/d1_supplementary_manifest.json", "d3/d2_supplementary_manifest.json", "spec/frozen.json", "d1/frozen_d1.json", "d2/frozen_d2.json", "d3/frozen_d3.json", "d1/baseline/model.json", "d1/baseline/predictions_fit.jsonl", "d2/subsets/s768_rows.jsonl", "d3/report/d3_report.json", "d1/report/d1_report.json", "dataset:MANIFEST.sha256.json"].iter().map(|s| s.to_string()).collect();
    for m in ["A", "M"] {
        for f in ["model.mpk", "meta.json", "opt_decay.mpk", "opt_nodecay.mpk"] {
            files.push(format!("d3/fits/{m}/final/{f}"));
        }
        files.push(format!("d3/fits/{m}/provenance.json"));
        files.push(format!("d3/fits/{m}/predictions.jsonl"));
    }
    let mut g = BTreeMap::new();
    for rel in files {
        let bytes = match rel.strip_prefix("dataset:") {
            Some(d) => a.read(&env.run.join(d))?,
            None => a.read(Path::new(&rel)).with_context(|| format!("hash {rel}"))?,
        };
        g.insert(rel, sha256_hex(&bytes));
    }
    let mut groups = BTreeMap::new();
    groups.insert("protocol".to_string(), g);
    let fz = FrozenD1 { created_utc: format!("unix:{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs()), run: "gen-001".into(), seed_fingerprint: "(not yet drawn)".into(), groups };
    let text = serde_json::to_string_pretty(&fz)?;
    write_new(&a, FROZEN_PROTOCOL, text.as_bytes())?;
    println!("frozen_protocol.json sha256 {}", sha256_hex(text.as_bytes()));
    Ok(())
}

// ------------------------------------------------------------------ preserve

fn cmd_preserve(args: &[String]) -> Result<()> {
    let env = Env::from(args)?;
    let a = env.access(Role::DataAudit)?;
    let label = arg(args, "--label").context("--label")?;
    let mut mismatches = Vec::new();
    let mut n = 0;
    for (mf, what) in [("d1/e1_supplementary_manifest.json", "E1"), ("d2/d1_supplementary_manifest.json", "D1"), ("d3/d2_supplementary_manifest.json", "D2"), ("g1/d3_supplementary_manifest.json", "D3")] {
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
        "documented_e1_deviation": "E1 audit receipt audit/gen-001_audit_receipt_v2.json was overwritten in place by the E1 post-campaign audit; pre-run copy preserved. G1 never writes to audit/; receipts are append-only.",
        "source": source_id()});
    write_new(&a, &format!("g1/receipts/preservation_{label}.json"), serde_json::to_string_pretty(&rep)?.as_bytes())?;
    println!("{}", serde_json::to_string_pretty(&rep)?);
    if !ok {
        std::process::exit(4);
    }
    Ok(())
}

// ------------------------------------------------------------------ generate

fn load_gen001_pool(a: &Access, run: &Path) -> Result<Vec<RootRec>> {
    let manifest: BTreeMap<String, String> = serde_json::from_slice(&a.read(&run.join("MANIFEST.sha256.json"))?)?;
    let b = a.read(&run.join("pool/roots.jsonl"))?;
    ensure!(manifest.get("pool/roots.jsonl") == Some(&sha256_hex(&b)), "gen-001 pool hash mismatch");
    jsonl(std::str::from_utf8(&b)?)
}

fn cmd_generate(args: &[String]) -> Result<()> {
    let t_start = Instant::now();
    let env = Env::from(args)?;
    let a = env.access(Role::G1Builder)?;
    let seed = env.seed(&a)?;
    let time_limit: u64 = arg(args, "--time-limit-secs").map(|s| s.parse()).transpose()?.unwrap_or(3600);
    let node_limit: u64 = arg(args, "--node-limit").map(|s| s.parse()).transpose()?.unwrap_or(5_000_000);
    let threads: usize = arg(args, "--threads").map(|s| s.parse()).transpose()?.unwrap_or(10);
    let round_size: u64 = arg(args, "--round-size").map(|s| s.parse()).transpose()?.unwrap_or(1000);
    ensure!(!a.custody().resolve(Path::new("g1/rows/g1_rows.jsonl"))?.exists(), "G1 data already generated; no regeneration");
    if arg(args, "--diagnose-rounds").is_none() {
        ensure!(!a.custody().resolve(Path::new("g1/meta/infeasibility_diagnosis.json"))?.exists(), "a failed construction was already diagnosed; the registered procedure is not retried");
    }
    // contamination-exclusion index from gen-001 roots and ALL their immediate children (data-only)
    let gen001 = load_gen001_pool(&a, &env.run)?;
    let idx = ExclusionIndex::from_pool(&gen001);
    let idx_text = serde_json::to_string(&idx)?;
    a.write(Path::new("g1/index/gen001_exclusion_index.json"), idx_text.as_bytes())?;
    eprintln!("[g1] seed fingerprint {} | exclusion index: {} roots, {} children | limits {time_limit}s, {node_limit} nodes, {threads} threads", seed.fingerprint(), idx.roots.len(), idx.children.len());
    let deadline = t_start + Duration::from_secs(time_limit);
    let mut stats: BTreeMap<&'static str, Fam> = Family::ALL.iter().map(|f| (f.name(), Fam::default())).collect();
    let mut seen: HashSet<String> = HashSet::new();
    let mut pool: Vec<RootRec> = Vec::new();
    let mut status = "wall_limit_reached".to_string();
    let mut result: Option<(Vec<RootRec>, Vec<recur64_v69::dataset::Group>, ExclusionStats, Vec<Example>, BTreeMap<String, usize>)> = None;
    let mut rounds = 0u64;
    let diagnose_rounds: Option<u64> = arg(args, "--diagnose-rounds").map(|s| s.parse()).transpose()?;
    let mut diag: Vec<serde_json::Value> = Vec::new();
    for round in 0..10_000u64 {
        if Instant::now() >= deadline {
            break;
        }
        let mut tasks: Vec<(Family, Board, cozy_chess::Color)> = Vec::new();
        let mut local: HashSet<String> = HashSet::new();
        for fam in Family::ALL {
            for k in 0..round_size {
                let s = stats.get_mut(fam.name()).unwrap();
                s.attempted += 1;
                match sample_root(&seed, fam, round * round_size + k) {
                    Err(Reject::Overlap) => s.illegal_overlap += 1,
                    Err(Reject::AdjacentKings) => s.illegal_adjacent_kings += 1,
                    Err(Reject::DefenderInCheck) => s.illegal_defender_in_check += 1,
                    Err(Reject::BoardInvalid) => s.illegal_other += 1,
                    Ok(r) => {
                        let key = key_hex(&canonical_key(&r.board, r.attacker));
                        if seen.contains(&key) || !local.insert(key) {
                            s.duplicate += 1;
                        } else {
                            tasks.push((fam, r.board, r.attacker));
                        }
                    }
                }
            }
        }
        let next = AtomicUsize::new(0);
        let results: Vec<std::sync::Mutex<Option<(Family, RootOutcome)>>> = (0..tasks.len()).map(|_| std::sync::Mutex::new(None)).collect();
        let timed_out = std::sync::atomic::AtomicBool::new(false);
        std::thread::scope(|sc| {
            for _ in 0..threads {
                sc.spawn(|| {
                    let mut o = Oracle::new(node_limit);
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= tasks.len() {
                            break;
                        }
                        if Instant::now() >= deadline {
                            timed_out.store(true, Ordering::Relaxed);
                            break;
                        }
                        let (fam, board, att) = &tasks[i];
                        *results[i].lock().unwrap() = Some((*fam, analyze_root(&mut o, *fam, board, *att)));
                    }
                });
            }
        });
        if timed_out.load(Ordering::Relaxed) {
            break; // incomplete round discarded whole (deterministic prefix of rounds)
        }
        for (t, slot) in tasks.iter().zip(results) {
            let (fam, out) = slot.into_inner().unwrap().expect("complete round");
            let s = stats.get_mut(fam.name()).unwrap();
            s.analyzed += 1;
            seen.insert(key_hex(&canonical_key(&t.1, t.2)));
            match out {
                RootOutcome::Accepted(r) => {
                    if r.depth == 2 {
                        s.accepted_m2 += 1
                    } else {
                        s.accepted_m3 += 1
                    }
                    pool.push(*r);
                }
                RootOutcome::RejectedM1 => s.rejected_m1 += 1,
                RootOutcome::RejectedBeyond => s.rejected_no_mate_within_3 += 1,
                RootOutcome::Unresolved => s.unresolved_node_limit += 1,
            }
        }
        rounds = round + 1;
        let (kept, groups, est) = apply_exclusion(&seed, pool.clone(), &idx);
        let (examples, counts) = select_g1(&seed, &kept, &groups);
        let short: usize = counts.values().map(|c| G1_QUOTA.saturating_sub(*c)).sum();
        eprintln!("[g1] round {round}: pool {} roots, kept {} after exclusion ({} groups), {short} examples short | {:.0}s", pool.len(), kept.len(), groups.len(), t_start.elapsed().as_secs_f64());
        if let Some(max_rounds) = diagnose_rounds {
            // DIAGNOSTIC ONLY (statistics; no rows/panel are written): strict whole-component exclusion vs a per-root exclusion variant
            let direct_bad = pool.iter().filter(|r| idx.roots.binary_search(&r.key).is_ok() || r.children.iter().any(|c| idx.children.binary_search(&c.key).is_ok())).count();
            let kept2: Vec<RootRec> = pool.iter().filter(|r| !(idx.roots.binary_search(&r.key).is_ok() || r.children.iter().any(|c| idx.children.binary_search(&c.key).is_ok()))).cloned().collect();
            let groups2 = build_groups(&seed, &kept2);
            let (_, counts2) = select_g1(&seed, &kept2, &groups2);
            let largest = groups.iter().map(|g| g.roots.len()).max().unwrap_or(0);
            let largest_pre = build_groups(&seed, &pool).iter().map(|g| g.roots.len()).max().unwrap_or(0);
            diag.push(json!({"round": round, "pool_roots": pool.len(), "roots_with_direct_gen001_overlap": direct_bad, "direct_overlap_fraction": direct_bad as f64 / pool.len() as f64, "largest_component_before_exclusion": largest_pre, "strict_rule": {"kept_roots": kept.len(), "groups": groups.len(), "largest_group": largest, "examples_short": short, "per_cell_available": counts}, "per_root_rule_variant": {"kept_roots": kept2.len(), "groups": groups2.len(), "examples_short": counts2.values().map(|c| G1_QUOTA.saturating_sub(*c)).sum::<usize>(), "per_cell_available": counts2}}));
            if round + 1 >= max_rounds {
                a.write(Path::new("g1/meta/infeasibility_diagnosis.json"), serde_json::to_string_pretty(&json!({"diagnostic_only": true, "no_rows_or_panel_written": true, "rounds": diag}))?.as_bytes())?;
                println!("diagnosis written after {max_rounds} rounds");
                return Ok(());
            }
        }
        if feasible(&counts) {
            status = "complete_feasible".into();
            result = Some((kept, groups, est, examples, counts));
            break;
        }
    }
    let Some((kept, groups, est, examples, counts)) = result else {
        let rep = json!({"status": status, "feasible": false, "rounds": rounds, "per_family_stats": stats, "pool_roots_before_exclusion": pool.len(), "wall_secs": t_start.elapsed().as_secs_f64()});
        a.write(Path::new("g1/meta/generation_report_FAILED.json"), serde_json::to_string_pretty(&rep)?.as_bytes())?;
        eprintln!("[g1] GENERATION INFEASIBLE within limits: {rep}");
        std::process::exit(3);
    };
    ensure!(examples.len() == G1_TOTAL, "example count");
    let mut sorted = examples.clone();
    sorted.sort_by(|x, y| x.id.cmp(&y.id));
    #[derive(Serialize)]
    struct Row<'x> {
        id: &'x str,
        fen: &'x str,
        budget: u8,
        label: bool,
    }
    let rows: Vec<Row> = sorted.iter().map(|e| Row { id: &e.id, fen: &e.fen, budget: e.budget, label: e.label }).collect();
    let (t_rows, t_meta, t_pool) = (text_of(&rows), text_of(&sorted), text_of(&kept));
    a.write(Path::new("g1/rows/g1_rows.jsonl"), t_rows.as_bytes())?;
    a.write(Path::new("g1/meta/g1_meta.jsonl"), t_meta.as_bytes())?;
    a.write(Path::new("g1/pool/roots.jsonl"), t_pool.as_bytes())?;
    let used_groups: HashSet<&str> = sorted.iter().map(|e| e.group_id.as_str()).collect();
    let used_roots: HashSet<&str> = sorted.iter().map(|e| e.root_id.as_str()).collect();
    let mut size_hist: BTreeMap<usize, usize> = BTreeMap::new();
    for g in &groups {
        *size_hist.entry(g.roots.len()).or_default() += 1;
    }
    let report = json!({
        "status": status, "feasible": true, "rounds": rounds, "round_size_per_family": round_size,
        "limits": {"wall_secs": time_limit, "node_limit_per_query": node_limit, "threads": threads},
        "wall_secs": t_start.elapsed().as_secs_f64(),
        "seed_fingerprint": seed.fingerprint(),
        "per_family_stats": stats, "pool_roots_before_exclusion": pool.len(),
        "exclusion": est, "exclusion_index": {"gen001_roots": idx.roots.len(), "gen001_children": idx.children.len(), "index_sha256": sha256_hex(idx_text.as_bytes())},
        "kept_roots": kept.len(), "groups": {"count": groups.len(), "size_histogram_roots": size_hist, "largest": groups.iter().map(|g| g.roots.len()).max()},
        "selected": {"examples": sorted.len(), "distinct_groups": used_groups.len(), "distinct_roots": used_roots.len(), "per_cell": counts},
        "source": source_id(),
    });
    a.write(Path::new("g1/meta/generation_report.json"), serde_json::to_string_pretty(&report)?.as_bytes())?;
    let mut man = BTreeMap::new();
    for (rel, t) in [("g1/rows/g1_rows.jsonl", &t_rows), ("g1/meta/g1_meta.jsonl", &t_meta), ("g1/pool/roots.jsonl", &t_pool)] {
        man.insert(rel.to_string(), sha256_hex(t.as_bytes()));
    }
    man.insert("g1/index/gen001_exclusion_index.json".into(), sha256_hex(idx_text.as_bytes()));
    man.insert("g1/meta/generation_report.json".into(), sha256_hex(&a.read(Path::new("g1/meta/generation_report.json"))?));
    a.write(Path::new("g1/MANIFEST.sha256.json"), serde_json::to_string_pretty(&man)?.as_bytes())?;
    println!("G1 generation complete: {} examples, {} roots kept ({} excluded in {} groups), {} groups, wall {:.1}s", sorted.len(), kept.len(), est.roots_removed, est.groups_removed, groups.len(), t_start.elapsed().as_secs_f64());
    Ok(())
}

// ------------------------------------------------------------------ audit

fn cmd_audit(args: &[String]) -> Result<()> {
    let t0 = Instant::now();
    let env = Env::from(args)?;
    let a = env.access(Role::G1Builder)?;
    let threads: usize = arg(args, "--threads").map(|s| s.parse()).transpose()?.unwrap_or(10);
    let ref_per_cell: usize = arg(args, "--ref-per-cell").map(|s| s.parse()).transpose()?.unwrap_or(8);
    let m3_roots: usize = arg(args, "--m3-roots").map(|s| s.parse()).transpose()?.unwrap_or(24);
    let mut failures: Vec<String> = Vec::new();
    macro_rules! check {
        ($n:expr, $ok:expr, $m:expr) => {
            if !$ok {
                failures.push(format!("[{}] {}", $n, $m));
            }
        };
    }
    let man: BTreeMap<String, String> = serde_json::from_slice(&a.read(Path::new("g1/MANIFEST.sha256.json"))?)?;
    for (f, h) in &man {
        check!("manifest", sha256_hex(&a.read(Path::new(f))?) == *h, format!("hash mismatch {f}"));
    }
    ensure!(failures.is_empty(), "manifest mismatch: {failures:?}");
    let rows = read_rows(&a.read_to_string(Path::new("g1/rows/g1_rows.jsonl"))?)?;
    let meta: Vec<Example> = jsonl(&a.read_to_string(Path::new("g1/meta/g1_meta.jsonl"))?)?;
    let pool: Vec<RootRec> = jsonl(&a.read_to_string(Path::new("g1/pool/roots.jsonl"))?)?;
    let mm: HashMap<&str, &Example> = meta.iter().map(|e| (e.id.as_str(), e)).collect();
    check!("rows", rows.len() == G1_TOTAL && meta.len() == G1_TOTAL && mm.len() == G1_TOTAL, "count/duplicate");
    for r in &rows {
        match mm.get(r.id.as_str()) {
            None => failures.push(format!("[rows] {} has no metadata", r.id)),
            Some(m) => {
                check!("rows", m.fen == r.fen && m.budget == r.budget && m.label == r.label, format!("row/meta differ {}", r.id));
                match featurize(&r.fen, r.budget) {
                    Ok((_, fam)) => check!("rows", fam.name() == m.family, format!("family differs {}", r.id)),
                    Err(e) => failures.push(format!("[featurize] {}: {e}", r.id)),
                }
            }
        }
    }
    // quotas / per-root cap
    let mut cells: BTreeMap<String, usize> = BTreeMap::new();
    let mut per_root: HashMap<(&str, bool), usize> = HashMap::new();
    for m in &meta {
        *cells.entry(format!("{}/n{}/{}", m.family, m.budget, m.label)).or_default() += 1;
        *per_root.entry((m.root_id.as_str(), m.label)).or_default() += 1;
    }
    check!("quota", cells.len() == 12 && cells.values().all(|c| *c == G1_QUOTA), format!("cells {cells:?}"));
    check!("cap", per_root.values().all(|c| *c <= 2), "per-root cap exceeded");
    // identities, membership, terminal exclusion
    let pool_by_id: HashMap<&str, &RootRec> = pool.iter().map(|r| (r.id.as_str(), r)).collect();
    for r in &pool {
        let rb: Board = r.fen.parse()?;
        let att = rb.side_to_move();
        check!("canon", key_hex(&canonical_key(&rb, att)) == r.key && key_id(&r.key) == r.id, format!("root identity {}", r.id));
        for c in &r.children {
            let cb: Board = c.fen.parse()?;
            check!("canon", key_hex(&canonical_key(&cb, att)) == c.key, format!("child identity {}", r.id));
        }
    }
    let mut taken: HashSet<(String, u8)> = HashSet::new();
    for m in &meta {
        check!("canon", taken.insert((m.key.clone(), m.budget)), format!("duplicate canonical child {}", m.id));
        let Some(root) = pool_by_id.get(m.root_id.as_str()) else {
            failures.push(format!("[membership] root {} missing", m.root_id));
            continue;
        };
        check!("membership", m.budget + 1 == root.depth && root.depth == m.root_depth && root.fen == m.root_fen, format!("root/budget {}", m.id));
        match root.children.iter().find(|c| c.mv == m.mv) {
            None => failures.push(format!("[membership] {} not a child of its root", m.id)),
            Some(c) => {
                check!("membership", c.fen == m.fen && c.key == m.key, format!("child identity {}", m.id));
                check!("terminal", c.status == "pos" || c.status == "neg", format!("terminal child selected {}", m.id));
                check!("membership", (c.status == "pos") == m.label, format!("stored status vs label {}", m.id));
            }
        }
    }
    // cross-gen-001 exclusion with full keys (data-only inspection of gen-001 pool)
    let gen001 = load_gen001_pool(&a, &env.run)?;
    let idx = ExclusionIndex::from_pool(&gen001);
    let (rs, cs): (HashSet<&str>, HashSet<&str>) = (idx.roots.iter().map(String::as_str).collect(), idx.children.iter().map(String::as_str).collect());
    let mut overlaps = 0;
    for r in &pool {
        if rs.contains(r.key.as_str()) || r.children.iter().any(|c| cs.contains(c.key.as_str())) {
            overlaps += 1;
        }
    }
    for m in &meta {
        if cs.contains(m.key.as_str()) || rs.contains(m.key.as_str()) {
            overlaps += 1;
        }
    }
    check!("exclusion", overlaps == 0, format!("{overlaps} canonical overlaps with gen-001"));
    let stored_idx: ExclusionIndex = serde_json::from_slice(&a.read(Path::new("g1/index/gen001_exclusion_index.json"))?)?;
    check!("exclusion", stored_idx.roots == idx.roots && stored_idx.children == idx.children, "stored exclusion index differs from the rebuilt one");
    // groups rebuilt
    let seed = env.seed(&a)?;
    let groups = build_groups(&seed, &pool);
    let mut rg: HashMap<&str, &str> = HashMap::new();
    for g in &groups {
        for &i in &g.roots {
            rg.insert(pool[i].id.as_str(), g.id.as_str());
        }
    }
    for m in &meta {
        check!("groups", rg.get(m.root_id.as_str()) == Some(&m.group_id.as_str()), format!("group id differs {}", m.id));
    }
    // fresh exact re-query of every example + empty-cache re-analysis of contributing roots
    let next = AtomicUsize::new(0);
    let bad = std::sync::Mutex::new(Vec::<String>::new());
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| {
                let mut o = Oracle::new(5_000_000);
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= meta.len() {
                        break;
                    }
                    let m = &meta[i];
                    o.clear_cache();
                    let cb: Board = m.fen.parse().unwrap();
                    let v = o.mate_within_child(&cb, !cb.side_to_move(), m.budget);
                    let ok = match v {
                        Verdict::Yes => m.label,
                        Verdict::No => !m.label,
                        Verdict::Unknown => false,
                    };
                    if !ok {
                        bad.lock().unwrap().push(format!("[exact-target] {} label {} oracle {:?}", m.id, m.label, v));
                    }
                }
            });
        }
    });
    let used: HashSet<&str> = meta.iter().map(|m| m.root_id.as_str()).collect();
    let used_roots: Vec<&RootRec> = pool.iter().filter(|r| used.contains(r.id.as_str())).collect();
    let next = AtomicUsize::new(0);
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| {
                let mut o = Oracle::new(5_000_000);
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= used_roots.len() {
                        break;
                    }
                    let r = used_roots[i];
                    let b: Board = r.fen.parse().unwrap();
                    let fam = Family::ALL.iter().copied().find(|f| f.name() == r.family).unwrap();
                    match analyze_root(&mut o, fam, &b, b.side_to_move()) {
                        RootOutcome::Accepted(r2) => {
                            if r2.depth != r.depth || r2.children.len() != r.children.len() || r2.children.iter().zip(&r.children).any(|(x, y)| x.mv != y.mv || x.status != y.status) {
                                bad.lock().unwrap().push(format!("[root-reanalysis] {} differs", r.id));
                            }
                        }
                        _ => bad.lock().unwrap().push(format!("[root-reanalysis] {} not accepted", r.id)),
                    }
                }
            });
        }
    });
    failures.extend(bad.into_inner().unwrap());
    // independent reference: child targets (per-cell sample) + bounded M3 minimal-depth sample
    let mut sample: Vec<&Example> = Vec::new();
    for f in Family::ALL {
        for b in [1u8, 2] {
            for c in [false, true] {
                let mut v: Vec<&Example> = meta.iter().filter(|e| e.family == f.name() && e.budget == b && e.label == c).collect();
                v.sort_by(|x, y| x.id.cmp(&y.id));
                let step = (v.len() / ref_per_cell.max(1)).max(1);
                sample.extend(v.into_iter().step_by(step).take(ref_per_cell));
            }
        }
    }
    let next = AtomicUsize::new(0);
    let rbad = std::sync::Mutex::new(Vec::<String>::new());
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= sample.len() {
                    break;
                }
                let e = sample[i];
                let b: Board = e.fen.parse().unwrap();
                if reference::child_target(&b, !b.side_to_move(), e.budget) != e.label {
                    rbad.lock().unwrap().push(format!("[reference] disagrees on {}", e.id));
                }
            });
        }
    });
    let mut m3: Vec<&RootRec> = pool.iter().filter(|r| r.depth == 3 && used.contains(r.id.as_str())).collect();
    m3.sort_by(|x, y| x.id.cmp(&y.id));
    let step = (m3.len() / m3_roots.max(1)).max(1);
    let m3s: Vec<&RootRec> = m3.into_iter().step_by(step).take(m3_roots).collect();
    let next = AtomicUsize::new(0);
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= m3s.len() {
                    break;
                }
                let r = m3s[i];
                let b: Board = r.fen.parse().unwrap();
                if reference::root_min_depth(&b, b.side_to_move(), 3) != Some(3) {
                    rbad.lock().unwrap().push(format!("[reference-M3] minimal depth differs for root {}", r.id));
                }
            });
        }
    });
    failures.extend(rbad.into_inner().unwrap());
    let ok = failures.is_empty();
    let rep = json!({"audit_pass": ok, "failures": failures, "examples": meta.len(), "roots_in_pool": pool.len(), "contributing_roots_reanalysed_empty_cache": used_roots.len(), "targets_requeried_exact": meta.len(), "reference_child_checks": sample.len(), "reference_m3_minimal_depth_checks": m3s.len(),
        "gen001_overlaps_found": overlaps, "gen001_index_sizes": {"roots": idx.roots.len(), "children": idx.children.len()},
        "reference_shared_dependencies": "cozy-chess move generator/legality (cross-checked vs brute-force is_legal in tests), FEN parser, domain definition; NOT the search, cache, node budget or rules::classify code",
        "audit_wall_secs": t0.elapsed().as_secs_f64(), "source": source_id()});
    write_new(&a, "g1/receipts/audit_receipt_g1.json", serde_json::to_string_pretty(&rep)?.as_bytes())?;
    println!("{}", serde_json::to_string_pretty(&rep)?);
    if !ok {
        std::process::exit(4);
    }
    Ok(())
}

// ------------------------------------------------------------------ intervention

fn cmd_intervention(args: &[String]) -> Result<()> {
    let env = Env::from(args)?;
    let a = env.access(Role::G1Builder)?;
    let seed = env.seed(&a)?;
    ensure!(a.read_to_string(Path::new("g1/receipts/audit_receipt_g1.json")).map(|t| t.contains("\"audit_pass\": true")).unwrap_or(false), "audit must pass before the intervention map is frozen");
    let meta: Vec<Example> = jsonl(&a.read_to_string(Path::new("g1/meta/g1_meta.jsonl"))?)?;
    let map = intervention_map(&seed, &meta)?; // label-independent: uses only family, budget and id
    let body = serde_json::to_string_pretty(&json!({"derangement": "cyclic shift of a keyed shuffle within family x budget cells; label independent", "stream": format!("{STREAM_G1_INTERVENTION}/<family>/<budget>"), "seed_fingerprint": seed.fingerprint(), "map": map}))?;
    write_new(&a, "g1/intervention/map.json", body.as_bytes())?;
    write_new(&a, "g1/intervention/map.sha256", sha256_hex(body.as_bytes()).as_bytes())?;
    println!("intervention map sha256 {}", sha256_hex(body.as_bytes()));
    Ok(())
}

// ------------------------------------------------------------------ freeze (before inference)

fn cmd_freeze(args: &[String]) -> Result<()> {
    let env = Env::from(args)?;
    let a = env.access(Role::DataAudit)?;
    ensure!(!a.custody().resolve(Path::new(FROZEN_G1))?.exists(), "G1 already frozen");
    let h = |rel: &str| -> Result<String> { Ok(sha256_hex(&a.read(Path::new(rel)).with_context(|| format!("hash {rel}"))?)) };
    let mut ev: Vec<String> = ["g1/G1_CONTRACT.md", "g1/config.json", FROZEN_PROTOCOL, "g1/rows/g1_rows.jsonl", "g1/MANIFEST.sha256.json", "g1/intervention/map.json", "g1/intervention/map.sha256", "g1/evaluator_source.json", "g1/receipts/evaluator_verification.json", "d1/baseline/model.json", "d1/baseline/predictions_fit.jsonl", "d2/subsets/s768_rows.jsonl"].iter().map(|s| s.to_string()).collect();
    for m in ["A", "M"] {
        for f in ["model.mpk", "meta.json"] {
            ev.push(format!("d3/fits/{m}/final/{f}"));
        }
        ev.push(format!("d3/fits/{m}/provenance.json"));
        ev.push(format!("d3/fits/{m}/predictions.jsonl"));
    }
    let agg: Vec<String> = ["g1/G1_CONTRACT.md", "g1/config.json", FROZEN_PROTOCOL, "g1/rows/g1_rows.jsonl", "g1/meta/g1_meta.jsonl", "g1/MANIFEST.sha256.json", "g1/intervention/map.json", "g1/receipts/audit_receipt_g1.json", "g1/receipts/evaluator_verification.json", "d3/report/d3_report.json", "d1/report/d1_report.json"].iter().map(|s| s.to_string()).collect();
    let mut groups = BTreeMap::new();
    for (n, list) in [("evaluator", ev), ("aggregator", agg)] {
        let mut m = BTreeMap::new();
        for rel in list {
            m.insert(rel.clone(), h(&rel)?);
        }
        groups.insert(n.to_string(), m);
    }
    let seed = env.seed(&a)?;
    let fz = FrozenD1 { created_utc: format!("unix:{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs()), run: "gen-001".into(), seed_fingerprint: seed.fingerprint(), groups };
    let text = serde_json::to_string_pretty(&fz)?;
    write_new(&a, FROZEN_G1, text.as_bytes())?;
    println!("frozen_g1.json sha256 {}", sha256_hex(text.as_bytes()));
    Ok(())
}

// ------------------------------------------------------------------ evaluator source record

fn cmd_source(args: &[String]) -> Result<()> {
    let env = Env::from(args)?;
    let a = env.access(Role::DataAudit)?;
    let sid = source_id();
    ensure!(sid.git_dirty_files == 0, "evaluator source must be committed and clean before it is frozen ({} dirty files)", sid.git_dirty_files);
    write_new(&a, "g1/evaluator_source.json", serde_json::to_string_pretty(&json!({"git_head": sid.git_head, "source_digest": source_digest(), "note": "executable source of the G1 evaluator (data crate + model crate + core rules + lock + toolchain)"}))?.as_bytes())?;
    println!("evaluator source {} digest {}", sid.git_head, sid.source_digest);
    Ok(())
}

// ------------------------------------------------------------------ aggregate

#[derive(Clone, Default)]
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
    fn acc(&self) -> f64 {
        (self.tp + self.tn) / self.n.max(1.0)
    }
    fn ba(&self) -> f64 {
        0.5 * (self.tp / (self.tp + self.fnn).max(1.0) + self.tn / (self.tn + self.fp).max(1.0))
    }
    fn bce(&self) -> f64 {
        self.bce / self.n.max(1.0)
    }
}

fn pct(v: &mut Vec<f64>, p: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() as f64 - 1.0) * p).round() as usize]
}

fn cmd_aggregate(args: &[String]) -> Result<()> {
    let env = Env::from(args)?;
    let a = env.access(Role::G1Aggregator)?;
    let seed = env.seed(&a)?;
    let frozen: FrozenD1 = serde_json::from_slice(&a.read(Path::new(FROZEN_G1))?)?;
    let nv = verify_group(&a, &env.run, &frozen, "aggregator")?;
    eprintln!("[g1 aggregate] {nv} frozen hashes verified");
    let meta: Vec<Example> = jsonl(&a.read_to_string(Path::new("g1/meta/g1_meta.jsonl"))?)?;
    ensure!(meta.len() == G1_TOTAL, "meta size");
    let mm: HashMap<&str, &Example> = meta.iter().map(|e| (e.id.as_str(), e)).collect();
    let map_v: serde_json::Value = serde_json::from_slice(&a.read(Path::new("g1/intervention/map.json"))?)?;
    let dmap: HashMap<String, String> = map_v["map"].as_object().unwrap().iter().map(|(k, v)| (k.clone(), v.as_str().unwrap().to_string())).collect();
    let audit: serde_json::Value = serde_json::from_slice(&a.read(Path::new("g1/receipts/audit_receipt_g1.json"))?)?;
    let audit_ok = audit["audit_pass"].as_bool() == Some(true);
    let fz_sha = sha256_hex(&a.read(Path::new(FROZEN_G1))?);
    // group index
    let mut gids: Vec<&str> = meta.iter().map(|e| e.group_id.as_str()).collect::<HashSet<_>>().into_iter().collect();
    gids.sort();
    let gpos: HashMap<&str, usize> = gids.iter().enumerate().map(|(i, g)| (*g, i)).collect();
    let ng = gids.len();
    // shared bootstrap resamples (paired): same group draws for every candidate and comparison
    let resamples = 5000usize;
    let mut rng = seed.stream(STREAM_G1_BOOTSTRAP, 0);
    let draws: Vec<Vec<usize>> = (0..resamples).map(|_| (0..ng).map(|_| rng.below(ng as u64) as usize).collect()).collect();
    let tol_neural = 2e-3;
    let tol_base = 1e-9;
    let mut report = serde_json::Map::new();
    let mut per_group: BTreeMap<String, (Vec<Counts>, Vec<Counts>)> = BTreeMap::new(); // real, derange-vs-recipient per group
    let mut points: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    let mut summ: BTreeMap<String, (f64, f64, f64, f64)> = BTreeMap::new(); // ba, acc, bce, drop
    let mut integrity_ok = audit_ok;
    for cand in ["A", "M", "B"] {
        let dir = format!("g1/eval/{cand}");
        let prov: serde_json::Value = serde_json::from_slice(&a.read(Path::new(&format!("{dir}/provenance.json")))?).with_context(|| format!("candidate {cand} evaluation missing/failed"))?;
        let pb = a.read(Path::new(&format!("{dir}/predictions.jsonl")))?;
        ensure!(prov["predictions_sha256"].as_str() == Some(sha256_hex(&pb).as_str()), "{cand}: predictions hash differs from provenance");
        ensure!(prov["frozen_g1_sha256"].as_str() == Some(fz_sha.as_str()), "{cand}: evaluation was not bound to the frozen G1 file");
        #[derive(serde::Deserialize)]
        struct P {
            id: String,
            mode: String,
            logit: f64,
            donor_id: Option<String>,
        }
        let preds: Vec<P> = jsonl(std::str::from_utf8(&pb)?)?;
        let mut real: HashMap<&str, f64> = HashMap::new();
        let mut der: Vec<&P> = Vec::new();
        let mut ers: Vec<&P> = Vec::new();
        for p in &preds {
            ensure!(mm.contains_key(p.id.as_str()) && p.logit.is_finite(), "{cand}: unknown id or non-finite logit {}", p.id);
            match p.mode.as_str() {
                "real" => ensure!(real.insert(&p.id, p.logit).is_none(), "duplicate real prediction"),
                "derange" => der.push(p),
                "erase" => ers.push(p),
                m => bail!("unknown mode {m}"),
            }
        }
        ensure!(real.len() == G1_TOTAL && der.len() == G1_TOTAL && ers.len() == G1_TOTAL, "{cand}: incomplete predictions");
        // real
        let mut cnt = Counts::default();
        let mut pg_real = vec![Counts::default(); ng];
        let (mut err_conf, mut err_n, mut ok_conf, mut ok_n) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let mut cells: BTreeMap<String, Counts> = BTreeMap::new();
        let mut brier = 0.0;
        let mut pos = Vec::new();
        let mut neg = Vec::new();
        for e in &meta {
            let z = real[e.id.as_str()];
            cnt.add(z, e.label);
            pg_real[gpos[e.group_id.as_str()]].add(z, e.label);
            cells.entry(format!("{}/n{}/{}", e.family, e.budget, if e.label { "pos" } else { "neg" })).or_default().add(z, e.label);
            brier += (1.0 / (1.0 + (-z).exp()) - e.label as u8 as f64).powi(2);
            if (z > 0.0) != e.label {
                err_conf += z.abs();
                err_n += 1.0;
            } else {
                ok_conf += z.abs();
                ok_n += 1.0;
            }
            if e.label { pos.push(z) } else { neg.push(z) }
        }
        let auroc = recur64_v69::d1_metrics::auroc(&meta.iter().map(|e| real[e.id.as_str()]).collect::<Vec<_>>(), &meta.iter().map(|e| e.label).collect::<Vec<_>>());
        // derangement
        let (mut rc, mut dc) = (Counts::default(), Counts::default());
        let mut pg_der = vec![Counts::default(); ng];
        let (mut maxdiff, mut agree) = (0f64, 0usize);
        for p in &der {
            let e = mm[p.id.as_str()];
            let donor = p.donor_id.as_deref().context("derange row without donor")?;
            ensure!(dmap.get(&p.id).map(String::as_str) == Some(donor), "{cand}: donor differs from the frozen map");
            let de = mm.get(donor).context("donor")?;
            ensure!(de.family == e.family && de.budget == e.budget, "donor outside cell");
            rc.add(p.logit, e.label);
            dc.add(p.logit, de.label);
            pg_der[gpos[e.group_id.as_str()]].add(p.logit, e.label);
            agree += (e.label == de.label) as usize;
            maxdiff = maxdiff.max((p.logit - real[donor]).abs());
        }
        let tol = if cand == "B" { tol_base } else { tol_neural };
        let donor_ok = maxdiff <= tol;
        let mut ec = Counts::default();
        let mut epos = 0.0;
        for p in &ers {
            let e = mm[p.id.as_str()];
            ec.add(p.logit, e.label);
            epos += (p.logit > 0.0) as u8 as f64;
        }
        integrity_ok &= donor_ok;
        summ.insert(cand.into(), (cnt.ba(), cnt.acc(), cnt.bce(), cnt.ba() - rc.ba()));
        per_group.insert(cand.into(), (pg_real, pg_der));
        let cell_json: BTreeMap<String, serde_json::Value> = cells.iter().map(|(k, c)| (k.clone(), json!({"n": c.n, "acc": c.acc(), "mean_bce": c.bce()}))).collect();
        let mean = |v: &Vec<f64>| v.iter().sum::<f64>() / v.len() as f64;
        let sd = |v: &Vec<f64>| { let m = mean(v); (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt() };
        points.insert(cand.into(), json!({
            "prediction_provenance_id": prov["provenance_id"], "predictions_sha256": prov["predictions_sha256"],
            "real": {"balanced_accuracy": cnt.ba(), "accuracy": cnt.acc(), "bce": cnt.bce(), "brier": brier / G1_TOTAL as f64, "auroc": auroc, "confusion": {"tn": cnt.tn, "fp": cnt.fp, "fn": cnt.fnn, "tp": cnt.tp},
                "logit_pos": {"mean": mean(&pos), "sd": sd(&pos)}, "logit_neg": {"mean": mean(&neg), "sd": sd(&neg)}, "mean_abs_logit_on_errors": if err_n > 0.0 { err_conf / err_n } else { f64::NAN }, "mean_abs_logit_on_correct": ok_conf / ok_n.max(1.0), "errors": err_n, "per_cell": cell_json},
            "derangement": {"balanced_accuracy_vs_recipient_labels": rc.ba(), "accuracy_vs_recipient_labels": rc.acc(), "balanced_accuracy_vs_donor_labels": dc.ba(), "accuracy_vs_donor_labels": dc.acc(), "donor_label_agreement_rate": agree as f64 / G1_TOTAL as f64, "max_abs_logit_diff_vs_donor_ordinary": maxdiff, "donor_prediction_tolerance": tol, "donor_predictions_consistent": donor_ok, "drop_pp": (cnt.ba() - rc.ba()) * 100.0},
            "erasure_ood_diagnostic": {"balanced_accuracy": ec.ba(), "accuracy": ec.acc(), "bce": ec.bce(), "predicted_positive_rate": epos / G1_TOTAL as f64},
            "timing": prov["timing"], "candidate_files": prov["candidate_files"], "evaluator_source": prov["evaluator_source"],
        }));
    }
    // paired cluster bootstrap with identical draws
    let mut ci: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    let metric = |c: &Counts, k: &str| -> f64 { match k { "ba" => c.ba(), "acc" => c.acc(), _ => c.bce() } };
    let tot = |v: &Vec<Counts>, d: &Vec<usize>| -> Counts { let mut t = Counts::default(); for &i in d { t.merge(&v[i]); } t };
    let mut samples: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for d in &draws {
        let real: BTreeMap<&str, Counts> = ["A", "M", "B"].iter().map(|c| (*c, tot(&per_group[*c].0, d))).collect();
        let der: BTreeMap<&str, Counts> = ["A", "M", "B"].iter().map(|c| (*c, tot(&per_group[*c].1, d))).collect();
        for c in ["A", "M", "B"] {
            for k in ["ba", "acc", "bce"] {
                samples.entry(format!("{c}.{k}")).or_default().push(metric(&real[c], k));
            }
            samples.entry(format!("{c}.drop_pp")).or_default().push((real[c].ba() - der[c].ba()) * 100.0);
        }
        for (x, y) in [("A", "B"), ("M", "B"), ("A", "M")] {
            for k in ["ba", "acc", "bce"] {
                samples.entry(format!("{x}-{y}.{k}")).or_default().push(metric(&real[x], k) - metric(&real[y], k));
            }
        }
    }
    for (k, v) in samples.iter_mut() {
        let (lo, hi) = (pct(v, 0.025), pct(v, 0.975));
        ci.insert(k.clone(), json!({"lo95": lo, "hi95": hi}));
    }
    // point differences and decision flags
    let pt = |k: &str| -> f64 { let (c, m) = k.split_once('.').unwrap(); let s = summ[c]; match m { "ba" => s.0, "acc" => s.1, "bce" => s.2, _ => s.3 * 100.0 } };
    let diff = |x: &str, y: &str, m: &str| -> f64 { pt(&format!("{x}.{m}")) - pt(&format!("{y}.{m}")) };
    let mut paired = serde_json::Map::new();
    for (x, y) in [("A", "B"), ("M", "B"), ("A", "M")] {
        for k in ["ba", "acc", "bce"] {
            paired.insert(format!("{x}-{y}.{k}"), json!({"point": diff(x, y, k), "ci": ci[&format!("{x}-{y}.{k}")]}));
        }
    }
    let mut decisions = serde_json::Map::new();
    for c in ["A", "M"] {
        let (ba, _acc, bce, drop) = summ[c];
        let transfer = ba >= 0.75 && bce <= 0.55 && drop * 100.0 >= 15.0 && integrity_ok;
        let gain = ba - summ["B"].0;
        let lo = ci[&format!("{c}-B.ba")]["lo95"].as_f64().unwrap();
        let improvement = gain >= 0.05 && lo > 0.0 && bce <= summ["B"].2;
        decisions.insert(c.into(), json!({"transfer_criterion_met": transfer, "ba_ge_75": ba >= 0.75, "bce_le_055": bce <= 0.55, "drop_ge_15pp": drop * 100.0 >= 15.0, "practical_improvement_over_baseline": improvement, "ba_gain_pp": gain * 100.0, "gain_ci_excludes_zero": lo > 0.0, "bce_no_worse_than_baseline": bce <= summ["B"].2}));
    }
    let b_transfers = summ["B"].0 >= 0.75 && summ["B"].2 <= 0.55;
    decisions.insert("B_meets_ba_bce_thresholds".into(), json!(b_transfers));
    let (at, ai) = (decisions["A"]["transfer_criterion_met"].as_bool().unwrap(), decisions["A"]["practical_improvement_over_baseline"].as_bool().unwrap());
    let mt = decisions["M"]["transfer_criterion_met"].as_bool().unwrap();
    let mut notes: Vec<String> = Vec::new();
    if at && ai { notes.push("A transfers and improves on the baseline: recommend a fresh controlled architecture/generalization campaign.".into()); }
    if mt && (diff("M", "A", "ba") >= 0.0 || diff("A", "M", "ba") < 0.02) { notes.push("M transfers with comparable or better quality than A: recommend M as the practical control/candidate (it is far cheaper).".into()); }
    if !at && !mt && b_transfers { notes.push("Neural candidates fail the transfer criterion while the baseline transfers: prioritize representation and generalization, not more fitting.".into()); }
    if !at && !mt && !b_transfers { notes.push("All candidates fail: audit distribution, domain and sampling before proposing more training.".into()); }
    if notes.is_empty() { notes.push("Mixed outcome: see per-candidate criteria; no single recommended path is forced by the pre-declared rules.".into()); }
    // fit-to-G1 gap
    let d3: serde_json::Value = serde_json::from_slice(&a.read(Path::new("d3/report/d3_report.json"))?)?;
    let d1: serde_json::Value = serde_json::from_slice(&a.read(Path::new("d1/report/d1_report.json"))?)?;
    let fit = json!({"A": d3["D3-A"]["snapshots"]["12000"]["bal_acc"], "M": d3["D3-M"]["snapshots"]["12000"]["bal_acc"], "B": d1["baseline"]["fit"]["bal_acc"]});
    let gap: BTreeMap<String, f64> = ["A", "M", "B"].iter().map(|c| (c.to_string(), (fit[*c].as_f64().unwrap() - summ[*c].0) * 100.0)).collect();
    report.insert("candidates".into(), json!(points));
    report.insert("point_summary".into(), json!(summ.iter().map(|(k, v)| (k.clone(), json!({"ba": v.0, "acc": v.1, "bce": v.2, "drop_ba_pp": v.3 * 100.0}))).collect::<BTreeMap<_, _>>()));
    report.insert("bootstrap".into(), json!({"resamples": resamples, "groups": ng, "unit": "connected G1 group_id", "paired": "identical group draws for every candidate and comparison", "intervals": ci}));
    report.insert("paired_differences".into(), serde_json::Value::Object(paired));
    report.insert("fit_balanced_accuracy".into(), fit);
    report.insert("fit_to_g1_gap_pp".into(), json!(gap));
    report.insert("decision_flags".into(), serde_json::Value::Object(decisions));
    report.insert("integrity_ok".into(), json!(integrity_ok));
    report.insert("recommendation_notes".into(), json!(notes));
    report.insert("label".into(), json!("G1 is same-domain generalization on one fresh evaluation-only panel; not an architecture, recurrence, hierarchy or move-selection claim."));
    report.insert("source".into(), serde_json::to_value(source_id())?);
    let text = serde_json::to_string_pretty(&serde_json::Value::Object(report))?;
    a.write(Path::new("g1/report/g1_report.json"), text.as_bytes())?;
    for n in &notes {
        println!("{n}");
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("protocol-freeze") => cmd_protocol_freeze(&args[2..]),
        Some("preserve") => cmd_preserve(&args[2..]),
        Some("generate") => cmd_generate(&args[2..]),
        Some("audit") => cmd_audit(&args[2..]),
        Some("intervention") => cmd_intervention(&args[2..]),
        Some("source") => cmd_source(&args[2..]),
        Some("freeze") => cmd_freeze(&args[2..]),
        Some("aggregate") => cmd_aggregate(&args[2..]),
        _ => bail!("usage: v69-g1 <protocol-freeze|preserve|generate|audit|intervention|source|freeze|aggregate> --artifacts DIR"),
    }
}
