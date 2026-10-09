//! v69-t1: data-side tooling for the T1 representation/generalization campaign (host only).
//!   preserve --label | protocol-freeze | generate | audit | baseline-refit | freeze
//! Every open goes through the role-restricted Access; receipts are append-only.

use anyhow::{Context, Result, bail, ensure};
use cozy_chess::Board;
use recur64_v69::access::{Access, Role};
use recur64_v69::canon::{canonical_key, key_hex, key_id};
use recur64_v69::custody::Custody;
use recur64_v69::d1::{BaselineModel, FrozenD1, N_BASELINE, baseline_features, fit_baseline};
use recur64_v69::d2::write_new;
use recur64_v69::dataset::{Partition, RootOutcome, RootRec, analyze_root};
use recur64_v69::features::{featurize, read_rows, transform_fen};
use recur64_v69::g1::ExclusionIndex;
use recur64_v69::generate::{Family, Reject, sample_root};
use recur64_v69::oracle::{Oracle, Verdict};
use recur64_v69::provenance::{sha256_hex, source_id};
use recur64_v69::reference;
use recur64_v69::streams::{MasterSeed, keyed_u64};
use recur64_v69::t1::*;
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
        Ok(Self { custody: Custody::new(Path::new(&arg(args, "--artifacts").context("--artifacts")?))?, run: PathBuf::from("gen-001"), seed_rel: PathBuf::from("t1/seed/t1_master_seed.hex") })
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

#[derive(Serialize)]
struct Row<'a> {
    id: &'a str,
    fen: &'a str,
    budget: u8,
    label: bool,
}

const FROZEN_PROTOCOL: &str = "t1/frozen_protocol.json";

// ------------------------------------------------------------------ preserve / protocol-freeze

fn cmd_preserve(args: &[String]) -> Result<()> {
    let env = Env::from(args)?;
    let a = env.access(Role::DataAudit)?;
    let label = arg(args, "--label").context("--label")?;
    let mut mismatches = Vec::new();
    let mut n = 0;
    for (mf, what) in [("d1/e1_supplementary_manifest.json", "E1"), ("d2/d1_supplementary_manifest.json", "D1"), ("d3/d2_supplementary_manifest.json", "D2"), ("g1r1/d3_supplementary_manifest.json", "D3"), ("g1r1/g1_attempt1_manifest.json", "G1-attempt1"), ("t1/g1r1_manifest.json", "G1-R1")] {
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
    let rep = json!({"label": label, "ok": ok, "files_checked": n, "mismatches": mismatches, "documented_e1_deviation": "E1 audit receipt was overwritten in place by the E1 post-campaign audit; pre-run copy preserved. T1 never writes to audit/; receipts are append-only.", "source": source_id()});
    write_new(&a, &format!("t1/receipts/preservation_{label}.json"), serde_json::to_string_pretty(&rep)?.as_bytes())?;
    println!("{}", serde_json::to_string_pretty(&rep)?);
    if !ok {
        std::process::exit(4);
    }
    Ok(())
}

fn cmd_protocol_freeze(args: &[String]) -> Result<()> {
    let env = Env::from(args)?;
    let a = env.access(Role::DataAudit)?;
    let amend = args.iter().any(|x| x == "--amend");
    let target = if amend { "t1/frozen_protocol_a1.json" } else { FROZEN_PROTOCOL };
    ensure!(!a.custody().resolve(Path::new(target))?.exists(), "protocol already frozen");
    if !amend {
        ensure!(!a.custody().resolve(&env.seed_rel)?.exists(), "protocol must be frozen BEFORE the T1 seed exists");
    } else {
        ensure!(!a.custody().resolve(Path::new("t1/rows/train.jsonl"))?.exists(), "amendment A1 must be frozen before any T1 rows exist");
    }
    let mut files = vec!["t1/T1_CONTRACT.md", "t1/config.json", "t1/g1r1_manifest.json", "t1/receipts/preservation_start.json", "g1r1/g1_attempt1_manifest.json", "g1r1/d3_supplementary_manifest.json", "d1/e1_supplementary_manifest.json", "d2/d1_supplementary_manifest.json", "d3/d2_supplementary_manifest.json", "d1/baseline/model.json", "g1r1/MANIFEST.sha256.json", "g1r1/report/g1_report.json", "dataset:MANIFEST.sha256.json"];
    if amend {
        files.push("t1/T1_AMENDMENT_A1.md");
        files.push(FROZEN_PROTOCOL);
    }
    let mut g = BTreeMap::new();
    for rel in files {
        let bytes = match rel.strip_prefix("dataset:") {
            Some(d) => a.read(&env.run.join(d))?,
            None => a.read(Path::new(rel)).with_context(|| format!("hash {rel}"))?,
        };
        g.insert(rel.to_string(), sha256_hex(&bytes));
    }
    let mut groups = BTreeMap::new();
    groups.insert("protocol".to_string(), g);
    let fz = FrozenD1 { created_utc: format!("unix:{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs()), run: "gen-001".into(), seed_fingerprint: "(not yet drawn)".into(), groups };
    let text = serde_json::to_string_pretty(&fz)?;
    write_new(&a, target, text.as_bytes())?;
    println!("frozen protocol sha256 {}", sha256_hex(text.as_bytes()));
    Ok(())
}

// ------------------------------------------------------------------ generate

fn load_prior_index(a: &Access, run: &Path) -> Result<ExclusionIndex> {
    let manifest: BTreeMap<String, String> = serde_json::from_slice(&a.read(&run.join("MANIFEST.sha256.json"))?)?;
    let b = a.read(&run.join("pool/roots.jsonl"))?;
    ensure!(manifest.get("pool/roots.jsonl") == Some(&sha256_hex(&b)), "gen-001 pool hash mismatch");
    let mut pool: Vec<RootRec> = jsonl(std::str::from_utf8(&b)?)?;
    let g1: Vec<RootRec> = jsonl(&a.read_to_string(Path::new("g1r1/pool/roots.jsonl"))?)?;
    pool.extend(g1);
    Ok(ExclusionIndex::from_pool(&pool))
}

fn direct_overlap(r: &RootRec, idx: &ExclusionIndex) -> bool {
    idx.roots.binary_search(&r.key).is_ok() || r.children.iter().any(|c| idx.children.binary_search(&c.key).is_ok())
}

fn cmd_generate(args: &[String]) -> Result<()> {
    let t_start = Instant::now();
    let env = Env::from(args)?;
    let a = env.access(Role::T1Builder)?;
    let seed = env.seed(&a)?;
    let time_limit: u64 = arg(args, "--time-limit-secs").map(|s| s.parse()).transpose()?.unwrap_or(3000);
    let node_limit: u64 = 5_000_000;
    let threads: usize = arg(args, "--threads").map(|s| s.parse()).transpose()?.unwrap_or(10);
    let round_size: u64 = arg(args, "--round-size").map(|s| s.parse()).transpose()?.unwrap_or(4000);
    ensure!(!a.custody().resolve(Path::new("t1/rows/train.jsonl"))?.exists(), "T1 data already generated; no regeneration");
    let idx = load_prior_index(&a, &env.run)?;
    let idx_text = serde_json::to_string(&idx)?;
    a.write(Path::new("t1/index/prior_exclusion_index.json"), idx_text.as_bytes())?;
    eprintln!("[t1] seed {} | prior exclusion index (gen-001 + G1 pool): {} roots, {} children", seed.fingerprint(), idx.roots.len(), idx.children.len());
    let deadline = t_start + Duration::from_secs(time_limit);
    let mut stats: BTreeMap<&'static str, Fam> = Family::ALL.iter().map(|f| (f.name(), Fam::default())).collect();
    let mut seen: HashSet<String> = HashSet::new();
    let mut kept: Vec<RootRec> = Vec::new();
    let (mut excluded, mut accepted_total) = (0usize, 0usize);
    let mut done: Option<(Vec<T1Meta>, BTreeMap<String, usize>)> = None;
    let mut status = "wall_limit_reached".to_string();
    let mut rounds = 0u64;
    for round in 0..100_000u64 {
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
            break;
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
                    accepted_total += 1;
                    if direct_overlap(&r, &idx) {
                        excluded += 1;
                    } else {
                        kept.push(*r);
                    }
                }
                RootOutcome::RejectedM1 => s.rejected_m1 += 1,
                RootOutcome::RejectedBeyond => s.rejected_no_mate_within_3 += 1,
                RootOutcome::Unresolved => s.unresolved_node_limit += 1,
            }
        }
        rounds = round + 1;
        if rounds % 3 == 0 {
            let parts = assign_roots(&seed, &kept);
            let (examples, counts) = select_t1(&seed, &kept, &parts);
            let short: usize = counts.iter().map(|(k, c)| quota(if k.starts_with("train/") { Partition::Fit } else if k.starts_with("val/") { Partition::Val } else { Partition::Test }).saturating_sub(*c)).sum();
            eprintln!("[t1] round {round}: accepted {accepted_total}, excluded {excluded}, kept {}, {short} examples short | {:.0}s", kept.len(), t_start.elapsed().as_secs_f64());
            if feasible(&counts) {
                status = "complete_feasible".into();
                done = Some((examples, counts));
                break;
            }
        }
    }
    let Some((mut examples, counts)) = done else {
        let rep = json!({"status": status, "feasible": false, "rounds": rounds, "per_family_stats": stats, "accepted": accepted_total, "excluded": excluded, "wall_secs": t_start.elapsed().as_secs_f64()});
        a.write(Path::new("t1/meta/generation_report_FAILED.json"), serde_json::to_string_pretty(&rep)?.as_bytes())?;
        eprintln!("[t1] GENERATION INFEASIBLE: {rep}");
        std::process::exit(3);
    };
    let group_counts = attach_groups(&seed, &mut examples, &kept);
    // write partitions, nested scales, metadata, contributing pool
    let mut written = BTreeMap::new();
    let mut manifest = BTreeMap::new();
    for p in [Partition::Fit, Partition::Val, Partition::Test] {
        let mut v: Vec<&T1Meta> = examples.iter().filter(|m| m.ex.partition == p).collect();
        v.sort_by(|x, y| x.ex.id.cmp(&y.ex.id));
        let rows: Vec<Row> = v.iter().map(|m| Row { id: &m.ex.id, fen: &m.ex.fen, budget: m.ex.budget, label: m.ex.label }).collect();
        let (tr, tm) = (text_of(&rows), text_of(&v));
        a.write(Path::new(&format!("t1/rows/{}.jsonl", pname(p))), tr.as_bytes())?;
        a.write(Path::new(&format!("t1/meta/{}.meta.jsonl", pname(p))), tm.as_bytes())?;
        manifest.insert(format!("t1/rows/{}.jsonl", pname(p)), sha256_hex(tr.as_bytes()));
        manifest.insert(format!("t1/meta/{}.meta.jsonl", pname(p)), sha256_hex(tm.as_bytes()));
        written.insert(pname(p).to_string(), v.len());
        if p == Partition::Fit {
            for k in SCALES {
                let sub: Vec<&&T1Meta> = v.iter().filter(|m| m.rank < k).collect();
                let rows: Vec<Row> = sub.iter().map(|m| Row { id: &m.ex.id, fen: &m.ex.fen, budget: m.ex.budget, label: m.ex.label }).collect();
                let t = text_of(&rows);
                a.write(Path::new(&format!("t1/rows/train_s{k}.jsonl")), t.as_bytes())?;
                manifest.insert(format!("t1/rows/train_s{k}.jsonl"), sha256_hex(t.as_bytes()));
                written.insert(format!("train_s{k}"), sub.len());
            }
        }
    }
    let used_roots: HashSet<&str> = examples.iter().map(|m| m.ex.root_id.as_str()).collect();
    let pool_used: Vec<&RootRec> = kept.iter().filter(|r| used_roots.contains(r.id.as_str())).collect();
    let tp = text_of(&pool_used);
    a.write(Path::new("t1/pool/roots.jsonl"), tp.as_bytes())?;
    manifest.insert("t1/pool/roots.jsonl".into(), sha256_hex(tp.as_bytes()));
    manifest.insert("t1/index/prior_exclusion_index.json".into(), sha256_hex(idx_text.as_bytes()));
    let mut gp: HashMap<&str, HashSet<&str>> = HashMap::new();
    for m in &examples {
        gp.entry(m.ex.partition.name()).or_default().insert(m.ex.group_id.as_str());
    }
    let report = json!({"status": status, "feasible": true, "rounds": rounds, "round_size_per_family": round_size, "wall_secs": t_start.elapsed().as_secs_f64(), "seed_fingerprint": seed.fingerprint(),
        "per_family_stats": stats, "accepted_roots": accepted_total, "excluded_direct_prior_overlap": excluded, "kept_roots": kept.len(), "prior_index": {"roots": idx.roots.len(), "children": idx.children.len()},
        "groups_among_contributing_roots_by_partition": group_counts,
        "examples_written": written, "per_cell": counts, "distinct_groups_by_partition": gp.iter().map(|(k, v)| (k.to_string(), v.len())).collect::<BTreeMap<_, _>>(), "source": source_id()});
    a.write(Path::new("t1/meta/generation_report.json"), serde_json::to_string_pretty(&report)?.as_bytes())?;
    manifest.insert("t1/meta/generation_report.json".into(), sha256_hex(&a.read(Path::new("t1/meta/generation_report.json"))?));
    a.write(Path::new("t1/MANIFEST.sha256.json"), serde_json::to_string_pretty(&manifest)?.as_bytes())?;
    println!("T1 generation complete: {written:?}; accepted {accepted_total}, excluded {excluded}, kept {}; wall {:.1}s", kept.len(), t_start.elapsed().as_secs_f64());
    Ok(())
}

// ------------------------------------------------------------------ audit

fn cmd_audit(args: &[String]) -> Result<()> {
    let t0 = Instant::now();
    let env = Env::from(args)?;
    let a = env.access(Role::T1Builder)?;
    let seed = env.seed(&a)?;
    let threads: usize = arg(args, "--threads").map(|s| s.parse()).transpose()?.unwrap_or(10);
    let mut failures: Vec<String> = Vec::new();
    macro_rules! check {
        ($n:expr, $ok:expr, $m:expr) => {
            if !$ok {
                failures.push(format!("[{}] {}", $n, $m));
            }
        };
    }
    let man: BTreeMap<String, String> = serde_json::from_slice(&a.read(Path::new("t1/MANIFEST.sha256.json"))?)?;
    for (f, h) in &man {
        check!("manifest", sha256_hex(&a.read(Path::new(f))?) == *h, format!("hash mismatch {f}"));
    }
    ensure!(failures.is_empty(), "manifest mismatch: {failures:?}");
    let mut all_meta: Vec<T1Meta> = Vec::new();
    let mut all_rows = Vec::new();
    for p in ["train", "val", "test"] {
        let rows = read_rows(&a.read_to_string(Path::new(&format!("t1/rows/{p}.jsonl")))?)?;
        let meta: Vec<T1Meta> = jsonl(&a.read_to_string(Path::new(&format!("t1/meta/{p}.meta.jsonl")))?)?;
        check!("rows", rows.len() == meta.len(), format!("{p}: {} rows vs {} meta", rows.len(), meta.len()));
        let mm: HashMap<&str, &T1Meta> = meta.iter().map(|m| (m.ex.id.as_str(), m)).collect();
        check!("rows", mm.len() == meta.len(), format!("{p}: duplicate ids"));
        for r in &rows {
            match mm.get(r.id.as_str()) {
                None => failures.push(format!("[rows] {} without metadata", r.id)),
                Some(m) => {
                    check!("rows", m.ex.fen == r.fen && m.ex.budget == r.budget && m.ex.label == r.label, format!("row/meta differ {}", r.id));
                    match featurize(&r.fen, r.budget) {
                        Ok((_, fam)) => check!("rows", fam.name() == m.ex.family, format!("family differs {}", r.id)),
                        Err(e) => failures.push(format!("[featurize] {}: {e}", r.id)),
                    }
                }
            }
        }
        all_rows.extend(rows);
        all_meta.extend(meta);
    }
    // nested scale files: exactly the rank<k prefix of train, per cell
    let train_ids: HashSet<&str> = all_meta.iter().filter(|m| m.ex.partition == Partition::Fit).map(|m| m.ex.id.as_str()).collect();
    let mut prev: HashSet<String> = HashSet::new();
    for k in SCALES {
        let rows = read_rows(&a.read_to_string(Path::new(&format!("t1/rows/train_s{k}.jsonl")))?)?;
        let ids: HashSet<String> = rows.iter().map(|r| r.id.clone()).collect();
        check!("scales", ids.len() == 12 * k.min(TRAIN_PER_CLASS_CELL) && rows.len() == ids.len(), format!("scale {k}: {} rows", rows.len()));
        check!("scales", ids.iter().all(|i| train_ids.contains(i.as_str())) && prev.is_subset(&ids), format!("scale {k} not nested in train"));
        let want: HashSet<String> = all_meta.iter().filter(|m| m.ex.partition == Partition::Fit && m.rank < k).map(|m| m.ex.id.clone()).collect();
        check!("scales", want == ids, format!("scale {k} is not the rank<k prefix"));
        prev = ids;
    }
    // quotas, caps, identities
    let mut cells: BTreeMap<String, usize> = BTreeMap::new();
    let mut per_root: HashMap<(&str, bool), usize> = HashMap::new();
    for m in &all_meta {
        *cells.entry(format!("{}/{}", pname(m.ex.partition), cell_name(&m.ex.family, m.ex.budget, m.ex.label))).or_default() += 1;
        *per_root.entry((m.ex.root_id.as_str(), m.ex.label)).or_default() += 1;
    }
    check!("quota", feasible(&cells) && cells.iter().all(|(k, c)| *c == quota(if k.starts_with("train/") { Partition::Fit } else if k.starts_with("val/") { Partition::Val } else { Partition::Test })), format!("cells {cells:?}"));
    check!("cap", per_root.values().all(|c| *c <= 2), "per-root cap");
    let pool: Vec<RootRec> = jsonl(&a.read_to_string(Path::new("t1/pool/roots.jsonl"))?)?;
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
    let mut taken: HashSet<String> = HashSet::new();
    for m in &all_meta {
        check!("canon", taken.insert(m.ex.key.clone()), format!("duplicate canonical child {}", m.ex.id));
        let Some(root) = pool_by_id.get(m.ex.root_id.as_str()) else {
            failures.push(format!("[membership] root {} missing", m.ex.root_id));
            continue;
        };
        check!("membership", m.ex.budget + 1 == root.depth && root.depth == m.ex.root_depth && root.fen == m.ex.root_fen, format!("root/budget {}", m.ex.id));
        match root.children.iter().find(|c| c.mv == m.ex.mv) {
            None => failures.push(format!("[membership] {} not a child of its root", m.ex.id)),
            Some(c) => {
                check!("membership", c.fen == m.ex.fen && c.key == m.ex.key, format!("child identity {}", m.ex.id));
                check!("terminal", c.status == "pos" || c.status == "neg", format!("terminal child selected {}", m.ex.id));
                check!("membership", (c.status == "pos") == m.ex.label, format!("stored status vs label {}", m.ex.id));
            }
        }
    }
    // identity exclusion against ALL prior V69 data (gen-001 + G1 pool), full keys
    let idx = load_prior_index(&a, &env.run)?;
    let (rs, cs): (HashSet<&str>, HashSet<&str>) = (idx.roots.iter().map(String::as_str).collect(), idx.children.iter().map(String::as_str).collect());
    let mut overlaps = 0;
    for r in &pool {
        overlaps += (rs.contains(r.key.as_str()) || r.children.iter().any(|c| cs.contains(c.key.as_str()))) as usize;
    }
    overlaps += all_meta.iter().filter(|m| cs.contains(m.ex.key.as_str()) || rs.contains(m.ex.key.as_str())).count();
    check!("exclusion", overlaps == 0, format!("{overlaps} canonical overlaps with prior V69 data"));
    let stored: ExclusionIndex = serde_json::from_slice(&a.read(Path::new("t1/index/prior_exclusion_index.json"))?)?;
    check!("exclusion", stored.roots == idx.roots && stored.children == idx.children, "stored exclusion index differs from the rebuilt one");
    // partition isolation: no group, root or canonical child identity across partitions
    let mut gpart: HashMap<&str, HashSet<Partition>> = HashMap::new();
    let mut rpart: HashMap<&str, HashSet<Partition>> = HashMap::new();
    let mut kpart: HashMap<&str, HashSet<Partition>> = HashMap::new();
    for m in &all_meta {
        gpart.entry(&m.ex.group_id).or_default().insert(m.ex.partition);
        rpart.entry(&m.ex.root_id).or_default().insert(m.ex.partition);
        kpart.entry(&m.ex.key).or_default().insert(m.ex.partition);
    }
    check!("isolation", gpart.values().all(|s| s.len() == 1) && rpart.values().all(|s| s.len() == 1) && kpart.values().all(|s| s.len() == 1), "a group/root/example-child identity spans partitions");
    // groups: rebuilt per partition from the contributing pool and compared (amendment A1)
    {
        let mut rebuilt = all_meta.clone();
        let mut pool_sorted = pool.clone();
        pool_sorted.sort_by(|x, y| x.id.cmp(&y.id));
        attach_groups(&seed, &mut rebuilt, &pool_sorted);
        let rb: HashMap<&str, &str> = rebuilt.iter().map(|m| (m.ex.id.as_str(), m.ex.group_id.as_str())).collect();
        for m in &all_meta {
            check!("groups", rb.get(m.ex.id.as_str()) == Some(&m.ex.group_id.as_str()), format!("group id differs {}", m.ex.id));
        }
    }
    // fresh exact re-query + empty-cache re-analysis
    let next = AtomicUsize::new(0);
    let bad = std::sync::Mutex::new(Vec::<String>::new());
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| {
                let mut o = Oracle::new(5_000_000);
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= all_meta.len() {
                        break;
                    }
                    let m = &all_meta[i].ex;
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
    let next = AtomicUsize::new(0);
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| {
                let mut o = Oracle::new(5_000_000);
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= pool.len() {
                        break;
                    }
                    let r = &pool[i];
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
    // symmetry augmentation: exact label invariance under all 8 board transforms (keyed sample, fresh oracle)
    let mut sym_sample: Vec<&T1Meta> = all_meta.iter().collect();
    sym_sample.sort_by_key(|m| keyed_u64(&seed, "t1/audit/sym", m.ex.id.as_bytes()));
    sym_sample.truncate(1500);
    let next = AtomicUsize::new(0);
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| {
                let mut o = Oracle::new(5_000_000);
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= sym_sample.len() {
                        break;
                    }
                    let m = &sym_sample[i].ex;
                    for t in 1..8 {
                        let fen = transform_fen(&m.fen, t).unwrap();
                        let cb: Board = fen.parse().unwrap();
                        o.clear_cache();
                        let ok = match o.mate_within_child(&cb, !cb.side_to_move(), m.budget) {
                            Verdict::Yes => m.label,
                            Verdict::No => !m.label,
                            Verdict::Unknown => false,
                        };
                        if !ok {
                            bad.lock().unwrap().push(format!("[symmetry] label changes under transform {t}: {}", m.id));
                        }
                    }
                }
            });
        }
    });
    failures.extend(bad.into_inner().unwrap());
    // independent reference sample + bounded M3 minimal-depth sample
    let mut sample: Vec<&T1Meta> = Vec::new();
    for p in [Partition::Fit, Partition::Val, Partition::Test] {
        for f in Family::ALL {
            for b in [1u8, 2] {
                for c in [false, true] {
                    let mut v: Vec<&T1Meta> = all_meta.iter().filter(|m| m.ex.partition == p && m.ex.family == f.name() && m.ex.budget == b && m.ex.label == c).collect();
                    v.sort_by(|x, y| x.ex.id.cmp(&y.ex.id));
                    let step = (v.len() / 6).max(1);
                    sample.extend(v.into_iter().step_by(step).take(6));
                }
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
                let e = &sample[i].ex;
                let b: Board = e.fen.parse().unwrap();
                if reference::child_target(&b, !b.side_to_move(), e.budget) != e.label {
                    rbad.lock().unwrap().push(format!("[reference] disagrees on {}", e.id));
                }
            });
        }
    });
    let mut m3: Vec<&RootRec> = pool.iter().filter(|r| r.depth == 3).collect();
    m3.sort_by(|x, y| x.id.cmp(&y.id));
    let step = (m3.len() / 30).max(1);
    let m3s: Vec<&RootRec> = m3.into_iter().step_by(step).take(30).collect();
    let next = AtomicUsize::new(0);
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= m3s.len() {
                    break;
                }
                let b: Board = m3s[i].fen.parse().unwrap();
                if reference::root_min_depth(&b, b.side_to_move(), 3) != Some(3) {
                    rbad.lock().unwrap().push(format!("[reference-M3] depth differs for root {}", m3s[i].id));
                }
            });
        }
    });
    failures.extend(rbad.into_inner().unwrap());
    let ok = failures.is_empty();
    let rep = json!({"audit_pass": ok, "failures": failures.iter().take(50).collect::<Vec<_>>(), "failure_count": failures.len(), "examples": all_meta.len(), "contributing_roots_reanalysed_empty_cache": pool.len(), "targets_requeried_exact": all_meta.len(),
        "symmetry_label_invariance": {"examples": sym_sample.len(), "transforms_per_example": 7, "oracle_queries": sym_sample.len() * 7}, "reference_child_checks": sample.len(), "reference_m3_minimal_depth_checks": m3s.len(), "prior_overlaps_found": overlaps,
        "prior_index_sizes": {"roots": idx.roots.len(), "children": idx.children.len()}, "reference_shared_dependencies": "cozy-chess move generator/legality, FEN parser, domain definition; NOT the search, cache, node budget or rules::classify code",
        "audit_wall_secs": t0.elapsed().as_secs_f64(), "source": source_id()});
    write_new(&a, "t1/receipts/audit_receipt_t1.json", serde_json::to_string_pretty(&rep)?.as_bytes())?;
    println!("{}", serde_json::to_string_pretty(&rep)?);
    if !ok {
        std::process::exit(4);
    }
    Ok(())
}

// ------------------------------------------------------------------ baseline refit (same 23 features, train partition only)

fn cmd_baseline_refit(args: &[String]) -> Result<()> {
    let env = Env::from(args)?;
    let a = env.access(Role::T1Builder)?;
    ensure!(!a.custody().resolve(Path::new("t1/baseline_refit/model.json"))?.exists(), "baseline already refit");
    let tr = read_rows(&a.read_to_string(Path::new("t1/rows/train.jsonl"))?)?;
    let va = read_rows(&a.read_to_string(Path::new("t1/rows/val.jsonl"))?)?;
    let xf: Vec<[f64; N_BASELINE]> = tr.iter().map(|r| baseline_features(&r.fen, r.budget)).collect::<Result<_>>()?;
    let yf: Vec<bool> = tr.iter().map(|r| r.label).collect();
    let model: BaselineModel = fit_baseline(&xf, &yf);
    let preds: Vec<serde_json::Value> = va.iter().map(|r| json!({"id": r.id, "update": 0, "logit": model.logit(&baseline_features(&r.fen, r.budget).unwrap())})).collect();
    let tp = text_of(&preds);
    a.write(Path::new("t1/baseline_refit/predictions_val.jsonl"), tp.as_bytes())?;
    a.write(Path::new("t1/baseline_refit/model.json"), serde_json::to_string_pretty(&json!({"model": model, "fit_rows": tr.len(), "recipe": "identical to the D1 baseline (23 features, 1000 GD steps, lr 0.05, L2 0.01, standardized on the fitting rows) but fit on the full T1 train partition", "source": source_id()}))?.as_bytes())?;
    println!("baseline refit on {} rows: objective {:.4} -> {:.4}", tr.len(), model.objective_first_step, model.final_train_objective);
    Ok(())
}

fn cmd_freeze_train(args: &[String]) -> Result<()> {
    let env = Env::from(args)?;
    let a = env.access(Role::DataAudit)?;
    let target = "t1/frozen_train.json";
    ensure!(!a.custody().resolve(Path::new(target))?.exists(), "training inputs already frozen");
    let rcpt: serde_json::Value = serde_json::from_slice(&a.read(Path::new("t1/receipts/audit_receipt_t1.json"))?)?;
    ensure!(rcpt["audit_pass"] == json!(true), "T1 audit did not pass; training inputs cannot be frozen");
    let files = ["t1/T1_CONTRACT.md", "t1/config.json", "t1/T1_AMENDMENT_A1.md", "t1/T1_AMENDMENT_A2.md", "t1/frozen_protocol.json", "t1/frozen_protocol_a1.json", "t1/receipts/audit_receipt_t1.json", "t1/MANIFEST.sha256.json", "t1/rows/train.jsonl", "t1/rows/train_s250.jsonl", "t1/rows/train_s1000.jsonl", "t1/rows/train_s2000.jsonl", "t1/rows/val.jsonl", "init/canonical_init.bin", "d1/mlp_init.bin", "t1/seed/t1_master_seed.hex"];
    let mut g = BTreeMap::new();
    for rel in files {
        g.insert(rel.to_string(), sha256_hex(&a.read(Path::new(rel)).with_context(|| format!("hash {rel}"))?));
    }
    let mut groups = BTreeMap::new();
    groups.insert("learner".to_string(), g);
    let seed = env.seed(&a)?;
    let fz = FrozenD1 { created_utc: format!("unix:{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs()), run: "gen-001".into(), seed_fingerprint: seed.fingerprint(), groups };
    let text = serde_json::to_string_pretty(&fz)?;
    write_new(&a, target, text.as_bytes())?;
    println!("frozen training inputs sha256 {}", sha256_hex(text.as_bytes()));
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("preserve") => cmd_preserve(&args[2..]),
        Some("protocol-freeze") => cmd_protocol_freeze(&args[2..]),
        Some("generate") => cmd_generate(&args[2..]),
        Some("audit") => cmd_audit(&args[2..]),
        Some("baseline-refit") => cmd_baseline_refit(&args[2..]),
        Some("freeze-train") => cmd_freeze_train(&args[2..]),
        _ => bail!("usage: v69-t1 <preserve|protocol-freeze|generate|audit|baseline-refit> --artifacts DIR"),
    }
}
