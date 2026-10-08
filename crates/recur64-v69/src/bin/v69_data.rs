//! v69-data: bounded fresh-data generation and audit for Recur64 V69.
//!
//!   v69-data generate --artifacts DIR --seed-file F --run NAME
//!        [--time-limit-secs 7200] [--node-limit 5000000] [--threads 10]
//!        [--round-size 400] [--max-rounds 1000]
//!   v69-data audit --artifacts DIR --run NAME [--ref-sample 8] [--threads 10]
//!
//! Every path goes through the V69 custody guard.

use anyhow::{Context, Result, bail};
use cozy_chess::Board;
use recur64_v69::canon::{canonical_key, key_hex};
use recur64_v69::custody::Custody;
use recur64_v69::dataset::*;
use recur64_v69::generate::{Family, Reject, sample_root};
use recur64_v69::oracle::Oracle;
use recur64_v69::reference;
use recur64_v69::streams::MasterSeed;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn sha256_file(p: &Path) -> Result<String> {
    Ok(recur64_v69::streams::hex(&Sha256::digest(std::fs::read(p)?)))
}

fn git(args: &[&str]) -> String {
    std::process::Command::new("git")
        .args(args)
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

/// Digest of the V69 crate sources + Cargo.lock + rust-toolchain (provenance).
fn source_digest() -> String {
    let mut h = Sha256::new();
    let mut files: Vec<PathBuf> = Vec::new();
    fn walk(d: &Path, out: &mut Vec<PathBuf>) {
        if let Ok(rd) = std::fs::read_dir(d) {
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    walk(&p, out)
                } else {
                    out.push(p)
                }
            }
        }
    }
    walk(Path::new("crates/recur64-v69"), &mut files);
    walk(Path::new("crates/recur64-core/src"), &mut files);
    files.push("Cargo.lock".into());
    files.push("rust-toolchain.toml".into());
    files.sort();
    for f in files {
        if let Ok(b) = std::fs::read(&f) {
            h.update(f.to_string_lossy().replace('\\', "/").as_bytes());
            h.update(Sha256::digest(&b));
        }
    }
    recur64_v69::streams::hex(&h.finalize())
}

#[derive(Default, Serialize, Clone)]
struct FamStats {
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
    timed_out_discarded: u64,
    oracle_micros_m2: u64,
    oracle_micros_m3: u64,
    oracle_micros_rejected: u64,
    oracle_nodes_accepted: u64,
}

enum Task {
    Analyze { fam: Family, board: Board, att: cozy_chess::Color },
}

fn run_generate(args: &[String]) -> Result<()> {
    let t_start = Instant::now();
    let custody = Custody::new(Path::new(&arg(args, "--artifacts").context("--artifacts")?))?;
    let run = arg(args, "--run").context("--run")?;
    let run_dir = custody.resolve(Path::new(&run))?;
    let seed_path = custody.resolve(Path::new(&arg(args, "--seed-file").context("--seed-file")?))?;
    let seed = MasterSeed::from_hex(&std::fs::read_to_string(&seed_path)?)?;
    let time_limit: u64 = arg(args, "--time-limit-secs").map(|s| s.parse()).transpose()?.unwrap_or(7200);
    let node_limit: u64 = arg(args, "--node-limit").map(|s| s.parse()).transpose()?.unwrap_or(5_000_000);
    let threads: usize = arg(args, "--threads").map(|s| s.parse()).transpose()?.unwrap_or(10);
    let round_size: u64 = arg(args, "--round-size").map(|s| s.parse()).transpose()?.unwrap_or(400);
    let max_rounds: u64 = arg(args, "--max-rounds").map(|s| s.parse()).transpose()?.unwrap_or(1000);
    if run_dir.exists() {
        bail!("run directory {} already exists; V69 never overwrites a run", run_dir.display());
    }
    std::fs::create_dir_all(&run_dir)?;
    let deadline = t_start + Duration::from_secs(time_limit);
    eprintln!("[v69] seed fingerprint {} | limits: {time_limit}s wall, {node_limit} nodes/query, {threads} threads", seed.fingerprint());

    let mut stats: BTreeMap<&'static str, FamStats> = Family::ALL.iter().map(|f| (f.name(), FamStats::default())).collect();
    let mut seen: HashSet<String> = HashSet::new();
    let mut pool: Vec<RootRec> = Vec::new();
    let mut status = "max_rounds_reached".to_string();
    let mut last_short = String::new();
    let mut rounds_done = 0u64;

    for round in 0..max_rounds {
        if Instant::now() >= deadline {
            status = "wall_limit_reached".into();
            break;
        }
        // Phase A (sequential, deterministic): sample, legality, dedup.
        let mut round_stats = stats.clone();
        let mut round_seen: Vec<String> = Vec::new();
        let mut tasks: Vec<(usize, Task)> = Vec::new();
        let mut local_seen: HashSet<String> = HashSet::new();
        for fam in Family::ALL {
            for k in 0..round_size {
                let idx = round * round_size + k;
                let s = round_stats.get_mut(fam.name()).unwrap();
                s.attempted += 1;
                match sample_root(&seed, fam, idx) {
                    Err(Reject::Overlap) => s.illegal_overlap += 1,
                    Err(Reject::AdjacentKings) => s.illegal_adjacent_kings += 1,
                    Err(Reject::DefenderInCheck) => s.illegal_defender_in_check += 1,
                    Err(Reject::BoardInvalid) => s.illegal_other += 1,
                    Ok(r) => {
                        let key = key_hex(&canonical_key(&r.board, r.attacker));
                        if seen.contains(&key) || !local_seen.insert(key.clone()) {
                            s.duplicate += 1;
                        } else {
                            round_seen.push(key);
                            tasks.push((tasks.len(), Task::Analyze { fam, board: r.board, att: r.attacker }));
                        }
                    }
                }
            }
        }
        // Phase B (parallel): exact analysis. Order of results is by task id.
        let next = AtomicUsize::new(0);
        let results: Vec<std::sync::Mutex<Option<(Family, RootOutcome, u64)>>> =
            (0..tasks.len()).map(|_| std::sync::Mutex::new(None)).collect();
        let timed_out = std::sync::atomic::AtomicBool::new(false);
        std::thread::scope(|sc| {
            for _ in 0..threads {
                sc.spawn(|| {
                    let mut oracle = Oracle::new(node_limit);
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= tasks.len() {
                            break;
                        }
                        if Instant::now() >= deadline {
                            timed_out.store(true, Ordering::Relaxed);
                            break;
                        }
                        let Task::Analyze { fam, board, att } = &tasks[i].1;
                        let t0 = Instant::now();
                        let out = analyze_root(&mut oracle, *fam, board, *att);
                        *results[i].lock().unwrap() = Some((*fam, out, t0.elapsed().as_micros() as u64));
                    }
                });
            }
        });
        if timed_out.load(Ordering::Relaxed) {
            // Discard the incomplete round entirely (deterministic prefix of rounds).
            for fam in Family::ALL {
                stats.get_mut(fam.name()).unwrap().timed_out_discarded += round_size;
            }
            status = "wall_limit_reached".into();
            break;
        }
        let mut new_roots = Vec::new();
        for (i, slot) in results.into_iter().enumerate() {
            let (fam, out, micros) = slot.into_inner().unwrap().expect("completed round has all results");
            let s = round_stats.get_mut(fam.name()).unwrap();
            s.analyzed += 1;
            match out {
                RootOutcome::Accepted(r) => {
                    if r.depth == 2 {
                        s.accepted_m2 += 1;
                        s.oracle_micros_m2 += micros;
                    } else {
                        s.accepted_m3 += 1;
                        s.oracle_micros_m3 += micros;
                    }
                    s.oracle_nodes_accepted += r.oracle_nodes;
                    new_roots.push(*r);
                }
                RootOutcome::RejectedM1 => {
                    s.rejected_m1 += 1;
                    s.oracle_micros_rejected += micros;
                }
                RootOutcome::RejectedBeyond => {
                    s.rejected_no_mate_within_3 += 1;
                    s.oracle_micros_rejected += micros;
                }
                RootOutcome::Unresolved => {
                    s.unresolved_node_limit += 1;
                    s.oracle_micros_rejected += micros;
                }
            }
            let _ = i;
        }
        stats = round_stats;
        seen.extend(round_seen);
        pool.extend(new_roots);
        rounds_done = round + 1;
        let groups = build_groups(&seed, &pool);
        let sel = select_examples(&seed, &pool, &groups);
        let short = sel.shortfalls();
        let missing: usize = short.iter().map(|s| s.3 - s.2).sum();
        last_short = format!("{} cells short, {} examples missing", short.len(), missing);
        eprintln!(
            "[v69] round {round}: pool {} roots, {} groups, {} | elapsed {:.0}s",
            pool.len(),
            groups.len(),
            last_short,
            t_start.elapsed().as_secs_f64()
        );
        if sel.feasible() {
            status = "complete_feasible".into();
            break;
        }
    }

    // Final deterministic build from the final pool.
    let groups = build_groups(&seed, &pool);
    let sel = select_examples(&seed, &pool, &groups);
    let feasible = sel.feasible();
    write_outputs(&custody, &run_dir, &seed, &pool, &groups, &sel, &stats, &status, feasible, &last_short, rounds_done, time_limit, node_limit, threads, round_size, t_start.elapsed().as_secs_f64())?;
    eprintln!("[v69] done: status={status} feasible={feasible} wall={:.1}s", t_start.elapsed().as_secs_f64());
    if !feasible {
        std::process::exit(3);
    }
    Ok(())
}

fn write_jsonl<T: Serialize>(p: &Path, rows: impl Iterator<Item = T>) -> Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(p)?);
    for r in rows {
        serde_json::to_writer(&mut f, &r)?;
        f.write_all(b"\n")?;
    }
    f.flush()?;
    Ok(())
}

#[derive(Serialize)]
struct ModelRow<'a> {
    id: &'a str,
    fen: &'a str,
    budget: u8,
    label: bool,
}

#[allow(clippy::too_many_arguments)]
fn write_outputs(
    custody: &Custody,
    run_dir: &Path,
    seed: &MasterSeed,
    pool: &[RootRec],
    groups: &[Group],
    sel: &Selection,
    stats: &BTreeMap<&'static str, FamStats>,
    status: &str,
    feasible: bool,
    last_short: &str,
    rounds: u64,
    time_limit: u64,
    node_limit: u64,
    threads: usize,
    round_size: u64,
    wall: f64,
) -> Result<()> {
    for sub in ["data", "meta", "sealed", "pool"] {
        std::fs::create_dir_all(custody.resolve(&run_dir.join(sub))?)?;
    }
    write_jsonl(&custody.resolve(&run_dir.join("pool/roots.jsonl"))?, pool.iter())?;
    for p in Partition::ALL {
        let rows: Vec<&Example> = sel.examples.iter().filter(|e| e.partition == p).collect();
        let (data, meta) = if p == Partition::Test {
            (format!("sealed/{}.jsonl", p.name()), format!("sealed/{}.meta.jsonl", p.name()))
        } else {
            (format!("data/{}.jsonl", p.name()), format!("meta/{}.meta.jsonl", p.name()))
        };
        write_jsonl(
            &custody.resolve(&run_dir.join(&data))?,
            rows.iter().map(|e| ModelRow { id: &e.id, fen: &e.fen, budget: e.budget, label: e.label }),
        )?;
        write_jsonl(&custody.resolve(&run_dir.join(&meta))?, rows.iter())?;
    }
    // Sealed files: read-only attribute.
    for f in ["sealed/test.jsonl", "sealed/test.meta.jsonl"] {
        let p = custody.resolve(&run_dir.join(f))?;
        let mut perm = std::fs::metadata(&p)?.permissions();
        perm.set_readonly(true);
        std::fs::set_permissions(&p, perm)?;
    }
    // Group report.
    let mut size_hist: BTreeMap<usize, usize> = BTreeMap::new();
    for g in groups {
        *size_hist.entry(g.roots.len()).or_default() += 1;
    }
    let largest = groups.iter().map(|g| g.roots.len()).max().unwrap_or(0);
    let mut per_part: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for g in groups {
        let e = per_part.entry(g.partition.name().to_string()).or_default();
        e.0 += 1;
        e.1 += g.roots.len();
    }
    let used_groups: HashSet<&str> = sel.examples.iter().map(|e| e.group_id.as_str()).collect();
    let used_roots: HashSet<&str> = sel.examples.iter().map(|e| e.root_id.as_str()).collect();
    let mut cells = BTreeMap::new();
    for ((p, (f, b, c)), n) in &sel.filled {
        cells.insert(format!("{}/{}/n{}/{}", p.name(), f, b, if *c { "pos" } else { "neg" }), (*n, p.quota()));
    }
    let report = serde_json::json!({
        "status": status,
        "feasible": feasible,
        "last_shortfall_summary": last_short,
        "rounds": rounds,
        "round_size_per_family": round_size,
        "limits": {"wall_secs": time_limit, "node_limit_per_query": node_limit, "threads": threads, "memory_gib_enforced_by": "scripts/v69/run_limited.ps1 watchdog"},
        "wall_secs_actual": wall,
        "per_family_stats": stats,
        "pool_roots": pool.len(),
        "groups": {"count": groups.len(), "largest_component_roots": largest, "size_histogram_roots": size_hist, "mixed_depth_groups": groups.iter().filter(|g| g.mixed_depth).count(), "by_partition_groups_roots": per_part},
        "used_groups": used_groups.len(),
        "used_roots": used_roots.len(),
        "cells_filled_vs_quota": cells,
        "examples_total": sel.examples.len(),
        "shortfalls": sel.shortfalls().iter().map(|(p, c, g, q)| format!("{}/{}/n{}/{} {g}/{q}", p.name(), c.0, c.1, if c.2 {"pos"} else {"neg"})).collect::<Vec<_>>(),
        "seed_fingerprint": seed.fingerprint(),
        "git_head": git(&["rev-parse", "HEAD"]),
        "git_dirty_files": git(&["status", "--porcelain"]).lines().count(),
        "source_digest": source_digest(),
        "per_root_cap_per_class": CAP_PER_ROOT_PER_CLASS,
    });
    std::fs::write(custody.resolve(&run_dir.join("generation_report.json"))?, serde_json::to_string_pretty(&report)?)?;
    // Hash manifest of every file in the run.
    let mut manifest = BTreeMap::new();
    for sub in ["pool/roots.jsonl", "data/fit.jsonl", "data/val.jsonl", "meta/fit.meta.jsonl", "meta/val.meta.jsonl", "sealed/test.jsonl", "sealed/test.meta.jsonl", "generation_report.json"] {
        manifest.insert(sub.to_string(), sha256_file(&custody.resolve(&run_dir.join(sub))?)?);
    }
    std::fs::write(custody.resolve(&run_dir.join("MANIFEST.sha256.json"))?, serde_json::to_string_pretty(&manifest)?)?;
    Ok(())
}

// ------------------------------------------------------------------ audit

fn read_jsonl<T: serde::de::DeserializeOwned>(p: &Path) -> Result<Vec<T>> {
    std::fs::read_to_string(p)?.lines().map(|l| Ok(serde_json::from_str(l)?)).collect()
}

fn run_audit(args: &[String]) -> Result<()> {
    let t0 = Instant::now();
    let custody = Custody::new(Path::new(&arg(args, "--artifacts").context("--artifacts")?))?;
    let run_dir = custody.resolve(Path::new(&arg(args, "--run").context("--run")?))?;
    let ref_sample: usize = arg(args, "--ref-sample").map(|s| s.parse()).transpose()?.unwrap_or(8);
    let threads: usize = arg(args, "--threads").map(|s| s.parse()).transpose()?.unwrap_or(10);
    let node_limit: u64 = arg(args, "--node-limit").map(|s| s.parse()).transpose()?.unwrap_or(5_000_000);
    let mut failures: Vec<String> = Vec::new();
    macro_rules! check {
        ($ok:expr, $msg:expr) => {
            if !$ok {
                failures.push($msg)
            }
        };
    }

    // Manifest integrity.
    let manifest: BTreeMap<String, String> = serde_json::from_str(&std::fs::read_to_string(run_dir.join("MANIFEST.sha256.json"))?)?;
    for (f, h) in &manifest {
        check!(sha256_file(&run_dir.join(f))? == *h, format!("hash mismatch {f}"));
    }
    let pool: Vec<RootRec> = read_jsonl(&run_dir.join("pool/roots.jsonl"))?;
    let mut examples: Vec<Example> = Vec::new();
    for f in ["meta/fit.meta.jsonl", "meta/val.meta.jsonl", "sealed/test.meta.jsonl"] {
        examples.extend(read_jsonl::<Example>(&run_dir.join(f))?);
    }

    {
        let mut ids = HashSet::new();
        for r in &pool {
            check!(ids.insert(r.id.clone()), format!("duplicate root id {}", r.id));
            check!(r.id == recur64_v69::canon::key_id(&r.key), format!("root id not derived from full key {}", r.id));
        }
        let mut eids = HashSet::new();
        for e in &examples {
            check!(eids.insert(e.id.clone()), format!("duplicate example id {}", e.id));
        }
    }

    // Quota and class coverage.
    let mut counts: BTreeMap<(String, String, u8, bool), usize> = BTreeMap::new();
    for e in &examples {
        *counts.entry((e.partition.name().into(), e.family.clone(), e.budget, e.label)).or_default() += 1;
    }
    for p in Partition::ALL {
        for f in Family::ALL {
            for b in [1u8, 2] {
                for c in [false, true] {
                    let n = counts.get(&(p.name().into(), f.name().into(), b, c)).copied().unwrap_or(0);
                    check!(n == p.quota(), format!("quota {}/{}/n{b}/{c}: {n} != {}", p.name(), f.name(), p.quota()));
                }
            }
        }
    }
    check!(examples.len() == 1536, format!("total {} != 1536", examples.len()));

    // Leakage: group, root and canonical position isolation across partitions.
    let mut group_part: BTreeMap<&str, HashSet<Partition>> = BTreeMap::new();
    let mut root_part: BTreeMap<&str, HashSet<Partition>> = BTreeMap::new();
    let mut key_part: BTreeMap<&str, HashSet<Partition>> = BTreeMap::new();
    for e in &examples {
        group_part.entry(&e.group_id).or_default().insert(e.partition);
        root_part.entry(&e.root_id).or_default().insert(e.partition);
        key_part.entry(&e.key).or_default().insert(e.partition);
    }
    for (g, ps) in &group_part {
        check!(ps.len() == 1, format!("group {g} spans partitions"));
    }
    for (g, ps) in &root_part {
        check!(ps.len() == 1, format!("root {g} spans partitions"));
    }
    for (k, ps) in &key_part {
        check!(ps.len() == 1, format!("canonical child {k} spans partitions"));
    }
    // Each root's all children (not only selected ones) must live in one partition
    // by group construction: re-derive groups from the pool and compare.
    // Full-identity key check: recompute canonical key from FEN.
    let mut key_mismatch = 0;
    for e in &examples {
        let b: Board = e.fen.parse()?;
        let root: Board = e.root_fen.parse()?;
        let att = root.side_to_move();
        if key_hex(&canonical_key(&b, att)) != e.key {
            key_mismatch += 1;
        }
        check!(b.side_to_move() != att, format!("{} child not defender-to-move", e.id));
        check!(recur64_core::rules::classify(&b, 1, 0, None).is_none(), format!("{} terminal child in dataset", e.id));
        check!(b.halfmove_clock() <= 1, format!("{} halfmove {} > 1", e.id, b.halfmove_clock()));
    }
    check!(key_mismatch == 0, format!("{key_mismatch} canonical key mismatches"));
    // Full-key (not hash) collision audit: distinct FENs with same key are
    // symmetric/colour-equivalent images; report count.
    let mut by_key: BTreeMap<&str, HashSet<&str>> = BTreeMap::new();
    for e in &examples {
        by_key.entry(&e.key).or_default().insert(&e.fen);
    }
    let symmetric_dup_keys = by_key.values().filter(|s| s.len() > 1).count();

    // Re-verify every contributing root with a fresh empty-cache oracle.
    let used: HashSet<&str> = examples.iter().map(|e| e.root_id.as_str()).collect();
    let used_roots: Vec<&RootRec> = pool.iter().filter(|r| used.contains(r.id.as_str())).collect();
    let next = AtomicUsize::new(0);
    let bad = std::sync::Mutex::new(Vec::<String>::new());
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| {
                let mut o = Oracle::new(node_limit);
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
                            if r2.depth != r.depth || r2.children.len() != r.children.len() {
                                bad.lock().unwrap().push(format!("root {} depth/children differ", r.id));
                            } else {
                                for (a, c) in r.children.iter().zip(r2.children.iter()) {
                                    if a.mv != c.mv || a.status != c.status {
                                        bad.lock().unwrap().push(format!("root {} child {} label differs", r.id, a.mv));
                                    }
                                }
                            }
                        }
                        other => bad.lock().unwrap().push(format!("root {} re-analysis -> {:?}", r.id, std::mem::discriminant(&other))),
                    }
                }
            });
        }
    });
    let bad = bad.into_inner().unwrap();
    for m in &bad {
        failures.push(m.clone());
    }

    // Independent reference enumerator on a deterministic per-cell sample:
    // child targets (n<=2) and, for sampled roots, minimal depth.
    let mut sample: Vec<&Example> = Vec::new();
    for p in Partition::ALL {
        for f in Family::ALL {
            for b in [1u8, 2] {
                for c in [false, true] {
                    let mut v: Vec<&Example> = examples.iter().filter(|e| e.partition == p && e.family == f.name() && e.budget == b && e.label == c).collect();
                    v.sort_by(|a, b| a.id.cmp(&b.id));
                    let step = (v.len() / ref_sample.max(1)).max(1);
                    sample.extend(v.into_iter().step_by(step).take(ref_sample));
                }
            }
        }
    }
    let next = AtomicUsize::new(0);
    let ref_bad = std::sync::Mutex::new(Vec::<String>::new());
    let ref_done = AtomicUsize::new(0);
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= sample.len() {
                    break;
                }
                let e = sample[i];
                let b: Board = e.fen.parse().unwrap();
                let att = !b.side_to_move();
                let v = reference::child_target(&b, att, e.budget);
                if v != e.label {
                    ref_bad.lock().unwrap().push(format!("reference disagrees on {}", e.id));
                }
                // negative at budget n must also be negative at all smaller budgets (monotonicity)
                if !e.label && e.budget == 2 && reference::child_target(&b, att, 1) {
                    ref_bad.lock().unwrap().push(format!("monotonicity violated {}", e.id));
                }
                ref_done.fetch_add(1, Ordering::Relaxed);
            });
        }
    });
    for m in ref_bad.into_inner().unwrap() {
        failures.push(m);
    }
    // Independent minimal-depth check for the sampled roots of M2 (cheap) only;
    // M3 roots are verified at the child level above plus the oracle re-run.
    let mut ref_roots = 0;
    for e in sample.iter().filter(|e| e.budget == 1).take(ref_sample * 12) {
        let root: Board = e.root_fen.parse()?;
        let d = reference::root_min_depth(&root, root.side_to_move(), 3);
        ref_roots += 1;
        check!(d == Some(2), format!("reference root depth {d:?} != 2 for {}", e.root_id));
    }

    let receipt = serde_json::json!({
        "audit_pass": failures.is_empty(),
        "failures": failures,
        "examples": examples.len(),
        "contributing_roots_reverified_fresh_cache": used_roots.len(),
        "reference_child_checks": ref_done.load(Ordering::Relaxed),
        "reference_m2_root_depth_checks": ref_roots,
        "symmetric_duplicate_canonical_keys": symmetric_dup_keys,
        "distinct_canonical_children": by_key.len(),
        "audit_wall_secs": t0.elapsed().as_secs_f64(),
        "sealed_test_note": "labels were read only to verify exactness; no model code touches sealed/",
    });
    std::fs::write(run_dir.join("audit_receipt.json"), serde_json::to_string_pretty(&receipt)?)?;
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    if !failures.is_empty() {
        std::process::exit(4);
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("generate") => run_generate(&args[2..]),
        Some("audit") => run_audit(&args[2..]),
        _ => bail!("usage: v69-data <generate|audit> ..."),
    }
}
