//! `recur64 v3-p4 ...`: the P4 data / ProofTrace / feasibility tooling.
//!
//! * `custody`: verify P25_DATA_V1 TRAIN and the HOLDOUT_C seal; inventory the
//!   exclusion datasets. Reads HOLDOUT_C for integrity only.
//! * `tune-gen`: generate, audit and verify `v3_tune_v1`.
//! * `trace-gen`: generate ProofTraceV1 shards (resumable).
//! * `trace-audit`: independent audit of every trace (resumable).
//! * `determinism`: regenerate sample shards on 1 and N threads and compare.
//! * `feasibility`: the frozen feasibility table from complete, audited traces.
//!
//! Nothing here evaluates a model, and no command traces or analyses HOLDOUT_C.

use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::{Args, Subcommand};

use recur64_runtime::proof::audit::{audit_targets, check_disjoint};
use recur64_runtime::proof::custody::{
    V3_TUNE_IDENTITY, V3_TUNE_PER_CELL, V3_TUNE_SEED, cell_counts, inventory, load_working_split,
    verify_holdout_c,
};
use recur64_runtime::proof::generator::{V3TuneSpec, generate_v3_tune};
use recur64_runtime::proof::targets::{ProofTargets, Split};
use recur64_runtime::proof::trace_store::{
    AuditManifest, SHARD_SIZE, TraceManifest, audit_traces, feasibility, generate_shard,
    generate_traces,
};

/// Stack for the worker thread (deep recursion in exhaustive search).
const STACK_BYTES: usize = 256 * 1024 * 1024;

#[derive(Subcommand, Debug)]
pub enum P4Cmd {
    /// Verify P25_DATA_V1 TRAIN, seal HOLDOUT_C (custody only) and inventory exclusions.
    Custody(CustodyArgs),
    /// Generate, audit and verify v3_tune_v1.
    TuneGen(TuneArgs),
    /// Generate ProofTraceV1 shards for a TRAIN or v3_tune_v1 dataset.
    TraceGen(TraceArgs),
    /// Independent audit of every trace shard.
    TraceAudit(TraceArgs),
    /// Regenerate sample shards on 1 and N threads and compare with the stored ones.
    Determinism(DetArgs),
    /// The frozen feasibility table from complete, audited traces.
    Feasibility(FeasArgs),
}

#[derive(Args, Debug)]
pub struct CustodyArgs {
    /// P25_DATA_V1 TRAIN (proof-train.json).
    #[arg(long)]
    pub train: PathBuf,
    /// HOLDOUT_C file (integrity check only).
    #[arg(long)]
    pub holdout_c: PathBuf,
    /// Comma-separated directories whose proof-*.json datasets are inventoried.
    #[arg(long)]
    pub inventory_dirs: String,
    #[arg(long)]
    pub output: PathBuf,
    #[arg(long)]
    pub seal_output: PathBuf,
    #[arg(long, default_value_t = 20)]
    pub threads: usize,
}

#[derive(Args, Debug)]
pub struct TuneArgs {
    #[arg(long)]
    pub inventory_dirs: String,
    /// Directory for proof-v3-tune-v1.json (git-ignored run storage).
    #[arg(long)]
    pub data_dir: PathBuf,
    #[arg(long)]
    pub manifest: PathBuf,
    #[arg(long, default_value_t = 20)]
    pub threads: usize,
    /// Skip the second, independent regeneration check.
    #[arg(long)]
    pub no_regeneration_check: bool,
}

#[derive(Args, Debug)]
pub struct TraceArgs {
    /// The dataset to trace (TRAIN or v3_tune_v1). Holdouts are refused.
    #[arg(long)]
    pub data: PathBuf,
    /// Trace directory (shards and manifests).
    #[arg(long)]
    pub dir: PathBuf,
    #[arg(long, default_value_t = 20)]
    pub threads: usize,
    #[arg(long, default_value_t = SHARD_SIZE)]
    pub shard_size: usize,
    /// Write a compact evidence summary here.
    #[arg(long)]
    pub evidence: Option<PathBuf>,
}

#[derive(Args, Debug)]
pub struct DetArgs {
    #[arg(long)]
    pub data: PathBuf,
    #[arg(long)]
    pub dir: PathBuf,
    /// Shards to regenerate (comma-separated indices).
    #[arg(long)]
    pub shards: String,
    #[arg(long, default_value_t = 20)]
    pub threads: usize,
    #[arg(long)]
    pub evidence: Option<PathBuf>,
}

#[derive(Args, Debug)]
pub struct FeasArgs {
    #[arg(long)]
    pub data: PathBuf,
    #[arg(long)]
    pub dir: PathBuf,
    /// "primary" (the frozen gate; TRAIN only) or "diagnostic".
    #[arg(long)]
    pub role: String,
    #[arg(long)]
    pub output: PathBuf,
}

fn dirs(s: &str) -> Vec<PathBuf> {
    s.split(',').map(|d| PathBuf::from(d.trim())).collect()
}

fn write_json(path: &Path, v: &serde_json::Value) -> anyhow::Result<()> {
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d)?;
    }
    std::fs::write(path, serde_json::to_vec_pretty(v)?)?;
    Ok(())
}

pub fn run(cmd: P4Cmd) -> anyhow::Result<()> {
    // Exhaustive search recurses; run on a thread with an explicit large stack.
    std::thread::Builder::new()
        .name("v3-p4".into())
        .stack_size(STACK_BYTES)
        .spawn(move || match cmd {
            P4Cmd::Custody(a) => custody(a),
            P4Cmd::TuneGen(a) => tune_gen(a),
            P4Cmd::TraceGen(a) => trace_gen(a),
            P4Cmd::TraceAudit(a) => trace_audit(a),
            P4Cmd::Determinism(a) => determinism(a),
            P4Cmd::Feasibility(a) => feasibility_cmd(a),
        })?
        .join()
        .map_err(|_| anyhow::anyhow!("v3-p4 thread panicked"))?
}

fn custody(a: CustodyArgs) -> anyhow::Result<()> {
    let t0 = Instant::now();
    // P25_DATA_V1 TRAIN: through ProofTargets::load (schema, contracts, digest).
    let train = load_working_split(&a.train, &[Split::Train])?;
    let cells = cell_counts(&train);
    let audit = audit_targets(&train, a.threads);
    anyhow::ensure!(
        audit.failures.is_empty(),
        "STOP: the independent label audit of P25_DATA_V1 TRAIN failed on {} of {}; first: {}",
        audit.failures.len(),
        audit.checked,
        audit.failures[0]
    );
    let mut seen_fen = std::collections::HashSet::new();
    let mut seen_canon = std::collections::HashSet::new();
    for p in &train.positions {
        anyhow::ensure!(
            seen_fen.insert(p.fen.as_str()),
            "{}: duplicate exact FEN in TRAIN",
            p.id
        );
        anyhow::ensure!(
            seen_canon.insert(p.canon.as_str()),
            "{}: duplicate canonical class in TRAIN",
            p.id
        );
        anyhow::ensure!(
            recur64_runtime::proof::generator::canonical_key(&p.fen) == p.canon,
            "{}: stored canonical key differs from the recomputed one",
            p.id
        );
    }

    // HOLDOUT_C: integrity only.
    let seal = verify_holdout_c(&a.holdout_c)?;

    // The inventory also reads HOLDOUT_C's canonical classes, for exclusion only.
    let inv = inventory(&dirs(&a.inventory_dirs))?;
    anyhow::ensure!(
        train
            .positions
            .iter()
            .all(|p| p.canon.len() == 64 && !p.id.is_empty()),
        "malformed TRAIN position"
    );
    let c_digest_in_inventory = inv.datasets.iter().any(|d| d.digest == seal.actual_digest);
    anyhow::ensure!(
        c_digest_in_inventory,
        "HOLDOUT_C is not part of the exclusion inventory"
    );

    // TRAIN is not itself part of the exclusion check against HOLDOUT_C by the
    // generator, so verify the disjointness the V2.5 record claims.
    let c = recur64_runtime::proof::targets::ProofTargets::load(&a.holdout_c)?;
    let c_canons: std::collections::HashSet<&str> =
        c.positions.iter().map(|p| p.canon.as_str()).collect();
    let c_fens: std::collections::HashSet<&str> =
        c.positions.iter().map(|p| p.fen.as_str()).collect();
    let train_overlap_canon = train
        .positions
        .iter()
        .filter(|p| c_canons.contains(p.canon.as_str()))
        .count();
    let train_overlap_fen = train
        .positions
        .iter()
        .filter(|p| c_fens.contains(p.fen.as_str()))
        .count();
    anyhow::ensure!(
        train_overlap_canon == 0 && train_overlap_fen == 0,
        "STOP: TRAIN overlaps HOLDOUT_C ({train_overlap_canon} canonical classes, {train_overlap_fen} FENs)"
    );

    let report = serde_json::json!({
        "schema": "v3_p4_custody_v1",
        "p25_data_v1_train": {
            "file": a.train.display().to_string().replace('\\', "/"),
            "split": train.split.label(),
            "positions": train.positions.len(),
            "digest": train.digest,
            "schema_version": train.schema,
            "history_contract": train.history_contract,
            "independent_label_audit": {"checked": audit.checked, "failures": 0},
            "unique_exact_fens": seen_fen.len(),
            "unique_canonical_classes": seen_canon.len(),
            "canonical_keys_recomputed_and_equal": true,
            "cells": cells,
            "disjoint_from_holdout_c": {"canonical_overlap": train_overlap_canon, "exact_fen_overlap": train_overlap_fen},
            "note": "the digest above is the one read from the file by ProofTargets::load; it was not hardcoded",
        },
        "holdout_c": {
            "file": seal.file,
            "expected_digest": seal.expected_digest,
            "verified_digest": seal.actual_digest,
            "positions": seal.positions,
            "evaluated": false,
            "use_in_p4": seal.permitted_p4_use,
        },
        "exclusion_inventory": inv,
        "limitations": [
            "HP/X1/X2 exact datasets are not present on this workstation in a compatible form (docs/evidence/x2 and the HP branch datasets are absent), so cross-line disjointness from them was NOT verified",
        ],
        "wall_s": t0.elapsed().as_secs_f64(),
    });
    write_json(&a.output, &report)?;
    write_json(&a.seal_output, &serde_json::to_value(&seal)?)?;
    println!(
        "P25_DATA_V1 TRAIN: {} positions, digest {}",
        train.positions.len(),
        train.digest
    );
    println!(
        "HOLDOUT_C: digest verified ({}), evaluated = false",
        seal.actual_digest
    );
    println!(
        "inventory: {} datasets, {} canonical classes",
        inv.datasets.len(),
        inv.excluded_canonical_classes
    );
    println!(
        "wrote {} and {}",
        a.output.display(),
        a.seal_output.display()
    );
    Ok(())
}

fn tune_gen(a: TuneArgs) -> anyhow::Result<()> {
    let t0 = Instant::now();
    let inv = inventory(&dirs(&a.inventory_dirs))?;
    let spec = V3TuneSpec {
        per_cell: V3_TUNE_PER_CELL,
        seed: V3_TUNE_SEED,
        threads: a.threads,
        exclude_canon: inv.canons.clone(),
        exclude_fen: inv.fens.clone(),
    };
    let out = generate_v3_tune(&spec)?;
    let set = &out.set;
    anyhow::ensure!(
        set.positions.len() == 6 * V3_TUNE_PER_CELL,
        "unexpected size {}",
        set.positions.len()
    );

    // Independent audit of every position.
    let audit = audit_targets(set, a.threads);
    anyhow::ensure!(
        audit.failures.is_empty(),
        "STOP: v3_tune_v1 audit failed on {} of {}; first: {}",
        audit.failures.len(),
        audit.checked,
        audit.failures[0]
    );
    // Hard disjointness: internal uniqueness, and against every inventoried dataset.
    let mut fens = std::collections::HashSet::new();
    let mut canons = std::collections::HashSet::new();
    for p in &set.positions {
        anyhow::ensure!(
            fens.insert(p.fen.as_str()),
            "{}: duplicate FEN inside v3_tune_v1",
            p.id
        );
        anyhow::ensure!(
            canons.insert(p.canon.as_str()),
            "{}: duplicate canonical class inside v3_tune_v1",
            p.id
        );
        anyhow::ensure!(
            !inv.canons.contains(&p.canon),
            "{}: canonical overlap with an earlier dataset",
            p.id
        );
        anyhow::ensure!(
            !inv.fens.contains(&p.fen),
            "{}: exact FEN overlap with an earlier dataset",
            p.id
        );
    }
    // Pairwise check against each inventoried dataset as well (belt and braces).
    let mut disjoint_checked = 0usize;
    for d in &inv.datasets {
        let other = ProofTargets::load(Path::new(&d.path))?;
        let mut renamed = other.clone();
        renamed.split = Split::Train;
        let mut mine = set.clone();
        mine.split = Split::Tune;
        // check_disjoint ignores same-split overlaps, so give the two distinct splits.
        check_disjoint(&[&mine, &renamed]).map_err(anyhow::Error::msg)?;
        disjoint_checked += 1;
    }

    std::fs::create_dir_all(&a.data_dir)?;
    let path = a.data_dir.join("proof-v3-tune-v1.json");
    set.save(&path)?;
    let reloaded = load_working_split(&path, &[Split::Tune])?;
    anyhow::ensure!(
        reloaded.digest == set.digest,
        "saved dataset digest differs"
    );

    // Deterministic regeneration from the seed.
    let regen = if a.no_regeneration_check {
        serde_json::json!({"performed": false})
    } else {
        let again = generate_v3_tune(&spec)?;
        anyhow::ensure!(
            again.set.digest == set.digest,
            "STOP: regeneration from the same seed produced digest {} not {}",
            again.set.digest,
            set.digest
        );
        serde_json::json!({"performed": true, "identical_digest": true})
    };

    let manifest = serde_json::json!({
        "schema": "v3_tune_v1_manifest",
        "identity": V3_TUNE_IDENTITY,
        "purpose": "V3 P5/P6 model selection and information-sufficiency work only",
        "seed": V3_TUNE_SEED,
        "seed_hex": format!("{V3_TUNE_SEED:#x}"),
        "history_contract": set.history_contract,
        "per_cell_target": V3_TUNE_PER_CELL,
        "positions": set.positions.len(),
        "digest": set.digest,
        "file": path.display().to_string().replace('\\', "/"),
        "cell_counts": cell_counts(set),
        "pool_accounting": out.cells,
        "exclusion": {
            "datasets": inv.datasets,
            "excluded_canonical_classes": inv.excluded_canonical_classes,
            "excluded_exact_fens": inv.excluded_exact_fens,
            "exclusion_manifest_digest": inv.exclusion_manifest_digest,
            "limitation": "HP/X1/X2 exact datasets are not present on this workstation in a compatible form; cross-line disjointness from them was NOT verified",
        },
        "independent_audit": {"checked": audit.checked, "failures": 0},
        "disjointness": {
            "internal_unique_fens": fens.len(),
            "internal_unique_canonical_classes": canons.len(),
            "overlap_with_any_inventoried_dataset": 0,
            "pairwise_datasets_checked": disjoint_checked,
        },
        "deterministic_regeneration": regen,
        "wall_s": t0.elapsed().as_secs_f64(),
    });
    write_json(&a.manifest, &manifest)?;
    println!(
        "v3_tune_v1: {} positions, digest {}",
        set.positions.len(),
        set.digest
    );
    for (k, v) in cell_counts(set) {
        println!("  {k}: {v}");
    }
    println!(
        "exclusion manifest digest {}",
        inv.exclusion_manifest_digest
    );
    println!("wrote {}", a.manifest.display());
    Ok(())
}

fn load_source(path: &Path) -> anyhow::Result<ProofTargets> {
    load_working_split(path, &[Split::Train, Split::Tune])
}

fn trace_gen(a: TraceArgs) -> anyhow::Result<()> {
    let src = load_source(&a.data)?;
    let t0 = Instant::now();
    let m = generate_traces(&src, &a.dir, a.shard_size, a.threads, &|i, n, reused| {
        eprintln!(
            "[{:>6.0}s] shard {}/{}{}",
            t0.elapsed().as_secs_f64(),
            i + 1,
            n,
            if reused { " (reused, validated)" } else { "" }
        );
    })?;
    println!(
        "traced {} positions in {} shards; manifest digest {}",
        m.source_positions,
        m.shards.len(),
        m.manifest_digest
    );
    if let Some(e) = a.evidence {
        write_json(
            &e,
            &serde_json::json!({
                "contract": m.trace_schema,
                "trace_definition": m.trace_definition,
                "generator_version": m.generator_version,
                "source_dataset_digest": m.source_dataset_digest,
                "source_split": m.source_split.label(),
                "traces": m.source_positions,
                "shard_size": m.shard_size,
                "shards": m.shards.len(),
                "trace_manifest_digest": m.manifest_digest,
                "wall_s_last_invocation": t0.elapsed().as_secs_f64(),
            }),
        )?;
    }
    Ok(())
}

fn trace_audit(a: TraceArgs) -> anyhow::Result<()> {
    let src = load_source(&a.data)?;
    let t0 = Instant::now();
    let m = audit_traces(&src, &a.dir, a.threads, &|i, n, reused| {
        eprintln!(
            "[{:>6.0}s] audit shard {}/{}{}",
            t0.elapsed().as_secs_f64(),
            i + 1,
            n,
            if reused { " (reused)" } else { "" }
        );
    })?;
    println!(
        "audited {} traces: {} failures; audit manifest digest {}",
        m.total_checked, m.total_failures, m.manifest_digest
    );
    if let Some(e) = a.evidence {
        let tm = TraceManifest::load(&a.dir)?;
        write_json(
            &e,
            &serde_json::json!({
                "contract": tm.trace_schema,
                "source_dataset_digest": tm.source_dataset_digest,
                "traces": tm.source_positions,
                "shards": tm.shards.len(),
                "trace_manifest_digest": tm.manifest_digest,
                "independent_audit": {
                    "checked": m.total_checked,
                    "failures": m.total_failures,
                    "audit_manifest_digest": m.manifest_digest,
                    "first_failures": m.first_failures,
                },
            }),
        )?;
    }
    anyhow::ensure!(
        m.ok(),
        "the independent audit reported {} failures",
        m.total_failures
    );
    Ok(())
}

fn determinism(a: DetArgs) -> anyhow::Result<()> {
    let src = load_source(&a.data)?;
    let tm = TraceManifest::load(&a.dir)?;
    anyhow::ensure!(
        tm.source_dataset_digest == src.digest,
        "trace manifest does not belong to this dataset"
    );
    let mut results = Vec::new();
    for idx in a.shards.split(',').map(|s| s.trim().parse::<usize>()) {
        let idx = idx?;
        let stored = tm
            .shards
            .iter()
            .find(|s| s.index == idx)
            .ok_or_else(|| anyhow::anyhow!("no shard {idx}"))?;
        let one = generate_shard(&src, idx, tm.shard_size, 1)?;
        let many = generate_shard(&src, idx, tm.shard_size, a.threads)?;
        let same = one.digest == many.digest && one.digest == stored.digest;
        results.push(serde_json::json!({
            "shard": idx,
            "positions": one.count,
            "digest_1_thread": one.digest,
            "digest_n_threads": many.digest,
            "stored_digest": stored.digest,
            "identical": same,
        }));
        anyhow::ensure!(
            same,
            "shard {idx}: output depends on the thread count or differs from the stored shard"
        );
        eprintln!(
            "shard {idx}: identical on 1 and {} threads and equal to the stored shard",
            a.threads
        );
    }
    if let Some(e) = a.evidence {
        write_json(
            &e,
            &serde_json::json!({"threads_compared": [1, a.threads], "shards": results, "all_identical": true}),
        )?;
    }
    Ok(())
}

fn feasibility_cmd(a: FeasArgs) -> anyhow::Result<()> {
    let src = load_source(&a.data)?;
    let f = feasibility(&src, &a.dir, &a.role)?;
    write_json(&a.output, &serde_json::to_value(&f)?)?;
    println!(
        "{:<8} {:>3} {:>7} {:>5} {:>6} {:>5} {:>5} {:>6}   C2      C4      C8      C16",
        "family", "M", "n", "qmin", "qmed", "p90", "p95", "qmax"
    );
    for c in f.cells.iter() {
        println!(
            "{:<8} {:>3} {:>7} {:>5} {:>6} {:>5} {:>5} {:>6}   {:.4}  {:.4}  {:.4}  {:.4}",
            c.family,
            c.mate_depth,
            c.n,
            c.q_min,
            c.q_median,
            c.q_p90,
            c.q_p95,
            c.q_max,
            c.c2,
            c.c4,
            c.c8,
            c.c16
        );
    }
    if let Some(p) = &f.primary {
        println!(
            "\nPRIMARY {} {}: n={} count<=8={} measured={} threshold={} pass={}\n{}",
            p.primary_cell,
            p.primary_metric,
            p.n,
            p.count_le_8,
            p.measured_value,
            p.threshold,
            p.pass,
            p.classification
        );
    }
    Ok(())
}

// Keep the unused import used when audit manifests are inspected elsewhere.
#[allow(dead_code)]
fn _audit_type(_: &AuditManifest) {}
