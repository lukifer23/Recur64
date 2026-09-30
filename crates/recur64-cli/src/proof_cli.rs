//! `recur64 proof ...` — exact proof-position tooling (`ProofTargetsV1`).
//!
//! * `proof bench`  — generator benchmark gate: measured rate, acceptance and
//!   branching per mate band, and a wall projection for a target scale, BEFORE
//!   any full generation.
//! * `proof gen`    — generate CONFIRM, TUNE, TRAIN from independent seeds
//!   (hard-disjoint), audit every position through the independent path, write
//!   datasets and a report.
//! * `proof audit`  — re-audit datasets and re-check disjointness.

use std::collections::HashSet;
use std::path::PathBuf;
use std::time::Instant;

use clap::{Args, Subcommand};

use recur64_runtime::proof::audit::{audit_targets, check_disjoint};
use recur64_runtime::proof::generator::{GenSpec, generate};
use recur64_runtime::proof::targets::{ProofTargets, Split};

#[derive(Subcommand, Debug)]
pub enum ProofCmd {
    /// Generator benchmark gate on a small sample.
    Bench(BenchArgs),
    /// Generate and audit CONFIRM/TUNE/TRAIN.
    Gen(GenArgs),
    /// Re-audit datasets and disjointness.
    Audit(AuditArgs),
    /// Exact (exhaustive) pool sizes per family, to size the datasets.
    Pool(PoolArgs),
}

#[derive(Args, Debug)]
pub struct PoolArgs {
    #[arg(long)]
    pub output: PathBuf,
    /// Comma-separated family names (default: all five).
    #[arg(long, default_value = "KQvK,KRvK,KQQvK,KQRvK,KRRvK")]
    pub families: String,
    #[arg(long, default_value_t = 3)]
    pub max_depth: u8,
    #[arg(long, default_value_t = 20)]
    pub threads: usize,
}

#[derive(Args, Debug)]
pub struct BenchArgs {
    #[arg(long)]
    pub output: PathBuf,
    /// Positions per (family, band) in the benchmark sample.
    #[arg(long, default_value_t = 20)]
    pub sample_per_cell: usize,
    #[arg(long, default_value = "1,2,3")]
    pub depths: String,
    #[arg(long, default_value_t = 20)]
    pub threads: usize,
    /// Target scale used for the wall projection (per family x band, TRAIN).
    #[arg(long, default_value_t = 1000)]
    pub target_train_per_cell: usize,
    #[arg(long, default_value_t = 100)]
    pub target_eval_per_cell: usize,
    #[arg(long, default_value_t = 7)]
    pub seed: u64,
}

#[derive(Args, Debug)]
pub struct GenArgs {
    #[arg(long)]
    pub output: PathBuf,
    #[arg(long, default_value_t = 1000)]
    pub train_per_cell: usize,
    #[arg(long, default_value_t = 100)]
    pub tune_per_cell: usize,
    #[arg(long, default_value_t = 100)]
    pub confirm_per_cell: usize,
    #[arg(long, default_value = "1,2,3")]
    pub depths: String,
    #[arg(long, default_value_t = 20)]
    pub threads: usize,
    /// Independent generator seeds, one per split.
    #[arg(long, default_value_t = 0x7A11_0001)]
    pub seed_train: u64,
    #[arg(long, default_value_t = 0x7A11_0002)]
    pub seed_tune: u64,
    #[arg(long, default_value_t = 0x7A11_0003)]
    pub seed_confirm: u64,
}

#[derive(Args, Debug)]
pub struct AuditArgs {
    #[arg(long)]
    pub dir: PathBuf,
    #[arg(long, default_value_t = 20)]
    pub threads: usize,
}

fn depths(s: &str) -> anyhow::Result<Vec<u8>> {
    let d: Vec<u8> = s
        .split(',')
        .map(|x| x.trim().parse())
        .collect::<Result<_, _>>()?;
    anyhow::ensure!(
        !d.is_empty() && d.iter().all(|x| (1..=5).contains(x)),
        "depths must be in 1..=5"
    );
    Ok(d)
}

pub fn run(cmd: ProofCmd) -> anyhow::Result<()> {
    match cmd {
        ProofCmd::Bench(a) => bench(a),
        ProofCmd::Gen(a) => gen_all(a),
        ProofCmd::Audit(a) => audit(a),
        ProofCmd::Pool(a) => pool(a),
    }
}

fn spec(split: Split, seed: u64, per_cell: usize, depths: Vec<u8>, threads: usize) -> GenSpec {
    GenSpec {
        split,
        seed,
        per_cell,
        depths,
        threads,
        max_tries_per_thread: 400_000_000,
        forbid_fen: HashSet::new(),
        forbid_canon: HashSet::new(),
    }
}

fn bench(a: BenchArgs) -> anyhow::Result<()> {
    let all = depths(&a.depths)?;
    let mut bands = Vec::new();
    let mut projected_train = 0.0f64;
    let mut projected_eval = 0.0f64;
    for &d in &all {
        let t = Instant::now();
        let (data, rep) = generate(&spec(
            Split::Train,
            a.seed,
            a.sample_per_cell,
            vec![d],
            a.threads,
        ))?;
        let wall = t.elapsed().as_secs_f64();
        let n = data.positions.len();
        // Every (family, band) cell costs about wall/5 per sample_per_cell.
        let per_position = wall / n as f64;
        let train = per_position * (a.target_train_per_cell * 5) as f64;
        let eval = per_position * (a.target_eval_per_cell * 5) as f64 * 2.0;
        projected_train += train;
        projected_eval += eval;
        let accepted: u64 = rep.cells.iter().map(|c| c.accepted as u64).sum();
        let found: u64 = rep.cells.iter().map(|c| c.found).sum();
        let (rf, ra): (u64, u64) = rep.cells.iter().fold((0, 0), |s, c| {
            (s.0 + c.rejected_fraction, s.1 + c.rejected_ambiguity)
        });
        bands.push(serde_json::json!({
            "band": format!("M{d}"),
            "accepted": n,
            "wall_s": wall,
            "positions_per_sec": n as f64 / wall,
            "sampled": rep.sampled,
            "acceptance_rate_of_sampled": accepted as f64 / rep.sampled.max(1) as f64,
            "found_at_depth": found,
            "rejected_fraction": rf,
            "rejected_ambiguity": ra,
            "acceptance_after_band_found": accepted as f64 / found.max(1) as f64,
            "max_legal_root_moves": rep.max_legal_root,
            "solver_nodes": rep.solver_nodes,
            "pool_by_family": rep.pool_by_family,
            "projected_train_wall_s": train,
            "projected_tune_plus_confirm_wall_s": eval,
        }));
    }
    let report = serde_json::json!({
        "sample_per_cell": a.sample_per_cell,
        "threads": a.threads,
        "target_train_per_cell": a.target_train_per_cell,
        "target_eval_per_cell": a.target_eval_per_cell,
        "bands": bands,
        "projected_total_wall_s": projected_train + projected_eval,
        "projected_total_wall_min": (projected_train + projected_eval) / 60.0,
    });
    std::fs::create_dir_all(&a.output)?;
    let path = a.output.join("proof-bench.json");
    std::fs::write(&path, serde_json::to_vec_pretty(&report)?)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    println!("wrote {}", path.display());
    Ok(())
}

fn gen_all(a: GenArgs) -> anyhow::Result<()> {
    use recur64_runtime::proof::generator::{ExhaustiveSpec, generate_exhaustive};
    let ds = depths(&a.depths)?;
    let t0 = Instant::now();
    let out = generate_exhaustive(&ExhaustiveSpec {
        depths: ds,
        train: a.train_per_cell,
        tune: a.tune_per_cell,
        confirm: a.confirm_per_cell,
        seed_train: a.seed_train,
        seed_tune: a.seed_tune,
        seed_confirm: a.seed_confirm,
        threads: a.threads,
    })?;
    let gen_wall = t0.elapsed().as_secs_f64();

    // Every position of every split is re-derived through the independent path.
    let mut audits = Vec::new();
    for t in [&out.confirm, &out.tune, &out.train] {
        let t1 = Instant::now();
        let r = audit_targets(t, a.threads);
        anyhow::ensure!(
            r.failures.is_empty(),
            "{} audit failed on {} of {} positions; first: {}",
            t.split.label(),
            r.failures.len(),
            r.checked,
            r.failures[0]
        );
        audits.push(serde_json::json!({
            "split": t.split.label(),
            "checked": r.checked,
            "failures": 0,
            "wall_s": t1.elapsed().as_secs_f64(),
        }));
    }
    check_disjoint(&[&out.train, &out.tune, &out.confirm]).map_err(anyhow::Error::msg)?;

    std::fs::create_dir_all(&a.output)?;
    let mut summary = Vec::new();
    for t in [&out.train, &out.tune, &out.confirm] {
        let path = a.output.join(format!("proof-{}.json", t.split.label()));
        t.save(&path)?;
        summary.push(dataset_summary(t, &path));
    }
    let plan: Vec<_> = out
        .plan
        .iter()
        .map(|(family, depth, pool, cp)| {
            serde_json::json!({ "family": family, "depth": depth, "eligible_pool": pool, "plan": cp })
        })
        .collect();
    let report = serde_json::json!({
        "datasets": summary,
        "disjoint": "exact FEN and symmetry-canonical key: no overlap across train/tune/confirm (verified)",
        "audit": audits,
        "cell_plan": plan,
        "pools": out.pools,
        "generation_wall_s": gen_wall,
    });
    let rp = a.output.join("proof-gen-report.json");
    std::fs::write(&rp, serde_json::to_vec_pretty(&report)?)?;
    println!("{}", serde_json::to_string_pretty(&report["datasets"])?);
    println!("wrote {}", rp.display());
    Ok(())
}

/// Sizes, digest and chance top-1 by split, depth and family.
fn dataset_summary(t: &ProofTargets, path: &std::path::Path) -> serde_json::Value {
    use std::collections::BTreeMap;
    let mut by_depth: BTreeMap<String, (usize, f64)> = BTreeMap::new();
    let mut by_family: BTreeMap<String, (usize, f64)> = BTreeMap::new();
    let mut total = 0.0f64;
    for p in &t.positions {
        let c = p.chance_top1 as f64;
        total += c;
        for (m, k) in [
            (&mut by_depth, format!("M{}", p.mate_depth)),
            (&mut by_family, p.family.clone()),
        ] {
            let e = m.entry(k).or_insert((0, 0.0));
            e.0 += 1;
            e.1 += c;
        }
    }
    let fmt = |m: BTreeMap<String, (usize, f64)>| -> serde_json::Value {
        m.into_iter()
            .map(|(k, (n, s))| {
                (
                    k,
                    serde_json::json!({ "positions": n, "chance_top1": s / n as f64 }),
                )
            })
            .collect::<serde_json::Map<_, _>>()
            .into()
    };
    serde_json::json!({
        "split": t.split.label(),
        "file": path.display().to_string(),
        "positions": t.positions.len(),
        "digest": t.digest,
        "seed": t.seed,
        "chance_top1_overall": total / t.positions.len().max(1) as f64,
        "by_depth": fmt(by_depth),
        "by_family": fmt(by_family),
    })
}

fn audit(a: AuditArgs) -> anyhow::Result<()> {
    let mut sets = Vec::new();
    for split in ["train", "tune", "confirm"] {
        let path = a.dir.join(format!("proof-{split}.json"));
        let t = ProofTargets::load(&path)?; // validates schema, contracts and digest
        let r = audit_targets(&t, a.threads);
        println!(
            "{split}: {} positions, digest {}, audit failures {}",
            r.checked,
            t.digest,
            r.failures.len()
        );
        anyhow::ensure!(r.failures.is_empty(), "{split}: {}", r.failures[0]);
        sets.push(t);
    }
    let refs: Vec<&ProofTargets> = sets.iter().collect();
    check_disjoint(&refs).map_err(anyhow::Error::msg)?;
    println!("splits are hard-disjoint (exact FEN and canonical key)");
    Ok(())
}

fn pool(a: PoolArgs) -> anyhow::Result<()> {
    use recur64_runtime::proof::generator::exhaustive_pool;
    use recur64_runtime::proof::targets::FAMILIES;
    std::fs::create_dir_all(&a.output)?;
    let mut reports = Vec::new();
    for name in a.families.split(',') {
        let name = name.trim();
        let fi = FAMILIES
            .iter()
            .position(|f| f.0 == name)
            .ok_or_else(|| anyhow::anyhow!("unknown family {name}"))?;
        let r = exhaustive_pool(fi, a.max_depth, a.threads)?;
        println!("{}", serde_json::to_string(&r)?);
        std::fs::write(
            a.output
                .join(format!("proof-pool-{}.json", name.to_lowercase())),
            serde_json::to_vec_pretty(&r)?,
        )?;
        reports.push(r);
    }
    Ok(())
}
