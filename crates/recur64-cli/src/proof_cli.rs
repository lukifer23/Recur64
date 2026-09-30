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
    /// P2.5: generate the three heavy-family evaluation holdouts (A, B, C).
    P25Holdouts(HoldoutArgs),
    /// P2.5-D: build P25_DATA_V1 (heavy cells scaled toward 5,000 unique positions).
    P25Data(DataArgs),
    /// Policy-only fixed-data training on ProofTargetsV1.
    Train(TrainArgs),
    /// Evaluate a saved checkpoint on a split (per-position results).
    Eval(EvalArgs),
    /// Paired per-position bootstrap between two sets of evaluations.
    Compare(CompareArgs),
    /// 2x2 factorial interaction (CF-C0)-(LF-L), paired by model-seed identity.
    Interaction(InteractionArgs),
}

#[derive(Args, Debug)]
pub struct DataArgs {
    /// Directory with the replacement P2 proof-{train,tune,confirm}.json.
    #[arg(long)]
    pub base: PathBuf,
    /// Comma-separated directories whose every proof-*.json dataset is excluded
    /// (retired splits, replacement splits, the holdouts).
    #[arg(long)]
    pub exclude_dirs: String,
    #[arg(long)]
    pub output: PathBuf,
    /// New positions wanted per heavy cell (existing 1,000 are kept on top).
    #[arg(long, default_value_t = 4000)]
    pub target_per_cell: usize,
    #[arg(long, default_value_t = 20)]
    pub threads: usize,
    /// Extension selection seed: 0x7A140001.
    #[arg(long, default_value_t = 2048131073)]
    pub seed: u64,
}

#[derive(Args, Debug)]
pub struct HoldoutArgs {
    #[arg(long)]
    pub output: PathBuf,
    /// Comma-separated directories holding proof-{train,tune,confirm}.json whose
    /// canonical classes are excluded.
    #[arg(long)]
    pub exclude_dirs: String,
    #[arg(long, default_value_t = 500)]
    pub per_cell: usize,
    #[arg(long, default_value_t = 20)]
    pub threads: usize,
    /// Independent holdout seeds (A, B, C): 0x7A130001..3.
    #[arg(long, default_value_t = 2048065537)]
    pub seed_a: u64,
    #[arg(long, default_value_t = 2048065538)]
    pub seed_b: u64,
    #[arg(long, default_value_t = 2048065539)]
    pub seed_c: u64,
}

#[derive(Args, Debug)]
pub struct TrainArgs {
    /// ProbeConfig TOML: model geometry, device and precision.
    #[arg(long)]
    pub config: PathBuf,
    /// Directory holding proof-{train,tune,confirm}.json.
    #[arg(long)]
    pub data: PathBuf,
    #[arg(long)]
    pub output: PathBuf,
    #[arg(long)]
    pub seed: u64,
    #[arg(long)]
    pub lr: f64,
    #[arg(long)]
    pub updates: usize,
    /// Warmup updates (default: 10% of updates, at least 5).
    #[arg(long)]
    pub warmup: Option<usize>,
    #[arg(long, default_value_t = 64)]
    pub micro: usize,
    #[arg(long, default_value_t = 4)]
    pub accum: usize,
    #[arg(long, default_value_t = 50)]
    pub eval_every: usize,
    /// Also evaluate CONFIRM at the end. CONFIRM is never touched otherwise; the
    /// summary records whether it was evaluated.
    #[arg(long)]
    pub eval_confirm: bool,
    /// Skip saving the checkpoint (LR screens).
    #[arg(long)]
    pub no_checkpoint: bool,
    /// Example sampler: cell_balanced_v1 (default, the P1b/P2 contract) or
    /// uniform_v0 (the historical P1a sampler).
    #[arg(long, default_value = "cell_balanced_v1")]
    pub sampler: String,
}

#[derive(Args, Debug)]
pub struct EvalArgs {
    #[arg(long)]
    pub config: PathBuf,
    /// Checkpoint directory; omit to evaluate a fresh (untrained) model.
    #[arg(long)]
    pub checkpoint: Option<PathBuf>,
    #[arg(long)]
    pub data: PathBuf,
    /// train | tune | confirm
    #[arg(long)]
    pub split: String,
    #[arg(long)]
    pub output: PathBuf,
    /// Seed for a fresh model's initialization.
    #[arg(long, default_value_t = 1)]
    pub seed: u64,
}

#[derive(Args, Debug)]
pub struct InteractionArgs {
    #[arg(long)]
    pub c0: String,
    #[arg(long)]
    pub cf: String,
    #[arg(long)]
    pub l: String,
    #[arg(long)]
    pub lf: String,
    #[arg(long, default_value_t = 10000)]
    pub resamples: usize,
    #[arg(long, default_value_t = 17)]
    pub seed: u64,
    #[arg(long)]
    pub output: PathBuf,
}

#[derive(Args, Debug)]
pub struct CompareArgs {
    /// Comma-separated eval files for side A (one per seed).
    #[arg(long)]
    pub a: String,
    /// Comma-separated eval files for side B (one per seed).
    #[arg(long)]
    pub b: String,
    #[arg(long, default_value_t = 10000)]
    pub resamples: usize,
    #[arg(long, default_value_t = 17)]
    pub seed: u64,
    /// Also report each seed separately, paired by model-seed identity (refused
    /// unless both sides carry unique, equal seed sets).
    #[arg(long)]
    pub per_seed: bool,
    #[arg(long)]
    pub output: PathBuf,
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
        ProofCmd::P25Holdouts(a) => holdouts_cmd(a),
        ProofCmd::P25Data(a) => p25_data_cmd(a),
        ProofCmd::Train(a) => train_cmd(a),
        ProofCmd::Eval(a) => eval_cmd(a),
        ProofCmd::Compare(a) => compare_cmd(a),
        ProofCmd::Interaction(a) => interaction_cmd(a),
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

use recur64_model::config::{Architecture, DeviceKind, ProbeConfig};
use recur64_model::net::NeuralModel;
use recur64_runtime::proof::compare::{EvalFile, compare};
use recur64_runtime::proof::train::{
    EvalSummary, PosResult, SamplerKind, TrainSpec, evaluate, prepare, summarize, train,
};
use recur64_runtime::{gpu_telemetry, model_io};

fn load_split(dir: &std::path::Path, split: &str) -> anyhow::Result<ProofTargets> {
    ProofTargets::load(&dir.join(format!("proof-{split}.json")))
}

/// Experimental-hygiene guard (not a security mechanism): make every CONFIRM
/// evaluation loud and leave a trace next to the datasets.
fn announce_confirm(data: &std::path::Path, digest: &str, who: &str, seed: u64) {
    announce("CONFIRM", "confirm-exposure.log", data, digest, who, seed);
}

/// Same guard for the P2.5 holdouts (`holdout_a|b|c`).
fn announce_holdout(data: &std::path::Path, split: &str, digest: &str, who: &str, seed: u64) {
    let label = split.to_uppercase();
    announce(&label, "holdout-exposure.log", data, digest, who, seed);
}

fn announce(label: &str, log: &str, data: &std::path::Path, digest: &str, who: &str, seed: u64) {
    eprintln!("{label} DATASET IS BEING EVALUATED:\n{digest}");
    let line = format!(
        "{} {label} digest={digest} model={who} seed={seed}\n",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    );
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(data.join(log))
    {
        let _ = f.write_all(line.as_bytes());
    }
}

fn probe_config(path: &std::path::Path) -> anyhow::Result<ProbeConfig> {
    let cfg = ProbeConfig::from_toml_str(&std::fs::read_to_string(path)?)?;
    cfg.model.validate()?;
    anyhow::ensure!(
        cfg.precision == recur64_model::config::Precision::Fp32,
        "fixed-data training is FP32 (fusion/autotune/TF32 are not part of V2.5)"
    );
    Ok(cfg)
}

/// Provenance recorded in every evaluation file.
struct Provenance<'a> {
    model: &'a str,
    model_id: &'a str,
    seed: u64,
    architecture: &'a str,
    updates: Option<usize>,
    peak_lr: Option<f64>,
    sampler: Option<String>,
}

fn write_eval(
    path: &std::path::Path,
    split: &str,
    digest: &str,
    prov: &Provenance,
    results: Vec<PosResult>,
) -> anyhow::Result<EvalSummary> {
    let summary = summarize(&results);
    let file = EvalFile {
        split: split.into(),
        dataset_digest: digest.into(),
        model: prov.model.into(),
        model_id: prov.model_id.into(),
        model_seed: Some(prov.seed),
        architecture: Some(prov.architecture.into()),
        training_updates: prov.updates,
        peak_lr: prov.peak_lr,
        sampler_version: prov.sampler.clone(),
        summary: summary.clone(),
        results,
    };
    std::fs::write(path, serde_json::to_vec(&file)?)?;
    Ok(summary)
}

fn model_id_of(dir: &std::path::Path) -> String {
    std::fs::read(dir.join("meta.json"))
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .and_then(|v| v["model_id"].as_str().map(str::to_string))
        .unwrap_or_default()
}

#[allow(clippy::too_many_arguments)]
fn run_train<B, MT, MI>(cfg: &ProbeConfig, a: &TrainArgs) -> anyhow::Result<serde_json::Value>
where
    B: burn::tensor::backend::AutodiffBackend,
    MT: NeuralModel<B> + burn::module::AutodiffModule<B, InnerModule = MI>,
    MI: NeuralModel<B::InnerBackend>,
{
    let device: burn::tensor::Device<B> = Default::default();
    let inner: burn::tensor::Device<B::InnerBackend> = Default::default();
    B::seed(&device, a.seed);
    let sampler = match a.sampler.as_str() {
        "cell_balanced_v1" => SamplerKind::CellBalancedV1,
        "uniform_v0" => SamplerKind::UniformV0,
        other => anyhow::bail!("unknown sampler '{other}'"),
    };
    let train_set = load_split(&a.data, "train")?;
    let tune_set = load_split(&a.data, "tune")?;
    let train_pos = prepare(&train_set)?;
    let tune_pos = prepare(&tune_set)?;

    let model = model_io::build_as::<B, MT>(&cfg.model, &device)?;
    let params = model.param_count();
    let mut optim = recur64_model::train::adamw::<B, MT>();
    let spec = TrainSpec {
        updates: a.updates,
        warmup: a.warmup.unwrap_or((a.updates / 10).max(5)),
        lr: a.lr,
        micro: a.micro,
        accum: a.accum,
        seed: a.seed,
        eval_every: a.eval_every,
        sampler,
    };
    let gpu = cfg.device == DeviceKind::Cuda;
    std::fs::create_dir_all(&a.output)?;
    let t0 = Instant::now();
    let (out, gpu_samples) = gpu_telemetry::monitor(gpu, || {
        train::<B, MT, MI, _>(
            model,
            &mut optim,
            &spec,
            &train_pos,
            &tune_pos,
            &device,
            &inner,
            |r| {
                if r.update % 25 == 0 {
                    eprintln!(
                        "update {:>4}  lr {:.2e}  loss {:.4}  gnorm {:.3}",
                        r.update, r.lr, r.report.total_loss, r.report.grad_norm
                    );
                }
            },
        )
    });
    let (model, run) = out?;
    let wall = t0.elapsed().as_secs_f64();

    let inner_model = burn::module::AutodiffModule::valid(&model);
    let mut model_id = String::new();
    if !a.no_checkpoint {
        let dir = a.output.join("checkpoint");
        let meta = recur64_model::checkpoint::CheckpointMeta::new(
            cfg.model.clone(),
            1,
            false,
            a.updates as u64,
            a.lr,
            a.seed,
            0,
            "proof-train",
            cfg.precision.label(),
        );
        recur64_model::checkpoint::save_training(&dir, &model, &optim, &meta)?;
        model_id = model_id_of(&dir);
    }
    let prov = Provenance {
        model: &cfg.name,
        model_id: &model_id,
        seed: a.seed,
        architecture: cfg.model.architecture.id(),
        updates: Some(a.updates),
        peak_lr: Some(a.lr),
        sampler: Some(sampler.label().to_string()),
    };
    let tune_results = evaluate::<B::InnerBackend, MI>(&inner_model, &inner, &tune_pos)?;
    let tune_summary = write_eval(
        &a.output.join("eval-tune.json"),
        "tune",
        &tune_set.digest,
        &prov,
        tune_results,
    )?;
    // Every TRAIN position, so an overfit gap is visible per family and cell
    // (this replaces the earlier first-1,208 slice, which was one family).
    let train_results = evaluate::<B::InnerBackend, MI>(&inner_model, &inner, &train_pos)?;
    let train_full = write_eval(
        &a.output.join("eval-train.json"),
        "train",
        &train_set.digest,
        &prov,
        train_results,
    )?;
    let confirm = if a.eval_confirm {
        let confirm_set = load_split(&a.data, "confirm")?;
        announce_confirm(&a.data, &confirm_set.digest, &cfg.name, a.seed);
        let confirm_pos = prepare(&confirm_set)?;
        let r = evaluate::<B::InnerBackend, MI>(&inner_model, &inner, &confirm_pos)?;
        let s = write_eval(
            &a.output.join("eval-confirm.json"),
            "confirm",
            &confirm_set.digest,
            &prov,
            r,
        )?;
        serde_json::json!({ "evaluated": true, "digest": confirm_set.digest, "summary": s })
    } else {
        serde_json::json!({ "evaluated": false })
    };

    let first = run.updates.first().map(|u| u.report.total_loss);
    let last = run.updates.last().map(|u| u.report.total_loss);
    let max_gnorm = run
        .updates
        .iter()
        .map(|u| u.report.grad_norm)
        .fold(0.0f32, f32::max);
    std::fs::write(
        a.output.join("updates.json"),
        serde_json::to_vec(&run.updates)?,
    )?;
    let summary = serde_json::json!({
        "name": cfg.name,
        "architecture": cfg.model.architecture.id(),
        "params": params,
        "spec": spec,
        "sampler_version": sampler.label(),
        "wall_s": wall,
        "sec_per_update": wall / a.updates as f64,
        "gpu": gpu_samples,
        "epochs_seen_over_all_positions": run.epochs_seen,
        "sampler_stats": run.sampler_stats,
        "loss_first": first,
        "loss_last": last,
        "max_grad_norm": max_gnorm,
        "all_finite": run.updates.iter().all(|u| u.report.total_loss.is_finite() && u.report.grad_norm.is_finite()),
        "train_digest": train_set.digest,
        "tune_digest": tune_set.digest,
        "tune_curve": run.tune_curve,
        "tune_final": tune_summary,
        "train_full_final": train_full,
        "confirm": confirm,
        "model_id": model_id,
    });
    std::fs::write(
        a.output.join("summary.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    Ok(summary)
}

fn train_cmd(a: TrainArgs) -> anyhow::Result<()> {
    use recur64_model::candidate::CandidateV25Model;
    use recur64_model::legacy_facts::LegacyFactsModel;
    use recur64_model::model::ProbeModel;
    let cfg = probe_config(&a.config)?;
    type Cpu = recur64_model::train::CpuTrainBackend;
    let summary = match (cfg.device, cfg.model.architecture) {
        (_, Architecture::ActiveSearchV3) => anyhow::bail!(
            "active_search_v3 is not supported by `proof train`: it has no fixed-data batched path,              because every budget above 0 needs the live query tool (use the v3 commands)"
        ),
        (DeviceKind::Cpu, Architecture::CandidateV25) => {
            run_train::<Cpu, CandidateV25Model<Cpu>, CandidateV25Model<burn::backend::Flex>>(
                &cfg, &a,
            )?
        }
        (DeviceKind::Cpu, Architecture::ProbeV1) => {
            run_train::<Cpu, ProbeModel<Cpu>, ProbeModel<burn::backend::Flex>>(&cfg, &a)?
        }
        (DeviceKind::Cpu, Architecture::LegacyFactsV25) => {
            run_train::<Cpu, LegacyFactsModel<Cpu>, LegacyFactsModel<burn::backend::Flex>>(
                &cfg, &a,
            )?
        }
        #[cfg(feature = "cuda")]
        (DeviceKind::Cuda, Architecture::CandidateV25) => {
            type G = burn::backend::Autodiff<burn::backend::Cuda>;
            run_train::<G, CandidateV25Model<G>, CandidateV25Model<burn::backend::Cuda>>(&cfg, &a)?
        }
        #[cfg(feature = "cuda")]
        (DeviceKind::Cuda, Architecture::ProbeV1) => {
            type G = burn::backend::Autodiff<burn::backend::Cuda>;
            run_train::<G, ProbeModel<G>, ProbeModel<burn::backend::Cuda>>(&cfg, &a)?
        }
        #[cfg(feature = "cuda")]
        (DeviceKind::Cuda, Architecture::LegacyFactsV25) => {
            type G = burn::backend::Autodiff<burn::backend::Cuda>;
            run_train::<G, LegacyFactsModel<G>, LegacyFactsModel<burn::backend::Cuda>>(&cfg, &a)?
        }
        #[cfg(not(feature = "cuda"))]
        (DeviceKind::Cuda, _) => {
            anyhow::bail!("CUDA support is not compiled; rebuild with --features cuda")
        }
    };
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

fn run_eval<B, MI>(cfg: &ProbeConfig, a: &EvalArgs) -> anyhow::Result<()>
where
    B: burn::tensor::backend::Backend,
    MI: NeuralModel<B>,
{
    let device: burn::tensor::Device<B> = Default::default();
    B::seed(&device, a.seed);
    let set = load_split(&a.data, &a.split)?;
    let (model, id, seed, updates, lr) = match &a.checkpoint {
        Some(dir) => {
            // The authoritative seed comes from checkpoint metadata, never from a name.
            let meta: serde_json::Value =
                serde_json::from_slice(&std::fs::read(dir.join("meta.json"))?)?;
            let seed = meta["seed"]
                .as_u64()
                .ok_or_else(|| anyhow::anyhow!("{}: meta.json carries no seed", dir.display()))?;
            (
                model_io::load_as::<B, MI>(dir, &cfg.model, &device)?,
                model_id_of(dir),
                seed,
                meta["update_counter"].as_u64().map(|u| u as usize),
                meta["lr"].as_f64(),
            )
        }
        None => (
            model_io::build_as::<B, MI>(&cfg.model, &device)?,
            "fresh".to_string(),
            a.seed,
            None,
            None,
        ),
    };
    if a.split == "confirm" {
        announce_confirm(&a.data, &set.digest, &cfg.name, seed);
    } else if a.split.starts_with("holdout") {
        announce_holdout(&a.data, &a.split, &set.digest, &cfg.name, seed);
    }
    let pos = prepare(&set)?;
    let results = evaluate::<B, MI>(&model, &device, &pos)?;
    let prov = Provenance {
        model: &cfg.name,
        model_id: &id,
        seed,
        architecture: cfg.model.architecture.id(),
        updates,
        peak_lr: lr,
        sampler: None,
    };
    let s = write_eval(&a.output, &a.split, &set.digest, &prov, results)?;
    println!("{}", serde_json::to_string_pretty(&s)?);
    Ok(())
}

fn eval_cmd(a: EvalArgs) -> anyhow::Result<()> {
    use recur64_model::candidate::CandidateV25Model;
    use recur64_model::legacy_facts::LegacyFactsModel;
    use recur64_model::model::ProbeModel;
    let cfg = probe_config(&a.config)?;
    match (cfg.device, cfg.model.architecture) {
        (_, Architecture::ActiveSearchV3) => anyhow::bail!(
            "active_search_v3 is not supported by `proof eval`: it has no fixed-data batched path,              because every budget above 0 needs the live query tool (use the v3 commands)"
        ),
        (DeviceKind::Cpu, Architecture::CandidateV25) => {
            run_eval::<burn::backend::Flex, CandidateV25Model<burn::backend::Flex>>(&cfg, &a)
        }
        (DeviceKind::Cpu, Architecture::ProbeV1) => {
            run_eval::<burn::backend::Flex, ProbeModel<burn::backend::Flex>>(&cfg, &a)
        }
        (DeviceKind::Cpu, Architecture::LegacyFactsV25) => {
            run_eval::<burn::backend::Flex, LegacyFactsModel<burn::backend::Flex>>(&cfg, &a)
        }
        #[cfg(feature = "cuda")]
        (DeviceKind::Cuda, Architecture::CandidateV25) => {
            run_eval::<burn::backend::Cuda, CandidateV25Model<burn::backend::Cuda>>(&cfg, &a)
        }
        #[cfg(feature = "cuda")]
        (DeviceKind::Cuda, Architecture::ProbeV1) => {
            run_eval::<burn::backend::Cuda, ProbeModel<burn::backend::Cuda>>(&cfg, &a)
        }
        #[cfg(feature = "cuda")]
        (DeviceKind::Cuda, Architecture::LegacyFactsV25) => {
            run_eval::<burn::backend::Cuda, LegacyFactsModel<burn::backend::Cuda>>(&cfg, &a)
        }
        #[cfg(not(feature = "cuda"))]
        (DeviceKind::Cuda, _) => {
            anyhow::bail!("CUDA support is not compiled; rebuild with --features cuda")
        }
    }
}

fn read_evals(list: &str) -> anyhow::Result<Vec<EvalFile>> {
    list.split(',')
        .map(|p| Ok(serde_json::from_slice(&std::fs::read(p.trim())?)?))
        .collect()
}

fn compare_cmd(a: CompareArgs) -> anyhow::Result<()> {
    let (fa, fb) = (read_evals(&a.a)?, read_evals(&a.b)?);
    if fa.iter().chain(&fb).any(|f| f.split == "confirm") {
        eprintln!(
            "NOTE: comparing CONFIRM evaluations (digest {})",
            fa[0].dataset_digest
        );
    }
    let v = compare(&fa, &fb, a.resamples, a.seed, a.per_seed).map_err(anyhow::Error::msg)?;
    std::fs::write(&a.output, serde_json::to_vec_pretty(&v)?)?;
    println!("{}", serde_json::to_string_pretty(&v)?);
    Ok(())
}

fn holdouts_cmd(a: HoldoutArgs) -> anyhow::Result<()> {
    use recur64_runtime::proof::generator::{HoldoutSpec, exclusion_digest, generate_holdouts};

    // Exclusion set: every canonical class of every earlier split.
    let mut exclude = HashSet::new();
    let mut excluded_datasets = Vec::new();
    let mut digests = Vec::new();
    for dir in a.exclude_dirs.split(',') {
        for split in ["train", "tune", "confirm"] {
            let path = std::path::Path::new(dir.trim()).join(format!("proof-{split}.json"));
            let t = ProofTargets::load(&path)?; // validates schema, contracts and digest
            for p in &t.positions {
                exclude.insert(p.canon.clone());
            }
            digests.push(t.digest.clone());
            excluded_datasets.push(serde_json::json!({
                "file": path.display().to_string(), "split": split,
                "positions": t.positions.len(), "digest": t.digest,
            }));
        }
    }
    let manifest_digest = exclusion_digest(&exclude, &digests);
    let t0 = Instant::now();
    let out = generate_holdouts(&HoldoutSpec {
        per_cell: a.per_cell,
        seeds: [a.seed_a, a.seed_b, a.seed_c],
        threads: a.threads,
        exclude_canon: exclude.clone(),
    })?;
    let wall = t0.elapsed().as_secs_f64();

    // Independent audit of every position, then disjointness (among holdouts and
    // against every excluded class, by canonical key and by exact FEN).
    let mut audits = Vec::new();
    for t in &out.sets {
        let r = audit_targets(t, a.threads);
        anyhow::ensure!(
            r.failures.is_empty(),
            "{} audit failed on {} of {}; first: {}",
            t.split.label(),
            r.failures.len(),
            r.checked,
            r.failures[0]
        );
        audits.push(
            serde_json::json!({ "split": t.split.label(), "checked": r.checked, "failures": 0 }),
        );
        for p in &t.positions {
            anyhow::ensure!(
                !exclude.contains(&p.canon),
                "{}: canonical class overlaps an excluded earlier split",
                p.id
            );
        }
    }
    let refs: Vec<&ProofTargets> = out.sets.iter().collect();
    check_disjoint(&refs).map_err(anyhow::Error::msg)?;

    std::fs::create_dir_all(&a.output)?;
    let mut metas = Vec::new();
    for t in &out.sets {
        let path = a.output.join(format!("proof-{}.json", t.split.label()));
        t.save(&path)?;
        let meta = dataset_summary(t, &path);
        std::fs::write(
            a.output
                .join(format!("{}-meta.json", t.split.label().replace('_', "-"))),
            serde_json::to_vec_pretty(&meta)?,
        )?;
        metas.push(meta);
    }
    let exclusion = serde_json::json!({
        "excluded_datasets": excluded_datasets,
        "excluded_canonical_classes": exclude.len(),
        "exclusion_manifest_digest": manifest_digest,
        "also_excluded_by_construction": "every P2.5 training-extension position (selected after the holdouts, excluding them)",
    });
    std::fs::write(
        a.output.join("exclusion-manifest.json"),
        serde_json::to_vec_pretty(&exclusion)?,
    )?;
    let accounting = serde_json::json!({
        "cells": out.cells,
        "pools": out.pools,
        "audit": audits,
        "disjoint": "holdouts A/B/C are mutually disjoint and disjoint from every excluded earlier split, by canonical class and exact FEN (verified)",
        "seeds": { "a": a.seed_a, "b": a.seed_b, "c": a.seed_c },
        "per_cell": a.per_cell,
        "generation_wall_s": wall,
    });
    std::fs::write(
        a.output.join("holdout-pool-accounting.json"),
        serde_json::to_vec_pretty(&accounting)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&metas)?);
    println!("{}", serde_json::to_string_pretty(&accounting["cells"])?);
    Ok(())
}

fn interaction_cmd(a: InteractionArgs) -> anyhow::Result<()> {
    use recur64_runtime::proof::compare::interaction;
    let v = interaction(
        &read_evals(&a.c0)?,
        &read_evals(&a.cf)?,
        &read_evals(&a.l)?,
        &read_evals(&a.lf)?,
        a.resamples,
        a.seed,
        true,
    )
    .map_err(anyhow::Error::msg)?;
    std::fs::write(&a.output, serde_json::to_vec_pretty(&v)?)?;
    println!("{}", serde_json::to_string_pretty(&v)?);
    Ok(())
}

/// Every `proof-*.json` dataset in `dir` (skips reports).
fn datasets_in(dir: &std::path::Path) -> anyhow::Result<Vec<ProofTargets>> {
    let mut out = Vec::new();
    let mut names: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|n| {
            n.starts_with("proof-")
                && n.ends_with(".json")
                && n != "proof-gen-report.json"
                && !n.starts_with("proof-pool-")
        })
        .collect();
    names.sort();
    for n in names {
        out.push(ProofTargets::load(&dir.join(n))?);
    }
    Ok(out)
}

fn p25_data_cmd(a: DataArgs) -> anyhow::Result<()> {
    use recur64_runtime::proof::generator::{ExtensionSpec, exclusion_digest, generate_extension};

    let base_train = load_split(&a.base, "train")?;
    let base_tune = load_split(&a.base, "tune")?;
    let base_confirm = load_split(&a.base, "confirm")?;

    let mut exclude = HashSet::new();
    let mut excluded = Vec::new();
    let mut digests = Vec::new();
    let mut holdouts = Vec::new();
    for dir in a.exclude_dirs.split(',') {
        for t in datasets_in(std::path::Path::new(dir.trim()))? {
            for p in &t.positions {
                exclude.insert(p.canon.clone());
            }
            excluded.push(serde_json::json!({
                "dir": dir.trim(), "split": t.split.label(),
                "positions": t.positions.len(), "digest": t.digest,
            }));
            digests.push(t.digest.clone());
            if t.split.label().starts_with("holdout") {
                holdouts.push(t);
            }
        }
    }
    anyhow::ensure!(
        holdouts.len() == 3,
        "expected holdouts A, B and C among the excluded datasets"
    );
    let manifest_digest = exclusion_digest(&exclude, &digests);
    let t0 = Instant::now();
    let ext = generate_extension(&ExtensionSpec {
        target_per_cell: a.target_per_cell,
        seed: a.seed,
        threads: a.threads,
        exclude_canon: exclude.clone(),
    })?;

    // Every added position must avoid every excluded class.
    for p in &ext.added {
        anyhow::ensure!(
            !exclude.contains(&p.canon),
            "{}: an added position overlaps an excluded class",
            p.id
        );
    }
    let added_n = ext.added.len();
    let mut positions = base_train.positions.clone();
    positions.extend(ext.added.iter().cloned());
    let filters = serde_json::json!({
        "dataset": "P25_DATA_V1",
        "base": "replacement P2 TRAIN (small families and the 1,000 heavy positions per cell kept unchanged)",
        "added_per_heavy_cell_target": a.target_per_cell,
        "exclusion_manifest_digest": manifest_digest,
        "pool_limit_rule": "holdouts first; the extension takes all that remains up to the target; nothing relaxed",
        "max_correct_fraction": recur64_runtime::proof::generator::MAX_CORRECT_FRACTION,
        "extension_seed": a.seed,
    });
    let combined = ProofTargets::new(Split::Train, a.seed, filters, positions);

    // Independent audit of the ENTIRE combined set, then disjointness from every
    // evaluation set (by canonical class and exact FEN).
    let audit = audit_targets(&combined, a.threads);
    anyhow::ensure!(
        audit.failures.is_empty(),
        "audit failed on {} of {}; first: {}",
        audit.failures.len(),
        audit.checked,
        audit.failures[0]
    );
    let mut eval_sets: Vec<&ProofTargets> = vec![&base_tune, &base_confirm];
    eval_sets.extend(holdouts.iter());
    for e in &eval_sets {
        let canon: HashSet<&str> = e.positions.iter().map(|p| p.canon.as_str()).collect();
        let fens: HashSet<&str> = e.positions.iter().map(|p| p.fen.as_str()).collect();
        for p in &combined.positions {
            anyhow::ensure!(
                !canon.contains(p.canon.as_str()) && !fens.contains(p.fen.as_str()),
                "{} overlaps evaluation set {}",
                p.id,
                e.split.label()
            );
        }
    }

    std::fs::create_dir_all(&a.output)?;
    let train_path = a.output.join("proof-train.json");
    combined.save(&train_path)?;
    base_tune.save(&a.output.join("proof-tune.json"))?;

    // Per-cell unique counts and oversampling at P2 length (102,400 examples / 15 cells).
    let mut counts: std::collections::BTreeMap<String, usize> = Default::default();
    for p in &combined.positions {
        *counts
            .entry(format!("{}-M{}", p.family, p.mate_depth))
            .or_default() += 1;
    }
    let per_cell_examples = 102_400f64 / 15.0;
    let cell_table: Vec<_> = counts
        .iter()
        .map(|(k, n)| {
            serde_json::json!({
                "cell": k, "unique_train_positions": n,
                "local_epochs_at_400_updates": per_cell_examples / *n as f64,
            })
        })
        .collect();
    let report = serde_json::json!({
        "dataset": "P25_DATA_V1",
        "file": train_path.display().to_string(),
        "digest": combined.digest,
        "total_positions": combined.positions.len(),
        "base_positions": base_train.positions.len(),
        "added_positions": added_n,
        "extension_seed": a.seed,
        "tune_digest": base_tune.digest,
        "exclusion_manifest_digest": manifest_digest,
        "excluded_datasets": excluded,
        "extension_cells": ext.cells,
        "unique_positions_per_cell": cell_table,
        "audit": { "checked": audit.checked, "failures": 0 },
        "disjoint": "no position of the combined TRAIN shares a canonical class or exact FEN with replacement TUNE/CONFIRM or holdouts A/B/C (verified); every ADDED position also avoids the retired splits and the replacement TRAIN (verified)",
        "wall_s": t0.elapsed().as_secs_f64(),
    });
    std::fs::write(
        a.output.join("p25-data-report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
