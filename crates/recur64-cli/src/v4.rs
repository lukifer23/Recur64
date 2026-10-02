//! `recur64 v4 ...`: the `evidence_belief_v4` research line (P0/P1: mechanism development on
//! TRAIN only).
//!
//! * `model-info`: describe the real V4 graph (parameters by group, contracts);
//! * `tune-gen` / `tune-verify`: generate, audit and seal `v4_tune_v1`, and re-verify custody
//!   (the seal and HOLDOUT_C's digest) without evaluating anything;
//! * `train`: one resumable stage run (A base, B evidence, C utility) on P25 TRAIN only;
//! * `measure`: the pre-registered mechanism measurements on `V4_TRAIN_DEV` (a held-out part of
//!   TRAIN) for stage-A/B/C/D finals;
//! * `bench`: timing and VRAM (root forward, one query, B2/B4/B8, probes, training).
//!
//! No command here opens `v4_tune_v1` for evaluation or HOLDOUT_C at all. A requested device that
//! is unavailable is an error, never a substitution.

use std::path::{Path, PathBuf};
use std::time::Instant;

use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use clap::{Args, Subcommand, ValueEnum};

use recur64_model::config::{Architecture, ModelConfig, ProbeConfig};
use recur64_runtime::gpu_telemetry::{monitor, sample_gpu};
use recur64_runtime::model_io;
use recur64_runtime::proof::audit::{audit_targets, check_disjoint};
use recur64_runtime::proof::custody::{
    V4_TUNE_IDENTITY, V4_TUNE_PER_CELL, V4_TUNE_POSITIONS, V4_TUNE_SEED, cell_counts, inventory,
    seal_v4_tune, verify_holdout_c,
};
use recur64_runtime::proof::generator::{V4TuneSpec, generate_v4_tune};
use recur64_runtime::proof::targets::{ProofTargets, Split};
use recur64_v4::data::V4Data;
use recur64_v4::model::EvidenceBeliefModel;
use recur64_v4::session::{RunOptions, Selection};
use recur64_v4::stage::{Recipe, Stage, Trainer, inference, load_model};
use recur64_v4::study::{
    SeedModel, collect_probes, dev1000, integration_report, mechanism_report, stage_a_report,
    utility_study,
};
use recur64_v4::train::{UtilityLoss, probe_batch};

use crate::v3_p5::{Dev, dispatch, write_json};

const STACK_BYTES: usize = 512 * 1024 * 1024;

#[derive(Subcommand, Debug)]
pub enum V4Cmd {
    /// Describe the real `evidence_belief_v4` graph.
    ModelInfo(ModelInfoArgs),
    /// Generate, audit and seal `v4_tune_v1` (never evaluated here).
    TuneGen(TuneArgs),
    /// Re-verify custody: the V4 seal and the HOLDOUT_C digest. Evaluates nothing.
    TuneVerify(VerifyArgs),
    /// One resumable stage run on P25 TRAIN.
    Train(TrainArgs),
    /// The pre-registered mechanism measurements on V4_TRAIN_DEV.
    Measure(MeasureArgs),
    /// Timing and VRAM measurements.
    Bench(BenchArgs),
    /// CUDA correctness smoke: CPU/CUDA parity of the real graph, a CUDA update, B0 immutability.
    CudaSmoke(SmokeArgs),
    /// Apply the frozen Stage-B LR-screen rule to the screen measurement reports.
    SelectLr(SelectLrArgs),
    /// Apply the frozen Stage-C loss-selection rule to the two utility reports.
    SelectLoss(SelectLossArgs),
}

#[derive(Args, Debug, Clone)]
pub struct SmokeArgs {
    #[arg(long)]
    pub train: PathBuf,
    /// Positions per forward comparison.
    #[arg(long, default_value_t = 16)]
    pub positions: usize,
    #[arg(long)]
    pub output: PathBuf,
}

#[derive(Args, Debug)]
pub struct SelectLrArgs {
    /// Comma-separated `measure --kind b` reports, one per screened LR (same order as --lrs).
    #[arg(long)]
    pub reports: String,
    #[arg(long)]
    pub lrs: String,
    #[arg(long)]
    pub output: PathBuf,
}

#[derive(Args, Debug)]
pub struct SelectLossArgs {
    #[arg(long)]
    pub ranking: PathBuf,
    #[arg(long)]
    pub regression: PathBuf,
    #[arg(long)]
    pub output: PathBuf,
}

#[derive(Args, Debug, Clone)]
pub struct ModelInfoArgs {
    #[arg(long)]
    pub config: Option<PathBuf>,
    #[arg(long)]
    pub json: Option<PathBuf>,
}

#[derive(Args, Debug)]
pub struct TuneArgs {
    /// Comma-separated directories whose proof-*.json datasets are excluded.
    #[arg(long)]
    pub inventory_dirs: String,
    /// Directory for proof-v4-tune-v1.json (git-ignored run storage).
    #[arg(long)]
    pub data_dir: PathBuf,
    #[arg(long)]
    pub manifest: PathBuf,
    #[arg(long)]
    pub seal_output: PathBuf,
    /// HOLDOUT_C file (integrity check only; its canonical classes are already excluded).
    #[arg(long)]
    pub holdout_c: PathBuf,
    #[arg(long, default_value_t = 20)]
    pub threads: usize,
    #[arg(long)]
    pub no_regeneration_check: bool,
}

#[derive(Args, Debug)]
pub struct VerifyArgs {
    #[arg(long)]
    pub data: PathBuf,
    #[arg(long)]
    pub seal: PathBuf,
    #[arg(long)]
    pub holdout_c: PathBuf,
    #[arg(long)]
    pub output: PathBuf,
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
pub enum StageArg {
    A,
    B,
    C,
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
pub enum LossArg {
    Ranking,
    Regression,
}

#[derive(Args, Debug, Clone)]
pub struct TrainArgs {
    /// P25_DATA_V1 TRAIN (the only dataset V4 trains on).
    #[arg(long)]
    pub train: PathBuf,
    #[arg(long, value_enum)]
    pub stage: StageArg,
    #[arg(long, value_enum)]
    pub device: Dev,
    #[arg(long)]
    pub seed: u64,
    #[arg(long)]
    pub updates: u64,
    #[arg(long)]
    pub lr: f64,
    /// Stage C only.
    #[arg(long, value_enum)]
    pub loss: Option<LossArg>,
    /// The previous stage's run directory (stages B and C).
    #[arg(long)]
    pub init: Option<PathBuf>,
    #[arg(long)]
    pub config: Option<PathBuf>,
    #[arg(long)]
    pub run_dir: PathBuf,
    /// Checkpoint every this many updates.
    #[arg(long, default_value_t = 100)]
    pub checkpoint_every: u64,
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
pub enum Kind {
    /// Stage A: B0 on V4_TRAIN_DEV.
    A,
    /// Stage B: questions A, B, C, G.
    B,
    /// Stage C: questions D, E, F.
    C,
    /// Stage D: integration smoke (diagnostic).
    D,
}

#[derive(Args, Debug, Clone)]
pub struct MeasureArgs {
    #[arg(long)]
    pub train: PathBuf,
    #[arg(long, value_enum)]
    pub kind: Kind,
    #[arg(long, value_enum)]
    pub device: Dev,
    /// Comma-separated run directories (one per seed).
    #[arg(long)]
    pub run_dirs: String,
    #[arg(long)]
    pub config: Option<PathBuf>,
    #[arg(long, default_value_t = 32)]
    pub micro: usize,
    #[arg(long)]
    pub output: PathBuf,
}

#[derive(Args, Debug, Clone)]
pub struct BenchArgs {
    #[arg(long)]
    pub train: PathBuf,
    #[arg(long, value_enum)]
    pub device: Dev,
    #[arg(long)]
    pub config: Option<PathBuf>,
    #[arg(long, default_value_t = 16)]
    pub batch: usize,
    #[arg(long, default_value_t = 5)]
    pub repeats: usize,
    #[arg(long)]
    pub output: PathBuf,
}

pub fn run(cmd: V4Cmd) -> anyhow::Result<()> {
    match cmd {
        V4Cmd::ModelInfo(a) => model_info(&a),
        V4Cmd::TuneGen(a) => big_stack("v4-tune", move || tune_gen(a)),
        V4Cmd::TuneVerify(a) => verify(&a),
        V4Cmd::Train(a) => {
            let dev = a.device;
            dispatch(dev, move |d| match d {
                crate::v3_p5::Dispatch::Cpu => train::<recur64_model::train::CpuTrainBackend>(&a, false),
                #[cfg(feature = "cuda")]
                crate::v3_p5::Dispatch::Cuda => {
                    train::<burn::backend::Autodiff<burn::backend::Cuda>>(&a, true)
                }
            })
        }
        V4Cmd::Measure(a) => {
            let dev = a.device;
            dispatch(dev, move |d| match d {
                crate::v3_p5::Dispatch::Cpu => measure::<recur64_model::train::CpuTrainBackend>(&a),
                #[cfg(feature = "cuda")]
                crate::v3_p5::Dispatch::Cuda => {
                    measure::<burn::backend::Autodiff<burn::backend::Cuda>>(&a)
                }
            })
        }
        V4Cmd::CudaSmoke(a) => big_stack("v4-cuda-smoke", move || cuda_smoke(a)),
        V4Cmd::SelectLr(a) => select_lr(&a),
        V4Cmd::SelectLoss(a) => select_loss(&a),
        V4Cmd::Bench(a) => {
            let dev = a.device;
            dispatch(dev, move |d| match d {
                crate::v3_p5::Dispatch::Cpu => bench::<recur64_model::train::CpuTrainBackend>(&a, "cpu"),
                #[cfg(feature = "cuda")]
                crate::v3_p5::Dispatch::Cuda => {
                    bench::<burn::backend::Autodiff<burn::backend::Cuda>>(&a, "cuda")
                }
            })
        }
    }
}

fn big_stack(name: &str, f: impl FnOnce() -> anyhow::Result<()> + Send + 'static) -> anyhow::Result<()> {
    std::thread::Builder::new()
        .name(name.into())
        .stack_size(STACK_BYTES)
        .spawn(f)?
        .join()
        .map_err(|_| anyhow::anyhow!("{name} thread panicked"))?
}

fn load_cfg(path: Option<&Path>) -> anyhow::Result<ModelConfig> {
    let cfg = match path {
        None => ModelConfig::evidence_belief_v4(),
        Some(p) => ProbeConfig::from_toml_str(&std::fs::read_to_string(p)?)?.model,
    };
    cfg.validate()?;
    anyhow::ensure!(
        cfg.architecture == Architecture::EvidenceBeliefV4,
        "{}: not an evidence_belief_v4 configuration",
        path.map_or("(default)".into(), |p| p.display().to_string())
    );
    cfg.check_recurrence(1)?;
    Ok(cfg)
}

// ---------------------------------------------------------------------------------------
// model-info
// ---------------------------------------------------------------------------------------

fn model_info(a: &ModelInfoArgs) -> anyhow::Result<()> {
    let cfg = load_cfg(a.config.as_deref())?;
    let e = cfg.evidence.clone().expect("validated");
    let device = Default::default();
    let m = EvidenceBeliefModel::<burn::backend::Flex>::new(cfg.clone(), &device);
    let total = m.num_params();
    let (base, evidence, utility) = m.group_counts();
    println!("architecture    : {}", cfg.architecture.id());
    println!(
        "base tower      : width={} heads={} ffn={} blocks={} (root executed once; B0 ends at z0)",
        cfg.width, cfg.heads, cfg.ffn, cfg.core_blocks
    );
    println!(
        "hypotheses      : dim={} heads={} ffn={} blocks={} facts_hidden={} (one token per legal root action)",
        e.candidate.dim, e.candidate.heads, e.candidate.ffn, e.candidate.blocks, e.candidate.facts_hidden
    );
    println!(
        "evidence encoder: width={} heads={} ffn={} blocks={} message_dim={} (bias-free, content-multiplied)",
        e.content_dim, e.content_heads, e.content_ffn, e.content_blocks, e.message_dim
    );
    println!(
        "belief update   : pair_dim={} key_dim={} trust_hidden={} delta_bound={}",
        e.pair_dim, e.key_dim, e.trust_hidden, e.delta_bound
    );
    println!(
        "utility path    : parent encoder blocks={} heads={} ffn={}, head hidden={}",
        e.query_blocks, e.query_heads, e.query_ffn, e.utility_hidden
    );
    println!("\ncontracts: {:?}", e.contracts);
    println!("\nparameter group                      count");
    for (n, c) in m.param_breakdown() {
        println!("{n:34} {c:>12}");
    }
    println!("{:34} {:>12}", "TOTAL UNIQUE", total);
    println!("base {base}  evidence {evidence}  utility-path {utility}");
    println!("fp32 parameter bytes : {}", total * 4);
    println!("recurrence           : 1 (extra compute is new exact information: the query budget)");
    println!("query budgets        : 0..=16 with the same weights; parameter count is budget independent");
    if let Some(j) = a.json.as_deref() {
        write_json(
            j,
            &serde_json::json!({
                "architecture": cfg.architecture.id(),
                "total_params": total, "param_bytes_fp32": total * 4,
                "groups": {"base": base, "evidence": evidence, "utility_path": utility},
                "breakdown": m.param_breakdown().into_iter().map(|(n, c)| serde_json::json!({"name": n, "params": c})).collect::<Vec<_>>(),
                "contracts": e.contracts, "recurrence": 1,
            }),
        )?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------
// v4_tune_v1 custody
// ---------------------------------------------------------------------------------------

fn dirs(s: &str) -> Vec<PathBuf> {
    s.split(',').map(|d| PathBuf::from(d.trim())).collect()
}

fn tune_gen(a: TuneArgs) -> anyhow::Result<()> {
    let t0 = Instant::now();
    // HOLDOUT_C: integrity only. Its classes are part of the inventory below.
    let seal_c = verify_holdout_c(&a.holdout_c)?;
    let inv = inventory(&dirs(&a.inventory_dirs))?;
    anyhow::ensure!(
        inv.datasets.iter().any(|d| d.split == "holdout_c"),
        "the exclusion inventory does not contain HOLDOUT_C"
    );
    anyhow::ensure!(
        inv.datasets.iter().any(|d| d.path.ends_with("proof-v3-tune-v1.json")),
        "the exclusion inventory does not contain V3_TUNE_V1"
    );
    let spec = V4TuneSpec {
        per_cell: V4_TUNE_PER_CELL,
        seed: V4_TUNE_SEED,
        threads: a.threads,
        exclude_canon: inv.canons.clone(),
        exclude_fen: inv.fens.clone(),
    };
    let out = generate_v4_tune(&spec)?;
    let set = &out.set;
    anyhow::ensure!(set.positions.len() == V4_TUNE_POSITIONS, "unexpected size {}", set.positions.len());
    let audit = audit_targets(set, a.threads);
    anyhow::ensure!(
        audit.failures.is_empty(),
        "STOP: v4_tune_v1 audit failed on {} of {}; first: {}",
        audit.failures.len(),
        audit.checked,
        audit.failures[0]
    );
    let mut fens = std::collections::HashSet::new();
    let mut canons = std::collections::HashSet::new();
    for p in &set.positions {
        anyhow::ensure!(fens.insert(p.fen.as_str()), "{}: duplicate FEN inside v4_tune_v1", p.id);
        anyhow::ensure!(canons.insert(p.canon.as_str()), "{}: duplicate canonical class inside v4_tune_v1", p.id);
        anyhow::ensure!(!inv.canons.contains(&p.canon), "{}: canonical overlap with an earlier dataset", p.id);
        anyhow::ensure!(!inv.fens.contains(&p.fen), "{}: exact FEN overlap with an earlier dataset", p.id);
    }
    let mut disjoint_checked = 0usize;
    for d in &inv.datasets {
        let other = ProofTargets::load(Path::new(&d.path))?;
        let mut renamed = other.clone();
        renamed.split = Split::Train;
        let mut mine = set.clone();
        mine.split = Split::Tune;
        check_disjoint(&[&mine, &renamed]).map_err(anyhow::Error::msg)?;
        disjoint_checked += 1;
    }
    std::fs::create_dir_all(&a.data_dir)?;
    let path = a.data_dir.join("proof-v4-tune-v1.json");
    set.save(&path)?;
    let seal = seal_v4_tune(&path)?;
    anyhow::ensure!(seal.digest == set.digest, "saved dataset digest differs");
    let regen = if a.no_regeneration_check {
        serde_json::json!({"performed": false})
    } else {
        let again = generate_v4_tune(&spec)?;
        anyhow::ensure!(
            again.set.digest == set.digest,
            "STOP: regeneration from the same seed produced digest {} not {}",
            again.set.digest,
            set.digest
        );
        serde_json::json!({"performed": true, "identical_digest": true})
    };
    let manifest = serde_json::json!({
        "schema": "v4_tune_v1_manifest",
        "identity": V4_TUNE_IDENTITY,
        "purpose": "V4 final science only; sealed, never evaluated during P0/P1 development",
        "seed": V4_TUNE_SEED, "seed_hex": format!("{V4_TUNE_SEED:#x}"),
        "history_contract": set.history_contract,
        "per_cell_target": V4_TUNE_PER_CELL,
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
        "holdout_c": {"digest_verified": seal_c.actual_digest, "evaluated": false, "sealed": true},
        "deterministic_regeneration": regen,
        "wall_s": t0.elapsed().as_secs_f64(),
    });
    write_json(&a.manifest, &manifest)?;
    write_json(&a.seal_output, &serde_json::to_value(&seal)?)?;
    println!("v4_tune_v1: {} positions, digest {}", set.positions.len(), set.digest);
    for (k, v) in cell_counts(set) {
        println!("  {k}: {v}");
    }
    println!("exclusion manifest digest {}", inv.exclusion_manifest_digest);
    println!("sealed = true, evaluated = false; wrote {} and {}", a.manifest.display(), a.seal_output.display());
    Ok(())
}

fn verify(a: &VerifyArgs) -> anyhow::Result<()> {
    let seal = seal_v4_tune(&a.data)?;
    let recorded: recur64_runtime::proof::custody::V4TuneSeal =
        serde_json::from_slice(&std::fs::read(&a.seal)?)?;
    anyhow::ensure!(
        recorded.digest == seal.digest && recorded.sealed && !recorded.evaluated,
        "the recorded V4 seal does not match the data on disk (or records an evaluation)"
    );
    let c = verify_holdout_c(&a.holdout_c)?;
    write_json(
        &a.output,
        &serde_json::json!({
            "v4_tune_v1": {"digest": seal.digest, "positions": seal.positions, "sealed": true, "evaluated": false},
            "holdout_c": {"digest": c.actual_digest, "positions": c.positions, "sealed": true, "evaluated": false},
        }),
    )?;
    println!("v4_tune_v1 {} sealed/unevaluated; HOLDOUT_C {} sealed/unevaluated", seal.digest, c.actual_digest);
    Ok(())
}

// ---------------------------------------------------------------------------------------
// train
// ---------------------------------------------------------------------------------------

fn tag(gpu: bool) -> &'static str {
    if gpu { "v4-cuda" } else { "v4-cpu" }
}

fn train<B: AutodiffBackend>(a: &TrainArgs, gpu: bool) -> anyhow::Result<()> {
    let device = B::Device::default();
    model_io::verify_device::<B>(&device)?;
    let data = V4Data::load(&a.train)?;
    let cfg = load_cfg(a.config.as_deref())?;
    let stage = match a.stage {
        StageArg::A => Stage::A,
        StageArg::B => Stage::B,
        StageArg::C => Stage::C,
    };
    let (init_model, init_id) = match stage {
        Stage::A => {
            anyhow::ensure!(a.init.is_none(), "stage A starts from scratch; --init is refused");
            (EvidenceBeliefModel::<B>::new(cfg.clone(), &device), None)
        }
        _ => {
            let dir = a.init.as_ref().ok_or_else(|| anyhow::anyhow!("--init is required for stages B and C"))?;
            let (m, meta) = load_model::<B>(dir, &cfg, &device)?;
            (m, Some(meta.model_id))
        }
    };
    let mut recipe = Recipe::new(stage, a.seed, a.updates, a.lr, &data, &cfg, init_id);
    if stage == Stage::C {
        let l = a.loss.ok_or_else(|| anyhow::anyhow!("--loss is required for stage C"))?;
        recipe = recipe.with_loss(match l {
            LossArg::Ranking => UtilityLoss::Ranking,
            LossArg::Regression => UtilityLoss::Regression,
        });
    } else {
        anyhow::ensure!(a.loss.is_none(), "--loss is a stage C option");
    }
    let resuming = a.run_dir.join("v4-state.json").exists();
    let mut trainer = if resuming {
        Trainer::<B>::load(&a.run_dir, recipe.clone(), &device)?
    } else {
        if a.run_dir.exists() && std::fs::read_dir(&a.run_dir)?.next().is_some() {
            anyhow::bail!(
                "{}: a run directory without a valid state is refused (inspect and remove it)",
                a.run_dir.display()
            );
        }
        std::fs::create_dir_all(&a.run_dir)?;
        Trainer::<B>::new(recipe.clone(), init_model)?
    };
    eprintln!(
        "[v4 train] stage {} seed {} updates {} lr {:e} recipe {} resumed_at {}",
        stage.label(),
        a.seed,
        a.updates,
        a.lr,
        &recipe.digest()[..12],
        trainer.updates_done
    );
    let t0 = Instant::now();
    let mut peak_vram = 0u64;
    while trainer.updates_done < a.updates {
        let rec = trainer.step(&data, &device)?;
        if let Some((mem, _, _)) = sample_gpu().filter(|_| gpu && rec.update % 10 == 0) {
            peak_vram = peak_vram.max(mem);
        }
        if rec.update % 25 == 0 || trainer.updates_done == a.updates {
            eprintln!(
                "[v4 train] update {:>5} lr {:.2e} loss {:.4} detail {} wall {:.2}s total {:.0}s vram_peak {}",
                rec.update,
                rec.lr,
                rec.loss,
                rec.detail,
                rec.wall_s,
                t0.elapsed().as_secs_f64(),
                peak_vram
            );
        }
        if trainer.updates_done % a.checkpoint_every == 0 && trainer.updates_done < a.updates {
            trainer.save(&a.run_dir, tag(gpu))?;
        }
    }
    let model_id = trainer.save(&a.run_dir, tag(gpu))?;
    let hist = &trainer.history;
    let tail = hist.iter().rev().take(25).map(|h| h.loss).collect::<Vec<_>>();
    write_json(
        &a.run_dir.join("run-summary.json"),
        &serde_json::json!({
            "schema": "v4_run_summary_v1",
            "stage": stage.label(), "seed": a.seed, "updates": trainer.updates_done,
            "recipe_digest": recipe.digest(), "model_id": model_id,
            "init_model_id": recipe.init_model_id,
            "first_loss": hist.first().map(|h| h.loss), "last25_mean_loss": tail.iter().sum::<f64>() / tail.len().max(1) as f64,
            "wall_s_this_process": t0.elapsed().as_secs_f64(),
            "resumed": resuming, "peak_vram_mb_sampled": (gpu).then_some(peak_vram),
            "device": if gpu { "cuda" } else { "cpu" },
            "v4_tune_v1_evaluated": false, "holdout_c_evaluated": false,
        }),
    )?;
    println!("stage {} seed {} done: {} updates, model id {}", stage.label(), a.seed, trainer.updates_done, model_id);
    Ok(())
}

// ---------------------------------------------------------------------------------------
// measure
// ---------------------------------------------------------------------------------------

fn measure<B: AutodiffBackend>(a: &MeasureArgs) -> anyhow::Result<()> {
    let device = B::Device::default();
    model_io::verify_device::<B>(&device)?;
    let idev = Default::default();
    let data = V4Data::load(&a.train)?;
    let cfg = load_cfg(a.config.as_deref())?;
    let mut models: Vec<(u64, EvidenceBeliefModel<B::InnerBackend>)> = Vec::new();
    for d in a.run_dirs.split(',') {
        let (m, meta) = load_model::<B>(Path::new(d.trim()), &cfg, &device)?;
        models.push((meta.seed, inference::<B>(&m)));
    }
    let seed_models: Vec<SeedModel<'_, B::InnerBackend>> = models
        .iter()
        .map(|(s, m)| SeedModel { seed: *s, model: m })
        .collect();
    let t0 = Instant::now();
    let mut report = match a.kind {
        Kind::A => stage_a_report(seed_models[0].model, &data, a.micro, &idev)?,
        Kind::B => mechanism_report(&seed_models, &data, a.micro, &idev)?,
        Kind::D => integration_report(&seed_models, &data, a.micro, &idev)?,
        Kind::C => {
            let idx = dev1000(&data);
            let mut per_seed = Vec::new();
            let mut noise_exact = true;
            for m in &seed_models {
                per_seed.push((m.seed, collect_probes(m.model, &data, &idx, a.micro.min(16), &idev)?));
                // Label noise: an identical probe repeated must give an identical label.
                let first: Vec<usize> = idx.iter().take(50).copied().collect();
                let (_, x) = probe_batch(m.model, &data, &first, 1, 8, 99, &idev)?;
                let (_, y) = probe_batch(m.model, &data, &first, 1, 8, 99, &idev)?;
                noise_exact &= x.iter().zip(&y).all(|(p, q)| p.u == q.u);
            }
            utility_study(&per_seed, noise_exact)?
        }
    };
    report["run_dirs"] = a.run_dirs.clone().into();
    report["wall_s"] = t0.elapsed().as_secs_f64().into();
    report["data"] = serde_json::json!({
        "train_digest": data.targets.digest, "fit_digest": data.fit_digest, "dev_digest": data.dev_digest,
        "fit": data.fit.len(), "dev": data.dev.len(),
        "v4_tune_v1_evaluated": false, "holdout_c_evaluated": false,
    });
    write_json(&a.output, &report)?;
    println!("wrote {}", a.output.display());
    Ok(())
}

// ---------------------------------------------------------------------------------------
// bench
// ---------------------------------------------------------------------------------------

fn bench<B: AutodiffBackend>(a: &BenchArgs, device_label: &str) -> anyhow::Result<()> {
    let device = B::Device::default();
    model_io::verify_device::<B>(&device)?;
    let data = V4Data::load(&a.train)?;
    let cfg = load_cfg(a.config.as_deref())?;
    let model = EvidenceBeliefModel::<B>::new(cfg.clone(), &device);
    let inf = inference::<B>(&model);
    let idev = Default::default();
    let idx: Vec<usize> = dev1000(&data).into_iter().take(a.batch).collect();
    let roots = data.roots(&idx)?;
    let time = |f: &mut dyn FnMut() -> anyhow::Result<()>| -> anyhow::Result<f64> {
        let _ = <B::InnerBackend as Backend>::sync(&idev);
        let t = Instant::now();
        for _ in 0..a.repeats {
            f()?;
        }
        let _ = <B::InnerBackend as Backend>::sync(&idev);
        Ok(t.elapsed().as_secs_f64() / a.repeats as f64)
    };
    // warm-up (kernel compilation / autotune)
    inf.run(&roots, &RunOptions::new(2), Selection::Fixed, 0, &idev)?;
    let mut rows = Vec::new();
    let mut b0_secs = 0.0;
    for budget in [0usize, 1, 2, 4, 8] {
        let (secs, samples) = monitor(device_label == "cuda", || {
            time(&mut || {
                inf.run(&roots, &RunOptions::new(budget), Selection::Fixed, 0, &idev).map(|_| ())
            })
        });
        let secs = secs?;
        rows.push(serde_json::json!({
            "budget": budget, "batch": a.batch, "seconds_per_batch": secs,
            "seconds_per_example": secs / a.batch as f64,
            "marginal_seconds_per_query_step_vs_b0": if budget == 0 { None } else { Some((secs - b0_secs) / budget as f64) },
            "gpu": samples,
        }));
        if budget == 0 {
            b0_secs = secs;
        }
    }
    let utility_rows = {
        let secs = time(&mut || {
            inf.run(&roots, &RunOptions::new(4), Selection::Utility, 0, &idev).map(|_| ())
        })?;
        serde_json::json!({"budget": 4, "selection": "utility", "seconds_per_batch": secs})
    };
    let probe = {
        let secs = time(&mut || probe_batch(&inf, &data, &idx, 2, 8, 7, &idev).map(|_| ()))?;
        serde_json::json!({"prefix": 2, "k": 8, "batch": a.batch, "seconds_per_batch": secs,
                           "probe_queries_per_second": (a.batch * 8) as f64 / secs})
    };
    // Training step cost and VRAM plateau at the three stage shapes.
    let mut training = Vec::new();
    for (stage, updates, lr) in [(Stage::A, 3u64, 3e-4), (Stage::B, 3, 3e-4)] {
        let m = EvidenceBeliefModel::<B>::new(cfg.clone(), &device);
        let init_id = (stage != Stage::A).then(|| "bench".to_string());
        let recipe = Recipe::new(stage, 1, updates, lr, &data, &cfg, init_id);
        let mut t = Trainer::<B>::new(recipe, m)?;
        let (res, samples) = monitor(device_label == "cuda", || -> anyhow::Result<Vec<f64>> {
            let mut walls = Vec::new();
            for _ in 0..updates {
                walls.push(t.step(&data, &device)?.wall_s);
            }
            Ok(walls)
        });
        training.push(serde_json::json!({"stage": stage.label(), "update_wall_s": res?, "gpu": samples}));
    }
    let doc = serde_json::json!({
        "schema": "v4_bench_v1", "device": device_label, "parameters": inf.num_params(),
        "inference": rows, "utility_selection": utility_rows, "probe": probe, "training": training,
        "note": "inference rows are per batch of positions with fixed_bfs selection; gpu.util_busy_mean is the host/device overhead indicator (low = host-bound)",
    });
    write_json(&a.output, &doc)?;
    println!("wrote {}", a.output.display());
    Ok(())
}

// ---------------------------------------------------------------------------------------
// frozen selection rules (docs/V4_EXPERIMENTS.md V4-E2)
// ---------------------------------------------------------------------------------------

fn read_json(p: &Path) -> anyhow::Result<serde_json::Value> {
    Ok(serde_json::from_slice(&std::fs::read(p)?)?)
}

/// B8 CE of the first (only) model under one schedule of a `measure --kind b` report.
fn b8_ce(report: &serde_json::Value, sched: &str) -> anyhow::Result<f64> {
    let table = report["per_schedule"][sched]["table"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("report has no per_schedule.{sched}.table"))?;
    let row = table
        .iter()
        .find(|r| r["budget"] == 8)
        .ok_or_else(|| anyhow::anyhow!("report has no B8 row"))?;
    row["per_seed"][0]["normal"]["ce"]
        .as_f64()
        .ok_or_else(|| anyhow::anyhow!("report has no B8 CE"))
}

/// Lowest mean `V4_TRAIN_DEV` CE at B8 over the FIXED and RANDOM schedules; ties go to the lower LR.
fn select_lr(a: &SelectLrArgs) -> anyhow::Result<()> {
    let reports: Vec<&str> = a.reports.split(',').map(str::trim).collect();
    let lrs: Vec<f64> = a
        .lrs
        .split(',')
        .map(|s| s.trim().parse::<f64>())
        .collect::<Result<_, _>>()?;
    anyhow::ensure!(reports.len() == lrs.len() && !lrs.is_empty(), "one report per LR");
    let mut rows = Vec::new();
    for (r, &lr) in reports.iter().zip(&lrs) {
        let doc = read_json(Path::new(r))?;
        let score = (b8_ce(&doc, "fixed")? + b8_ce(&doc, "random")?) / 2.0;
        rows.push((lr, score, (*r).to_string()));
    }
    let best = rows
        .iter()
        .min_by(|x, y| {
            x.1.partial_cmp(&y.1)
                .expect("finite")
                .then(x.0.partial_cmp(&y.0).expect("finite"))
        })
        .expect("non-empty");
    write_json(
        &a.output,
        &serde_json::json!({
            "rule": "lowest mean V4_TRAIN_DEV CE at B8 over FIXED and RANDOM; ties to the lower LR",
            "candidates": rows.iter().map(|(lr, s, r)| serde_json::json!({"lr": lr, "mean_ce_b8": s, "report": r})).collect::<Vec<_>>(),
            "selected_lr": best.0,
        }),
    )?;
    println!("selected stage-B LR {:e} (mean B8 CE {:.4})", best.0, best.1);
    Ok(())
}

/// Higher pooled DEV1000 per-state Spearman wins; within 0.02 the ranking loss is chosen.
fn select_loss(a: &SelectLossArgs) -> anyhow::Result<()> {
    let sp = |p: &Path| -> anyhow::Result<f64> {
        read_json(p)?["E"]["spearman_per_state"]["mean"]
            .as_f64()
            .ok_or_else(|| anyhow::anyhow!("{}: no E.spearman_per_state.mean", p.display()))
    };
    let (rk, rg) = (sp(&a.ranking)?, sp(&a.regression)?);
    let chosen = if rg - rk >= 0.02 { "regression" } else { "ranking" };
    write_json(
        &a.output,
        &serde_json::json!({
            "rule": "higher pooled DEV1000 mean per-state Spearman; if within 0.02 choose ranking",
            "spearman_ranking": rk, "spearman_regression": rg, "selected_loss": chosen,
        }),
    )?;
    println!("selected utility loss: {chosen} (spearman ranking {rk:.4}, regression {rg:.4})");
    Ok(())
}

// ---------------------------------------------------------------------------------------
// CUDA correctness smoke
// ---------------------------------------------------------------------------------------

#[cfg(not(feature = "cuda"))]
fn cuda_smoke(_a: SmokeArgs) -> anyhow::Result<()> {
    anyhow::bail!("CUDA support is not compiled; rebuild with --features cuda (no CPU substitution is made)")
}

/// The real V4 graph on `Autodiff<Cuda>`: CPU/CUDA forward parity from identical weights at B0 and
/// B4, a CUDA backward + AdamW update with the base detached, B0 bit-identical on the GPU after
/// it, and a CUDA checkpoint round trip. Nothing is substituted: a missing GPU is an error.
#[cfg(feature = "cuda")]
fn cuda_smoke(a: SmokeArgs) -> anyhow::Result<()> {
    use recur64_model::checkpoint::{CheckpointMeta, save_training};
    use recur64_model::train::{CpuTrainBackend, adamw};
    type C = burn::backend::Autodiff<burn::backend::Cuda>;
    type G = CpuTrainBackend;
    let t0 = Instant::now();
    let cdev = Default::default();
    model_io::verify_device::<C>(&cdev)?;
    let gdev = Default::default();
    let vram_start = sample_gpu();
    let data = V4Data::load(&a.train)?;
    let cfg = ModelConfig::evidence_belief_v4();
    cfg.validate()?;
    let idx: Vec<usize> = dev1000(&data).into_iter().take(a.positions).collect();
    let roots = data.roots(&idx)?;

    // Identical weights on both devices: save on the CPU backend, load on CUDA.
    let tmp = std::env::temp_dir().join(format!("recur64-v4-cuda-smoke-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    let cpu_model = EvidenceBeliefModel::<G>::new(cfg.clone(), &gdev);
    let meta = CheckpointMeta::new(cfg.clone(), 1, false, 0, 3e-4, 1, 0, "v4-cpu", "fp32");
    save_training::<G, _, _>(&tmp.join("checkpoint"), &cpu_model, &adamw::<G, EvidenceBeliefModel<G>>(), &meta)?;
    let (cuda_model, _) = load_model::<C>(&tmp, &cfg, &cdev)?;
    let cpu_inf = inference::<G>(&cpu_model);
    let cuda_inf = inference::<C>(&cuda_model);
    let idev = Default::default();

    let valid: Vec<usize> = idx.iter().map(|&i| data.position(i).legal.len()).collect();
    let mut parity = Vec::new();
    for budget in [0usize, 4] {
        let c = cpu_inf.run(&roots, &RunOptions::new(budget), Selection::Fixed, 0, &gdev)?;
        let g = cuda_inf.run(&roots, &RunOptions::new(budget), Selection::Fixed, 0, &idev)?;
        let w = c.logits.dims()[1];
        let (cz, gz) = (
            c.logits.into_data().to_vec::<f32>().unwrap(),
            g.logits.into_data().to_vec::<f32>().unwrap(),
        );
        let mut max_abs = 0.0f32;
        let mut argmax_equal = true;
        for (r, &vw) in valid.iter().enumerate() {
            let (a_, b_) = (&cz[r * w..r * w + vw], &gz[r * w..r * w + vw]);
            for (x, y) in a_.iter().zip(b_) {
                max_abs = max_abs.max((x - y).abs());
            }
            let am = |v: &[f32]| v.iter().enumerate().fold(0, |m, (i, &x)| if x > v[m] { i } else { m });
            argmax_equal &= am(a_) == am(b_);
        }
        parity.push(serde_json::json!({"budget": budget, "positions": idx.len(), "max_abs_logit_difference": max_abs,
            "argmax_equal_everywhere": argmax_equal, "tolerance": 2e-3, "pass": max_abs <= 2e-3 && argmax_equal}));
    }

    // CUDA update with the base detached; B0 must be bit-identical afterwards.
    let b0 = |m: &EvidenceBeliefModel<<C as AutodiffBackend>::InnerBackend>| -> anyhow::Result<Vec<f64>> {
        Ok(recur64_v4::train::eval_policy(m, &data, &idx, 0, recur64_v4::train::EvalSel::Fixed, 8, &idev)?.ce)
    };
    let recipe = Recipe::new(Stage::B, 1, 3, 3e-4, &data, &cfg, Some("cuda-smoke".into()));
    let mut trainer = Trainer::<C>::new(recipe.clone(), cuda_model)?;
    let before = b0(&inference::<C>(&trainer.model))?;
    let mut losses = Vec::new();
    for _ in 0..3 {
        losses.push(trainer.step(&data, &cdev)?.loss);
    }
    let after = b0(&inference::<C>(&trainer.model))?;
    let b0_identical = before == after;
    let moved = {
        let o = inference::<C>(&trainer.model).run(&roots, &RunOptions::new(4), Selection::Fixed, 0, &idev)?;
        o.delta.into_data().to_vec::<f32>().unwrap().iter().any(|d| d.abs() > 0.0)
    };

    // Checkpoint round trip on CUDA.
    let rt = tmp.join("rt");
    std::fs::create_dir_all(&rt)?;
    trainer.save(&rt, "v4-cuda")?;
    let loaded = Trainer::<C>::load(&rt, recipe, &cdev)?;
    let rt_ok = b0(&inference::<C>(&loaded.model))? == after && loaded.updates_done == 3;
    let vram_end = sample_gpu();
    let _ = std::fs::remove_dir_all(&tmp);
    let pass = parity.iter().all(|p| p["pass"] == true)
        && b0_identical
        && moved
        && rt_ok
        && losses.iter().all(|l| l.is_finite());
    write_json(
        &a.output,
        &serde_json::json!({
            "schema": "v4_cuda_smoke_v1",
            "device": "cuda (Autodiff<Cuda>), FP32 (no TF32)",
            "parameters": cuda_inf.num_params(),
            "forward_parity_cpu_vs_cuda": parity,
            "cuda_stage_b_update": {"losses": losses, "b0_bit_identical_after_updates": b0_identical, "evidence_moved": moved},
            "checkpoint_round_trip_on_cuda": rt_ok,
            "vram_mb_start_end": [vram_start.map(|v| v.0), vram_end.map(|v| v.0)],
            "pass": pass, "wall_s": t0.elapsed().as_secs_f64(),
            "tested": "the real graph ran on the CUDA device; parity checked against the CPU backend with identical weights",
        }),
    )?;
    println!("v4 cuda smoke: pass = {pass} (wall {:.1}s)", t0.elapsed().as_secs_f64());
    anyhow::ensure!(pass, "CUDA smoke failed; see {}", a.output.display());
    Ok(())
}
