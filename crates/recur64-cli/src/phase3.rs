//! Phase 3 commands: gen-openings | eval-policy | pilot.

use std::path::PathBuf;

use burn::tensor::backend::AutodiffBackend;
use clap::Args;

use recur64_eval::{OpeningSuite, generate_openings};
use recur64_model::checkpoint::{CheckpointMeta, save_training};
use recur64_model::train::adamw;
use recur64_runtime::{
    CancelToken, RunConfig, RunDir, RunMetadata, SyncEvaluator, eval_policy, model_io, run_pilot,
};

type CpuTrain = burn::backend::Autodiff<burn::backend::Flex>;

#[derive(Args, Debug)]
pub struct GenOpeningsArgs {
    #[arg(long, default_value = "configs/openings-v1.toml")]
    pub output: PathBuf,
    #[arg(long, default_value_t = 12)]
    pub count: usize,
    #[arg(long, default_value_t = 6)]
    pub plies: usize,
    #[arg(long, default_value_t = 20260923)]
    pub seed: u64,
}

#[derive(Args, Debug)]
pub struct EvalPolicyArgs {
    #[arg(long)]
    pub config: PathBuf,
    /// Checkpoint directory for the model being evaluated.
    #[arg(long)]
    pub checkpoint: PathBuf,
    #[arg(long)]
    pub output: PathBuf,
    #[arg(long, default_value_t = 40)]
    pub games: u32,
}

#[derive(Args, Debug)]
pub struct PilotArgs {
    #[arg(long)]
    pub config: PathBuf,
    #[arg(long)]
    pub run_dir: PathBuf,
    #[arg(long, default_value_t = false)]
    pub force: bool,
}

#[derive(Args, Debug)]
pub struct FreezeReferenceArgs {
    #[arg(long)]
    pub config: PathBuf,
    #[arg(long)]
    pub output: PathBuf,
}

fn freeze_reference_impl<B: AutodiffBackend>(
    cfg: &RunConfig,
    output: &std::path::Path,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        !output.exists(),
        "reference checkpoint already exists: {}",
        output.display()
    );
    let device: B::Device = Default::default();
    B::seed(&device, cfg.seed);
    let model = model_io::build::<B>(&cfg.model, &device);
    let optim = adamw::<B, _>();
    let mut meta = CheckpointMeta::new(
        cfg.model.clone(),
        cfg.recurrence,
        false,
        0,
        cfg.lr,
        cfg.seed,
        0,
        format!("{} ({})", cfg.device, cfg.precision),
        cfg.precision.clone(),
    );
    meta.run_id = cfg.run_id.clone();
    meta.git_revision = RunMetadata::new(cfg).git_revision;
    save_training(output, &model, &optim, &meta)?;
    std::fs::write(output.join("config.toml"), toml::to_string_pretty(cfg)?)?;
    let saved: CheckpointMeta = serde_json::from_slice(&std::fs::read(output.join("meta.json"))?)?;
    println!(
        "reference model_id={} seed={} git={:?} scientific_hash={} resolved_hash={}",
        saved.model_id,
        cfg.seed,
        saved.git_revision,
        cfg.scientific_config_hash(),
        cfg.resolved_config_hash()
    );
    Ok(())
}

pub fn run_freeze_reference(args: FreezeReferenceArgs) -> anyhow::Result<()> {
    let cfg = RunConfig::from_toml_str(&std::fs::read_to_string(args.config)?)?;
    cfg.ensure_supported()?;
    if let Some(path) = &cfg.opening_suite {
        OpeningSuite::load(std::path::Path::new(path))?;
    }
    match cfg.device.as_str() {
        "cpu" => freeze_reference_impl::<CpuTrain>(&cfg, &args.output),
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                freeze_reference_impl::<burn::backend::Autodiff<burn::backend::Cuda>>(
                    &cfg,
                    &args.output,
                )
            }
            #[cfg(not(feature = "cuda"))]
            {
                anyhow::bail!("CUDA support is not compiled; rebuild with --features cuda")
            }
        }
        other => anyhow::bail!("unknown device '{other}'"),
    }
}

pub fn run_gen_openings(args: GenOpeningsArgs) -> anyhow::Result<()> {
    let openings = generate_openings(args.count, args.plies, args.seed);
    let suite = OpeningSuite {
        version: 1,
        provenance: format!(
            "Recur64 deterministic legal prefixes: count={}, plies={}, seed={}",
            args.count, args.plies, args.seed
        ),
        openings,
    };
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    suite.save(&args.output)?;
    println!(
        "wrote {} openings to {}",
        suite.len(),
        args.output.display()
    );
    Ok(())
}

fn eval_policy_impl<B: AutodiffBackend>(
    cfg: &RunConfig,
    checkpoint: &std::path::Path,
    output: &std::path::Path,
    games: u32,
) -> anyhow::Result<()> {
    let device: B::Device = Default::default();
    let model = model_io::load::<B>(checkpoint, &cfg.model, &device)?;
    let ev = SyncEvaluator::new(model, cfg.recurrence, device);
    let openings = match &cfg.opening_suite {
        Some(p) => OpeningSuite::load(std::path::Path::new(p))?.openings,
        None => vec![recur64_core::GameState::startpos().to_fen()],
    };
    let result = eval_policy::raw_policy_vs_random(
        &ev,
        games,
        cfg.temperature,
        cfg.ply_cap,
        cfg.seed,
        &openings,
    )?;
    std::fs::create_dir_all(output)?;
    std::fs::write(
        output.join("eval-policy.json"),
        serde_json::to_vec_pretty(&result)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

pub fn run_eval_policy(args: EvalPolicyArgs) -> anyhow::Result<()> {
    let cfg = RunConfig::from_toml_str(&std::fs::read_to_string(&args.config)?)?;
    match cfg.device.as_str() {
        "cpu" => eval_policy_impl::<CpuTrain>(&cfg, &args.checkpoint, &args.output, args.games),
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                eval_policy_impl::<burn::backend::Autodiff<burn::backend::Cuda>>(
                    &cfg,
                    &args.checkpoint,
                    &args.output,
                    args.games,
                )
            }
            #[cfg(not(feature = "cuda"))]
            {
                anyhow::bail!("CUDA support is not compiled; rebuild with --features cuda")
            }
        }
        other => anyhow::bail!("unknown device '{other}'"),
    }
}

fn pilot_impl<B: AutodiffBackend>(
    cfg: &RunConfig,
    run_dir: &std::path::Path,
    force: bool,
) -> anyhow::Result<()> {
    let cancel = CancelToken::new();
    cancel.install_handler()?;
    let dir = RunDir::create(run_dir, force)?;
    let report = run_pilot::<B>(cfg, &dir, &cancel)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

pub fn run_pilot_cmd(args: PilotArgs) -> anyhow::Result<()> {
    let cfg = RunConfig::from_toml_str(&std::fs::read_to_string(&args.config)?)?;
    match cfg.device.as_str() {
        "cpu" => pilot_impl::<CpuTrain>(&cfg, &args.run_dir, args.force),
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                pilot_impl::<burn::backend::Autodiff<burn::backend::Cuda>>(
                    &cfg,
                    &args.run_dir,
                    args.force,
                )
            }
            #[cfg(not(feature = "cuda"))]
            {
                anyhow::bail!("CUDA support is not compiled; rebuild with --features cuda")
            }
        }
        other => anyhow::bail!("unknown device '{other}'"),
    }
}
