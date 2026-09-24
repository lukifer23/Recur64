//! Phase 2 commands: selfplay | replay-audit | train | arena | run | report.

use std::path::{Path, PathBuf};

use burn::tensor::backend::AutodiffBackend;
use clap::Args;

use recur64_eval::{ArenaConfig, run_arena as eval_run_arena};
use recur64_model::train::adamw;
use recur64_runtime::replay::{ReplayReader, audit_dir};
use recur64_runtime::{
    CancelToken, RunConfig, RunDir, SyncEvaluator, collect_only, model_io, read_metadata,
    run as run_coordinator,
};

type CpuTrain = burn::backend::Autodiff<burn::backend::Flex>;

fn load_config(path: &Path) -> anyhow::Result<RunConfig> {
    let text = std::fs::read_to_string(path)?;
    RunConfig::from_toml_str(&text)
}

/// Hardware-scheduling overrides. These let a sweep vary scheduling parameters
/// from the command line without editing scientific config files. They override
/// only hardware scheduling keys, never model geometry or search settings.
#[derive(Args, Debug, Default, Clone)]
pub struct ScheduleOverrides {
    /// Number of self-play games to launch.
    #[arg(long)]
    pub active_games: Option<u32>,
    /// Number of self-play worker threads.
    #[arg(long)]
    pub cpu_workers: Option<usize>,
    /// Maximum inference batch size.
    #[arg(long)]
    pub max_inference_batch: Option<usize>,
    /// Batch coalescing timeout in microseconds.
    #[arg(long)]
    pub batch_timeout_us: Option<u64>,
    /// Simulations per move (search budget).
    #[arg(long)]
    pub simulations_per_move: Option<u32>,
    /// Ply cap before a game is truncated.
    #[arg(long)]
    pub ply_cap: Option<u32>,
}

impl ScheduleOverrides {
    fn apply(&self, cfg: &mut RunConfig) {
        if let Some(v) = self.active_games {
            cfg.active_games = v;
        }
        if let Some(v) = self.cpu_workers {
            cfg.cpu_workers = v;
        }
        if let Some(v) = self.max_inference_batch {
            cfg.max_inference_batch = v;
        }
        if let Some(v) = self.batch_timeout_us {
            cfg.batch_timeout_us = v;
        }
        if let Some(v) = self.simulations_per_move {
            cfg.simulations_per_move = v;
        }
        if let Some(v) = self.ply_cap {
            cfg.ply_cap = v;
        }
    }
}

#[derive(Args, Debug)]
pub struct SelfplayArgs {
    #[arg(long)]
    pub config: PathBuf,
    #[arg(long)]
    pub output: PathBuf,
    #[command(flatten)]
    pub schedule: ScheduleOverrides,
}

#[derive(Args, Debug)]
pub struct ReplayAuditArgs {
    #[arg(long)]
    pub input: PathBuf,
}

#[derive(Args, Debug)]
pub struct TrainArgs {
    #[arg(long)]
    pub config: PathBuf,
    #[arg(long)]
    pub replay: PathBuf,
    /// Starting checkpoint directory.
    #[arg(long)]
    pub checkpoint: PathBuf,
    /// Output candidate checkpoint directory.
    #[arg(long)]
    pub output: PathBuf,
}

#[derive(Args, Debug)]
pub struct ArenaArgs {
    #[arg(long)]
    pub config: PathBuf,
    #[arg(long)]
    pub reference: PathBuf,
    #[arg(long)]
    pub candidate: PathBuf,
    #[arg(long)]
    pub output: PathBuf,
}

#[derive(Args, Debug)]
pub struct RunArgs {
    #[arg(long)]
    pub config: PathBuf,
    #[arg(long)]
    pub run_dir: PathBuf,
    #[arg(long, default_value_t = false)]
    pub force: bool,
    #[command(flatten)]
    pub schedule: ScheduleOverrides,
}

#[derive(Args, Debug)]
pub struct ReportArgs {
    #[arg(long)]
    pub run_dir: PathBuf,
}

// --- backend-generic implementations ---

fn selfplay_impl<B: AutodiffBackend>(cfg: &RunConfig, output: &Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(output)?;
    let cancel = CancelToken::new();
    let metrics = collect_only::<B>(cfg, output, &cancel)?;
    println!(
        "wrote {} games ({} plies) to {}",
        metrics.games,
        metrics.plies,
        output.display()
    );
    println!("{}", serde_json::to_string_pretty(&metrics)?);
    Ok(())
}

fn train_impl<B: AutodiffBackend>(
    cfg: &RunConfig,
    replay: &Path,
    checkpoint: &Path,
    output: &Path,
) -> anyhow::Result<()> {
    let device: B::Device = Default::default();
    let games = ReplayReader::open(replay)?.read_all_games()?;
    let model = model_io::load::<B>(checkpoint, &cfg.model, &device)?;
    let mut optim = adamw::<B, _>();
    let learner_cfg = recur64_runtime::LearnerConfig {
        batch_size: cfg.train_batch,
        accumulation_steps: cfg.accumulation_steps,
        max_updates: cfg.max_updates,
        lr: cfg.lr,
        warmup_updates: cfg.resolved_warmup(),
        planned_updates: cfg.resolved_planned_updates(),
        start_update: 0,
        recurrence: cfg.recurrence,
        seed: cfg.seed,
        deadline: None,
    };
    let (trained, report) =
        recur64_runtime::train_from_games(model, &mut optim, &games, &learner_cfg, &device)
            .map_err(anyhow::Error::msg)?;
    let mut meta = recur64_model::checkpoint::CheckpointMeta::new(
        cfg.model.clone(),
        cfg.recurrence,
        false,
        report.updates as u64,
        cfg.lr,
        cfg.seed,
        0,
        format!("{} ({})", cfg.device, cfg.precision),
        cfg.precision.clone(),
    );
    meta.run_id = cfg.run_id.clone();
    meta.update_counter = report.updates as u64;
    recur64_model::checkpoint::save_training(output, &trained, &optim, &meta)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn arena_impl<B: AutodiffBackend>(
    cfg: &RunConfig,
    reference: &Path,
    candidate: &Path,
    output: &Path,
) -> anyhow::Result<()> {
    let device: <B::InnerBackend as burn::tensor::backend::BackendTypes>::Device =
        Default::default();
    let ref_model = model_io::load::<B::InnerBackend>(reference, &cfg.model, &device)?;
    let cand_model = model_io::load::<B::InnerBackend>(candidate, &cfg.model, &device)?;
    let ref_ev = SyncEvaluator::new(ref_model, cfg.recurrence, device.clone());
    let cand_ev = SyncEvaluator::new(cand_model, cfg.recurrence, device);
    let openings = match &cfg.opening_suite {
        Some(p) => recur64_eval::OpeningSuite::load(std::path::Path::new(p))?.openings,
        None => Vec::new(),
    };
    let arena_cfg = ArenaConfig {
        games: cfg.arena_games,
        simulations: cfg.simulations_per_move,
        c_puct: cfg.c_puct,
        recurrence: cfg.recurrence,
        ply_cap: cfg.ply_cap,
        seed: cfg.seed,
        openings,
    };
    let result = eval_run_arena(
        &ref_ev,
        &cand_ev,
        &reference.display().to_string(),
        &candidate.display().to_string(),
        &arena_cfg,
    )?;
    std::fs::create_dir_all(output)?;
    std::fs::write(
        output.join("arena.json"),
        serde_json::to_vec_pretty(&result)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

fn run_impl<B: AutodiffBackend>(
    cfg: &RunConfig,
    run_dir: &Path,
    force: bool,
) -> anyhow::Result<()> {
    let cancel = CancelToken::new();
    cancel.install_handler()?;
    let dir = RunDir::create(run_dir, force)?;
    let report = run_coordinator::<B>(cfg, &dir, &cancel)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    if report.status != "completed" {
        anyhow::bail!("run finished with status {}", report.status);
    }
    Ok(())
}

// --- command entry points ---

pub fn run_selfplay(args: SelfplayArgs) -> anyhow::Result<()> {
    let mut cfg = load_config(&args.config)?;
    args.schedule.apply(&mut cfg);
    cfg.ensure_supported()?;
    match cfg.device.as_str() {
        "cpu" => selfplay_impl::<CpuTrain>(&cfg, &args.output),
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                selfplay_impl::<burn::backend::Autodiff<burn::backend::Cuda>>(&cfg, &args.output)
            }
            #[cfg(not(feature = "cuda"))]
            {
                anyhow::bail!("CUDA support is not compiled; rebuild with --features cuda")
            }
        }
        other => anyhow::bail!("unknown device '{other}'"),
    }
}

pub fn run_replay_audit(args: ReplayAuditArgs) -> anyhow::Result<()> {
    let report = audit_dir(&args.input)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    if !report.ok() {
        anyhow::bail!("replay audit failed with {} errors", report.errors.len());
    }
    Ok(())
}

pub fn run_train(args: TrainArgs) -> anyhow::Result<()> {
    let cfg = load_config(&args.config)?;
    cfg.ensure_supported()?;
    match cfg.device.as_str() {
        "cpu" => train_impl::<CpuTrain>(&cfg, &args.replay, &args.checkpoint, &args.output),
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                train_impl::<burn::backend::Autodiff<burn::backend::Cuda>>(
                    &cfg,
                    &args.replay,
                    &args.checkpoint,
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

pub fn run_arena(args: ArenaArgs) -> anyhow::Result<()> {
    let cfg = load_config(&args.config)?;
    cfg.ensure_supported()?;
    match cfg.device.as_str() {
        "cpu" => arena_impl::<CpuTrain>(&cfg, &args.reference, &args.candidate, &args.output),
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                arena_impl::<burn::backend::Autodiff<burn::backend::Cuda>>(
                    &cfg,
                    &args.reference,
                    &args.candidate,
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

pub fn run_run(args: RunArgs) -> anyhow::Result<()> {
    let mut cfg = load_config(&args.config)?;
    args.schedule.apply(&mut cfg);
    cfg.ensure_supported()?;
    match cfg.device.as_str() {
        "cpu" => run_impl::<CpuTrain>(&cfg, &args.run_dir, args.force),
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                run_impl::<burn::backend::Autodiff<burn::backend::Cuda>>(
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

pub fn run_report(args: ReportArgs) -> anyhow::Result<()> {
    let meta = read_metadata(&args.run_dir)?;
    println!("{}", serde_json::to_string_pretty(&meta)?);
    let report_path = args.run_dir.join("report").join("report.json");
    if report_path.exists() {
        println!("{}", std::fs::read_to_string(report_path)?);
    } else {
        println!("(no report yet)");
    }
    Ok(())
}
