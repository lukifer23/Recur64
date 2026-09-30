//! `recur64 x15 gen-targets | audit-targets` - the fixed-data reasoning set.
//!
//! The teacher is a Recur64 probe checkpoint searched with deterministic PUCT
//! (no root noise, one leaf in flight). Positions come from real self-play
//! replay games as `start_fen + exact prefix`, so repetition state is exact.

use std::path::PathBuf;
use std::time::Instant;

use burn::prelude::*;
use clap::Args;

use recur64_runtime::evaluator::SyncEvaluator;
use recur64_runtime::reasoning_targets::{
    self as rt, Provenance, ReasoningTargetsV1, TeacherContract, select_positions,
};
use recur64_runtime::replay::ReplayReader;
use recur64_runtime::{RunConfig, model_io};

#[derive(Args, Debug)]
pub struct GenTargetsArgs {
    /// RunConfig of the teacher's run (model geometry, recurrence, device).
    #[arg(long)]
    pub teacher_config: PathBuf,
    /// Teacher checkpoint directory.
    #[arg(long)]
    pub teacher_checkpoint: PathBuf,
    /// Replay directory to draw exact-history positions from.
    #[arg(long)]
    pub replay: PathBuf,
    #[arg(long)]
    pub out: PathBuf,
    #[arg(long, default_value_t = 32)]
    pub positions: usize,
    /// Simulation ladder, shallowest first (last = deep teacher).
    #[arg(long, value_delimiter = ',', default_value = "16,32,64,128")]
    pub ladder: Vec<u32>,
    #[arg(long, default_value_t = 20260929)]
    pub seed: u64,
    #[arg(long, default_value_t = 1.0)]
    pub c_puct: f32,
    /// Worker threads sharing the one teacher model (labels are identical for any count).
    #[arg(long, default_value_t = 4)]
    pub threads: usize,
}

#[derive(Args, Debug)]
pub struct AuditTargetsArgs {
    #[arg(long)]
    pub targets: PathBuf,
}

fn generate<B: Backend>(cfg: &RunConfig, args: &GenTargetsArgs) -> anyhow::Result<()> {
    anyhow::ensure!(
        args.ladder.windows(2).all(|w| w[0] < w[1]) && !args.ladder.is_empty(),
        "ladder must be strictly increasing"
    );
    let device: B::Device = Default::default();
    let model = model_io::load::<B>(&args.teacher_checkpoint, &cfg.model, &device)?;
    let model_id = std::fs::read(args.teacher_checkpoint.join("meta.json"))
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .and_then(|v| {
            v.get("model_id")
                .and_then(|m| m.as_str())
                .map(str::to_owned)
        })
        .ok_or_else(|| anyhow::anyhow!("teacher checkpoint has no model_id"))?;
    let evaluator = SyncEvaluator::new(model, cfg.recurrence, device);

    let games = ReplayReader::open(&args.replay)?.read_all_games()?;
    anyhow::ensure!(!games.is_empty(), "replay has no games");
    let candidates = select_positions(&games, args.positions, args.seed)?;
    let teacher = TeacherContract {
        checkpoint: args.teacher_checkpoint.display().to_string(),
        model_id,
        architecture: "probe_v1".into(),
        recurrence: cfg.recurrence,
        c_puct: args.c_puct,
        leaves_in_flight: 1,
        root_noise: false,
        ladder: args.ladder.clone(),
    };
    println!(
        "teacher {} (model_id {}...), ladder {:?}, {} positions from {} replay games",
        teacher.checkpoint,
        &teacher.model_id[..12.min(teacher.model_id.len())],
        teacher.ladder,
        candidates.len(),
        games.len()
    );
    let start = Instant::now();
    let positions = rt::label(
        candidates,
        &evaluator,
        &teacher,
        args.seed,
        args.threads,
        &|done, total| {
            if done % 8 == 0 || done == total {
                println!(
                    "  labelled {done}/{total} ({:.0}s)",
                    start.elapsed().as_secs_f64()
                );
            }
        },
    )?;
    let provenance = Provenance {
        git_rev: recur64_runtime::provenance::git_revision()
            .unwrap_or("unknown")
            .into(),
        created_unix_s: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
    };
    let targets = ReasoningTargetsV1::new(teacher, args.seed, provenance, positions);
    let n = rt::audit(&targets)?;
    targets.save(&args.out)?;
    let by_cat = |c: &str| targets.positions.iter().filter(|p| p.category == c).count();
    let val = targets
        .positions
        .iter()
        .filter(|p| p.split == "val")
        .count();
    println!(
        "wrote {} ({} positions audited move-for-move); digest {}",
        args.out.display(),
        n,
        targets.digest
    );
    println!(
        "  categories: {}; split: train {} / val {}",
        rt::CATEGORIES
            .iter()
            .map(|c| format!("{c}={}", by_cat(c)))
            .collect::<Vec<_>>()
            .join(" "),
        targets.positions.len() - val,
        val
    );
    Ok(())
}

pub fn run_gen(args: GenTargetsArgs) -> anyhow::Result<()> {
    let cfg = RunConfig::from_toml_str(&std::fs::read_to_string(&args.teacher_config)?)?;
    cfg.ensure_supported()?;
    match cfg.device.as_str() {
        "cpu" => generate::<burn::backend::Flex>(&cfg, &args),
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                // In-process scoped sampler (joined on exit): the RTX's own
                // utilization, so "is the GPU actually working" is a number.
                let (result, gpu) = recur64_runtime::gpu_telemetry::monitor(true, || {
                    generate::<burn::backend::Cuda>(&cfg, &args)
                });
                println!(
                    "gpu telemetry   : peak_vram={:?} MiB util_max={:?}% util_busy_mean={:?} temp_max={:?} C ({} samples)",
                    gpu.peak_vram_mb, gpu.util_max, gpu.util_busy_mean, gpu.temp_max_c, gpu.samples
                );
                result
            }
            #[cfg(not(feature = "cuda"))]
            {
                anyhow::bail!("CUDA support is not compiled; rebuild with --features cuda")
            }
        }
        other => anyhow::bail!("unsupported device {other:?}"),
    }
}

pub fn run_audit(args: AuditTargetsArgs) -> anyhow::Result<()> {
    let targets = ReasoningTargetsV1::load(&args.targets)?;
    let n = rt::audit(&targets)?;
    println!(
        "OK: {n} positions reconstruct move-for-move; digest {}",
        targets.digest
    );
    Ok(())
}
