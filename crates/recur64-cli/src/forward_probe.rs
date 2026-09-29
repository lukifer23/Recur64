//! `recur64 forward-probe` — raw network outputs and forward latency on real
//! positions.
//!
//! Reconstructs the first `--positions` plies of a replay, then:
//! - writes the network's policy and WDL for every position
//!   (`outputs.json`), so two builds (e.g. with and without kernel fusion)
//!   can be compared for numerical parity on identical inputs;
//! - times the batched forward (upload, forward, readback: the inference
//!   owner's `evaluate_batch`) at each `--batches` size after warmup
//!   (`timing.json`), the forward-latency-vs-batch curve.
//!
//! Diagnostic only: no search, no training.

use std::path::PathBuf;
use std::time::Instant;

use burn::tensor::backend::AutodiffBackend;
use clap::Args;

use recur64_core::{ActionId, GameState, StandardMove};
use recur64_runtime::replay::{ReplayReader, sampler::example_for_ply};
use recur64_runtime::{BatchEvaluator, BatchedModel, RunConfig, model_io};

type CpuTrain = burn::backend::Autodiff<burn::backend::Flex>;

#[derive(Args, Debug)]
pub struct ForwardProbeArgs {
    #[arg(long)]
    pub config: PathBuf,
    #[arg(long)]
    pub checkpoint: PathBuf,
    #[arg(long)]
    pub replay: PathBuf,
    #[arg(long)]
    pub output: PathBuf,
    /// Positions taken from the start of the replay (in game-id order).
    #[arg(long, default_value_t = 1024)]
    pub positions: usize,
    /// Batch sizes to time (comma-separated).
    #[arg(long, default_value = "1,8,16,32,64,128,256")]
    pub batches: String,
    /// Timed repetitions per batch size (after 3 warmup calls).
    #[arg(long, default_value_t = 20)]
    pub reps: usize,
}

#[derive(serde::Serialize)]
struct Output {
    policy: Vec<f32>,
    wdl: [f32; 3],
}

#[derive(serde::Serialize)]
struct Timing {
    batch: usize,
    reps: usize,
    mean_ms: f64,
    p50_ms: f64,
    min_ms: f64,
    max_ms: f64,
    positions_per_sec: f64,
}

fn run_impl<B: AutodiffBackend>(cfg: &RunConfig, args: &ForwardProbeArgs) -> anyhow::Result<()> {
    let mut games = ReplayReader::open(&args.replay)?.read_all_games()?;
    games.sort_by_key(|g| g.game_id);
    let mut positions = Vec::with_capacity(args.positions);
    'games: for game in &games {
        let mut state = GameState::from_fen(&game.start_fen)?;
        for ply in &game.plies {
            if positions.len() >= args.positions {
                break 'games;
            }
            let ex = example_for_ply(&state, game.outcome.unwrap_or(1), ply)
                .map_err(|e| anyhow::anyhow!("game {}: {e}", game.game_id))?;
            positions.push((ex.observation, ex.legal));
            let id = ActionId::from_index(ply.selected as u32)?;
            let (from, to, promo) = id.to_physical(state.perspective());
            let promotion = if promo.is_none() { None } else { Some(promo) };
            state.apply(StandardMove::new(from, to, promotion))?;
        }
    }
    anyhow::ensure!(!positions.is_empty(), "replay has no positions");

    let device = Default::default();
    let model = model_io::load::<B::InnerBackend>(&args.checkpoint, &cfg.model, &device)?;
    let batched = BatchedModel::new(model, cfg.recurrence, device);

    // Outputs in fixed batches of 64, the same inputs for every build.
    let mut outputs = Vec::with_capacity(positions.len());
    for chunk in positions.chunks(64) {
        let obs: Vec<_> = chunk.iter().map(|p| p.0.clone()).collect();
        let legal: Vec<_> = chunk.iter().map(|p| p.1.clone()).collect();
        let out = batched
            .evaluate_batch(&obs, &legal)
            .map_err(|e| anyhow::anyhow!("forward failed: {e}"))?;
        outputs.extend(out.into_iter().map(|r| Output {
            policy: r.policy,
            wdl: r.wdl,
        }));
    }

    let sizes: Vec<usize> = args
        .batches
        .split(',')
        .map(|s| s.trim().parse())
        .collect::<Result<_, _>>()?;
    let mut timings = Vec::new();
    for &size in &sizes {
        anyhow::ensure!(size >= 1, "batch sizes must be >= 1");
        // Cycle through the positions so every call sees real, varied inputs.
        let batch_at = |k: usize| {
            let idx = (0..size).map(|i| (k * size + i) % positions.len());
            let obs: Vec<_> = idx.clone().map(|i| positions[i].0.clone()).collect();
            let legal: Vec<_> = idx.map(|i| positions[i].1.clone()).collect();
            (obs, legal)
        };
        for k in 0..3 {
            let (obs, legal) = batch_at(k);
            batched
                .evaluate_batch(&obs, &legal)
                .map_err(|e| anyhow::anyhow!("forward failed: {e}"))?;
        }
        let mut ms = Vec::with_capacity(args.reps);
        for k in 0..args.reps {
            let (obs, legal) = batch_at(k + 3);
            let t = Instant::now();
            batched
                .evaluate_batch(&obs, &legal)
                .map_err(|e| anyhow::anyhow!("forward failed: {e}"))?;
            ms.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let mean = ms.iter().sum::<f64>() / ms.len() as f64;
        let mut sorted = ms.clone();
        sorted.sort_by(f64::total_cmp);
        let t = Timing {
            batch: size,
            reps: ms.len(),
            mean_ms: mean,
            p50_ms: sorted[sorted.len() / 2],
            min_ms: sorted[0],
            max_ms: sorted[sorted.len() - 1],
            positions_per_sec: size as f64 / (mean / 1e3),
        };
        println!(
            "batch {:>4}: mean {:>7.2} ms  p50 {:>7.2}  min {:>7.2}  max {:>7.2}  -> {:>8.0} pos/s",
            t.batch, t.mean_ms, t.p50_ms, t.min_ms, t.max_ms, t.positions_per_sec
        );
        timings.push(t);
    }

    std::fs::create_dir_all(&args.output)?;
    std::fs::write(
        args.output.join("outputs.json"),
        serde_json::to_vec(&outputs)?,
    )?;
    std::fs::write(
        args.output.join("timing.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "checkpoint": args.checkpoint,
            "positions": positions.len(),
            "git_revision": option_env!("RECUR64_GIT_SHA"),
            "timings": timings,
        }))?,
    )?;
    println!(
        "wrote {} outputs and {} timings to {}",
        outputs.len(),
        timings.len(),
        args.output.display()
    );
    Ok(())
}

pub fn run(args: ForwardProbeArgs) -> anyhow::Result<()> {
    let cfg = RunConfig::from_toml_str(&std::fs::read_to_string(&args.config)?)?;
    cfg.ensure_supported()?;
    match cfg.device.as_str() {
        "cpu" => run_impl::<CpuTrain>(&cfg, &args),
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                run_impl::<burn::backend::Autodiff<burn::backend::Cuda>>(&cfg, &args)
            }
            #[cfg(not(feature = "cuda"))]
            {
                anyhow::bail!("CUDA support is not compiled; rebuild with --features cuda")
            }
        }
        other => anyhow::bail!("unknown device '{other}'"),
    }
}
