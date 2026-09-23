//! Stage A profiling: deliberate CUDA warmup and a bounded batching sweep.
//!
//! Measures self-play throughput and batcher behavior across
//! (active_games, max_batch, batch_timeout, simulations) cells. Warmup cost is
//! recorded separately so cold JIT/autotune never distorts steady-state numbers.

use std::time::{Duration, Instant};

use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

use recur64_core::{ActionId, GameState, ObservationV1};
use recur64_search::{Rng, SelfPlayConfig, play_game_from};

use crate::config::RunConfig;
use crate::inference::{BatchEvaluator, BatchedModel, InferenceConfig, InferenceOwner};
use crate::model_io;

/// One sweep cell.
#[derive(Debug, Clone, Copy)]
pub struct SweepCellSpec {
    pub active_games: u32,
    pub max_batch: usize,
    pub batch_timeout_us: u64,
    pub simulations: u32,
}

/// The default grids. `full` is still bounded; obvious pathological cells are
/// pruned by the caller if they OOM.
pub fn grid(small: bool) -> Vec<SweepCellSpec> {
    let mut out = Vec::new();
    if small {
        for active_games in [32, 64, 128] {
            for max_batch in [32, 64] {
                out.push(SweepCellSpec {
                    active_games,
                    max_batch,
                    batch_timeout_us: 500,
                    simulations: 8,
                });
            }
        }
    } else {
        for active_games in [32, 64, 128, 256] {
            for max_batch in [16, 32, 64, 128] {
                out.push(SweepCellSpec {
                    active_games,
                    max_batch,
                    batch_timeout_us: 500,
                    simulations: 16,
                });
            }
        }
    }
    out
}

/// A measured sweep result.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SweepCellResult {
    pub active_games: u32,
    pub max_batch: usize,
    pub batch_timeout_us: u64,
    pub simulations: u32,
    pub games: u64,
    pub positions: u64,
    pub requests: u64,
    pub batches: u64,
    pub batch_mean: f64,
    pub batch_p50: u32,
    pub batch_p95: u32,
    pub batch_max: u64,
    pub queue_wait_us_p50: u64,
    pub queue_wait_us_p95: u64,
    pub forward_us_mean: f64,
    pub collect_secs: f64,
    pub games_per_hour: f64,
    pub positions_per_sec: f64,
    pub peak_vram_mb: Option<u64>,
}

/// Warm the inference path for a range of batch sizes; returns elapsed seconds.
pub fn warmup<B: AutodiffBackend>(
    cfg: &RunConfig,
    max_batch: usize,
    device: &B::Device,
) -> anyhow::Result<f64> {
    let inner_device: Device<B::InnerBackend> = Default::default();
    let model = model_io::build::<B::InnerBackend>(&cfg.model, &inner_device);
    let batched = BatchedModel::new(model, cfg.recurrence, inner_device);
    let legal: Vec<ActionId> = (0..20).map(|i| ActionId::from_index(i).unwrap()).collect();
    let obs = ObservationV1::zeroed();

    let start = Instant::now();
    let mut size = 1usize;
    while size <= max_batch.max(1) {
        let observations: Vec<ObservationV1> = (0..size).map(|_| obs.clone()).collect();
        let legal_lists: Vec<Vec<ActionId>> = (0..size).map(|_| legal.clone()).collect();
        let _ = batched.evaluate_batch(&observations, &legal_lists);
        size *= 2;
    }
    let _ = device;
    Ok(start.elapsed().as_secs_f64())
}

/// Sample peak VRAM (MiB) via nvidia-smi, best-effort.
fn sample_vram_mb() -> Option<u64> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    s.lines().next()?.trim().parse::<u64>().ok()
}

/// Run one sweep cell: `games` self-play games, measuring throughput.
pub fn run_cell<B: AutodiffBackend>(
    cfg: &RunConfig,
    cell: SweepCellSpec,
    games: u64,
) -> anyhow::Result<SweepCellResult> {
    let inner_device: Device<B::InnerBackend> = Default::default();
    let model = model_io::build::<B::InnerBackend>(&cfg.model, &inner_device);
    let batched = BatchedModel::new(model, cfg.recurrence, inner_device);
    let owner = InferenceOwner::spawn(
        batched,
        InferenceConfig {
            max_batch: cell.max_batch,
            batch_timeout: Duration::from_micros(cell.batch_timeout_us),
            ..InferenceConfig::default()
        },
    );
    let ev = owner.evaluator();
    let sp = SelfPlayConfig {
        simulations_per_move: cell.simulations,
        c_puct: cfg.c_puct,
        temperature: cfg.temperature,
        ply_cap: cfg.ply_cap,
        recurrence: cfg.recurrence,
    };

    // `active_games` is the concurrency: one game per thread, so that many
    // leaf-evaluation requests are in flight and the batcher can coalesce them.
    let concurrency = (cell.active_games as usize).max(1);
    let next = std::sync::atomic::AtomicU64::new(0);
    let positions = std::sync::atomic::AtomicU64::new(0);

    let mut peak_vram = sample_vram_mb();
    let collect_start = Instant::now();
    std::thread::scope(|scope| {
        for _ in 0..concurrency {
            let ev = &ev;
            let next = &next;
            let positions = &positions;
            let start_fen = cfg.start_fen.clone();
            let seed = cfg.seed;
            scope.spawn(move || {
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    if i >= games {
                        break;
                    }
                    let mut rng = Rng::new(seed.wrapping_add(i));
                    let start = match &start_fen {
                        Some(f) => match GameState::from_fen(f) {
                            Ok(s) => s,
                            Err(_) => break,
                        },
                        None => GameState::startpos(),
                    };
                    if let Ok(game) = play_game_from(ev, &sp, &mut rng, start) {
                        positions.fetch_add(
                            game.plies.len() as u64,
                            std::sync::atomic::Ordering::SeqCst,
                        );
                    }
                }
            });
        }
    });
    let positions = positions.load(std::sync::atomic::Ordering::SeqCst);
    if let Some(v) = sample_vram_mb() {
        peak_vram = Some(peak_vram.map_or(v, |p| p.max(v)));
    }
    let collect_secs = collect_start.elapsed().as_secs_f64().max(1e-6);
    let m = owner.metrics().snapshot();
    owner.shutdown();

    Ok(SweepCellResult {
        active_games: cell.active_games,
        max_batch: cell.max_batch,
        batch_timeout_us: cell.batch_timeout_us,
        simulations: cell.simulations,
        games,
        positions,
        requests: m.submitted,
        batches: m.batches,
        batch_mean: m.batch_size_mean,
        batch_p50: m.batch_size_p50,
        batch_p95: m.batch_size_p95,
        batch_max: m.batch_size_max,
        queue_wait_us_p50: m.queue_wait_us_p50,
        queue_wait_us_p95: m.queue_wait_us_p95,
        forward_us_mean: m.forward_us_mean,
        collect_secs,
        games_per_hour: games as f64 / collect_secs * 3600.0,
        positions_per_sec: positions as f64 / collect_secs,
        peak_vram_mb: peak_vram,
    })
}
