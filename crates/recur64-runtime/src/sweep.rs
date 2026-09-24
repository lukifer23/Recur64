//! Stage A profiling: deliberate CUDA warmup and a bounded batching sweep.
//!
//! Measures self-play throughput and batcher behavior across
//! (active_games, max_batch, batch_timeout, simulations) cells. Warmup cost is
//! recorded separately so cold JIT/autotune never distorts steady-state numbers.

use std::time::{Duration, Instant};

use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

use recur64_core::{ActionId, ObservationV1};

use crate::cancel::CancelToken;
use crate::config::RunConfig;
use crate::coordinator::collect_parallel;
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

/// HP RTX 2050 schedule candidates; same game count for every cell.
pub fn hp_grid() -> Vec<SweepCellSpec> {
    [
        (6, 8, 500),
        (8, 16, 500),
        (12, 16, 500),
        (12, 32, 500),
        (12, 32, 1000),
        (12, 32, 2000),
        (16, 32, 1000),
        (16, 64, 1000),
        (24, 32, 1000),
        (24, 64, 2000),
    ]
    .into_iter()
    .map(
        |(active_games, max_batch, batch_timeout_us)| SweepCellSpec {
            active_games,
            max_batch,
            batch_timeout_us,
            simulations: 16,
        },
    )
    .collect()
}

/// A measured sweep result.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SweepCellResult {
    pub active_games: u32,
    pub max_batch: usize,
    pub batch_timeout_us: u64,
    pub simulations: u32,
    pub games: u64,
    pub requested_games: u64,
    pub peak_in_flight: usize,
    pub errors: u64,
    pub terminations: std::collections::BTreeMap<String, u64>,
    pub mean_target_entropy: f64,
    pub mean_top1_visit_share: f64,
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
    pub evaluations_per_sec: f64,
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
        batched
            .evaluate_batch(&observations, &legal_lists)
            .map_err(|e| anyhow::anyhow!("GPU warmup failed at batch {size}: {e}"))?;
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
    let mut cell_cfg = cfg.clone();
    cell_cfg.games_per_cycle = Some(u32::try_from(games)?);
    cell_cfg.concurrent_games = Some(cell.active_games);
    cell_cfg.cpu_workers = cell.active_games as usize;
    cell_cfg.simulations_per_move = cell.simulations;
    let sampling = std::sync::atomic::AtomicBool::new(true);
    let peak_vram = std::sync::Mutex::new(sample_vram_mb());
    let collect_start = Instant::now();
    let records = std::thread::scope(|scope| {
        let monitor = scope.spawn(|| {
            while sampling.load(std::sync::atomic::Ordering::Relaxed) {
                if let Some(v) = sample_vram_mb() {
                    let mut peak = peak_vram.lock().expect("VRAM sampler mutex");
                    *peak = Some(peak.map_or(v, |p| p.max(v)));
                }
                std::thread::sleep(Duration::from_millis(500));
            }
        });
        let deadline = Instant::now() + Duration::from_secs(cfg.run_budget_minutes.max(1) * 60);
        let result = collect_parallel(&cell_cfg, &ev, &CancelToken::new(), deadline, 0);
        sampling.store(false, std::sync::atomic::Ordering::Relaxed);
        monitor.join().expect("VRAM sampler thread");
        result
    })?;
    let positions = records.iter().map(|g| g.plies.len() as u64).sum::<u64>();
    let mut terminations = std::collections::BTreeMap::new();
    let mut entropy = 0.0;
    let mut top1 = 0.0f64;
    for game in &records {
        *terminations.entry(game.termination.clone()).or_insert(0) += 1;
        for ply in &game.plies {
            entropy -= ply
                .target
                .iter()
                .map(|(_, p)| {
                    let p = *p as f64;
                    if p > 0.0 { p * p.ln() } else { 0.0 }
                })
                .sum::<f64>();
            top1 += ply
                .target
                .iter()
                .map(|(_, p)| *p as f64)
                .fold(0.0, f64::max);
        }
    }
    let collect_secs = collect_start.elapsed().as_secs_f64().max(1e-6);
    let m = owner.metrics().snapshot();
    owner.shutdown();

    Ok(SweepCellResult {
        active_games: cell.active_games,
        max_batch: cell.max_batch,
        batch_timeout_us: cell.batch_timeout_us,
        simulations: cell.simulations,
        games: records.len() as u64,
        requested_games: games,
        peak_in_flight: m.peak_in_flight,
        errors: m.errors,
        terminations,
        mean_target_entropy: entropy / positions.max(1) as f64,
        mean_top1_visit_share: top1 / positions.max(1) as f64,
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
        games_per_hour: records.len() as f64 / collect_secs * 3600.0,
        positions_per_sec: positions as f64 / collect_secs,
        evaluations_per_sec: m.completed as f64 / collect_secs,
        peak_vram_mb: *peak_vram.lock().expect("VRAM sampler mutex"),
    })
}
