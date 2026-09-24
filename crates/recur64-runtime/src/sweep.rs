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
use crate::coordinator::{SelfPlayMetrics, collect_parallel, selfplay_metrics};
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

/// Main workstation coarse schedule candidates (RTX 2000 Ada 16 GB, 24c/24t).
///
/// Search is sequential within a game, so each game has at most one leaf in
/// flight and a batch can never exceed `active_games`. A cap at or above the
/// concurrency therefore does not bind; this coarse pass varies concurrency
/// with a non-binding cap. Binding-cap and timeout variants are run afterwards
/// as single cells around the leaders. Candidate values are derived from the
/// workstation CPU topology and the Phase 3 Stage A priors; they are a starting
/// matrix, not a frozen schedule.
pub fn workstation_grid() -> Vec<SweepCellSpec> {
    [
        (16, 16, 500),
        (24, 24, 500),
        (32, 32, 500),
        (48, 48, 500),
        (64, 64, 500),
        (96, 96, 500),
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
    /// Scientific identity after applying this cell's search override.
    pub scientific_config_hash: String,
    /// Full runtime identity after applying every cell override.
    pub resolved_config_hash: String,
    pub games: u64,
    pub requested_games: u64,
    pub peak_in_flight: usize,
    pub errors: u64,
    pub terminations: std::collections::BTreeMap<String, u64>,
    /// All-ply target health (kept for comparability with earlier cells).
    pub mean_target_entropy: f64,
    pub mean_top1_visit_share: f64,
    pub positions: u64,
    pub trainable_positions: u64,
    pub trainable_positions_per_sec: f64,
    /// Full data-health record: W/D/L, lengths, all/trainable target health.
    pub selfplay: SelfPlayMetrics,
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
    /// Mean GPU utilization over samples with utilization > 0.
    pub gpu_util_busy_mean: Option<f64>,
    pub gpu_util_max: Option<u64>,
    pub gpu_temp_max_c: Option<u64>,
    pub gpu_samples: u64,
}

#[derive(Default)]
struct GpuSamples {
    peak_vram_mb: Option<u64>,
    busy_util_sum: u64,
    busy_samples: u64,
    util_max: Option<u64>,
    temp_max: Option<u64>,
    samples: u64,
}

/// Warm the inference path for a range of batch sizes; returns elapsed seconds.
pub fn warmup<B: AutodiffBackend>(
    cfg: &RunConfig,
    max_batch: usize,
    device: &B::Device,
) -> anyhow::Result<f64> {
    let inner_device: Device<B::InnerBackend> = Default::default();
    <B::InnerBackend as Backend>::seed(&inner_device, cfg.seed);
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

/// Sample (memory.used MiB, utilization %, temperature C) via nvidia-smi.
/// `None` when nvidia-smi is unavailable; the report then shows no GPU data.
fn sample_gpu() -> Option<(u64, u64, u64)> {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.used,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    let mut fields = s
        .lines()
        .next()?
        .split(',')
        .map(|f| f.trim().parse::<u64>());
    Some((
        fields.next()?.ok()?,
        fields.next()?.ok()?,
        fields.next()?.ok()?,
    ))
}

impl GpuSamples {
    fn record(&mut self, (mem, util, temp): (u64, u64, u64)) {
        self.samples += 1;
        self.peak_vram_mb = Some(self.peak_vram_mb.map_or(mem, |p| p.max(mem)));
        self.util_max = Some(self.util_max.map_or(util, |p| p.max(util)));
        self.temp_max = Some(self.temp_max.map_or(temp, |p| p.max(temp)));
        if util > 0 {
            self.busy_util_sum += util;
            self.busy_samples += 1;
        }
    }
}

fn cell_config(cfg: &RunConfig, cell: SweepCellSpec, games: u64) -> anyhow::Result<RunConfig> {
    let mut cell_cfg = cfg.clone();
    cell_cfg.games_per_cycle = Some(u32::try_from(games)?);
    cell_cfg.concurrent_games = Some(cell.active_games);
    cell_cfg.cpu_workers = cell.active_games as usize;
    cell_cfg.max_inference_batch = cell.max_batch;
    cell_cfg.batch_timeout_us = cell.batch_timeout_us;
    cell_cfg.simulations_per_move = cell.simulations;
    Ok(cell_cfg)
}

/// Run one sweep cell: `games` self-play games, measuring throughput.
pub fn run_cell<B: AutodiffBackend>(
    cfg: &RunConfig,
    cell: SweepCellSpec,
    games: u64,
    checkpoint: Option<&std::path::Path>,
) -> anyhow::Result<SweepCellResult> {
    let inner_device: Device<B::InnerBackend> = Default::default();
    <B::InnerBackend as Backend>::seed(&inner_device, cfg.seed);
    let model = match checkpoint {
        Some(path) => model_io::load::<B::InnerBackend>(path, &cfg.model, &inner_device)?,
        None => model_io::build::<B::InnerBackend>(&cfg.model, &inner_device),
    };
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
    let cell_cfg = cell_config(cfg, cell, games)?;
    let scientific_config_hash = cell_cfg.scientific_config_hash()?;
    let resolved_config_hash = cell_cfg.resolved_config_hash();
    let sampling = std::sync::atomic::AtomicBool::new(true);
    let gpu = std::sync::Mutex::new(GpuSamples::default());
    if let Some(v) = sample_gpu() {
        gpu.lock().expect("GPU sampler mutex").record(v);
    }
    let collect_start = Instant::now();
    let records = std::thread::scope(|scope| {
        let monitor = scope.spawn(|| {
            while sampling.load(std::sync::atomic::Ordering::Relaxed) {
                if let Some(v) = sample_gpu() {
                    gpu.lock().expect("GPU sampler mutex").record(v);
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
    let collect_secs = collect_start.elapsed().as_secs_f64().max(1e-6);
    let m = owner.metrics().snapshot();
    owner.shutdown();
    let selfplay = selfplay_metrics(&cell_cfg, &records, m.clone());
    let positions = selfplay.plies;
    let gpu = gpu.into_inner().expect("GPU sampler mutex");

    Ok(SweepCellResult {
        active_games: cell.active_games,
        max_batch: cell.max_batch,
        batch_timeout_us: cell.batch_timeout_us,
        simulations: cell.simulations,
        scientific_config_hash,
        resolved_config_hash,
        games: records.len() as u64,
        requested_games: games,
        peak_in_flight: m.peak_in_flight,
        errors: m.errors,
        terminations: selfplay.terminations.clone(),
        mean_target_entropy: selfplay.target_health.all.mean_entropy,
        mean_top1_visit_share: selfplay.target_health.all.mean_top1_visit_share,
        positions,
        trainable_positions: selfplay.trainable_positions,
        trainable_positions_per_sec: selfplay.trainable_positions as f64 / collect_secs,
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
        peak_vram_mb: gpu.peak_vram_mb,
        gpu_util_busy_mean: (gpu.busy_samples > 0)
            .then(|| gpu.busy_util_sum as f64 / gpu.busy_samples as f64),
        gpu_util_max: gpu.util_max,
        gpu_temp_max_c: gpu.temp_max,
        gpu_samples: gpu.samples,
        selfplay,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cell_hashes_name_effective_overrides() {
        let base = RunConfig::from_toml_str(include_str!("../../../configs/smoke.toml"))
            .expect("parse smoke config");
        let schedule_a = SweepCellSpec {
            active_games: 8,
            max_batch: 8,
            batch_timeout_us: 500,
            simulations: 16,
        };
        let schedule_b = SweepCellSpec {
            active_games: 16,
            max_batch: 16,
            batch_timeout_us: 1000,
            simulations: 16,
        };
        let search_b = SweepCellSpec {
            simulations: 32,
            ..schedule_b
        };
        let a = cell_config(&base, schedule_a, 16).expect("resolve cell a");
        let b = cell_config(&base, schedule_b, 16).expect("resolve cell b");
        let search = cell_config(&base, search_b, 16).expect("resolve search cell");

        assert_eq!(
            a.scientific_config_hash().unwrap(),
            b.scientific_config_hash().unwrap(),
            "hardware scheduling must not change scientific identity"
        );
        assert_ne!(a.resolved_config_hash(), b.resolved_config_hash());
        assert_ne!(
            b.scientific_config_hash().unwrap(),
            search.scientific_config_hash().unwrap(),
            "search budget must change scientific identity"
        );
        assert_ne!(b.resolved_config_hash(), search.resolved_config_hash());
    }
}
