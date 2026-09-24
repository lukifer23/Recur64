//! `recur64 bench-lifecycle` — GPU inference-owner lifecycle probe.
//!
//! Repeats owner creation (model load from disk), batched evaluation and
//! owner shutdown, and records VRAM before / peak / after plus latency per
//! repetition, to detect memory or latency accumulation across lifecycles.
//! No training happens. Modes:
//!
//! - `one`: one owner plays a self-arena and raw-vs-random.
//! - `two`: two simultaneous owners play the arena and raw-vs-parent.
//! - `pilot`: the pilot's own `evaluate_candidate` with parent == reference
//!   (pre-promotion shape).
//! - `pilot-promoted`: `evaluate_candidate` with the parent labelled distinct
//!   from the reference, forcing the post-promotion longitudinal arena.
//!
//! All models are the same frozen checkpoint, so every game is a real
//! workload but no result here is a strength measurement.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use clap::Args;

use recur64_eval::{ArenaConfig, OpeningSuite, run_arena};
use recur64_runtime::eval_policy::{raw_policy_vs_parent, raw_policy_vs_random};
use recur64_runtime::gpu_telemetry::{self, GpuSamples};
use recur64_runtime::{EvalModels, MetricsSnapshot, RunConfig, evaluate_candidate, spawn_owner};

type CpuTrain = burn::backend::Autodiff<burn::backend::Flex>;

#[derive(Args, Debug)]
pub struct BenchLifecycleArgs {
    #[arg(long)]
    pub config: PathBuf,
    /// Frozen checkpoint used for every logical model.
    #[arg(long)]
    pub checkpoint: PathBuf,
    #[arg(long)]
    pub output: PathBuf,
    /// Comma-separated modes: one, two, pilot, pilot-promoted.
    #[arg(long, default_value = "one,two,pilot,pilot-promoted")]
    pub modes: String,
    #[arg(long, default_value_t = 8)]
    pub reps: u32,
    /// Overrides the config's arena_games (raw matches use max(4, this)).
    #[arg(long, default_value_t = 8)]
    pub arena_games: u32,
    /// Seconds to wait after shutdown before the post-shutdown VRAM sample.
    #[arg(long, default_value_t = 1.0)]
    pub settle_secs: f64,
}

#[derive(Debug, Clone, serde::Serialize)]
struct RepResult {
    mode: String,
    rep: u32,
    vram_before_mb: Option<u64>,
    vram_after_shutdown_mb: Option<u64>,
    secs: f64,
    evaluations: u64,
    evaluations_per_sec: f64,
    /// Batch-weighted mean forward latency across owners.
    forward_us_mean: f64,
    batch_mean: f64,
    errors: u64,
    owners_spawned: u32,
    max_resident_owners: u32,
    gpu: GpuSamples,
}

fn combine(snaps: &[MetricsSnapshot]) -> (u64, f64, f64, u64) {
    let completed: u64 = snaps.iter().map(|m| m.completed).sum();
    let batches: u64 = snaps.iter().map(|m| m.batches).sum();
    let fwd = snaps
        .iter()
        .map(|m| m.forward_us_mean * m.batches as f64)
        .sum::<f64>()
        / batches.max(1) as f64;
    let batch_mean = completed as f64 / batches.max(1) as f64;
    (
        completed,
        fwd,
        batch_mean,
        snaps.iter().map(|m| m.errors).sum(),
    )
}

/// One repetition of a mode; returns (inference snapshots, spawned, resident).
fn run_mode<B: Backend>(
    mode: &str,
    cfg: &RunConfig,
    ckpt: &Path,
    model_id: &str,
    openings: &[String],
    rep: u32,
    device: &B::Device,
) -> anyhow::Result<(Vec<MetricsSnapshot>, u32, u32)> {
    let concurrency = cfg.collection_shape()?.1;
    let seed = cfg.seed.wrapping_add(rep as u64);
    let arena_cfg = ArenaConfig {
        games: cfg.arena_games,
        simulations: cfg.simulations_per_move,
        c_puct: cfg.c_puct,
        recurrence: cfg.recurrence,
        ply_cap: cfg.ply_cap,
        seed,
        openings: openings.to_vec(),
        concurrency,
    };
    let raw_games = cfg.arena_games.max(4);
    match mode {
        "one" => {
            let owner = spawn_owner::<B>(ckpt, cfg, device)?;
            let ev = owner.evaluator();
            run_arena(&ev, &ev, model_id, model_id, &arena_cfg)?;
            raw_policy_vs_random(
                &ev,
                raw_games,
                cfg.temperature,
                cfg.ply_cap,
                seed,
                openings,
                concurrency,
            )?;
            drop(ev);
            let snap = owner.metrics().snapshot();
            owner.shutdown();
            Ok((vec![snap], 1, 1))
        }
        "two" => {
            let a = spawn_owner::<B>(ckpt, cfg, device)?;
            let b = spawn_owner::<B>(ckpt, cfg, device)?;
            let (ea, eb) = (a.evaluator(), b.evaluator());
            run_arena(&ea, &eb, model_id, model_id, &arena_cfg)?;
            raw_policy_vs_parent(
                &ea,
                &eb,
                raw_games,
                cfg.ply_cap,
                seed,
                openings,
                concurrency,
            )?;
            drop((ea, eb));
            let snaps = vec![a.metrics().snapshot(), b.metrics().snapshot()];
            a.shutdown();
            b.shutdown();
            Ok((snaps, 2, 2))
        }
        "pilot" | "pilot-promoted" => {
            let parent_label = if mode == "pilot" {
                model_id.to_string()
            } else {
                format!("{model_id}-as-promoted-parent")
            };
            let models = EvalModels {
                parent_dir: ckpt,
                parent_model_id: &parent_label,
                candidate_dir: ckpt,
                candidate_model_id: model_id,
                reference_dir: ckpt,
                reference_model_id: model_id,
            };
            let out = evaluate_candidate::<B>(cfg, &models, rep, openings, concurrency, device)?;
            anyhow::ensure!(
                out.reference_arena_is_parent_arena == (mode == "pilot"),
                "unexpected reference-arena reuse for mode {mode}"
            );
            Ok((
                out.inference.into_iter().map(|(_, s)| s).collect(),
                out.owners_spawned,
                out.max_resident_owners,
            ))
        }
        other => anyhow::bail!("unknown mode '{other}' (one | two | pilot | pilot-promoted)"),
    }
}

fn run_impl<B: AutodiffBackend>(cfg: &RunConfig, args: &BenchLifecycleArgs) -> anyhow::Result<()> {
    let device: Device<B::InnerBackend> = Default::default();
    let gpu_on = cfg.device == "cuda";
    let model_id = std::fs::read(args.checkpoint.join("meta.json"))
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .and_then(|v| {
            v.get("model_id")
                .and_then(|m| m.as_str())
                .map(str::to_owned)
        })
        .ok_or_else(|| anyhow::anyhow!("checkpoint has no model_id in meta.json"))?;
    let openings = match &cfg.opening_suite {
        Some(p) => OpeningSuite::load(Path::new(p))?.openings,
        None => Vec::new(),
    };
    let modes: Vec<String> = args
        .modes
        .split(',')
        .map(|m| m.trim().to_string())
        .collect();
    let initial_vram_mb = gpu_on
        .then(gpu_telemetry::sample_gpu)
        .flatten()
        .map(|s| s.0);
    let mut results = Vec::new();
    for mode in &modes {
        for rep in 0..args.reps {
            let vram_before_mb = gpu_on
                .then(gpu_telemetry::sample_gpu)
                .flatten()
                .map(|s| s.0);
            let ((out, secs), gpu) = gpu_telemetry::monitor(gpu_on, || {
                let start = Instant::now();
                let out = run_mode::<B::InnerBackend>(
                    mode,
                    cfg,
                    &args.checkpoint,
                    &model_id,
                    &openings,
                    rep,
                    &device,
                );
                (out, start.elapsed().as_secs_f64().max(1e-9))
            });
            let (snaps, owners_spawned, max_resident_owners) = out?;
            std::thread::sleep(Duration::from_secs_f64(args.settle_secs.max(0.0)));
            let vram_after_shutdown_mb = gpu_on
                .then(gpu_telemetry::sample_gpu)
                .flatten()
                .map(|s| s.0);
            let (evaluations, forward_us_mean, batch_mean, errors) = combine(&snaps);
            let r = RepResult {
                mode: mode.clone(),
                rep,
                vram_before_mb,
                vram_after_shutdown_mb,
                secs,
                evaluations,
                evaluations_per_sec: evaluations as f64 / secs,
                forward_us_mean,
                batch_mean,
                errors,
                owners_spawned,
                max_resident_owners,
                gpu,
            };
            println!(
                "{:<15} rep {:>2}: vram before/peak/after = {:?}/{:?}/{:?} MB | {:.1}s ev/s={:.1} fwd={:.0}us batch={:.2} err={} owners={}/{} util={:?} temp={:?}C",
                r.mode,
                r.rep,
                r.vram_before_mb,
                r.gpu.peak_vram_mb,
                r.vram_after_shutdown_mb,
                r.secs,
                r.evaluations_per_sec,
                r.forward_us_mean,
                r.batch_mean,
                r.errors,
                r.owners_spawned,
                r.max_resident_owners,
                r.gpu.util_busy_mean.map(|u| u.round()),
                r.gpu.temp_max_c
            );
            results.push(r);
        }
    }
    std::fs::create_dir_all(&args.output)?;
    let report = serde_json::json!({
        "checkpoint": args.checkpoint,
        "checkpoint_model_id": model_id,
        "device": cfg.device,
        "precision": cfg.precision,
        "simulations_per_move": cfg.simulations_per_move,
        "arena_games": cfg.arena_games,
        "concurrency": cfg.collection_shape()?.1,
        "max_inference_batch": cfg.max_inference_batch,
        "batch_timeout_us": cfg.batch_timeout_us,
        "initial_vram_mb": initial_vram_mb,
        "git_revision": option_env!("RECUR64_GIT_SHA"),
        "reps": results,
    });
    std::fs::write(
        args.output.join("bench-lifecycle.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("wrote {}/bench-lifecycle.json", args.output.display());
    Ok(())
}

pub fn run(args: BenchLifecycleArgs) -> anyhow::Result<()> {
    let mut cfg = RunConfig::from_toml_str(&std::fs::read_to_string(&args.config)?)?;
    cfg.arena_games = args.arena_games;
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
