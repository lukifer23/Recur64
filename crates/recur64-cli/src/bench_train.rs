//! `recur64 bench-train` — learner throughput for physical-batch layouts.
//!
//! Loads a full training checkpoint (weights + Adam state), samples real
//! replay through `train_from_store`, and times updates for each
//! `physical x accumulation` layout at one fixed effective batch. Warmup
//! updates (JIT/autotune) are run first and excluded from timing. Run one
//! layout per process when comparing peak VRAM: the device memory pool is
//! process-wide.

use std::path::PathBuf;
use std::time::Instant;

use burn::tensor::backend::AutodiffBackend;
use clap::Args;

use recur64_model::checkpoint::load_training;
use recur64_model::train::adamw;
use recur64_runtime::gpu_telemetry::{self, GpuSamples};
use recur64_runtime::{LearnerConfig, ReplayStore, RunConfig, model_io, train_from_store};

type CpuTrain = burn::backend::Autodiff<burn::backend::Flex>;

#[derive(Args, Debug)]
pub struct BenchTrainArgs {
    #[arg(long)]
    pub config: PathBuf,
    /// Full training checkpoint (e.g. the frozen reference).
    #[arg(long)]
    pub checkpoint: PathBuf,
    /// Replay directory with real self-play games.
    #[arg(long)]
    pub replay: PathBuf,
    #[arg(long)]
    pub output: PathBuf,
    /// Comma-separated `physical x accumulation` layouts, e.g. `64x4,128x2`.
    /// All layouts must share one effective batch.
    #[arg(long, default_value = "32x8,64x4,128x2")]
    pub layouts: String,
    /// Timed optimizer updates per layout.
    #[arg(long, default_value_t = 20)]
    pub updates: usize,
    /// Untimed warmup updates per layout.
    #[arg(long, default_value_t = 2)]
    pub warmup_updates: usize,
}

#[derive(Debug, Clone, serde::Serialize)]
struct LayoutResult {
    physical_batch: usize,
    accumulation_steps: usize,
    effective_batch: usize,
    warmup_updates: usize,
    warmup_secs: f64,
    updates: usize,
    examples_consumed: u64,
    secs: f64,
    examples_per_sec: f64,
    step_ms_mean: f64,
    first_loss: f32,
    last_loss: f32,
    max_grad_norm: f32,
    all_finite: bool,
    error: Option<String>,
    gpu: GpuSamples,
}

fn parse_layouts(s: &str) -> anyhow::Result<Vec<(usize, usize)>> {
    let layouts = s
        .split(',')
        .map(|l| {
            let (p, a) = l
                .trim()
                .split_once('x')
                .ok_or_else(|| anyhow::anyhow!("layout '{l}' is not PHYSICALxACCUM"))?;
            Ok((p.parse::<usize>()?, a.parse::<usize>()?))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    anyhow::ensure!(!layouts.is_empty(), "no layouts given");
    let effective = layouts[0].0 * layouts[0].1;
    anyhow::ensure!(
        layouts.iter().all(|(p, a)| p * a == effective && *p > 0),
        "every layout must have the same effective batch ({effective})"
    );
    Ok(layouts)
}

fn run_impl<B: AutodiffBackend>(cfg: &RunConfig, args: &BenchTrainArgs) -> anyhow::Result<()> {
    let device: B::Device = Default::default();
    let layouts = parse_layouts(&args.layouts)?;
    let store = ReplayStore::open(&args.replay)?;
    anyhow::ensure!(store.sampleable() > 0, "replay has no trainable positions");
    let (warmup, planned) = cfg.lr_schedule();
    let gpu_on = cfg.device == "cuda";
    let mut results = Vec::new();

    for (physical, accum) in layouts {
        let learner = |max_updates: usize, start_update: u64| LearnerConfig {
            batch_size: physical,
            accumulation_steps: accum,
            max_updates,
            lr: cfg.lr,
            warmup_updates: warmup,
            planned_updates: planned,
            start_update,
            recurrence: cfg.recurrence,
            seed: cfg.seed,
            deadline: None,
            current_cycle_first_game_id: None,
            games_per_cycle: 0,
        };
        let (model, mut optim, _) = load_training(
            &args.checkpoint,
            model_io::build::<B>(&cfg.model, &device),
            adamw::<B, _>(),
            &device,
        )?;
        let warm_start = Instant::now();
        let warmed = train_from_store(
            &store,
            model,
            &mut optim,
            &learner(args.warmup_updates, 0),
            &device,
        );
        let warmup_secs = warm_start.elapsed().as_secs_f64();
        let model = match warmed {
            Ok((m, _)) => m,
            Err(e) => anyhow::bail!("warmup failed for {physical}x{accum}: {e}"),
        };
        let ((timed, secs), gpu) = gpu_telemetry::monitor(gpu_on, || {
            let start = Instant::now();
            let r = train_from_store(
                &store,
                model,
                &mut optim,
                &learner(args.updates, args.warmup_updates as u64),
                &device,
            );
            (r, start.elapsed().as_secs_f64().max(1e-9))
        });
        let r = match timed {
            Ok((_, report)) => LayoutResult {
                physical_batch: physical,
                accumulation_steps: accum,
                effective_batch: physical * accum,
                warmup_updates: args.warmup_updates,
                warmup_secs,
                updates: report.updates,
                examples_consumed: report.examples_consumed,
                secs,
                examples_per_sec: report.examples_consumed as f64 / secs,
                step_ms_mean: secs * 1000.0 / report.updates.max(1) as f64,
                first_loss: report.first_loss,
                last_loss: report.last_loss,
                max_grad_norm: report
                    .metrics
                    .iter()
                    .map(|m| m.grad_norm)
                    .fold(0.0, f32::max),
                all_finite: report.metrics.iter().all(|m| {
                    m.total_loss.is_finite()
                        && m.policy_loss.is_finite()
                        && m.wdl_loss.is_finite()
                        && m.grad_norm.is_finite()
                }),
                error: None,
                gpu,
            },
            Err(e) => LayoutResult {
                physical_batch: physical,
                accumulation_steps: accum,
                effective_batch: physical * accum,
                warmup_updates: args.warmup_updates,
                warmup_secs,
                updates: 0,
                examples_consumed: 0,
                secs,
                examples_per_sec: 0.0,
                step_ms_mean: 0.0,
                first_loss: f32::NAN,
                last_loss: f32::NAN,
                max_grad_norm: f32::NAN,
                all_finite: false,
                error: Some(e),
                gpu,
            },
        };
        println!(
            "layout {}x{} (eff {}): updates={} ex/s={:.1} step={:.1}ms loss {:.4}->{:.4} max_grad={:.3} finite={} vram peak={:?}MB temp={:?}C err={:?}",
            r.physical_batch,
            r.accumulation_steps,
            r.effective_batch,
            r.updates,
            r.examples_per_sec,
            r.step_ms_mean,
            r.first_loss,
            r.last_loss,
            r.max_grad_norm,
            r.all_finite,
            r.gpu.peak_vram_mb,
            r.gpu.temp_max_c,
            r.error
        );
        results.push(r);
    }

    std::fs::create_dir_all(&args.output)?;
    let checkpoint_model_id = std::fs::read(args.checkpoint.join("meta.json"))
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .and_then(|v| v.get("model_id").cloned());
    let report = serde_json::json!({
        "checkpoint": args.checkpoint,
        "checkpoint_model_id": checkpoint_model_id,
        "replay": args.replay,
        "replay_total_games": store.total_games(),
        "replay_sampleable_positions": store.sampleable(),
        "device": cfg.device,
        "precision": cfg.precision,
        "model": cfg.model,
        "lr": cfg.lr,
        "lr_schedule": {"warmup_updates": warmup, "planned_updates": planned},
        "git_revision": option_env!("RECUR64_GIT_SHA"),
        "layouts": results,
    });
    std::fs::write(
        args.output.join("bench-train.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("wrote {}/bench-train.json", args.output.display());
    Ok(())
}

pub fn run(args: BenchTrainArgs) -> anyhow::Result<()> {
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

#[cfg(test)]
mod tests {
    use super::parse_layouts;

    #[test]
    fn layouts_must_share_one_effective_batch() {
        assert_eq!(
            parse_layouts("32x8, 64x4,128x2").unwrap(),
            [(32, 8), (64, 4), (128, 2)]
        );
        assert!(parse_layouts("64x4,64x2").is_err());
        assert!(parse_layouts("64").is_err());
    }
}
