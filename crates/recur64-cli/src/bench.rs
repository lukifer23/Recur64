//! `recur64 bench` — bounded Phase 0 benchmark matrix.
//!
//! Writes raw JSON and a markdown summary to the output directory. GPU timings
//! synchronize the device; if synchronization cannot be performed the run fails
//! rather than publishing an invalid number.

use std::path::{Path, PathBuf};
use std::time::Instant;

use burn::backend::Flex;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use clap::Args;
use recur64_model::config::ProbeConfig;
use recur64_model::fixture::SynthFixture;
use recur64_model::model::ProbeModel;
use recur64_model::train::{adamw, train_step};

#[cfg(feature = "cuda")]
use burn::backend::Cuda;

#[derive(Args, Debug)]
pub struct BenchArgs {
    #[arg(long)]
    pub config: PathBuf,
    #[arg(long)]
    pub output: PathBuf,
    /// Device: `cpu` (Burn Flex) or `cuda` (Burn CUDA).
    #[arg(long, default_value = "cpu")]
    pub device: String,
    /// Inference batch sizes.
    #[arg(long, value_delimiter = ',', default_value = "1,16,64,128")]
    pub inference_batches: Vec<usize>,
    /// Training physical batch sizes.
    #[arg(long, value_delimiter = ',', default_value = "32")]
    pub train_batches: Vec<usize>,
    /// Recurrence counts (overrides config when provided).
    #[arg(long, value_delimiter = ',')]
    pub recurrences: Vec<usize>,
    #[arg(long, default_value_t = 2)]
    pub warmup: usize,
    #[arg(long, default_value_t = 5)]
    pub iters: usize,
    #[arg(long, default_value_t = 2)]
    pub train_steps: usize,
    #[arg(long, default_value_t = false)]
    pub skip_training: bool,
    #[arg(long, default_value_t = 1)]
    pub seed: u64,
}

#[derive(serde::Serialize)]
struct Case {
    kind: &'static str,
    batch: usize,
    recurrence: usize,
    executed_blocks: usize,
    cold_ms: f64,
    warm_ms: f64,
    examples_per_sec: f64,
    finite: bool,
}

#[derive(serde::Serialize)]
struct BenchReport {
    recur64_version: String,
    burn_version: String,
    config: String,
    precision: String,
    device: String,
    synchronized: bool,
    unique_params: usize,
    unique_blocks: usize,
    notes: Vec<String>,
    cases: Vec<Case>,
}

fn all_finite<B: Backend>(out: &recur64_model::model::ModelOutput<B>) -> bool {
    out.readouts.iter().all(|r| {
        r.policy
            .log_probs
            .clone()
            .into_data()
            .to_vec::<f32>()
            .map(|v| v.iter().all(|x| x.is_finite()))
            .unwrap_or(false)
            && r.wdl_logits
                .clone()
                .into_data()
                .to_vec::<f32>()
                .map(|v| v.iter().all(|x| x.is_finite()))
                .unwrap_or(false)
    })
}

pub fn run_bench(args: BenchArgs) -> anyhow::Result<()> {
    let text = std::fs::read_to_string(&args.config)?;
    let cfg = ProbeConfig::from_toml_str(&text)?;
    recur64_model::precision::ensure_supported(cfg.precision, cfg.device)?;

    let recurrences: Vec<usize> = if args.recurrences.is_empty() {
        cfg.recurrence.clone()
    } else {
        args.recurrences.clone()
    };

    match args.device.as_str() {
        "cpu" => run_with::<Flex, burn::backend::Autodiff<Flex>>(
            &cfg,
            &args,
            &recurrences,
            "cpu (Burn Flex)",
            false,
        ),
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                run_with::<Cuda, burn::backend::Autodiff<Cuda>>(
                    &cfg,
                    &args,
                    &recurrences,
                    "cuda (Burn 0.21.0)",
                    true,
                )
            }
            #[cfg(not(feature = "cuda"))]
            {
                anyhow::bail!("CUDA support is not compiled; rebuild with --features cuda")
            }
        }
        other => anyhow::bail!("unknown device '{other}' (expected cpu or cuda)"),
    }
}

fn run_with<B, TB>(
    cfg: &ProbeConfig,
    args: &BenchArgs,
    recurrences: &[usize],
    label: &str,
    synchronized: bool,
) -> anyhow::Result<()>
where
    B: Backend,
    TB: AutodiffBackend,
{
    // Both backends address the same physical device (index 0); each backend
    // exposes its own device type, so construct them independently.
    let device: B::Device = Default::default();
    let tb_device: TB::Device = Default::default();
    let model = ProbeModel::<B>::new(cfg.model.clone(), &device);
    let unique_params = model.num_params();

    let mut notes = vec![
        format!("{label}, {} precision.", cfg.precision.label()),
        "Cold = first invocation; warm = mean of steady-state iterations.".to_string(),
        "Not chess learning; synthetic fixtures only.".to_string(),
    ];
    if synchronized {
        notes.push("Device synchronized around each timed region.".to_string());
    } else {
        notes.push("CPU path: no device synchronization required.".to_string());
    }

    let mut cases: Vec<Case> = Vec::new();

    // Inference.
    for &batch in &args.inference_batches {
        let fx = SynthFixture::new(batch, cfg.model.in_features, args.seed);
        let (board, cands, _t) = fx.tensors::<B>(&device);
        for &r in recurrences {
            B::sync(&device)?;
            let t0 = Instant::now();
            let out = model.forward_r(board.clone(), &cands, r, false);
            B::sync(&device)?;
            let cold = t0.elapsed().as_secs_f64();
            let finite = all_finite(&out);
            for _ in 0..args.warmup {
                let _ = model.forward_r(board.clone(), &cands, r, false);
            }
            B::sync(&device)?;
            let t1 = Instant::now();
            for _ in 0..args.iters {
                let _ = model.forward_r(board.clone(), &cands, r, false);
            }
            B::sync(&device)?;
            let warm = t1.elapsed().as_secs_f64() / args.iters as f64;
            cases.push(Case {
                kind: "inference",
                batch,
                recurrence: r,
                executed_blocks: cfg.model.executed_blocks_final(r),
                cold_ms: cold * 1000.0,
                warm_ms: warm * 1000.0,
                examples_per_sec: batch as f64 / warm,
                finite,
            });
            println!(
                "inference  batch={batch:<4} R={r} blocks={:<3} cold={:.2}ms warm={:.2}ms ex/s={:.1}",
                cfg.model.executed_blocks_final(r),
                cold * 1000.0,
                warm * 1000.0,
                batch as f64 / warm
            );
            // A/B in the same process (per-process autotune variance is
            // large): per-forward index rebuild vs a cached index tensor,
            // alternated twice, each iteration synchronized like the real
            // inference owner (which reads results every batch).
            let rel = model.rel_index_tensor(&device);
            for round in 0..2 {
                for cached in [false, true] {
                    B::sync(&device)?;
                    let t = Instant::now();
                    for _ in 0..args.iters {
                        let _ = if cached {
                            model.forward_r_with_rel_idx(board.clone(), &cands, r, false, &rel)
                        } else {
                            model.forward_r(board.clone(), &cands, r, false)
                        };
                        B::sync(&device)?;
                    }
                    let ms = t.elapsed().as_secs_f64() * 1000.0 / args.iters as f64;
                    cases.push(Case {
                        kind: if cached {
                            "inference-sync-rel-cached"
                        } else {
                            "inference-sync-rel-rebuilt"
                        },
                        batch,
                        recurrence: r,
                        executed_blocks: cfg.model.executed_blocks_final(r),
                        cold_ms: 0.0,
                        warm_ms: ms,
                        examples_per_sec: batch as f64 * 1000.0 / ms,
                        finite,
                    });
                    println!(
                        "  rel-index {} round {round} batch={batch:<4} R={r} sync-per-forward={ms:.3}ms",
                        if cached { "cached " } else { "rebuilt" }
                    );
                }
            }
        }
    }

    // Training.
    if !args.skip_training {
        for &batch in &args.train_batches {
            let fx = SynthFixture::new(batch, cfg.model.in_features, args.seed);
            let (board, cands, targets) = fx.tensors::<TB>(&tb_device);
            for &r in recurrences {
                let mut m = ProbeModel::<TB>::new(cfg.model.clone(), &tb_device);
                let mut optim = adamw::<TB, ProbeModel<TB>>();
                TB::sync(&tb_device)?;
                let t0 = Instant::now();
                let (m2, loss) = train_step(
                    m,
                    &mut optim,
                    board.clone(),
                    &cands,
                    &targets,
                    r,
                    cfg.deep_supervision,
                    3e-4,
                );
                m = m2;
                TB::sync(&tb_device)?;
                let cold = t0.elapsed().as_secs_f64();
                let finite = loss
                    .into_data()
                    .to_vec::<f32>()
                    .map(|v| v.iter().all(|x| x.is_finite()))
                    .unwrap_or(false);
                let steps = args.train_steps.max(1);
                TB::sync(&tb_device)?;
                let t1 = Instant::now();
                for _ in 0..steps.saturating_sub(1) {
                    let (m2, _) = train_step(
                        m,
                        &mut optim,
                        board.clone(),
                        &cands,
                        &targets,
                        r,
                        cfg.deep_supervision,
                        3e-4,
                    );
                    m = m2;
                }
                TB::sync(&tb_device)?;
                let warm = t1.elapsed().as_secs_f64() / steps as f64;
                cases.push(Case {
                    kind: "training",
                    batch,
                    recurrence: r,
                    executed_blocks: if cfg.deep_supervision {
                        cfg.model.executed_blocks_deep_supervision(r)
                    } else {
                        cfg.model.executed_blocks_final(r)
                    },
                    cold_ms: cold * 1000.0,
                    warm_ms: warm * 1000.0,
                    examples_per_sec: batch as f64 / warm,
                    finite,
                });
                println!(
                    "training   batch={batch:<4} R={r} cold={:.2}ms warm={:.2}ms ex/s={:.1} finite={finite}",
                    cold * 1000.0,
                    warm * 1000.0,
                    batch as f64 / warm
                );
            }
        }
    }

    let report = BenchReport {
        recur64_version: recur64_model::VERSION.to_string(),
        burn_version: recur64_model::BURN_VERSION.to_string(),
        config: cfg.name.clone(),
        precision: cfg.precision.label().to_string(),
        device: label.to_string(),
        synchronized,
        unique_params,
        unique_blocks: cfg.model.unique_blocks(),
        notes,
        cases,
    };

    write_report(&args.output, &report)?;
    Ok(())
}

fn write_report(output: &Path, report: &BenchReport) -> anyhow::Result<()> {
    std::fs::create_dir_all(output)?;
    let json = serde_json::to_vec_pretty(report)?;
    std::fs::write(output.join("bench.json"), &json)?;

    let mut md = String::new();
    md.push_str(&format!("# Recur64 bench: {}\n\n", report.config));
    md.push_str(&format!(
        "- recur64 {} | burn {} | {} | device {} | synchronized {}\n",
        report.recur64_version,
        report.burn_version,
        report.precision,
        report.device,
        report.synchronized
    ));
    md.push_str(&format!(
        "- unique params: {} | unique blocks: {}\n\n",
        report.unique_params, report.unique_blocks
    ));
    for n in &report.notes {
        md.push_str(&format!("- {n}\n"));
    }
    md.push_str("\n| kind | batch | R | blocks | cold ms | warm ms | ex/s | finite |\n");
    md.push_str("|---|---:|---:|---:|---:|---:|---:|:--:|\n");
    for c in &report.cases {
        md.push_str(&format!(
            "| {} | {} | {} | {} | {:.2} | {:.2} | {:.1} | {} |\n",
            c.kind,
            c.batch,
            c.recurrence,
            c.executed_blocks,
            c.cold_ms,
            c.warm_ms,
            c.examples_per_sec,
            c.finite
        ));
    }
    std::fs::write(output.join("bench.md"), md)?;
    println!(
        "\nwrote {} and bench.md",
        output.join("bench.json").display()
    );
    Ok(())
}
