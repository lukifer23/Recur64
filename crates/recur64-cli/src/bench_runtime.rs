//! `recur64 bench-runtime` — Stage A warmup + batching/active-game sweep.

use std::path::PathBuf;

use burn::tensor::backend::AutodiffBackend;
use clap::Args;

use recur64_runtime::sweep::{SweepCellResult, grid, run_cell, warmup, workstation_grid};
use recur64_runtime::{RunConfig, SweepCellSpec};

#[derive(Args, Debug)]
pub struct BenchRuntimeArgs {
    #[arg(long)]
    pub config: PathBuf,
    #[arg(long)]
    pub output: PathBuf,
    /// Frozen inference/training checkpoint whose weights are reused in every cell.
    #[arg(long)]
    pub checkpoint: Option<PathBuf>,
    /// `small`, `full`, or `workstation` (main-workstation schedule candidates).
    #[arg(long, default_value = "small")]
    pub grid: String,
    #[arg(long, default_value_t = 32)]
    pub games_per_cell: u64,
    /// If any override is given, run a single cell built from these values.
    #[arg(long)]
    pub active: Option<u32>,
    #[arg(long)]
    pub max_batch: Option<usize>,
    #[arg(long)]
    pub timeout_us: Option<u64>,
    #[arg(long)]
    pub simulations: Option<u32>,
}

fn run_impl<B: AutodiffBackend>(
    cfg: &RunConfig,
    output: &std::path::Path,
    args: &BenchRuntimeArgs,
) -> anyhow::Result<()> {
    let device: B::Device = Default::default();
    let full = args.grid == "full";
    let games_per_cell = args.games_per_cell;
    let single = args.active.is_some()
        || args.max_batch.is_some()
        || args.timeout_us.is_some()
        || args.simulations.is_some();
    let cells: Vec<SweepCellSpec> = if single {
        vec![SweepCellSpec {
            active_games: args.active.unwrap_or(64),
            max_batch: args.max_batch.unwrap_or(64),
            batch_timeout_us: args.timeout_us.unwrap_or(500),
            simulations: args.simulations.unwrap_or(8),
        }]
    } else if args.grid == "workstation" {
        workstation_grid()
    } else {
        grid(!full)
    };
    let max_batch = cells.iter().map(|c| c.max_batch).max().unwrap_or(1);

    let warmup_secs = warmup::<B>(cfg, max_batch, &device)?;
    println!("warmup: {warmup_secs:.2}s (batch sizes up to {max_batch})");

    let mut results: Vec<SweepCellResult> = Vec::new();
    for cell in cells {
        let r = run_cell::<B>(cfg, cell, games_per_cell, args.checkpoint.as_deref())?;
        println!(
            "active={:<4} batch={:<4} timeout={:<5}us sims={:<3} | games={}/{} ev/s={:.1} games/h={:>7.1} pos/s={:>6.1} train_pos/s={:>6.1} batch mean/p50/p95={:.2}/{}/{} wait p95={}us vram={:?}MB util={:?}/{:?} temp={:?}C err={}",
            r.active_games,
            r.max_batch,
            r.batch_timeout_us,
            r.simulations,
            r.games,
            r.requested_games,
            r.evaluations_per_sec,
            r.games_per_hour,
            r.positions_per_sec,
            r.trainable_positions_per_sec,
            r.batch_mean,
            r.batch_p50,
            r.batch_p95,
            r.queue_wait_us_p95,
            r.peak_vram_mb,
            r.gpu_util_busy_mean.map(|u| u.round()),
            r.gpu_util_max,
            r.gpu_temp_max_c,
            r.errors
        );
        results.push(r);
    }

    std::fs::create_dir_all(output)?;
    let report = serde_json::json!({
        "warmup_secs": warmup_secs,
        "games_per_cell": games_per_cell,
        "model": cfg.model,
        "device": cfg.device,
        "checkpoint": args.checkpoint,
        "checkpoint_model_id": args.checkpoint.as_ref().and_then(|c| {
            std::fs::read(c.join("meta.json"))
                .ok()
                .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
                .and_then(|v| v.get("model_id").cloned())
        }),
        "scientific_config_hash": cfg.scientific_config_hash()?,
        "resolved_config_hash": cfg.resolved_config_hash(),
        "git_revision": recur64_runtime::RunMetadata::new(cfg).git_revision,
        "cells": results,
    });
    std::fs::write(
        output.join("sweep.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;

    let mut md = String::from("# Recur64 runtime sweep (Stage A)\n\n");
    md.push_str(&format!(
        "- warmup: {warmup_secs:.2}s | games/cell: {games_per_cell}\n\n"
    ));
    md.push_str("| active | batch | timeout us | sims | ev/s | games/h | pos/s | trainable pos/s | batch mean/p50/p95 | wait p95 us | vram MB |\n");
    md.push_str("|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|\n");
    for r in &results {
        md.push_str(&format!(
            "| {} | {} | {} | {} | {:.1} | {:.1} | {:.1} | {:.1} | {:.2}/{}/{} | {} | {} |\n",
            r.active_games,
            r.max_batch,
            r.batch_timeout_us,
            r.simulations,
            r.evaluations_per_sec,
            r.games_per_hour,
            r.positions_per_sec,
            r.trainable_positions_per_sec,
            r.batch_mean,
            r.batch_p50,
            r.batch_p95,
            r.queue_wait_us_p95,
            r.peak_vram_mb
                .map(|v| v.to_string())
                .unwrap_or_else(|| "?".into())
        ));
    }
    std::fs::write(output.join("sweep.md"), md)?;
    println!("\nwrote {}/sweep.json and sweep.md", output.display());
    Ok(())
}

pub fn run(args: BenchRuntimeArgs) -> anyhow::Result<()> {
    let text = std::fs::read_to_string(&args.config)?;
    let cfg = RunConfig::from_toml_str(&text)?;
    cfg.ensure_supported()?;
    match cfg.device.as_str() {
        "cpu" => {
            run_impl::<burn::backend::Autodiff<burn::backend::Flex>>(&cfg, &args.output, &args)
        }
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                run_impl::<burn::backend::Autodiff<burn::backend::Cuda>>(&cfg, &args.output, &args)
            }
            #[cfg(not(feature = "cuda"))]
            {
                anyhow::bail!("CUDA support is not compiled; rebuild with --features cuda")
            }
        }
        other => anyhow::bail!("unknown device '{other}'"),
    }
}
