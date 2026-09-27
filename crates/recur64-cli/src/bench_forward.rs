//! `recur64 bench-forward`: fast (~1-2 min) throughput probe of the exact
//! production inference batch path (`BatchedModel::evaluate_batch`: host
//! encoding tensors, uploads, forward, readbacks, post-processing) on real,
//! reproducible positions, with an output-parity check against a baseline.
//!
//! Used for the throughput optimization loop: change one thing, re-run, keep
//! it only if it is faster *and* parity holds.

use std::path::{Path, PathBuf};
use std::time::Instant;

use burn::prelude::*;
use clap::Args;

use recur64_core::{GameState, ObservationV1, StandardMove, encode_observation_v1};
use recur64_eval::OpeningSuite;
use recur64_model::checkpoint::CheckpointMeta;
use recur64_runtime::{BatchEvaluator, BatchedModel, Rng, RunConfig, model_io};

#[derive(Args, Debug)]
pub struct BenchForwardArgs {
    #[arg(long)]
    pub config: PathBuf,
    #[arg(long)]
    pub checkpoint: PathBuf,
    #[arg(long)]
    pub output: PathBuf,
    /// Recurrence values to measure (default: the config's).
    #[arg(long, value_delimiter = ',')]
    pub recurrences: Vec<usize>,
    #[arg(long, value_delimiter = ',', default_value = "1,4,8,16,32,64")]
    pub batches: Vec<usize>,
    #[arg(long, default_value_t = 3)]
    pub warmup: usize,
    #[arg(long, default_value_t = 20)]
    pub iters: usize,
    /// Distinct positions (openings + seeded random playouts).
    #[arg(long, default_value_t = 256)]
    pub positions: usize,
    /// Round candidate widths up to fixed buckets (fewer tensor shapes).
    #[arg(long, default_value_t = false)]
    pub bucket_candidates: bool,
    /// Earlier bench-forward JSON to compare outputs against (parity).
    #[arg(long)]
    pub baseline: Option<PathBuf>,
}

/// Reproducible non-terminal positions: each opening, then 0..=40 seeded
/// random legal plies.
fn positions(openings: &[String], n: usize) -> anyhow::Result<Vec<GameState>> {
    let starts: Vec<String> = if openings.is_empty() {
        vec![GameState::startpos().to_fen()]
    } else {
        openings.to_vec()
    };
    let mut rng = Rng::new(20260927);
    let mut out = Vec::with_capacity(n);
    let mut i = 0usize;
    while out.len() < n {
        let mut s = GameState::from_fen(&starts[i % starts.len()])?;
        let plies = (rng.next_u64() % 41) as usize;
        for _ in 0..plies {
            if s.termination().is_some() {
                break;
            }
            let legal = s.legal_actions();
            let a = legal[(rng.next_u64() % legal.len() as u64) as usize];
            let (from, to, promo) = a.to_physical(s.perspective());
            s.apply(StandardMove::new(
                from,
                to,
                (!promo.is_none()).then_some(promo),
            ))?;
        }
        if s.termination().is_none() {
            out.push(s);
        }
        i += 1;
    }
    Ok(out)
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * q).round() as usize]
}

fn run_impl<B: Backend>(cfg: &RunConfig, args: &BenchForwardArgs) -> anyhow::Result<()> {
    let device: B::Device = Default::default();
    let meta: CheckpointMeta =
        serde_json::from_slice(&std::fs::read(args.checkpoint.join("meta.json"))?)?;
    let openings = match &cfg.opening_suite {
        Some(p) => OpeningSuite::load(Path::new(p))?.openings,
        None => Vec::new(),
    };
    let states = positions(&openings, args.positions)?;
    let obs: Vec<ObservationV1> = states.iter().map(encode_observation_v1).collect();
    let legal: Vec<Vec<recur64_core::ActionId>> =
        states.iter().map(|s| s.legal_actions()).collect();
    let recurrences = if args.recurrences.is_empty() {
        vec![cfg.recurrence]
    } else {
        args.recurrences.clone()
    };

    let baseline: Option<serde_json::Value> = match &args.baseline {
        Some(p) => Some(serde_json::from_slice(&std::fs::read(p)?)?),
        None => None,
    };
    let mut cells = Vec::new();
    let mut outputs = serde_json::Map::new();
    let mut parity = serde_json::Map::new();
    for &r in &recurrences {
        let model = model_io::load::<B>(&args.checkpoint, &meta.model, &device)?;
        let bm = BatchedModel::new(model, r, device.clone())
            .with_candidate_buckets(args.bucket_candidates);
        // Parity outputs: every position once, in batches of 16.
        let mut pol: Vec<f32> = Vec::new();
        let mut val: Vec<f32> = Vec::new();
        for (o, l) in obs.chunks(16).zip(legal.chunks(16)) {
            for res in bm
                .evaluate_batch(o, l)
                .map_err(|e| anyhow::anyhow!("{e}"))?
            {
                pol.extend(res.policy);
                val.push(res.value);
            }
        }
        if let Some(base) = baseline
            .as_ref()
            .and_then(|b| b["outputs"].get(format!("r{r}")))
        {
            let bp: Vec<f32> = serde_json::from_value(base["policy"].clone())?;
            let bv: Vec<f32> = serde_json::from_value(base["value"].clone())?;
            anyhow::ensure!(
                bp.len() == pol.len() && bv.len() == val.len(),
                "baseline shape differs"
            );
            let md = |a: &[f32], b: &[f32]| {
                a.iter()
                    .zip(b)
                    .map(|(x, y)| (x - y).abs())
                    .fold(0f32, f32::max)
            };
            parity.insert(
                format!("r{r}"),
                serde_json::json!({"max_abs_policy_diff": md(&pol, &bp), "max_abs_value_diff": md(&val, &bv)}),
            );
        }
        outputs.insert(
            format!("r{r}"),
            serde_json::json!({"policy": pol, "value": val}),
        );

        for &b in &args.batches {
            let b = b.min(obs.len());
            let before = bm.phase_times();
            let mut times = Vec::with_capacity(args.iters);
            for it in 0..(args.warmup + args.iters) {
                let start = (it * b) % (obs.len() - b + 1);
                let t0 = Instant::now();
                bm.evaluate_batch(&obs[start..start + b], &legal[start..start + b])
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                if it >= args.warmup {
                    times.push(t0.elapsed().as_secs_f64() * 1e3);
                }
            }
            times.sort_by(|a, c| a.partial_cmp(c).unwrap());
            let p50 = quantile(&times, 0.5);
            let after = bm.phase_times();
            let n = (after.batches - before.batches).max(1) as f64;
            let per = |x: u64, y: u64| (x - y) as f64 / n / 1e3;
            let phases = serde_json::json!({
                "prepare_ms": per(after.prepare_us, before.prepare_us),
                "forward_submit_ms": per(after.forward_us, before.forward_us),
                "readback_ms": per(after.readback_us, before.readback_us),
                "post_ms": per(after.post_us, before.post_us),
            });
            println!(
                "R={r} batch={b:<3} p10={:.2}ms p50={p50:.2}ms p90={:.2}ms evals/s={:.0}",
                quantile(&times, 0.1),
                quantile(&times, 0.9),
                b as f64 / (p50 / 1e3)
            );
            cells.push(serde_json::json!({
                "recurrence": r, "batch": b, "p10_ms": quantile(&times, 0.1), "p50_ms": p50,
                "p90_ms": quantile(&times, 0.9), "evals_per_sec_p50": b as f64 / (p50 / 1e3),
                "mean_phase_ms_incl_warmup": phases,
            }));
            println!("         phases/batch (mean incl. warmup): {phases}");
        }
    }
    for (k, v) in &parity {
        println!("parity {k}: {v}");
    }
    let report = serde_json::json!({
        "kind": "recur64-bench-forward-v1",
        "checkpoint": args.checkpoint.display().to_string(),
        "model_id": meta.model_id,
        "device": cfg.device,
        "positions": obs.len(),
        "bucket_candidates": args.bucket_candidates,
        "build_features": {"fusion": cfg!(feature = "fusion"), "autotune": cfg!(feature = "autotune")},
        "cells": cells,
        "parity_vs_baseline": parity,
        "baseline": args.baseline.as_ref().map(|p| p.display().to_string()),
        "git_revision": recur64_runtime::provenance::git_revision(),
        "outputs": outputs,
    });
    std::fs::create_dir_all(&args.output)?;
    std::fs::write(
        args.output.join("bench-forward.json"),
        serde_json::to_vec(&report)?,
    )?;
    println!("wrote {}/bench-forward.json", args.output.display());
    Ok(())
}

pub fn run(args: BenchForwardArgs) -> anyhow::Result<()> {
    let cfg = RunConfig::from_toml_str(&std::fs::read_to_string(&args.config)?)?;
    cfg.ensure_supported()?;
    match cfg.device.as_str() {
        "cpu" => run_impl::<burn::backend::Flex>(&cfg, &args),
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                run_impl::<burn::backend::Cuda>(&cfg, &args)
            }
            #[cfg(not(feature = "cuda"))]
            {
                anyhow::bail!("CUDA support is not compiled; rebuild with --features cuda")
            }
        }
        other => anyhow::bail!("unknown device '{other}'"),
    }
}
