//! `recur64 eval-arena` — batched searched arena between two checkpoints.
//!
//! Runs the pilot's arena path (batched inference owners, paired colours,
//! the frozen opening suite) outside a pilot, so evaluation-contract variants
//! (D45: sampled opening phase, root noise) can be measured on the same pair
//! of models without training.

use std::path::{Path, PathBuf};

use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use clap::Args;

use recur64_eval::{OpeningSuite, run_arena};
use recur64_runtime::gpu_telemetry;
use recur64_runtime::{RunConfig, spawn_owner};

type CpuTrain = burn::backend::Autodiff<burn::backend::Flex>;

#[derive(Args, Debug)]
pub struct EvalArenaArgs {
    #[arg(long)]
    pub config: PathBuf,
    /// Reference (parent) checkpoint.
    #[arg(long)]
    pub reference: PathBuf,
    /// Candidate checkpoint.
    #[arg(long)]
    pub candidate: PathBuf,
    #[arg(long)]
    pub output: PathBuf,
    /// Override the config's arena_games.
    #[arg(long)]
    pub arena_games: Option<u32>,
    /// Override arena_sample_plies (0 = argmax from ply 0).
    #[arg(long)]
    pub sample_plies: Option<u32>,
    /// Override arena_root_dirichlet_epsilon.
    #[arg(long)]
    pub noise_epsilon: Option<f32>,
    /// Seed offset (the pilot uses the cycle index).
    #[arg(long, default_value_t = 0)]
    pub seed_offset: u64,
}

fn model_id(dir: &Path) -> anyhow::Result<String> {
    std::fs::read(dir.join("meta.json"))
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .and_then(|v| {
            v.get("model_id")
                .and_then(|m| m.as_str())
                .map(str::to_owned)
        })
        .ok_or_else(|| anyhow::anyhow!("{} has no model_id", dir.display()))
}

fn run_impl<B: AutodiffBackend>(cfg: &RunConfig, args: &EvalArenaArgs) -> anyhow::Result<()> {
    let device: Device<B::InnerBackend> = Default::default();
    let openings = match &cfg.opening_suite {
        Some(p) => OpeningSuite::load(Path::new(p))?.openings,
        None => Vec::new(),
    };
    let (ref_id, cand_id) = (model_id(&args.reference)?, model_id(&args.candidate)?);
    let concurrency = cfg.collection_shape()?.1;
    let arena_cfg = cfg.arena_config(args.seed_offset, openings, concurrency);
    let ref_owner = spawn_owner::<B::InnerBackend>(&args.reference, cfg, &device)?;
    let cand_owner = spawn_owner::<B::InnerBackend>(&args.candidate, cfg, &device)?;
    let (ref_ev, cand_ev) = (ref_owner.evaluator(), cand_owner.evaluator());
    let start = std::time::Instant::now();
    let (result, gpu) = gpu_telemetry::monitor(cfg.device == "cuda", || {
        run_arena(&ref_ev, &cand_ev, &ref_id, &cand_id, &arena_cfg)
    });
    let secs = start.elapsed().as_secs_f64();
    let result = result.map_err(|e| anyhow::anyhow!("arena failed: {e}"))?;
    drop((ref_ev, cand_ev));
    let inference = [
        ref_owner.metrics().snapshot(),
        cand_owner.metrics().snapshot(),
    ];
    ref_owner.shutdown();
    cand_owner.shutdown();
    let report = serde_json::json!({
        "reference_model_id": ref_id,
        "candidate_model_id": cand_id,
        "scientific_config_hash": cfg.scientific_config_hash()?,
        "arena": {
            "games": arena_cfg.games,
            "simulations": arena_cfg.simulations,
            "sample_plies": arena_cfg.sample_plies,
            "root_dirichlet_alpha": arena_cfg.root_dirichlet_alpha,
            "root_dirichlet_epsilon": arena_cfg.root_dirichlet_epsilon,
            "seed": arena_cfg.seed,
            "concurrency": arena_cfg.concurrency,
        },
        "result": result,
        "secs": secs,
        "inference_errors": inference.iter().map(|m| m.errors).sum::<u64>(),
        "gpu": gpu,
        "git_revision": option_env!("RECUR64_GIT_SHA"),
    });
    std::fs::create_dir_all(&args.output)?;
    std::fs::write(
        args.output.join("eval-arena.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!(
        "arena sample_plies={:?} eps={} games={}: cand W/D/L {}/{}/{} trunc {} score {:.3} decisive {} terminations {:?} ({:.0}s)",
        arena_cfg.sample_plies,
        arena_cfg.root_dirichlet_epsilon,
        result.games,
        result.candidate_wins,
        result.draws,
        result.reference_wins,
        result.truncated,
        result.candidate_score,
        result.decisive_games,
        result.terminations,
        secs
    );
    Ok(())
}

pub fn run(args: EvalArenaArgs) -> anyhow::Result<()> {
    let mut cfg = RunConfig::from_toml_str(&std::fs::read_to_string(&args.config)?)?;
    if let Some(g) = args.arena_games {
        cfg.arena_games = g;
    }
    if let Some(p) = args.sample_plies {
        cfg.arena_sample_plies = (p > 0).then_some(p);
    }
    if let Some(e) = args.noise_epsilon {
        cfg.arena_root_dirichlet_epsilon = e;
    }
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
