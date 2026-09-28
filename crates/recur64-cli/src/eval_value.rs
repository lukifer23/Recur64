//! `recur64 eval-value`: a checkpoint's value (WDL) and policy cross-entropy
//! on a fixed held-out replay, with no training.
//!
//! This is the P2 primary metric. Every arm is scored on the identical
//! positions and targets, unlike each arm's fresh-data loss, which is measured
//! on that arm's own self-play. Read-only.

use std::path::PathBuf;

use burn::prelude::*;
use clap::Args;

use recur64_model::checkpoint::CheckpointMeta;
use recur64_model::loss::{policy_ce, wdl_ce};
use recur64_runtime::{ReplayStore, RunConfig, build_batch_tensors, model_io};

#[derive(Args, Debug)]
pub struct EvalValueArgs {
    #[arg(long)]
    pub config: PathBuf,
    #[arg(long)]
    pub checkpoint: PathBuf,
    /// Held-out replay directory.
    #[arg(long)]
    pub replay: PathBuf,
    /// Recurrence to evaluate at (default: the config's).
    #[arg(long)]
    pub recurrence: Option<usize>,
    #[arg(long, default_value_t = 64)]
    pub batch: usize,
    #[arg(long)]
    pub output: PathBuf,
}

fn scalar<B: Backend>(t: Tensor<B, 1>) -> f64 {
    t.into_data()
        .to_vec::<f32>()
        .map(|v| v[0] as f64)
        .unwrap_or(f64::NAN)
}

fn run_impl<B: Backend>(cfg: &RunConfig, args: &EvalValueArgs) -> anyhow::Result<()> {
    let device: B::Device = Default::default();
    let meta: CheckpointMeta =
        serde_json::from_slice(&std::fs::read(args.checkpoint.join("meta.json"))?)?;
    let model = model_io::load::<B>(&args.checkpoint, &meta.model, &device)?;
    let r = args.recurrence.unwrap_or(cfg.recurrence);
    let store = ReplayStore::open(&args.replay)?;
    let coords = store.coordinates();
    anyhow::ensure!(
        !coords.is_empty(),
        "held-out replay has no trainable positions"
    );
    let (mut n, mut wdl_sum, mut pol_sum) = (0usize, 0f64, 0f64);
    for chunk in coords.chunks(args.batch.max(1)) {
        let examples = chunk
            .iter()
            .map(|&c| store.example(c))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| anyhow::anyhow!(e))?;
        let refs: Vec<&_> = examples.iter().collect();
        let (board, cands, targets) = build_batch_tensors::<B>(&refs, &device);
        let out = model.forward_r(board, &cands, r, false);
        let readout = out
            .readouts
            .last()
            .ok_or_else(|| anyhow::anyhow!("no readout"))?;
        // Both losses are batch means; weight by batch size.
        let b = examples.len();
        wdl_sum += scalar(wdl_ce(&readout.wdl_logits, &targets.wdl_target)) * b as f64;
        pol_sum += scalar(policy_ce(&readout.policy, &targets.policy_target)) * b as f64;
        n += b;
    }
    let report = serde_json::json!({
        "kind": "recur64-eval-value-v1",
        "checkpoint": args.checkpoint.display().to_string(),
        "model_id": meta.model_id,
        "update_counter": meta.update_counter,
        "recurrence": r,
        "replay": args.replay.display().to_string(),
        "positions": n,
        "games": store.total_games(),
        "mean_wdl_ce": wdl_sum / n as f64,
        "mean_policy_ce": pol_sum / n as f64,
        "uniform_wdl_ce": 3f64.ln(),
        "git_revision": recur64_runtime::provenance::git_revision(),
    });
    std::fs::create_dir_all(&args.output)?;
    std::fs::write(
        args.output.join("eval-value.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!(
        "eval-value R={r} positions={n}: wdl_ce={:.4} policy_ce={:.4} (uniform wdl {:.4})",
        wdl_sum / n as f64,
        pol_sum / n as f64,
        3f64.ln()
    );
    Ok(())
}

pub fn run(args: EvalValueArgs) -> anyhow::Result<()> {
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
