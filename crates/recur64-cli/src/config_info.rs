//! `recur64 config-info`: resolved run config, identities, schedule and
//! workload bounds, for pre-flight records (H3.5B P10). Read-only.

use std::path::PathBuf;

use clap::Args;

use recur64_runtime::RunConfig;

#[derive(Args, Debug)]
pub struct ConfigInfoArgs {
    #[arg(long)]
    pub config: PathBuf,
    /// New trainable positions per cycle to evaluate the update plan at
    /// (repeatable), e.g. expected, stress and theoretical workloads.
    #[arg(long = "positions")]
    pub positions: Vec<u64>,
    #[arg(long)]
    pub output: Option<PathBuf>,
}

pub fn run(args: ConfigInfoArgs) -> anyhow::Result<()> {
    let cfg = RunConfig::from_toml_str(&std::fs::read_to_string(&args.config)?)?;
    let (warmup, planned) = cfg.lr_schedule();
    let plans = args
        .positions
        .iter()
        .map(|&p| {
            Ok(serde_json::json!({"new_trainable_positions": p, "plan": cfg.update_plan(p)?}))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let out = serde_json::json!({
        "config_path": args.config.display().to_string(),
        "scientific_config_hash": cfg.scientific_config_hash()?,
        "resolved_config_hash": cfg.resolved_config_hash(),
        "scientific_identity": cfg.scientific_identity()?,
        "head_version": recur64_model::model::HEAD_VERSION,
        "reference_model_id": cfg.reference_model_id,
        "search_leaves_in_flight": cfg.search_leaves_in_flight,
        "simulations_per_move": cfg.simulations_per_move,
        "arena": {
            "games": cfg.arena_games,
            "sample_plies": cfg.arena_sample_plies,
            "root_dirichlet_epsilon": cfg.arena_root_dirichlet_epsilon,
            "rng_policy": cfg.arena_rng_policy,
        },
        "trainer_policy": cfg.trainer_policy,
        "lr_schedule": {"warmup_updates": warmup, "planned_updates": planned},
        "max_updates_per_cycle": cfg.max_updates,
        "update_plans": plans,
        "health_stops": cfg.health_stops,
        "hardware_schedule": {
            "collection_shape": cfg.collection_shape()?,
            "cpu_workers": cfg.cpu_workers,
            "max_inference_batch": cfg.max_inference_batch,
            "batch_timeout_us": cfg.batch_timeout_us,
            "train_batch": cfg.train_batch,
            "accumulation_steps": cfg.accumulation_steps,
            "effective_batch": cfg.effective_batch(),
        },
        "execution_bounds": {
            "cycles": cfg.cycles,
            "run_budget_minutes": cfg.run_budget_minutes,
            "position_budget": cfg.position_budget,
        },
        "git_revision": recur64_runtime::provenance::git_revision(),
        "git_branch": recur64_runtime::provenance::git_branch(),
        "resolved_config": cfg,
    });
    let text = serde_json::to_string_pretty(&out)?;
    if let Some(path) = &args.output {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, &text)?;
    }
    println!("{text}");
    Ok(())
}
