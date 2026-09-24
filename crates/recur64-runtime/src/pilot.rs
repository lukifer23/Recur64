//! Bounded multi-cycle pilot: COLLECT → AUDIT → TRAIN → EVALUATE, repeated.
//!
//! Each cycle freezes a snapshot for self-play, collects games, audits them,
//! trains a candidate, and evaluates it against the reference. A conservative
//! snapshot policy decides whether the candidate becomes the next self-play
//! snapshot. The loop is bounded by cycle count, wall-clock, and position budget
//! — it is not an autonomous long run.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

use recur64_eval::{ArenaConfig, ArenaResult, OpeningSuite, run_arena};
use recur64_model::checkpoint::{CheckpointMeta, load_training, save_training};
use recur64_model::train::adamw;

use crate::SyncEvaluator;
use crate::cancel::CancelToken;
use crate::config::{RunConfig, SnapshotPolicy};
use crate::coordinator::{SelfPlayMetrics, collect_parallel, selfplay_metrics};
use crate::eval_policy::{
    RawMatchResult, RawParentResult, raw_policy_vs_parent, raw_policy_vs_random,
};
use crate::inference::{BatchedModel, InferenceConfig, InferenceOwner};
use crate::learner::{LearnerConfig, TrainReport, train_from_store};
use crate::model_io;
use crate::replay::{ReplayHeader, ReplayStore, ReplayWriter, audit_dir, enforce_capacity};
use crate::run_dir::{LineageRecord, RunDir, RunStatus};

/// One cycle's report.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CycleReport {
    pub cycle: u32,
    pub games: u64,
    pub positions: u64,
    pub new_trainable_positions: u64,
    pub mean_target_entropy: f64,
    pub mean_top1_visit_share: f64,
    pub requested_examples: f64,
    pub requested_updates: usize,
    pub scheduled_updates: usize,
    pub selfplay: SelfPlayMetrics,
    pub audit_ok: bool,
    pub train: Option<TrainReport>,
    pub arena: Option<ArenaResult>,
    pub reference_arena: Option<ArenaResult>,
    pub raw: Option<RawMatchResult>,
    pub raw_parent: Option<RawParentResult>,
    pub snapshot_model_id: String,
    pub candidate_model_id: String,
    pub replay_positions: u64,
    pub reuse_ratio: f64,
    pub reuse_shortfall_reason: Option<String>,
    pub decision: String,
    pub wall_secs: f64,
}

/// Whole-pilot report.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PilotReport {
    pub run_id: String,
    pub status: String,
    pub cycles: Vec<CycleReport>,
    pub total_positions: u64,
    pub elapsed_secs: f64,
}

fn backend_label(cfg: &RunConfig) -> String {
    format!("{} ({})", cfg.device, cfg.precision)
}

fn read_model_id(dir: &Path) -> String {
    std::fs::read(dir.join("meta.json"))
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .and_then(|v| {
            v.get("model_id")
                .and_then(|m| m.as_str())
                .map(|s| s.to_string())
        })
        .unwrap_or_default()
}

fn load_openings(cfg: &RunConfig) -> anyhow::Result<Vec<String>> {
    match &cfg.opening_suite {
        Some(p) => Ok(OpeningSuite::load(Path::new(p))?.openings),
        None => Ok(Vec::new()),
    }
}

fn copy_dir(src: &Path, dst: &Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            std::fs::copy(entry.path(), dst.join(entry.file_name()))?;
        }
    }
    Ok(())
}

fn arena_informative_for_promotion(arena: &ArenaResult) -> bool {
    arena.truncated < arena.games && arena.candidate_wins + arena.reference_wins > 0
}

/// Run a bounded multi-cycle pilot.
pub fn run_pilot<B: AutodiffBackend>(
    cfg: &RunConfig,
    run_dir: &RunDir,
    cancel: &CancelToken,
) -> anyhow::Result<PilotReport> {
    let started = Instant::now();
    let deadline = started + Duration::from_secs(cfg.run_budget_minutes.max(1) * 60);
    run_dir.write_config_and_metadata(cfg)?;

    let b_device: B::Device = Default::default();
    let inner_device: Device<B::InnerBackend> = Default::default();

    // Reference checkpoint (random F10).
    let reference_model = model_io::build::<B>(&cfg.model, &b_device);
    let reference_optim = adamw::<B, _>();
    let mut ref_meta = CheckpointMeta::new(
        cfg.model.clone(),
        cfg.recurrence,
        false,
        0,
        cfg.lr,
        cfg.seed,
        0,
        backend_label(cfg),
        cfg.precision.clone(),
    );
    ref_meta.run_id = cfg.run_id.clone();
    ref_meta.git_revision = option_env!("RECUR64_GIT_SHA").map(str::to_owned);
    save_training(
        &run_dir.reference_ckpt(),
        &reference_model,
        &reference_optim,
        &ref_meta,
    )?;
    let reference_model_id = read_model_id(&run_dir.reference_ckpt());

    let openings = load_openings(cfg)?;
    let mut snapshot_dir: PathBuf = run_dir.reference_ckpt();
    let mut snapshot_model_id = reference_model_id.clone();
    let mut cumulative_updates = 0u64;
    let mut total_positions = 0u64;
    let mut total_games_collected = 0u64;
    let mut cycles = Vec::new();
    let mut status = "completed".to_string();

    for cycle in 0..cfg.cycles {
        if cancel.is_cancelled() {
            status = "interrupted".into();
            break;
        }
        if Instant::now() > deadline {
            status = "budget_exhausted".into();
            break;
        }
        if cfg.position_budget.is_some_and(|b| total_positions >= b) {
            status = "position_budget_exhausted".into();
            break;
        }

        let cycle_start = Instant::now();
        let parent_model_id = snapshot_model_id.clone();
        let optimizer_step_start = cumulative_updates;

        // COLLECT
        let inference_model =
            model_io::load::<B::InnerBackend>(&snapshot_dir, &cfg.model, &inner_device)?;
        let batched = BatchedModel::new(inference_model, cfg.recurrence, inner_device.clone());
        let owner = InferenceOwner::spawn(
            batched,
            InferenceConfig {
                max_batch: cfg.max_inference_batch,
                batch_timeout: Duration::from_micros(cfg.batch_timeout_us),
                ..InferenceConfig::default()
            },
        );
        let evaluator = owner.evaluator();
        let collected = collect_parallel(cfg, &evaluator, cancel, deadline, total_games_collected);
        let inference_metrics = owner.metrics().snapshot();
        owner.shutdown();
        let records = collected?;
        let selfplay = selfplay_metrics(cfg, &records, inference_metrics);
        let games = records.len() as u64;
        total_games_collected += cfg.collection_shape()?.0 as u64;
        let positions: u64 = records.iter().map(|g| g.plies.len() as u64).sum();
        let new_trainable_positions: u64 = records
            .iter()
            .filter(|g| g.outcome.is_some())
            .map(|g| g.plies.len() as u64)
            .sum();
        let target_stats: Vec<(f64, f64)> = records
            .iter()
            .flat_map(|g| &g.plies)
            .map(|ply| {
                let entropy = -ply
                    .target
                    .iter()
                    .map(|(_, p)| {
                        let p = *p as f64;
                        if p > 0.0 { p * p.ln() } else { 0.0 }
                    })
                    .sum::<f64>();
                let top1 = ply
                    .target
                    .iter()
                    .map(|(_, p)| *p as f64)
                    .fold(0.0, f64::max);
                (entropy, top1)
            })
            .collect();
        let mean_target_entropy =
            target_stats.iter().map(|v| v.0).sum::<f64>() / target_stats.len().max(1) as f64;
        let mean_top1_visit_share =
            target_stats.iter().map(|v| v.1).sum::<f64>() / target_stats.len().max(1) as f64;
        if games < cfg.collection_shape()?.0 as u64 {
            status = "budget_exhausted_during_collect".into();
            break;
        }
        let mut header = ReplayHeader::new(
            cfg.run_id.clone(),
            snapshot_model_id.clone(),
            backend_label(cfg),
            cfg.precision.clone(),
        );
        header.git_revision = option_env!("RECUR64_GIT_SHA").map(str::to_owned);
        let mut writer = ReplayWriter::open_append(&run_dir.replay(), header, cfg.shard_max_games)?;
        for r in records {
            writer.push(r)?;
        }
        writer.finish()?;
        enforce_capacity(&run_dir.replay(), cfg.replay_max_positions)?;
        total_positions += positions;

        // AUDIT
        let audit = audit_dir(&run_dir.replay())?;
        if !audit.ok() {
            run_dir.update_status(cfg, RunStatus::Failed, Some("replay audit failed".into()))?;
            anyhow::bail!("replay audit failed: {:?}", audit.errors);
        }

        // TRAIN
        let store = ReplayStore::open(&run_dir.replay())?;
        let replay_positions = store.total_positions();
        let (train_model, mut optim, parent_meta) = load_training(
            &snapshot_dir,
            model_io::build::<B>(&cfg.model, &b_device),
            adamw::<B, _>(),
            &b_device,
        )?;
        anyhow::ensure!(
            parent_meta.update_counter == cumulative_updates
                && parent_meta.lr_schedule_step == cumulative_updates,
            "accepted optimizer trajectory mismatch at cycle {cycle}"
        );
        let (scheduled_updates, requested_examples) = cfg.reuse_updates(new_trainable_positions)?;
        let requested_updates = (requested_examples / cfg.effective_batch() as f64).ceil() as usize;
        // The schedule spans the whole pilot (cycles x per-cycle updates), not
        // just one cycle, so warmup/decay behave as intended.
        let planned = cfg
            .resolved_planned_updates()
            .max(cfg.max_updates as u64 * cfg.cycles.max(1) as u64);
        let warmup = cfg.warmup_updates.unwrap_or((planned / 10).clamp(10, 1000));
        let learner_cfg = LearnerConfig {
            batch_size: cfg.train_batch,
            accumulation_steps: cfg.accumulation_steps,
            max_updates: scheduled_updates,
            lr: cfg.lr,
            warmup_updates: warmup,
            planned_updates: planned,
            start_update: cumulative_updates,
            recurrence: cfg.recurrence,
            seed: cfg.seed.wrapping_add(cycle as u64),
            deadline: Some(deadline),
        };
        let (candidate_model_id, train_report) =
            match train_from_store(&store, train_model, &mut optim, &learner_cfg, &b_device) {
                Ok((trained, report)) => {
                    let mut meta = CheckpointMeta::new(
                        cfg.model.clone(),
                        cfg.recurrence,
                        false,
                        cumulative_updates + report.updates as u64,
                        cfg.lr,
                        cfg.seed,
                        0,
                        backend_label(cfg),
                        cfg.precision.clone(),
                    );
                    meta.run_id = cfg.run_id.clone();
                    meta.git_revision = option_env!("RECUR64_GIT_SHA").map(str::to_owned);
                    meta.update_counter = cumulative_updates + report.updates as u64;
                    meta.lr_schedule_step = cumulative_updates + report.updates as u64;
                    save_training(&run_dir.candidate_ckpt(), &trained, &optim, &meta)?;
                    (read_model_id(&run_dir.candidate_ckpt()), Some(report))
                }
                Err(e) if e.contains("no trainable") => {
                    copy_dir(&snapshot_dir, &run_dir.candidate_ckpt())?;
                    (snapshot_model_id.clone(), None)
                }
                Err(e) => return Err(anyhow::anyhow!(e)),
            };

        // EVALUATE
        let parent_infer =
            model_io::load::<B::InnerBackend>(&snapshot_dir, &cfg.model, &inner_device)?;
        let ref_infer = model_io::load::<B::InnerBackend>(
            &run_dir.reference_ckpt(),
            &cfg.model,
            &inner_device,
        )?;
        let cand_infer = model_io::load::<B::InnerBackend>(
            &run_dir.candidate_ckpt(),
            &cfg.model,
            &inner_device,
        )?;
        let ref_ev = SyncEvaluator::new(ref_infer, cfg.recurrence, inner_device.clone());
        let parent_ev = SyncEvaluator::new(parent_infer, cfg.recurrence, inner_device.clone());
        let cand_ev = SyncEvaluator::new(cand_infer, cfg.recurrence, inner_device.clone());
        let arena_cfg = ArenaConfig {
            games: cfg.arena_games,
            simulations: cfg.simulations_per_move,
            c_puct: cfg.c_puct,
            recurrence: cfg.recurrence,
            ply_cap: cfg.ply_cap,
            seed: cfg.seed.wrapping_add(cycle as u64),
            openings: openings.clone(),
        };
        let arena = run_arena(
            &parent_ev,
            &cand_ev,
            &parent_model_id,
            &candidate_model_id,
            &arena_cfg,
        )?;
        let reference_arena = run_arena(
            &ref_ev,
            &cand_ev,
            &reference_model_id,
            &candidate_model_id,
            &arena_cfg,
        )?;
        let raw = raw_policy_vs_random(
            &cand_ev,
            cfg.arena_games.max(4),
            cfg.temperature,
            cfg.ply_cap,
            cfg.seed.wrapping_add(cycle as u64),
            &openings,
        )?;
        let raw_parent = raw_policy_vs_parent(
            &cand_ev,
            &parent_ev,
            cfg.arena_games.max(4),
            cfg.ply_cap,
            cfg.seed.wrapping_add(cycle as u64),
            &openings,
        )?;

        // SNAPSHOT DECISION (conservative).
        let achieved_reuse = if new_trainable_positions == 0 {
            0.0
        } else {
            train_report
                .as_ref()
                .map(|r| r.examples_consumed as f64 / new_trainable_positions as f64)
                .unwrap_or(0.0)
        };
        let healthy = audit.ok()
            && selfplay.inference.errors == 0
            && achieved_reuse >= cfg.replay_reuse_target * 0.8
            && train_report
                .as_ref()
                .map(|r| {
                    r.updates > 0
                        && r.metrics.iter().all(|m| {
                            m.total_loss.is_finite()
                                && m.policy_loss.is_finite()
                                && m.wdl_loss.is_finite()
                                && m.grad_norm.is_finite()
                                && m.policy_entropy.is_finite()
                        })
                })
                .unwrap_or(false);
        let decision = match cfg.snapshot_policy {
            SnapshotPolicy::FrozenReference => "continue".to_string(),
            SnapshotPolicy::Conservative => {
                if healthy
                    && arena_informative_for_promotion(&arena)
                    && arena.candidate_score >= cfg.promotion_score_floor
                {
                    let snap = run_dir.checkpoints().join(format!("snapshot-{cycle:03}"));
                    copy_dir(&run_dir.candidate_ckpt(), &snap)?;
                    snapshot_dir = snap;
                    snapshot_model_id = candidate_model_id.clone();
                    cumulative_updates = optimizer_step_start
                        + train_report.as_ref().map(|r| r.updates as u64).unwrap_or(0);
                    "promote".to_string()
                } else {
                    "continue".to_string()
                }
            }
        };

        let reuse_ratio = if new_trainable_positions > 0 {
            train_report
                .as_ref()
                .map(|r| r.examples_consumed as f64 / new_trainable_positions as f64)
                .unwrap_or(0.0)
        } else {
            0.0
        };
        let reuse_shortfall_reason = if requested_updates > scheduled_updates {
            Some("max_updates safety cap".to_string())
        } else if train_report
            .as_ref()
            .is_some_and(|r| r.updates < scheduled_updates)
        {
            Some("run deadline or replay exhaustion".to_string())
        } else if reuse_ratio < cfg.replay_reuse_target * 0.8 {
            Some("effective-batch rounding or insufficient trainable positions".to_string())
        } else {
            None
        };

        let cycle_report = CycleReport {
            cycle,
            games,
            positions,
            new_trainable_positions,
            mean_target_entropy,
            mean_top1_visit_share,
            requested_examples,
            requested_updates,
            scheduled_updates,
            selfplay,
            audit_ok: audit.ok(),
            train: train_report,
            arena: Some(arena),
            reference_arena: Some(reference_arena),
            raw: Some(raw),
            raw_parent: Some(raw_parent),
            snapshot_model_id: snapshot_model_id.clone(),
            candidate_model_id: candidate_model_id.clone(),
            replay_positions,
            reuse_ratio,
            reuse_shortfall_reason,
            decision: decision.clone(),
            wall_secs: cycle_start.elapsed().as_secs_f64(),
        };
        run_dir.append_lineage(&LineageRecord {
            cycle,
            run_id: cfg.run_id.clone(),
            parent_model_id: parent_model_id.clone(),
            candidate_model_id: candidate_model_id.clone(),
            promoted_model_id: snapshot_model_id.clone(),
            replay_model_ids: vec![parent_model_id],
            new_positions: positions,
            new_trainable_positions,
            examples_consumed: cycle_report
                .train
                .as_ref()
                .map(|t| t.examples_consumed)
                .unwrap_or(0),
            optimizer_step_start,
            optimizer_step_end: optimizer_step_start
                + cycle_report
                    .train
                    .as_ref()
                    .map(|t| t.updates as u64)
                    .unwrap_or(0),
            wall_clock_secs: cycle_report.wall_secs,
            arena_candidate_score: cycle_report.arena.as_ref().map(|a| a.candidate_score),
            snapshot_decision: decision,
            config_hash: cfg.config_hash(),
            scientific_config_hash: cfg.scientific_config_hash(),
            resolved_config_hash: cfg.resolved_config_hash(),
            git_revision: option_env!("RECUR64_GIT_SHA").map(str::to_owned),
            git_branch: option_env!("RECUR64_GIT_BRANCH").map(str::to_owned),
            seed: cfg.seed,
        })?;
        println!(
            "cycle {cycle}: games={games} positions={positions} decision={} arena_score={:.3} wall={:.1}s",
            cycle_report.decision,
            cycle_report
                .arena
                .as_ref()
                .map(|a| a.candidate_score)
                .unwrap_or(0.0),
            cycle_report.wall_secs
        );
        cycles.push(cycle_report);
    }

    let report = PilotReport {
        run_id: cfg.run_id.clone(),
        status: status.clone(),
        cycles,
        total_positions,
        elapsed_secs: started.elapsed().as_secs_f64(),
    };
    std::fs::create_dir_all(run_dir.report())?;
    std::fs::write(
        run_dir.report().join("pilot.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    let final_status = if status == "completed" {
        RunStatus::Completed
    } else {
        RunStatus::Interrupted
    };
    run_dir.update_status(cfg, final_status, Some(status))?;
    Ok(report)
}

#[cfg(test)]
mod promotion_tests {
    use super::*;
    #[test]
    fn draw_only_and_all_truncated_arenas_are_uninformative() {
        let mut arena = ArenaResult {
            games: 4,
            candidate_wins: 0,
            reference_wins: 0,
            draws: 4,
            truncated: 0,
            candidate_score: 0.5,
            score_ci_low: 0.5,
            score_ci_high: 0.5,
            opening_count: 1,
            terminations: Default::default(),
            model_reference: "parent".into(),
            model_candidate: "candidate".into(),
        };
        assert!(!arena_informative_for_promotion(&arena));
        arena.draws = 0;
        arena.truncated = 4;
        assert!(!arena_informative_for_promotion(&arena));
        arena.truncated = 0;
        arena.candidate_wins = 1;
        assert!(arena_informative_for_promotion(&arena));
    }
}
