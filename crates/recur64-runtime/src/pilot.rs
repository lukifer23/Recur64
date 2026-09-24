//! Bounded multi-cycle pilot: COLLECT → AUDIT → TRAIN → EVALUATE, repeated.
//!
//! Each cycle freezes a snapshot for self-play, collects games, audits them,
//! trains a candidate, and evaluates it against its parent and the frozen
//! reference. A conservative snapshot policy decides whether the candidate
//! becomes the next self-play snapshot. The loop is bounded by cycle count,
//! wall-clock, and position budget — it is not an autonomous long run.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

use recur64_eval::{ArenaConfig, ArenaResult, OpeningSuite, run_arena};
use recur64_model::checkpoint::{CheckpointMeta, load_training, save_training};
use recur64_model::train::adamw;

use crate::cancel::CancelToken;
use crate::config::{PROMOTION_RULE_VERSION, RunConfig, SnapshotPolicy};
use crate::coordinator::{SelfPlayMetrics, collect_parallel, selfplay_metrics};
use crate::eval_policy::{
    PolicyDiagnostics, RawMatchResult, RawParentResult, policy_diagnostics, raw_policy_vs_parent,
    raw_policy_vs_random,
};
use crate::gpu_telemetry::{self, GpuSamples};
use crate::inference::{BatchedModel, InferenceConfig, InferenceOwner, MetricsSnapshot};
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
    /// First replay game id of this cycle (ids are contiguous per cycle).
    pub first_game_id: u64,
    pub requested_examples: f64,
    pub requested_updates: usize,
    pub scheduled_updates: usize,
    /// Updates the learner actually completed.
    pub completed_updates: usize,
    pub max_updates: usize,
    /// The `max_updates` safety cap reduced the requested work this cycle.
    pub max_updates_cap_bound: bool,
    pub selfplay: SelfPlayMetrics,
    pub audit_ok: bool,
    pub train: Option<TrainReport>,
    pub arena: Option<ArenaResult>,
    pub reference_arena: Option<ArenaResult>,
    /// True when the parent was the frozen reference, so `reference_arena` is
    /// the parent arena (same models, same deterministic games), not rerun.
    pub reference_arena_is_parent_arena: bool,
    pub raw: Option<RawMatchResult>,
    pub raw_parent: Option<RawParentResult>,
    pub parent_model_id: String,
    pub snapshot_model_id: String,
    pub candidate_model_id: String,
    /// Positions in result games across the active replay (sampleable).
    pub replay_positions: u64,
    pub replay_total_games: u64,
    pub reuse_ratio: f64,
    /// `examples_consumed / all new plies` (keeps truncated mass visible).
    pub reuse_ratio_all_new_positions: f64,
    pub reuse_shortfall_reason: Option<String>,
    pub decision: String,
    /// Why a conservative decision held (empty on promote).
    pub hold_reasons: Vec<String>,
    pub optimizer_step_start: u64,
    /// Accepted-trajectory optimizer step after this cycle's decision.
    pub accepted_optimizer_step_after: u64,
    /// Inference owners created / simultaneously resident during EVALUATE.
    pub eval_owners_spawned: u32,
    pub eval_max_resident_owners: u32,
    pub eval_inference: Vec<(String, MetricsSnapshot)>,
    pub gpu: CycleGpu,
    pub collect_secs: f64,
    pub train_secs: f64,
    pub eval_secs: f64,
    pub wall_secs: f64,
}

/// Per-phase GPU telemetry for one cycle (empty on CPU runs).
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct CycleGpu {
    pub vram_at_cycle_start_mb: Option<u64>,
    pub collect: GpuSamples,
    pub train: GpuSamples,
    pub eval: GpuSamples,
}

/// The three logical models compared in one cycle's evaluation.
#[derive(Debug, Clone, Copy)]
pub struct EvalModels<'a> {
    pub parent_dir: &'a Path,
    pub parent_model_id: &'a str,
    pub candidate_dir: &'a Path,
    pub candidate_model_id: &'a str,
    pub reference_dir: &'a Path,
    pub reference_model_id: &'a str,
}

/// Every evaluation of one candidate, plus owner-lifecycle accounting.
#[derive(Debug, Clone, serde::Serialize)]
pub struct EvalOutcome {
    pub arena: ArenaResult,
    pub reference_arena: ArenaResult,
    /// True when the parent was the frozen reference, so `reference_arena` is
    /// the parent arena (same models, same deterministic games), not rerun.
    pub reference_arena_is_parent_arena: bool,
    pub raw: RawMatchResult,
    pub raw_parent: RawParentResult,
    pub owners_spawned: u32,
    pub max_resident_owners: u32,
    /// Inference metrics per spawned owner (role, snapshot).
    pub inference: Vec<(String, MetricsSnapshot)>,
}

/// Experiment identity written to `identity.json` before cycle 0.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PilotIdentity {
    pub run_id: String,
    pub scientific_config_hash: String,
    pub resolved_config_hash: String,
    pub scientific_identity: serde_json::Value,
    pub reference_model_id: String,
    pub reference_source: String,
    pub opening_suite_digest: Option<String>,
    pub promotion_rule: String,
    pub git_revision: Option<String>,
    pub git_branch: Option<String>,
}

/// Baseline T0: the reference network before any training.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BaselineReport {
    pub reference_model_id: String,
    pub raw_vs_random: RawMatchResult,
    pub policy: PolicyDiagnostics,
}

/// Whole-pilot report.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PilotReport {
    pub run_id: String,
    pub status: String,
    pub identity: PilotIdentity,
    pub baseline: Option<BaselineReport>,
    pub cycles: Vec<CycleReport>,
    pub total_positions: u64,
    pub elapsed_secs: f64,
    /// Soft wall budget and how far the run went past it (evaluation does not
    /// yet check the deadline, so an overrun is recorded, never hidden).
    pub run_budget_secs: f64,
    pub budget_overrun_secs: f64,
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

/// Conservative promotion on the candidate-vs-parent arena. Returns the hold
/// reasons; empty means promote. A score of 0.5 is a tie and never promotes,
/// and nothing promotes without `promotion_min_decisive_games` won-or-lost
/// games (see [`PROMOTION_RULE_VERSION`]).
fn promotion_holds(cfg: &RunConfig, healthy: bool, arena: &ArenaResult) -> Vec<String> {
    let mut reasons = Vec::new();
    if !healthy {
        reasons.push("unhealthy".to_string());
    }
    if arena.decisive_games < cfg.promotion_min_decisive_games.max(1) {
        reasons.push("arena_uninformative".to_string());
    } else if arena.candidate_score <= 0.5 || arena.candidate_score < cfg.promotion_score_floor {
        reasons.push("score_not_above_parent".to_string());
    }
    reasons
}

/// Spawn a batched inference owner for a checkpoint using the run schedule.
pub fn spawn_owner<B: Backend>(
    dir: &Path,
    cfg: &RunConfig,
    device: &B::Device,
) -> anyhow::Result<InferenceOwner> {
    let model = model_io::load::<B>(dir, &cfg.model, device)?;
    Ok(InferenceOwner::spawn(
        BatchedModel::new(model, cfg.recurrence, device.clone()),
        InferenceConfig {
            max_batch: cfg.max_inference_batch,
            batch_timeout: Duration::from_micros(cfg.batch_timeout_us),
            ..InferenceConfig::default()
        },
    ))
}

/// Evaluate a candidate against its parent, the frozen reference, and random
/// play. Match order and seeds are part of the evaluation contract:
/// searched candidate-vs-parent, searched candidate-vs-reference (reused when
/// parent == reference), raw candidate-vs-random, raw candidate-vs-parent, all
/// seeded `cfg.seed + cycle`.
pub fn evaluate_candidate<B: Backend>(
    cfg: &RunConfig,
    models: &EvalModels<'_>,
    cycle: u32,
    openings: &[String],
    eval_concurrency: usize,
    device: &B::Device,
) -> anyhow::Result<EvalOutcome> {
    let parent_owner = spawn_owner::<B>(models.parent_dir, cfg, device)?;
    let cand_owner = spawn_owner::<B>(models.candidate_dir, cfg, device)?;
    let ref_owner = spawn_owner::<B>(models.reference_dir, cfg, device)?;
    let (owners_spawned, max_resident_owners) = (3, 3);
    let (parent_ev, cand_ev, ref_ev) = (
        parent_owner.evaluator(),
        cand_owner.evaluator(),
        ref_owner.evaluator(),
    );
    let arena_cfg = ArenaConfig {
        games: cfg.arena_games,
        simulations: cfg.simulations_per_move,
        c_puct: cfg.c_puct,
        recurrence: cfg.recurrence,
        ply_cap: cfg.ply_cap,
        seed: cfg.seed.wrapping_add(cycle as u64),
        openings: openings.to_vec(),
        concurrency: eval_concurrency,
    };
    let arena = run_arena(
        &parent_ev,
        &cand_ev,
        models.parent_model_id,
        models.candidate_model_id,
        &arena_cfg,
    )?;
    // Until the first promotion the parent *is* the frozen reference, so
    // the longitudinal arena would replay the identical deterministic
    // comparison. Reuse it and say so.
    let reference_arena_is_parent_arena = models.parent_model_id == models.reference_model_id;
    let reference_arena = if reference_arena_is_parent_arena {
        arena.clone()
    } else {
        run_arena(
            &ref_ev,
            &cand_ev,
            models.reference_model_id,
            models.candidate_model_id,
            &arena_cfg,
        )?
    };
    let raw = raw_policy_vs_random(
        &cand_ev,
        cfg.arena_games.max(4),
        cfg.temperature,
        cfg.ply_cap,
        cfg.seed.wrapping_add(cycle as u64),
        openings,
        eval_concurrency,
    )?;
    let raw_parent = raw_policy_vs_parent(
        &cand_ev,
        &parent_ev,
        cfg.arena_games.max(4),
        cfg.ply_cap,
        cfg.seed.wrapping_add(cycle as u64),
        openings,
        eval_concurrency,
    )?;
    drop((parent_ev, cand_ev, ref_ev));
    let inference = vec![
        ("parent".to_string(), parent_owner.metrics().snapshot()),
        ("candidate".to_string(), cand_owner.metrics().snapshot()),
        ("reference".to_string(), ref_owner.metrics().snapshot()),
    ];
    parent_owner.shutdown();
    cand_owner.shutdown();
    ref_owner.shutdown();
    Ok(EvalOutcome {
        inference,
        arena,
        reference_arena,
        reference_arena_is_parent_arena,
        raw,
        raw_parent,
        owners_spawned,
        max_resident_owners,
    })
}

/// Install the frozen reference (if configured) or a fresh seeded one, and
/// return its model id.
fn install_reference<B: AutodiffBackend>(
    cfg: &RunConfig,
    run_dir: &RunDir,
    device: &B::Device,
) -> anyhow::Result<String> {
    if let Some(src) = &cfg.reference_checkpoint {
        let src = Path::new(src);
        let meta: CheckpointMeta = serde_json::from_slice(&std::fs::read(src.join("meta.json"))?)?;
        anyhow::ensure!(
            serde_json::to_value(&meta.model)? == serde_json::to_value(&cfg.model)?,
            "reference checkpoint model config differs from the run config"
        );
        anyhow::ensure!(
            meta.recurrence == cfg.recurrence,
            "reference checkpoint recurrence differs from the run config"
        );
        anyhow::ensure!(
            meta.update_counter == 0 && meta.lr_schedule_step == 0,
            "reference checkpoint is not an untrained step-0 checkpoint"
        );
        if let Some(expected) = &cfg.reference_model_id {
            anyhow::ensure!(
                &meta.model_id == expected,
                "reference model_id {} does not match configured {expected}",
                meta.model_id
            );
        }
        copy_dir(src, &run_dir.reference_ckpt())?;
        let copied = read_model_id(&run_dir.reference_ckpt());
        anyhow::ensure!(
            copied == meta.model_id,
            "reference copy changed its model_id"
        );
        return Ok(copied);
    }
    anyhow::ensure!(
        cfg.reference_model_id.is_none(),
        "reference_model_id is set but reference_checkpoint is not"
    );
    let reference_model = model_io::build::<B>(&cfg.model, device);
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
    Ok(read_model_id(&run_dir.reference_ckpt()))
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
    B::seed(&b_device, cfg.seed);
    let (games_per_cycle, eval_concurrency) = cfg.collection_shape()?;
    let gpu_on = cfg.device == "cuda";

    let reference_model_id = install_reference::<B>(cfg, run_dir, &b_device)?;
    let scientific_config_hash = cfg.scientific_config_hash()?;
    let resolved_config_hash = cfg.resolved_config_hash();
    let identity = PilotIdentity {
        run_id: cfg.run_id.clone(),
        scientific_config_hash: scientific_config_hash.clone(),
        resolved_config_hash: resolved_config_hash.clone(),
        scientific_identity: cfg.scientific_identity()?,
        reference_model_id: reference_model_id.clone(),
        reference_source: cfg
            .reference_checkpoint
            .clone()
            .unwrap_or_else(|| "fresh seeded init".into()),
        opening_suite_digest: cfg.opening_suite_digest()?,
        promotion_rule: PROMOTION_RULE_VERSION.to_string(),
        git_revision: option_env!("RECUR64_GIT_SHA").map(str::to_owned),
        git_branch: option_env!("RECUR64_GIT_BRANCH").map(str::to_owned),
    };
    std::fs::write(
        run_dir.root.join("identity.json"),
        serde_json::to_vec_pretty(&identity)?,
    )?;
    println!(
        "identity: scientific={} resolved={} reference={} git={:?}",
        identity.scientific_config_hash,
        identity.resolved_config_hash,
        identity.reference_model_id,
        identity.git_revision
    );

    let openings = load_openings(cfg)?;

    // BASELINE T0: the raw reference policy, before any training.
    let baseline = {
        let owner = spawn_owner::<B::InnerBackend>(&run_dir.reference_ckpt(), cfg, &inner_device)?;
        let ev = owner.evaluator();
        let raw_vs_random = raw_policy_vs_random(
            &ev,
            cfg.arena_games.max(4),
            cfg.temperature,
            cfg.ply_cap,
            cfg.seed,
            &openings,
            eval_concurrency,
        )?;
        let policy = policy_diagnostics(&ev, &openings)?;
        drop(ev);
        owner.shutdown();
        BaselineReport {
            reference_model_id: reference_model_id.clone(),
            raw_vs_random,
            policy,
        }
    };
    std::fs::write(
        run_dir.eval().join("baseline-t0.json"),
        serde_json::to_vec_pretty(&baseline)?,
    )?;

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
        let mut gpu = CycleGpu {
            vram_at_cycle_start_mb: gpu_on
                .then(gpu_telemetry::sample_gpu)
                .flatten()
                .map(|s| s.0),
            ..CycleGpu::default()
        };
        let first_game_id = total_games_collected;
        let owner = spawn_owner::<B::InnerBackend>(&snapshot_dir, cfg, &inner_device)?;
        let evaluator = owner.evaluator();
        let ((collected, collect_secs), collect_gpu) = gpu_telemetry::monitor(gpu_on, || {
            let collect_start = Instant::now();
            let collected = collect_parallel(cfg, &evaluator, cancel, deadline, first_game_id);
            (collected, collect_start.elapsed().as_secs_f64())
        });
        gpu.collect = collect_gpu;
        let inference_metrics = owner.metrics().snapshot();
        drop(evaluator);
        owner.shutdown();
        let records = collected?;
        let selfplay = selfplay_metrics(cfg, &records, inference_metrics);
        let games = records.len() as u64;
        total_games_collected += games_per_cycle as u64;
        let positions: u64 = records.iter().map(|g| g.plies.len() as u64).sum();
        let new_trainable_positions = selfplay.trainable_positions;
        if games < games_per_cycle as u64 {
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
        let train_start = Instant::now();
        let store = ReplayStore::open(&run_dir.replay())?;
        let replay_positions = store.total_positions();
        let replay_total_games = store.total_games() as u64;
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
        let plan = cfg.update_plan(new_trainable_positions)?;
        let (scheduled_updates, requested_examples, requested_updates) = (
            plan.scheduled_updates,
            plan.requested_examples,
            plan.requested_updates,
        );
        // One schedule for the whole trajectory; the same function feeds the
        // scientific hash, so the hash names the schedule actually trained.
        let (warmup, planned) = cfg.lr_schedule();
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
            current_cycle_first_game_id: Some(first_game_id),
            games_per_cycle: games_per_cycle as u64,
        };
        let (trained, train_gpu) = gpu_telemetry::monitor(gpu_on, || {
            train_from_store(&store, train_model, &mut optim, &learner_cfg, &b_device)
        });
        gpu.train = train_gpu;
        let (candidate_model_id, train_report) = match trained {
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
        let train_secs = train_start.elapsed().as_secs_f64();

        // EVALUATE: batched owners, games in parallel, reports in game order.
        let eval_start = Instant::now();
        let candidate_dir = run_dir.candidate_ckpt();
        let reference_dir = run_dir.reference_ckpt();
        let models = EvalModels {
            parent_dir: &snapshot_dir,
            parent_model_id: &parent_model_id,
            candidate_dir: &candidate_dir,
            candidate_model_id: &candidate_model_id,
            reference_dir: &reference_dir,
            reference_model_id: &reference_model_id,
        };
        let (outcome, eval_gpu) = gpu_telemetry::monitor(gpu_on, || {
            evaluate_candidate::<B::InnerBackend>(
                cfg,
                &models,
                cycle,
                &openings,
                eval_concurrency,
                &inner_device,
            )
        });
        gpu.eval = eval_gpu;
        let EvalOutcome {
            arena,
            reference_arena,
            reference_arena_is_parent_arena,
            raw,
            raw_parent,
            owners_spawned: eval_owners_spawned,
            max_resident_owners: eval_max_resident_owners,
            inference: eval_inference,
        } = outcome?;
        let eval_secs = eval_start.elapsed().as_secs_f64();

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
        let hold_reasons = match cfg.snapshot_policy {
            SnapshotPolicy::FrozenReference => vec!["frozen_reference_policy".to_string()],
            SnapshotPolicy::Conservative => promotion_holds(cfg, healthy, &arena),
        };
        let decision = match cfg.snapshot_policy {
            SnapshotPolicy::FrozenReference => "continue".to_string(),
            SnapshotPolicy::Conservative => {
                if hold_reasons.is_empty() {
                    let snap = run_dir.checkpoints().join(format!("snapshot-{cycle:03}"));
                    copy_dir(&run_dir.candidate_ckpt(), &snap)?;
                    snapshot_dir = snap;
                    snapshot_model_id = candidate_model_id.clone();
                    cumulative_updates = optimizer_step_start
                        + train_report.as_ref().map(|r| r.updates as u64).unwrap_or(0);
                    "promote".to_string()
                } else {
                    "hold".to_string()
                }
            }
        };

        let reuse_ratio = achieved_reuse;
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
            first_game_id,
            requested_examples,
            requested_updates,
            scheduled_updates,
            completed_updates: train_report.as_ref().map(|r| r.updates).unwrap_or(0),
            max_updates: plan.max_updates,
            max_updates_cap_bound: plan.cap_bound,
            selfplay,
            audit_ok: audit.ok(),
            train: train_report,
            arena: Some(arena),
            reference_arena: Some(reference_arena),
            reference_arena_is_parent_arena,
            raw: Some(raw),
            raw_parent: Some(raw_parent),
            parent_model_id: parent_model_id.clone(),
            snapshot_model_id: snapshot_model_id.clone(),
            candidate_model_id: candidate_model_id.clone(),
            replay_positions,
            replay_total_games,
            reuse_ratio,
            reuse_ratio_all_new_positions: 0.0,
            reuse_shortfall_reason,
            decision: decision.clone(),
            hold_reasons,
            optimizer_step_start,
            accepted_optimizer_step_after: cumulative_updates,
            eval_owners_spawned,
            eval_max_resident_owners,
            eval_inference,
            gpu,
            collect_secs,
            train_secs,
            eval_secs,
            wall_secs: cycle_start.elapsed().as_secs_f64(),
        };
        let consumed = cycle_report
            .train
            .as_ref()
            .map(|t| t.examples_consumed)
            .unwrap_or(0);
        let cycle_report = CycleReport {
            reuse_ratio_all_new_positions: if positions > 0 {
                consumed as f64 / positions as f64
            } else {
                0.0
            },
            ..cycle_report
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
            examples_consumed: consumed,
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
            scientific_config_hash: scientific_config_hash.clone(),
            resolved_config_hash: resolved_config_hash.clone(),
            git_revision: option_env!("RECUR64_GIT_SHA").map(str::to_owned),
            git_branch: option_env!("RECUR64_GIT_BRANCH").map(str::to_owned),
            seed: cfg.seed,
        })?;
        println!(
            "cycle {cycle}: games={games} positions={positions} trainable={new_trainable_positions} updates={} decision={} {:?} parent_arena={:.3} (decisive {}) wall={:.1}s (collect {:.0}s train {:.0}s eval {:.0}s)",
            cycle_report.train.as_ref().map(|t| t.updates).unwrap_or(0),
            cycle_report.decision,
            cycle_report.hold_reasons,
            cycle_report
                .arena
                .as_ref()
                .map(|a| a.candidate_score)
                .unwrap_or(0.0),
            cycle_report
                .arena
                .as_ref()
                .map(|a| a.decisive_games)
                .unwrap_or(0),
            cycle_report.wall_secs,
            collect_secs,
            train_secs,
            eval_secs
        );
        // Persist a partial report after every cycle so a crash keeps evidence.
        std::fs::create_dir_all(run_dir.report())?;
        std::fs::write(
            run_dir.report().join(format!("cycle-{cycle:03}.json")),
            serde_json::to_vec_pretty(&cycle_report)?,
        )?;
        cycles.push(cycle_report);
    }
    let elapsed_secs = started.elapsed().as_secs_f64();
    let run_budget_secs = (cfg.run_budget_minutes.max(1) * 60) as f64;

    let report = PilotReport {
        run_id: cfg.run_id.clone(),
        status: status.clone(),
        identity,
        baseline: Some(baseline),
        cycles,
        total_positions,
        elapsed_secs,
        run_budget_secs,
        budget_overrun_secs: (elapsed_secs - run_budget_secs).max(0.0),
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

    fn cfg() -> RunConfig {
        RunConfig::from_toml_str(
            "run_id = 't'\n[model]\nwidth = 32\nheads = 4\nffn = 64\ninput_blocks = 0\ncore_blocks = 1\noutput_blocks = 0\n",
        )
        .unwrap()
    }

    fn arena(wins: u32, losses: u32, draws: u32, truncated: u32) -> ArenaResult {
        let games = wins + losses + draws + truncated;
        let decided = games - truncated;
        ArenaResult {
            games,
            candidate_wins: wins,
            reference_wins: losses,
            draws,
            truncated,
            candidate_score: if decided == 0 {
                0.5
            } else {
                (wins as f64 + 0.5 * draws as f64) / decided as f64
            },
            decisive_games: wins + losses,
            informative: wins + losses > 0,
            score_ci_low: 0.0,
            score_ci_high: 1.0,
            opening_count: 1,
            terminations: Default::default(),
            model_reference: "parent".into(),
            model_candidate: "candidate".into(),
        }
    }

    #[test]
    fn promotion_requires_decisive_evidence_and_beating_the_parent() {
        let c = cfg();
        assert_eq!(c.promotion_min_decisive_games, 4);
        let holds = |a: &ArenaResult| promotion_holds(&c, true, a);
        assert_eq!(
            holds(&arena(0, 0, 20, 0)),
            ["arena_uninformative"],
            "draw-only"
        );
        assert_eq!(
            holds(&arena(0, 0, 0, 20)),
            ["arena_uninformative"],
            "all truncated"
        );
        assert_eq!(
            holds(&arena(1, 0, 19, 0)),
            ["arena_uninformative"],
            "1 decisive"
        );
        // 1 loss + 19 draws (0.475) cleared the old 0.35 floor.
        assert_eq!(holds(&arena(0, 1, 19, 0)), ["arena_uninformative"]);
        assert_eq!(
            holds(&arena(2, 3, 15, 0)),
            ["score_not_above_parent"],
            "0.475"
        );
        assert_eq!(
            holds(&arena(2, 2, 16, 0)),
            ["score_not_above_parent"],
            "tie"
        );
        assert!(holds(&arena(3, 1, 16, 0)).is_empty(), "0.55, 4 decisive");
        assert_eq!(
            promotion_holds(&c, false, &arena(3, 1, 16, 0)),
            ["unhealthy"]
        );
    }
}
