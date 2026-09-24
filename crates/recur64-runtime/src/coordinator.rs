//! Bounded run coordinator: COLLECT → AUDIT → TRAIN → EVALUATE → REPORT.
//!
//! One GPU owner is used for self-play; training and arena run after it is shut
//! down, so the single accelerator is never fought over. Every phase is bounded
//! by the config and the wall-clock budget, and interruption leaves a truthful
//! status and a recoverable checkpoint.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::{Duration, Instant};

use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

use recur64_core::{GameState, StandardMove};
use recur64_eval::{ArenaConfig, ArenaResult, OpeningSuite, run_arena};
use recur64_model::checkpoint::{CheckpointMeta, save_training};
use recur64_model::train::adamw;

use crate::SyncEvaluator;
use crate::cancel::CancelToken;
use crate::config::RunConfig;
use crate::inference::{BatchedModel, InferenceConfig, InferenceOwner, MetricsSnapshot};
use crate::learner::{LearnerConfig, TrainReport, train_from_games};
use crate::model_io;
use crate::replay::{GameRecord, ReplayHeader, ReplayWriter, SearchRecord, audit_dir};
use crate::run_dir::{RunDir, RunStatus};
use recur64_search::{Evaluator, SelfPlayConfig, play_game_from};

/// Self-play concurrency and data-health metrics. `peak_in_flight_evaluations`
/// is the direct evidence that games executed *concurrently* rather than
/// sequentially: a configured `active_games`/`cpu_workers` only counts if this
/// rises above 1. The termination/outcome fields are the data-health record
/// required before trusting a learning run.
///
/// Illegal selected actions are impossible by construction: `play_game_from`
/// only applies moves returned by the rules engine's legal-action list, and the
/// replay audit re-verifies legality on read.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SelfPlayMetrics {
    pub games_requested: u64,
    pub games_completed: u64,
    pub failed_games: u64,
    pub concurrent_games: usize,
    pub requested_active_games: u32,
    pub cpu_workers: usize,
    pub games: u64,
    pub plies: u64,
    pub mean_game_plies: f64,
    pub peak_in_flight_evaluations: usize,
    /// Termination label -> count (checkmate, stalemate, threefold_repetition,
    /// fifty_move_rule, insufficient_material, truncated, aborted).
    pub terminations: BTreeMap<String, u64>,
    /// Completed-game results from the board's perspective.
    pub white_wins: u64,
    pub draws: u64,
    pub black_wins: u64,
    /// Games without a terminal result (truncated/aborted); excluded by the
    /// learner and never labelled as draws.
    pub truncated: u64,
    pub draw_share: f64,
    pub repetition_share: f64,
    /// Plies in games with a result (the learner's reuse denominator).
    pub trainable_positions: u64,
    pub trainable_games: u64,
    /// Search-target health over every generated ply and over trainable plies.
    pub target_health: TargetHealth,
    pub inference: MetricsSnapshot,
}

/// Mean visit-target entropy and top-1 share over a set of plies.
#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct TargetStats {
    pub positions: u64,
    pub mean_entropy: f64,
    pub mean_top1_visit_share: f64,
}

/// Target health for all generated plies and for the plies the learner can
/// actually use (games with a result). Computed post hoc from records.
#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct TargetHealth {
    pub all: TargetStats,
    pub trainable: TargetStats,
}

/// Compute [`TargetHealth`] from game records (no search hot-loop cost).
pub fn target_health(records: &[GameRecord]) -> TargetHealth {
    let mut sums = [(0u64, 0f64, 0f64); 2];
    for game in records {
        for ply in &game.plies {
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
            let slots: &[usize] = if game.outcome.is_some() {
                &[0, 1]
            } else {
                &[0]
            };
            for &i in slots {
                sums[i].0 += 1;
                sums[i].1 += entropy;
                sums[i].2 += top1;
            }
        }
    }
    let stats = |(n, e, t): (u64, f64, f64)| TargetStats {
        positions: n,
        mean_entropy: e / n.max(1) as f64,
        mean_top1_visit_share: t / n.max(1) as f64,
    };
    TargetHealth {
        all: stats(sums[0]),
        trainable: stats(sums[1]),
    }
}

/// Build self-play metrics from collected game records.
pub(crate) fn selfplay_metrics(
    cfg: &RunConfig,
    records: &[GameRecord],
    inference: MetricsSnapshot,
) -> SelfPlayMetrics {
    let mut terminations: BTreeMap<String, u64> = BTreeMap::new();
    let (mut white_wins, mut draws, mut black_wins, mut truncated) = (0u64, 0u64, 0u64, 0u64);
    for r in records {
        *terminations.entry(r.termination.clone()).or_insert(0) += 1;
        match r.outcome {
            Some(0) => white_wins += 1,
            Some(1) => draws += 1,
            Some(2) => black_wins += 1,
            _ => truncated += 1,
        }
    }
    let plies: u64 = records.iter().map(|r| r.plies.len() as u64).sum();
    let games = records.len() as u64;
    let repetition_share = terminations
        .get("threefold_repetition")
        .copied()
        .unwrap_or(0) as f64
        / games.max(1) as f64;
    SelfPlayMetrics {
        games_requested: cfg.collection_shape().map(|s| s.0 as u64).unwrap_or(0),
        games_completed: games,
        failed_games: 0,
        concurrent_games: cfg.collection_shape().map(|s| s.1).unwrap_or(0),
        requested_active_games: cfg.active_games,
        cpu_workers: cfg.cpu_workers,
        games,
        plies,
        mean_game_plies: if games == 0 {
            0.0
        } else {
            plies as f64 / games as f64
        },
        peak_in_flight_evaluations: inference.peak_in_flight,
        terminations,
        white_wins,
        draws,
        black_wins,
        truncated,
        draw_share: draws as f64 / games.max(1) as f64,
        repetition_share,
        trainable_positions: records
            .iter()
            .filter(|g| g.outcome.is_some())
            .map(|g| g.plies.len() as u64)
            .sum(),
        trainable_games: games - truncated,
        target_health: target_health(records),
        inference,
    }
}

/// Outcome of a bounded run.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RunReport {
    pub run_id: String,
    pub status: String,
    pub games_collected: u64,
    pub audit_ok: bool,
    pub audit_errors: Vec<String>,
    pub train: Option<TrainReport>,
    pub arena: Option<ArenaResult>,
    pub reference_model_id: String,
    pub candidate_model_id: String,
    pub inference: Option<MetricsSnapshot>,
    pub selfplay: Option<SelfPlayMetrics>,
    pub elapsed_secs: f64,
}

fn read_model_id(dir: &Path) -> String {
    let path = dir.join("meta.json");
    std::fs::read(path)
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .and_then(|v| {
            v.get("model_id")
                .and_then(|m| m.as_str())
                .map(|s| s.to_string())
        })
        .unwrap_or_default()
}

/// Play `cfg.active_games` games with `cfg.active_games` concurrent worker
/// threads, pulling game indices from a shared atomic queue. `active_games` is
/// therefore the real concurrency: many leaf requests are in flight and the
/// batcher can coalesce them.
///
/// Both `run` and `collect_only` use this single path, so a configured
/// concurrency value always means real concurrent execution. (The Phase 2
/// `collect_only` played games sequentially, and the Phase 2 coordinator
/// bounded concurrency by `cpu_workers`, producing tiny batches.)
pub(crate) fn collect_parallel(
    cfg: &RunConfig,
    evaluator: &dyn Evaluator,
    cancel: &CancelToken,
    deadline: Instant,
    game_id_base: u64,
) -> anyhow::Result<Vec<GameRecord>> {
    let sp = SelfPlayConfig {
        simulations_per_move: cfg.simulations_per_move,
        c_puct: cfg.c_puct,
        temperature: cfg.temperature,
        ply_cap: cfg.ply_cap,
        recurrence: cfg.recurrence,
        argmax_after_ply: cfg.argmax_after_ply,
        root_dirichlet_alpha: cfg.root_dirichlet_alpha,
        root_dirichlet_epsilon: cfg.root_dirichlet_epsilon,
    };
    let search_record = SearchRecord {
        simulations: cfg.simulations_per_move,
        c_puct: cfg.c_puct,
        temperature: cfg.temperature,
        recurrence: cfg.recurrence,
    };

    let (games_total, concurrency) = cfg.collection_shape()?;
    let games_total = games_total as usize;
    let start_state = match &cfg.start_fen {
        Some(f) => GameState::from_fen(f)
            .map_err(|e| anyhow::anyhow!("invalid configured start_fen: {e}"))?,
        None => GameState::startpos(),
    };
    let next = std::sync::atomic::AtomicU64::new(0);
    let mut records: Vec<GameRecord> = Vec::new();

    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for _ in 0..concurrency {
            let ev = evaluator;
            let cancel = cancel.clone();
            let search_record = search_record.clone();
            let seed = cfg.seed;
            let start_state = start_state.clone();
            let next = &next;
            handles.push(scope.spawn(move || {
                let mut out = Vec::new();
                let mut failures = Vec::new();
                loop {
                    if cancel.is_cancelled() || Instant::now() > deadline {
                        break;
                    }
                    let game_index =
                        next.fetch_add(1, std::sync::atomic::Ordering::SeqCst) as usize;
                    if game_index >= games_total {
                        break;
                    }
                    // `game_id_base` advances between pilot cycles. Derive the
                    // RNG seed from the global id as well, otherwise every
                    // cycle deterministically repeats the first cycle's games.
                    let game_id = game_id_base.wrapping_add(game_index as u64);
                    let game_seed = selfplay_game_seed(seed, game_id);
                    let mut rng = recur64_search::Rng::new(game_seed);
                    match play_game_from(ev, &sp, &mut rng, start_state.clone()) {
                        Ok(mut g) => {
                            g.seed = game_seed;
                            out.push(GameRecord::from_selfplay(
                                game_id,
                                &g,
                                search_record.clone(),
                            ));
                        }
                        Err(e) => failures.push(format!("game {game_id}: {e}")),
                    }
                }
                (out, failures)
            }));
        }
        let mut errors = Vec::new();
        for h in handles {
            match h.join() {
                Ok((mut v, mut failures)) => {
                    records.append(&mut v);
                    errors.append(&mut failures);
                }
                Err(_) => errors.push("self-play worker panicked".to_string()),
            }
        }
        if !errors.is_empty() {
            return Err(anyhow::anyhow!(
                "{} self-play games failed: {}",
                errors.len(),
                errors
                    .iter()
                    .take(4)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join("; ")
            ));
        }
        Ok(())
    })?;
    Ok(records)
}

fn selfplay_game_seed(base_seed: u64, global_game_id: u64) -> u64 {
    base_seed.wrapping_add(global_game_id)
}

/// Run the bounded vertical slice.
pub fn run<B: AutodiffBackend>(
    cfg: &RunConfig,
    run_dir: &RunDir,
    cancel: &CancelToken,
) -> anyhow::Result<RunReport> {
    let started = Instant::now();
    let deadline = started + Duration::from_secs(cfg.run_budget_minutes.max(1) * 60);
    run_dir.write_config_and_metadata(cfg)?;

    let b_device: B::Device = Default::default();
    let inner_device: Device<B::InnerBackend> = Default::default();
    B::seed(&b_device, cfg.seed);

    // --- Save the reference checkpoint (fresh Micro weights). ---
    let reference_model = model_io::build::<B>(&cfg.model, &b_device);
    let reference_optim = adamw::<B, _>();
    let meta = CheckpointMeta::new(
        cfg.model.clone(),
        cfg.recurrence,
        false,
        0,
        cfg.lr,
        cfg.seed,
        0,
        format!("{} ({})", cfg.device, cfg.precision),
        cfg.precision.clone(),
    );
    let mut meta = meta;
    meta.run_id = cfg.run_id.clone();
    meta.git_revision = option_env!("RECUR64_GIT_SHA").map(str::to_owned);
    save_training(
        &run_dir.reference_ckpt(),
        &reference_model,
        &reference_optim,
        &meta,
    )?;
    let reference_model_id = read_model_id(&run_dir.reference_ckpt());

    if cancel.is_cancelled() {
        run_dir.update_status(
            cfg,
            RunStatus::Interrupted,
            Some("cancelled before collect".into()),
        )?;
        return Ok(interrupted_report(cfg, started, reference_model_id));
    }

    // --- COLLECT ---
    let mut header = ReplayHeader::new(
        cfg.run_id.clone(),
        reference_model_id.clone(),
        format!("{} ({})", cfg.device, cfg.precision),
        cfg.precision.clone(),
    );
    header.git_revision = option_env!("RECUR64_GIT_SHA").map(str::to_owned);
    let inference_model =
        model_io::load::<B::InnerBackend>(&run_dir.reference_ckpt(), &cfg.model, &inner_device)?;
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

    let records = collect_parallel(cfg, &evaluator, cancel, deadline, 0)?;

    let inference_metrics = owner.metrics().snapshot();
    owner.shutdown();
    let selfplay = selfplay_metrics(cfg, &records, inference_metrics.clone());

    let mut writer = ReplayWriter::new(&run_dir.replay(), header, cfg.shard_max_games)?;
    for r in records {
        writer.push(r)?;
    }
    let manifest = writer.finish()?;
    let games_collected = manifest.games;

    if cancel.is_cancelled() {
        run_dir.update_status(
            cfg,
            RunStatus::Interrupted,
            Some(format!(
                "cancelled during collect; {games_collected} games published"
            )),
        )?;
        let mut report = interrupted_report(cfg, started, reference_model_id);
        report.games_collected = games_collected;
        report.inference = Some(inference_metrics);
        report.selfplay = Some(selfplay);
        return Ok(report);
    }

    // --- AUDIT ---
    let audit = audit_dir(&run_dir.replay())?;
    if !audit.ok() {
        run_dir.update_status(cfg, RunStatus::Failed, Some("replay audit failed".into()))?;
        anyhow::bail!("replay audit failed: {:?}", audit.errors);
    }

    // --- TRAIN ---
    let all_games = crate::replay::ReplayReader::open(&run_dir.replay())?.read_all_games()?;
    let new_trainable_positions: u64 = all_games
        .iter()
        .filter(|g| g.outcome.is_some())
        .map(|g| g.plies.len() as u64)
        .sum();
    let (scheduled_updates, _) = cfg.reuse_updates(new_trainable_positions)?;
    let (warmup_updates, planned_updates) = cfg.lr_schedule();
    let train_model = model_io::load::<B>(&run_dir.reference_ckpt(), &cfg.model, &b_device)?;
    let mut optim = adamw::<B, _>();
    let learner_cfg = LearnerConfig {
        batch_size: cfg.train_batch,
        accumulation_steps: cfg.accumulation_steps,
        max_updates: scheduled_updates,
        lr: cfg.lr,
        warmup_updates,
        planned_updates,
        start_update: 0,
        recurrence: cfg.recurrence,
        seed: cfg.seed,
        deadline: Some(deadline),
        current_cycle_first_game_id: Some(0),
        games_per_cycle: cfg.collection_shape()?.0 as u64,
    };
    let (train_report, candidate_model_id) =
        match train_from_games(train_model, &mut optim, &all_games, &learner_cfg, &b_device) {
            Ok((trained, report)) => {
                let mut cand_meta = CheckpointMeta::new(
                    cfg.model.clone(),
                    cfg.recurrence,
                    false,
                    report.updates as u64,
                    cfg.lr,
                    cfg.seed,
                    0,
                    format!("{} ({})", cfg.device, cfg.precision),
                    cfg.precision.clone(),
                );
                cand_meta.run_id = cfg.run_id.clone();
                cand_meta.git_revision = option_env!("RECUR64_GIT_SHA").map(str::to_owned);
                cand_meta.update_counter = report.updates as u64;
                save_training(&run_dir.candidate_ckpt(), &trained, &optim, &cand_meta)?;
                (Some(report), read_model_id(&run_dir.candidate_ckpt()))
            }
            Err(e) if e.contains("no trainable examples") => {
                // No completed games produced a result; publish the reference as
                // the candidate and report that no training occurred.
                let reference_model =
                    model_io::load::<B>(&run_dir.reference_ckpt(), &cfg.model, &b_device)?;
                let mut cand_meta = CheckpointMeta::new(
                    cfg.model.clone(),
                    cfg.recurrence,
                    false,
                    0,
                    cfg.lr,
                    cfg.seed,
                    0,
                    format!("{} ({})", cfg.device, cfg.precision),
                    cfg.precision.clone(),
                );
                cand_meta.run_id = cfg.run_id.clone();
                cand_meta.git_revision = option_env!("RECUR64_GIT_SHA").map(str::to_owned);
                save_training(
                    &run_dir.candidate_ckpt(),
                    &reference_model,
                    &optim,
                    &cand_meta,
                )?;
                (None, read_model_id(&run_dir.candidate_ckpt()))
            }
            Err(e) => return Err(anyhow::anyhow!(e)),
        };

    // --- EVALUATE ---
    let ref_infer =
        model_io::load::<B::InnerBackend>(&run_dir.reference_ckpt(), &cfg.model, &inner_device)?;
    let cand_infer =
        model_io::load::<B::InnerBackend>(&run_dir.candidate_ckpt(), &cfg.model, &inner_device)?;
    let ref_ev = SyncEvaluator::new(ref_infer, cfg.recurrence, inner_device.clone());
    let cand_ev = SyncEvaluator::new(cand_infer, cfg.recurrence, inner_device);
    let openings = match &cfg.opening_suite {
        Some(p) => OpeningSuite::load(std::path::Path::new(p))?.openings,
        None => Vec::new(),
    };
    let arena_cfg = ArenaConfig {
        games: cfg.arena_games,
        simulations: cfg.simulations_per_move,
        c_puct: cfg.c_puct,
        recurrence: cfg.recurrence,
        ply_cap: cfg.ply_cap,
        seed: cfg.seed,
        openings,
        concurrency: 1,
    };
    let arena = run_arena(
        &ref_ev,
        &cand_ev,
        &reference_model_id,
        &candidate_model_id,
        &arena_cfg,
    )?;

    // --- REPORT ---
    let report = RunReport {
        run_id: cfg.run_id.clone(),
        status: "completed".to_string(),
        games_collected,
        audit_ok: true,
        audit_errors: Vec::new(),
        train: train_report,
        arena: Some(arena),
        reference_model_id,
        candidate_model_id,
        inference: Some(inference_metrics),
        selfplay: Some(selfplay),
        elapsed_secs: started.elapsed().as_secs_f64(),
    };
    write_report(run_dir, &report)?;
    run_dir.update_status(cfg, RunStatus::Completed, None)?;
    Ok(report)
}

fn interrupted_report(cfg: &RunConfig, started: Instant, reference_model_id: String) -> RunReport {
    RunReport {
        run_id: cfg.run_id.clone(),
        status: "interrupted".to_string(),
        games_collected: 0,
        audit_ok: false,
        audit_errors: Vec::new(),
        train: None,
        arena: None,
        reference_model_id,
        candidate_model_id: String::new(),
        inference: None,
        selfplay: None,
        elapsed_secs: started.elapsed().as_secs_f64(),
    }
}

/// Write `report/report.json` and `report/report.md`.
pub fn write_report(run_dir: &RunDir, report: &RunReport) -> anyhow::Result<()> {
    std::fs::create_dir_all(run_dir.report())?;
    std::fs::write(
        run_dir.report().join("report.json"),
        serde_json::to_vec_pretty(report)?,
    )?;

    let mut md = String::from("# Recur64 run report\n\n");
    md.push_str(&format!("- run_id: {}\n", report.run_id));
    md.push_str(&format!("- status: {}\n", report.status));
    md.push_str(&format!("- games collected: {}\n", report.games_collected));
    md.push_str(&format!("- audit ok: {}\n", report.audit_ok));
    md.push_str(&format!(
        "- reference model: {}\n- candidate model: {}\n",
        report.reference_model_id, report.candidate_model_id
    ));
    if let Some(t) = &report.train {
        md.push_str(&format!(
            "- train: {} examples, {} updates, loss {:.4} -> {:.4}\n",
            t.examples, t.updates, t.first_loss, t.last_loss
        ));
    }
    if let Some(a) = &report.arena {
        md.push_str(&format!(
            "- arena: {} games, W/D/L {}/{}/{}, truncated {}, candidate score {:.3}\n",
            a.games, a.candidate_wins, a.draws, a.reference_wins, a.truncated, a.candidate_score
        ));
    }
    if let Some(m) = &report.inference {
        md.push_str(&format!(
            "- inference: {} requests, {} batches, mean batch {:.2} (p95 {}), queue wait p95 {} us\n",
            m.submitted, m.batches, m.batch_size_mean, m.batch_size_p95, m.queue_wait_us_p95
        ));
    }
    if let Some(s) = &report.selfplay {
        md.push_str(&format!(
            "- selfplay: {} games, {} plies, mean {:.1} plies/game, active_games {} / workers {}, peak in-flight evals {}\n",
            s.games,
            s.plies,
            s.mean_game_plies,
            s.requested_active_games,
            s.cpu_workers,
            s.peak_in_flight_evaluations
        ));
        md.push_str(&format!(
            "- results: W/D/L {}/{}/{}, truncated {}\n",
            s.white_wins, s.draws, s.black_wins, s.truncated
        ));
        let terms: Vec<String> = s
            .terminations
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect();
        md.push_str(&format!("- terminations: {}\n", terms.join(", ")));
        md.push_str(&format!(
            "- concurrency: batch mean {:.2} / p50 {} / p95 {} / max {}\n",
            s.inference.batch_size_mean,
            s.inference.batch_size_p50,
            s.inference.batch_size_p95,
            s.inference.batch_size_max
        ));
    }
    md.push_str(&format!("- elapsed: {:.1}s\n", report.elapsed_secs));
    std::fs::write(run_dir.report().join("report.md"), md)?;
    Ok(())
}

/// Helper for the `selfplay` CLI: play games and write replay without training.
/// Uses the same parallel collect path as `run`, so `active_games`/`cpu_workers`
/// reflect real concurrent execution.
pub fn collect_only<B: AutodiffBackend>(
    cfg: &RunConfig,
    replay_dir: &Path,
    cancel: &CancelToken,
) -> anyhow::Result<SelfPlayMetrics> {
    let inner_device: Device<B::InnerBackend> = Default::default();
    let b_device: B::Device = Default::default();
    let reference_model = model_io::build::<B>(&cfg.model, &b_device);
    let reference_optim = adamw::<B, _>();
    let mut meta = CheckpointMeta::new(
        cfg.model.clone(),
        cfg.recurrence,
        false,
        0,
        cfg.lr,
        cfg.seed,
        0,
        format!("{} ({})", cfg.device, cfg.precision),
        cfg.precision.clone(),
    );
    meta.run_id = cfg.run_id.clone();
    meta.git_revision = option_env!("RECUR64_GIT_SHA").map(str::to_owned);
    let tmp_ckpt = replay_dir.join("_ref");
    save_training(&tmp_ckpt, &reference_model, &reference_optim, &meta)?;
    let model_id = read_model_id(&tmp_ckpt);
    let header = ReplayHeader::new(
        cfg.run_id.clone(),
        model_id,
        format!("{} ({})", cfg.device, cfg.precision),
        cfg.precision.clone(),
    );

    let inference_model = model_io::load::<B::InnerBackend>(&tmp_ckpt, &cfg.model, &inner_device)?;
    let batched = BatchedModel::new(inference_model, cfg.recurrence, inner_device.clone());
    let owner = InferenceOwner::spawn(
        batched,
        InferenceConfig {
            max_batch: cfg.max_inference_batch,
            batch_timeout: Duration::from_micros(cfg.batch_timeout_us),
            ..InferenceConfig::default()
        },
    );
    let ev = owner.evaluator();
    let deadline = Instant::now() + Duration::from_secs(cfg.run_budget_minutes.max(1) * 60);
    let records = collect_parallel(cfg, &ev, cancel, deadline, 0)?;
    let inference_metrics = owner.metrics().snapshot();
    owner.shutdown();

    let metrics = selfplay_metrics(cfg, &records, inference_metrics);
    let mut writer = ReplayWriter::new(replay_dir, header, cfg.shard_max_games)?;
    for r in records {
        writer.push(r)?;
    }
    let manifest = writer.finish()?;
    let _ = std::fs::remove_dir_all(&tmp_ckpt);
    Ok(SelfPlayMetrics {
        games: manifest.games,
        ..metrics
    })
}

/// Reconstruct a game's UCI move list (used by tests and diagnostics).
pub fn game_uci_moves(record: &GameRecord) -> anyhow::Result<Vec<String>> {
    let mut state = GameState::from_fen(&record.start_fen)?;
    let mut out = Vec::new();
    for ply in &record.plies {
        let id = recur64_core::ActionId::from_index(ply.selected as u32)?;
        let perspective = state.perspective();
        let (from, to, promo) = id.to_physical(perspective);
        let promotion = if promo.is_none() { None } else { Some(promo) };
        let mv = StandardMove::new(from, to, promotion);
        out.push(mv.to_uci());
        state.apply(mv)?;
    }
    Ok(out)
}

#[cfg(test)]
mod collection_tests {
    use super::*;
    use recur64_search::{EvalError, EvalRequest, EvalResult};
    struct Failing;
    impl Evaluator for Failing {
        fn evaluate(&self, _: EvalRequest<'_>) -> Result<EvalResult, EvalError> {
            Err(EvalError::Backend("injected failure".into()))
        }
    }
    fn cfg() -> RunConfig {
        RunConfig::from_toml_str("run_id = 'test'\ngames_per_cycle = 2\nconcurrent_games = 2\n[model]\nwidth = 32\nheads = 4\nffn = 64\ninput_blocks = 0\ncore_blocks = 1\noutput_blocks = 0\n").unwrap()
    }
    #[test]
    fn invalid_fen_and_failed_game_are_visible() {
        let mut config = cfg();
        config.start_fen = Some("not a FEN".into());
        let deadline = Instant::now() + Duration::from_secs(2);
        let err =
            collect_parallel(&config, &Failing, &CancelToken::new(), deadline, 0).unwrap_err();
        assert!(err.to_string().contains("invalid configured start_fen"));
        config.start_fen = None;
        let err =
            collect_parallel(&config, &Failing, &CancelToken::new(), deadline, 0).unwrap_err();
        assert!(err.to_string().contains("2 self-play games failed"));
    }

    #[test]
    fn game_seed_uses_global_game_id_across_cycles() {
        let base_seed = 17u64;
        let games_per_cycle = 24u64;
        let first_cycle: Vec<_> = (0..games_per_cycle)
            .map(|index| selfplay_game_seed(base_seed, index))
            .collect();
        let second_cycle: Vec<_> = (0..games_per_cycle)
            .map(|index| selfplay_game_seed(base_seed, games_per_cycle + index))
            .collect();
        assert!(first_cycle.iter().all(|seed| !second_cycle.contains(seed)));
        assert_eq!(second_cycle[0], 41);
    }
}
