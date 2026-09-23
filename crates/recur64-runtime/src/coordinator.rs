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
use recur64_eval::{ArenaConfig, ArenaResult, run_arena};
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
    pub inference: MetricsSnapshot,
}

/// Build self-play metrics from collected game records.
fn selfplay_metrics(
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
    SelfPlayMetrics {
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

/// Play `cfg.active_games` games across `cfg.cpu_workers` worker threads.
///
/// This is the **only** parallel self-play path: both `run` and `collect_only`
/// call it, so a configured concurrency value always corresponds to real
/// concurrent execution. The workers block on the single inference owner, which
/// coalesces their requests into batches (see `MetricsSnapshot`).
fn collect_parallel(
    cfg: &RunConfig,
    evaluator: &dyn Evaluator,
    cancel: &CancelToken,
    deadline: Instant,
) -> Vec<GameRecord> {
    let sp = SelfPlayConfig {
        simulations_per_move: cfg.simulations_per_move,
        c_puct: cfg.c_puct,
        temperature: cfg.temperature,
        ply_cap: cfg.ply_cap,
        recurrence: cfg.recurrence,
    };
    let search_record = SearchRecord {
        simulations: cfg.simulations_per_move,
        c_puct: cfg.c_puct,
        temperature: cfg.temperature,
        recurrence: cfg.recurrence,
    };

    let workers = cfg.cpu_workers.max(1);
    let games_total = cfg.active_games as usize;
    let mut records: Vec<GameRecord> = Vec::new();

    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for w in 0..workers {
            let per = games_total / workers;
            let extra = if w < games_total % workers { 1 } else { 0 };
            let start = w * per + w.min(games_total % workers);
            let count = per + extra;
            let ev = evaluator;
            let cancel = cancel.clone();
            let search_record = search_record.clone();
            let seed = cfg.seed;
            let start_fen = cfg.start_fen.clone();
            handles.push(scope.spawn(move || {
                let mut out = Vec::with_capacity(count);
                for i in 0..count {
                    if cancel.is_cancelled() || Instant::now() > deadline {
                        break;
                    }
                    let game_index = start + i;
                    let game_seed = seed.wrapping_add(game_index as u64);
                    let start_state = match &start_fen {
                        Some(f) => match GameState::from_fen(f) {
                            Ok(s) => s,
                            Err(_) => continue,
                        },
                        None => GameState::startpos(),
                    };
                    let mut rng = recur64_search::Rng::new(game_seed);
                    if let Ok(mut g) = play_game_from(ev, &sp, &mut rng, start_state) {
                        g.seed = game_seed;
                        out.push(GameRecord::from_selfplay(
                            game_index as u64,
                            &g,
                            search_record.clone(),
                        ));
                    }
                }
                out
            }));
        }
        for h in handles {
            if let Ok(mut v) = h.join() {
                records.append(&mut v);
            }
        }
    });
    records
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
    let header = ReplayHeader::new(
        cfg.run_id.clone(),
        reference_model_id.clone(),
        format!("{} ({})", cfg.device, cfg.precision),
        cfg.precision.clone(),
    );
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

    let records = collect_parallel(cfg, &evaluator, cancel, deadline);

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
    let train_model = model_io::load::<B>(&run_dir.reference_ckpt(), &cfg.model, &b_device)?;
    let mut optim = adamw::<B, _>();
    let learner_cfg = LearnerConfig {
        batch_size: cfg.train_batch,
        max_updates: cfg.max_updates,
        lr: cfg.lr,
        recurrence: cfg.recurrence,
        seed: cfg.seed,
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
    let arena_cfg = ArenaConfig {
        games: cfg.arena_games,
        simulations: cfg.simulations_per_move,
        c_puct: cfg.c_puct,
        recurrence: cfg.recurrence,
        ply_cap: cfg.ply_cap,
        seed: cfg.seed,
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
    let records = collect_parallel(cfg, &ev, cancel, deadline);
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
