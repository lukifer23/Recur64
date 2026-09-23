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

use recur64_core::GameState;
use recur64_eval::{ArenaConfig, ArenaResult, OpeningSuite, run_arena};
use recur64_model::checkpoint::{CheckpointMeta, save_training};
use recur64_model::train::adamw;

use crate::SyncEvaluator;
use crate::cancel::CancelToken;
use crate::config::{RunConfig, SnapshotPolicy};
use crate::eval_policy::{RawMatchResult, raw_policy_vs_random};
use crate::inference::{BatchedModel, InferenceConfig, InferenceOwner};
use crate::learner::{LearnerConfig, TrainReport, train_from_store};
use crate::model_io;
use crate::replay::{
    GameRecord, ReplayHeader, ReplayStore, ReplayWriter, SearchRecord, audit_dir, enforce_capacity,
};
use crate::run_dir::{LineageRecord, RunDir, RunStatus};
use recur64_search::{SelfPlayConfig, play_game_from};

/// One cycle's report.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CycleReport {
    pub cycle: u32,
    pub games: u64,
    pub positions: u64,
    pub audit_ok: bool,
    pub train: Option<TrainReport>,
    pub arena: Option<ArenaResult>,
    pub raw: Option<RawMatchResult>,
    pub snapshot_model_id: String,
    pub candidate_model_id: String,
    pub replay_positions: u64,
    pub reuse_ratio: f64,
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

fn load_openings(cfg: &RunConfig) -> Vec<String> {
    match &cfg.opening_suite {
        Some(p) => OpeningSuite::load(Path::new(p))
            .map(|s| s.openings)
            .unwrap_or_default(),
        None => Vec::new(),
    }
}

/// Collect `active_games` concurrent self-play games with the snapshot model.
fn collect_games<B: AutodiffBackend>(
    cfg: &RunConfig,
    snapshot_dir: &Path,
    inner_device: &Device<B::InnerBackend>,
    cancel: &CancelToken,
    deadline: Instant,
    game_id_base: u64,
) -> anyhow::Result<Vec<GameRecord>> {
    let inference_model =
        model_io::load::<B::InnerBackend>(snapshot_dir, &cfg.model, inner_device)?;
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
    let concurrency = (cfg.active_games as usize).max(1);
    let games_total = cfg.active_games as usize;
    let next = std::sync::atomic::AtomicU64::new(0);
    let mut records = Vec::new();

    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for _ in 0..concurrency {
            let ev = &ev;
            let cancel = cancel.clone();
            let search_record = search_record.clone();
            let seed = cfg.seed;
            let start_fen = cfg.start_fen.clone();
            let next = &next;
            handles.push(scope.spawn(move || {
                let mut out = Vec::new();
                loop {
                    if cancel.is_cancelled() || Instant::now() > deadline {
                        break;
                    }
                    let gi = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst) as usize;
                    if gi >= games_total {
                        break;
                    }
                    let game_seed = seed.wrapping_add(gi as u64);
                    let start = match &start_fen {
                        Some(f) => match GameState::from_fen(f) {
                            Ok(s) => s,
                            Err(_) => continue,
                        },
                        None => GameState::startpos(),
                    };
                    let mut rng = recur64_search::Rng::new(game_seed);
                    if let Ok(mut g) = play_game_from(ev, &sp, &mut rng, start) {
                        g.seed = game_seed;
                        out.push(GameRecord::from_selfplay(
                            game_id_base + gi as u64,
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

    owner.shutdown();
    Ok(records)
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
    save_training(
        &run_dir.reference_ckpt(),
        &reference_model,
        &reference_optim,
        &ref_meta,
    )?;
    let reference_model_id = read_model_id(&run_dir.reference_ckpt());

    let openings = load_openings(cfg);
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

        // COLLECT
        let records = collect_games::<B>(
            cfg,
            &snapshot_dir,
            &inner_device,
            cancel,
            deadline,
            total_games_collected,
        )?;
        let games = records.len() as u64;
        total_games_collected += games;
        let positions: u64 = records.iter().map(|g| g.plies.len() as u64).sum();
        let header = ReplayHeader::new(
            cfg.run_id.clone(),
            snapshot_model_id.clone(),
            backend_label(cfg),
            cfg.precision.clone(),
        );
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
        let train_model = model_io::load::<B>(&snapshot_dir, &cfg.model, &b_device)?;
        let mut optim = adamw::<B, _>();
        // The schedule spans the whole pilot (cycles x per-cycle updates), not
        // just one cycle, so warmup/decay behave as intended.
        let planned = cfg
            .resolved_planned_updates()
            .max(cfg.max_updates as u64 * cfg.cycles.max(1) as u64);
        let warmup = cfg.warmup_updates.unwrap_or((planned / 10).clamp(10, 1000));
        let learner_cfg = LearnerConfig {
            batch_size: cfg.train_batch,
            accumulation_steps: cfg.accumulation_steps,
            max_updates: cfg.max_updates,
            lr: cfg.lr,
            warmup_updates: warmup,
            planned_updates: planned,
            start_update: cumulative_updates,
            recurrence: cfg.recurrence,
            seed: cfg.seed.wrapping_add(cycle as u64),
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
                    meta.update_counter = cumulative_updates + report.updates as u64;
                    meta.lr_schedule_step = cumulative_updates + report.updates as u64;
                    save_training(&run_dir.candidate_ckpt(), &trained, &optim, &meta)?;
                    cumulative_updates += report.updates as u64;
                    (read_model_id(&run_dir.candidate_ckpt()), Some(report))
                }
                Err(e) if e.contains("no trainable") => {
                    copy_dir(&snapshot_dir, &run_dir.candidate_ckpt())?;
                    (snapshot_model_id.clone(), None)
                }
                Err(e) => return Err(anyhow::anyhow!(e)),
            };

        // EVALUATE
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

        // SNAPSHOT DECISION (conservative).
        let healthy = audit.ok()
            && train_report
                .as_ref()
                .map(|r| r.updates > 0)
                .unwrap_or(false);
        let decision = match cfg.snapshot_policy {
            SnapshotPolicy::FrozenReference => "continue".to_string(),
            SnapshotPolicy::Conservative => {
                if healthy && arena.candidate_score >= cfg.promotion_score_floor {
                    let snap = run_dir.checkpoints().join(format!("snapshot-{cycle:03}"));
                    copy_dir(&run_dir.candidate_ckpt(), &snap)?;
                    snapshot_dir = snap;
                    snapshot_model_id = candidate_model_id.clone();
                    "promote".to_string()
                } else {
                    "continue".to_string()
                }
            }
        };

        let reuse_ratio = if positions > 0 {
            train_report
                .as_ref()
                .map(|r| r.examples_consumed as f64 / positions as f64)
                .unwrap_or(0.0)
        } else {
            0.0
        };

        let cycle_report = CycleReport {
            cycle,
            games,
            positions,
            audit_ok: audit.ok(),
            train: train_report,
            arena: Some(arena),
            raw: Some(raw),
            snapshot_model_id: snapshot_model_id.clone(),
            candidate_model_id: candidate_model_id.clone(),
            replay_positions,
            reuse_ratio,
            decision: decision.clone(),
            wall_secs: cycle_start.elapsed().as_secs_f64(),
        };
        run_dir.append_lineage(&LineageRecord {
            cycle,
            run_id: cfg.run_id.clone(),
            parent_model_id: snapshot_model_id.clone(),
            candidate_model_id: candidate_model_id.clone(),
            replay_model_ids: vec![snapshot_model_id.clone()],
            new_positions: positions,
            examples_consumed: cycle_report
                .train
                .as_ref()
                .map(|t| t.examples_consumed)
                .unwrap_or(0),
            optimizer_step_start: cumulative_updates.saturating_sub(
                cycle_report
                    .train
                    .as_ref()
                    .map(|t| t.updates as u64)
                    .unwrap_or(0),
            ),
            optimizer_step_end: cumulative_updates,
            wall_clock_secs: cycle_report.wall_secs,
            arena_candidate_score: cycle_report.arena.as_ref().map(|a| a.candidate_score),
            snapshot_decision: decision,
            config_hash: cfg.config_hash(),
            git_revision: None,
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
