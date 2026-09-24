//! T2.9: the learner reconstructs real positions and performs real updates.

use burn::backend::Flex;

use recur64_core::{ActionId, Color, GameState, PromotionCode, StandardMove, Termination};
use recur64_model::config::ModelConfig;
use recur64_model::model::ProbeModel;
use recur64_model::train::{CpuTrainBackend, adamw};
use recur64_runtime::learner::lr_at;
use recur64_runtime::learner::{build_examples, train_from_games};
use recur64_runtime::replay::{GameRecord, PlyRecord, SearchRecord};
use recur64_runtime::{LearnerConfig, SelfPlayConfig, SyncEvaluator, play_game_seeded};

fn micro() -> ModelConfig {
    ModelConfig {
        width: 192,
        heads: 6,
        ffn: 384,
        input_blocks: 0,
        core_blocks: 4,
        output_blocks: 0,
        squares: 64,
        in_features: 119,
        policy_dim: 128,
        wdl_classes: 3,
        promo_codes: 5,
        rms_eps: 1e-5,
    }
}

/// Build a real, auditable game record from UCI moves and a known result.
fn record_from_uci(game_id: u64, moves: &[&str], outcome: u8, termination: &str) -> GameRecord {
    let mut state = GameState::startpos();
    let start_fen = state.to_fen();
    let mut plies = Vec::new();
    for m in moves {
        let perspective = state.perspective();
        let mv = StandardMove::from_uci(state.board(), m).unwrap();
        let id = ActionId::from_physical(
            mv.from,
            mv.to,
            mv.promotion.unwrap_or(PromotionCode::NONE),
            perspective,
        );
        assert!(state.legal_actions().contains(&id), "{m} should be legal");
        plies.push(PlyRecord {
            selected: id.index() as u16,
            target: vec![(id.index() as u16, 1.0)],
            visits_total: 1,
            side_to_move: if state.side_to_move() == Color::White {
                0
            } else {
                1
            },
        });
        state.apply(mv).unwrap();
    }
    GameRecord {
        game_id,
        start_fen,
        seed: 0,
        search: SearchRecord {
            simulations: 1,
            c_puct: 1.0,
            temperature: 0.0,
            recurrence: 1,
        },
        plies,
        termination: termination.to_string(),
        outcome: Some(outcome),
    }
}

/// Fool's mate: 1.f3 e5 2.g4 Qh4# (black wins).
fn fools_mate(game_id: u64) -> GameRecord {
    record_from_uci(game_id, &["f2f3", "e7e5", "g2g4", "d8h4"], 2, "checkmate")
}

#[test]
fn examples_are_reconstructed_with_correct_perspective() {
    let gs = vec![fools_mate(0)];
    let (examples, used, skipped) = build_examples(&gs).unwrap();
    assert_eq!(used, 1);
    assert_eq!(skipped, 0);
    assert_eq!(examples.len(), 4);
    // White to move (black wins) -> loss; black to move -> win.
    assert_eq!(examples[0].wdl, 2);
    assert_eq!(examples[1].wdl, 0);
    assert_eq!(examples[2].wdl, 2);
    assert_eq!(examples[3].wdl, 0);
    for ex in &examples {
        assert!(!ex.legal.is_empty());
        assert_eq!(ex.policy.len(), ex.legal.len());
        let sum: f32 = ex.policy.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "policy sum {sum}");
    }
}

#[test]
fn truncated_games_are_excluded() {
    let device = Default::default();
    let model = ProbeModel::<Flex>::new(micro(), &device);
    let ev = SyncEvaluator::new(model, 1, device);
    let cfg = SelfPlayConfig {
        simulations_per_move: 2,
        c_puct: 1.0,
        temperature: 1.0,
        ply_cap: 4,
        recurrence: 1,
    };
    let g = play_game_seeded(&ev, &cfg, 1).unwrap();
    let mut rec = GameRecord::from_selfplay(
        0,
        &g,
        SearchRecord {
            simulations: 2,
            c_puct: 1.0,
            temperature: 1.0,
            recurrence: 1,
        },
    );
    rec.outcome = None;
    rec.termination = Termination::Truncated.label().to_string();
    let (_examples, used, skipped) = build_examples(&[rec]).unwrap();
    assert_eq!(used, 0);
    assert_eq!(skipped, 1);
}

#[test]
fn lr_schedule_warmup_then_cosine() {
    let base = 3e-4;
    let warmup = 10u64;
    let planned = 100u64;
    // Warmup ramps up.
    assert!(lr_at(0, base, warmup, planned) < lr_at(9, base, warmup, planned));
    assert!((lr_at(9, base, warmup, planned) - base).abs() < 1e-12);
    // Cosine decays after warmup and ends near zero.
    assert!(lr_at(50, base, warmup, planned) < base);
    assert!(lr_at(100, base, warmup, planned) < 1e-9);
    // Monotone non-increasing after warmup.
    let mut prev = f64::INFINITY;
    for step in warmup..=planned {
        let lr = lr_at(step, base, warmup, planned);
        assert!(lr <= prev + 1e-15, "lr rose at step {step}");
        prev = lr;
    }
}

#[test]
fn accumulation_consumes_effective_batch_and_logs_metrics() {
    let device = Default::default();
    let gs = vec![fools_mate(0), fools_mate(1)];
    let model = ProbeModel::<CpuTrainBackend>::new(micro(), &device);
    let mut optim = adamw::<CpuTrainBackend, ProbeModel<CpuTrainBackend>>();
    let cfg = LearnerConfig {
        batch_size: 2,
        accumulation_steps: 2,
        max_updates: 2,
        lr: 3e-3,
        warmup_updates: 0,
        planned_updates: 2,
        start_update: 0,
        recurrence: 1,
        seed: 3,
        deadline: None,
        ..Default::default()
    };
    let (_, report) = train_from_games(model, &mut optim, &gs, &cfg, &device).unwrap();
    assert_eq!(report.updates, 2);
    // 2 updates x 2 accumulation steps x 2 micro-batch = 8 examples consumed.
    assert_eq!(report.examples_consumed, 8);
    assert_eq!(report.metrics.len(), 2);
    for m in &report.metrics {
        assert!(m.total_loss.is_finite());
        assert!(m.policy_loss.is_finite());
        assert!(m.wdl_loss.is_finite());
        assert!(m.grad_norm.is_finite() && m.grad_norm > 0.0);
        assert!(m.policy_entropy.is_finite());
        assert!(m.lr > 0.0);
    }
}

#[test]
fn training_updates_parameters_and_loss_is_finite() {
    let device = Default::default();
    let gs = vec![fools_mate(0), fools_mate(1)];
    let model = ProbeModel::<CpuTrainBackend>::new(micro(), &device);
    let before = model.core_weight_scalar();
    let mut optim = adamw::<CpuTrainBackend, ProbeModel<CpuTrainBackend>>();
    let cfg = LearnerConfig {
        batch_size: 2,
        accumulation_steps: 1,
        max_updates: 2,
        lr: 3e-3,
        warmup_updates: 0,
        planned_updates: 2,
        start_update: 0,
        recurrence: 1,
        seed: 7,
        deadline: None,
        ..Default::default()
    };
    let (trained, report) = train_from_games(model, &mut optim, &gs, &cfg, &device).unwrap();
    assert_eq!(report.updates, 2);
    assert!(report.first_loss.is_finite());
    assert!(report.last_loss.is_finite());
    assert_eq!(report.loss_curve.len(), 2);
    let after = trained.core_weight_scalar();
    assert!(
        (before - after).abs() > 0.0,
        "training must move parameters: {before} -> {after}"
    );
}

#[test]
fn replay_accounting_and_sample_provenance_are_truthful() {
    let device = Default::default();
    let tiny = ModelConfig {
        width: 32,
        heads: 4,
        ffn: 64,
        core_blocks: 1,
        ..micro()
    };
    let model = ProbeModel::<CpuTrainBackend>::new(tiny, &device);
    let mut optim = adamw::<CpuTrainBackend, _>();
    // Cycle 0 = games 0,1; cycle 1 (current) = games 2,3 and a truncated game 4.
    let mut truncated = fools_mate(4);
    truncated.outcome = None;
    truncated.termination = "truncated".into();
    let gs = vec![
        fools_mate(0),
        fools_mate(1),
        fools_mate(2),
        fools_mate(3),
        truncated,
    ];
    let cfg = LearnerConfig {
        batch_size: 8,
        accumulation_steps: 1,
        max_updates: 2,
        planned_updates: 2,
        current_cycle_first_game_id: Some(2),
        games_per_cycle: 2,
        ..LearnerConfig::default()
    };
    let (_, report) = train_from_games(model, &mut optim, &gs, &cfg, &device).unwrap();
    assert_eq!(report.games_used, 4);
    assert_eq!(report.games_skipped, 1);
    assert_eq!(report.replay_total_games, 5);
    assert_eq!(report.sampleable_positions, 16);
    assert_eq!(report.examples_consumed, 16);
    // Every distinct example was consumed once: half from each cycle.
    assert_eq!(report.current_cycle_sample_fraction, Some(0.5));
    assert_eq!(report.mean_sample_age_cycles, Some(0.5));
}

#[test]
fn target_health_separates_trainable_plies() {
    let mut truncated = fools_mate(1);
    truncated.outcome = None;
    for ply in &mut truncated.plies {
        ply.target = vec![(ply.selected, 0.5), (ply.selected.wrapping_add(1), 0.5)];
    }
    let h = recur64_runtime::coordinator::target_health(&[fools_mate(0), truncated]);
    assert_eq!(h.all.positions, 8);
    assert_eq!(h.trainable.positions, 4);
    assert_eq!(h.trainable.mean_entropy, 0.0);
    assert_eq!(h.trainable.mean_top1_visit_share, 1.0);
    assert!((h.all.mean_entropy - 0.5 * std::f64::consts::LN_2).abs() < 1e-12);
    assert!((h.all.mean_top1_visit_share - 0.75).abs() < 1e-12);
}
