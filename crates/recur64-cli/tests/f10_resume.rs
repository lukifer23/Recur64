//! T3.7: F10 checkpoint/resume preserves optimizer moments and the LR schedule.
//!
//! A run trained continuously must match a run that is saved after K updates and
//! resumed for the remaining updates, on deterministic CPU FP32.

use burn::backend::Autodiff;
use burn::backend::Flex;

use recur64_core::{ActionId, Color, GameState, PromotionCode, StandardMove};
use recur64_model::checkpoint::{CheckpointMeta, load_training, save_training};
use recur64_model::config::ModelConfig;
use recur64_model::model::ProbeModel;
use recur64_model::train::{adamw, train_step_reporting};
use recur64_runtime::learner::{build_batch_tensors, lr_at};
use recur64_runtime::replay::sampler::TrainingExample;
use recur64_runtime::replay::{GameRecord, PlyRecord, SearchRecord};

type B = Autodiff<Flex>;

fn f10() -> ModelConfig {
    ModelConfig {
        width: 384,
        heads: 12,
        ffn: 768,
        input_blocks: 0,
        core_blocks: 8,
        output_blocks: 0,
        squares: 64,
        in_features: 119,
        policy_dim: 128,
        wdl_classes: 3,
        promo_codes: 5,
        rms_eps: 1e-5,
    }
}

fn fools_mate(game_id: u64) -> GameRecord {
    let moves = ["f2f3", "e7e5", "g2g4", "d8h4"];
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
        termination: "checkmate".into(),
        outcome: Some(2),
    }
}

fn tmp(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("recur64_f10_resume_{name}"));
    let _ = std::fs::remove_dir_all(&d);
    d
}

#[test]
fn f10_resume_preserves_optimizer_and_schedule() {
    let device = Default::default();
    let examples: Vec<TrainingExample> =
        recur64_runtime::build_examples(&[fools_mate(0), fools_mate(1)])
            .unwrap()
            .0;

    let model0 = ProbeModel::<B>::new(f10(), &device);
    let optim0 = adamw::<B, _>();
    let lr_base = 3e-4;
    let warmup = 1u64;
    let planned = 4u64;

    // Run A: continuous 4 updates.
    let mut model_a = model0.clone();
    let mut optim_a = optim0.clone();
    let mut loss_a = f32::NAN;
    for (i, ex) in examples.iter().enumerate().take(4) {
        let ex = [ex];
        let (board, cands, targets) = build_batch_tensors::<B>(&ex, &device);
        let lr = lr_at(i as u64, lr_base, warmup, planned);
        let (m, r) =
            train_step_reporting(model_a, &mut optim_a, board, &cands, &targets, 1, false, lr);
        model_a = m;
        loss_a = r.total_loss;
    }

    // Run B: 2 updates, save, reload, 2 more.
    let mut model_b = model0.clone();
    let mut optim_b = optim0.clone();
    for (i, ex) in examples.iter().enumerate().take(2) {
        let ex = [ex];
        let (board, cands, targets) = build_batch_tensors::<B>(&ex, &device);
        let lr = lr_at(i as u64, lr_base, warmup, planned);
        let (m, _) =
            train_step_reporting(model_b, &mut optim_b, board, &cands, &targets, 1, false, lr);
        model_b = m;
    }

    let dir = tmp("ckpt");
    let _ = std::fs::remove_dir_all(&dir);
    let mut meta = CheckpointMeta::new(f10(), 1, false, 2, lr_base, 0, 0, "cpu", "fp32");
    meta.lr_schedule_step = 2;
    meta.update_counter = 2;
    save_training(&dir, &model_b, &optim_b, &meta).expect("save");

    // Reload into a template that shares the original parameter identities.
    let template = model0.clone();
    let (mut model_b2, mut optim_b2, loaded) =
        load_training(&dir, template, optim0.clone(), &device).expect("load");
    assert_eq!(loaded.lr_schedule_step, 2, "schedule step preserved");
    assert_eq!(loaded.update_counter, 2);

    let mut loss_b = f32::NAN;
    for (i, ex) in examples.iter().enumerate().take(4).skip(2) {
        let ex = [ex];
        let (board, cands, targets) = build_batch_tensors::<B>(&ex, &device);
        let lr = lr_at(i as u64, lr_base, warmup, planned);
        let (m, r) = train_step_reporting(
            model_b2,
            &mut optim_b2,
            board,
            &cands,
            &targets,
            1,
            false,
            lr,
        );
        model_b2 = m;
        loss_b = r.total_loss;
    }

    let dl = (loss_a - loss_b).abs();
    let dw = (model_a.core_weight_scalar() - model_b2.core_weight_scalar()).abs();
    println!("continuous loss={loss_a} resumed loss={loss_b} |dl|={dl} |dw|={dw}");
    assert!(dl < 1e-4, "resumed loss {loss_b} vs continuous {loss_a}");
    assert!(dw < 1e-5, "core weight drift {dw}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// The pilot reloads a promoted checkpoint into a *freshly built* module (new
/// random weights, new `ParamId`s) and a fresh AdamW, then continues through
/// the real learner. That path must reproduce the continuous trajectory.
#[test]
fn pilot_reload_into_fresh_module_preserves_optimizer_trajectory() {
    use recur64_runtime::learner::{LearnerConfig, train_from_games};
    let device = Default::default();
    let cfg = ModelConfig {
        width: 64,
        heads: 4,
        ffn: 128,
        core_blocks: 2,
        ..f10()
    };
    let games = [fools_mate(0), fools_mate(1)];
    let learner = |max_updates: usize, start_update: u64| LearnerConfig {
        batch_size: 4,
        accumulation_steps: 2,
        max_updates,
        lr: 3e-4,
        warmup_updates: 1,
        planned_updates: 4,
        start_update,
        seed: 7,
        ..LearnerConfig::default()
    };

    let model0 = ProbeModel::<B>::new(cfg.clone(), &device);
    let (model_a, report_a) = train_from_games(
        model0.clone(),
        &mut adamw::<B, _>(),
        &games,
        &learner(4, 0),
        &device,
    )
    .unwrap();

    let mut optim_b = adamw::<B, _>();
    let (model_b, report_b1) = train_from_games(
        model0.clone(),
        &mut optim_b,
        &games,
        &learner(2, 0),
        &device,
    )
    .unwrap();
    let dir = tmp("fresh_template");
    let mut meta = CheckpointMeta::new(cfg.clone(), 1, false, 2, 3e-4, 0, 0, "cpu", "fp32");
    meta.update_counter = 2;
    meta.lr_schedule_step = 2;
    save_training(&dir, &model_b, &optim_b, &meta).expect("save");
    drop((model_b, optim_b, model0));

    // Exactly what pilot.rs does: fresh build, fresh optimizer, load_training.
    let fresh = ProbeModel::<B>::new(cfg.clone(), &device);
    let (model_b2, mut optim_b2, loaded) =
        load_training(&dir, fresh, adamw::<B, _>(), &device).expect("load");
    assert_eq!(loaded.lr_schedule_step, 2);
    let (model_b3, report_b2) =
        train_from_games(model_b2, &mut optim_b2, &games, &learner(2, 2), &device).unwrap();

    assert_eq!(report_a.updates, 4);
    assert_eq!(report_b1.updates + report_b2.updates, 4);
    for (a, b) in report_a.metrics[2..].iter().zip(&report_b2.metrics) {
        assert_eq!(a.update, b.update, "global update index continues");
        assert_eq!(a.lr, b.lr, "LR schedule continues");
    }
    let la = report_a.metrics[3].total_loss;
    let lb = report_b2.metrics[1].total_loss;
    let dl = (la - lb).abs();
    let dw = (model_a.core_weight_scalar() - model_b3.core_weight_scalar()).abs();
    println!(
        "fresh-template resume: continuous loss={la} resumed loss={lb} |dl|={dl} |dw|={dw} bit_exact={}",
        dl == 0.0 && dw == 0.0
    );
    assert!(dl < 1e-4, "resumed loss {lb} vs continuous {la}");
    assert!(dw < 1e-5, "core weight drift {dw}");

    // Negative control: a fresh AdamW with the *same* weights must diverge, or
    // this test could not detect lost moments.
    let fresh = ProbeModel::<B>::new(cfg, &device);
    let (model_c, _, _) = load_training(&dir, fresh, adamw::<B, _>(), &device).unwrap();
    let (model_c2, _) = train_from_games(
        model_c,
        &mut adamw::<B, _>(),
        &games,
        &learner(2, 2),
        &device,
    )
    .unwrap();
    let dw_reset = (model_a.core_weight_scalar() - model_c2.core_weight_scalar()).abs();
    println!("reset-moments control |dw|={dw_reset}");
    assert!(
        dw_reset > dw,
        "control must show moment loss ({dw_reset} vs {dw})"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
