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
