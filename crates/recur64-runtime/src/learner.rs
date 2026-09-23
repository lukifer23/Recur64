//! The first learner: policy + WDL cross-entropy from real Recur64 replay.
//!
//! Games are reconstructed from their start FEN and selected moves, so the
//! learner trains on genuine legal positions and genuine search targets. Games
//! without a result (truncated/aborted) are excluded: they must never be turned
//! into draw labels.

use burn::optim::Optimizer;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Int, TensorData};

use recur64_core::{
    ActionId, Color, GameState, ObservationV1, StandardMove, encode_observation_v1,
};
use recur64_model::action::CandidateBatch;
use recur64_model::loss::Targets;
use recur64_model::model::{CandidateTensors, ProbeModel};
use recur64_model::train::train_step;

use crate::replay::schema::GameRecord;
use recur64_search::Rng;

/// Learner configuration.
#[derive(Debug, Clone)]
pub struct LearnerConfig {
    pub batch_size: usize,
    pub max_updates: usize,
    pub lr: f64,
    pub recurrence: usize,
    pub seed: u64,
}

impl Default for LearnerConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            max_updates: 10,
            lr: 3e-4,
            recurrence: 1,
            seed: 0,
        }
    }
}

/// One training example reconstructed from replay.
#[derive(Debug, Clone)]
pub struct TrainingExample {
    pub observation: ObservationV1,
    pub legal: Vec<ActionId>,
    /// Target distribution aligned to `legal`.
    pub policy: Vec<f32>,
    /// WDL class from the side-to-move perspective: 0 win, 1 draw, 2 loss.
    pub wdl: i64,
}

/// Report of a training run.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TrainReport {
    pub examples: usize,
    pub games_used: usize,
    pub games_skipped: usize,
    pub updates: usize,
    pub first_loss: f32,
    pub last_loss: f32,
    pub loss_curve: Vec<(usize, f32)>,
}

/// WDL class from the side-to-move perspective.
fn wdl_class(outcome: u8, side: Color) -> i64 {
    match outcome {
        1 => 1, // draw
        0 => {
            if side == Color::White {
                0
            } else {
                2
            }
        }
        2 => {
            if side == Color::Black {
                0
            } else {
                2
            }
        }
        _ => 1,
    }
}

/// Reconstruct training examples from completed games.
pub fn build_examples(
    games: &[GameRecord],
) -> Result<(Vec<TrainingExample>, usize, usize), String> {
    let mut examples = Vec::new();
    let mut used = 0usize;
    let mut skipped = 0usize;
    for game in games {
        let Some(outcome) = game.outcome else {
            skipped += 1;
            continue;
        };
        used += 1;
        let mut state =
            GameState::from_fen(&game.start_fen).map_err(|e| format!("start FEN: {e}"))?;
        for (i, ply) in game.plies.iter().enumerate() {
            let legal = state.legal_actions();
            let mut policy = vec![0.0f32; legal.len()];
            for (idx, prob) in &ply.target {
                let pos = legal
                    .iter()
                    .position(|a| a.index() == *idx as u32)
                    .ok_or_else(|| {
                        format!("game {} ply {i}: target action not legal", game.game_id)
                    })?;
                policy[pos] += *prob;
            }
            let observation = encode_observation_v1(&state);
            let wdl = wdl_class(outcome, state.side_to_move());
            examples.push(TrainingExample {
                observation,
                legal: legal.clone(),
                policy,
                wdl,
            });

            let perspective = state.perspective();
            let id = ActionId::from_index(ply.selected as u32)
                .map_err(|e| format!("game {} ply {i}: {e}", game.game_id))?;
            let (from, to, promo) = id.to_physical(perspective);
            let promotion = if promo.is_none() { None } else { Some(promo) };
            state
                .apply(StandardMove::new(from, to, promotion))
                .map_err(|e| format!("game {} ply {i}: {e}", game.game_id))?;
        }
    }
    Ok((examples, used, skipped))
}

/// Build model tensors for a batch of examples.
fn build_batch<B: Backend>(
    batch: &[&TrainingExample],
    device: &B::Device,
) -> (Tensor<B, 3>, CandidateTensors<B>, Targets<B>) {
    let b = batch.len();
    let mut board_data = Vec::with_capacity(b * 64 * 119);
    for ex in batch {
        board_data.extend_from_slice(ex.observation.as_slice());
    }
    let board = Tensor::<B, 3>::from_data(TensorData::new(board_data, [b, 64, 119]), device);

    let lists: Vec<Vec<(u32, u32, u8)>> = batch
        .iter()
        .map(|ex| {
            ex.legal
                .iter()
                .map(|id| {
                    let (from, to, promo) = id.decode();
                    (from as u32, to as u32, promo.code())
                })
                .collect()
        })
        .collect();
    let cb = CandidateBatch::from_lists(&lists);
    let width = cb.width;
    let cands = CandidateTensors::from_batch(&cb, device);

    let mut policy_data = vec![0.0f32; b * width];
    for (i, ex) in batch.iter().enumerate() {
        for (k, p) in ex.policy.iter().enumerate() {
            policy_data[i * width + k] = *p;
        }
    }
    let policy_target = Tensor::<B, 2>::from_data(TensorData::new(policy_data, [b, width]), device);
    let wdl_target = Tensor::<B, 1, Int>::from_data(
        TensorData::new(batch.iter().map(|e| e.wdl).collect::<Vec<_>>(), [b]),
        device,
    );

    (
        board,
        cands,
        Targets {
            policy_target,
            wdl_target,
        },
    )
}

/// Train `model` on reconstructed replay for a bounded number of updates.
pub fn train_from_games<B, O>(
    mut model: ProbeModel<B>,
    optim: &mut O,
    games: &[GameRecord],
    cfg: &LearnerConfig,
    device: &B::Device,
) -> Result<(ProbeModel<B>, TrainReport), String>
where
    B: AutodiffBackend,
    O: Optimizer<ProbeModel<B>, B>,
{
    let (mut examples, used, skipped) = build_examples(games)?;
    if examples.is_empty() {
        return Err("no trainable examples (all games truncated/aborted?)".into());
    }

    // Deterministic shuffle.
    let mut rng = Rng::new(cfg.seed);
    for i in (1..examples.len()).rev() {
        let j = (rng.next_u64() as usize) % (i + 1);
        examples.swap(i, j);
    }

    let batch_size = cfg.batch_size.max(1);
    let mut first_loss = f32::NAN;
    let mut last_loss = f32::NAN;
    let mut loss_curve = Vec::new();
    let mut updates = 0usize;

    for update in 0..cfg.max_updates {
        let start = (update * batch_size) % examples.len();
        let end = (start + batch_size).min(examples.len());
        let batch: Vec<&TrainingExample> = examples[start..end].iter().collect();
        if batch.is_empty() {
            break;
        }
        let (board, cands, targets) = build_batch::<B>(&batch, device);
        let (m, loss) = train_step(
            model,
            optim,
            board,
            &cands,
            &targets,
            cfg.recurrence,
            false,
            cfg.lr,
        );
        model = m;
        let l = loss
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| format!("loss read: {e}"))?[0];
        if !l.is_finite() {
            return Err(format!("non-finite loss at update {update}: {l}"));
        }
        if update == 0 {
            first_loss = l;
        }
        last_loss = l;
        loss_curve.push((update, l));
        updates += 1;
    }

    Ok((
        model,
        TrainReport {
            examples: examples.len(),
            games_used: used,
            games_skipped: skipped,
            updates,
            first_loss,
            last_loss,
            loss_curve,
        },
    ))
}
