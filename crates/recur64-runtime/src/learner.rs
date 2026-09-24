//! The learner: policy + WDL cross-entropy from real Recur64 replay.
//!
//! Games are reconstructed from their start FEN and selected moves, so the
//! learner trains on genuine legal positions and genuine search targets. Games
//! without a result (truncated/aborted) are excluded.
//!
//! Supports gradient accumulation (effective batch), a warmup + cosine LR
//! schedule, and per-update metrics (total/policy/WDL loss, grad norm, LR,
//! policy entropy) with visible health guards.
//!
//! Two entry points:
//! - [`train_from_games`] materializes every example (small runs / tests).
//! - [`train_from_store`] samples on demand from a [`ReplayStore`], bounding
//!   memory regardless of replay capacity.

use burn::module::{Module, ModuleVisitor, Param};
use burn::optim::{GradientsAccumulator, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Int, TensorData};
use std::marker::PhantomData;
use std::time::Instant;

use recur64_core::{ActionId, GameState, StandardMove};
use recur64_model::action::CandidateBatch;
use recur64_model::loss::{Targets, model_loss, policy_ce, policy_entropy, wdl_ce};
use recur64_model::model::{CandidateTensors, ProbeModel};
use recur64_model::train::global_grad_norm;
use recur64_search::Rng;

use crate::replay::sampler::{ReplayStore, TrainingExample, example_for_ply};
use crate::replay::schema::GameRecord;

/// Learner configuration.
#[derive(Debug, Clone)]
pub struct LearnerConfig {
    /// Physical (micro-batch) size per forward/backward.
    pub batch_size: usize,
    /// Gradient-accumulation steps; effective batch = `batch_size * this`.
    pub accumulation_steps: usize,
    pub max_updates: usize,
    /// Peak (base) learning rate.
    pub lr: f64,
    pub warmup_updates: u64,
    pub planned_updates: u64,
    /// Global update index this training segment starts at (for schedule
    /// continuity across cycles and resume).
    pub start_update: u64,
    pub recurrence: usize,
    pub seed: u64,
    pub deadline: Option<Instant>,
    /// First replay game id generated in the current cycle. Game ids are
    /// contiguous per cycle, so `source_game_id >= this` marks fresh data.
    pub current_cycle_first_game_id: Option<u64>,
    /// Games per cycle, used to express sample age in cycles.
    pub games_per_cycle: u64,
}

impl Default for LearnerConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            accumulation_steps: 1,
            max_updates: 10,
            lr: 3e-4,
            warmup_updates: 0,
            planned_updates: 10,
            start_update: 0,
            recurrence: 1,
            seed: 0,
            deadline: None,
            current_cycle_first_game_id: None,
            games_per_cycle: 0,
        }
    }
}

/// Per-update training metrics.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct UpdateMetrics {
    pub update: usize,
    pub total_loss: f32,
    pub policy_loss: f32,
    pub wdl_loss: f32,
    pub grad_norm: f32,
    pub lr: f64,
    pub policy_entropy: f32,
}

struct MeanGradVisitor<'a, B: AutodiffBackend> {
    grads: &'a mut GradientsParams,
    divisor: f32,
    _backend: PhantomData<B>,
}

impl<B: AutodiffBackend> ModuleVisitor<B> for MeanGradVisitor<'_, B> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        if let Some(grad) = self.grads.remove::<B::InnerBackend, D>(param.id) {
            self.grads
                .register::<B::InnerBackend, D>(param.id, grad / self.divisor);
        }
    }
}

fn mean_gradients<B: AutodiffBackend>(
    grads: &mut GradientsParams,
    model: &ProbeModel<B>,
    examples: usize,
) {
    model.visit(&mut MeanGradVisitor::<B> {
        grads,
        divisor: examples as f32,
        _backend: PhantomData,
    });
}

/// Report of a training run.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TrainReport {
    /// Distinct trainable positions available (sampleable/materialized).
    pub examples: usize,
    pub examples_consumed: u64,
    /// Games whose plies could be trained on (they have a result).
    pub games_used: usize,
    /// Result-less (truncated/aborted) games that were not sampleable.
    pub games_skipped: usize,
    /// All games in the replay the learner read.
    pub replay_total_games: usize,
    /// Positions eligible for sampling (result games only).
    pub sampleable_positions: usize,
    /// Share of consumed examples from games generated this cycle. `None`
    /// when the caller did not identify the current cycle.
    pub current_cycle_sample_fraction: Option<f64>,
    /// Mean age of consumed examples, in cycles (0 = this cycle).
    pub mean_sample_age_cycles: Option<f64>,
    pub updates: usize,
    pub first_loss: f32,
    pub last_loss: f32,
    pub loss_curve: Vec<(usize, f32)>,
    pub metrics: Vec<UpdateMetrics>,
}

fn scalar<B: Backend>(t: Tensor<B, 1>) -> f32 {
    t.into_data()
        .to_vec::<f32>()
        .ok()
        .and_then(|v| v.first().copied())
        .unwrap_or(f32::NAN)
}

/// Learning rate at a given update: linear warmup then cosine decay.
pub fn lr_at(step: u64, base: f64, warmup: u64, planned: u64) -> f64 {
    if warmup > 0 && step < warmup {
        base * (step + 1) as f64 / warmup as f64
    } else {
        let denom = planned.saturating_sub(warmup).max(1) as f64;
        let progress = (step.saturating_sub(warmup) as f64 / denom).min(1.0);
        base * (0.5 * (1.0 + (std::f64::consts::PI * progress).cos())).max(0.0)
    }
}

/// Reconstruct training examples from completed games (materializes all).
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
            let mut example = example_for_ply(&state, outcome, ply)
                .map_err(|e| format!("game {} ply {i}: {e}", game.game_id))?;
            example.source_game_id = game.game_id;
            examples.push(example);
            let id = ActionId::from_index(ply.selected as u32)
                .map_err(|e| format!("game {} ply {i}: {e}", game.game_id))?;
            let perspective = state.perspective();
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
pub fn build_batch_tensors<B: Backend>(
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

#[allow(clippy::too_many_arguments)]
fn run_updates<B, O, F>(
    mut model: ProbeModel<B>,
    optim: &mut O,
    cfg: &LearnerConfig,
    device: &B::Device,
    total_examples: usize,
    games_used: usize,
    games_skipped: usize,
    mut next_batch: F,
) -> Result<(ProbeModel<B>, TrainReport), String>
where
    B: AutodiffBackend,
    O: Optimizer<ProbeModel<B>, B>,
    F: FnMut(usize) -> Result<Vec<TrainingExample>, String>,
{
    let accum = cfg.accumulation_steps.max(1);
    let micro = cfg.batch_size.max(1);
    let planned = cfg.planned_updates.max(1);
    let mut loss_curve = Vec::new();
    let mut metrics = Vec::new();
    let mut consumed = 0u64;
    let mut fresh = 0u64;
    let mut age_sum = 0f64;

    for update in 0..cfg.max_updates {
        if cfg
            .deadline
            .is_some_and(|deadline| Instant::now() >= deadline)
        {
            break;
        }
        let global_update = cfg.start_update + update as u64;
        let lr = lr_at(global_update, cfg.lr, cfg.warmup_updates, planned);
        let mut accumulator = GradientsAccumulator::<ProbeModel<B>>::new();
        let mut components = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let mut micro_count = 0usize;
        let mut update_examples = 0usize;

        for _ in 0..accum {
            let examples = next_batch(micro)?;
            if examples.is_empty() {
                break;
            }
            micro_count += 1;
            update_examples += examples.len();
            consumed += examples.len() as u64;
            if let Some(first) = cfg.current_cycle_first_game_id {
                for ex in &examples {
                    if ex.source_game_id >= first {
                        fresh += 1;
                    } else if let Some(age) =
                        (first - ex.source_game_id - 1).checked_div(cfg.games_per_cycle)
                    {
                        age_sum += (age + 1) as f64;
                    }
                }
            }
            let refs: Vec<&TrainingExample> = examples.iter().collect();
            let (board, cands, targets) = build_batch_tensors::<B>(&refs, device);
            let out = model.forward_r(board, &cands, cfg.recurrence, false);
            let readout = &out.readouts[0];
            let policy_loss = scalar(policy_ce(&readout.policy, &targets.policy_target));
            let wdl_loss = scalar(wdl_ce(&readout.wdl_logits, &targets.wdl_target));
            let entropy = scalar(policy_entropy(&readout.policy));
            let loss = model_loss(&out, &targets);
            let total = scalar(loss.clone());
            let grads =
                GradientsParams::from_grads((loss * examples.len() as f32).backward(), &model);
            accumulator.accumulate(&model, grads);
            let weight = examples.len() as f32;
            components.0 += total * weight;
            components.1 += policy_loss * weight;
            components.2 += wdl_loss * weight;
            components.3 += entropy * weight;
        }
        if micro_count == 0 {
            break;
        }

        let mut grads = accumulator.grads();
        mean_gradients(&mut grads, &model, update_examples);
        let grad_norm = global_grad_norm(&grads, &model);
        let (total, policy_loss, wdl_loss, entropy) = (
            components.0 / update_examples as f32,
            components.1 / update_examples as f32,
            components.2 / update_examples as f32,
            components.3 / update_examples as f32,
        );
        if !total.is_finite() || !grad_norm.is_finite() {
            return Err(format!(
                "non-finite loss/grad at update {update}: loss={total} grad={grad_norm}"
            ));
        }
        model = optim.step(lr, model, grads);

        metrics.push(UpdateMetrics {
            update: global_update as usize,
            total_loss: total,
            policy_loss,
            wdl_loss,
            grad_norm,
            lr,
            policy_entropy: entropy,
        });
        loss_curve.push((update, total));
    }

    let first_loss = metrics.first().map(|m| m.total_loss).unwrap_or(f32::NAN);
    let last_loss = metrics.last().map(|m| m.total_loss).unwrap_or(f32::NAN);
    let updates = metrics.len();
    Ok((
        model,
        TrainReport {
            examples: total_examples,
            examples_consumed: consumed,
            games_used,
            games_skipped,
            replay_total_games: games_used + games_skipped,
            sampleable_positions: total_examples,
            current_cycle_sample_fraction: cfg
                .current_cycle_first_game_id
                .filter(|_| consumed > 0)
                .map(|_| fresh as f64 / consumed as f64),
            mean_sample_age_cycles: cfg
                .current_cycle_first_game_id
                .filter(|_| consumed > 0 && cfg.games_per_cycle > 0)
                .map(|_| age_sum / consumed as f64),
            updates,
            first_loss,
            last_loss,
            loss_curve,
            metrics,
        },
    ))
}

/// Train on materialized examples (small runs / tests).
pub fn train_from_games<B, O>(
    model: ProbeModel<B>,
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
    let mut rng = Rng::new(cfg.seed);
    for i in (1..examples.len()).rev() {
        let j = (rng.next_u64() as usize) % (i + 1);
        examples.swap(i, j);
    }
    let total = examples.len();
    let mut cursor = 0usize;
    run_updates(
        model,
        optim,
        cfg,
        device,
        total,
        used,
        skipped,
        |batch_size| {
            if cursor + batch_size > examples.len() {
                cursor = 0; // wrap for reuse
            }
            let end = (cursor + batch_size).min(examples.len());
            let out = examples[cursor..end].to_vec();
            cursor = end;
            Ok(out)
        },
    )
}

/// Train by sampling on demand from a [`ReplayStore`] (bounded memory).
pub fn train_from_store<B, O>(
    store: &ReplayStore,
    model: ProbeModel<B>,
    optim: &mut O,
    cfg: &LearnerConfig,
    device: &B::Device,
) -> Result<(ProbeModel<B>, TrainReport), String>
where
    B: AutodiffBackend,
    O: Optimizer<ProbeModel<B>, B>,
{
    if store.sampleable() == 0 {
        return Err("no trainable examples (all games truncated/aborted?)".into());
    }
    let mut rng = Rng::new(cfg.seed);
    run_updates(
        model,
        optim,
        cfg,
        device,
        store.sampleable(),
        store.trainable_games(),
        store.total_games() - store.trainable_games(),
        |batch_size| store.sample_batch(batch_size, &mut rng),
    )
}
