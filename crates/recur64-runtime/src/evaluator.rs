//! Synchronous neural evaluator over the Phase 0/1 model.
//!
//! Used for the internal arena (few games, one model at a time) and as a
//! correctness reference for the batched inference owner. It performs a
//! single-position forward pass through [`ProbeModel`] and returns policy and
//! value from the side-to-move perspective.

use burn::prelude::*;
use burn::tensor::TensorData;

use recur64_model::action::CandidateBatch;
use recur64_model::candidate::facts_tensor;
use recur64_model::model::{CandidateTensors, ProbeModel};
use recur64_model::net::NeuralModel;
use recur64_search::{EvalError, EvalRequest, EvalResult, Evaluator};

/// A single-model, single-position evaluator.
pub struct SyncEvaluator<B: Backend, M: NeuralModel<B> = ProbeModel<B>> {
    model: M,
    recurrence: usize,
    device: B::Device,
}

impl<B: Backend, M: NeuralModel<B>> SyncEvaluator<B, M> {
    pub fn new(model: M, recurrence: usize, device: B::Device) -> Self {
        Self {
            model,
            recurrence,
            device,
        }
    }

    pub fn model(&self) -> &M {
        &self.model
    }
}

/// Softmax over three logits.
fn softmax3(logits: [f32; 3]) -> [f32; 3] {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps = logits.map(|v| (v - max).exp());
    let sum = exps.iter().sum::<f32>();
    // NaN/Inf logits propagate as NaN so the caller's finiteness check refuses
    // them; a uniform fallback would hide them as value 0.
    exps.map(|v| v / sum)
}

impl<B: Backend, M: NeuralModel<B>> Evaluator for SyncEvaluator<B, M> {
    fn needs_candidate_facts(&self) -> bool {
        self.model.needs_candidate_facts()
    }

    fn evaluate(&self, request: EvalRequest<'_>) -> Result<EvalResult, EvalError> {
        if request.legal.is_empty() {
            return Err(EvalError::Invalid(
                "cannot evaluate a position with no legal actions".into(),
            ));
        }

        // Canonical legal actions -> model candidate tuples (canonical squares).
        let tuples: Vec<(u32, u32, u8)> = request
            .legal
            .iter()
            .map(|id| {
                let (from, to, promo) = id.decode();
                (from as u32, to as u32, promo.code())
            })
            .collect();
        let cb = CandidateBatch::from_lists(&[tuples]);
        let cands = CandidateTensors::from_batch(&cb, &self.device);

        let board = Tensor::<B, 3>::from_data(
            TensorData::new(request.observation.as_slice().to_vec(), [1, 64, 119]),
            &self.device,
        );

        let facts = if self.model.needs_candidate_facts() {
            let rows = request.facts.filter(|f| f.len() == request.legal.len());
            let rows = rows.ok_or_else(|| {
                EvalError::Invalid(
                    "this model requires CandidateFacts aligned to the legal actions".into(),
                )
            })?;
            Some(
                facts_tensor::<B>(&[rows], cb.width, &self.device)
                    .map_err(|e| EvalError::Invalid(format!("facts tensor: {e}")))?,
            )
        } else {
            None
        };

        let out = self
            .model
            .forward_inputs(board, &cands, facts, self.recurrence, false);
        let readout = out
            .readouts
            .first()
            .ok_or_else(|| EvalError::Backend("model produced no readout".into()))?;

        let log_probs = readout
            .policy
            .log_probs
            .clone()
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| EvalError::Backend(format!("policy read failed: {e}")))?;

        let n = request.legal.len();
        let mut policy: Vec<f32> = (0..n).map(|k| log_probs[k].exp()).collect();
        let sum: f32 = policy.iter().sum();
        // A NaN/Inf or all-zero legal policy is a model failure, never a
        // uniform prior (no silent fallback).
        if !(sum.is_finite() && sum > 0.0) {
            return Err(EvalError::Backend(format!(
                "non-finite or zero legal policy mass ({sum})"
            )));
        }
        for p in policy.iter_mut() {
            *p /= sum;
        }

        let wdl_logits = readout
            .wdl_logits
            .clone()
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| EvalError::Backend(format!("wdl read failed: {e}")))?;
        let wdl = softmax3([wdl_logits[0], wdl_logits[1], wdl_logits[2]]);
        let value = wdl[0] - wdl[2];

        if !value.is_finite() || policy.iter().any(|p| !p.is_finite()) {
            return Err(EvalError::Backend("non-finite evaluation output".into()));
        }

        Ok(EvalResult { policy, value, wdl })
    }
}
