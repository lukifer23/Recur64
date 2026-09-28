//! Losses for the Phase 0 probe.
//!
//! Policy cross-entropy is computed only over supplied legal candidates
//! (padded entries contribute exactly zero and are excluded). Terminal
//! positions have no legal candidates and are excluded from the policy term
//! entirely — they never enter an all-masked softmax.

use burn::prelude::*;
use burn::tensor::activation;

use crate::model::{ModelOutput, PolicyOutput, Readout};

/// Targets for one probe batch.
pub struct Targets<B: Backend> {
    /// `[batch, width]` target probability over candidates (0 on padding).
    pub policy_target: Tensor<B, 2>,
    /// `[batch]` WDL class index in `{0=win, 1=draw, 2=loss}`.
    pub wdl_target: Tensor<B, 1, Int>,
    /// `[batch]` 1.0 where the WDL target is supervised, 0.0 where it is not
    /// (a truncated game has no result). `None` = every row is supervised,
    /// which keeps the unmasked mean bit-identical to every earlier path.
    pub wdl_mask: Option<Tensor<B, 1>>,
}

/// Policy cross-entropy over legal candidates, averaged over valid positions.
pub fn policy_ce<B: Backend>(out: &PolicyOutput<B>, target: &Tensor<B, 2>) -> Tensor<B, 1> {
    let per = (out.log_probs.clone() * target.clone())
        .sum_dim(1)
        .squeeze_dim::<1>(1); // [batch]
    let per = per.mask_fill(out.valid.clone().bool_not(), 0.0);
    let n = out.valid.clone().float().sum().clamp(1.0, f32::MAX);
    (-per.sum()) / n
}

/// WDL cross-entropy from side-to-move perspective.
pub fn wdl_ce<B: Backend>(logits: &Tensor<B, 2>, target: &Tensor<B, 1, Int>) -> Tensor<B, 1> {
    let logp = activation::log_softmax(logits.clone(), 1);
    let picked = logp
        .gather(1, target.clone().unsqueeze_dim::<2>(1))
        .squeeze_dim::<1>(1);
    -picked.mean()
}

/// WDL cross-entropy averaged over supervised rows only (`mask` = 1.0).
/// `None` is exactly [`wdl_ce`].
pub fn wdl_ce_masked<B: Backend>(
    logits: &Tensor<B, 2>,
    target: &Tensor<B, 1, Int>,
    mask: Option<&Tensor<B, 1>>,
) -> Tensor<B, 1> {
    let Some(mask) = mask else {
        return wdl_ce(logits, target);
    };
    let logp = activation::log_softmax(logits.clone(), 1);
    let picked = logp
        .gather(1, target.clone().unsqueeze_dim::<2>(1))
        .squeeze_dim::<1>(1);
    let n = mask.clone().sum().clamp(1.0, f32::MAX);
    -(picked * mask.clone()).sum() / n
}

/// Mean predicted-policy entropy over valid positions (nats).
pub fn policy_entropy<B: Backend>(out: &PolicyOutput<B>) -> Tensor<B, 1> {
    let p = out.log_probs.clone().exp();
    let per = -(p * out.log_probs.clone()).sum_dim(1).squeeze_dim::<1>(1);
    let per = per.mask_fill(out.valid.clone().bool_not(), 0.0);
    let n = out.valid.clone().float().sum().clamp(1.0, f32::MAX);
    per.sum() / n
}

/// Combined loss for one readout.
pub fn readout_loss<B: Backend>(r: &Readout<B>, t: &Targets<B>) -> Tensor<B, 1> {
    policy_ce(&r.policy, &t.policy_target)
        + wdl_ce_masked(&r.wdl_logits, &t.wdl_target, t.wdl_mask.as_ref())
}

/// Mean of the per-readout losses. Averaging (rather than summing) means
/// increasing recurrence does not scale the loss.
pub fn model_loss<B: Backend>(out: &ModelOutput<B>, t: &Targets<B>) -> Tensor<B, 1> {
    assert!(!out.readouts.is_empty(), "no readouts to compute loss over");
    let mut total: Option<Tensor<B, 1>> = None;
    for r in &out.readouts {
        let l = readout_loss(r, t);
        total = Some(match total {
            None => l,
            Some(acc) => acc + l,
        });
    }
    total.unwrap() / out.readouts.len() as f32
}
