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

// --- X15 deep supervision ------------------------------------------------------

/// Which target each supervised thought is trained against, and with what
/// weight. Pure index/weight arithmetic so the mapping is unit-testable
/// without tensors.
///
/// `targets` are ordered by search depth (shallowest rung first, the deepest
/// teacher last), i.e. the `ReasoningTargetsV1` ladder. Returns
/// `(target_index, weight)` per **readout** the forward pass produced:
///
/// * `final_only_v1`: one readout (the final thought), the deepest target,
///   weight 1.
/// * `same_target_v1`: `t` readouts, all against the deepest target; the final
///   thought has weight 1 and every intermediate `intermediate_weight`.
/// * `progressive_search_v1`: `t` readouts; thought `i` against rung `i`. The
///   ladder must have exactly `t` rungs: a target is never reused or
///   reinterpreted to fill a missing rung. Weights as for `same_target_v1`.
pub fn thought_supervision(
    mode: crate::experimental::DeepSupervisionMode,
    intermediate_weight: f32,
    thoughts: usize,
    ladder_len: usize,
) -> anyhow::Result<Vec<(usize, f32)>> {
    use crate::experimental::DeepSupervisionMode as M;
    anyhow::ensure!(thoughts >= 1, "at least one thought is required");
    anyhow::ensure!(ladder_len >= 1, "at least one target is required");
    let weight = |i: usize| {
        if i + 1 == thoughts {
            1.0
        } else {
            intermediate_weight
        }
    };
    Ok(match mode {
        M::FinalOnlyV1 => vec![(ladder_len - 1, 1.0)],
        M::SameTargetV1 => (0..thoughts).map(|i| (ladder_len - 1, weight(i))).collect(),
        M::ProgressiveSearchV1 => {
            anyhow::ensure!(
                ladder_len == thoughts,
                "progressive_search_v1 with {thoughts} thoughts needs exactly {thoughts} \
                 search rungs, got {ladder_len}; select the rungs explicitly rather than \
                 reusing a target"
            );
            (0..thoughts).map(|i| (i, weight(i))).collect()
        }
    })
}

/// Weighted training loss over a Chimera forward pass. `readouts` must be the
/// **training-semantics** output of the configured mode (`forward_thoughts`),
/// not a diagnostic forward: a readout-count mismatch is refused so that
/// diagnostic readouts can never leak into supervision.
pub fn thought_loss<B: Backend>(
    readouts: &[Readout<B>],
    targets: &[Targets<B>],
    mode: crate::experimental::DeepSupervisionMode,
    intermediate_weight: f32,
    thoughts: usize,
) -> anyhow::Result<Tensor<B, 1>> {
    let plan = thought_supervision(mode, intermediate_weight, thoughts, targets.len())?;
    anyhow::ensure!(
        plan.len() == readouts.len(),
        "{} supervises {} readouts but the forward pass produced {} (was a diagnostic \
         forward passed to the training loss?)",
        mode.label(),
        plan.len(),
        readouts.len()
    );
    let mut total: Option<Tensor<B, 1>> = None;
    for (r, (idx, w)) in readouts.iter().zip(plan) {
        let l = readout_loss(r, &targets[idx]) * w;
        total = Some(match total {
            None => l,
            Some(acc) => acc + l,
        });
    }
    Ok(total.expect("plan is non-empty"))
}

#[cfg(test)]
mod thought_tests {
    use super::*;
    use crate::experimental::DeepSupervisionMode as M;

    #[test]
    fn final_only_supervises_the_final_thought_against_the_deepest_target() {
        assert_eq!(
            thought_supervision(M::FinalOnlyV1, 0.25, 4, 4).unwrap(),
            vec![(3, 1.0)]
        );
        assert_eq!(
            thought_supervision(M::FinalOnlyV1, 0.25, 4, 1).unwrap(),
            vec![(0, 1.0)]
        );
    }

    #[test]
    fn same_target_weights_intermediates_and_reuses_the_deepest_target() {
        let plan = thought_supervision(M::SameTargetV1, 0.25, 3, 4).unwrap();
        assert_eq!(plan, vec![(3, 0.25), (3, 0.25), (3, 1.0)]);
        let plan = thought_supervision(M::SameTargetV1, 0.5, 2, 1).unwrap();
        assert_eq!(plan, vec![(0, 0.5), (0, 1.0)]);
    }

    #[test]
    fn progressive_maps_thought_i_to_rung_i_and_refuses_a_short_or_long_ladder() {
        let plan = thought_supervision(M::ProgressiveSearchV1, 0.25, 4, 4).unwrap();
        assert_eq!(plan, vec![(0, 0.25), (1, 0.25), (2, 0.25), (3, 1.0)]);
        assert!(thought_supervision(M::ProgressiveSearchV1, 0.25, 4, 2).is_err());
        assert!(thought_supervision(M::ProgressiveSearchV1, 0.25, 2, 4).is_err());
    }
}
