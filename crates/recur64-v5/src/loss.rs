//! Stable correct-set objective and reference metrics.

use burn::prelude::*;
use burn::tensor::Bool;

use crate::MASKED_LOGIT;

fn masked_logsumexp<B: Backend>(x: Tensor<B, 2>, mask: Tensor<B, 2, Bool>) -> Tensor<B, 1> {
    let masked = x.mask_fill(mask.clone().bool_not(), MASKED_LOGIT);
    let max = masked.clone().max_dim(1);
    let sum = ((masked - max.clone().expand(mask.dims())).exp() * mask.float()).sum_dim(1);
    max.squeeze_dim::<1>(1) + sum.clamp(1.0e-30, f32::MAX).log().squeeze_dim::<1>(1)
}

pub fn correct_set_loss<B: Backend>(
    logits: Tensor<B, 2>,
    legal: Tensor<B, 2, Bool>,
    correct: Tensor<B, 2, Bool>,
) -> Tensor<B, 1> {
    // Batch construction validates nonempty legal/correct intersections. Keep
    // the differentiable loss device-resident; a host read here would retain
    // and synchronize the training graph on every microbatch.
    let legal_lse = masked_logsumexp(logits.clone(), legal.clone());
    let correct_lse = masked_logsumexp(logits, legal.bool_and(correct));
    (legal_lse - correct_lse).mean()
}

pub fn reference_set_loss(logits: &[f32], legal: &[bool], correct: &[bool]) -> f64 {
    fn lse(xs: impl Iterator<Item = f64>) -> f64 {
        let v: Vec<f64> = xs.collect();
        let m = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        m + (v.iter().map(|x| (x - m).exp()).sum::<f64>()).ln()
    }
    lse(logits
        .iter()
        .zip(legal)
        .filter(|x| *x.1)
        .map(|x| f64::from(*x.0)))
        - lse(logits
            .iter()
            .zip(legal)
            .zip(correct)
            .filter(|x| *x.0.1 && *x.1)
            .map(|x| f64::from(*x.0.0)))
}
