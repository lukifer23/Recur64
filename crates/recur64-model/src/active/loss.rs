//! Selector (process) loss: cross-entropy of the selector distribution against a
//! target distribution over admissible frontier edges.
//!
//! The target is supplied by a [`super::run::QueryScript`]; it is uniform over a
//! set (see `docs/V3_RESEARCH_PLAN.md`, ProofTraceV1), so choosing one admissible
//! edge before another is never penalised.

use burn::prelude::*;
use burn::tensor::activation;

use super::run::SelectorStep;

/// Mean negative log-likelihood over every (step, example) that has a target.
/// `None` when no step carries a target.
pub fn selector_loss<B: Backend>(steps: &[SelectorStep<B>]) -> Option<Tensor<B, 1>> {
    let mut total: Option<Tensor<B, 1>> = None;
    let mut count = 0.0f32;
    for s in steps {
        let n_targets = s.has_target.iter().filter(|&&t| t).count();
        if n_targets == 0 {
            continue;
        }
        let device = s.logits.device();
        let logp = activation::log_softmax(s.logits.clone(), 1);
        let nll = (s.target.clone() * logp)
            .sum_dim(1)
            .neg()
            .squeeze_dim::<1>(1);
        let weight = Tensor::<B, 1>::from_data(
            burn::tensor::TensorData::new(
                s.has_target
                    .iter()
                    .map(|&t| f32::from(u8::from(t)))
                    .collect::<Vec<_>>(),
                [s.has_target.len()],
            ),
            &device,
        );
        let part = (nll * weight).sum();
        total = Some(match total {
            Some(t) => t + part,
            None => part,
        });
        count += n_targets as f32;
    }
    total.map(|t| t / count.max(1.0))
}
