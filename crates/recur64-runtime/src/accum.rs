//! One optimizer update over gradient-accumulated micro-batches, for any
//! [`NeuralModel`].
//!
//! The reduction is the learner's, unchanged: each micro-batch loss is scaled
//! by its example count before `backward`, the per-parameter gradients are
//! summed, then divided by the total example count, so the update sees an
//! example-weighted mean over the effective batch
//! (`OPTIMIZER_CONTRACT`: `grad_reduction=example_weighted_mean_over_effective_batch`).

use burn::module::AutodiffModule;
use burn::optim::{GradientsAccumulator, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

use recur64_model::loss::{Targets, model_loss, policy_ce, policy_entropy, wdl_ce};
use recur64_model::model::CandidateTensors;
use recur64_model::net::NeuralModel;
use recur64_model::train::global_grad_norm;

use crate::learner::mean_gradients;

/// Which loss an update optimizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LossMode {
    /// Policy cross-entropy only (exact-proof pretraining; no WDL loss).
    PolicyOnly,
    /// Policy + WDL, the learner's loss.
    Full,
}

/// One micro-batch of prepared tensors.
pub struct MicroBatch<B: Backend> {
    pub board: Tensor<B, 3>,
    pub cands: CandidateTensors<B>,
    /// `[b, width, 8]` for architectures that consume CandidateFacts.
    pub facts: Option<Tensor<B, 3>>,
    pub targets: Targets<B>,
    pub examples: usize,
}

/// Loss components (example-weighted means) and gradient norm of one update.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct UpdateReport {
    pub total_loss: f32,
    pub policy_loss: f32,
    pub wdl_loss: f32,
    pub policy_entropy: f32,
    pub grad_norm: f32,
    pub examples: usize,
}

fn scalar<B: Backend>(t: Tensor<B, 1>) -> f32 {
    t.into_data()
        .to_vec::<f32>()
        .ok()
        .and_then(|v| v.first().copied())
        .unwrap_or(f32::NAN)
}

/// Run one optimizer update over `micros`. Errors visibly on a non-finite loss
/// or gradient norm; the optimizer step is not taken in that case.
pub fn accumulated_update<B, M, O, I>(
    model: M,
    optim: &mut O,
    micros: I,
    lr: f64,
    mode: LossMode,
) -> Result<(M, UpdateReport), String>
where
    B: AutodiffBackend,
    M: NeuralModel<B> + AutodiffModule<B>,
    O: Optimizer<M, B>,
    I: IntoIterator<Item = MicroBatch<B>>,
{
    let mut accumulator = GradientsAccumulator::<M>::new();
    let (mut total, mut policy, mut wdl, mut entropy) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let mut examples = 0usize;
    for micro in micros {
        let n = micro.examples;
        if n == 0 {
            continue;
        }
        let out = model.forward_inputs(micro.board, &micro.cands, micro.facts, 1, false);
        let readout = &out.readouts[0];
        let p = scalar(policy_ce(&readout.policy, &micro.targets.policy_target));
        let w = scalar(wdl_ce(&readout.wdl_logits, &micro.targets.wdl_target));
        let e = scalar(policy_entropy(&readout.policy));
        let loss = match mode {
            LossMode::PolicyOnly => policy_ce(&readout.policy, &micro.targets.policy_target),
            LossMode::Full => model_loss(&out, &micro.targets),
        };
        let l = scalar(loss.clone());
        let grads = GradientsParams::from_grads((loss * n as f32).backward(), &model);
        accumulator.accumulate(&model, grads);
        let weight = n as f32;
        total += l * weight;
        policy += p * weight;
        wdl += w * weight;
        entropy += e * weight;
        examples += n;
    }
    if examples == 0 {
        return Err("update received no examples".into());
    }
    let mut grads = accumulator.grads();
    mean_gradients(&mut grads, &model, examples);
    let grad_norm = global_grad_norm::<B, M>(&grads, &model);
    let d = examples as f32;
    let report = UpdateReport {
        total_loss: total / d,
        policy_loss: policy / d,
        wdl_loss: wdl / d,
        policy_entropy: entropy / d,
        grad_norm,
        examples,
    };
    if !report.total_loss.is_finite() || !grad_norm.is_finite() {
        return Err(format!(
            "non-finite loss/grad: loss={} grad={grad_norm}",
            report.total_loss
        ));
    }
    let model = optim.step(lr, model, grads);
    Ok((model, report))
}
