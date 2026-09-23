//! Real optimizer updates for the Phase 0 probe.

use burn::module::AutodiffModule;
use burn::optim::{
    AdamW, AdamWConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor,
    grad_clipping::GradientClippingConfig,
};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

use crate::loss::{Targets, model_loss};
use crate::model::{CandidateTensors, ProbeModel};

/// CPU FP32 autodiff backend used for correctness work in Phase 0.
pub type CpuTrainBackend = burn::backend::Autodiff<burn::backend::Flex>;

/// AdamW with the initial Phase 0 settings (master spec §7.1). These are
/// starting values to test, not claims of optimality.
pub fn adamw<B, M>() -> OptimizerAdaptor<AdamW, M, B>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    AdamWConfig::new()
        .with_weight_decay(1e-4)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init::<B, M>()
}

/// One forward/backward/optimizer step. Returns the updated model and the
/// (pre-update) loss.
#[allow(clippy::too_many_arguments)]
pub fn train_step<B, O>(
    model: ProbeModel<B>,
    optim: &mut O,
    board: Tensor<B, 3>,
    cands: &CandidateTensors<B>,
    targets: &Targets<B>,
    r: usize,
    deep: bool,
    lr: f64,
) -> (ProbeModel<B>, Tensor<B, 1>)
where
    B: AutodiffBackend,
    O: Optimizer<ProbeModel<B>, B>,
{
    let out = model.forward_r(board, cands, r, deep);
    let loss = model_loss(&out, targets);
    let grads = GradientsParams::from_grads(loss.backward(), &model);
    let model = optim.step(lr, model, grads);
    (model, loss)
}
