//! Real optimizer updates for the Phase 0 probe.

use std::marker::PhantomData;

use burn::module::{AutodiffModule, ModuleVisitor, Param};
use burn::optim::{
    AdamW, AdamWConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor,
    grad_clipping::GradientClippingConfig,
};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

use crate::loss::{Targets, model_loss, policy_ce, wdl_ce};
use crate::model::{CandidateTensors, ProbeModel};
use crate::net::NeuralModel;

/// CPU FP32 autodiff backend used for correctness work in Phase 0.
pub type CpuTrainBackend = burn::backend::Autodiff<burn::backend::Flex>;

/// Versioned description of the optimizer semantics [`adamw`] builds plus the
/// learner's gradient reduction. It is part of the scientific config hash, so
/// any change here must change this string. Burn clips each parameter tensor's
/// L2 norm separately (not one global norm) before AdamW sees the gradient.
pub const OPTIMIZER_CONTRACT: &str = "adamw-v1:burn=0.21.0,beta1=0.9,beta2=0.999,eps=1e-5,\
weight_decay=1e-4,cautious_wd=false,amsgrad=false,clip=per_parameter_l2_norm@1.0,\
grad_reduction=example_weighted_mean_over_effective_batch";

/// AdamW with the initial Phase 0 settings (master spec §7.1). These are
/// starting values to test, not claims of optimality. Betas and epsilon equal
/// the Burn 0.21.0 defaults and are set explicitly to match
/// [`OPTIMIZER_CONTRACT`].
pub fn adamw<B, M>() -> OptimizerAdaptor<AdamW, M, B>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    AdamWConfig::new()
        .with_beta_1(0.9)
        .with_beta_2(0.999)
        .with_epsilon(1e-5)
        .with_weight_decay(1e-4)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init::<B, M>()
}

/// One forward/backward/optimizer step for any [`NeuralModel`]. `facts` is the
/// `[b, width, 8]` CandidateFacts tensor for architectures that consume it.
/// Returns the updated model and the (pre-update) loss.
#[allow(clippy::too_many_arguments)]
pub fn train_step_any<B, M, O>(
    model: M,
    optim: &mut O,
    board: Tensor<B, 3>,
    cands: &CandidateTensors<B>,
    facts: Option<Tensor<B, 3>>,
    targets: &Targets<B>,
    r: usize,
    deep: bool,
    lr: f64,
) -> (M, Tensor<B, 1>)
where
    B: AutodiffBackend,
    M: NeuralModel<B> + AutodiffModule<B>,
    O: Optimizer<M, B>,
{
    let out = model.forward_inputs(board, cands, facts, r, deep);
    let loss = model_loss(&out, targets);
    let grads = GradientsParams::from_grads(loss.backward(), &model);
    let model = optim.step(lr, model, grads);
    (model, loss)
}

/// One forward/backward/optimizer step of the Probe model. Returns the updated
/// model and the (pre-update) loss.
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
    train_step_any(model, optim, board, cands, None, targets, r, deep, lr)
}

/// Loss components and gradient norm for one optimizer step.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct StepReport {
    pub total_loss: f32,
    pub policy_loss: f32,
    pub wdl_loss: f32,
    pub grad_norm: f32,
}

fn scalar1<B: Backend>(t: Tensor<B, 1>) -> f32 {
    t.into_data()
        .to_vec::<f32>()
        .ok()
        .and_then(|v| v.first().copied())
        .unwrap_or(f32::NAN)
}

struct GradNormVisitor<'a, B: AutodiffBackend> {
    grads: &'a GradientsParams,
    sum_sq: f64,
    _p: PhantomData<B>,
}

impl<B: AutodiffBackend> ModuleVisitor<B> for GradNormVisitor<'_, B> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        if let Some(g) = self.grads.get::<B::InnerBackend, D>(param.id) {
            let sq = scalar1((g.clone() * g).sum());
            if sq.is_finite() {
                self.sum_sq += sq as f64;
            }
        }
    }
}

/// Global L2 norm of the parameter gradients.
pub fn global_grad_norm<B: AutodiffBackend, M: AutodiffModule<B>>(
    grads: &GradientsParams,
    model: &M,
) -> f32 {
    let mut visitor = GradNormVisitor::<B> {
        grads,
        sum_sq: 0.0,
        _p: PhantomData,
    };
    model.visit(&mut visitor);
    visitor.sum_sq.sqrt() as f32
}

/// One forward/backward/optimizer step of any [`NeuralModel`] returning loss
/// components and gradient norm.
#[allow(clippy::too_many_arguments)]
pub fn train_step_reporting_any<B, M, O>(
    model: M,
    optim: &mut O,
    board: Tensor<B, 3>,
    cands: &CandidateTensors<B>,
    facts: Option<Tensor<B, 3>>,
    targets: &Targets<B>,
    r: usize,
    deep: bool,
    lr: f64,
) -> (M, StepReport)
where
    B: AutodiffBackend,
    M: NeuralModel<B> + AutodiffModule<B>,
    O: Optimizer<M, B>,
{
    let out = model.forward_inputs(board, cands, facts, r, deep);
    let readout = &out.readouts[0];
    let policy_loss = scalar1(policy_ce(&readout.policy, &targets.policy_target));
    let wdl_loss = scalar1(wdl_ce(&readout.wdl_logits, &targets.wdl_target));
    let loss = model_loss(&out, targets);
    let total_loss = scalar1(loss.clone());
    let grads = GradientsParams::from_grads(loss.backward(), &model);
    let grad_norm = global_grad_norm::<B, M>(&grads, &model);
    let model = optim.step(lr, model, grads);
    (
        model,
        StepReport {
            total_loss,
            policy_loss,
            wdl_loss,
            grad_norm,
        },
    )
}

/// One forward/backward/optimizer step of the Probe model returning loss
/// components and gradient norm.
#[allow(clippy::too_many_arguments)]
pub fn train_step_reporting<B, O>(
    model: ProbeModel<B>,
    optim: &mut O,
    board: Tensor<B, 3>,
    cands: &CandidateTensors<B>,
    targets: &Targets<B>,
    r: usize,
    deep: bool,
    lr: f64,
) -> (ProbeModel<B>, StepReport)
where
    B: AutodiffBackend,
    O: Optimizer<ProbeModel<B>, B>,
{
    train_step_reporting_any(model, optim, board, cands, None, targets, r, deep, lr)
}
