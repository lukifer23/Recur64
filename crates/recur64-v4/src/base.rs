//! Root hypothesis bank and immutable base belief (`base_readout_v4_v1`).
//!
//! The root board is encoded once. The reused V3 [`RootPath`] (board encoder + candidate
//! tokens + one candidate block, contract `v25_root_encoder_v1` / `candidate_token_v3_root_v1`,
//! V4 weights trained from scratch) yields one token `h_i` per legal root action, and
//! `z0_i = BaseReadout(h_i, root_context)` is the immutable baseline logit. B0 inference ends
//! here: it never touches an evidence module.

use burn::module::Module;
use burn::nn::{Initializer, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation;
use recur64_model::active::modules::RootPath;
use crate::POLICY_INIT_STD;
use recur64_model::config::{EvidenceConfig, ModelConfig};
use recur64_model::model::CandidateTensors;

use crate::MASKED_LOGIT;
use crate::util::rows;

/// `z0_i = out(gelu(hidden([h_i, root_node])))`. The output layer has no bias: a constant added
/// to every candidate is cancelled by the softmax and would be an inert parameter.
#[derive(Module, Debug)]
pub struct BaseReadout<B: Backend> {
    hidden: Linear<B>,
    out: Linear<B>,
}

impl<B: Backend> BaseReadout<B> {
    pub fn new(e: &EvidenceConfig, device: &B::Device) -> Self {
        let d = e.token_dim();
        Self {
            hidden: LinearConfig::new(2 * d, e.base_hidden)
                .with_bias(true)
                .init(device),
            out: LinearConfig::new(e.base_hidden, 1)
                .with_bias(false)
                .with_initializer(Initializer::Normal {
                    mean: 0.0,
                    std: POLICY_INIT_STD,
                })
                .init(device),
        }
    }

    /// Raw logits `[b, w]` (padding not yet masked).
    pub fn logits(&self, tokens: Tensor<B, 3>, root_node: Tensor<B, 2>) -> Tensor<B, 2> {
        let [b, w, d] = tokens.dims();
        let ctx = root_node.unsqueeze_dim::<3>(1).expand([b, w, d]);
        let x = Tensor::cat(vec![tokens, ctx], 2);
        let h = activation::gelu(rows(&self.hidden, x));
        rows(&self.out, h).squeeze_dim::<2>(2)
    }

    pub fn param_breakdown(&self) -> Vec<(&'static str, usize)> {
        vec![(
            "base.readout",
            self.hidden.num_params() + self.out.num_params(),
        )]
    }
}

/// Output of the single root execution.
pub struct BaseStage<B: Backend> {
    /// `[b, w, cd]` hypothesis tokens.
    pub tokens: Tensor<B, 3>,
    /// `[b, qd]` pooled root context.
    pub root_node: Tensor<B, 2>,
    /// `[b, w]` base logits `z0`; padding is `MASKED_LOGIT`.
    pub z0: Tensor<B, 2>,
    pub wdl_logits: Tensor<B, 2>,
}

#[derive(Module, Debug)]
pub struct BaseTower<B: Backend> {
    root: RootPath<B>,
    readout: BaseReadout<B>,
}

impl<B: Backend> BaseTower<B> {
    pub fn new(cfg: &ModelConfig, device: &B::Device) -> Self {
        let e = cfg.evidence.as_ref().expect("evidence geometry");
        Self {
            root: RootPath::new(cfg, &e.shared_geometry(), device),
            readout: BaseReadout::new(e, device),
        }
    }

    /// Execute the root once. This is the whole of B0.
    pub fn forward(
        &self,
        cfg: &ModelConfig,
        board: Tensor<B, 3>,
        cands: &CandidateTensors<B>,
        facts: Tensor<B, 3>,
    ) -> BaseStage<B> {
        assert!(
            cands.width > 0,
            "candidate batch contains no legal candidates; terminal-only batches have no policy path"
        );
        let e = cfg.evidence.as_ref().expect("evidence geometry");
        let stage = self.root.forward(cfg, &e.shared_geometry(), board, cands, facts);
        let z0 = self
            .readout
            .logits(stage.tokens.clone(), stage.root_node.clone())
            .mask_fill(cands.mask.clone().bool_not(), MASKED_LOGIT);
        BaseStage {
            tokens: stage.tokens,
            root_node: stage.root_node,
            z0,
            wdl_logits: stage.wdl_logits,
        }
    }

    pub fn param_breakdown(&self) -> Vec<(&'static str, usize)> {
        let mut v = self.root.param_breakdown();
        v.extend(self.readout.param_breakdown());
        v
    }
}
