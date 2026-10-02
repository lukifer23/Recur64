//! Query utility head (`query_utility_head_v1`).
//!
//! Predicts, BEFORE the child is seen, the expected marginal value of executing a frontier
//! edge for the root decision: `U(e | S) = CE_before - CE_after_e` (see the research plan).
//!
//! Law D / invariant 13 are structural: [`QueryUtilityHead::score`] takes only a
//! [`FrontierEdgeView`], whose every field is a function of information that already exists
//! before the edge is executed (the base belief, the ledger so far, the parent state's own
//! encoding and the action's from/to squares read from the PARENT, and scalar frontier
//! structure). A view has no field that could hold the unseen child's content, and the session
//! builds it without ever calling `QueryManager::query`.

use burn::module::Module;
use burn::nn::{Initializer, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation;
use recur64_model::active::features::EDGE_FEATS;
use crate::POLICY_INIT_STD;
use recur64_model::config::EvidenceConfig;

use crate::MASKED_LOGIT;
use crate::util::rows;

/// Scalar belief features per frontier edge:
/// `[z_b - max z, p_b, entropy, p_max, top-2 margin, messages / 16, ||delta||, delta_b]`.
pub const BELIEF_FEATS: usize = 8;

/// Everything the utility head may read about a frontier edge. No child content.
pub struct FrontierEdgeView<B: Backend> {
    /// `[b, F, qd]` action embedding built from the parent's own square features
    /// (root edges: the root hypothesis token).
    pub edge_emb: Tensor<B, 3>,
    /// `[b, F, qd]` pooled encoding of the parent state (already known to the model).
    pub parent_pool: Tensor<B, 3>,
    /// `[b, F, cd]` hypothesis token of the edge's root branch.
    pub branch_token: Tensor<B, 3>,
    /// `[b, F, BELIEF_FEATS]` current belief summary for the edge's branch.
    pub belief: Tensor<B, 3>,
    /// `[b, F, M]` mean of the evidence messages acquired so far.
    pub ledger_summary: Tensor<B, 3>,
    /// `[b, F, EDGE_FEATS]` depth / remaining budget / parity of the edge.
    pub edge_feats: Tensor<B, 3>,
    /// `[b, F]` true for real frontier slots.
    pub mask: Tensor<B, 2, Bool>,
}

#[derive(Module, Debug)]
pub struct QueryUtilityHead<B: Backend> {
    hidden: Linear<B>,
    out: Linear<B>,
}

impl<B: Backend> QueryUtilityHead<B> {
    pub fn new(e: &EvidenceConfig, device: &B::Device) -> Self {
        let d = e.token_dim();
        let input = 2 * d + d + BELIEF_FEATS + e.message_dim + EDGE_FEATS;
        Self {
            hidden: LinearConfig::new(input, e.utility_hidden)
                .with_bias(true)
                .init(device),
            out: LinearConfig::new(e.utility_hidden, 1)
                .with_bias(true)
                .with_initializer(Initializer::Normal {
                    mean: 0.0,
                    std: POLICY_INIT_STD,
                })
                .init(device),
        }
    }

    /// Predicted utility `[b, F]` in nats; padding is `MASKED_LOGIT`.
    pub fn score(&self, view: &FrontierEdgeView<B>) -> Tensor<B, 2> {
        let x = Tensor::cat(
            vec![
                view.edge_emb.clone(),
                view.parent_pool.clone(),
                view.branch_token.clone(),
                view.belief.clone(),
                view.ledger_summary.clone(),
                view.edge_feats.clone(),
            ],
            2,
        );
        let h = activation::gelu(rows(&self.hidden, x));
        rows(&self.out, h)
            .squeeze_dim::<2>(2)
            .mask_fill(view.mask.clone().bool_not(), MASKED_LOGIT)
    }

    pub fn param_breakdown(&self) -> Vec<(&'static str, usize)> {
        vec![(
            "utility.head",
            self.hidden.num_params() + self.out.num_params(),
        )]
    }
}
