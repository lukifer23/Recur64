//! Belief update (`belief_update_gated_sum_v1`).
//!
//! `z_t = z0 + delta_z_t`. Hypothesis tokens read the ledger and produce a per-candidate
//! evidence delta. For hypothesis `i` and message `m_j`:
//!
//! ```text
//! s_ij       = Wo . ( gelu(Wv m_j)  (*)  (1 + tanh(Wh h_i)) )          scalar, content-multiplied
//! trust_j    = tanh( (wt . gelu(Wt m_j))^2 )                            in [0, 1), 0 iff zero
//! gate_ij    = sigmoid( q_i . k_j / sqrt(K) + route[class(i, j)] )      routing only
//! contrib_ij = trust_j * gate_ij * s_ij
//! delta_i    = B * tanh( sum_j contrib_ij / B )                         bounded
//! ```
//!
//! Every path from a message to `delta` is bias-free and f(0)=0 and ends in a product with
//! `trust_j * s_ij`, so a zero message contributes exactly zero whatever the gate (routing)
//! says. The sum over `j` makes the update permutation invariant. `z0` is only ever read.

use burn::module::{Module, Param};
use burn::nn::{Initializer, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::{Distribution, activation};
use recur64_model::config::EvidenceConfig;

use crate::ledger::{EvidenceLedger, ROUTE_CLASSES};

/// Output of one belief update.
pub struct BeliefOutput<B: Backend> {
    /// `[b, w]` bounded additive delta, exactly 0 at padding.
    pub delta: Tensor<B, 2>,
    /// `[b, J]` per-message trust (empty ledger: `None`).
    pub trust: Option<Tensor<B, 2>>,
    /// `[b, w, J]` per-(candidate, message) trusted, gated contribution before the bound.
    pub contrib: Option<Tensor<B, 3>>,
}

#[derive(Module, Debug)]
pub struct BeliefUpdate<B: Backend> {
    /// Content key of a message (bias-free): only used inside the routing gate.
    key_proj: Linear<B>,
    /// Query from the (detached) hypothesis token.
    q_proj: Linear<B>,
    val_proj: Linear<B>,
    hyp_gate: Linear<B>,
    out: Linear<B>,
    trust_h: Linear<B>,
    trust_o: Linear<B>,
    route_bias: Param<Tensor<B, 1>>,
    key_dim: usize,
    delta_bound: f64,
}

impl<B: Backend> BeliefUpdate<B> {
    pub fn new(e: &EvidenceConfig, device: &B::Device) -> Self {
        let (cd, m, p, k, t) = (
            e.token_dim(),
            e.message_dim,
            e.pair_dim,
            e.key_dim,
            e.trust_hidden,
        );
        let nb = |i: usize, o: usize| LinearConfig::new(i, o).with_bias(false).init(device);
        Self {
            key_proj: nb(m, k),
            q_proj: LinearConfig::new(cd, k).with_bias(true).init(device),
            val_proj: nb(m, p),
            hyp_gate: LinearConfig::new(cd, p).with_bias(true).init(device),
            out: LinearConfig::new(p, 1)
                .with_bias(false)
                .with_initializer(Initializer::Normal {
                    mean: 0.0,
                    std: 0.05,
                })
                .init(device),
            trust_h: nb(m, t),
            trust_o: nb(t, 1),
            route_bias: Param::from_tensor(Tensor::random(
                [ROUTE_CLASSES],
                Distribution::Normal(0.0, 0.01),
                device,
            )),
            key_dim: k,
            delta_bound: e.delta_bound,
        }
    }

    /// `h: [b, w, cd]` hypothesis tokens, `mask: [b, w]` valid candidates.
    pub fn forward(
        &self,
        h: Tensor<B, 3>,
        mask: Tensor<B, 2, Bool>,
        ledger: &EvidenceLedger<B>,
    ) -> BeliefOutput<B> {
        let device = h.device();
        let [b, w, _] = h.dims();
        let (Some(msgs), Some(present)) = (&ledger.messages, &ledger.present) else {
            // Empty ledger: the update is the identity by definition (delta exactly 0).
            return BeliefOutput {
                delta: Tensor::zeros([b, w], &device),
                trust: None,
                contrib: None,
            };
        };
        let j = msgs.dims()[1];
        let m = msgs.clone();

        let t_pre = self
            .trust_o
            .forward(activation::gelu(self.trust_h.forward(m.clone())))
            .squeeze_dim::<2>(2);
        let trust = (t_pre.clone() * t_pre).tanh(); // [b, J]

        let p = self.val_proj.forward(m.clone()); // [b, J, P]
        let pd = p.dims()[2];
        let val = activation::gelu(p).unsqueeze_dim::<4>(1).expand([b, w, j, pd]);
        let hg = self
            .hyp_gate
            .forward(h.clone())
            .tanh()
            .add_scalar(1.0)
            .unsqueeze_dim::<4>(2)
            .expand([b, w, j, pd]);
        let s = self.out.forward(hg * val).squeeze_dim::<3>(3); // [b, w, J]

        let q = self.q_proj.forward(h); // [b, w, K]
        let k = self.key_proj.forward(m); // [b, J, K]
        let logits = q
            .matmul(k.swap_dims(1, 2))
            .mul_scalar(1.0 / (self.key_dim as f32).sqrt());
        let classes = ledger.route_classes(w, &device);
        let route = self
            .route_bias
            .val()
            .select(0, classes)
            .reshape([b, w, j]);
        let gate = activation::sigmoid(logits + route);

        let pres = present.clone().unsqueeze_dim::<3>(1).expand([b, w, j]);
        let tr = trust.clone().unsqueeze_dim::<3>(1).expand([b, w, j]);
        let contrib = tr * gate * s * pres;
        let raw = contrib.clone().sum_dim(2).squeeze_dim::<2>(2);
        let bound = self.delta_bound as f32;
        let delta = raw
            .div_scalar(bound)
            .tanh()
            .mul_scalar(bound)
            .mask_fill(mask.bool_not(), 0.0);
        BeliefOutput {
            delta,
            trust: Some(trust),
            contrib: Some(contrib),
        }
    }

    pub fn param_breakdown(&self) -> Vec<(&'static str, usize)> {
        vec![
            (
                "belief.routing_gate",
                self.key_proj.num_params() + self.q_proj.num_params() + self.route_bias.num_params(),
            ),
            (
                "belief.interaction",
                self.val_proj.num_params() + self.hyp_gate.num_params() + self.out.num_params(),
            ),
            (
                "belief.trust",
                self.trust_h.num_params() + self.trust_o.num_params(),
            ),
        ]
    }
}
