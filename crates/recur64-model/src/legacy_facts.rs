//! P2.5 LF (`legacy_facts_v25`): the legacy head-v2 policy plus CandidateFacts.
//!
//! The board encoder and the source/destination/promotion policy are exactly the
//! historical [`ProbeModel`] (wrapped unchanged, so its record layout and identity
//! are untouched). CandidateFacts enter as a small candidate-local POLICY DELTA:
//!
//! `final_logit_i = base_logit_i + Linear_nobias(hidden -> 1)(GELU(Linear(8 -> hidden)(facts_i)))`
//!
//! and then the usual masked legal softmax. There is no gain multiplier and no
//! zero-gated path; the final layer has a small NONZERO initialization so the fresh
//! policy stays near the legacy prior while every facts parameter receives gradient
//! from update 1. Facts do not feed the WDL head (the legacy value path is unchanged).

use burn::module::{Module, ModuleVisitor, Param};
use burn::nn::{Initializer, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::{Int, activation};

use crate::config::{Architecture, CANDIDATE_FACT_FIELDS, ModelConfig};
use crate::model::{CandidateTensors, ModelOutput, ProbeModel, linear_rows};

/// Std of the final fact-delta layer's weights (structural, same rule as the
/// CandidateV25 policy scorer; fixed before any result).
const FACT_DELTA_INIT_STD: f64 = 0.01;

#[derive(Module, Debug)]
pub struct LegacyFactsModel<B: Backend> {
    probe: ProbeModel<B>,
    facts1: Linear<B>,
    facts2: Linear<B>,
    cfg: ModelConfig,
}

impl<B: Backend> LegacyFactsModel<B> {
    pub fn new(cfg: ModelConfig, device: &B::Device) -> Self {
        cfg.validate()
            .expect("valid legacy_facts_v25 configuration");
        assert_eq!(
            cfg.architecture,
            Architecture::LegacyFactsV25,
            "LegacyFactsModel requires architecture legacy_facts_v25"
        );
        let hidden = cfg
            .legacy_facts
            .as_ref()
            .expect("facts-delta geometry")
            .facts_hidden;
        // The wrapped legacy model sees a plain probe_v1 configuration.
        let mut probe_cfg = cfg.clone();
        probe_cfg.architecture = Architecture::ProbeV1;
        probe_cfg.legacy_facts = None;
        let model = Self {
            probe: ProbeModel::new(probe_cfg, device),
            facts1: LinearConfig::new(CANDIDATE_FACT_FIELDS, hidden)
                .with_bias(true)
                .init(device),
            // No bias: a constant added to every candidate of a row is cancelled by the
            // softmax, so it would be an inert parameter with zero policy gradient.
            facts2: LinearConfig::new(hidden, 1)
                .with_bias(false)
                .with_initializer(Initializer::Normal {
                    mean: 0.0,
                    std: FACT_DELTA_INIT_STD,
                })
                .init(device),
            cfg,
        };
        model.force_init();
        model
    }

    /// Materialize every lazily initialized parameter (see `ProbeModel::new`).
    pub fn force_init(&self) {
        struct Init;
        impl<B: Backend> ModuleVisitor<B> for Init {
            fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
                let _ = param.val();
            }
            fn visit_int<const D: usize>(&mut self, param: &Param<Tensor<B, D, Int>>) {
                let _ = param.val();
            }
            fn visit_bool<const D: usize>(&mut self, param: &Param<Tensor<B, D, Bool>>) {
                let _ = param.val();
            }
        }
        let mut visitor = Init;
        self.visit(&mut visitor);
    }

    pub fn config(&self) -> &ModelConfig {
        &self.cfg
    }

    /// The wrapped, unmodified legacy model (tests compare against it).
    pub fn probe(&self) -> &ProbeModel<B> {
        &self.probe
    }

    /// Candidate-local logit delta `[b, width]` from `facts` `[b, width, 8]`.
    pub fn fact_delta(&self, facts: Tensor<B, 3>) -> Tensor<B, 2> {
        let h = activation::gelu(linear_rows(&self.facts1, facts));
        linear_rows(&self.facts2, h).squeeze_dim::<2>(2)
    }

    /// One pass. `facts` is `[b, width, 8]`, aligned with the candidate tensors.
    pub fn forward(
        &self,
        board: Tensor<B, 3>,
        cands: &CandidateTensors<B>,
        facts: Tensor<B, 3>,
    ) -> ModelOutput<B> {
        let [b, w] = cands.mask.dims();
        assert_eq!(
            facts.dims(),
            [b, w, CANDIDATE_FACT_FIELDS],
            "facts tensor must be [batch, candidate width, {CANDIDATE_FACT_FIELDS}]"
        );
        let delta = self.fact_delta(facts);
        self.probe.forward_with_logit_delta(board, cands, delta)
    }

    pub fn num_params(&self) -> usize {
        Module::num_params(self)
    }

    /// Exact parameter count by group; sums to [`Self::num_params`].
    pub fn param_breakdown(&self) -> Vec<(&'static str, usize)> {
        let mut v = self.probe.param_breakdown();
        v.push((
            "facts_delta_mlp",
            self.facts1.num_params() + self.facts2.num_params(),
        ));
        v
    }

    /// Identity of the first fact-delta weight (gradient tests).
    pub fn facts_weight_id(&self) -> burn::module::ParamId {
        self.facts1.weight.id
    }

    /// Identity of the first fact-delta layer's bias (gradient tests).
    pub fn facts_bias_id(&self) -> burn::module::ParamId {
        self.facts1.bias.as_ref().expect("facts1 has a bias").id
    }

    /// Whether the final fact-delta layer has a bias (it must not).
    pub fn delta_layer_has_bias(&self) -> bool {
        self.facts2.bias.is_some()
    }

    /// Identity of the final fact-delta weight (gradient tests).
    pub fn delta_weight_id(&self) -> burn::module::ParamId {
        self.facts2.weight.id
    }
}
