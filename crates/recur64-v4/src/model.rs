//! `evidence_belief_v4`: the assembled network.
//!
//! Parameter groups (all independent of the query budget):
//! * base: `BaseTower` (root encoder, hypothesis tokens, `BaseReadout`) produces `z0`;
//! * evidence: `EvidenceEncoder` + `BeliefUpdate` (bias-free, content-causal);
//! * utility: the parent/action `QueryEncoder` (reused V3 module, utility path only) and the
//!   `QueryUtilityHead`.

use burn::module::{Module, ModuleVisitor, Param};
use burn::prelude::*;
use burn::tensor::{Int, activation};
use recur64_model::active::modules::QueryEncoder;
use recur64_model::config::{Architecture, EvidenceConfig, ModelConfig};
use recur64_model::model::{CandidateTensors, ModelOutput, PolicyOutput, Readout};
use recur64_model::net::NeuralModel;

use crate::base::{BaseStage, BaseTower};
use crate::belief::BeliefUpdate;
use crate::content::EvidenceEncoder;
use crate::utility::QueryUtilityHead;

#[derive(Module, Debug)]
pub struct EvidenceBeliefModel<B: Backend> {
    pub(crate) base: BaseTower<B>,
    pub(crate) encoder: EvidenceEncoder<B>,
    pub(crate) update: BeliefUpdate<B>,
    pub(crate) state: QueryEncoder<B>,
    pub(crate) utility: QueryUtilityHead<B>,
    cfg: ModelConfig,
}

impl<B: Backend> EvidenceBeliefModel<B> {
    pub fn new(cfg: ModelConfig, device: &B::Device) -> Self {
        cfg.validate().expect("valid evidence_belief_v4 configuration");
        assert_eq!(
            cfg.architecture,
            Architecture::EvidenceBeliefV4,
            "EvidenceBeliefModel requires architecture evidence_belief_v4"
        );
        let e = cfg.evidence.clone().expect("evidence geometry");
        let model = Self {
            base: BaseTower::new(&cfg, device),
            encoder: EvidenceEncoder::new(&e, cfg.rms_eps, device),
            update: BeliefUpdate::new(&e, device),
            state: QueryEncoder::new(&cfg, &e.shared_geometry(), device),
            utility: QueryUtilityHead::new(&e, device),
            cfg,
        };
        model.force_init();
        model
    }

    /// Materialise every lazily initialised parameter.
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
        self.visit(&mut Init);
    }

    pub fn config(&self) -> &ModelConfig {
        &self.cfg
    }

    pub fn evidence(&self) -> &EvidenceConfig {
        self.cfg.evidence.as_ref().expect("evidence geometry")
    }

    /// B0: the root once. No evidence module executes.
    pub fn base_stage(
        &self,
        board: Tensor<B, 3>,
        cands: &CandidateTensors<B>,
        facts: Tensor<B, 3>,
    ) -> BaseStage<B> {
        self.base.forward(&self.cfg, board, cands, facts)
    }

    /// Masked log-probabilities from logits `z` (padding exactly 0 in the output).
    pub fn log_probs(z: Tensor<B, 2>, cands_mask: &Tensor<B, 2, Bool>) -> Tensor<B, 2> {
        let invalid = cands_mask.clone().bool_not();
        activation::log_softmax(z, 1).mask_fill(invalid, 0.0)
    }

    pub fn num_params(&self) -> usize {
        Module::num_params(self)
    }

    /// Exact parameter count by subsystem; sums to [`Self::num_params`].
    pub fn param_breakdown(&self) -> Vec<(&'static str, usize)> {
        let mut v = self.base.param_breakdown();
        v.extend(self.encoder.param_breakdown());
        v.extend(self.update.param_breakdown());
        v.extend(self.state.param_breakdown());
        v.extend(self.utility.param_breakdown());
        v
    }

    /// `(base, evidence, utility)` parameter counts.
    pub fn group_counts(&self) -> (usize, usize, usize) {
        let sum = |p: &str| -> usize {
            self.param_breakdown()
                .into_iter()
                .filter(|(n, _)| n.starts_with(p))
                .map(|(_, c)| c)
                .sum()
        };
        let base = sum("root.") + sum("base.");
        let evidence = sum("evidence.") + sum("belief.");
        let utility = sum("query.") + sum("utility.");
        (base, evidence, utility)
    }
}

impl<B: Backend> NeuralModel<B> for EvidenceBeliefModel<B> {
    const ARCHITECTURE: Architecture = Architecture::EvidenceBeliefV4;

    fn build(cfg: &ModelConfig, device: &B::Device) -> anyhow::Result<Self> {
        cfg.validate()?;
        anyhow::ensure!(
            cfg.architecture == Architecture::EvidenceBeliefV4,
            "cannot build an evidence_belief_v4 model from a {} configuration",
            cfg.architecture.id()
        );
        Ok(EvidenceBeliefModel::new(cfg.clone(), device))
    }

    fn model_config(&self) -> &ModelConfig {
        self.config()
    }

    fn needs_candidate_facts(&self) -> bool {
        true
    }

    /// The trait's batched path has no query tool, so it is explicitly B0: the base belief only.
    fn forward_inputs(
        &self,
        board: Tensor<B, 3>,
        cands: &CandidateTensors<B>,
        facts: Option<Tensor<B, 3>>,
        recurrence: usize,
        deep_supervision: bool,
    ) -> ModelOutput<B> {
        assert_eq!(
            recurrence, 1,
            "evidence_belief_v4 has no recurrence; recurrence {recurrence} is refused"
        );
        assert!(
            !deep_supervision,
            "evidence_belief_v4 has no per-step readouts; deep supervision is refused"
        );
        let facts =
            facts.expect("evidence_belief_v4 requires CandidateFactsV1 (facts tensor is None)");
        let stage = self.base_stage(board, cands, facts);
        let log_probs = Self::log_probs(stage.z0, &cands.mask);
        ModelOutput {
            readouts: vec![Readout {
                policy: PolicyOutput {
                    log_probs,
                    mask: cands.mask.clone(),
                    valid: cands.valid.clone(),
                    base_all: None,
                },
                wdl_logits: stage.wdl_logits,
            }],
            executed_blocks: self.cfg.core_blocks,
        }
    }

    fn param_count(&self) -> usize {
        self.num_params()
    }

    fn param_groups(&self) -> Vec<(&'static str, usize)> {
        self.param_breakdown()
    }
}

impl<B: Backend> EvidenceBeliefModel<B> {
    /// Encode a batch of query content into evidence messages `[n, message_dim]`.
    pub fn encode_content(&self, content: &crate::content::ContentBatch<B>) -> Tensor<B, 2> {
        self.encoder.forward(content)
    }

    /// Run the belief update for explicit hypothesis tokens and a ledger (no session needed).
    pub fn belief_update(
        &self,
        tokens: Tensor<B, 3>,
        mask: Tensor<B, 2, Bool>,
        ledger: &crate::ledger::EvidenceLedger<B>,
    ) -> crate::belief::BeliefOutput<B> {
        self.update.forward(tokens, mask, ledger)
    }
}
