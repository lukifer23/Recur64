//! The interface the runtime, trainer and checkpoint code need from a network.
//!
//! Implemented by [`ProbeModel`] (historical control) and [`CandidateV25Model`].
//! Runtime dispatch happens on `ModelConfig::architecture` by monomorphizing over
//! the concrete model type, not by an enum inside the model: a Burn `Module`
//! enum would change the recorded checkpoint layout and break every historical
//! Probe checkpoint.

use burn::module::Module;
use burn::prelude::*;

use crate::active::ActiveSearchModel;
use crate::all_info::AllInfoModel;
use crate::candidate::CandidateV25Model;
use crate::config::{Architecture, ModelConfig};
use crate::legacy_facts::LegacyFactsModel;
use crate::model::{CandidateTensors, ModelOutput, ProbeModel};

/// A trainable, checkpointable chess network.
pub trait NeuralModel<B: Backend>: Module<B> + Clone + Send + Sync + 'static {
    /// The architecture this type implements.
    const ARCHITECTURE: Architecture;

    /// Build a fresh, eagerly initialized network. Errors visibly on a
    /// configuration that is not this architecture.
    fn build(cfg: &ModelConfig, device: &B::Device) -> anyhow::Result<Self>;

    fn model_config(&self) -> &ModelConfig;

    /// Whether `forward_inputs` consumes `CandidateFactsV1`.
    fn needs_candidate_facts(&self) -> bool;

    /// One forward pass. `facts` is `[b, width, 8]` when
    /// [`Self::needs_candidate_facts`], else `None`. `recurrence` and
    /// `deep_supervision` are Probe concepts; the candidate model refuses
    /// anything but one pass.
    fn forward_inputs(
        &self,
        board: Tensor<B, 3>,
        cands: &CandidateTensors<B>,
        facts: Option<Tensor<B, 3>>,
        recurrence: usize,
        deep_supervision: bool,
    ) -> ModelOutput<B>;

    fn param_count(&self) -> usize;
    fn param_groups(&self) -> Vec<(&'static str, usize)>;
}

impl<B: Backend> NeuralModel<B> for ProbeModel<B> {
    const ARCHITECTURE: Architecture = Architecture::ProbeV1;

    fn build(cfg: &ModelConfig, device: &B::Device) -> anyhow::Result<Self> {
        cfg.validate()?;
        anyhow::ensure!(
            cfg.architecture == Architecture::ProbeV1,
            "cannot build a probe_v1 model from a {} configuration",
            cfg.architecture.id()
        );
        Ok(ProbeModel::new(cfg.clone(), device))
    }

    fn model_config(&self) -> &ModelConfig {
        self.config()
    }

    fn needs_candidate_facts(&self) -> bool {
        false
    }

    fn forward_inputs(
        &self,
        board: Tensor<B, 3>,
        cands: &CandidateTensors<B>,
        facts: Option<Tensor<B, 3>>,
        recurrence: usize,
        deep_supervision: bool,
    ) -> ModelOutput<B> {
        assert!(
            facts.is_none(),
            "probe_v1 does not consume CandidateFacts; refusing to silently drop them"
        );
        self.forward_r(board, cands, recurrence, deep_supervision)
    }

    fn param_count(&self) -> usize {
        self.num_params()
    }

    fn param_groups(&self) -> Vec<(&'static str, usize)> {
        self.param_breakdown()
    }
}

impl<B: Backend> NeuralModel<B> for CandidateV25Model<B> {
    const ARCHITECTURE: Architecture = Architecture::CandidateV25;

    fn build(cfg: &ModelConfig, device: &B::Device) -> anyhow::Result<Self> {
        cfg.validate()?;
        anyhow::ensure!(
            cfg.architecture == Architecture::CandidateV25,
            "cannot build a candidate_v25 model from a {} configuration",
            cfg.architecture.id()
        );
        Ok(CandidateV25Model::new(cfg.clone(), device))
    }

    fn model_config(&self) -> &ModelConfig {
        self.config()
    }

    fn needs_candidate_facts(&self) -> bool {
        true
    }

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
            "candidate_v25 is one pass; recurrence {recurrence} is refused"
        );
        assert!(
            !deep_supervision,
            "candidate_v25 has no recurrent readouts; deep supervision is refused"
        );
        let facts = facts.expect("candidate_v25 requires CandidateFactsV1 (facts tensor is None)");
        self.forward(board, cands, facts)
    }

    fn param_count(&self) -> usize {
        self.num_params()
    }

    fn param_groups(&self) -> Vec<(&'static str, usize)> {
        self.param_breakdown()
    }
}

impl<B: Backend> NeuralModel<B> for LegacyFactsModel<B> {
    const ARCHITECTURE: Architecture = Architecture::LegacyFactsV25;

    fn build(cfg: &ModelConfig, device: &B::Device) -> anyhow::Result<Self> {
        cfg.validate()?;
        anyhow::ensure!(
            cfg.architecture == Architecture::LegacyFactsV25,
            "cannot build a legacy_facts_v25 model from a {} configuration",
            cfg.architecture.id()
        );
        Ok(LegacyFactsModel::new(cfg.clone(), device))
    }

    fn model_config(&self) -> &ModelConfig {
        self.config()
    }

    fn needs_candidate_facts(&self) -> bool {
        true
    }

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
            "legacy_facts_v25 is one pass; recurrence {recurrence} is refused"
        );
        assert!(
            !deep_supervision,
            "legacy_facts_v25 has no recurrent readouts; deep supervision is refused"
        );
        let facts =
            facts.expect("legacy_facts_v25 requires CandidateFactsV1 (facts tensor is None)");
        self.forward(board, cands, facts)
    }

    fn param_count(&self) -> usize {
        self.num_params()
    }

    fn param_groups(&self) -> Vec<(&'static str, usize)> {
        self.param_breakdown()
    }
}

impl<B: Backend> NeuralModel<B> for ActiveSearchModel<B> {
    const ARCHITECTURE: Architecture = Architecture::ActiveSearchV3;

    fn build(cfg: &ModelConfig, device: &B::Device) -> anyhow::Result<Self> {
        cfg.validate()?;
        anyhow::ensure!(
            cfg.architecture == Architecture::ActiveSearchV3,
            "cannot build an active_search_v3 model from a {} configuration",
            cfg.architecture.id()
        );
        Ok(ActiveSearchModel::new(cfg.clone(), device))
    }

    fn model_config(&self) -> &ModelConfig {
        self.config()
    }

    fn needs_candidate_facts(&self) -> bool {
        true
    }

    /// The trait's batched path has no query tool, so it is explicitly budget 0:
    /// one root encoding and zero queries. Budgets above 0 go through
    /// [`ActiveSearchModel::run`], never through this method.
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
            "active_search_v3 has no recurrence; recurrence {recurrence} is refused"
        );
        assert!(
            !deep_supervision,
            "active_search_v3 has no per-step readouts; deep supervision is refused"
        );
        let facts =
            facts.expect("active_search_v3 requires CandidateFactsV1 (facts tensor is None)");
        self.forward_b0(board, cands, facts)
    }

    fn param_count(&self) -> usize {
        self.num_params()
    }

    fn param_groups(&self) -> Vec<(&'static str, usize)> {
        self.param_breakdown()
    }
}

impl<B: Backend> NeuralModel<B> for AllInfoModel<B> {
    const ARCHITECTURE: Architecture = Architecture::AllInfoV1;

    fn build(cfg: &ModelConfig, device: &B::Device) -> anyhow::Result<Self> {
        cfg.validate()?;
        anyhow::ensure!(
            cfg.architecture == Architecture::AllInfoV1,
            "cannot build an all_info_v1 model from a {} configuration",
            cfg.architecture.id()
        );
        Ok(AllInfoModel::new(cfg.clone(), device))
    }

    fn model_config(&self) -> &ModelConfig {
        self.config()
    }

    fn needs_candidate_facts(&self) -> bool {
        true
    }

    /// The trait's batched path carries no tree, so it is explicitly refused: ALL-INFO is
    /// defined only on the exhaustive depth-2 tree (`AllInfoModel::forward_trees`).
    fn forward_inputs(
        &self,
        _board: Tensor<B, 3>,
        _cands: &CandidateTensors<B>,
        _facts: Option<Tensor<B, 3>>,
        _recurrence: usize,
        _deep_supervision: bool,
    ) -> ModelOutput<B> {
        self.no_tree_forward()
    }

    fn param_count(&self) -> usize {
        self.num_params()
    }

    fn param_groups(&self) -> Vec<(&'static str, usize)> {
        self.param_breakdown()
    }
}
