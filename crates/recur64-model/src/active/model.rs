//! `active_search_v3`: one root encoding plus a budgeted number of exact state
//! queries. The same weights run at every budget; budget 0 is the same model
//! with zero queries, zero query-encoder executions and zero planner updates.

use burn::module::{Module, ModuleVisitor, Param};
use burn::prelude::*;
use burn::tensor::{Int, activation};

use crate::config::{ActiveConfig, Architecture, ModelConfig};
use crate::model::{CandidateTensors, ModelOutput, PolicyOutput, Readout};

use super::modules::{Planner, QueryEncoder, RootPath, RootReadout, RootStage, Selector};

/// The V3 active-search network.
#[derive(Module, Debug)]
pub struct ActiveSearchModel<B: Backend> {
    pub(crate) root: RootPath<B>,
    pub(crate) query: QueryEncoder<B>,
    pub(crate) planner: Planner<B>,
    pub(crate) selector: Selector<B>,
    pub(crate) readout: RootReadout<B>,
    cfg: ModelConfig,
}

impl<B: Backend> ActiveSearchModel<B> {
    pub fn new(cfg: ModelConfig, device: &B::Device) -> Self {
        cfg.validate()
            .expect("valid active_search_v3 configuration");
        assert_eq!(
            cfg.architecture,
            Architecture::ActiveSearchV3,
            "ActiveSearchModel requires architecture active_search_v3"
        );
        let a = cfg.active.clone().expect("active geometry");
        let model = Self {
            root: RootPath::new(&cfg, &a, device),
            query: QueryEncoder::new(&cfg, &a, device),
            planner: Planner::new(&cfg, &a, device),
            selector: Selector::new(&a, device),
            readout: RootReadout::new(&a, device),
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
        let mut visitor = Init;
        self.visit(&mut visitor);
    }

    pub fn config(&self) -> &ModelConfig {
        &self.cfg
    }

    pub fn active(&self) -> &ActiveConfig {
        self.cfg.active.as_ref().expect("active geometry")
    }

    /// Execute the root once.
    pub fn root_stage(
        &self,
        board: Tensor<B, 3>,
        cands: &CandidateTensors<B>,
        facts: Tensor<B, 3>,
    ) -> RootStage<B> {
        assert!(
            cands.width > 0,
            "candidate batch contains no legal candidates; terminal-only batches \
             have no policy path and must bypass neural evaluation"
        );
        super::counters::note_root_stage();
        self.root
            .forward(&self.cfg, self.active(), board, cands, facts)
    }

    /// Masked log-probabilities over the root legal candidates.
    pub fn policy(
        &self,
        tokens: Tensor<B, 3>,
        branch: Tensor<B, 3>,
        workspace: Tensor<B, 3>,
        cands: &CandidateTensors<B>,
    ) -> PolicyOutput<B> {
        let [b, w] = cands.mask.dims();
        let logits = self.readout.logits(tokens, branch, workspace);
        // Padding -> -inf; terminal rows -> 0 so the softmax is never all-masked.
        let invalid = cands.mask.clone().bool_not();
        let logits = logits.mask_fill(invalid.clone(), f32::NEG_INFINITY);
        let terminal_row = cands
            .valid
            .clone()
            .bool_not()
            .unsqueeze_dim::<2>(1)
            .expand([b, w]);
        let logits = logits.mask_fill(terminal_row, 0.0);
        let log_probs = activation::log_softmax(logits, 1).mask_fill(invalid, 0.0);
        PolicyOutput {
            log_probs,
            mask: cands.mask.clone(),
            valid: cands.valid.clone(),
            base_all: None,
        }
    }

    /// Budget 0: root encoding, root tokens, zero queries.
    pub fn forward_b0(
        &self,
        board: Tensor<B, 3>,
        cands: &CandidateTensors<B>,
        facts: Tensor<B, 3>,
    ) -> ModelOutput<B> {
        let b = board.dims()[0];
        let stage = self.root_stage(board, cands, facts);
        let ws = self.planner.initial_workspace(b);
        let br = self.planner.initial_branch(stage.tokens.clone());
        let policy = self.policy(stage.tokens, br, ws, cands);
        ModelOutput {
            readouts: vec![Readout {
                policy,
                wdl_logits: stage.wdl_logits,
            }],
            executed_blocks: self.cfg.core_blocks,
        }
    }

    pub fn num_params(&self) -> usize {
        Module::num_params(self)
    }

    /// Exact parameter count by subsystem; sums to [`Self::num_params`].
    pub fn param_breakdown(&self) -> Vec<(&'static str, usize)> {
        let mut v = self.root.param_breakdown();
        v.extend(self.query.param_breakdown());
        v.extend(self.planner.param_breakdown());
        v.extend(self.selector.param_breakdown());
        v.extend(self.readout.param_breakdown());
        v
    }

    /// Parameters that receive no gradient while STOP is masked.
    pub fn stop_head_params(&self) -> usize {
        self.selector
            .param_breakdown()
            .into_iter()
            .filter(|(n, _)| *n == "selector.stop_head")
            .map(|(_, c)| c)
            .sum()
    }
}
