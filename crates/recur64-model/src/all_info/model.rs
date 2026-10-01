//! `all_info_v1`: the P6 information-sufficiency control.
//!
//! A separately trained model that receives the exhaustive raw depth-2 tree of a root at
//! once. It shares the V2.5 root encoder, the root candidate tokens and the
//! `query_state_encoder_v1` architecture with `active_search_v3` (its weights are its
//! own), and replaces the selector and the sequential planner by a permutation-invariant
//! set integrator:
//!
//! 1. the root is encoded exactly once (root candidate tokens, root CandidateFacts only);
//! 2. every depth-1 and depth-2 state is encoded by ONE shared `QueryEncoder` call;
//! 3. each state becomes a token that carries only structure (own / parent / incoming
//!    edge embedding, root candidate token of its branch, depth, terminal, in-check);
//! 4. per root branch, a masked set-attention block mixes the branch's tokens and an
//!    attention pool (query from the root candidate token) summarises them;
//! 5. a masked set-attention block mixes the branch summaries, and the V3 root readout
//!    scores every legal root candidate.
//!
//! No positional encoding of reply order exists anywhere, so serialisation order cannot
//! carry an answer signal. Nothing proof-derived is an input.

use burn::module::{Module, ModuleVisitor, Param};
use burn::nn::{Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::prelude::*;
use burn::tensor::{Distribution, Int, TensorData, activation};

use recur64_core::ActionId;

use crate::active::modules::{QueryEncoder, RootPath, RootReadout};
use crate::candidate::{CandidateBlock, CandidateInputs};
use crate::config::{AllInfoConfig, Architecture, CandidateConfig, ModelConfig};
use crate::model::{CandidateTensors, ModelOutput, PolicyOutput, Readout, linear_rows};

use super::batch::TreeBatch;
use super::tree::AllInfoTree;

const MASKED_KEY: f32 = -1.0e9;

/// Set integrator over the tokens of each root branch and over the branches.
#[derive(Module, Debug)]
pub struct TreeIntegrator<B: Backend> {
    /// `[own, parent, incoming edge, root candidate token, terminal, in_check] -> d`.
    state_proj: Linear<B>,
    /// Depth embedding: row 0 = depth-1 state, row 1 = depth-2 reply.
    depth_emb: Param<Tensor<B, 2>>,
    token_norm: RmsNorm<B>,
    state_block: CandidateBlock<B>,
    pool_q: Linear<B>,
    pool_k: Linear<B>,
    pool_v: Linear<B>,
    pool_out: Linear<B>,
    pool_norm: RmsNorm<B>,
    global_block: CandidateBlock<B>,
    heads: usize,
    head_dim: usize,
}

impl<B: Backend> TreeIntegrator<B> {
    pub fn new(a: &AllInfoConfig, eps: f64, device: &B::Device) -> Self {
        let d = a.query_dim;
        let block_cfg = CandidateConfig {
            dim: d,
            heads: a.set_heads,
            ffn: a.set_ffn,
            blocks: 1,
            ..a.candidate.clone()
        };
        Self {
            state_proj: LinearConfig::new(4 * d + 2, d).with_bias(true).init(device),
            depth_emb: Param::from_tensor(Tensor::random(
                [2, d],
                Distribution::Normal(0.0, 0.02),
                device,
            )),
            token_norm: RmsNormConfig::new(d).with_epsilon(eps).init(device),
            state_block: CandidateBlock::new(&block_cfg, eps, device),
            pool_q: LinearConfig::new(d, d).with_bias(true).init(device),
            // No key bias: a constant added to every key cancels in the softmax.
            pool_k: LinearConfig::new(d, d).with_bias(false).init(device),
            pool_v: LinearConfig::new(d, d).with_bias(true).init(device),
            pool_out: LinearConfig::new(d, d).with_bias(true).init(device),
            pool_norm: RmsNormConfig::new(d).with_epsilon(eps).init(device),
            global_block: CandidateBlock::new(&block_cfg, eps, device),
            heads: a.set_heads,
            head_dim: d / a.set_heads,
        }
    }

    /// `tokens_in: [b*w*t, 4d+2]`, `depth: [b*w*t]`, `root_tok: [b*w, d]`,
    /// `slot_pad: [b*w, t]` (true = no state), `branch_pad: [b, w]` (true = padding).
    /// Returns branch summaries `[b, w, d]` after cross-branch mixing.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        tokens_in: Tensor<B, 2>,
        depth: Tensor<B, 1, Int>,
        root_tok: Tensor<B, 2>,
        slot_pad: Tensor<B, 2, Bool>,
        branch_pad: Tensor<B, 2, Bool>,
        b: usize,
        w: usize,
        t: usize,
    ) -> Tensor<B, 3> {
        let d = root_tok.dims()[1];
        let (h, hd) = (self.heads, self.head_dim);
        let x = self.state_proj.forward(tokens_in) + self.depth_emb.val().select(0, depth);
        let x = self.token_norm.forward(x).reshape([b * w, t, d]);
        let x = self.state_block.forward(x, slot_pad.clone());

        // Attention pooling: the query comes from the branch's root candidate token.
        let q = self
            .pool_q
            .forward(root_tok.clone())
            .reshape([b * w, h, 1, hd]);
        let split = |m: Tensor<B, 3>| m.reshape([b * w, t, h, hd]).swap_dims(1, 2);
        let k = split(linear_rows(&self.pool_k, x.clone()));
        let v = split(linear_rows(&self.pool_v, x));
        let logits = q
            .matmul(k.swap_dims(2, 3))
            .mul_scalar(1.0 / (hd as f32).sqrt());
        let pad = slot_pad
            .unsqueeze_dim::<3>(1)
            .unsqueeze_dim::<4>(1)
            .expand([b * w, h, 1, t]);
        let attn = activation::softmax(logits.mask_fill(pad, MASKED_KEY), 3).matmul(v);
        let pooled = self.pool_out.forward(attn.reshape([b * w, d]));
        let branch = self.pool_norm.forward(pooled + root_tok).reshape([b, w, d]);
        self.global_block.forward(branch, branch_pad)
    }

    pub fn param_breakdown(&self) -> Vec<(&'static str, usize)> {
        vec![
            (
                "integrator.state_projection",
                self.state_proj.num_params()
                    + self.depth_emb.num_params()
                    + self.token_norm.num_params(),
            ),
            ("integrator.state_block", self.state_block.num_params()),
            (
                "integrator.branch_pool",
                self.pool_q.num_params()
                    + self.pool_k.num_params()
                    + self.pool_v.num_params()
                    + self.pool_out.num_params()
                    + self.pool_norm.num_params(),
            ),
            ("integrator.global_block", self.global_block.num_params()),
        ]
    }
}

/// Accounting of one ALL-INFO forward.
#[derive(Debug, Clone)]
pub struct AllInfoAccounting {
    /// `(depth-1 states, depth-2 states)` per example.
    pub states: Vec<(usize, usize)>,
    pub root_encodes: usize,
    pub query_encoder_calls: usize,
    pub query_encoder_states: usize,
}

pub struct AllInfoOutput<B: Backend> {
    pub readout: Readout<B>,
    pub accounting: AllInfoAccounting,
}

/// The P6 ALL-INFO network.
#[derive(Module, Debug)]
pub struct AllInfoModel<B: Backend> {
    pub(crate) root: RootPath<B>,
    pub(crate) query: QueryEncoder<B>,
    pub(crate) integrator: TreeIntegrator<B>,
    pub(crate) readout: RootReadout<B>,
    cfg: ModelConfig,
}

impl<B: Backend> AllInfoModel<B> {
    pub fn new(cfg: ModelConfig, device: &B::Device) -> Self {
        cfg.validate().expect("valid all_info_v1 configuration");
        assert_eq!(
            cfg.architecture,
            Architecture::AllInfoV1,
            "AllInfoModel requires architecture all_info_v1"
        );
        let a = cfg.all_info.clone().expect("all_info geometry");
        let shared = a.shared_geometry();
        let model = Self {
            root: RootPath::new(&cfg, &shared, device),
            query: QueryEncoder::new(&cfg, &shared, device),
            integrator: TreeIntegrator::new(&a, cfg.rms_eps, device),
            readout: RootReadout::new(&shared, device),
            cfg,
        };
        model.force_init();
        model
    }

    /// Materialise every lazily initialised parameter.
    pub fn force_init(&self) {
        struct Force;
        impl<B: Backend> ModuleVisitor<B> for Force {
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
        self.visit(&mut Force);
    }

    pub fn config(&self) -> &ModelConfig {
        &self.cfg
    }

    /// The shared root path (identity checks against `active_search_v3`).
    pub fn root_path(&self) -> &RootPath<B> {
        &self.root
    }

    /// The shared query-state encoder (identity checks against `active_search_v3`).
    pub fn query_encoder(&self) -> &QueryEncoder<B> {
        &self.query
    }

    fn all_info(&self) -> &AllInfoConfig {
        self.cfg.all_info.as_ref().expect("all_info geometry")
    }

    pub fn num_params(&self) -> usize {
        Module::num_params(self)
    }

    /// Exact parameter count by subsystem; sums to [`Self::num_params`].
    pub fn param_breakdown(&self) -> Vec<(&'static str, usize)> {
        let mut v = self.root.param_breakdown();
        v.extend(self.query.param_breakdown());
        v.extend(self.integrator.param_breakdown());
        v.extend(self.readout.param_breakdown());
        v
    }

    /// Masked log-probabilities over the root legal candidates (same masking rules as the
    /// V3 readout: padding exactly 0, terminal rows neutral).
    fn policy(
        &self,
        tokens: Tensor<B, 3>,
        branch: Tensor<B, 3>,
        workspace: Tensor<B, 3>,
        cands: &CandidateTensors<B>,
    ) -> PolicyOutput<B> {
        let [b, w] = cands.mask.dims();
        let logits = self.readout.logits(tokens, branch, workspace);
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

    /// One forward over a batch of roots with their exhaustive depth-2 trees. `inputs` must
    /// be built from the same roots (root observation, legal candidates, root facts); the
    /// tree of example `i` must belong to root `i` (checked against the candidate list).
    pub fn forward_trees(
        &self,
        inputs: &CandidateInputs<B>,
        trees: &[AllInfoTree],
        device: &B::Device,
    ) -> anyhow::Result<AllInfoOutput<B>> {
        let a = self.all_info();
        let shared = a.shared_geometry();
        let d = a.query_dim;
        let b = trees.len();
        let w = inputs.cands.width;
        anyhow::ensure!(
            inputs.board.dims()[0] == b,
            "{} trees for a batch of {}",
            b,
            inputs.board.dims()[0]
        );
        let tb = TreeBatch::build(trees, w)?;
        let (n1, n2) = (tb.n1, tb.n2);
        let n = n1 + n2;
        anyhow::ensure!(n1 > 0, "a batch without any root move");

        // 1. root, exactly once.
        let stage = self.root.forward(
            &self.cfg,
            &shared,
            inputs.board.clone(),
            &inputs.cands,
            inputs.facts.clone(),
        );

        // 2. every future state through ONE shared query-encoder call.
        let obs = Tensor::<B, 3>::from_data(TensorData::new(tb.obs.clone(), [n, 64, 119]), device);
        let (sq, pooled) = self.query.forward(obs);
        let sq1 = sq.slice([0..n1, 0..64, 0..d]);
        let aw = tb.action_width;
        let idx2 = |v: &[i32]| {
            Tensor::<B, 2, Int>::from_data(TensorData::new(v.to_vec(), [n1, aw]), device)
        };
        let act = self
            .query
            .action_embeddings(&sq1, idx2(&tb.from), idx2(&tb.to), idx2(&tb.promo));

        // 3. tables and per-slot gathers.
        let zero = Tensor::<B, 2>::zeros([1, d], device);
        let node_table = Tensor::cat(vec![stage.root_node.clone(), pooled, zero.clone()], 0);
        let root_rows = stage.tokens.clone().reshape([b * w, d]);
        let edge_table = Tensor::cat(vec![root_rows.clone(), act.reshape([n1 * aw, d]), zero], 0);
        let slots = tb.slots;
        let p = b * w * slots;
        let ix =
            |v: &[i32]| Tensor::<B, 1, Int>::from_data(TensorData::new(v.to_vec(), [p]), device);
        let own = node_table.clone().select(0, ix(&tb.own));
        let parent = node_table.select(0, ix(&tb.parent));
        let edge = edge_table.clone().select(0, ix(&tb.edge));
        let rtok = edge_table.select(0, ix(&tb.root_token));
        let flags = Tensor::<B, 2>::from_data(TensorData::new(tb.flags.clone(), [p, 2]), device);
        let tokens_in = Tensor::cat(vec![own, parent, edge, rtok, flags], 1);
        let slot_pad = Tensor::<B, 2, Bool>::from_data(
            TensorData::new(tb.slot_pad.clone(), [b * w, slots]),
            device,
        );
        let branch_pad = inputs.cands.mask.clone().bool_not();

        // 4-5. set integration and readout.
        let branches = self.integrator.forward(
            tokens_in,
            ix(&tb.depth),
            root_rows,
            slot_pad,
            branch_pad,
            b,
            w,
            slots,
        );
        // Workspace analogue: the mean of the valid branch summaries (no parameters).
        let valid = inputs.cands.mask.clone().float().unsqueeze_dim::<3>(2);
        let count = valid.clone().sum_dim(1).clamp_min(1.0);
        let workspace = (branches.clone() * valid).sum_dim(1) / count;
        let policy = self.policy(stage.tokens, branches, workspace, &inputs.cands);
        Ok(AllInfoOutput {
            readout: Readout {
                policy,
                wdl_logits: stage.wdl_logits,
            },
            accounting: AllInfoAccounting {
                states: tb.counts,
                root_encodes: 1,
                query_encoder_calls: 1,
                query_encoder_states: n,
            },
        })
    }

    /// Verify a tree belongs to a root by comparing its branch actions with the root's
    /// candidate list (ascending `ActionId`).
    pub fn check_tree_matches_candidates(
        tree: &AllInfoTree,
        legal: &[ActionId],
    ) -> anyhow::Result<()> {
        let acts: Vec<u16> = tree.root.legal_actions.clone();
        anyhow::ensure!(
            acts.len() == legal.len()
                && acts
                    .iter()
                    .zip(legal)
                    .all(|(a, l)| u32::from(*a) == l.index()),
            "the ALL-INFO tree's root moves differ from the root candidate list"
        );
        Ok(())
    }

    /// Placeholder-free guard used by `NeuralModel::forward_inputs`.
    pub fn no_tree_forward(&self) -> ! {
        panic!(
            "all_info_v1 consumes the exhaustive depth-2 tree; a tree-less forward has no \
             defined meaning and is refused (use AllInfoModel::forward_trees)"
        )
    }

    /// Unused-output helper for the generic `ModelOutput` shape.
    pub fn as_model_output(readout: Readout<B>) -> ModelOutput<B> {
        ModelOutput {
            readouts: vec![readout],
            executed_blocks: 0,
        }
    }
}
