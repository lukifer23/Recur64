//! Neural components of `active_search_v3`.
//!
//! * [`RootPath`]: the V2.5 CF root (board encoder, candidate tokens, one
//!   candidate block). Executed exactly once per decision.
//! * [`QueryEncoder`]: the cheap shared encoder of a queried state.
//! * [`Planner`]: one shared, gated, RMS-normalised update per new exact state.
//! * [`Selector`]: scores frontier edges (plus a STOP logit).
//! * [`RootReadout`]: sparse legal-candidate policy from tokens, branch memory
//!   and workspace.

use burn::module::Param;
use burn::nn::{Initializer, Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::prelude::*;
use burn::tensor::{Distribution, Int, TensorData, activation};

use crate::candidate::{CandidateBlock, POLICY_INIT_STD};
use crate::config::{ActiveConfig, Architecture, CANDIDATE_FACT_FIELDS, ModelConfig};
use crate::model::{Block, CandidateTensors, linear_rows, rel_index_data};

use super::features::{EDGE_FEATS, EVENT_FEATS, STOP_FEATS};

/// Finite stand-in for -inf in masked logits, so a zero target times a masked
/// log-probability is exactly zero rather than NaN.
pub const MASKED_LOGIT: f32 = -1.0e9;

/// `[b, n, d]` rows selected by `idx: [b, f]` -> `[b, f, d]`.
pub fn gather_rows<B: Backend>(src: Tensor<B, 3>, idx: Tensor<B, 2, Int>) -> Tensor<B, 3> {
    let [b, _n, d] = src.dims();
    let f = idx.dims()[1];
    src.gather(1, idx.unsqueeze_dim::<3>(2).expand([b, f, d]))
}

/// `[b, n, d]` row selected by `idx: [b]` -> `[b, d]`.
pub fn gather_one<B: Backend>(src: Tensor<B, 3>, idx: Tensor<B, 1, Int>) -> Tensor<B, 2> {
    let [b, _n, d] = src.dims();
    src.gather(1, idx.reshape([b, 1, 1]).expand([b, 1, d]))
        .reshape([b, d])
}

/// RMS over every element (scalar).
pub fn rms_all<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, 1> {
    (x.clone() * x).mean().sqrt()
}

// ---------------------------------------------------------------------------
// Root path (V2.5 CF geometry)
// ---------------------------------------------------------------------------

/// Root board encoder and candidate-token construction, reimplementing the
/// V2.5 `candidate_v25` geometry under contracts `v25_root_encoder_v1` and
/// `candidate_token_v3_root_v1`. The V2.5 policy scorer is not carried over;
/// the V3 readout replaces it.
#[derive(Module, Debug)]
pub struct RootPath<B: Backend> {
    input_proj: Linear<B>,
    square_emb: Param<Tensor<B, 2>>,
    blocks: Vec<Block<B>>,
    final_norm: RmsNorm<B>,
    from_to_proj: Linear<B>,
    global_proj: Linear<B>,
    promo_emb: Param<Tensor<B, 2>>,
    facts1: Linear<B>,
    facts2: Linear<B>,
    token_norm: RmsNorm<B>,
    cand_blocks: Vec<CandidateBlock<B>>,
    /// Pooled root board (width) -> node embedding (query_dim).
    node_proj: Linear<B>,
    /// Neutral WDL head on the pooled board (zero-initialised, as V2.5).
    wdl: Linear<B>,
}

impl<B: Backend> RootPath<B> {
    pub fn new(cfg: &ModelConfig, a: &ActiveConfig, device: &B::Device) -> Self {
        let (d, cd, eps) = (cfg.width, a.candidate.dim, cfg.rms_eps);
        Self {
            input_proj: LinearConfig::new(cfg.in_features, d)
                .with_bias(true)
                .init(device),
            square_emb: Param::from_tensor(Tensor::random(
                [cfg.squares, d],
                Distribution::Normal(0.0, 0.02),
                device,
            )),
            blocks: (0..cfg.core_blocks)
                .map(|_| Block::new(cfg, device))
                .collect(),
            final_norm: RmsNormConfig::new(d).with_epsilon(eps).init(device),
            from_to_proj: LinearConfig::new(2 * d, cd).with_bias(true).init(device),
            global_proj: LinearConfig::new(d, cd).with_bias(true).init(device),
            promo_emb: Param::from_tensor(Tensor::random(
                [cfg.promo_codes, cd],
                Distribution::Normal(0.0, 0.02),
                device,
            )),
            // Ordinary (non-zero) initialisation: facts are a first-class token input.
            facts1: LinearConfig::new(CANDIDATE_FACT_FIELDS, a.candidate.facts_hidden)
                .with_bias(true)
                .init(device),
            facts2: LinearConfig::new(a.candidate.facts_hidden, cd)
                .with_bias(true)
                .init(device),
            token_norm: RmsNormConfig::new(cd).with_epsilon(eps).init(device),
            cand_blocks: (0..a.candidate.blocks)
                .map(|_| CandidateBlock::new(&a.candidate, eps, device))
                .collect(),
            node_proj: LinearConfig::new(d, a.query_dim)
                .with_bias(true)
                .init(device),
            wdl: LinearConfig::new(d, cfg.wdl_classes)
                .with_bias(true)
                .with_initializer(Initializer::Zeros)
                .init(device),
        }
    }

    /// `[b, 64, in_features] -> [b, 64, width]`.
    pub fn encode_board(&self, cfg: &ModelConfig, board: Tensor<B, 3>) -> Tensor<B, 3> {
        let device = board.device();
        let (h, s) = (cfg.heads, cfg.squares);
        let rel_idx = Tensor::<B, 2, Int>::from_data(
            TensorData::new(rel_index_data(h, s), [h, s * s]),
            &device,
        );
        let p = linear_rows(&self.input_proj, board);
        let [b, s, d] = p.dims();
        let mut x = p + self.square_emb.val().reshape([1, s, d]).expand([b, s, d]);
        for block in &self.blocks {
            x = block.forward(x, rel_idx.clone());
        }
        self.final_norm.forward(x)
    }

    /// Candidate tokens `[b, w, cand_dim]` (before the candidate block).
    fn raw_tokens(
        &self,
        cand_dim: usize,
        y: &Tensor<B, 3>,
        cands: &CandidateTensors<B>,
        facts: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let [b, _s, d] = y.dims();
        let w = cands.width;
        let gather_sq = |idx: &Tensor<B, 2, Int>| {
            y.clone()
                .gather(1, idx.clone().unsqueeze_dim::<3>(2).expand([b, w, d]))
        };
        let from_to = linear_rows(
            &self.from_to_proj,
            Tensor::cat(
                vec![gather_sq(&cands.from_idx), gather_sq(&cands.to_idx)],
                2,
            ),
        );
        let pooled = y.clone().mean_dim(1).squeeze_dim::<2>(1);
        let global = self
            .global_proj
            .forward(pooled)
            .unsqueeze_dim::<3>(1)
            .expand([b, w, cand_dim]);
        let promo = self
            .promo_emb
            .val()
            .select(0, cands.promo_code.clone().reshape([b * w]))
            .reshape([b, w, cand_dim]);
        let facts = linear_rows(
            &self.facts2,
            activation::gelu(linear_rows(&self.facts1, facts)),
        );
        self.token_norm.forward(from_to + global + promo + facts)
    }

    /// Run the root once: pooled root node embedding `[b, query_dim]`, final
    /// candidate tokens `[b, w, cand_dim]` and the neutral WDL logits.
    pub fn forward(
        &self,
        cfg: &ModelConfig,
        a: &ActiveConfig,
        board: Tensor<B, 3>,
        cands: &CandidateTensors<B>,
        facts: Tensor<B, 3>,
    ) -> RootStage<B> {
        let y = self.encode_board(cfg, board);
        let pooled = y.clone().mean_dim(1).squeeze_dim::<2>(1);
        let wdl_logits = self.wdl.forward(pooled.clone());
        let root_node = self.node_proj.forward(pooled);
        let mut tokens = self.raw_tokens(a.candidate.dim, &y, cands, facts);
        let key_pad = cands.mask.clone().bool_not();
        for block in &self.cand_blocks {
            tokens = block.forward(tokens, key_pad.clone());
        }
        RootStage {
            root_node,
            tokens,
            wdl_logits,
        }
    }

    pub fn param_breakdown(&self) -> Vec<(&'static str, usize)> {
        vec![
            ("root.input_projection", self.input_proj.num_params()),
            ("root.square_embeddings", self.square_emb.num_params()),
            ("root.board_blocks", self.blocks.num_params()),
            ("root.final_norm", self.final_norm.num_params()),
            ("root.from_to_projection", self.from_to_proj.num_params()),
            ("root.global_projection", self.global_proj.num_params()),
            ("root.promotion_embedding", self.promo_emb.num_params()),
            (
                "root.facts_encoder",
                self.facts1.num_params() + self.facts2.num_params(),
            ),
            ("root.token_norm", self.token_norm.num_params()),
            ("root.candidate_blocks", self.cand_blocks.num_params()),
            ("root.node_projection", self.node_proj.num_params()),
            ("root.wdl_head", self.wdl.num_params()),
        ]
    }
}

/// Output of the single root execution.
pub struct RootStage<B: Backend> {
    /// `[b, query_dim]`
    pub root_node: Tensor<B, 2>,
    /// `[b, w, cand_dim]`
    pub tokens: Tensor<B, 3>,
    pub wdl_logits: Tensor<B, 2>,
}

// ---------------------------------------------------------------------------
// Query-state encoder
// ---------------------------------------------------------------------------

/// Shared lightweight encoder of one queried state (`query_state_encoder_v1`).
/// Parameters are independent of the budget and shared by every node and step.
#[derive(Module, Debug)]
pub struct QueryEncoder<B: Backend> {
    input_proj: Linear<B>,
    square_emb: Param<Tensor<B, 2>>,
    blocks: Vec<Block<B>>,
    final_norm: RmsNorm<B>,
    act_proj: Linear<B>,
    act_promo: Param<Tensor<B, 2>>,
    act_norm: RmsNorm<B>,
    heads: usize,
    squares: usize,
}

impl<B: Backend> QueryEncoder<B> {
    pub fn new(cfg: &ModelConfig, a: &ActiveConfig, device: &B::Device) -> Self {
        let qd = a.query_dim;
        let qcfg = ModelConfig {
            width: qd,
            heads: a.query_heads,
            ffn: a.query_ffn,
            input_blocks: 0,
            core_blocks: a.query_blocks,
            output_blocks: 0,
            squares: cfg.squares,
            in_features: cfg.in_features,
            policy_dim: cfg.policy_dim,
            wdl_classes: cfg.wdl_classes,
            promo_codes: cfg.promo_codes,
            rms_eps: cfg.rms_eps,
            architecture: Architecture::ProbeV1,
            candidate: None,
            legacy_facts: None,
            active: None,
            all_info: None,
            evidence: None,
        };
        Self {
            input_proj: LinearConfig::new(cfg.in_features, qd)
                .with_bias(true)
                .init(device),
            square_emb: Param::from_tensor(Tensor::random(
                [cfg.squares, qd],
                Distribution::Normal(0.0, 0.02),
                device,
            )),
            blocks: (0..a.query_blocks)
                .map(|_| Block::new(&qcfg, device))
                .collect(),
            final_norm: RmsNormConfig::new(qd)
                .with_epsilon(cfg.rms_eps)
                .init(device),
            act_proj: LinearConfig::new(2 * qd, qd).with_bias(true).init(device),
            act_promo: Param::from_tensor(Tensor::random(
                [cfg.promo_codes, qd],
                Distribution::Normal(0.0, 0.02),
                device,
            )),
            act_norm: RmsNormConfig::new(qd)
                .with_epsilon(cfg.rms_eps)
                .init(device),
            heads: a.query_heads,
            squares: cfg.squares,
        }
    }

    /// `[b, 64, in_features] -> (square features [b, 64, qd], pooled [b, qd])`.
    pub fn forward(&self, obs: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 2>) {
        super::counters::note_query_encoder();
        let device = obs.device();
        let (h, s) = (self.heads, self.squares);
        let rel_idx = Tensor::<B, 2, Int>::from_data(
            TensorData::new(rel_index_data(h, s), [h, s * s]),
            &device,
        );
        let p = linear_rows(&self.input_proj, obs);
        let [b, s, d] = p.dims();
        let mut x = p + self.square_emb.val().reshape([1, s, d]).expand([b, s, d]);
        for block in &self.blocks {
            x = block.forward(x, rel_idx.clone());
        }
        let x = self.final_norm.forward(x);
        let pooled = x.clone().mean_dim(1).squeeze_dim::<2>(1);
        (x, pooled)
    }

    /// Raw legal-action embeddings `[b, a, qd]` built from the node's own square
    /// features only (no successor information).
    pub fn action_embeddings(
        &self,
        squares: &Tensor<B, 3>,
        from_idx: Tensor<B, 2, Int>,
        to_idx: Tensor<B, 2, Int>,
        promo_code: Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        let [b, w] = from_idx.dims();
        let qd = squares.dims()[2];
        let h_from = gather_rows(squares.clone(), from_idx);
        let h_to = gather_rows(squares.clone(), to_idx);
        let proj = linear_rows(&self.act_proj, Tensor::cat(vec![h_from, h_to], 2));
        let promo = self
            .act_promo
            .val()
            .select(0, promo_code.reshape([b * w]))
            .reshape([b, w, qd]);
        self.act_norm.forward(proj + promo)
    }

    pub fn param_breakdown(&self) -> Vec<(&'static str, usize)> {
        vec![
            ("query.input_projection", self.input_proj.num_params()),
            ("query.square_embeddings", self.square_emb.num_params()),
            ("query.blocks", self.blocks.num_params()),
            ("query.final_norm", self.final_norm.num_params()),
            (
                "query.action_head",
                self.act_proj.num_params()
                    + self.act_promo.num_params()
                    + self.act_norm.num_params(),
            ),
        ]
    }
}

// ---------------------------------------------------------------------------
// Planner
// ---------------------------------------------------------------------------

/// Gate activations of one planner update (for diagnostics).
pub struct PlanGates<B: Backend> {
    /// `[b, K, d]`
    pub workspace: Tensor<B, 3>,
    /// `[b, d]`
    pub branch: Tensor<B, 2>,
}

/// Result of one planner update.
pub struct PlanOut<B: Backend> {
    /// `[b, K, d]`
    pub workspace: Tensor<B, 3>,
    /// `[b, R, d]`
    pub branch: Tensor<B, 3>,
    pub gates: PlanGates<B>,
}

/// One shared gated update per new exact state (`active_planner_v1`).
///
/// `S' = RMSNorm((1 - u) * S + u * proposal)` for both the workspace and the
/// selected branch token, so no state accumulates an unbounded residual.
#[derive(Module, Debug)]
pub struct Planner<B: Backend> {
    ws_init: Param<Tensor<B, 2>>,
    branch_init: Linear<B>,
    branch_init_norm: RmsNorm<B>,
    event_proj: Linear<B>,
    event_norm: RmsNorm<B>,
    ws_norm: RmsNorm<B>,
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    out_proj: Linear<B>,
    ffn_norm: RmsNorm<B>,
    ffn1: Linear<B>,
    ffn2: Linear<B>,
    gate: Linear<B>,
    out_norm: RmsNorm<B>,
    b_in: Linear<B>,
    b_out: Linear<B>,
    b_gate: Linear<B>,
    b_norm: RmsNorm<B>,
    heads: usize,
    head_dim: usize,
}

impl<B: Backend> Planner<B> {
    pub fn new(cfg: &ModelConfig, a: &ActiveConfig, device: &B::Device) -> Self {
        let (d, eps) = (a.query_dim, cfg.rms_eps);
        let rms = |n: usize| RmsNormConfig::new(n).with_epsilon(eps).init(device);
        let lin = |i: usize, o: usize| LinearConfig::new(i, o).with_bias(true).init(device);
        // A key bias adds one constant to every key of a query and cancels in the
        // softmax (an inert parameter with identically zero gradient), so the key
        // projection has none.
        let lin_nb = |i: usize, o: usize| LinearConfig::new(i, o).with_bias(false).init(device);
        Self {
            ws_init: Param::from_tensor(Tensor::random(
                [a.workspace_tokens, d],
                Distribution::Normal(0.0, 1.0),
                device,
            )),
            branch_init: lin(d, d),
            branch_init_norm: rms(d),
            // edge, parent, child, root token, branch memory + event features
            event_proj: lin(5 * d + EVENT_FEATS, d),
            event_norm: rms(d),
            ws_norm: rms(d),
            q_proj: lin(d, d),
            k_proj: lin_nb(d, d),
            v_proj: lin(d, d),
            out_proj: lin(d, d),
            ffn_norm: rms(d),
            ffn1: lin(d, a.planner_ffn),
            ffn2: lin(a.planner_ffn, d),
            gate: lin(2 * d, d),
            out_norm: rms(d),
            b_in: lin(3 * d, d),
            b_out: lin(d, d),
            b_gate: lin(2 * d, d),
            b_norm: rms(d),
            heads: a.planner_heads,
            head_dim: d / a.planner_heads,
        }
    }

    /// Initial workspace `[b, K, d]` (learned tokens, identical for every example).
    pub fn initial_workspace(&self, b: usize) -> Tensor<B, 3> {
        let [k, d] = self.ws_init.val().dims();
        self.ws_init.val().reshape([1, k, d]).expand([b, k, d])
    }

    /// Initial branch memory `[b, R, d]` from the root candidate tokens.
    pub fn initial_branch(&self, tokens: Tensor<B, 3>) -> Tensor<B, 3> {
        self.branch_init_norm
            .forward(linear_rows(&self.branch_init, tokens))
    }

    /// One update.
    ///
    /// * `workspace` `[b, K, d]`, `branch` `[b, R, d]`
    /// * `parts`: the five `[b, d]` event inputs (edge, parent node, child node,
    ///   root candidate token, selected branch memory) and `[b, EVENT_FEATS]`
    /// * `branch_onehot` `[b, R, 1]`: the selected branch (zero rows for examples
    ///   that are not updated)
    ///
    /// Every row is a real update: the caller passes only the examples that
    /// received a new exact state (compacted), so rows executed equal queries.
    pub fn update(
        &self,
        workspace: Tensor<B, 3>,
        branch: Tensor<B, 3>,
        event_in: Tensor<B, 2>,
        branch_onehot: Tensor<B, 3>,
    ) -> PlanOut<B> {
        super::counters::note_planner_update();
        let [b, k, d] = workspace.dims();
        let (h, hd) = (self.heads, self.head_dim);

        // Event token.
        let ev = self.event_norm.forward(self.event_proj.forward(event_in));

        // Workspace: tokens attend over [event; tokens].
        let n = self.ws_norm.forward(workspace.clone());
        let kv = Tensor::cat(vec![ev.clone().unsqueeze_dim::<3>(1), n.clone()], 1);
        let split = |t: Tensor<B, 3>, len: usize| t.reshape([b, len, h, hd]).swap_dims(1, 2);
        let q = split(linear_rows(&self.q_proj, n.clone()), k);
        let kk = split(linear_rows(&self.k_proj, kv.clone()), k + 1);
        let v = split(linear_rows(&self.v_proj, kv), k + 1);
        let logits = q
            .matmul(kk.swap_dims(2, 3))
            .mul_scalar(1.0 / (hd as f32).sqrt());
        let attn = activation::softmax(logits, 3).matmul(v);
        let attn = attn.swap_dims(1, 2).reshape([b, k, d]);
        let hdn = n + linear_rows(&self.out_proj, attn);
        let f = linear_rows(
            &self.ffn2,
            activation::gelu(linear_rows(&self.ffn1, self.ffn_norm.forward(hdn.clone()))),
        );
        let proposal = hdn + f;
        let u = activation::sigmoid(linear_rows(
            &self.gate,
            Tensor::cat(vec![workspace.clone(), proposal.clone()], 2),
        ));
        let ws_new = self
            .out_norm
            .forward(workspace.clone() * (u.clone().neg() + 1.0) + u.clone() * proposal);

        // Branch memory of the selected root candidate.
        let ctx = ws_new.clone().mean_dim(1).squeeze_dim::<2>(1);
        let br_sel = (branch.clone() * branch_onehot.clone())
            .sum_dim(1)
            .squeeze_dim::<2>(1);
        let bp = self.b_out.forward(activation::gelu(
            self.b_in
                .forward(Tensor::cat(vec![br_sel.clone(), ev, ctx], 1)),
        ));
        let ub = activation::sigmoid(
            self.b_gate
                .forward(Tensor::cat(vec![br_sel.clone(), bp.clone()], 1)),
        );
        let row = self
            .b_norm
            .forward(br_sel * (ub.clone().neg() + 1.0) + ub.clone() * bp);
        let m = branch_onehot;
        let br_out = branch.clone() * (m.clone().neg() + 1.0) + row.unsqueeze_dim::<3>(1) * m;
        PlanOut {
            workspace: ws_new,
            branch: br_out,
            gates: PlanGates {
                workspace: u,
                branch: ub,
            },
        }
    }

    pub fn param_breakdown(&self) -> Vec<(&'static str, usize)> {
        vec![
            (
                "planner.initial_memory",
                self.ws_init.num_params()
                    + self.branch_init.num_params()
                    + self.branch_init_norm.num_params(),
            ),
            (
                "planner.event",
                self.event_proj.num_params() + self.event_norm.num_params(),
            ),
            (
                "planner.workspace_update",
                self.ws_norm.num_params()
                    + self.q_proj.num_params()
                    + self.k_proj.num_params()
                    + self.v_proj.num_params()
                    + self.out_proj.num_params()
                    + self.ffn_norm.num_params()
                    + self.ffn1.num_params()
                    + self.ffn2.num_params()
                    + self.gate.num_params()
                    + self.out_norm.num_params(),
            ),
            (
                "planner.branch_update",
                self.b_in.num_params()
                    + self.b_out.num_params()
                    + self.b_gate.num_params()
                    + self.b_norm.num_params(),
            ),
        ]
    }
}

// ---------------------------------------------------------------------------
// Selector
// ---------------------------------------------------------------------------

/// Scores frontier edges and carries a STOP logit (`active_selector_v1`).
///
/// STOP is masked in the primary experiment, so its parameters receive exactly
/// zero gradient until `adaptive_stop_v1`; they are reported separately.
#[derive(Module, Debug)]
pub struct Selector<B: Backend> {
    hidden: Linear<B>,
    out: Linear<B>,
    stop_hidden: Linear<B>,
    stop_out: Linear<B>,
}

impl<B: Backend> Selector<B> {
    pub fn new(a: &ActiveConfig, device: &B::Device) -> Self {
        let (d, h) = (a.query_dim, a.selector_hidden);
        let small = |i: usize, o: usize| {
            LinearConfig::new(i, o)
                .with_bias(true)
                .with_initializer(Initializer::Normal {
                    mean: 0.0,
                    std: POLICY_INIT_STD,
                })
                .init(device)
        };
        Self {
            // edge, parent, root token, branch memory, workspace mean + edge features
            hidden: LinearConfig::new(5 * d + EDGE_FEATS, h)
                .with_bias(true)
                .init(device),
            out: small(h, 1),
            stop_hidden: LinearConfig::new(d + STOP_FEATS, h)
                .with_bias(true)
                .init(device),
            stop_out: small(h, 1),
        }
    }

    /// Edge scores `[b, f]` from the concatenated `[b, f, 5d + EDGE_FEATS]` input.
    pub fn edge_logits(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        let h = activation::gelu(linear_rows(&self.hidden, input));
        linear_rows(&self.out, h).squeeze_dim::<2>(2)
    }

    /// STOP logit `[b, 1]` from `[b, d + STOP_FEATS]`.
    pub fn stop_logit(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        self.stop_out
            .forward(activation::gelu(self.stop_hidden.forward(input)))
    }

    pub fn param_breakdown(&self) -> Vec<(&'static str, usize)> {
        vec![
            (
                "selector.edge_scorer",
                self.hidden.num_params() + self.out.num_params(),
            ),
            (
                "selector.stop_head",
                self.stop_hidden.num_params() + self.stop_out.num_params(),
            ),
        ]
    }
}

// ---------------------------------------------------------------------------
// Root policy readout
// ---------------------------------------------------------------------------

/// Sparse root-candidate policy from tokens, branch memory and workspace
/// (`root_policy_v3_v1`). The same weights at every budget.
#[derive(Module, Debug)]
pub struct RootReadout<B: Backend> {
    ctx_proj: Linear<B>,
    hidden: Linear<B>,
    out: Linear<B>,
}

impl<B: Backend> RootReadout<B> {
    pub fn new(a: &ActiveConfig, device: &B::Device) -> Self {
        let d = a.query_dim;
        Self {
            ctx_proj: LinearConfig::new(d, d).with_bias(true).init(device),
            hidden: LinearConfig::new(3 * d, a.readout_hidden)
                .with_bias(true)
                .init(device),
            out: LinearConfig::new(a.readout_hidden, 1)
                .with_bias(true)
                .with_initializer(Initializer::Normal {
                    mean: 0.0,
                    std: POLICY_INIT_STD,
                })
                .init(device),
        }
    }

    /// Raw candidate logits `[b, w]` (padding not yet masked).
    pub fn logits(
        &self,
        tokens: Tensor<B, 3>,
        branch: Tensor<B, 3>,
        workspace: Tensor<B, 3>,
    ) -> Tensor<B, 2> {
        let [b, w, d] = tokens.dims();
        let ctx = self
            .ctx_proj
            .forward(workspace.mean_dim(1).squeeze_dim::<2>(1))
            .unsqueeze_dim::<3>(1)
            .expand([b, w, d]);
        let x = Tensor::cat(vec![tokens, branch, ctx], 2);
        let h = activation::gelu(linear_rows(&self.hidden, x));
        linear_rows(&self.out, h).squeeze_dim::<2>(2)
    }

    pub fn param_breakdown(&self) -> Vec<(&'static str, usize)> {
        vec![(
            "readout",
            self.ctx_proj.num_params() + self.hidden.num_params() + self.out.num_params(),
        )]
    }
}
