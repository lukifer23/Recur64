//! Chimera V2: one-shot board encoder, candidate tokens, exact-world-model tool
//! tokens and a bounded recurrent planner. See `docs/HP_X2_ARCHITECTURE.md`.
//!
//! ```text
//! B  = BoardEncoderV2(board [, visual])                 # ONE pass, immutable afterwards
//! M  = CandidateTokens(B, legal moves, candidate facts)
//! S  = SuccessorEncoder(world.successors)               # visible from thought 2 (progressive)
//! R  = ReplySetEncoder(world.reply_sets)                # visible from thought 3 (progressive)
//! Z0 = workspace slots + proj(mean B)
//! for t in 1..=T:  Z = PlannerStep_t(Z ; B, M, S_t, R_t)   # shared, gated, RMS-normalised
//! policy = MLP(M, S, R, ctx(Z))       wdl = Linear(mean B, mean Z)
//! ```

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use burn::module::{Module, Param};
use burn::nn::{Initializer, Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::prelude::*;
use burn::tensor::{Distribution, Int, TensorData, activation};

use crate::chimera::VisualEncoder;
use crate::config::ModelConfig;
use crate::experimental::{
    CANDIDATE_FACT_FIELDS, ExperimentalConfig, InfoSchedule, VisualProviderKind,
};
use crate::model::{Block, CandidateTensors, PolicyOutput, Readout, linear_rows, rel_index_data};

const SQUARES: usize = 64;

/// One-hot codes per square in a packed successor board (0 empty, 1-6 mover, 7-12 opponent).
pub const BOARD_CODES: usize = 13;
/// Exact global flags per successor: terminal one-hot (5), in-check, reply count,
/// castling nibble (4), en-passant present, halfmove clock.
pub const SUCC_FLAG_DIM: usize = 13;
/// Features per reply token: 8 facts, terminal one-hot (5), check, 7 next-player summary.
pub const REPLY_FEAT_DIM: usize = 21;

/// The world-model information as tensors (already unpacked and normalised).
pub struct WorldTensors<B: Backend> {
    /// `[b, W, 64, BOARD_CODES]` one-hot successor placement in the child's canonical frame.
    pub succ_board: Tensor<B, 4>,
    /// `[b, W, SUCC_FLAG_DIM]`.
    pub succ_flags: Tensor<B, 3>,
    /// `[b, W, R, REPLY_FEAT_DIM]`.
    pub reply_feats: Tensor<B, 4>,
    /// `[b, W, R]` 1.0 for a real reply, 0.0 for padding.
    pub reply_mask: Tensor<B, 3>,
}

/// Everything a V2 forward pass consumes.
pub struct ChimeraV2Input<B: Backend> {
    /// Canonical Observation V1 `[b, 64, 119]`.
    pub board: Tensor<B, 3>,
    /// Canonical board image `[b, 3, 64, 64]` in `0..1` (only when the visual path is on).
    pub visual: Option<Tensor<B, 4>>,
    /// `CandidateFactsV1` `[b, W, CANDIDATE_FACT_FIELDS]`, padded candidates zero.
    pub cand_facts: Tensor<B, 3>,
    /// The world model (absent for the root-only control).
    pub world: Option<WorldTensors<B>>,
}

/// Forward options. They change what is observed, never the function computed.
#[derive(Clone, Copy, Debug, Default)]
pub struct V2Options {
    /// Read out after every thought (measurement mode). Otherwise only the final thought.
    pub all_readouts: bool,
}

/// Per-thought planner diagnostics, each `[b]` (gate and share fields averaged).
pub struct PlannerDiag<B: Backend> {
    pub thought: usize,
    /// Mean |Z|.
    pub mean_abs: Tensor<B, 1>,
    /// RMS(Z).
    pub rms: Tensor<B, 1>,
    /// Mean |Z_t - Z_{t-1}| (against the pre-step state).
    pub delta: Tensor<B, 1>,
    pub gate_mean: Tensor<B, 1>,
    pub gate_std: Tensor<B, 1>,
    /// Contribution shares (mean |delta| fractions) of board / candidates / successor / reply.
    pub share_board: Tensor<B, 1>,
    pub share_cand: Tensor<B, 1>,
    pub share_succ: Tensor<B, 1>,
    pub share_reply: Tensor<B, 1>,
}

/// What a forward pass executed (compute accounting).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ComputeCounts {
    pub board_encoder_runs: usize,
    pub planner_steps: usize,
    /// Successor tokens encoded (candidates x batch), 0 when not needed.
    pub successor_encodes: usize,
    /// Reply-set summaries encoded.
    pub reply_set_encodes: usize,
}

pub struct ChimeraV2Output<B: Backend> {
    /// The final readout, or one per thought in `all_readouts` mode.
    pub readouts: Vec<Readout<B>>,
    pub diag: Vec<PlannerDiag<B>>,
    pub counts: ComputeCounts,
}

/// Masked multi-head attention with independent query / source / output widths.
#[derive(Module, Debug)]
struct MAttn<B: Backend> {
    norm_q: RmsNorm<B>,
    norm_src: RmsNorm<B>,
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    out_proj: Linear<B>,
    heads: usize,
    head_dim: usize,
}

impl<B: Backend> MAttn<B> {
    fn new(
        dq: usize,
        dsrc: usize,
        dattn: usize,
        dout: usize,
        heads: usize,
        eps: f64,
        device: &B::Device,
    ) -> Self {
        assert!(
            dattn.is_multiple_of(heads),
            "attention width {dattn} not divisible by {heads}"
        );
        Self {
            norm_q: RmsNormConfig::new(dq).with_epsilon(eps).init(device),
            norm_src: RmsNormConfig::new(dsrc).with_epsilon(eps).init(device),
            q_proj: LinearConfig::new(dq, dattn).with_bias(true).init(device),
            k_proj: LinearConfig::new(dsrc, dattn).with_bias(true).init(device),
            v_proj: LinearConfig::new(dsrc, dattn).with_bias(true).init(device),
            out_proj: LinearConfig::new(dattn, dout).with_bias(true).init(device),
            heads,
            head_dim: dattn / heads,
        }
    }

    /// `q [b, nq, dq]`, `src [b, ns, dsrc]`, optional `key_mask [b, ns]` (1 = real).
    fn forward(
        &self,
        q: Tensor<B, 3>,
        src: Tensor<B, 3>,
        key_mask: Option<&Tensor<B, 2>>,
    ) -> Tensor<B, 3> {
        let [b, nq, _] = q.dims();
        let [_, ns, _] = src.dims();
        let (h, hd) = (self.heads, self.head_dim);
        let qh = linear_rows(&self.q_proj, self.norm_q.forward(q))
            .reshape([b, nq, h, hd])
            .swap_dims(1, 2);
        let src_n = self.norm_src.forward(src);
        let kh = linear_rows(&self.k_proj, src_n.clone())
            .reshape([b, ns, h, hd])
            .swap_dims(1, 2);
        let vh = linear_rows(&self.v_proj, src_n)
            .reshape([b, ns, h, hd])
            .swap_dims(1, 2);
        let mut logits = qh
            .matmul(kh.swap_dims(2, 3))
            .mul_scalar(1.0 / (hd as f32).sqrt());
        if let Some(m) = key_mask {
            // 1 -> 0, 0 -> -1e9 (finite, so an all-masked row stays finite).
            let add = m
                .clone()
                .mul_scalar(1.0e9)
                .sub_scalar(1.0e9)
                .reshape([b, 1, 1, ns]);
            logits = logits + add;
        }
        let out = activation::softmax(logits, 3).matmul(vh);
        let out = out.swap_dims(1, 2).reshape([b, nq, h * hd]);
        linear_rows(&self.out_proj, out)
    }
}

fn mean_abs3<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 1> {
    let flat = x.abs().mean_dim(2).squeeze_dim::<2>(2);
    flat.mean_dim(1).squeeze_dim::<1>(1)
}

/// The Chimera V2 model.
#[derive(Module, Debug)]
pub struct ChimeraV2Model<B: Backend> {
    // --- board encoder (executed once) ---
    input_proj: Linear<B>,
    square_emb: Param<Tensor<B, 2>>,
    fuse_norm: RmsNorm<B>,
    prelude: Vec<Block<B>>,
    body: Vec<Block<B>>,
    board_norm: RmsNorm<B>,
    // --- optional one-shot visual perception ---
    visual: VisualEncoder<B>,
    visual_gate: Param<Tensor<B, 1>>,
    // --- candidate tokens ---
    cand_mix: Linear<B>,
    promo_emb: Param<Tensor<B, 2>>,
    facts_enc: Linear<B>,
    cand_norm: RmsNorm<B>,
    // --- successor encoder ---
    succ_sq: Linear<B>,
    succ_sq_emb: Param<Tensor<B, 2>>,
    succ_flat: Linear<B>,
    succ_flags: Linear<B>,
    succ_norm: RmsNorm<B>,
    // --- reply-set encoder ---
    reply_in1: Linear<B>,
    reply_in2: Linear<B>,
    reply_cand: Linear<B>,
    reply_self: MAttn<B>,
    reply_norm: RmsNorm<B>,
    pool_q: Param<Tensor<B, 2>>,
    reply_pool: MAttn<B>,
    reply_out: Linear<B>,
    reply_out_norm: RmsNorm<B>,
    tool_type_emb: Param<Tensor<B, 2>>,
    // --- planner ---
    ws_emb: Param<Tensor<B, 2>>,
    ws_init: Linear<B>,
    z_init_norm: RmsNorm<B>,
    thought_emb: Param<Tensor<B, 2>>,
    z_norm: RmsNorm<B>,
    self_attn: MAttn<B>,
    x_board: MAttn<B>,
    x_cand: MAttn<B>,
    x_succ: MAttn<B>,
    x_reply: MAttn<B>,
    ffn_norm: RmsNorm<B>,
    ffn1: Linear<B>,
    ffn2: Linear<B>,
    gate: Linear<B>,
    state_norm: RmsNorm<B>,
    // --- readout ---
    ctx_attn: MAttn<B>,
    pol1: Linear<B>,
    pol2: Linear<B>,
    shortcut: Linear<B>,
    wdl: Linear<B>,

    cfg: ModelConfig,
    exp: ExperimentalConfig,
    /// Counts board-encoder executions (proof that it runs once per decision).
    board_runs: Arc<AtomicUsize>,
}

impl<B: Backend> ChimeraV2Model<B> {
    pub fn new(cfg: ModelConfig, exp: ExperimentalConfig, device: &B::Device) -> Self {
        assert!(
            exp.is_chimera_v2(),
            "ChimeraV2Model requires architecture = chimera_v2"
        );
        let v = exp.v2.clone();
        let d = cfg.width;
        let (c, p, dr, sh) = (v.cand_dim, v.planner_dim, v.reply_dim, v.succ_hidden);
        let eps = cfg.rms_eps;
        let mk = |n: usize| -> Vec<Block<B>> { (0..n).map(|_| Block::new(&cfg, device)).collect() };
        let init_param = |shape: [usize; 2], std: f64| {
            Param::from_tensor(Tensor::random(
                shape,
                Distribution::Normal(0.0, std),
                device,
            ))
        };
        let lin = |i: usize, o: usize| LinearConfig::new(i, o).with_bias(true).init(device);
        let rms = |n: usize| RmsNormConfig::new(n).with_epsilon(eps).init(device);
        let model = Self {
            input_proj: LinearConfig::new(cfg.in_features, d)
                .with_bias(true)
                .init(device),
            square_emb: init_param([SQUARES, d], 0.02),
            fuse_norm: rms(d),
            prelude: mk(cfg.input_blocks),
            body: mk(cfg.core_blocks),
            board_norm: rms(d),
            visual: VisualEncoder::new(&exp.visual, d, device),
            visual_gate: Param::from_tensor(Tensor::from_data(
                TensorData::new(vec![0.0f32], [1]),
                device,
            )),
            cand_mix: lin(2 * d, c),
            promo_emb: init_param([5, c], 0.02),
            facts_enc: lin(CANDIDATE_FACT_FIELDS, c),
            cand_norm: rms(c),
            succ_sq: lin(BOARD_CODES, sh),
            succ_sq_emb: init_param([SQUARES, sh], 0.02),
            succ_flat: lin(SQUARES * sh, c),
            succ_flags: lin(SUCC_FLAG_DIM, c),
            succ_norm: rms(c),
            reply_in1: lin(REPLY_FEAT_DIM, dr),
            reply_in2: lin(dr, dr),
            reply_cand: lin(c, dr),
            reply_self: MAttn::new(dr, dr, dr, dr, 4, eps, device),
            reply_norm: rms(dr),
            pool_q: init_param([1, dr], 0.02),
            reply_pool: MAttn::new(dr, dr, dr, dr, 4, eps, device),
            reply_out: lin(dr, c),
            reply_out_norm: rms(c),
            tool_type_emb: init_param([3, c], 0.02),
            ws_emb: init_param([v.workspace_tokens, p], 0.02),
            ws_init: lin(d, p),
            z_init_norm: rms(p),
            thought_emb: init_param([v.max_thoughts, p], 0.02),
            z_norm: rms(p),
            self_attn: MAttn::new(p, p, p, p, v.planner_heads, eps, device),
            x_board: MAttn::new(p, d, p, p, v.planner_heads, eps, device),
            x_cand: MAttn::new(p, c, p, p, v.planner_heads, eps, device),
            x_succ: MAttn::new(p, c, p, p, v.planner_heads, eps, device),
            x_reply: MAttn::new(p, c, p, p, v.planner_heads, eps, device),
            ffn_norm: rms(p),
            ffn1: lin(p, v.planner_ffn),
            ffn2: lin(v.planner_ffn, p),
            gate: lin(2 * p, p),
            state_norm: rms(p),
            ctx_attn: MAttn::new(c, p, c, c, 4, eps, device),
            pol1: lin(4 * c, c),
            pol2: lin(c, 1),
            shortcut: lin(CANDIDATE_FACT_FIELDS, 1),
            wdl: LinearConfig::new(d + p, cfg.wdl_classes)
                .with_bias(true)
                .with_initializer(Initializer::Zeros)
                .init(device),
            cfg,
            exp,
            board_runs: Arc::new(AtomicUsize::new(0)),
        };
        model.force_init();
        model
    }

    /// Materialise every lazily initialised parameter (deterministic seeding, and so
    /// that timing measurements do not include first-use initialisation).
    pub fn force_init(&self) {
        struct Init;
        impl<B: Backend> burn::module::ModuleVisitor<B> for Init {
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
        let mut v = Init;
        self.visit(&mut v);
    }

    pub fn config(&self) -> &ModelConfig {
        &self.cfg
    }

    pub fn experimental(&self) -> &ExperimentalConfig {
        &self.exp
    }

    pub fn num_params(&self) -> usize {
        Module::num_params(self)
    }

    /// Board-encoder executions since construction (test / accounting hook).
    pub fn board_encoder_runs(&self) -> usize {
        self.board_runs.load(Ordering::SeqCst)
    }

    /// Trainable parameters by subsystem.
    pub fn param_breakdown(&self) -> Vec<(&'static str, usize)> {
        let bl = |v: &Vec<Block<B>>| -> usize { v.iter().map(Module::num_params).sum() };
        vec![
            (
                "board encoder (input proj + squares + prelude + body)",
                self.input_proj.num_params()
                    + self.square_emb.num_params()
                    + self.fuse_norm.num_params()
                    + bl(&self.prelude)
                    + bl(&self.body)
                    + self.board_norm.num_params(),
            ),
            (
                "visual encoder + fusion gate",
                self.visual.num_params() + self.visual_gate.num_params(),
            ),
            (
                "candidate encoder (B[from],B[to],promo)",
                self.cand_mix.num_params()
                    + self.promo_emb.num_params()
                    + self.cand_norm.num_params(),
            ),
            ("candidate facts encoder", self.facts_enc.num_params()),
            (
                "successor encoder",
                self.succ_sq.num_params()
                    + self.succ_sq_emb.num_params()
                    + self.succ_flat.num_params()
                    + self.succ_flags.num_params()
                    + self.succ_norm.num_params(),
            ),
            (
                "reply-set encoder",
                self.reply_in1.num_params()
                    + self.reply_in2.num_params()
                    + self.reply_cand.num_params()
                    + self.reply_self.num_params()
                    + self.reply_norm.num_params()
                    + self.pool_q.num_params()
                    + self.reply_pool.num_params()
                    + self.reply_out.num_params()
                    + self.reply_out_norm.num_params()
                    + self.tool_type_emb.num_params(),
            ),
            (
                "planner",
                self.ws_emb.num_params()
                    + self.ws_init.num_params()
                    + self.z_init_norm.num_params()
                    + self.thought_emb.num_params()
                    + self.z_norm.num_params()
                    + self.self_attn.num_params()
                    + self.x_board.num_params()
                    + self.x_cand.num_params()
                    + self.x_succ.num_params()
                    + self.x_reply.num_params()
                    + self.ffn_norm.num_params()
                    + self.ffn1.num_params()
                    + self.ffn2.num_params()
                    + self.gate.num_params()
                    + self.state_norm.num_params(),
            ),
            (
                "policy head",
                self.ctx_attn.num_params()
                    + self.pol1.num_params()
                    + self.pol2.num_params()
                    + self.shortcut.num_params(),
            ),
            ("WDL head", self.wdl.num_params()),
        ]
    }

    fn rel_idx(&self, device: &B::Device) -> Tensor<B, 2, Int> {
        let data = rel_index_data(self.cfg.heads, SQUARES);
        Tensor::<B, 2, Int>::from_data(
            TensorData::new(data, [self.cfg.heads, SQUARES * SQUARES]),
            device,
        )
    }

    /// The board encoder. Executes exactly once per decision.
    fn encode_board(&self, board: Tensor<B, 3>, visual: Option<Tensor<B, 4>>) -> Tensor<B, 3> {
        self.board_runs.fetch_add(1, Ordering::SeqCst);
        let device = board.device();
        let rel = self.rel_idx(&device);
        let p = linear_rows(&self.input_proj, board);
        let [b, s, d] = p.dims();
        let mut x = p + self.square_emb.val().reshape([1, s, d]).expand([b, s, d]);
        let visual_on =
            self.exp.visual.enabled && self.exp.visual.provider == VisualProviderKind::RenderV1;
        if let (true, Some(img)) = (visual_on, visual) {
            let tokens = self.visual.forward(img); // [b, 64, d]
            let g = activation::sigmoid(self.visual_gate.val()).reshape([1, 1, 1]);
            x = x + tokens * g;
        }
        let mut h = self.fuse_norm.forward(x);
        for blk in self.prelude.iter().chain(self.body.iter()) {
            h = blk.forward(h, rel.clone());
        }
        self.board_norm.forward(h)
    }

    fn gather_squares(board: &Tensor<B, 3>, idx: &Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [b, _, d] = board.dims();
        let w = idx.dims()[1];
        board
            .clone()
            .gather(1, idx.clone().unsqueeze_dim::<3>(2).expand([b, w, d]))
    }

    fn candidate_tokens(
        &self,
        board: &Tensor<B, 3>,
        cands: &CandidateTensors<B>,
        facts: &Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let from = Self::gather_squares(board, &cands.from_idx);
        let to = Self::gather_squares(board, &cands.to_idx);
        let mixed = linear_rows(&self.cand_mix, Tensor::cat(vec![from, to], 2));
        let [b, w, c] = mixed.dims();
        let promo = self
            .promo_emb
            .val()
            .reshape([1, 5, c])
            .expand([b, 5, c])
            .gather(
                1,
                cands
                    .promo_code
                    .clone()
                    .unsqueeze_dim::<3>(2)
                    .expand([b, w, c]),
            );
        let f = linear_rows(&self.facts_enc, facts.clone());
        let tok = self.cand_norm.forward(mixed + promo + f);
        // Padded candidates carry nothing.
        tok * cands.mask.clone().float().unsqueeze_dim::<3>(2)
    }

    fn type_row(&self, row: usize, like: &Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, w, c] = like.dims();
        self.tool_type_emb
            .val()
            .narrow(0, row, 1)
            .reshape([1, 1, c])
            .expand([b, w, c])
    }

    fn successor_tokens(&self, cand: &Tensor<B, 3>, world: &WorldTensors<B>) -> Tensor<B, 3> {
        let [b, w, _, _] = world.succ_board.dims();
        let sh = self.exp.v2.succ_hidden;
        let sq = world
            .succ_board
            .clone()
            .reshape([b * w, SQUARES, BOARD_CODES]);
        let h = activation::gelu(linear_rows(&self.succ_sq, sq));
        let h = h + self
            .succ_sq_emb
            .val()
            .reshape([1, SQUARES, sh])
            .expand([b * w, SQUARES, sh]);
        let flat = self.succ_flat.forward(h.reshape([b * w, SQUARES * sh]));
        let c = flat.dims()[1];
        let s = flat.reshape([b, w, c]);
        let f = linear_rows(&self.succ_flags, world.succ_flags.clone());
        let tok = self.succ_norm.forward(s + f + cand.clone());
        (tok.clone() + self.type_row(0, &tok))
            * (cand.clone().abs().sum_dim(2).greater_elem(0.0)).float()
    }

    fn reply_tokens(&self, cand: &Tensor<B, 3>, world: &WorldTensors<B>) -> Tensor<B, 3> {
        let [b, w, r, rf] = world.reply_feats.dims();
        let dr = self.exp.v2.reply_dim;
        let feats = world.reply_feats.clone().reshape([b * w, r, rf]);
        let mask = world.reply_mask.clone().reshape([b * w, r]);
        let e = activation::gelu(linear_rows(&self.reply_in1, feats));
        let e = linear_rows(&self.reply_in2, e);
        let cc = linear_rows(&self.reply_cand, cand.clone())
            .reshape([b * w, 1, dr])
            .expand([b * w, r, dr]);
        let e = e + cc;
        let sa = self.reply_self.forward(e.clone(), e.clone(), Some(&mask));
        let e = self.reply_norm.forward(e + sa);
        let q = self.pool_q.val().reshape([1, 1, dr]).expand([b * w, 1, dr]);
        let pooled = self.reply_pool.forward(q, e, Some(&mask)); // [b*w, 1, dr]
        let any = mask.max_dim(1).reshape([b * w, 1, 1]); // 1 if any real reply
        let pooled = (pooled * any).reshape([b, w, dr]);
        let out = linear_rows(&self.reply_out, pooled);
        let tok = self.reply_out_norm.forward(out + cand.clone());
        (tok.clone() + self.type_row(1, &tok))
            * (cand.clone().abs().sum_dim(2).greater_elem(0.0)).float()
    }

    /// One planner step. Returns the next state and its diagnostics.
    #[allow(clippy::too_many_arguments)]
    fn planner_step(
        &self,
        z: Tensor<B, 3>,
        t: usize,
        board: &Tensor<B, 3>,
        cand: &Tensor<B, 3>,
        cand_mask: &Tensor<B, 2>,
        succ: Option<&Tensor<B, 3>>,
        reply: Option<&Tensor<B, 3>>,
    ) -> (Tensor<B, 3>, PlannerDiag<B>) {
        let [b, k, p] = z.dims();
        let te = self
            .thought_emb
            .val()
            .narrow(0, t - 1, 1)
            .reshape([1, 1, p])
            .expand([b, k, p]);
        let n = self.z_norm.forward(z.clone() + te);
        let a = self.self_attn.forward(n.clone(), n.clone(), None);
        let q = n.clone() + a;
        let db = self.x_board.forward(q.clone(), board.clone(), None);
        let dm = self
            .x_cand
            .forward(q.clone(), cand.clone(), Some(cand_mask));
        let ds = succ.map(|s| self.x_succ.forward(q.clone(), s.clone(), Some(cand_mask)));
        let dr = reply.map(|r| self.x_reply.forward(q.clone(), r.clone(), Some(cand_mask)));
        let mut sum = q + db.clone() + dm.clone();
        if let Some(s) = &ds {
            sum = sum + s.clone();
        }
        if let Some(r) = &dr {
            sum = sum + r.clone();
        }
        let proposal = linear_rows(
            &self.ffn2,
            activation::gelu(linear_rows(&self.ffn1, self.ffn_norm.forward(sum))),
        );
        let u = activation::sigmoid(linear_rows(
            &self.gate,
            Tensor::cat(vec![n.clone(), proposal.clone()], 2),
        ));
        let keep = u.clone().neg().add_scalar(1.0);
        let z_next = self.state_norm.forward(n * keep + proposal * u.clone());

        let zeros1 = || Tensor::<B, 1>::zeros([b], &z_next.device());
        let (mb, mm) = (mean_abs3(db), mean_abs3(dm));
        let ms = ds.map(mean_abs3).unwrap_or_else(zeros1);
        let mr = dr.map(mean_abs3).unwrap_or_else(zeros1);
        let total = (mb.clone() + mm.clone() + ms.clone() + mr.clone()).add_scalar(1e-9);
        let gate_flat = u.reshape([b, k * p]);
        let gate_mean = gate_flat.clone().mean_dim(1).squeeze_dim::<1>(1);
        let gate_var = (gate_flat - gate_mean.clone().unsqueeze_dim::<2>(1))
            .powf_scalar(2.0)
            .mean_dim(1)
            .squeeze_dim::<1>(1);
        let diag = PlannerDiag {
            thought: t,
            mean_abs: mean_abs3(z_next.clone()),
            rms: z_next
                .clone()
                .powf_scalar(2.0)
                .mean_dim(2)
                .squeeze_dim::<2>(2)
                .mean_dim(1)
                .squeeze_dim::<1>(1)
                .sqrt(),
            delta: mean_abs3(z_next.clone() - z),
            gate_mean,
            gate_std: gate_var.sqrt(),
            share_board: mb / total.clone(),
            share_cand: mm / total.clone(),
            share_succ: ms / total.clone(),
            share_reply: mr / total,
        };
        (z_next, diag)
    }

    #[allow(clippy::too_many_arguments)]
    fn readout(
        &self,
        cand: &Tensor<B, 3>,
        succ: Option<&Tensor<B, 3>>,
        reply: Option<&Tensor<B, 3>>,
        z: &Tensor<B, 3>,
        board_pool: &Tensor<B, 2>,
        cands: &CandidateTensors<B>,
        facts: &Tensor<B, 3>,
    ) -> Readout<B> {
        let ctx = self.ctx_attn.forward(cand.clone(), z.clone(), None);
        let zero = || Tensor::<B, 3>::zeros(cand.dims(), &cand.device());
        let feat = Tensor::cat(
            vec![
                cand.clone(),
                succ.cloned().unwrap_or_else(zero),
                reply.cloned().unwrap_or_else(zero),
                ctx,
            ],
            2,
        );
        let h = activation::gelu(linear_rows(&self.pol1, feat));
        let mut logits = linear_rows(&self.pol2, h).squeeze_dim::<2>(2);
        if self.exp.v2.fact_shortcut {
            logits = logits + linear_rows(&self.shortcut, facts.clone()).squeeze_dim::<2>(2);
        }
        let [b, w] = logits.dims();
        // Mask padding with -inf; replace fully-terminal rows with 0 (as the probe readout does).
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
        let z_pool = z.clone().mean_dim(1).squeeze_dim::<2>(1);
        let wdl_logits = self
            .wdl
            .forward(Tensor::cat(vec![board_pool.clone(), z_pool], 1));
        Readout {
            policy: PolicyOutput {
                log_probs,
                mask: cands.mask.clone(),
                valid: cands.valid.clone(),
                base_all: Tensor::zeros([b, 1, 1], &cand.device()),
            },
            wdl_logits,
        }
    }

    /// Forward with `t` thoughts under the configured information schedule.
    pub fn forward(
        &self,
        input: &ChimeraV2Input<B>,
        cands: &CandidateTensors<B>,
        t: usize,
        opts: V2Options,
    ) -> ChimeraV2Output<B> {
        let max_t = self.exp.v2.max_thoughts;
        assert!(t >= 1 && t <= max_t, "thoughts {t} outside 1..={max_t}");
        let sched: InfoSchedule = self.exp.v2.info_schedule;
        let mut counts = ComputeCounts::default();

        // (1) board, ONCE.
        let runs_before = self.board_runs.load(Ordering::SeqCst);
        let board = self.encode_board(input.board.clone(), input.visual.clone());
        counts.board_encoder_runs = self.board_runs.load(Ordering::SeqCst) - runs_before;

        // (2) candidate tokens.
        let cand = self.candidate_tokens(&board, cands, &input.cand_facts);
        let cand_mask = cands.mask.clone().float();

        // (3) tool tokens, only if some thought <= t will see them.
        let need_s = (1..=t).any(|k| sched.successors_visible(k));
        let need_r = (1..=t).any(|k| sched.replies_visible(k));
        let [b, w, _] = cand.dims();
        let world = input.world.as_ref();
        assert!(
            world.is_some() || (!need_s && !need_r),
            "the info schedule needs world-model tensors but none were supplied"
        );
        let succ = match (need_s, world) {
            (true, Some(wm)) => {
                counts.successor_encodes = b * w;
                Some(self.successor_tokens(&cand, wm))
            }
            _ => None,
        };
        let reply = match (need_r, world) {
            (true, Some(wm)) => {
                counts.reply_set_encodes = b * w;
                Some(self.reply_tokens(&cand, wm))
            }
            _ => None,
        };

        // (4) planner.
        let d = board.dims()[2];
        let board_pool = board.clone().mean_dim(1).squeeze_dim::<2>(1);
        let p = self.exp.v2.planner_dim;
        let k = self.exp.v2.workspace_tokens;
        let z0 = self.ws_emb.val().reshape([1, k, p]).expand([b, k, p])
            + linear_rows(&self.ws_init, board_pool.clone().reshape([b, 1, d])).expand([b, k, p]);
        let mut z = self.z_init_norm.forward(z0);
        let mut readouts = Vec::new();
        let mut diag = Vec::new();
        for step in 1..=t {
            let sv = sched
                .successors_visible(step)
                .then_some(())
                .and(succ.as_ref());
            let rv = sched
                .replies_visible(step)
                .then_some(())
                .and(reply.as_ref());
            let (zn, dg) = self.planner_step(z, step, &board, &cand, &cand_mask, sv, rv);
            z = zn;
            counts.planner_steps += 1;
            diag.push(dg);
            if opts.all_readouts || step == t {
                readouts.push(self.readout(
                    &cand,
                    sv,
                    rv,
                    &z,
                    &board_pool,
                    cands,
                    &input.cand_facts,
                ));
            }
        }
        ChimeraV2Output {
            readouts,
            diag,
            counts,
        }
    }
}
