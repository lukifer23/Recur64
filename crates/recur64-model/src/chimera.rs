//! X15 "Chimera": the novel-architecture laboratory model.
//!
//! Four independently gated pathways:
//!
//! 1. **Symbolic geometry-aware square-token transformer prelude**
//!    (authoritative: the policy reads this path, and it is what keeps the
//!    network's chess identity legible).
//! 2. **Explicit recurrent latent reasoning / scratchpad**: `K` persistent
//!    `[b, K, Daux]` thought slots, initialized from learned embeddings plus a
//!    projection of the pooled symbolic board, updated once per thought.
//! 3. **Deterministic chess coprocessor** exposed to the reasoning bus through
//!    cross-attention, with a native implementation and a real WebAssembly one
//!    that must agree byte for byte.
//! 4. **Literal canonical visual-board pathway**: a small residual CNN over a
//!    deterministic top-down render of the position.
//!
//! A retrieval pathway has a designed interface
//! ([`crate::experimental::RetrievalProviderKind`]) but is **inactive in X1**.
//!
//! # Thought loop (the architecture contract)
//!
//! ```text
//! x  = input_proj(board) + square_emb                     # [b, 64, D]
//! S0 = input_blocks(x)                                    # prelude
//! Z0 = latent_emb + latent_init(mean(S0))                 # [b, K, Daux], conditioned at t=0
//! for t in 1..=T:
//!     S  = core_blocks(inject_norm(S + x * alpha))        # shared recurrent square core
//!     Z += xattn_sq(Z, S)                                 # always on
//!     Z += g_compute * xattn_compute(Z, compute_tokens)   # gated
//!     Z += g_visual  * xattn_visual(Z, visual_tokens)     # gated
//!     Z += latent_ffn(latent_norm(Z))
//!     S += g_reason  * project_back(mean_K(Z))            # gated feedback
//!     readout_t = sparse_readout(output_blocks(S), wdl_from_latent(mean_K(Z)))
//! final output = readout_T
//! ```
//!
//! Every gate is `sigmoid(logit)` initialized at `0.5`, so no pathway is
//! switched off at initialization and every subsystem receives a non-zero
//! gradient on the first step (asserted by
//! `crates/recur64-runtime/tests/x15_modules.rs`).

use burn::module::{Module, ModuleVisitor, Param};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{Initializer, Linear, LinearConfig, PaddingConfig2d, RmsNorm, RmsNormConfig};
use burn::optim::GradientsParams;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Distribution, Int, TensorData, activation};

use crate::experimental::{
    Architecture, CANDIDATE_FACT_FIELDS, CHIMERA_HEAD_VERSION, ExperimentalConfig,
    VisualProviderKind,
};
use crate::model::{
    Block, CandidateTensors, NamedParam, Readout, ReadoutHeads, linear_rows, rel_index_data,
    sparse_readout,
};

/// Number of squares in the symbolic path.
const SQUARES: usize = 64;

/// One auxiliary cross-attention module at the reasoning width.
///
/// The query is the reasoning state (`Daux`); the keys and values come from a
/// source token set of width `Dsrc` (the trunk width for the square source, the
/// reasoning width for the compute and visual token sets).
#[derive(Module, Debug)]
pub struct CrossAttn<B: Backend> {
    norm_q: RmsNorm<B>,
    norm_src: RmsNorm<B>,
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    out_proj: Linear<B>,
    heads: usize,
    head_dim: usize,
}

impl<B: Backend> CrossAttn<B> {
    fn new(dsrc: usize, daux: usize, heads: usize, eps: f64, device: &B::Device) -> Self {
        assert!(
            daux.is_multiple_of(heads),
            "aux width {daux} not divisible by aux heads {heads}"
        );
        Self {
            norm_q: RmsNormConfig::new(daux).with_epsilon(eps).init(device),
            norm_src: RmsNormConfig::new(dsrc).with_epsilon(eps).init(device),
            q_proj: LinearConfig::new(daux, daux).with_bias(true).init(device),
            k_proj: LinearConfig::new(dsrc, daux).with_bias(true).init(device),
            v_proj: LinearConfig::new(dsrc, daux).with_bias(true).init(device),
            out_proj: LinearConfig::new(daux, daux).with_bias(true).init(device),
            heads,
            head_dim: daux / heads,
        }
    }

    /// `q` is `[b, k, Daux]`, `src` is `[b, n, Dsrc]`.
    fn forward(&self, q: Tensor<B, 3>, src: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, k, _] = q.dims();
        let [_, n, _] = src.dims();
        let h = self.heads;
        let hd = self.head_dim;

        // Flattened to one 2-D matmul per projection (same map as the rank-3
        // `Linear::forward`; see `linear_rows`).
        let qh = linear_rows(&self.q_proj, self.norm_q.forward(q))
            .reshape([b, k, h, hd])
            .swap_dims(1, 2); // [b, h, k, hd]
        let src_n = self.norm_src.forward(src);
        let kh = linear_rows(&self.k_proj, src_n.clone())
            .reshape([b, n, h, hd])
            .swap_dims(1, 2);
        let vh = linear_rows(&self.v_proj, src_n)
            .reshape([b, n, h, hd])
            .swap_dims(1, 2);

        let scale = 1.0 / (hd as f32).sqrt();
        let logits = qh.matmul(kh.swap_dims(2, 3)).mul_scalar(scale);
        let attn = activation::softmax(logits, 3);
        let out = attn.matmul(vh); // [b, h, k, hd]
        let out = out.swap_dims(1, 2).reshape([b, k, self.head_dim * h]);
        linear_rows(&self.out_proj, out)
    }
}

/// A small residual convolutional block.
#[derive(Module, Debug)]
pub struct ConvBlock<B: Backend> {
    c1: Conv2d<B>,
    c2: Conv2d<B>,
}

impl<B: Backend> ConvBlock<B> {
    fn new(channels: usize, device: &B::Device) -> Self {
        let conv = || {
            Conv2dConfig::new([channels, channels], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device)
        };
        Self {
            c1: conv(),
            c2: conv(),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let h = activation::gelu(self.c1.forward(x.clone()));
        let h = self.c2.forward(h);
        x + h
    }
}

/// The residual CNN that turns a canonical RGB board image into 64 spatial
/// tokens at the reasoning width.
///
/// The stride plan is `log2(resolution / 8)` stride-2 stages, so the encoder
/// emits exactly one token per board square. Residual blocks sit between the
/// downsampling stages (they are stored flat; the per-stage stride is derived
/// from the downsampling-stage count, so no extra module state is needed).
#[derive(Module, Debug)]
pub struct VisualEncoder<B: Backend> {
    stem: Conv2d<B>,
    downs: Vec<Conv2d<B>>,
    blocks: Vec<ConvBlock<B>>,
    proj: Conv2d<B>,
    out_channels: usize,
}

impl<B: Backend> VisualEncoder<B> {
    fn new(cfg: &crate::experimental::VisualConfig, daux: usize, device: &B::Device) -> Self {
        let c = cfg.channels;
        let stages = (cfg.resolution / 8).trailing_zeros() as usize;
        assert!(
            stages >= 2,
            "visual resolution {} needs at least two stride-2 stages",
            cfg.resolution
        );
        let stem = Conv2dConfig::new([3, c], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .init(device);
        let mut downs = Vec::new();
        let mut blocks = Vec::new();
        for stage in 0..stages {
            let out_ch = if stage + 1 == stages { daux } else { c };
            downs.push(
                Conv2dConfig::new([c, out_ch], [3, 3])
                    .with_stride([2, 2])
                    .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                    .init(device),
            );
            if stage + 1 < stages {
                for _ in 0..cfg.blocks {
                    blocks.push(ConvBlock::new(c, device));
                }
            }
        }
        let proj = Conv2dConfig::new([daux, daux], [1, 1]).init(device);
        Self {
            stem,
            downs,
            blocks,
            proj,
            out_channels: daux,
        }
    }

    /// The reasoning width this encoder projects into.
    pub fn out_channels(&self) -> usize {
        self.out_channels
    }

    /// `[b, 3, S, S]` in `0..1` -> `[b, 64, Daux]`.
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        let mut h = activation::gelu(self.stem.forward(x));
        let n = self.downs.len();
        let intermediate = n - 1;
        let per_stage = self.blocks.len() / intermediate.max(1);
        for (i, down) in self.downs.iter().enumerate() {
            h = activation::gelu(down.forward(h));
            if i < intermediate {
                for b in &self.blocks[i * per_stage..(i + 1) * per_stage] {
                    h = b.forward(h);
                }
            }
        }
        let [b, _c, hh, ww] = h.dims();
        assert_eq!(
            (hh, ww),
            (8, 8),
            "the visual encoder must land on an 8x8 token grid, got {hh}x{ww}"
        );
        let h = self.proj.forward(h);
        h.reshape([b, self.out_channels, SQUARES]).swap_dims(1, 2)
    }
}

/// The optional inputs a thought loop can consume.
pub struct ChimeraInput<B: Backend> {
    /// Canonical Observation V1 `[b, 64, 119]`.
    pub board: Tensor<B, 3>,
    /// `ComputeBankV1` bytes as `[b, tokens, fields]` floats in `0..255`.
    pub compute: Option<Tensor<B, 3>>,
    /// Canonical board image `[b, 3, S, S]` in `0..1`.
    pub visual: Option<Tensor<B, 4>>,
    /// `CandidateFactsV1` as `[b, width, CANDIDATE_FACT_FIELDS]` (zero on
    /// padded candidates), in candidate order.
    pub cand_facts: Option<Tensor<B, 3>>,
}

/// Per-thought diagnostics. All are `[b]` (or `[b, 3]` for WDL) so a probe can
/// read them back and a training step can ignore them.
pub struct ThoughtMetrics<B: Backend> {
    /// Predicted-policy entropy over legal candidates (nats).
    pub policy_entropy: Tensor<B, 1>,
    /// `[b, 3]` WDL probabilities.
    pub wdl: Tensor<B, 2>,
    /// Mean reasoning-state norm.
    pub latent_norm: Tensor<B, 1>,
    /// Mean change in the reasoning state since the previous thought.
    pub latent_delta_norm: Tensor<B, 1>,
    /// Mean absolute contribution of the square cross-attention.
    pub square_pathway: Tensor<B, 1>,
    /// Mean absolute contribution of the gated compute cross-attention.
    pub compute_pathway: Tensor<B, 1>,
    /// Mean absolute contribution of the gated visual cross-attention.
    pub visual_pathway: Tensor<B, 1>,
    /// The learned residual scales actually used.
    pub gate_compute: Tensor<B, 1>,
    pub gate_visual: Tensor<B, 1>,
    pub gate_reason: Tensor<B, 1>,
    /// `KL(policy_t || policy_{t-1})` over legal candidates (nats), per
    /// position. Zero on the first row of a forward pass.
    pub policy_kl_prev: Tensor<B, 1>,
    /// L1 distance between this and the previous row's WDL probabilities,
    /// per position. Zero on the first row of a forward pass.
    pub wdl_l1_prev: Tensor<B, 1>,
}

/// Forward-pass options. These affect only what is *observed*, never the
/// function the network computes or the gradients of the normal path.
#[derive(Clone, Copy, Debug, Default)]
pub struct ForwardOptions {
    /// Read out (policy + WDL + metrics) after **every** thought regardless of
    /// the configured deep-supervision mode. This is a measurement mode: it
    /// costs `output_blocks` extra block evaluations per intermediate thought
    /// and does not change the final readout, the parameters, or the
    /// checkpoint/scientific identity.
    pub diagnostic_readouts: bool,
}

/// The result of a Chimera forward pass.
pub struct ChimeraOutput<B: Backend> {
    /// One readout per thought that reads out (only the last for
    /// `final_only_v1`; every thought in diagnostic mode).
    pub readouts: Vec<Readout<B>>,
    /// One metrics row per readout, aligned with `readouts`.
    pub thoughts: Vec<ThoughtMetrics<B>>,
    /// Transformer blocks executed (accounting only).
    pub executed_blocks: usize,
}

/// The X15 "Chimera" model.
#[derive(Module, Debug)]
pub struct ChimeraModel<B: Backend> {
    // --- symbolic trunk (authoritative) ---
    input_proj: Linear<B>,
    square_emb: Param<Tensor<B, 2>>,
    input_blocks: Vec<Block<B>>,
    core_blocks: Vec<Block<B>>,
    output_blocks: Vec<Block<B>>,
    inject_norm: RmsNorm<B>,
    alpha_logit: Param<Tensor<B, 1>>,

    // --- heads ---
    final_norm: RmsNorm<B>,
    wdl: Linear<B>,
    source_proj: Linear<B>,
    dest_proj: Linear<B>,
    promo1: Linear<B>,
    promo2: Linear<B>,
    wdl_from_latent: Linear<B>,

    // --- latent reasoning ---
    latent_emb: Param<Tensor<B, 2>>,
    latent_init: Linear<B>,
    latent_norm: RmsNorm<B>,
    latent_ffn1: Linear<B>,
    latent_ffn2: Linear<B>,
    xattn_sq: CrossAttn<B>,
    xattn_compute: CrossAttn<B>,
    xattn_visual: CrossAttn<B>,
    feedback: Linear<B>,
    gate_compute_logit: Param<Tensor<B, 1>>,
    gate_visual_logit: Param<Tensor<B, 1>>,
    gate_reason_logit: Param<Tensor<B, 1>>,

    // --- deterministic compute tokens ---
    compute_proj: Linear<B>,
    compute_type_emb: Param<Tensor<B, 2>>,

    // --- per-candidate exact facts (bias on the policy logits) ---
    facts_l1: Linear<B>,
    facts_l2: Linear<B>,

    // --- visual pathway ---
    visual: VisualEncoder<B>,

    cfg: crate::config::ModelConfig,
    exp: ExperimentalConfig,
}

impl<B: Backend> ChimeraModel<B> {
    /// Build a fresh Chimera model. The experimental config must already have
    /// validated (`ExperimentalConfig::validate`).
    pub fn new(
        cfg: crate::config::ModelConfig,
        exp: ExperimentalConfig,
        device: &B::Device,
    ) -> Self {
        assert!(
            exp.is_chimera(),
            "ChimeraModel requires architecture = chimera_v1"
        );
        let d = cfg.width;
        let daux = exp.reasoning.aux_width;
        let heads = exp.reasoning.aux_heads;
        let eps = cfg.rms_eps;
        let mk = |n: usize| -> Vec<Block<B>> { (0..n).map(|_| Block::new(&cfg, device)).collect() };
        let init_param = |shape: [usize; 2]| {
            Param::from_tensor(Tensor::random(
                shape,
                Distribution::Normal(0.0, 0.02),
                device,
            ))
        };

        let model = Self {
            input_proj: LinearConfig::new(cfg.in_features, d)
                .with_bias(true)
                .init(device),
            square_emb: init_param([cfg.squares, d]),
            input_blocks: mk(cfg.input_blocks),
            core_blocks: mk(cfg.core_blocks),
            output_blocks: mk(cfg.output_blocks),
            inject_norm: RmsNormConfig::new(d).with_epsilon(eps).init(device),
            alpha_logit: Param::from_tensor(Tensor::from_data(
                TensorData::new(vec![(0.1f32 / 0.9).ln()], [1]),
                device,
            )),
            final_norm: RmsNormConfig::new(d).with_epsilon(eps).init(device),
            wdl: LinearConfig::new(d, cfg.wdl_classes)
                .with_bias(true)
                .with_initializer(Initializer::Zeros)
                .init(device),
            source_proj: LinearConfig::new(d, cfg.policy_dim)
                .with_bias(true)
                .init(device),
            dest_proj: LinearConfig::new(d, cfg.policy_dim)
                .with_bias(true)
                .init(device),
            promo1: LinearConfig::new(d * 3, cfg.policy_dim)
                .with_bias(true)
                .init(device),
            promo2: LinearConfig::new(cfg.policy_dim, cfg.promo_codes - 1)
                .with_bias(true)
                .init(device),
            // Zero-init keeps the reasoning term neutral at step 0 while still
            // receiving gradient from the first update.
            wdl_from_latent: LinearConfig::new(daux, cfg.wdl_classes)
                .with_bias(true)
                .with_initializer(Initializer::Zeros)
                .init(device),
            latent_emb: init_param([exp.reasoning_tokens, daux]),
            latent_init: LinearConfig::new(d, daux).with_bias(true).init(device),
            latent_norm: RmsNormConfig::new(daux).with_epsilon(eps).init(device),
            latent_ffn1: LinearConfig::new(daux, exp.reasoning.latent_ffn)
                .with_bias(true)
                .init(device),
            latent_ffn2: LinearConfig::new(exp.reasoning.latent_ffn, daux)
                .with_bias(true)
                .init(device),
            xattn_sq: CrossAttn::new(d, daux, heads, eps, device),
            xattn_compute: CrossAttn::new(daux, daux, heads, eps, device),
            xattn_visual: CrossAttn::new(daux, daux, heads, eps, device),
            feedback: LinearConfig::new(daux, d).with_bias(true).init(device),
            // All three gates sit at sigmoid(0) = 0.5: fully open, but small
            // enough that the auxiliary pathways do not dominate the trunk at
            // initialization.
            gate_compute_logit: Param::from_tensor(Tensor::from_data(
                TensorData::new(vec![0.0f32], [1]),
                device,
            )),
            gate_visual_logit: Param::from_tensor(Tensor::from_data(
                TensorData::new(vec![0.0f32], [1]),
                device,
            )),
            gate_reason_logit: Param::from_tensor(Tensor::from_data(
                TensorData::new(vec![0.0f32], [1]),
                device,
            )),
            compute_proj: LinearConfig::new(exp.compute.fields, daux)
                .with_bias(true)
                .init(device),
            compute_type_emb: init_param([2, daux]),
            facts_l1: LinearConfig::new(CANDIDATE_FACT_FIELDS, exp.candidate_facts.hidden)
                .with_bias(true)
                .init(device),
            // Zero-init: a fresh network ignores the facts until it learns to use them.
            facts_l2: LinearConfig::new(exp.candidate_facts.hidden, 1)
                .with_bias(true)
                .with_initializer(Initializer::Zeros)
                .init(device),
            visual: VisualEncoder::new(&exp.visual, daux, device),
            cfg,
            exp,
        };
        model.force_init();
        model
    }

    /// The architecture this model is.
    pub fn architecture(&self) -> Architecture {
        Architecture::ChimeraV1
    }

    /// The readout-head function version.
    pub fn head_version(&self) -> u32 {
        CHIMERA_HEAD_VERSION
    }

    pub fn config(&self) -> &crate::config::ModelConfig {
        &self.cfg
    }

    pub fn experimental(&self) -> &ExperimentalConfig {
        &self.exp
    }

    /// Force every parameter to materialize its value.
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
        let mut v = Init;
        self.visit(&mut v);
    }

    fn gate(p: &Param<Tensor<B, 1>>) -> Tensor<B, 1> {
        activation::sigmoid(p.val())
    }

    fn alpha(&self) -> Tensor<B, 3> {
        activation::sigmoid(self.alpha_logit.val()).reshape([1, 1, 1])
    }

    fn rel_idx(&self, device: &B::Device) -> Tensor<B, 2, Int> {
        let data = rel_index_data(self.cfg.heads, self.cfg.squares);
        Tensor::<B, 2, Int>::from_data(
            TensorData::new(data, [self.cfg.heads, self.cfg.squares * self.cfg.squares]),
            device,
        )
    }

    fn run_blocks(
        &self,
        blocks: &[Block<B>],
        x: Tensor<B, 3>,
        rel: &Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        let mut h = x;
        for b in blocks {
            h = b.forward(h, rel.clone());
        }
        h
    }

    fn mean_over_tokens(x: Tensor<B, 3>) -> Tensor<B, 2> {
        x.mean_dim(1).squeeze_dim::<2>(1)
    }

    fn mean_abs(x: Tensor<B, 3>) -> Tensor<B, 1> {
        let flat = x.abs().mean_dim(2).squeeze_dim::<2>(2); // [b, n]
        flat.mean_dim(1).squeeze_dim::<1>(1)
    }

    /// The fact MLP's raw output `[b, width]` for supplied facts, regardless of
    /// whether the pathway is enabled (diagnostics only).
    pub fn facts_bias_raw(&self, facts: Tensor<B, 3>) -> Tensor<B, 2> {
        let h = activation::gelu(linear_rows(&self.facts_l1, facts));
        linear_rows(&self.facts_l2, h)
            .squeeze_dim::<2>(2)
            .mul_scalar(self.exp.candidate_facts.gain)
    }

    /// The per-candidate policy bias from `CandidateFactsV1`, `[b, width]`, or
    /// `None` when the pathway is off or no facts were supplied. When it is off
    /// the facts parameters are not executed, so the output does not depend on
    /// them.
    fn facts_bias(&self, input: &ChimeraInput<B>) -> Option<Tensor<B, 2>> {
        if !self.exp.candidate_facts.enabled {
            return None;
        }
        input.cand_facts.as_ref().map(|facts| {
            let h = activation::gelu(linear_rows(&self.facts_l1, facts.clone()));
            linear_rows(&self.facts_l2, h)
                .squeeze_dim::<2>(2)
                .mul_scalar(self.exp.candidate_facts.gain)
        })
    }

    /// The compute tokens projected into the reasoning width, with a
    /// square-vs-global type embedding. `None` when there is no bank.
    fn compute_tokens(&self, compute: &Tensor<B, 3>, device: &B::Device) -> Tensor<B, 3> {
        let proj = linear_rows(&self.compute_proj, compute.clone()); // [b, T, daux]
        let [b, n, _] = proj.dims();
        let globals = self
            .exp
            .compute
            .tokens
            .saturating_sub(recur64_coproc::SQUARES);
        let mut types = vec![0i32; n];
        for slot in types.iter_mut().skip(n.saturating_sub(globals)) {
            *slot = 1;
        }
        let types = Tensor::<B, 2, Int>::from_data(TensorData::new(types, [1, n]), device)
            .expand([b, n])
            .unsqueeze_dim::<3>(2)
            .expand([b, n, self.exp.reasoning.aux_width]);
        let te = self
            .compute_type_emb
            .val()
            .reshape([1, 2, self.exp.reasoning.aux_width])
            .expand([b, 2, self.exp.reasoning.aux_width])
            .gather(1, types);
        proj + te
    }

    /// Training-semantics forward with `t` thoughts: readouts follow the
    /// configured deep-supervision mode. With `reasoning.enabled = false` this
    /// is the symbolic-only control (see `forward_symbolic`).
    pub fn forward_thoughts(
        &self,
        input: &ChimeraInput<B>,
        cands: &CandidateTensors<B>,
        t: usize,
    ) -> ChimeraOutput<B> {
        self.forward_with(input, cands, t, ForwardOptions::default())
    }

    /// Measurement forward: identical network, identical final readout, but a
    /// policy/WDL readout and metrics row after every thought.
    pub fn forward_thoughts_diagnostic(
        &self,
        input: &ChimeraInput<B>,
        cands: &CandidateTensors<B>,
        t: usize,
    ) -> ChimeraOutput<B> {
        self.forward_with(
            input,
            cands,
            t,
            ForwardOptions {
                diagnostic_readouts: true,
            },
        )
    }

    pub fn forward_with(
        &self,
        input: &ChimeraInput<B>,
        cands: &CandidateTensors<B>,
        t: usize,
        opts: ForwardOptions,
    ) -> ChimeraOutput<B> {
        assert!(t >= 1, "thought steps must be >= 1");
        assert!(
            t <= 8,
            "thought steps {t} exceeds the supported maximum of 8"
        );
        let exp = &self.exp;
        assert!(
            exp.reasoning.enabled || t == 1,
            "thought steps > 1 requires the reasoning latents"
        );
        let device = input.board.device();
        let rel = self.rel_idx(&device);
        let daux = exp.reasoning.aux_width;
        let k = exp.reasoning_tokens;

        let x = {
            let p = linear_rows(&self.input_proj, input.board.clone());
            let [b, s, d] = p.dims();
            let emb = self.square_emb.val().reshape([1, s, d]).expand([b, s, d]);
            p + emb
        };
        let mut s_state = self.run_blocks(&self.input_blocks, x.clone(), &rel);
        let inject = x * self.alpha();

        if !exp.reasoning.enabled {
            return self.forward_symbolic(s_state, inject, &rel, cands, self.facts_bias(input));
        }

        // Position-conditioned latent initialization.
        let pooled = Self::mean_over_tokens(s_state.clone()); // [b, D]
        let [bb, _] = pooled.dims();
        let emb = self
            .latent_emb
            .val()
            .reshape([1, k, daux])
            .expand([bb, k, daux]);
        let mut z = emb
            + self
                .latent_init
                .forward(pooled)
                .reshape([bb, 1, daux])
                .expand([bb, k, daux]);
        z = self.latent_norm.forward(z);

        let compute_tokens = input
            .compute
            .as_ref()
            .filter(|_| exp.compute.enabled)
            .map(|c| self.compute_tokens(c, &device));

        let visual_tokens = match (&input.visual, exp.visual.enabled, exp.visual.provider) {
            (Some(v), true, VisualProviderKind::RenderV1) => Some(self.visual.forward(v.clone())),
            _ => None,
        };

        let g_compute = Self::gate(&self.gate_compute_logit);
        let g_visual = Self::gate(&self.gate_visual_logit);
        let g_reason = Self::gate(&self.gate_reason_logit);

        let facts_bias = self.facts_bias(input);
        let mut readouts = Vec::new();
        let mut thoughts = Vec::new();
        let mut prev_z: Option<Tensor<B, 3>> = None;
        let mut prev_readout: Option<(Tensor<B, 2>, Tensor<B, 2>)> = None;

        for step in 1..=t {
            s_state = self.run_blocks(
                &self.core_blocks,
                self.inject_norm.forward(s_state + inject.clone()),
                &rel,
            );

            let d_sq = self.xattn_sq.forward(z.clone(), s_state.clone());
            let square_pathway = Self::mean_abs(d_sq.clone());
            z = z + d_sq;

            let mut compute_pathway = None;
            if let Some(c) = compute_tokens.as_ref() {
                let d = self.xattn_compute.forward(z.clone(), c.clone());
                compute_pathway = Some(Self::mean_abs(d.clone()));
                let g = g_compute.clone().reshape([1, 1, 1]);
                z = z + d * g;
            }

            let mut visual_pathway = None;
            if let Some(v) = visual_tokens.as_ref() {
                let d = self.xattn_visual.forward(z.clone(), v.clone());
                visual_pathway = Some(Self::mean_abs(d.clone()));
                let g = g_visual.clone().reshape([1, 1, 1]);
                z = z + d * g;
            }

            let ff = linear_rows(
                &self.latent_ffn2,
                activation::gelu(linear_rows(&self.latent_ffn1, z.clone())),
            );
            z = z + ff;

            let delta = prev_z
                .as_ref()
                .map(|p| Self::mean_abs(z.clone() - p.clone()));
            prev_z = Some(z.clone());

            {
                let fb = linear_rows(&self.feedback, z.clone()); // [b, K, D]
                let fb = Self::mean_over_tokens(fb).unsqueeze_dim::<3>(1); // [b, 1, D]
                let g = g_reason.clone().reshape([1, 1, 1]);
                s_state = s_state + fb * g;
            }

            if step == t || exp.deep_supervision.reads_intermediate() || opts.diagnostic_readouts {
                let y = self.run_blocks(&self.output_blocks, s_state.clone(), &rel);
                let latent_wdl = self
                    .wdl_from_latent
                    .forward(Self::mean_over_tokens(z.clone()));
                let readout = sparse_readout(
                    &self.readout_heads(),
                    y,
                    Some(latent_wdl),
                    cands,
                    facts_bias.clone(),
                );
                let entropy = self.policy_entropy(&readout);
                let zeros = || {
                    Tensor::<B, 1>::zeros(
                        [readout.wdl_logits.dims()[0]],
                        &readout.wdl_logits.device(),
                    )
                };
                let wdl_now = activation::softmax(readout.wdl_logits.clone(), 1);
                let (policy_kl_prev, wdl_l1_prev) = match prev_readout.as_ref() {
                    Some((prev_readout_policy, prev_wdl)) => (
                        Self::policy_kl(&readout, prev_readout_policy),
                        (wdl_now.clone() - prev_wdl.clone())
                            .abs()
                            .sum_dim(1)
                            .squeeze_dim::<1>(1),
                    ),
                    None => (zeros(), zeros()),
                };
                prev_readout = Some((readout.policy.log_probs.clone(), wdl_now.clone()));
                thoughts.push(ThoughtMetrics {
                    policy_entropy: entropy,
                    wdl: wdl_now,
                    policy_kl_prev,
                    wdl_l1_prev,
                    latent_norm: Self::mean_abs(z.clone()),
                    latent_delta_norm: delta.unwrap_or_else(zeros),
                    square_pathway: square_pathway.clone(),
                    compute_pathway: compute_pathway.clone().unwrap_or_else(zeros),
                    visual_pathway: visual_pathway.clone().unwrap_or_else(zeros),
                    gate_compute: g_compute.clone(),
                    gate_visual: g_visual.clone(),
                    gate_reason: g_reason.clone(),
                });
                readouts.push(readout);
            }
        }

        let intermediate_reads =
            exp.deep_supervision.reads_intermediate() || opts.diagnostic_readouts;
        ChimeraOutput {
            readouts,
            thoughts,
            executed_blocks: self.cfg.executed_blocks_final(t)
                + if intermediate_reads {
                    self.cfg.output_blocks * (t - 1)
                } else {
                    0
                },
        }
    }

    /// The symbolic-only control: symbolic input/prelude, the shared square
    /// core once, the output blocks, and the historical-style sparse head.
    ///
    /// No latent state is created, no cross-attention, latent FFN, compute or
    /// visual path runs, there is no latent feedback, and no latent WDL term
    /// is added. The output is therefore a function of the symbolic
    /// parameters only (`input_proj`, `square_emb`, input/core/output blocks,
    /// `inject_norm`, `alpha_logit`, and the heads other than
    /// `wdl_from_latent`).
    fn forward_symbolic(
        &self,
        s_state: Tensor<B, 3>,
        inject: Tensor<B, 3>,
        rel: &Tensor<B, 2, Int>,
        cands: &CandidateTensors<B>,
        facts_bias: Option<Tensor<B, 2>>,
    ) -> ChimeraOutput<B> {
        let s_state = self.run_blocks(
            &self.core_blocks,
            self.inject_norm.forward(s_state + inject),
            rel,
        );
        let y = self.run_blocks(&self.output_blocks, s_state, rel);
        let readout = sparse_readout(&self.readout_heads(), y, None, cands, facts_bias);
        let device = readout.wdl_logits.device();
        let b = readout.wdl_logits.dims()[0];
        let zeros = || Tensor::<B, 1>::zeros([b], &device);
        let one_zero = || Tensor::<B, 1>::zeros([1], &device);
        let thoughts = vec![ThoughtMetrics {
            policy_entropy: self.policy_entropy(&readout),
            wdl: activation::softmax(readout.wdl_logits.clone(), 1),
            latent_norm: zeros(),
            latent_delta_norm: zeros(),
            square_pathway: zeros(),
            compute_pathway: zeros(),
            visual_pathway: zeros(),
            gate_compute: one_zero(),
            gate_visual: one_zero(),
            gate_reason: one_zero(),
            policy_kl_prev: zeros(),
            wdl_l1_prev: zeros(),
        }];
        ChimeraOutput {
            readouts: vec![readout],
            thoughts,
            executed_blocks: self.cfg.executed_blocks_final(1),
        }
    }

    /// `KL(current || previous)` over legal candidates, per position. Padded
    /// candidates hold `log_prob = 0` and are masked out, as are terminal rows.
    fn policy_kl(current: &Readout<B>, prev_log_probs: &Tensor<B, 2>) -> Tensor<B, 1> {
        let lp = current.policy.log_probs.clone();
        let p = lp.clone().exp();
        let mask = current.policy.mask.clone().float();
        let per = (p * (lp - prev_log_probs.clone()) * mask)
            .sum_dim(1)
            .squeeze_dim::<1>(1);
        per * current.policy.valid.clone().float()
    }

    fn policy_entropy(&self, readout: &Readout<B>) -> Tensor<B, 1> {
        let p = readout.policy.log_probs.clone().exp();
        let per = -(p * readout.policy.log_probs.clone())
            .sum_dim(1)
            .squeeze_dim::<1>(1);
        let per = per.mask_fill(readout.policy.valid.clone().bool_not(), 0.0);
        let n = readout
            .policy
            .valid
            .clone()
            .float()
            .sum()
            .clamp(1.0, f32::MAX);
        per.sum() / n
    }

    fn readout_heads(&self) -> ReadoutHeads<'_, B> {
        ReadoutHeads {
            final_norm: &self.final_norm,
            wdl: &self.wdl,
            source_proj: &self.source_proj,
            dest_proj: &self.dest_proj,
            promo1: &self.promo1,
            promo2: &self.promo2,
        }
    }

    /// Total unique parameters (shared blocks counted once).
    pub fn num_params(&self) -> usize {
        Module::num_params(self)
    }

    /// Parameter count by subsystem, so visual and auxiliary growth is never
    /// hidden.
    pub fn param_breakdown(&self) -> Vec<(&'static str, usize)> {
        vec![
            (
                "symbolic (input_proj + square_emb + prelude)",
                self.input_proj.num_params()
                    + self.square_emb.num_params()
                    + self.input_blocks.num_params()
                    + self.inject_norm.num_params()
                    + self.alpha_logit.num_params(),
            ),
            (
                "recurrent square core (shared)",
                self.core_blocks.num_params(),
            ),
            ("output blocks", self.output_blocks.num_params()),
            (
                "reasoning latents + bus",
                self.latent_emb.num_params()
                    + self.latent_init.num_params()
                    + self.latent_norm.num_params()
                    + self.latent_ffn1.num_params()
                    + self.latent_ffn2.num_params()
                    + self.feedback.num_params()
                    + self.gate_compute_logit.num_params()
                    + self.gate_visual_logit.num_params()
                    + self.gate_reason_logit.num_params(),
            ),
            (
                "compute embedding + xattn",
                self.compute_proj.num_params()
                    + self.compute_type_emb.num_params()
                    + self.xattn_compute.num_params(),
            ),
            (
                "visual CNN + xattn",
                self.visual.num_params() + self.xattn_visual.num_params(),
            ),
            ("square xattn", self.xattn_sq.num_params()),
            (
                "heads",
                self.final_norm.num_params()
                    + self.source_proj.num_params()
                    + self.dest_proj.num_params()
                    + self.promo1.num_params()
                    + self.promo2.num_params()
                    + self.wdl.num_params()
                    + self.wdl_from_latent.num_params(),
            ),
            (
                "candidate facts MLP",
                self.facts_l1.num_params() + self.facts_l2.num_params(),
            ),
        ]
    }

    /// Every float parameter in visit order, as `(v0000, dims, values)`.
    ///
    /// The order is the derived `Module::visit` order (struct field order, then
    /// vector index), which is deterministic for a given source revision. The
    /// names are positional by design: this exists for the semantic weight
    /// digest, which needs a stable encoding, not for human readability.
    pub fn named_float_params(&self) -> anyhow::Result<Vec<NamedParam>> {
        struct Collect {
            out: Vec<NamedParam>,
            n: usize,
        }
        impl<B: Backend> ModuleVisitor<B> for Collect {
            fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
                let t = param.val();
                let dims = t.dims().to_vec();
                let values = t.into_data().to_vec::<f32>().ok();
                self.out.push(NamedParam {
                    name: format!("v{:04}", self.n),
                    dims,
                    values,
                });
                self.n += 1;
            }
        }
        let mut c = Collect {
            out: Vec::new(),
            n: 0,
        };
        self.visit(&mut c);
        Ok(c.out)
    }
}

/// Per-subsystem gradient reports for the autodiff backend.
impl<B: AutodiffBackend> ChimeraModel<B> {
    /// Per-subsystem gradient L2 norms from a backward pass. A subsystem whose
    /// norm is exactly zero is reported as `0.0` rather than hidden — that is
    /// what the module-gradient probe asserts against.
    pub fn subsystem_grad_norms(&self, grads: &GradientsParams) -> Vec<(&'static str, f32)> {
        let mut norms = vec![
            (
                "symbolic",
                module_grad_norm(&self.input_proj, grads)
                    + module_grad_norm(&self.square_emb, grads)
                    + module_grad_norm(&self.input_blocks, grads),
            ),
            ("recurrent_core", module_grad_norm(&self.core_blocks, grads)),
            (
                "output_blocks",
                module_grad_norm(&self.output_blocks, grads),
            ),
            (
                "reasoning_latents",
                module_grad_norm(&self.latent_emb, grads)
                    + module_grad_norm(&self.latent_init, grads)
                    + module_grad_norm(&self.latent_ffn1, grads)
                    + module_grad_norm(&self.latent_ffn2, grads)
                    + module_grad_norm(&self.feedback, grads)
                    + module_grad_norm(&self.xattn_sq, grads),
            ),
            (
                "compute",
                module_grad_norm(&self.compute_proj, grads)
                    + module_grad_norm(&self.compute_type_emb, grads)
                    + module_grad_norm(&self.xattn_compute, grads),
            ),
            (
                "visual",
                module_grad_norm(&self.visual, grads) + module_grad_norm(&self.xattn_visual, grads),
            ),
            (
                "heads",
                module_grad_norm(&self.final_norm, grads)
                    + module_grad_norm(&self.source_proj, grads)
                    + module_grad_norm(&self.dest_proj, grads)
                    + module_grad_norm(&self.promo1, grads)
                    + module_grad_norm(&self.promo2, grads)
                    + module_grad_norm(&self.wdl, grads)
                    + module_grad_norm(&self.wdl_from_latent, grads),
            ),
            (
                "gates",
                module_grad_norm(&self.gate_compute_logit, grads)
                    + module_grad_norm(&self.gate_visual_logit, grads)
                    + module_grad_norm(&self.gate_reason_logit, grads),
            ),
        ];
        // Reported only when the pathway is on, so a disabled pathway is not
        // mistaken for a starved one.
        if self.exp.candidate_facts.enabled {
            norms.push((
                "candidate_facts",
                module_grad_norm(&self.facts_l1, grads) + module_grad_norm(&self.facts_l2, grads),
            ));
        }
        norms
    }
}

/// Blocks a Chimera config executes at `t` thoughts (accounting helper).
pub fn executed_blocks(
    cfg: &crate::config::ModelConfig,
    exp: &ExperimentalConfig,
    t: usize,
) -> usize {
    cfg.executed_blocks_final(t)
        + if exp.deep_supervision.reads_intermediate() {
            cfg.output_blocks * (t - 1)
        } else {
            0
        }
}

fn scalar_f32<B: Backend>(t: Tensor<B, 1>) -> f32 {
    t.into_data()
        .to_vec::<f32>()
        .ok()
        .and_then(|v| v.first().copied())
        .unwrap_or(f32::NAN)
}

fn module_grad_norm<B: AutodiffBackend, M: Module<B>>(m: &M, grads: &GradientsParams) -> f32 {
    struct V<'a, B: AutodiffBackend> {
        grads: &'a GradientsParams,
        sum_sq: f64,
        _p: std::marker::PhantomData<B>,
    }
    impl<B: AutodiffBackend> ModuleVisitor<B> for V<'_, B> {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
            if let Some(g) = self.grads.get::<B::InnerBackend, D>(param.id) {
                let s = scalar_f32::<B::InnerBackend>((g.clone() * g).sum());
                self.sum_sq += s as f64;
            }
        }
    }
    let mut v = V::<B> {
        grads,
        sum_sq: 0.0,
        _p: std::marker::PhantomData,
    };
    m.visit(&mut v);
    v.sum_sq.sqrt() as f32
}
