//! Workstation V2.5 candidate transformer (`candidate_v25`).
//!
//! One pass, no recurrence. A geometry-aware board transformer encodes the 64
//! squares; every legal move becomes a first-class token built from the encoded
//! from/to squares, a pooled board vector, a promotion embedding and the exact
//! `CandidateFactsV1` vector; one small masked self-attention block lets the
//! candidates interact; a scorer reads one logit per candidate.
//!
//! The WDL head reads the pooled board only (never the candidate facts), so the
//! value head keeps the mainline recipe and the experiment stays interpretable.

use burn::module::{Module, ModuleVisitor, Param};
use burn::nn::{Initializer, Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::prelude::*;
use burn::tensor::{Distribution, Int, TensorData, activation};

use crate::config::{Architecture, CANDIDATE_FACT_FIELDS, CandidateConfig, ModelConfig};
use crate::model::{
    Block, CandidateTensors, ModelOutput, PolicyOutput, Readout, linear_rows, rel_index_data,
};

/// Additive value that removes a padded key from attention. Finite (not -inf)
/// so a fully masked row (a terminal position) softmaxes to uniform, never NaN;
/// `exp(-1e9)` is exactly 0 in fp32, so padded keys carry no weight.
const MASKED_KEY: f32 = -1.0e9;

/// Std of the final policy scorer's weights. Small and nonzero: the fresh policy
/// is near-uniform, yet gradient reaches the facts encoder from update 1.
const POLICY_INIT_STD: f64 = 0.01;

/// Pre-norm masked self-attention + GeLU FFN over candidate tokens.
#[derive(Module, Debug)]
struct CandidateBlock<B: Backend> {
    norm1: RmsNorm<B>,
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    out_proj: Linear<B>,
    norm2: RmsNorm<B>,
    ffn1: Linear<B>,
    ffn2: Linear<B>,
    heads: usize,
    head_dim: usize,
}

impl<B: Backend> CandidateBlock<B> {
    fn new(c: &CandidateConfig, eps: f64, device: &B::Device) -> Self {
        let d = c.dim;
        Self {
            norm1: RmsNormConfig::new(d).with_epsilon(eps).init(device),
            q_proj: LinearConfig::new(d, d).with_bias(true).init(device),
            k_proj: LinearConfig::new(d, d).with_bias(true).init(device),
            v_proj: LinearConfig::new(d, d).with_bias(true).init(device),
            out_proj: LinearConfig::new(d, d).with_bias(true).init(device),
            norm2: RmsNormConfig::new(d).with_epsilon(eps).init(device),
            ffn1: LinearConfig::new(d, c.ffn).with_bias(true).init(device),
            ffn2: LinearConfig::new(c.ffn, d).with_bias(true).init(device),
            heads: c.heads,
            head_dim: d / c.heads,
        }
    }

    /// `key_pad`: `[b, w]`, true where the candidate slot is padding.
    fn forward(&self, x: Tensor<B, 3>, key_pad: Tensor<B, 2, Bool>) -> Tensor<B, 3> {
        let [b, w, d] = x.dims();
        let (h, hd) = (self.heads, self.head_dim);
        let n = self.norm1.forward(x.clone());
        let split = |t: Tensor<B, 3>| t.reshape([b, w, h, hd]).swap_dims(1, 2); // [b,h,w,hd]
        let q = split(linear_rows(&self.q_proj, n.clone()));
        let k = split(linear_rows(&self.k_proj, n.clone()));
        let v = split(linear_rows(&self.v_proj, n));
        let logits = q
            .matmul(k.swap_dims(2, 3))
            .mul_scalar(1.0 / (hd as f32).sqrt());
        let pad = key_pad
            .unsqueeze_dim::<3>(1)
            .unsqueeze_dim::<4>(1)
            .expand([b, h, w, w]);
        let logits = logits.mask_fill(pad, MASKED_KEY);
        let attn = activation::softmax(logits, 3).matmul(v);
        let attn = attn.swap_dims(1, 2).reshape([b, w, d]);
        let x = x + linear_rows(&self.out_proj, attn);
        let f = linear_rows(
            &self.ffn2,
            activation::gelu(linear_rows(&self.ffn1, self.norm2.forward(x.clone()))),
        );
        x + f
    }
}

/// The V2.5 candidate transformer.
#[derive(Module, Debug)]
pub struct CandidateV25Model<B: Backend> {
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
    policy1: Linear<B>,
    policy2: Linear<B>,
    wdl: Linear<B>,
    cfg: ModelConfig,
}

impl<B: Backend> CandidateV25Model<B> {
    pub fn new(cfg: ModelConfig, device: &B::Device) -> Self {
        cfg.validate().expect("valid candidate_v25 configuration");
        assert_eq!(
            cfg.architecture,
            Architecture::CandidateV25,
            "CandidateV25Model requires architecture candidate_v25"
        );
        let c = cfg.candidate.clone().expect("candidate geometry");
        let (d, cd, eps) = (cfg.width, c.dim, cfg.rms_eps);
        let model = Self {
            input_proj: LinearConfig::new(cfg.in_features, d)
                .with_bias(true)
                .init(device),
            square_emb: Param::from_tensor(Tensor::random(
                [cfg.squares, d],
                Distribution::Normal(0.0, 0.02),
                device,
            )),
            blocks: (0..cfg.core_blocks)
                .map(|_| Block::new(&cfg, device))
                .collect(),
            final_norm: RmsNormConfig::new(d).with_epsilon(eps).init(device),
            from_to_proj: LinearConfig::new(2 * d, cd).with_bias(true).init(device),
            global_proj: LinearConfig::new(d, cd).with_bias(true).init(device),
            promo_emb: Param::from_tensor(Tensor::random(
                [cfg.promo_codes, cd],
                Distribution::Normal(0.0, 0.02),
                device,
            )),
            // Normal (nonzero) initialization on both facts layers: the facts
            // path is a first-class input, not a zero-gated side channel.
            facts1: LinearConfig::new(CANDIDATE_FACT_FIELDS, c.facts_hidden)
                .with_bias(true)
                .init(device),
            facts2: LinearConfig::new(c.facts_hidden, cd)
                .with_bias(true)
                .init(device),
            token_norm: RmsNormConfig::new(cd).with_epsilon(eps).init(device),
            cand_blocks: (0..c.blocks)
                .map(|_| CandidateBlock::new(&c, eps, device))
                .collect(),
            policy1: LinearConfig::new(cd, c.policy_hidden)
                .with_bias(true)
                .init(device),
            policy2: LinearConfig::new(c.policy_hidden, 1)
                .with_bias(true)
                .with_initializer(Initializer::Normal {
                    mean: 0.0,
                    std: POLICY_INIT_STD,
                })
                .init(device),
            // Zero-init, as in mainline head v2: fresh WDL is exactly uniform.
            wdl: LinearConfig::new(d, cfg.wdl_classes)
                .with_bias(true)
                .with_initializer(Initializer::Zeros)
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

    fn rel_idx(&self, device: &B::Device) -> Tensor<B, 2, Int> {
        let (h, s) = (self.cfg.heads, self.cfg.squares);
        Tensor::<B, 2, Int>::from_data(TensorData::new(rel_index_data(h, s), [h, s * s]), device)
    }

    /// Encode the board: `[b, 64, in_features] -> [b, 64, width]`.
    pub fn encode_board(&self, board: Tensor<B, 3>) -> Tensor<B, 3> {
        let device = board.device();
        let rel_idx = self.rel_idx(&device);
        let p = linear_rows(&self.input_proj, board);
        let [b, s, d] = p.dims();
        let mut x = p + self.square_emb.val().reshape([1, s, d]).expand([b, s, d]);
        for block in &self.blocks {
            x = block.forward(x, rel_idx.clone());
        }
        self.final_norm.forward(x)
    }

    /// Build the candidate tokens `[b, w, cand_dim]` from the encoded board.
    ///
    /// `facts` is `[b, w, 8]`, zero on padding. When the model is the C0
    /// ablation (`facts_enabled == false`) the facts input is replaced by zeros:
    /// the facts modules exist and are counted, but carry no information.
    pub fn candidate_tokens(
        &self,
        y: &Tensor<B, 3>,
        cands: &CandidateTensors<B>,
        facts: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let [b, _s, d] = y.dims();
        let w = cands.width;
        let cd = self.cfg.candidate.as_ref().expect("candidate geometry").dim;
        let gather_sq = |idx: &Tensor<B, 2, Int>| {
            y.clone()
                .gather(1, idx.clone().unsqueeze_dim::<3>(2).expand([b, w, d]))
        };
        let h_from = gather_sq(&cands.from_idx);
        let h_to = gather_sq(&cands.to_idx);
        let from_to = linear_rows(&self.from_to_proj, Tensor::cat(vec![h_from, h_to], 2));

        let pooled = y.clone().mean_dim(1).squeeze_dim::<2>(1); // [b, d]
        let global = self
            .global_proj
            .forward(pooled)
            .unsqueeze_dim::<3>(1)
            .expand([b, w, cd]);

        let promo = self
            .promo_emb
            .val()
            .select(0, cands.promo_code.clone().reshape([b * w]))
            .reshape([b, w, cd]);

        let facts = if self.cfg.candidate.as_ref().is_some_and(|c| c.facts_enabled) {
            facts
        } else {
            facts.zeros_like()
        };
        let facts = linear_rows(
            &self.facts2,
            activation::gelu(linear_rows(&self.facts1, facts)),
        );
        self.token_norm.forward(from_to + global + promo + facts)
    }

    /// One forward pass. `board` is `[b, 64, in_features]`, `facts` is
    /// `[b, w, 8]` aligned with the candidate tensors.
    pub fn forward(
        &self,
        board: Tensor<B, 3>,
        cands: &CandidateTensors<B>,
        facts: Tensor<B, 3>,
    ) -> ModelOutput<B> {
        assert!(
            cands.width > 0,
            "candidate batch contains no legal candidates; terminal-only batches \
             have no policy path and must bypass neural evaluation"
        );
        let [b, w] = cands.mask.dims();
        assert_eq!(
            facts.dims(),
            [b, w, CANDIDATE_FACT_FIELDS],
            "facts tensor must be [batch, candidate width, {CANDIDATE_FACT_FIELDS}]"
        );
        let y = self.encode_board(board);
        let pooled = y.clone().mean_dim(1).squeeze_dim::<2>(1);
        let wdl_logits = self.wdl.forward(pooled);

        let mut tokens = self.candidate_tokens(&y, cands, facts);
        let key_pad = cands.mask.clone().bool_not();
        for block in &self.cand_blocks {
            tokens = block.forward(tokens, key_pad.clone());
        }
        let hidden = activation::gelu(linear_rows(&self.policy1, tokens));
        let logits = linear_rows(&self.policy2, hidden).squeeze_dim::<2>(2); // [b, w]

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

        ModelOutput {
            readouts: vec![Readout {
                policy: PolicyOutput {
                    log_probs,
                    mask: cands.mask.clone(),
                    valid: cands.valid.clone(),
                    base_all: None,
                },
                wdl_logits,
            }],
            executed_blocks: self.cfg.core_blocks,
        }
    }

    pub fn num_params(&self) -> usize {
        Module::num_params(self)
    }

    /// Exact parameter count by subsystem; sums to [`Self::num_params`].
    pub fn param_breakdown(&self) -> Vec<(&'static str, usize)> {
        vec![
            ("input_projection", self.input_proj.num_params()),
            ("square_embeddings", self.square_emb.num_params()),
            ("board_blocks", self.blocks.num_params()),
            ("final_norm", self.final_norm.num_params()),
            ("from_to_projection", self.from_to_proj.num_params()),
            ("global_projection", self.global_proj.num_params()),
            ("promotion_embedding", self.promo_emb.num_params()),
            (
                "facts_encoder",
                self.facts1.num_params() + self.facts2.num_params(),
            ),
            ("token_norm", self.token_norm.num_params()),
            ("candidate_blocks", self.cand_blocks.num_params()),
            (
                "policy_scorer",
                self.policy1.num_params() + self.policy2.num_params(),
            ),
            ("wdl_head", self.wdl.num_params()),
        ]
    }

    /// Identity of the facts encoder's first weight (gradient tests).
    pub fn facts_weight_id(&self) -> burn::module::ParamId {
        self.facts1.weight.id
    }

    /// Identity of the final policy scorer's weight (gradient tests).
    pub fn policy_weight_id(&self) -> burn::module::ParamId {
        self.policy2.weight.id
    }
}

/// `[b, width, 8]` facts tensor from per-position rows, zero on padding.
///
/// Errors visibly if any position has more rows than `width` (never truncates).
pub fn facts_tensor<B: Backend>(
    rows: &[&[recur64_core::CandidateFactsV1]],
    width: usize,
    device: &B::Device,
) -> anyhow::Result<Tensor<B, 3>> {
    let mut data = vec![0.0f32; rows.len() * width * CANDIDATE_FACT_FIELDS];
    for (i, r) in rows.iter().enumerate() {
        anyhow::ensure!(
            r.len() <= width,
            "position {i} has {} candidate fact rows, wider than the candidate width {width}",
            r.len()
        );
        for (k, row) in r.iter().enumerate() {
            let at = (i * width + k) * CANDIDATE_FACT_FIELDS;
            data[at..at + CANDIDATE_FACT_FIELDS].copy_from_slice(row);
        }
    }
    Ok(Tensor::from_data(
        TensorData::new(data, [rows.len(), width, CANDIDATE_FACT_FIELDS]),
        device,
    ))
}

/// Everything `CandidateV25Model::forward` consumes for a batch of positions.
pub struct CandidateInputs<B: Backend> {
    pub board: Tensor<B, 3>,
    pub cands: CandidateTensors<B>,
    pub facts: Tensor<B, 3>,
}

impl<B: Backend> CandidateInputs<B> {
    /// Build from authoritative states: observation, legal candidates and
    /// `CandidateFactsV1` all come from the same `GameState`, in legal-action
    /// order.
    pub fn from_states(
        states: &[recur64_core::GameState],
        device: &B::Device,
    ) -> anyhow::Result<Self> {
        let facts: Vec<Vec<recur64_core::CandidateFactsV1>> =
            states.iter().map(recur64_core::candidate_facts).collect();
        let obs: Vec<recur64_core::ObservationV1> = states
            .iter()
            .map(recur64_core::encode_observation_v1)
            .collect();
        let legal: Vec<Vec<recur64_core::ActionId>> =
            states.iter().map(|s| s.legal_actions()).collect();
        let refs: Vec<&[recur64_core::CandidateFactsV1]> =
            facts.iter().map(Vec::as_slice).collect();
        let obs_refs: Vec<&recur64_core::ObservationV1> = obs.iter().collect();
        Self::from_parts(&obs_refs, &legal, &refs, device)
    }

    /// Build from already-computed parts (the inference and learner paths).
    pub fn from_parts(
        observations: &[&recur64_core::ObservationV1],
        legal: &[Vec<recur64_core::ActionId>],
        facts: &[&[recur64_core::CandidateFactsV1]],
        device: &B::Device,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            observations.len() == legal.len() && legal.len() == facts.len(),
            "observation / legal / facts batch sizes differ"
        );
        for (i, (l, f)) in legal.iter().zip(facts).enumerate() {
            anyhow::ensure!(
                l.len() == f.len(),
                "position {i}: {} legal actions but {} candidate fact rows",
                l.len(),
                f.len()
            );
        }
        let b = observations.len();
        let mut board = Vec::with_capacity(b * 64 * 119);
        for o in observations {
            board.extend_from_slice(o.as_slice());
        }
        let board = Tensor::<B, 3>::from_data(TensorData::new(board, [b, 64, 119]), device);
        let lists: Vec<Vec<(u32, u32, u8)>> = legal
            .iter()
            .map(|l| {
                l.iter()
                    .map(|id| {
                        let (f, t, p) = id.decode();
                        (f as u32, t as u32, p.code())
                    })
                    .collect()
            })
            .collect();
        let cb = crate::action::CandidateBatch::from_lists(&lists);
        let facts = facts_tensor::<B>(facts, cb.width, device)?;
        let cands = CandidateTensors::from_batch(&cb, device);
        Ok(Self {
            board,
            cands,
            facts,
        })
    }
}
