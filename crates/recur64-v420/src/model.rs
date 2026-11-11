//! `recurrent_depth_v420`: the V4 base path (board transformer -> candidate tokens -> candidate
//! block -> readout) with the board transformer's blocks made explicit and optionally SHARED.
//!
//! `unique_blocks == 1` applies one block `k` times (recurrence, parameter count independent of
//! `k`); `unique_blocks == k` applies distinct blocks once each (the unshared controls).

use burn::module::{Module, Param};
use burn::nn::{Initializer, Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::prelude::*;
use burn::tensor::{Distribution, Int, TensorData, activation};
use recur64_model::candidate::CandidateInputs;
use recur64_model::config::{CANDIDATE_FACT_FIELDS, CandidateConfig, ModelConfig};
use recur64_model::model::Block;

use crate::MASKED_LOGIT;

pub const IDENTITY: &str = "recurrent_depth_v420";
const MASKED_KEY: f32 = -1.0e9;
const POLICY_INIT_STD: f64 = 0.01;
const BASE_HIDDEN: usize = 256;

/// Row-major relative-displacement bucket per `(from, to)` square pair, repeated per head
/// (`[heads, squares * squares]`), the layout `Block`'s relative-position bias indexes.
pub fn rel_index_data(heads: usize, squares: usize) -> Vec<i32> {
    assert_eq!(squares, 64, "V420 assumes an 8x8 board");
    let mut v = Vec::with_capacity(heads * squares * squares);
    for _ in 0..heads {
        for i in 0..squares {
            let (ri, fi) = (i / 8, i % 8);
            for j in 0..squares {
                let (rj, fj) = (j / 8, j % 8);
                v.push((rj as i32 - ri as i32 + 7) * 15 + (fj as i32 - fi as i32 + 7));
            }
        }
    }
    v
}

fn rows<B: Backend>(l: &Linear<B>, x: Tensor<B, 3>) -> Tensor<B, 3> {
    let [b, s, d] = x.dims();
    let y = l.forward(x.reshape([b * s, d]));
    let o = y.dims()[1];
    y.reshape([b, s, o])
}

/// Pre-norm masked self-attention + GeLU FFN over candidate tokens (same function as the V2.5
/// candidate block; the original is crate-private, so it is re-stated here).
#[derive(Module, Debug)]
pub struct CandBlock<B: Backend> {
    norm1: RmsNorm<B>,
    q: Linear<B>,
    k: Linear<B>,
    v: Linear<B>,
    o: Linear<B>,
    norm2: RmsNorm<B>,
    f1: Linear<B>,
    f2: Linear<B>,
    heads: usize,
    head_dim: usize,
}

impl<B: Backend> CandBlock<B> {
    fn new(c: &CandidateConfig, eps: f64, device: &B::Device) -> Self {
        let d = c.dim;
        let lin = |i, o| LinearConfig::new(i, o).with_bias(true).init(device);
        Self {
            norm1: RmsNormConfig::new(d).with_epsilon(eps).init(device),
            q: lin(d, d),
            k: lin(d, d),
            v: lin(d, d),
            o: lin(d, d),
            norm2: RmsNormConfig::new(d).with_epsilon(eps).init(device),
            f1: lin(d, c.ffn),
            f2: lin(c.ffn, d),
            heads: c.heads,
            head_dim: d / c.heads,
        }
    }

    fn forward(&self, x: Tensor<B, 3>, key_pad: Tensor<B, 2, Bool>) -> Tensor<B, 3> {
        let [b, w, d] = x.dims();
        let (h, hd) = (self.heads, self.head_dim);
        let n = self.norm1.forward(x.clone());
        let split = |t: Tensor<B, 3>| t.reshape([b, w, h, hd]).swap_dims(1, 2);
        let q = split(rows(&self.q, n.clone()));
        let k = split(rows(&self.k, n.clone()));
        let v = split(rows(&self.v, n));
        let logits = q.matmul(k.swap_dims(2, 3)).mul_scalar(1.0 / (hd as f32).sqrt());
        let pad = key_pad.unsqueeze_dim::<3>(1).unsqueeze_dim::<4>(1).expand([b, h, w, w]);
        let attn = activation::softmax(logits.mask_fill(pad, MASKED_KEY), 3).matmul(v);
        let x = x + rows(&self.o, attn.swap_dims(1, 2).reshape([b, w, d]));
        let f = rows(&self.f2, activation::gelu(rows(&self.f1, self.norm2.forward(x.clone()))));
        x + f
    }
}

/// One arm's geometry. Everything but `unique_blocks` is the V4 base geometry.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Geometry {
    pub width: usize,
    pub heads: usize,
    pub ffn: usize,
    pub unique_blocks: usize,
    pub cand_dim: usize,
    pub cand_heads: usize,
    pub cand_ffn: usize,
    pub facts_hidden: usize,
}

impl Geometry {
    /// V4 base geometry (`candidate_v25` primary: width 640, 10 heads, FFN 1280, 256-wide tokens).
    pub fn v4(unique_blocks: usize) -> Self {
        let m = ModelConfig::candidate_v25(true);
        let c = m.candidate.clone().expect("candidate geometry");
        Self {
            width: m.width,
            heads: m.heads,
            ffn: m.ffn,
            unique_blocks,
            cand_dim: c.dim,
            cand_heads: c.heads,
            cand_ffn: c.ffn,
            facts_hidden: c.facts_hidden,
        }
    }

    /// Tiny geometry for CPU tests only.
    pub fn tiny(unique_blocks: usize) -> Self {
        Self {
            width: 32,
            heads: 4,
            ffn: 64,
            unique_blocks,
            cand_dim: 16,
            cand_heads: 2,
            cand_ffn: 32,
            facts_hidden: 8,
        }
    }

    fn model_config(&self) -> ModelConfig {
        let mut m = ModelConfig::candidate_v25(true);
        m.width = self.width;
        m.heads = self.heads;
        m.ffn = self.ffn;
        m.core_blocks = self.unique_blocks;
        m
    }

    fn cand_config(&self) -> CandidateConfig {
        CandidateConfig {
            dim: self.cand_dim,
            heads: self.cand_heads,
            ffn: self.cand_ffn,
            blocks: 1,
            facts_hidden: self.facts_hidden,
            ..CandidateConfig::default()
        }
    }
}

#[derive(Module, Debug)]
pub struct DepthModel<B: Backend> {
    input_proj: Linear<B>,
    square_emb: Param<Tensor<B, 2>>,
    blocks: Vec<Block<B>>,
    final_norm: RmsNorm<B>,
    from_to: Linear<B>,
    global: Linear<B>,
    promo: Param<Tensor<B, 2>>,
    facts1: Linear<B>,
    facts2: Linear<B>,
    token_norm: RmsNorm<B>,
    cand: CandBlock<B>,
    node: Linear<B>,
    hidden: Linear<B>,
    out: Linear<B>,
    heads: usize,
    squares: usize,
    unique_blocks: usize,
    cand_dim: usize,
}

impl<B: Backend> DepthModel<B> {
    pub fn new(g: &Geometry, device: &B::Device) -> Self {
        let m = g.model_config();
        let (d, cd, eps) = (g.width, g.cand_dim, m.rms_eps);
        let lin = |i, o| LinearConfig::new(i, o).with_bias(true).init(device);
        let model = Self {
            input_proj: lin(m.in_features, d),
            square_emb: Param::from_tensor(Tensor::random(
                [m.squares, d],
                Distribution::Normal(0.0, 0.02),
                device,
            )),
            blocks: (0..g.unique_blocks).map(|_| Block::new(&m, device)).collect(),
            final_norm: RmsNormConfig::new(d).with_epsilon(eps).init(device),
            from_to: lin(2 * d, cd),
            global: lin(d, cd),
            promo: Param::from_tensor(Tensor::random(
                [m.promo_codes, cd],
                Distribution::Normal(0.0, 0.02),
                device,
            )),
            facts1: lin(CANDIDATE_FACT_FIELDS, g.facts_hidden),
            facts2: lin(g.facts_hidden, cd),
            token_norm: RmsNormConfig::new(cd).with_epsilon(eps).init(device),
            cand: CandBlock::new(&g.cand_config(), eps, device),
            node: lin(d, cd),
            hidden: lin(2 * cd, BASE_HIDDEN),
            out: LinearConfig::new(BASE_HIDDEN, 1)
                .with_bias(false)
                .with_initializer(Initializer::Normal {
                    mean: 0.0,
                    std: POLICY_INIT_STD,
                })
                .init(device),
            heads: m.heads,
            squares: m.squares,
            unique_blocks: g.unique_blocks,
            cand_dim: cd,
        };
        for p in [&model.square_emb, &model.promo] {
            let _ = p.val();
        }
        model
    }

    pub fn unique_blocks(&self) -> usize {
        self.unique_blocks
    }

    pub fn num_params_total(&self) -> usize {
        Module::num_params(self)
    }

    /// Block applied at depth step `i`: shared blocks repeat, unshared blocks are used once each.
    fn block_at(&self, i: usize) -> &Block<B> {
        &self.blocks[i % self.unique_blocks]
    }

    /// Masked root logits `[b, w]` after `k` applications of the board block(s).
    /// Unshared arms (`unique_blocks > 1`) execute exactly `unique_blocks` blocks: `k` must equal it.
    pub fn logits(&self, inputs: &CandidateInputs<B>, k: usize) -> anyhow::Result<Tensor<B, 2>> {
        anyhow::ensure!(k >= 1, "depth K must be at least 1");
        anyhow::ensure!(
            self.unique_blocks == 1 || k == self.unique_blocks,
            "an unshared model with {} blocks executes exactly {} applications, not K={k}",
            self.unique_blocks,
            self.unique_blocks
        );
        let device = inputs.board.device();
        let (h, s) = (self.heads, self.squares);
        let rel = Tensor::<B, 2, Int>::from_data(
            TensorData::new(rel_index_data(h, s), [h, s * s]),
            &device,
        );
        let p = rows(&self.input_proj, inputs.board.clone());
        let [b, s, d] = p.dims();
        let mut x = p + self.square_emb.val().reshape([1, s, d]).expand([b, s, d]);
        for i in 0..k {
            x = self.block_at(i).forward(x, rel.clone());
        }
        let y = self.final_norm.forward(x);

        let cands = &inputs.cands;
        let (w, cd) = (cands.width, self.cand_dim);
        let gather = |idx: &Tensor<B, 2, Int>| {
            y.clone().gather(1, idx.clone().unsqueeze_dim::<3>(2).expand([b, w, d]))
        };
        let from_to = rows(
            &self.from_to,
            Tensor::cat(vec![gather(&cands.from_idx), gather(&cands.to_idx)], 2),
        );
        let pooled = y.clone().mean_dim(1).squeeze_dim::<2>(1);
        let global = self.global.forward(pooled.clone()).unsqueeze_dim::<3>(1).expand([b, w, cd]);
        let promo = self
            .promo
            .val()
            .select(0, cands.promo_code.clone().reshape([b * w]))
            .reshape([b, w, cd]);
        let facts = rows(
            &self.facts2,
            activation::gelu(rows(&self.facts1, inputs.facts.clone())),
        );
        let mut tokens = self.token_norm.forward(from_to + global + promo + facts);
        tokens = self.cand.forward(tokens, cands.mask.clone().bool_not());

        let root = self.node.forward(pooled).unsqueeze_dim::<3>(1).expand([b, w, cd]);
        let hid = activation::gelu(rows(&self.hidden, Tensor::cat(vec![tokens, root], 2)));
        let z = rows(&self.out, hid).squeeze_dim::<2>(2);
        Ok(z.mask_fill(cands.mask.clone().bool_not(), MASKED_LOGIT))
    }
}
