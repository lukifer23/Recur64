//! Transition evidence encoder (`evidence_encoder_v4_v1`).
//!
//! Law B (content causality) is enforced by construction, not learned:
//!
//! * every `Linear` on the content path has NO bias;
//! * every activation satisfies f(0) = 0 (GELU);
//! * normalisation is scale-only (`RmsNorm` has a gain, no shift, and maps 0 to 0);
//! * the per-square "position" signal is a MULTIPLICATIVE gain `(1 + g_s)` on content-derived
//!   tokens plus a relative-displacement bias inside attention logits, never an additive
//!   learned vector, so a position cannot create evidence from zero content.
//!
//! Therefore an all-zero [`ContentBatch`] produces an exactly-zero message, whatever the
//! routing metadata is. The message is built from CONTENT only: root, parent and child
//! observations, the known action's content (one-hot from/to/promotion, which is all-zero for
//! "no content") and the child's check/terminal flags. Branch identity, parent links and depth
//! are routing metadata and never enter this module.

use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::prelude::*;
use burn::tensor::{Distribution, Int, TensorData, activation};
use recur64_core::ActionId;
use recur64_model::config::EvidenceConfig;
use recur64_model::model::RelPosBias;

use crate::util::{rel_index_data, rows};
use crate::{IN_FEATURES, SQUARES};

/// One-hot from (64) + one-hot to (64) + promotion one-hot N/B/R/Q (4).
pub const ACTION_CONTENT: usize = 132;
/// Child in-check, child terminal.
pub const FLAG_CONTENT: usize = 2;

/// Content of a batch of queried transitions. All five tensors are CONTENT.
pub struct ContentBatch<B: Backend> {
    /// `[n, 64, 119]` root observation.
    pub root: Tensor<B, 3>,
    /// `[n, 64, 119]` parent observation (already known to the model).
    pub parent: Tensor<B, 3>,
    /// `[n, 64, 119]` returned child observation.
    pub child: Tensor<B, 3>,
    /// `[n, 132]` action content.
    pub action: Tensor<B, 2>,
    /// `[n, 2]` child in-check / terminal.
    pub flags: Tensor<B, 2>,
}

/// Host-side action content: one-hot from/to/promotion.
pub fn action_content(action: u16) -> anyhow::Result<[f32; ACTION_CONTENT]> {
    let (from, to, promo) = ActionId::from_index(u32::from(action))?.decode();
    let mut v = [0.0f32; ACTION_CONTENT];
    v[from as usize] = 1.0;
    v[64 + to as usize] = 1.0;
    let code = promo.code();
    if code > 0 {
        v[128 + usize::from(code) - 1] = 1.0;
    }
    Ok(v)
}

impl<B: Backend> ContentBatch<B> {
    pub fn len(&self) -> usize {
        self.root.dims()[0]
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Build from host data. Each observation slice must be `64 * 119` long.
    pub fn from_host(
        root: &[&[f32]],
        parent: &[&[f32]],
        child: &[&[f32]],
        actions: &[u16],
        flags: &[[f32; FLAG_CONTENT]],
        device: &B::Device,
    ) -> anyhow::Result<Self> {
        let n = root.len();
        anyhow::ensure!(
            n > 0 && parent.len() == n && child.len() == n && actions.len() == n && flags.len() == n,
            "content batch inputs differ in length"
        );
        let obs = |rows: &[&[f32]]| -> anyhow::Result<Tensor<B, 3>> {
            let mut buf = Vec::with_capacity(n * SQUARES * IN_FEATURES);
            for r in rows {
                anyhow::ensure!(r.len() == SQUARES * IN_FEATURES, "bad observation length");
                buf.extend_from_slice(r);
            }
            Ok(Tensor::from_data(
                TensorData::new(buf, [n, SQUARES, IN_FEATURES]),
                device,
            ))
        };
        let mut act = Vec::with_capacity(n * ACTION_CONTENT);
        for &a in actions {
            act.extend_from_slice(&action_content(a)?);
        }
        let mut fl = Vec::with_capacity(n * FLAG_CONTENT);
        for f in flags {
            fl.extend_from_slice(f);
        }
        Ok(Self {
            root: obs(root)?,
            parent: obs(parent)?,
            child: obs(child)?,
            action: Tensor::from_data(TensorData::new(act, [n, ACTION_CONTENT]), device),
            flags: Tensor::from_data(TensorData::new(fl, [n, FLAG_CONTENT]), device),
        })
    }

    /// Every content tensor set to exactly zero (evaluation-only ablation and invariant tests).
    pub fn zeroed(&self) -> Self {
        Self {
            root: self.root.clone().zeros_like(),
            parent: self.parent.clone().zeros_like(),
            child: self.child.clone().zeros_like(),
            action: self.action.clone().zeros_like(),
            flags: self.flags.clone().zeros_like(),
        }
    }

    /// Row `i` receives the content of row `(i + shift) % n` (evaluation-only shuffle).
    pub fn rolled(&self, shift: usize) -> Self {
        let n = self.len();
        let idx: Vec<i32> = (0..n).map(|i| ((i + shift) % n) as i32).collect();
        let device = self.root.device();
        let idx = Tensor::<B, 1, Int>::from_data(TensorData::new(idx, [n]), &device);
        Self {
            root: self.root.clone().select(0, idx.clone()),
            parent: self.parent.clone().select(0, idx.clone()),
            child: self.child.clone().select(0, idx.clone()),
            action: self.action.clone().select(0, idx.clone()),
            flags: self.flags.clone().select(0, idx),
        }
    }
}

/// Bias-free transformer block over the 64 squares (pre-RMSNorm attention + GELU FFN).
#[derive(Module, Debug)]
pub struct ContentBlock<B: Backend> {
    norm1: RmsNorm<B>,
    q: Linear<B>,
    k: Linear<B>,
    v: Linear<B>,
    o: Linear<B>,
    rel: RelPosBias<B>,
    norm2: RmsNorm<B>,
    f1: Linear<B>,
    f2: Linear<B>,
    heads: usize,
    head_dim: usize,
}

impl<B: Backend> ContentBlock<B> {
    pub fn new(dim: usize, heads: usize, ffn: usize, eps: f64, device: &B::Device) -> Self {
        let lin = |i: usize, o: usize| LinearConfig::new(i, o).with_bias(false).init(device);
        Self {
            norm1: RmsNormConfig::new(dim).with_epsilon(eps).init(device),
            q: lin(dim, dim),
            k: lin(dim, dim),
            v: lin(dim, dim),
            o: lin(dim, dim),
            rel: RelPosBias::<B>::new(heads, device),
            norm2: RmsNormConfig::new(dim).with_epsilon(eps).init(device),
            f1: lin(dim, ffn),
            f2: lin(ffn, dim),
            heads,
            head_dim: dim / heads,
        }
    }

    fn attention(&self, x: Tensor<B, 3>, rel_idx: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [b, s, d] = x.dims();
        let (h, hd) = (self.heads, self.head_dim);
        let split = |t: Tensor<B, 3>| t.reshape([b, s, h, hd]).swap_dims(1, 2);
        let q = split(rows(&self.q, x.clone()));
        let k = split(rows(&self.k, x.clone()));
        let v = split(rows(&self.v, x));
        let logits = q
            .matmul(k.swap_dims(2, 3))
            .mul_scalar(1.0 / (hd as f32).sqrt());
        let logits = logits + self.rel.bias(rel_idx, s, h);
        let out = activation::softmax(logits, 3).matmul(v);
        rows(&self.o, out.swap_dims(1, 2).reshape([b, s, d]))
    }

    pub fn forward(&self, x: Tensor<B, 3>, rel_idx: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let a = x.clone() + self.attention(self.norm1.forward(x), rel_idx);
        let f = rows(&self.f2, activation::gelu(rows(&self.f1, self.norm2.forward(a.clone()))));
        a + f
    }
}

/// `EvidenceEncoder(root, parent, child, action, flags) -> message`, zero iff content is zero.
#[derive(Module, Debug)]
pub struct EvidenceEncoder<B: Backend> {
    in_proj: Linear<B>,
    sq_gain: Param<Tensor<B, 2>>,
    blocks: Vec<ContentBlock<B>>,
    final_norm: RmsNorm<B>,
    pool_proj: Linear<B>,
    act_proj: Linear<B>,
    flag_proj: Linear<B>,
    heads: usize,
}

impl<B: Backend> EvidenceEncoder<B> {
    pub fn new(e: &EvidenceConfig, eps: f64, device: &B::Device) -> Self {
        let lin = |i: usize, o: usize| LinearConfig::new(i, o).with_bias(false).init(device);
        let d = e.content_dim;
        Self {
            in_proj: lin(4 * IN_FEATURES, d),
            sq_gain: Param::from_tensor(Tensor::random(
                [SQUARES, d],
                Distribution::Normal(0.0, 0.1),
                device,
            )),
            blocks: (0..e.content_blocks)
                .map(|_| ContentBlock::new(d, e.content_heads, e.content_ffn, eps, device))
                .collect(),
            final_norm: RmsNormConfig::new(d).with_epsilon(eps).init(device),
            pool_proj: lin(d, e.message_dim),
            act_proj: lin(ACTION_CONTENT, e.message_dim),
            flag_proj: lin(FLAG_CONTENT, e.message_dim),
            heads: e.content_heads,
        }
    }

    /// `[n, message_dim]`. Exactly zero when every content tensor is exactly zero.
    pub fn forward(&self, c: &ContentBatch<B>) -> Tensor<B, 2> {
        let device = c.root.device();
        let delta = c.child.clone() - c.parent.clone();
        let x = Tensor::cat(
            vec![delta, c.child.clone(), c.parent.clone(), c.root.clone()],
            2,
        );
        let [n, s, _] = x.dims();
        let mut t = rows(&self.in_proj, x);
        let d = t.dims()[2];
        let gain = self
            .sq_gain
            .val()
            .reshape([1, s, d])
            .expand([n, s, d])
            .add_scalar(1.0);
        t = t * gain;
        let rel_idx = Tensor::<B, 2, Int>::from_data(
            TensorData::new(rel_index_data(self.heads, s), [self.heads, s * s]),
            &device,
        );
        for block in &self.blocks {
            t = block.forward(t, rel_idx.clone());
        }
        let pooled = self.final_norm.forward(t).mean_dim(1).squeeze_dim::<2>(1);
        self.pool_proj.forward(pooled)
            + self.act_proj.forward(c.action.clone())
            + self.flag_proj.forward(c.flags.clone())
    }

    pub fn param_breakdown(&self) -> Vec<(&'static str, usize)> {
        vec![
            (
                "evidence.input",
                self.in_proj.num_params() + self.sq_gain.num_params(),
            ),
            ("evidence.blocks", self.blocks.num_params()),
            ("evidence.norm", self.final_norm.num_params()),
            (
                "evidence.message_heads",
                self.pool_proj.num_params()
                    + self.act_proj.num_params()
                    + self.flag_proj.num_params(),
            ),
        ]
    }
}
