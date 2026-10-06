//! Exact V5 returned-encoder geometry, independently owned by V6.
use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::prelude::*;
use burn::tensor::{Bool, Distribution, Int, TensorData, activation};
use recur64_v5::config::V5Config;
use recur64_v5::{IN_FEATURES, PAYLOAD_FLAGS, SQUARES};
const REL_BUCKETS: usize = 225;
pub(crate) fn attention_softmax<B: Backend, const N: usize>(
    x: Tensor<B, N>,
    dim: usize,
) -> Tensor<B, N> {
    let m = x.clone().detach().max_dim(dim);
    let e = (x - m).exp();
    e.clone() / e.sum_dim(dim)
}
pub(crate) fn rows<B: Backend>(linear: &Linear<B>, x: Tensor<B, 3>) -> Tensor<B, 3> {
    let [b, s, d] = x.dims();
    let y = linear.forward(x.reshape([b * s, d]));
    let o = y.dims()[1];
    y.reshape([b, s, o])
}

pub(crate) fn rel_index_data(heads: usize) -> Vec<i32> {
    let mut out = Vec::with_capacity(heads * SQUARES * SQUARES);
    for _ in 0..heads {
        for i in 0..SQUARES {
            for j in 0..SQUARES {
                let (ir, iff) = (i / 8, i % 8);
                let (jr, jf) = (j / 8, j % 8);
                out.push(((jr as i32 - ir as i32 + 7) * 15) + jf as i32 - iff as i32 + 7);
            }
        }
    }
    out
}

#[derive(Module, Debug)]
struct SquareBias<B: Backend> {
    table: Param<Tensor<B, 2>>,
}

impl<B: Backend> SquareBias<B> {
    fn new(heads: usize, device: &B::Device) -> Self {
        Self {
            table: Param::from_tensor(Tensor::random(
                [heads, REL_BUCKETS],
                Distribution::Normal(0.0, 0.02),
                device,
            )),
        }
    }

    fn forward(&self, idx: Tensor<B, 2, Int>, heads: usize) -> Tensor<B, 4> {
        self.table
            .val()
            .gather(1, idx)
            .reshape([heads, SQUARES, SQUARES])
            .unsqueeze_dim::<4>(0)
    }
}

#[derive(Module, Debug)]
pub(crate) struct SquareBlock<B: Backend> {
    norm1: RmsNorm<B>,
    q: Linear<B>,
    k: Linear<B>,
    v: Linear<B>,
    o: Linear<B>,
    rel: SquareBias<B>,
    norm2: RmsNorm<B>,
    f1: Linear<B>,
    f2: Linear<B>,
    heads: usize,
    head_dim: usize,
}

impl<B: Backend> SquareBlock<B> {
    pub fn new(cfg: &V5Config, device: &B::Device) -> Self {
        let (d, h) = (cfg.width, cfg.heads);
        Self {
            norm1: RmsNormConfig::new(d).with_epsilon(cfg.rms_eps).init(device),
            q: LinearConfig::new(d, d).with_bias(true).init(device),
            k: LinearConfig::new(d, d).with_bias(true).init(device),
            v: LinearConfig::new(d, d).with_bias(true).init(device),
            o: LinearConfig::new(d, d).with_bias(true).init(device),
            rel: SquareBias::new(h, device),
            norm2: RmsNormConfig::new(d).with_epsilon(cfg.rms_eps).init(device),
            f1: LinearConfig::new(d, cfg.ffn).with_bias(true).init(device),
            f2: LinearConfig::new(cfg.ffn, d).with_bias(true).init(device),
            heads: h,
            head_dim: d / h,
        }
    }

    pub(crate) fn forward(&self, x: Tensor<B, 3>, rel_idx: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [b, s, d] = x.dims();
        let (h, hd) = (self.heads, self.head_dim);
        let n = self.norm1.forward(x.clone());
        let split = |t: Tensor<B, 3>| t.reshape([b, s, h, hd]).swap_dims(1, 2);
        let q = split(rows(&self.q, n.clone()));
        let k = split(rows(&self.k, n.clone()));
        let v = split(rows(&self.v, n));
        let logits = q
            .matmul(k.swap_dims(2, 3))
            .mul_scalar(1.0 / (hd as f32).sqrt())
            + self.rel.forward(rel_idx, h);
        let a = attention_softmax(logits, 3)
            .matmul(v)
            .swap_dims(1, 2)
            .reshape([b, s, d]);
        let x = x + rows(&self.o, a);
        let f = rows(
            &self.f2,
            activation::gelu(rows(&self.f1, self.norm2.forward(x.clone()))),
        );
        x + f
    }
}

#[derive(Module, Debug)]
pub struct StateEncoder<B: Backend> {
    input: Linear<B>,
    square: Param<Tensor<B, 2>>,
    blocks: Vec<SquareBlock<B>>,
    final_norm: RmsNorm<B>,
    slots: Param<Tensor<B, 2>>,
    q: Linear<B>,
    k: Linear<B>,
    v: Linear<B>,
    o: Linear<B>,
    flags: Linear<B>,
    output_norm: RmsNorm<B>,
    cfg: V5Config,
}

impl<B: Backend> StateEncoder<B> {
    pub fn new(cfg: &V5Config, device: &B::Device) -> Self {
        let d = cfg.width;
        Self {
            input: LinearConfig::new(IN_FEATURES, d)
                .with_bias(true)
                .init(device),
            square: Param::from_tensor(Tensor::random(
                [SQUARES, d],
                Distribution::Normal(0.0, 0.02),
                device,
            )),
            blocks: (0..cfg.state_blocks)
                .map(|_| SquareBlock::new(cfg, device))
                .collect(),
            final_norm: RmsNormConfig::new(d).with_epsilon(cfg.rms_eps).init(device),
            slots: Param::from_tensor(Tensor::random(
                [cfg.state_slots, d],
                Distribution::Normal(0.0, 0.02),
                device,
            )),
            q: LinearConfig::new(d, d).with_bias(true).init(device),
            k: LinearConfig::new(d, d).with_bias(true).init(device),
            v: LinearConfig::new(d, d).with_bias(true).init(device),
            o: LinearConfig::new(d, d).with_bias(true).init(device),
            flags: LinearConfig::new(PAYLOAD_FLAGS, d)
                .with_bias(true)
                .init(device),
            output_norm: RmsNormConfig::new(d).with_epsilon(cfg.rms_eps).init(device),
            cfg: cfg.clone(),
        }
    }

    pub fn forward(
        &self,
        states: Tensor<B, 4>,
        flags: Tensor<B, 3>,
        node_mask: Tensor<B, 2, Bool>,
    ) -> Tensor<B, 4> {
        let [b, qn, _, _] = states.dims();
        let n = b * qn;
        let d = self.cfg.width;
        let h = self.cfg.heads;
        let hd = d / h;
        let device = states.device();
        let mut x = rows(&self.input, states.reshape([n, SQUARES, IN_FEATURES]))
            + self
                .square
                .val()
                .reshape([1, SQUARES, d])
                .expand([n, SQUARES, d]);
        let rel = Tensor::from_data(
            TensorData::new(rel_index_data(h), [h, SQUARES * SQUARES]),
            &device,
        );
        for block in &self.blocks {
            x = block.forward(x, rel.clone());
        }
        x = self.final_norm.forward(x);
        let slots = self
            .slots
            .val()
            .reshape([1, self.cfg.state_slots, d])
            .expand([n, self.cfg.state_slots, d]);
        let split_q = |t: Tensor<B, 3>| t.reshape([n, self.cfg.state_slots, h, hd]).swap_dims(1, 2);
        let split_s = |t: Tensor<B, 3>| t.reshape([n, SQUARES, h, hd]).swap_dims(1, 2);
        let qq = split_q(rows(&self.q, slots));
        let kk = split_s(rows(&self.k, x.clone()));
        let vv = split_s(rows(&self.v, x));
        let pooled = attention_softmax(
            qq.matmul(kk.swap_dims(2, 3))
                .mul_scalar(1.0 / (hd as f32).sqrt()),
            3,
        )
        .matmul(vv)
        .swap_dims(1, 2)
        .reshape([n, self.cfg.state_slots, d]);
        let flag = self
            .flags
            .forward(flags.reshape([n, PAYLOAD_FLAGS]))
            .unsqueeze_dim::<3>(1)
            .expand([n, self.cfg.state_slots, d]);
        let out = self
            .output_norm
            .forward(rows(&self.o, pooled) + flag)
            .reshape([b, qn, self.cfg.state_slots, d]);
        mask_returned_payload(out, node_mask)
    }
}

fn mask_returned_payload<B: Backend>(out: Tensor<B, 4>, mask: Tensor<B, 2, Bool>) -> Tensor<B, 4> {
    let [b, q, s, d] = out.dims();
    out * mask.float().reshape([b, q, 1, 1]).expand([b, q, s, d])
}
