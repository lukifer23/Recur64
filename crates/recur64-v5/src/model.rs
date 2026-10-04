//! The V5 root tower, returned-state encoder and paired shared relational loop.

use burn::module::{Module, ModuleVisitor, Param};
use burn::nn::{Initializer, Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::prelude::*;
use burn::tensor::{Bool, Distribution, Int, TensorData, activation};
use recur64_core::{ActionId, GameState, candidate_facts, encode_root_relative_observation_v1};
use recur64_model::action::CandidateBatch;
use recur64_model::candidate::facts_tensor;
use recur64_model::model::CandidateTensors;
use sha2::{Digest, Sha256};

use crate::config::V5Config;
use crate::graph::{AcquiredGraph, AcquiredNode};
use crate::{
    ACTION_GEOMETRY, FACT_FIELDS, IN_FEATURES, MASKED_LOGIT, PAYLOAD_FLAGS, SQUARES,
    STRUCTURAL_FEATURES,
};

const REL_BUCKETS: usize = 225;
const RELATIONS: usize = 14;
const REL_DEFAULT: usize = 0;
const REL_SAME_NODE: usize = 1;
const REL_PARENT: usize = 2;
const REL_CHILD: usize = 3;
const REL_SIBLING: usize = 4;
const REL_SAME_BRANCH: usize = 5;
const REL_DIFFERENT_BRANCH: usize = 6;
const REL_E_OWNER_H: usize = 7;
const REL_E_OTHER_H: usize = 8;
const REL_ROOT_CONTEXT: usize = 9;
const REL_H_SELF: usize = 10;
const REL_H_OTHER: usize = 11;
const REL_H_OWNED_E: usize = 12;
const REL_H_OTHER_E: usize = 13;

#[cfg(test)]
std::thread_local! {
    static LEGACY_SOFTMAX: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Test-only parity reference: the real reader with the former Burn dispatch.
/// No alternate execution switch exists in production.
#[cfg(test)]
pub(crate) fn with_legacy_softmax<T>(f: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            LEGACY_SOFTMAX.set(self.0);
        }
    }
    let _restore = Restore(LEGACY_SOFTMAX.replace(true));
    f()
}

/// The exact pinned Burn 0.21 default softmax equation, on both backends.
/// Only the numerical max-shift is detached, as in the framework default;
/// neither stream nor any recurrent state/iteration is detached.
fn attention_softmax<B: Backend, const N: usize>(x: Tensor<B, N>, dim: usize) -> Tensor<B, N> {
    #[cfg(test)]
    if LEGACY_SOFTMAX.get() {
        return activation::softmax(x, dim);
    }
    let max = x.clone().detach().max_dim(dim);
    let exp = (x - max).exp();
    exp.clone() / exp.sum_dim(dim)
}

fn rows<B: Backend>(linear: &Linear<B>, x: Tensor<B, 3>) -> Tensor<B, 3> {
    let [b, s, d] = x.dims();
    let y = linear.forward(x.reshape([b * s, d]));
    let o = y.dims()[1];
    y.reshape([b, s, o])
}

fn rel_index_data(heads: usize) -> Vec<i32> {
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
struct SquareBlock<B: Backend> {
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
    fn new(cfg: &V5Config, device: &B::Device) -> Self {
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

    fn forward(&self, x: Tensor<B, 3>, rel_idx: Tensor<B, 2, Int>) -> Tensor<B, 3> {
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
struct MaskedBlock<B: Backend> {
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

impl<B: Backend> MaskedBlock<B> {
    fn new(cfg: &V5Config, device: &B::Device) -> Self {
        let (d, h) = (cfg.width, cfg.heads);
        Self {
            norm1: RmsNormConfig::new(d).with_epsilon(cfg.rms_eps).init(device),
            q: LinearConfig::new(d, d).with_bias(true).init(device),
            k: LinearConfig::new(d, d).with_bias(true).init(device),
            v: LinearConfig::new(d, d).with_bias(true).init(device),
            o: LinearConfig::new(d, d).with_bias(true).init(device),
            norm2: RmsNormConfig::new(d).with_epsilon(cfg.rms_eps).init(device),
            f1: LinearConfig::new(d, cfg.ffn).with_bias(true).init(device),
            f2: LinearConfig::new(cfg.ffn, d).with_bias(true).init(device),
            heads: h,
            head_dim: d / h,
        }
    }

    fn forward(&self, x: Tensor<B, 3>, valid: Tensor<B, 2, Bool>) -> Tensor<B, 3> {
        let [b, s, d] = x.dims();
        let (h, hd) = (self.heads, self.head_dim);
        let n = self.norm1.forward(x.clone());
        let split = |t: Tensor<B, 3>| t.reshape([b, s, h, hd]).swap_dims(1, 2);
        let q = split(rows(&self.q, n.clone()));
        let k = split(rows(&self.k, n.clone()));
        let v = split(rows(&self.v, n));
        let logits = q
            .matmul(k.swap_dims(2, 3))
            .mul_scalar(1.0 / (hd as f32).sqrt());
        let pad = valid
            .clone()
            .bool_not()
            .unsqueeze_dim::<3>(1)
            .unsqueeze_dim::<4>(1)
            .expand([b, h, s, s]);
        let a = attention_softmax(logits.mask_fill(pad, MASKED_LOGIT), 3)
            .matmul(v)
            .swap_dims(1, 2)
            .reshape([b, s, d]);
        let x = x + rows(&self.o, a);
        let f = rows(
            &self.f2,
            activation::gelu(rows(&self.f1, self.norm2.forward(x.clone()))),
        );
        (x + f).mask_fill(
            valid.bool_not().unsqueeze_dim::<3>(2).expand([b, s, d]),
            0.0,
        )
    }
}

#[derive(Module, Debug)]
struct RootTower<B: Backend> {
    input: Linear<B>,
    square: Param<Tensor<B, 2>>,
    blocks: Vec<SquareBlock<B>>,
    final_norm: RmsNorm<B>,
    candidate_input: Linear<B>,
    candidate_norm: RmsNorm<B>,
    candidate_blocks: Vec<MaskedBlock<B>>,
    baseline_hidden: Linear<B>,
    baseline_out: Linear<B>,
    cfg: V5Config,
}

pub struct BaseOutput<B: Backend> {
    pub context: Tensor<B, 3>,
    pub pooled: Tensor<B, 2>,
    pub hypotheses: Tensor<B, 3>,
    pub z0: Tensor<B, 2>,
}

impl<B: Backend> RootTower<B> {
    fn new(cfg: &V5Config, device: &B::Device) -> Self {
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
            blocks: (0..cfg.root_blocks)
                .map(|_| SquareBlock::new(cfg, device))
                .collect(),
            final_norm: RmsNormConfig::new(d).with_epsilon(cfg.rms_eps).init(device),
            candidate_input: LinearConfig::new(3 * d + ACTION_GEOMETRY + FACT_FIELDS, d)
                .with_bias(true)
                .init(device),
            candidate_norm: RmsNormConfig::new(d).with_epsilon(cfg.rms_eps).init(device),
            candidate_blocks: (0..cfg.candidate_blocks)
                .map(|_| MaskedBlock::new(cfg, device))
                .collect(),
            baseline_hidden: LinearConfig::new(2 * d, cfg.readout_hidden)
                .with_bias(true)
                .init(device),
            baseline_out: LinearConfig::new(cfg.readout_hidden, 1)
                .with_bias(false)
                .with_initializer(Initializer::Normal {
                    mean: 0.0,
                    std: 0.01,
                })
                .init(device),
            cfg: cfg.clone(),
        }
    }

    fn rel_idx(&self, device: &B::Device) -> Tensor<B, 2, Int> {
        Tensor::from_data(
            TensorData::new(
                rel_index_data(self.cfg.heads),
                [self.cfg.heads, SQUARES * SQUARES],
            ),
            device,
        )
    }

    fn forward(
        &self,
        board: Tensor<B, 3>,
        cands: &CandidateTensors<B>,
        geom: Tensor<B, 3>,
        facts: Tensor<B, 3>,
    ) -> BaseOutput<B> {
        let device = board.device();
        let [b, _, _] = board.dims();
        let w = cands.width;
        let d = self.cfg.width;
        let mut c = rows(&self.input, board)
            + self
                .square
                .val()
                .reshape([1, SQUARES, d])
                .expand([b, SQUARES, d]);
        let rel = self.rel_idx(&device);
        for block in &self.blocks {
            c = block.forward(c, rel.clone());
        }
        c = self.final_norm.forward(c);
        let pooled = c.clone().mean_dim(1).squeeze_dim::<2>(1);
        let gather = |idx: &Tensor<B, 2, Int>| {
            c.clone()
                .gather(1, idx.clone().unsqueeze_dim::<3>(2).expand([b, w, d]))
        };
        let from = gather(&cands.from_idx);
        let to = gather(&cands.to_idx);
        let global = pooled.clone().unsqueeze_dim::<3>(1).expand([b, w, d]);
        let mut h = rows(
            &self.candidate_input,
            Tensor::cat(vec![from, to, global, geom, facts], 2),
        );
        h = self.candidate_norm.forward(h);
        for block in &self.candidate_blocks {
            h = block.forward(h, cands.mask.clone());
        }
        let ctx = pooled.clone().unsqueeze_dim::<3>(1).expand([b, w, d]);
        let hidden = activation::gelu(rows(
            &self.baseline_hidden,
            Tensor::cat(vec![h.clone(), ctx], 2),
        ));
        let z0 = rows(&self.baseline_out, hidden)
            .squeeze_dim::<2>(2)
            .mask_fill(cands.mask.clone().bool_not(), MASKED_LOGIT);
        BaseOutput {
            context: c,
            pooled,
            hypotheses: h,
            z0,
        }
    }
}

#[derive(Module, Debug)]
struct StateEncoder<B: Backend> {
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
    fn new(cfg: &V5Config, device: &B::Device) -> Self {
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

    fn forward(
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
        out.mask_fill(
            node_mask
                .bool_not()
                .unsqueeze_dim::<3>(2)
                .unsqueeze_dim::<4>(3)
                .expand([b, qn, self.cfg.state_slots, d]),
            0.0,
        )
    }
}

#[derive(Module, Debug)]
struct RelationalBlock<B: Backend> {
    state_norm: RmsNorm<B>,
    recall: Linear<B>,
    memory_norm: RmsNorm<B>,
    q: Linear<B>,
    k: Linear<B>,
    v: Linear<B>,
    o: Linear<B>,
    relation: Param<Tensor<B, 2>>,
    mid_norm: RmsNorm<B>,
    f1: Linear<B>,
    f2: Linear<B>,
    out_norm: RmsNorm<B>,
    heads: usize,
    head_dim: usize,
    alpha: f32,
}

impl<B: Backend> RelationalBlock<B> {
    fn new(cfg: &V5Config, device: &B::Device) -> Self {
        let d = cfg.width;
        Self {
            state_norm: RmsNormConfig::new(d).with_epsilon(cfg.rms_eps).init(device),
            recall: LinearConfig::new(2 * d, d).with_bias(true).init(device),
            memory_norm: RmsNormConfig::new(d).with_epsilon(cfg.rms_eps).init(device),
            q: LinearConfig::new(d, d).with_bias(true).init(device),
            k: LinearConfig::new(d, d).with_bias(true).init(device),
            v: LinearConfig::new(d, d).with_bias(true).init(device),
            o: LinearConfig::new(d, d).with_bias(true).init(device),
            relation: Param::from_tensor(Tensor::random(
                [cfg.heads, RELATIONS],
                Distribution::Normal(0.0, 0.02),
                device,
            )),
            mid_norm: RmsNormConfig::new(d).with_epsilon(cfg.rms_eps).init(device),
            f1: LinearConfig::new(d, cfg.ffn).with_bias(true).init(device),
            f2: LinearConfig::new(cfg.ffn, d).with_bias(true).init(device),
            out_norm: RmsNormConfig::new(d).with_epsilon(cfg.rms_eps).init(device),
            heads: cfg.heads,
            head_dim: d / cfg.heads,
            alpha: cfg.residual_alpha as f32,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_inner(
        &self,
        state: Tensor<B, 3>,
        anchor: Tensor<B, 3>,
        memory: Tensor<B, 3>,
        query_valid: Tensor<B, 2, Bool>,
        memory_valid: Tensor<B, 2, Bool>,
        relation_onehot: Tensor<B, 4>,
        remove_relations: bool,
        trace_attention: bool,
    ) -> (Tensor<B, 3>, Option<Tensor<B, 4>>) {
        let [b, qs, d] = state.dims();
        let ms = memory.dims()[1];
        let (h, hd) = (self.heads, self.head_dim);
        let u = rows(
            &self.recall,
            Tensor::cat(vec![self.state_norm.forward(state.clone()), anchor], 2),
        );
        let mem = self.memory_norm.forward(memory);
        let q = rows(&self.q, u).reshape([b, qs, h, hd]).swap_dims(1, 2);
        let k = rows(&self.k, mem.clone())
            .reshape([b, ms, h, hd])
            .swap_dims(1, 2);
        let v = rows(&self.v, mem).reshape([b, ms, h, hd]).swap_dims(1, 2);
        let mut logits = q
            .matmul(k.swap_dims(2, 3))
            .mul_scalar(1.0 / (hd as f32).sqrt());
        if !remove_relations {
            let rb = relation_onehot
                .reshape([b * qs * ms, RELATIONS])
                .matmul(self.relation.val().swap_dims(0, 1))
                .reshape([b, qs, ms, h])
                .swap_dims(2, 3)
                .swap_dims(1, 2);
            logits = logits + rb;
        }
        let pad = memory_valid
            .bool_not()
            .unsqueeze_dim::<3>(1)
            .unsqueeze_dim::<4>(1)
            .expand([b, h, qs, ms]);
        let weights = attention_softmax(logits.mask_fill(pad, MASKED_LOGIT), 3);
        let traced = trace_attention.then(|| weights.clone());
        let a = weights.matmul(v).swap_dims(1, 2).reshape([b, qs, d]);
        let t = state + rows(&self.o, a).mul_scalar(self.alpha);
        let f = rows(
            &self.f2,
            activation::gelu(rows(&self.f1, self.mid_norm.forward(t.clone()))),
        );
        (
            self.out_norm
                .forward(t + f.mul_scalar(self.alpha))
                .mask_fill(
                    query_valid
                        .bool_not()
                        .unsqueeze_dim::<3>(2)
                        .expand([b, qs, d]),
                    0.0,
                ),
            traced,
        )
    }
}

#[derive(Module, Debug)]
pub struct CounterfactualRelationalLoop<B: Backend> {
    root: RootTower<B>,
    state: StateEncoder<B>,
    hypothesis_init: Linear<B>,
    evidence_init: Linear<B>,
    evidence: RelationalBlock<B>,
    hypothesis: RelationalBlock<B>,
    correction_hidden: Linear<B>,
    correction_out: Linear<B>,
    cfg: V5Config,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Treatment {
    Normal,
    AllPayloadNull,
    NoRelationBias,
    NoHypothesisFeedback,
}

pub struct PairedOutput<B: Backend> {
    pub logits: Tensor<B, 2>,
    pub z0: Tensor<B, 2>,
    pub raw_delta: Tensor<B, 2>,
    pub centered_delta: Tensor<B, 2>,
    pub factual_h: Tensor<B, 3>,
    pub null_h: Tensor<B, 3>,
}

pub struct StreamLoopTensorTrace<B: Backend> {
    pub evidence_before: Tensor<B, 3>,
    pub evidence_after: Tensor<B, 3>,
    pub hypothesis_before: Tensor<B, 3>,
    pub hypothesis_after: Tensor<B, 3>,
    pub evidence_attention: Tensor<B, 4>,
    pub hypothesis_attention: Tensor<B, 4>,
}

pub struct PairedTracedOutput<B: Backend> {
    pub output: PairedOutput<B>,
    pub factual: Vec<StreamLoopTensorTrace<B>>,
    pub null: Vec<StreamLoopTensorTrace<B>>,
}

impl<B: Backend> CounterfactualRelationalLoop<B> {
    pub fn new(cfg: V5Config, device: &B::Device) -> Self {
        cfg.validate().expect("valid frozen V5 config");
        let d = cfg.width;
        let model = Self {
            root: RootTower::new(&cfg, device),
            state: StateEncoder::new(&cfg, device),
            hypothesis_init: LinearConfig::new(d, d).with_bias(true).init(device),
            evidence_init: LinearConfig::new(2 * d + STRUCTURAL_FEATURES, d)
                .with_bias(true)
                .init(device),
            evidence: RelationalBlock::new(&cfg, device),
            hypothesis: RelationalBlock::new(&cfg, device),
            correction_hidden: LinearConfig::new(d, cfg.readout_hidden)
                .with_bias(true)
                .init(device),
            correction_out: LinearConfig::new(cfg.readout_hidden, 1)
                .with_bias(false)
                .with_initializer(Initializer::Normal {
                    mean: 0.0,
                    std: 0.01,
                })
                .init(device),
            cfg,
        };
        model.force_init();
        model
    }

    pub fn force_init(&self) {
        struct Init;
        impl<B: Backend> ModuleVisitor<B> for Init {
            fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<B, D>>) {
                let _ = p.val();
            }
            fn visit_int<const D: usize>(&mut self, p: &Param<Tensor<B, D, Int>>) {
                let _ = p.val();
            }
            fn visit_bool<const D: usize>(&mut self, p: &Param<Tensor<B, D, Bool>>) {
                let _ = p.val();
            }
        }
        self.visit(&mut Init);
    }

    pub fn config(&self) -> &V5Config {
        &self.cfg
    }
    pub fn num_params(&self) -> usize {
        Module::num_params(self)
    }
    /// Content identity of every FP32 parameter, independent of generated
    /// ParamIds and recorder map ordering. Used for exact resume verification.
    pub fn parameter_digest(&self) -> anyhow::Result<String> {
        self.parameter_digest_scope(false)
    }
    pub fn baseline_parameter_digest(&self) -> anyhow::Result<String> {
        self.parameter_digest_scope(true)
    }
    fn parameter_digest_scope(&self, baseline_only: bool) -> anyhow::Result<String> {
        struct Visitor {
            hash: Sha256,
            error: Option<anyhow::Error>,
        }
        impl<B: Backend> ModuleVisitor<B> for Visitor {
            fn visit_float<const N: usize>(&mut self, p: &Param<Tensor<B, N>>) {
                if self.error.is_some() {
                    return;
                }
                let tensor = p.val();
                self.hash.update((N as u64).to_le_bytes());
                for dim in tensor.dims() {
                    self.hash.update((dim as u64).to_le_bytes());
                }
                match tensor.into_data().to_vec::<f32>() {
                    Ok(values) => {
                        for value in values {
                            self.hash.update(value.to_bits().to_le_bytes());
                        }
                    }
                    Err(error) => self.error = Some(anyhow::anyhow!("{error:?}")),
                }
            }
        }
        let mut visitor = Visitor {
            hash: Sha256::new(),
            error: None,
        };
        if baseline_only {
            self.root.visit(&mut visitor);
        } else {
            self.visit(&mut visitor);
        }
        if let Some(error) = visitor.error {
            return Err(error);
        }
        Ok(format!("{:x}", visitor.hash.finalize()))
    }
    pub fn param_breakdown(&self) -> Vec<(&'static str, usize)> {
        vec![
            ("root", self.root.num_params()),
            ("state_encoder", self.state.num_params()),
            ("hypothesis_adapter", self.hypothesis_init.num_params()),
            ("evidence_initializer", self.evidence_init.num_params()),
            ("evidence_block", self.evidence.num_params()),
            ("hypothesis_block", self.hypothesis.num_params()),
            (
                "correction_readout",
                self.correction_hidden.num_params() + self.correction_out.num_params(),
            ),
        ]
    }

    pub fn base(&self, input: &V5Inputs<B>) -> BaseOutput<B> {
        self.base_root(&RootInputs {
            batch: input.batch,
            root: input.root.clone(),
            cands: CandidateTensors {
                base_idx: input.cands.base_idx.clone(),
                from_idx: input.cands.from_idx.clone(),
                to_idx: input.cands.to_idx.clone(),
                promo_idx: input.cands.promo_idx.clone(),
                promo_code: input.cands.promo_code.clone(),
                mask: input.cands.mask.clone(),
                valid: input.cands.valid.clone(),
                width: input.cands.width,
            },
            candidate_geometry: input.candidate_geometry.clone(),
            facts: input.facts.clone(),
        })
    }

    /// Q0 path: exactly one root/candidate execution and no state encoder or
    /// relational-loop execution.
    pub fn base_root(&self, input: &RootInputs<B>) -> BaseOutput<B> {
        self.root.forward(
            input.root.clone(),
            &input.cands,
            input.candidate_geometry.clone(),
            input.facts.clone(),
        )
    }

    pub fn paired(
        &self,
        input: &V5Inputs<B>,
        loops: usize,
        treatment: Treatment,
    ) -> PairedOutput<B> {
        assert!(loops > 0, "R must be positive for reader execution");
        let base = self.base(input);
        self.paired_with_base(input, base, loops, treatment)
    }

    /// Execute the reader from an explicitly supplied base result.
    ///
    /// Stage B supplies graph-free base tensors here, so no frozen baseline
    /// activation graph is ever constructed.
    pub fn paired_with_base(
        &self,
        input: &V5Inputs<B>,
        base: BaseOutput<B>,
        loops: usize,
        treatment: Treatment,
    ) -> PairedOutput<B> {
        self.paired_with_base_payload_mask(input, base, loops, treatment, None)
    }

    /// Execute the matched streams with an optional factual payload-token mask.
    ///
    /// The mask is applied after the returned-state encoder, at the declared
    /// payload-anchor boundary. This makes partial composition interventions an
    /// exact zero intervention even though the encoder contains biases.
    pub fn paired_with_base_payload_mask(
        &self,
        input: &V5Inputs<B>,
        base: BaseOutput<B>,
        loops: usize,
        treatment: Treatment,
        factual_payload_mask: Option<Tensor<B, 2, Bool>>,
    ) -> PairedOutput<B> {
        self.paired_profiled(
            input,
            base,
            loops,
            treatment,
            factual_payload_mask,
            &mut |_| {},
        )
    }

    /// The production paired computation with optional completion observers.
    /// Observers receive names only, never tensors or autodiff state.
    pub(crate) fn paired_profiled(
        &self,
        input: &V5Inputs<B>,
        base: BaseOutput<B>,
        loops: usize,
        treatment: Treatment,
        factual_payload_mask: Option<Tensor<B, 2, Bool>>,
        phase: &mut dyn FnMut(&str),
    ) -> PairedOutput<B> {
        assert!(loops > 0, "R must be positive for reader execution");
        let mut factual_x = self
            .state
            .forward(
                input.states.clone(),
                input.flags.clone(),
                input.node_mask.clone(),
            )
            .reshape([input.batch, input.evidence_tokens, self.cfg.width]);
        phase("returned_state_encoder");
        if let Some(mask) = factual_payload_mask {
            assert_eq!(
                mask.dims(),
                [input.batch, input.evidence_tokens],
                "payload mask shape mismatch"
            );
            factual_x = factual_x.mask_fill(
                mask.bool_not().unsqueeze_dim::<3>(2).expand([
                    input.batch,
                    input.evidence_tokens,
                    self.cfg.width,
                ]),
                0.0,
            );
        }
        let null_x = factual_x.clone().zeros_like();
        let factual_anchor = if treatment == Treatment::AllPayloadNull {
            null_x.clone()
        } else {
            factual_x
        };
        phase("payload_boundary");
        let factual_h = self
            .run_stream_inner(
                &base,
                input,
                factual_anchor,
                loops,
                treatment,
                false,
                "factual",
                phase,
            )
            .0;
        let null_h = self
            .run_stream_inner(&base, input, null_x, loops, treatment, false, "null", phase)
            .0;
        let read = |h: Tensor<B, 3>| {
            rows(
                &self.correction_out,
                activation::gelu(rows(&self.correction_hidden, h)),
            )
            .squeeze_dim::<2>(2)
        };
        let raw = read(factual_h.clone()) - read(null_h.clone());
        // FP32 is unchanged; match dtype for the test-only FP64 reference.
        let valid = input
            .cands
            .mask
            .clone()
            .float()
            .cast(burn::tensor::FloatDType::from(raw.dtype()));
        let count = valid.clone().sum_dim(1).clamp(1.0, f32::MAX);
        let mean = (raw.clone() * valid).sum_dim(1) / count;
        let centered = (raw.clone() - mean.expand([input.batch, input.cands.width]))
            .mask_fill(input.cands.mask.clone().bool_not(), 0.0);
        let logits = base.z0.clone() + centered.clone();
        phase("paired_readout_centering");
        PairedOutput {
            logits,
            z0: base.z0,
            raw_delta: raw,
            centered_delta: centered,
            factual_h,
            null_h,
        }
    }

    /// Evaluation-only traced execution of the same paired computation.
    pub fn paired_traced_with_base_payload_mask(
        &self,
        input: &V5Inputs<B>,
        base: BaseOutput<B>,
        loops: usize,
        treatment: Treatment,
        factual_payload_mask: Option<Tensor<B, 2, Bool>>,
    ) -> PairedTracedOutput<B> {
        assert!(loops > 0, "R must be positive for reader execution");
        let mut factual_x = self
            .state
            .forward(
                input.states.clone(),
                input.flags.clone(),
                input.node_mask.clone(),
            )
            .reshape([input.batch, input.evidence_tokens, self.cfg.width]);
        if let Some(mask) = factual_payload_mask {
            assert_eq!(mask.dims(), [input.batch, input.evidence_tokens]);
            factual_x = factual_x.mask_fill(
                mask.bool_not().unsqueeze_dim::<3>(2).expand([
                    input.batch,
                    input.evidence_tokens,
                    self.cfg.width,
                ]),
                0.0,
            );
        }
        let null_x = factual_x.clone().zeros_like();
        let factual_anchor = if treatment == Treatment::AllPayloadNull {
            null_x.clone()
        } else {
            factual_x
        };
        let (factual_h, factual) = self.run_stream_inner(
            &base,
            input,
            factual_anchor,
            loops,
            treatment,
            true,
            "factual",
            &mut |_| {},
        );
        let (null_h, null) = self.run_stream_inner(
            &base,
            input,
            null_x,
            loops,
            treatment,
            true,
            "null",
            &mut |_| {},
        );
        let read = |h: Tensor<B, 3>| {
            rows(
                &self.correction_out,
                activation::gelu(rows(&self.correction_hidden, h)),
            )
            .squeeze_dim::<2>(2)
        };
        let raw = read(factual_h.clone()) - read(null_h.clone());
        let valid = input
            .cands
            .mask
            .clone()
            .float()
            .cast(burn::tensor::FloatDType::from(raw.dtype()));
        let count = valid.clone().sum_dim(1).clamp(1.0, f32::MAX);
        let mean = (raw.clone() * valid).sum_dim(1) / count;
        let centered = (raw.clone() - mean.expand([input.batch, input.cands.width]))
            .mask_fill(input.cands.mask.clone().bool_not(), 0.0);
        PairedTracedOutput {
            output: PairedOutput {
                logits: base.z0.clone() + centered.clone(),
                z0: base.z0,
                raw_delta: raw,
                centered_delta: centered,
                factual_h,
                null_h,
            },
            factual,
            null,
        }
    }

    #[allow(clippy::too_many_arguments)] // One shared execution path for normal, traced and profiled reads.
    fn run_stream_inner(
        &self,
        base: &BaseOutput<B>,
        input: &V5Inputs<B>,
        anchor: Tensor<B, 3>,
        loops: usize,
        treatment: Treatment,
        trace: bool,
        stream: &str,
        phase: &mut dyn FnMut(&str),
    ) -> (Tensor<B, 3>, Vec<StreamLoopTensorTrace<B>>) {
        let [b, _w, d] = base.hypotheses.dims();
        let e = input.evidence_tokens;
        let h0 = rows(&self.hypothesis_init, base.hypotheses.clone());
        let owner = base.hypotheses.clone().gather(
            1,
            input
                .owner_idx
                .clone()
                .unsqueeze_dim::<3>(2)
                .expand([b, e, d]),
        );
        let mut ev = rows(
            &self.evidence_init,
            Tensor::cat(vec![anchor.clone(), owner, input.structural.clone()], 2),
        );
        ev = ev.mask_fill(
            input
                .evidence_mask
                .clone()
                .bool_not()
                .unsqueeze_dim::<3>(2)
                .expand([b, e, d]),
            0.0,
        );
        let mut h = h0.clone();
        let mut traces = Vec::with_capacity(if trace { loops } else { 0 });
        let root_valid = Tensor::<B, 2, Bool>::ones([b, SQUARES], &base.context.device());
        phase(&format!("{stream}.initialize"));
        for iteration in 0..loops {
            let supplied_h = if treatment == Treatment::NoHypothesisFeedback {
                h0.clone()
            } else {
                h.clone()
            };
            let ev_mem = Tensor::cat(vec![ev.clone(), supplied_h, base.context.clone()], 1);
            let ev_valid = Tensor::cat(
                vec![
                    input.evidence_mask.clone(),
                    input.cands.mask.clone(),
                    root_valid.clone(),
                ],
                1,
            );
            let evidence_before = trace.then(|| ev.clone());
            let (next_ev, evidence_attention) = self.evidence.forward_inner(
                ev,
                anchor.clone(),
                ev_mem,
                input.evidence_mask.clone(),
                ev_valid,
                input.evidence_rel.clone(),
                treatment == Treatment::NoRelationBias,
                trace,
            );
            ev = next_ev;
            phase(&format!("{stream}.r{}.evidence", iteration + 1));
            let h_mem = Tensor::cat(vec![h.clone(), ev.clone(), base.context.clone()], 1);
            let h_valid = Tensor::cat(
                vec![
                    input.cands.mask.clone(),
                    input.evidence_mask.clone(),
                    root_valid.clone(),
                ],
                1,
            );
            let hypothesis_before = trace.then(|| h.clone());
            let (next_h, hypothesis_attention) = self.hypothesis.forward_inner(
                h,
                base.hypotheses.clone(),
                h_mem,
                input.cands.mask.clone(),
                h_valid,
                input.hypothesis_rel.clone(),
                treatment == Treatment::NoRelationBias,
                trace,
            );
            h = next_h;
            phase(&format!("{stream}.r{}.hypothesis", iteration + 1));
            if trace {
                traces.push(StreamLoopTensorTrace {
                    evidence_before: evidence_before.expect("trace evidence"),
                    evidence_after: ev.clone(),
                    hypothesis_before: hypothesis_before.expect("trace hypothesis"),
                    hypothesis_after: h.clone(),
                    evidence_attention: evidence_attention.expect("trace evidence attention"),
                    hypothesis_attention: hypothesis_attention.expect("trace hypothesis attention"),
                });
            }
        }
        (h, traces)
    }
}

impl<B: burn::tensor::backend::AutodiffBackend> CounterfactualRelationalLoop<B> {
    /// Compute the complete immutable baseline path on the graph-free inner
    /// backend and lift its outputs into autodiff as constants.
    pub fn base_frozen(
        &self,
        examples: &[(&GameState, &AcquiredGraph)],
        device: &B::Device,
    ) -> anyhow::Result<BaseOutput<B>> {
        self.base_frozen_profiled(examples, device, &mut |_| {})
    }

    pub(crate) fn base_frozen_profiled(
        &self,
        examples: &[(&GameState, &AcquiredGraph)],
        device: &B::Device,
        phase: &mut dyn FnMut(&str),
    ) -> anyhow::Result<BaseOutput<B>> {
        use burn::module::AutodiffModule;
        let inner = self.valid();
        phase("frozen_model_view");
        let inputs = V5Inputs::<B::InnerBackend>::from_examples_profiled(examples, device, phase)?;
        let base = inner.base(&inputs);
        phase("frozen_root_encoder_and_candidate_path");
        let lifted = BaseOutput {
            context: Tensor::from_inner(base.context),
            pooled: Tensor::from_inner(base.pooled),
            hypotheses: Tensor::from_inner(base.hypotheses),
            z0: Tensor::from_inner(base.z0),
        };
        phase("frozen_base_lift");
        Ok(lifted)
    }
}

pub struct V5Inputs<B: Backend> {
    pub batch: usize,
    pub evidence_tokens: usize,
    pub root: Tensor<B, 3>,
    pub cands: CandidateTensors<B>,
    pub candidate_geometry: Tensor<B, 3>,
    pub facts: Tensor<B, 3>,
    pub states: Tensor<B, 4>,
    pub flags: Tensor<B, 3>,
    pub node_mask: Tensor<B, 2, Bool>,
    pub evidence_mask: Tensor<B, 2, Bool>,
    pub structural: Tensor<B, 3>,
    pub owner_idx: Tensor<B, 2, Int>,
    pub evidence_rel: Tensor<B, 4>,
    pub hypothesis_rel: Tensor<B, 4>,
}

pub struct RootInputs<B: Backend> {
    pub batch: usize,
    pub root: Tensor<B, 3>,
    pub cands: CandidateTensors<B>,
    pub candidate_geometry: Tensor<B, 3>,
    pub facts: Tensor<B, 3>,
}

fn root_geom(id: ActionId) -> [f32; ACTION_GEOMETRY] {
    let (from, to, promo) = id.decode();
    let (ff, fr) = ((from as usize % 8) as f32, (from as usize / 8) as f32);
    let (tf, tr) = ((to as usize % 8) as f32, (to as usize / 8) as f32);
    let mut g = [0.0; ACTION_GEOMETRY];
    g[0] = ff / 7.0;
    g[1] = fr / 7.0;
    g[2] = tf / 7.0;
    g[3] = tr / 7.0;
    g[4] = (tf - ff) / 7.0;
    g[5] = (tr - fr) / 7.0;
    g[6 + promo.code() as usize] = 1.0;
    g
}

fn set_relation(buf: &mut [f32], b: usize, q: usize, m: usize, qs: usize, ms: usize, rel: usize) {
    buf[(((b * qs + q) * ms + m) * RELATIONS) + rel] = 1.0;
}

fn ev_relation(a: &AcquiredNode, b: &AcquiredNode) -> usize {
    if a.storage_id == b.storage_id {
        REL_SAME_NODE
    } else if a.parent == Some(b.storage_id) {
        REL_PARENT
    } else if b.parent == Some(a.storage_id) {
        REL_CHILD
    } else if a.parent == b.parent {
        REL_SIBLING
    } else if a.root_candidate == b.root_candidate {
        REL_SAME_BRANCH
    } else {
        REL_DIFFERENT_BRANCH
    }
}

impl<B: Backend> RootInputs<B> {
    pub fn from_roots(roots: &[&GameState], device: &B::Device) -> anyhow::Result<Self> {
        Self::from_roots_profiled(roots, device, &mut |_| {})
    }

    pub(crate) fn from_roots_profiled(
        roots: &[&GameState],
        device: &B::Device,
        phase: &mut dyn FnMut(&str),
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(!roots.is_empty(), "empty V5 root batch");
        let batch = roots.len();
        let legal: Vec<Vec<ActionId>> = roots.iter().map(|r| r.legal_actions()).collect();
        anyhow::ensure!(
            legal.iter().all(|x| !x.is_empty()),
            "terminal root in V5 batch"
        );
        let lists: Vec<Vec<(u32, u32, u8)>> = legal
            .iter()
            .map(|list| {
                list.iter()
                    .map(|id| {
                        let (from, to, promo) = id.decode();
                        (from as u32, to as u32, promo.code())
                    })
                    .collect()
            })
            .collect();
        let batch_desc = CandidateBatch::from_lists(&lists);
        let cands = CandidateTensors::from_batch(&batch_desc, device);
        let width = batch_desc.width;
        phase("root_legal_lists_and_candidate_upload");
        let mut board = Vec::with_capacity(batch * SQUARES * IN_FEATURES);
        let mut geometry = vec![0.0; batch * width * ACTION_GEOMETRY];
        phase("root_host_allocation");
        let all_facts: Vec<Vec<recur64_core::CandidateFactsV1>> =
            roots.iter().map(|root| candidate_facts(root)).collect();
        phase("root_candidate_facts_exact_cpu");
        for (row, root) in roots.iter().enumerate() {
            let obs = encode_root_relative_observation_v1(root, root.side_to_move());
            board.extend_from_slice(obs.as_slice());
            for (column, id) in legal[row].iter().enumerate() {
                geometry[(row * width + column) * ACTION_GEOMETRY
                    ..(row * width + column + 1) * ACTION_GEOMETRY]
                    .copy_from_slice(&root_geom(*id));
            }
        }
        let refs: Vec<&[recur64_core::CandidateFactsV1]> =
            all_facts.iter().map(Vec::as_slice).collect();
        let inputs = Self {
            batch,
            root: Tensor::from_data(
                TensorData::new(board, [batch, SQUARES, IN_FEATURES]),
                device,
            ),
            cands,
            candidate_geometry: Tensor::from_data(
                TensorData::new(geometry, [batch, width, ACTION_GEOMETRY]),
                device,
            ),
            facts: facts_tensor::<B>(&refs, width, device)?,
        };
        phase("root_observation_geometry_and_fact_upload");
        Ok(inputs)
    }
}

impl<B: Backend> V5Inputs<B> {
    /// Expand per-node intervention choices to the four-slot payload boundary.
    pub fn payload_token_mask(
        &self,
        active_nodes: &[Vec<bool>],
        device: &B::Device,
    ) -> anyhow::Result<Tensor<B, 2, Bool>> {
        anyhow::ensure!(
            active_nodes.len() == self.batch,
            "payload intervention batch mismatch"
        );
        let node_width = self.evidence_tokens / 4;
        let mut mask = vec![false; self.batch * self.evidence_tokens];
        for (row, active) in active_nodes.iter().enumerate() {
            anyhow::ensure!(
                active.len() <= node_width,
                "payload intervention node width exceeds input"
            );
            for (node, &enabled) in active.iter().enumerate() {
                for slot in 0..4 {
                    mask[row * self.evidence_tokens + node * 4 + slot] = enabled;
                }
            }
        }
        Ok(Tensor::from_data(
            TensorData::new(mask, [self.batch, self.evidence_tokens]),
            device,
        ))
    }

    pub fn from_examples(
        examples: &[(&GameState, &AcquiredGraph)],
        device: &B::Device,
    ) -> anyhow::Result<Self> {
        Self::from_examples_profiled(examples, device, &mut |_| {})
    }

    pub(crate) fn from_examples_profiled(
        examples: &[(&GameState, &AcquiredGraph)],
        device: &B::Device,
        phase: &mut dyn FnMut(&str),
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(!examples.is_empty(), "empty V5 batch");
        let batch = examples.len();
        let roots: Vec<&GameState> = examples.iter().map(|x| x.0).collect();
        let legal: Vec<Vec<ActionId>> = roots.iter().map(|r| r.legal_actions()).collect();
        anyhow::ensure!(
            legal.iter().all(|x| !x.is_empty()),
            "terminal root in V5 batch"
        );
        let root_inputs = RootInputs::<B>::from_roots_profiled(&roots, device, phase)?;
        let w = root_inputs.cands.width;
        let qn = examples.iter().map(|x| x.1.actual_q).max().unwrap_or(0);
        anyhow::ensure!(qn > 0, "reader batch requires Q>0");
        let e = qn * 4;
        let mut states = vec![0.0; batch * qn * SQUARES * IN_FEATURES];
        let mut fl = vec![0.0; batch * qn * PAYLOAD_FLAGS];
        let mut node_mask = vec![false; batch * qn];
        let mut ev_mask = vec![false; batch * e];
        let mut structural = vec![0.0; batch * e * STRUCTURAL_FEATURES];
        let mut owner = vec![0i32; batch * e];
        for (bi, (root_state, g)) in examples.iter().enumerate() {
            g.verify()?;
            anyhow::ensure!(g.actual_q <= qn, "internal Q padding error");
            let expected = if root_state.side_to_move() == recur64_core::Color::White {
                "white"
            } else {
                "black"
            };
            anyhow::ensure!(g.root_player == expected, "graph/root player mismatch");
            for (ni, n) in g.nodes.iter().enumerate() {
                anyhow::ensure!(
                    n.root_candidate < legal[bi].len(),
                    "graph owner outside legal candidates"
                );
                states[((bi * qn + ni) * SQUARES * IN_FEATURES)
                    ..((bi * qn + ni + 1) * SQUARES * IN_FEATURES)]
                    .copy_from_slice(&n.payload.observation);
                fl[((bi * qn + ni) * PAYLOAD_FLAGS)..((bi * qn + ni + 1) * PAYLOAD_FLAGS)]
                    .copy_from_slice(&n.payload.flags);
                node_mask[bi * qn + ni] = true;
                for slot in 0..4 {
                    let ti = ni * 4 + slot;
                    ev_mask[bi * e + ti] = true;
                    owner[bi * e + ti] = n.root_candidate as i32;
                    let at = (bi * e + ti) * STRUCTURAL_FEATURES;
                    structural[at + n.depth as usize - 1] = 1.0;
                    structural[at + crate::TURN_FEATURE_OFFSET + usize::from(!n.root_to_move)] =
                        1.0;
                    structural[at + crate::SLOT_FEATURE_OFFSET + slot] = 1.0;
                    structural[at + crate::ACTION_FEATURE_OFFSET
                        ..at + crate::ACTION_FEATURE_OFFSET + ACTION_GEOMETRY]
                        .copy_from_slice(&n.action_geometry);
                }
            }
        }
        let mut erel = vec![0.0; batch * e * (e + w + SQUARES) * RELATIONS];
        let mut hrel = vec![0.0; batch * w * (w + e + SQUARES) * RELATIONS];
        for (bi, (_, g)) in examples.iter().enumerate() {
            let em = e + w + SQUARES;
            let hm = w + e + SQUARES;
            for qi in 0..e {
                let ni = qi / 4;
                if ni >= g.nodes.len() {
                    for mi in 0..em {
                        set_relation(&mut erel, bi, qi, mi, e, em, REL_DEFAULT);
                    }
                    continue;
                }
                for mi in 0..e {
                    let nj = mi / 4;
                    let r = if nj < g.nodes.len() {
                        ev_relation(&g.nodes[ni], &g.nodes[nj])
                    } else {
                        REL_DEFAULT
                    };
                    set_relation(&mut erel, bi, qi, mi, e, em, r);
                }
                for j in 0..w {
                    let r = if j == g.nodes[ni].root_candidate {
                        REL_E_OWNER_H
                    } else {
                        REL_E_OTHER_H
                    };
                    set_relation(&mut erel, bi, qi, e + j, e, em, r);
                }
                for j in 0..SQUARES {
                    set_relation(&mut erel, bi, qi, e + w + j, e, em, REL_ROOT_CONTEXT);
                }
            }
            for qi in 0..w {
                for j in 0..w {
                    set_relation(
                        &mut hrel,
                        bi,
                        qi,
                        j,
                        w,
                        hm,
                        if qi == j { REL_H_SELF } else { REL_H_OTHER },
                    );
                }
                for mi in 0..e {
                    let ni = mi / 4;
                    let r = if ni < g.nodes.len() && g.nodes[ni].root_candidate == qi {
                        REL_H_OWNED_E
                    } else {
                        REL_H_OTHER_E
                    };
                    set_relation(&mut hrel, bi, qi, w + mi, w, hm, r);
                }
                for j in 0..SQUARES {
                    set_relation(&mut hrel, bi, qi, w + e + j, w, hm, REL_ROOT_CONTEXT);
                }
            }
        }
        let inputs = Self {
            batch,
            evidence_tokens: e,
            root: root_inputs.root,
            cands: root_inputs.cands,
            candidate_geometry: root_inputs.candidate_geometry,
            facts: root_inputs.facts,
            states: Tensor::from_data(
                TensorData::new(states, [batch, qn, SQUARES, IN_FEATURES]),
                device,
            ),
            flags: Tensor::from_data(TensorData::new(fl, [batch, qn, PAYLOAD_FLAGS]), device),
            node_mask: Tensor::from_data(TensorData::new(node_mask, [batch, qn]), device),
            evidence_mask: Tensor::from_data(TensorData::new(ev_mask, [batch, e]), device),
            structural: Tensor::from_data(
                TensorData::new(structural, [batch, e, STRUCTURAL_FEATURES]),
                device,
            ),
            owner_idx: Tensor::from_data(TensorData::new(owner, [batch, e]), device),
            evidence_rel: Tensor::from_data(
                TensorData::new(erel, [batch, e, e + w + SQUARES, RELATIONS]),
                device,
            ),
            hypothesis_rel: Tensor::from_data(
                TensorData::new(hrel, [batch, w, w + e + SQUARES, RELATIONS]),
                device,
            ),
        };
        phase("graph_packet_encoding_relations_and_upload");
        Ok(inputs)
    }
}

#[cfg(test)]
mod recall_tests {
    use super::*;
    use crate::graph::{EpisodeKey, Schedule, acquire};
    type B = burn::backend::Autodiff<burn::backend::Flex>;

    #[test]
    fn all_depth_fields_are_disjoint_from_turn_slot_action_and_padding() {
        type Cpu = burn::backend::Flex;
        let _guard = crate::CPU_TEST_RNG
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let device = Default::default();
        let (root, deep) = crate::graph::depth_chain_fixture(16);
        let shallow = deep.prefix(5).unwrap();
        let inputs =
            V5Inputs::<Cpu>::from_examples(&[(&root, &deep), (&root, &shallow)], &device).unwrap();
        let structural = inputs
            .structural
            .clone()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(crate::DEPTH_FEATURES, 16);
        assert_eq!(STRUCTURAL_FEATURES, 33);
        for (row, graph) in [&deep, &shallow].into_iter().enumerate() {
            for node in 0..16 {
                for slot in 0..4 {
                    let at = (row * 64 + node * 4 + slot) * STRUCTURAL_FEATURES;
                    let fields = &structural[at..at + STRUCTURAL_FEATURES];
                    if let Some(n) = graph.nodes.get(node) {
                        let mut expected = [0.0; STRUCTURAL_FEATURES];
                        expected[usize::from(n.depth) - 1] = 1.0;
                        expected[crate::TURN_FEATURE_OFFSET + usize::from(!n.root_to_move)] = 1.0;
                        expected[crate::SLOT_FEATURE_OFFSET + slot] = 1.0;
                        expected[crate::ACTION_FEATURE_OFFSET..]
                            .copy_from_slice(&n.action_geometry);
                        assert_eq!(fields, expected, "row={row} node={node} slot={slot}");
                        assert_eq!(fields[..16].iter().sum::<f32>(), 1.0);
                    } else {
                        assert!(fields.iter().all(|v| *v == 0.0));
                    }
                }
            }
        }
        <Cpu as Backend>::seed(&device, 5301);
        let model =
            CounterfactualRelationalLoop::<Cpu>::new(crate::config::V5Config::default(), &device);
        let output = model.paired(&inputs, 1, Treatment::AllPayloadNull);
        assert!(
            output
                .centered_delta
                .into_data()
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|v| *v == 0.0)
        );
    }

    #[test]
    fn immutable_anchor_has_a_direct_gradient_path_at_every_loop_in_both_blocks() {
        let _guard = crate::CPU_TEST_RNG
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let device = Default::default();
        <B as Backend>::seed(&device, 5301);
        let model = CounterfactualRelationalLoop::<B>::new(V5Config::default(), &device);
        let root = GameState::from_fen("6k1/8/8/8/8/8/4Q3/3RK3 w - - 0 1").unwrap();
        let graph = acquire(
            &root,
            EpisodeKey {
                position_id: "recall-gradient-fixture".into(),
                schedule: Schedule::UniformFrontierV1,
                run_seed: 5301,
                occurrence_ordinal: 0,
            },
            4,
            None,
        )
        .unwrap();
        let input = V5Inputs::<B>::from_examples(&[(&root, &graph)], &device).unwrap();
        let base = model.base_frozen(&[(&root, &graph)], &device).unwrap();
        let anchors = model
            .state
            .forward(
                input.states.clone(),
                input.flags.clone(),
                input.node_mask.clone(),
            )
            .reshape([1, input.evidence_tokens, 256]);
        let trace = model.paired_traced_with_base_payload_mask(
            &input,
            model.base_frozen(&[(&root, &graph)], &device).unwrap(),
            4,
            Treatment::Normal,
            None,
        );
        let root_valid = Tensor::<B, 2, Bool>::ones([1, SQUARES], &device);
        for (iteration, state) in trace.factual.iter().enumerate() {
            for evidence in [true, false] {
                // Freeze mutable state and memory for this read so an anchor
                // gradient cannot be explained by its initialization or an
                // earlier loop. Exercise the production block at each actual
                // loop state, with only its immutable recall input tracked.
                let (block, anchor, mutable, memory, query_mask, memory_mask, relation) =
                    if evidence {
                        (
                            &model.evidence,
                            anchors.clone().detach().require_grad(),
                            state.evidence_before.clone().detach(),
                            Tensor::cat(
                                vec![
                                    state.evidence_before.clone(),
                                    state.hypothesis_before.clone(),
                                    base.context.clone(),
                                ],
                                1,
                            )
                            .detach(),
                            input.evidence_mask.clone(),
                            Tensor::cat(
                                vec![
                                    input.evidence_mask.clone(),
                                    input.cands.mask.clone(),
                                    root_valid.clone(),
                                ],
                                1,
                            ),
                            input.evidence_rel.clone(),
                        )
                    } else {
                        (
                            &model.hypothesis,
                            base.hypotheses.clone().detach().require_grad(),
                            state.hypothesis_before.clone().detach(),
                            Tensor::cat(
                                vec![
                                    state.hypothesis_before.clone(),
                                    state.evidence_after.clone(),
                                    base.context.clone(),
                                ],
                                1,
                            )
                            .detach(),
                            input.cands.mask.clone(),
                            Tensor::cat(
                                vec![
                                    input.cands.mask.clone(),
                                    input.evidence_mask.clone(),
                                    root_valid.clone(),
                                ],
                                1,
                            ),
                            input.hypothesis_rel.clone(),
                        )
                    };
                let (output, _) = block.forward_inner(
                    mutable,
                    anchor.clone(),
                    memory,
                    query_mask,
                    memory_mask,
                    relation,
                    false,
                    false,
                );
                let objective = output.clone().slice([0..1, 0..1, 0..1]).sum()
                    - output.slice([0..1, 1..2, 1..2]).sum();
                let grads = objective.backward();
                let gradient = anchor
                    .grad(&grads)
                    .expect("immutable recall input gradient");
                let values = gradient.into_data().to_vec::<f32>().unwrap();
                let norm = values.iter().map(|v| v * v).sum::<f32>().sqrt();
                println!(
                    "V5_RECALL_EVIDENCE {}",
                    serde_json::json!({
                        "loop": iteration + 1, "block": if evidence { "evidence" } else { "hypothesis" },
                        "direct_anchor_gradient_l2": norm,
                    })
                );
                assert!(
                    values.iter().all(|v| v.is_finite()) && norm > 1.0e-9,
                    "direct recall gradient missing: iteration={} evidence={evidence} norm={norm:e}",
                    iteration + 1
                );
            }
        }
    }
}
