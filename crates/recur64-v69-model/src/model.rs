//! V69 two-clock workspace model (frozen in docs/v69/MODEL_SPEC.md).

use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::{Int, TensorData, activation};
use recur64_v69::features::{Features, N_PIECE_CODES, N_SCALARS};

pub const D: usize = 192;
pub const HEADS: usize = 6;
pub const HEAD_DIM: usize = D / HEADS;
pub const FFN: usize = 576;
pub const N_WS: usize = 8;
pub const N_SQ: usize = 64;
/// Fixed residual scale applied to every sublayer.
pub const RES_SCALE: f64 = 0.1;
pub const LN_EPS: f64 = 1e-5;

fn ln<B: Backend>(device: &B::Device) -> LayerNorm<B> {
    LayerNormConfig::new(D).with_epsilon(LN_EPS).init(device)
}
fn lin<B: Backend>(i: usize, o: usize, device: &B::Device) -> Linear<B> {
    LinearConfig::new(i, o).with_bias(true).init(device)
}

#[derive(Module, Debug)]
pub struct Attn<B: Backend> {
    q: Linear<B>,
    k: Linear<B>,
    v: Linear<B>,
    o: Linear<B>,
}

impl<B: Backend> Attn<B> {
    fn new(device: &B::Device) -> Self {
        Self { q: lin(D, D, device), k: lin(D, D, device), v: lin(D, D, device), o: lin(D, D, device) }
    }
    /// Scaled dot-product attention, scale 1/sqrt(head_dim), no mask, no dropout.
    fn forward(&self, xq: Tensor<B, 3>, xkv: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, nq, _] = xq.dims();
        let nk = xkv.dims()[1];
        let q = self.q.forward(xq).reshape([b, nq, HEADS, HEAD_DIM]).swap_dims(1, 2);
        let k = self.k.forward(xkv.clone()).reshape([b, nk, HEADS, HEAD_DIM]).swap_dims(1, 2);
        let v = self.v.forward(xkv).reshape([b, nk, HEADS, HEAD_DIM]).swap_dims(1, 2);
        let logits = q.matmul(k.swap_dims(2, 3)).mul_scalar(1.0 / (HEAD_DIM as f32).sqrt());
        let w = activation::softmax(logits, 3);
        let out = w.matmul(v).swap_dims(1, 2).reshape([b, nq, D]);
        self.o.forward(out)
    }
}

#[derive(Module, Debug)]
pub struct Ffn<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
}

impl<B: Backend> Ffn<B> {
    fn new(device: &B::Device) -> Self {
        Self { l1: lin(D, FFN, device), l2: lin(FFN, D, device) }
    }
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.l2.forward(activation::gelu(self.l1.forward(x)))
    }
}

#[derive(Module, Debug)]
pub struct EncBlock<B: Backend> {
    ln1: LayerNorm<B>,
    attn: Attn<B>,
    ln2: LayerNorm<B>,
    ffn: Ffn<B>,
}

impl<B: Backend> EncBlock<B> {
    fn new(device: &B::Device) -> Self {
        Self { ln1: ln(device), attn: Attn::new(device), ln2: ln(device), ffn: Ffn::new(device) }
    }
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let h = self.ln1.forward(x.clone());
        let x = x + self.attn.forward(h.clone(), h).mul_scalar(RES_SCALE);
        let f = self.ffn.forward(self.ln2.forward(x.clone()));
        x + f.mul_scalar(RES_SCALE)
    }
}

/// Shared recurrent block. `fast` updates the board (self-attention over the
/// current board plus the fixed encoded board E as a read-only refresh,
/// attention to the workspace, FFN); `slow` updates the workspace
/// (self-attention, attention to the board, FFN). Same structure, separate
/// parameters.
#[derive(Module, Debug)]
pub struct ClockBlock<B: Backend> {
    ln_self: LayerNorm<B>,
    self_attn: Attn<B>,
    ln_q: LayerNorm<B>,
    ln_kv: LayerNorm<B>,
    cross: Attn<B>,
    ln_ff: LayerNorm<B>,
    ffn: Ffn<B>,
}

impl<B: Backend> ClockBlock<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            ln_self: ln(device),
            self_attn: Attn::new(device),
            ln_q: ln(device),
            ln_kv: ln(device),
            cross: Attn::new(device),
            ln_ff: ln(device),
            ffn: Ffn::new(device),
        }
    }

    /// `own` is the stream being updated; `other` the stream it attends to;
    /// `refresh` (fast block only) is the fixed encoded board E.
    fn forward(&self, own: Tensor<B, 3>, other: Tensor<B, 3>, refresh: Option<Tensor<B, 3>>) -> Tensor<B, 3> {
        let h = self.ln_self.forward(own.clone());
        let kv = match refresh {
            Some(e) => Tensor::cat(vec![h.clone(), self.ln_self.forward(e)], 1),
            None => h.clone(),
        };
        let x = own + self.self_attn.forward(h, kv).mul_scalar(RES_SCALE);
        let q = self.ln_q.forward(x.clone());
        let kv = self.ln_kv.forward(other);
        let x = x + self.cross.forward(q, kv).mul_scalar(RES_SCALE);
        let f = self.ffn.forward(self.ln_ff.forward(x.clone()));
        x + f.mul_scalar(RES_SCALE)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arm {
    A,
    B,
    C,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Fast,
    Slow,
    Read,
}

impl Arm {
    pub const ALL: [Arm; 3] = [Arm::A, Arm::B, Arm::C];

    pub fn name(self) -> &'static str {
        match self {
            Arm::A => "A",
            Arm::B => "B",
            Arm::C => "C",
        }
    }
    pub fn parse(s: &str) -> anyhow::Result<Arm> {
        match s {
            "A" => Ok(Arm::A),
            "B" => Ok(Arm::B),
            "C" => Ok(Arm::C),
            _ => anyhow::bail!("unknown arm {s}"),
        }
    }
    /// Executable schedule.
    pub fn schedule(self) -> Vec<Op> {
        let mut v = Vec::new();
        match self {
            Arm::A => v.extend([Op::Fast, Op::Slow, Op::Read]),
            Arm::B => {
                for cycle in 1..=4 {
                    v.extend([Op::Fast, Op::Fast, Op::Slow]);
                    if matches!(cycle, 1 | 2 | 4) {
                        v.push(Op::Read);
                    }
                }
            }
            Arm::C => {
                for pair in 1..=6 {
                    v.extend([Op::Fast, Op::Slow]);
                    if matches!(pair, 2 | 4 | 6) {
                        v.push(Op::Read);
                    }
                }
            }
        }
        v
    }
    pub fn n_readouts(self) -> usize {
        self.schedule().iter().filter(|o| **o == Op::Read).count()
    }
    pub fn n_calls(self) -> usize {
        self.schedule().iter().filter(|o| **o != Op::Read).count()
    }
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    piece_emb: Embedding<B>,
    square_emb: Embedding<B>,
    scalar_proj: Linear<B>,
    budget_emb: Embedding<B>,
    att_emb: Embedding<B>,
    enc0: EncBlock<B>,
    enc1: EncBlock<B>,
    enc_ln: LayerNorm<B>,
    slot_emb: Embedding<B>,
    ws_pool_ln: LayerNorm<B>,
    ws_pool_proj: Linear<B>,
    fast: ClockBlock<B>,
    slow: ClockBlock<B>,
    head_ln: LayerNorm<B>,
    head1: Linear<B>,
    head2: Linear<B>,
}

/// Device-resident inputs. No label, id or order information.
pub struct Batch<B: Backend> {
    pub piece: Tensor<B, 2, Int>,
    pub scalars: Tensor<B, 2>,
    pub budget: Tensor<B, 2, Int>,
    pub att: Tensor<B, 2, Int>,
    pub n: usize,
}

impl<B: Backend> Batch<B> {
    pub fn from_features(fs: &[&Features], device: &B::Device) -> Self {
        let n = fs.len();
        let piece: Vec<i32> = fs.iter().flat_map(|f| f.piece.iter().map(|&p| p as i32)).collect();
        let scalars: Vec<f32> = fs.iter().flat_map(|f| f.scalars.iter().copied()).collect();
        let budget: Vec<i32> = fs.iter().map(|f| f.budget as i32).collect();
        let att: Vec<i32> = fs.iter().map(|f| f.attacker_to_move as i32).collect();
        Self {
            piece: Tensor::from_data(TensorData::new(piece, [n, N_SQ]), device),
            scalars: Tensor::from_data(TensorData::new(scalars, [n, N_SCALARS]), device),
            budget: Tensor::from_data(TensorData::new(budget, [n, 1]), device),
            att: Tensor::from_data(TensorData::new(att, [n, 1]), device),
            n,
        }
    }
}

/// Qualification-only forward hooks (all off in scientific runs).
pub struct FwdOpts<B: Backend> {
    /// Zero-valued leaf probes added to (board, workspace) after each block call.
    pub probes: Option<Vec<(Tensor<B, 3>, Tensor<B, 3>)>>,
    /// Negative control: detach both states after this (1-based) call.
    pub detach_after_call: Option<usize>,
    /// Synchronize the device after every block call (profile mode).
    pub sync_each_call: bool,
}

impl<B: Backend> Default for FwdOpts<B> {
    fn default() -> Self {
        Self { probes: None, detach_after_call: None, sync_each_call: false }
    }
}

impl<B: Backend> Model<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            piece_emb: EmbeddingConfig::new(N_PIECE_CODES, D).init(device),
            square_emb: EmbeddingConfig::new(N_SQ, D).init(device),
            scalar_proj: lin(N_SCALARS, D, device),
            budget_emb: EmbeddingConfig::new(4, D).init(device),
            att_emb: EmbeddingConfig::new(2, D).init(device),
            enc0: EncBlock::new(device),
            enc1: EncBlock::new(device),
            enc_ln: ln(device),
            slot_emb: EmbeddingConfig::new(N_WS, D).init(device),
            ws_pool_ln: ln(device),
            ws_pool_proj: lin(D, D, device),
            fast: ClockBlock::new(device),
            slow: ClockBlock::new(device),
            head_ln: ln(device),
            head1: lin(D, D, device),
            head2: lin(D, 1, device),
        }
    }

    /// Board encoder output E [b,64,D] and initial workspace [b,8,D].
    pub fn encode(&self, x: &Batch<B>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let tok = self.piece_emb.forward(x.piece.clone())
            + self.square_emb.weight.val().unsqueeze_dim::<3>(0)
            + self.scalar_proj.forward(x.scalars.clone()).unsqueeze_dim::<3>(1)
            + self.budget_emb.forward(x.budget.clone())
            + self.att_emb.forward(x.att.clone());
        let tok = self.enc1.forward(self.enc0.forward(tok));
        let e = self.enc_ln.forward(tok);
        let pooled = e.clone().mean_dim(1); // [b,1,D]
        let p = self.ws_pool_proj.forward(self.ws_pool_ln.forward(pooled)); // [b,1,D]
        let ws = self.slot_emb.weight.val().unsqueeze_dim::<3>(0) + p; // [b,8,D]
        (e, ws)
    }

    fn readout(&self, ws: Tensor<B, 3>) -> Tensor<B, 1> {
        let b = ws.dims()[0];
        let pooled = ws.mean_dim(1).reshape([b, D]);
        let h = activation::gelu(self.head1.forward(self.head_ln.forward(pooled)));
        self.head2.forward(h).reshape([b])
    }

    /// Logits [b] at every prescribed readout of `arm`. State is created from the
    /// batch inputs only (reset per example); full BPTT (no detach, no carry).
    pub fn forward_ex(&self, x: &Batch<B>, arm: Arm, opts: &FwdOpts<B>) -> Vec<Tensor<B, 1>> {
        let (e, mut ws) = self.encode(x);
        let mut board = e.clone();
        let mut outs = Vec::new();
        let mut call = 0usize;
        for op in arm.schedule() {
            match op {
                Op::Fast => {
                    board = self.fast.forward(board, ws.clone(), Some(e.clone()));
                    call += 1;
                }
                Op::Slow => {
                    ws = self.slow.forward(ws, board.clone(), None);
                    call += 1;
                }
                Op::Read => {
                    outs.push(self.readout(ws.clone()));
                    continue;
                }
            }
            if let Some(pr) = &opts.probes {
                let (pb, pw) = &pr[call - 1];
                board = board + pb.clone();
                ws = ws + pw.clone();
            }
            if opts.detach_after_call == Some(call) {
                board = board.detach();
                ws = ws.detach();
            }
            if opts.sync_each_call {
                let _ = B::sync(&ws.device());
            }
        }
        outs
    }

    pub fn forward(&self, x: &Batch<B>, arm: Arm) -> Vec<Tensor<B, 1>> {
        self.forward_ex(x, arm, &FwdOpts::default())
    }
}

/// Numerically stable BCE-with-logits, mean over the batch: max(z,0) - z*y + ln(1+exp(-|z|)).
pub fn bce_mean<B: Backend>(z: Tensor<B, 1>, y: Tensor<B, 1>) -> Tensor<B, 1> {
    // max(z,0) is written (z+|z|)/2 so the subgradient at z = 0 is the symmetric 1/2
    // (clamp_min has a one-sided gradient at exactly 0, found by qualification).
    let t = (z.clone() + z.clone().abs()).mul_scalar(0.5) - z.clone() * y + z.abs().neg().exp().add_scalar(1.0).log();
    t.mean()
}

/// Mean over prescribed readouts of the batch-mean BCE.
pub fn arm_loss<B: Backend>(outs: &[Tensor<B, 1>], y: &Tensor<B, 1>) -> Tensor<B, 1> {
    let mut acc: Option<Tensor<B, 1>> = None;
    for z in outs {
        let l = bce_mean(z.clone(), y.clone());
        acc = Some(match acc {
            Some(a) => a + l,
            None => l,
        });
    }
    acc.unwrap().div_scalar(outs.len() as f32)
}
