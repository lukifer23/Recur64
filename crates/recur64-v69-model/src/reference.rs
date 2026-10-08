//! Independent f64 host implementation of the V69 model (qualification only).
//!
//! Plain nested loops, no Burn, no GPU. It exists to (a) verify that the CUDA
//! graph computes the specified function, (b) measure the actual numeric error of
//! the CUDA f32 path (Burn/cubek may select TF32 tensor-core matmuls for f32
//! tensors; strict FP32 matmul cannot be forced without custom kernels), and
//! (c) supply exact finite-difference gradients in f64.

use crate::inventory::ParamInfo;
use crate::model::{Arm, D, FFN, HEAD_DIM, HEADS, LN_EPS, N_WS, Op, RES_SCALE};
use recur64_v69::features::Features;
use std::collections::HashMap;

pub struct RefWeights {
    pub map: HashMap<String, (Vec<usize>, Vec<f64>)>,
}

impl RefWeights {
    pub fn new(inv: &[ParamInfo], values: &[(Vec<usize>, Vec<f32>)]) -> Self {
        let map = inv
            .iter()
            .zip(values)
            .map(|(p, (s, v))| (format!("{}.{}", p.path, p.leaf), (s.clone(), v.iter().map(|x| *x as f64).collect())))
            .collect();
        Self { map }
    }
    fn get(&self, k: &str) -> &Vec<f64> {
        &self.map.get(k).unwrap_or_else(|| panic!("missing weight {k}")).1
    }
    pub fn perturbed(&self, key: &str, dir: &[f64], eps: f64) -> Self {
        let mut map = self.map.clone();
        for (x, d) in map.get_mut(key).unwrap().1.iter_mut().zip(dir) {
            *x += eps * d;
        }
        Self { map }
    }
}

fn erf(x: f64) -> f64 {
    if x.abs() > 4.5 {
        return x.signum();
    }
    // Maclaurin series; cancellation is harmless in f64 for |x| <= 6/sqrt(2)*... (arguments here are <= ~5)
    let mut term = x;
    let mut sum = x;
    let x2 = x * x;
    for n in 1..200 {
        term *= -x2 / n as f64;
        let add = term / (2 * n + 1) as f64;
        sum += add;
        if add.abs() < 1e-18 {
            break;
        }
    }
    2.0 / std::f64::consts::PI.sqrt() * sum
}

fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// y[n,dout] = x[n,din] w[din,dout] + b
fn lin(w: &RefWeights, name: &str, x: &[f64], n: usize, din: usize, dout: usize) -> Vec<f64> {
    let wt = w.get(&format!("{name}.weight"));
    let b = w.get(&format!("{name}.bias"));
    let mut y = vec![0.0; n * dout];
    for i in 0..n {
        let xr = &x[i * din..(i + 1) * din];
        let yr = &mut y[i * dout..(i + 1) * dout];
        yr.copy_from_slice(b);
        for (k, xv) in xr.iter().enumerate() {
            let wr = &wt[k * dout..(k + 1) * dout];
            for (o, wv) in wr.iter().enumerate() {
                yr[o] += xv * wv;
            }
        }
    }
    y
}

fn layer_norm(w: &RefWeights, name: &str, x: &[f64], n: usize) -> Vec<f64> {
    let g = w.get(&format!("{name}.gamma"));
    let b = w.get(&format!("{name}.beta"));
    let mut y = vec![0.0; n * D];
    for i in 0..n {
        let r = &x[i * D..(i + 1) * D];
        let mean = r.iter().sum::<f64>() / D as f64;
        let var = r.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / D as f64;
        let inv = 1.0 / (var + LN_EPS).sqrt();
        for j in 0..D {
            y[i * D + j] = (r[j] - mean) * inv * g[j] + b[j];
        }
    }
    y
}

fn attention(w: &RefWeights, name: &str, xq: &[f64], nq: usize, xkv: &[f64], nk: usize) -> Vec<f64> {
    let q = lin(w, &format!("{name}.q"), xq, nq, D, D);
    let k = lin(w, &format!("{name}.k"), xkv, nk, D, D);
    let v = lin(w, &format!("{name}.v"), xkv, nk, D, D);
    let scale = 1.0 / (HEAD_DIM as f64).sqrt();
    let mut cat = vec![0.0; nq * D];
    for h in 0..HEADS {
        for i in 0..nq {
            let mut sc = vec![0.0; nk];
            for j in 0..nk {
                let mut s = 0.0;
                for d in 0..HEAD_DIM {
                    s += q[i * D + h * HEAD_DIM + d] * k[j * D + h * HEAD_DIM + d];
                }
                sc[j] = s * scale;
            }
            let mx = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut z = 0.0;
            for s in sc.iter_mut() {
                *s = (*s - mx).exp();
                z += *s;
            }
            for d in 0..HEAD_DIM {
                let mut o = 0.0;
                for j in 0..nk {
                    o += sc[j] / z * v[j * D + h * HEAD_DIM + d];
                }
                cat[i * D + h * HEAD_DIM + d] = o;
            }
        }
    }
    lin(w, &format!("{name}.o"), &cat, nq, D, D)
}

fn ffn(w: &RefWeights, name: &str, x: &[f64], n: usize) -> Vec<f64> {
    let h: Vec<f64> = lin(w, &format!("{name}.l1"), x, n, D, FFN).into_iter().map(gelu).collect();
    lin(w, &format!("{name}.l2"), &h, n, FFN, D)
}

fn add_scaled(x: &mut [f64], y: &[f64]) {
    for (a, b) in x.iter_mut().zip(y) {
        *a += RES_SCALE * b;
    }
}

fn enc_block(w: &RefWeights, name: &str, x: &mut Vec<f64>) {
    let h = layer_norm(w, &format!("{name}.ln1"), x, 64);
    let a = attention(w, &format!("{name}.attn"), &h, 64, &h, 64);
    add_scaled(x, &a);
    let h2 = layer_norm(w, &format!("{name}.ln2"), x, 64);
    let f = ffn(w, &format!("{name}.ffn"), &h2, 64);
    add_scaled(x, &f);
}

fn clock_block(w: &RefWeights, name: &str, own: &mut Vec<f64>, n_own: usize, other: &[f64], n_other: usize, refresh: Option<&[f64]>) {
    let h = layer_norm(w, &format!("{name}.ln_self"), own, n_own);
    let (kv, nkv) = match refresh {
        Some(e) => {
            let mut c = h.clone();
            c.extend(layer_norm(w, &format!("{name}.ln_self"), e, 64));
            (c, n_own + 64)
        }
        None => (h.clone(), n_own),
    };
    let a = attention(w, &format!("{name}.self_attn"), &h, n_own, &kv, nkv);
    add_scaled(own, &a);
    let q = layer_norm(w, &format!("{name}.ln_q"), own, n_own);
    let kvo = layer_norm(w, &format!("{name}.ln_kv"), other, n_other);
    let c = attention(w, &format!("{name}.cross"), &q, n_own, &kvo, n_other);
    add_scaled(own, &c);
    let hf = layer_norm(w, &format!("{name}.ln_ff"), own, n_own);
    let f = ffn(w, &format!("{name}.ffn"), &hf, n_own);
    add_scaled(own, &f);
}

/// Logits at every prescribed readout of `arm` for one example.
pub fn ref_forward(w: &RefWeights, f: &Features, arm: Arm) -> Vec<f64> {
    let piece = w.get("piece_emb.weight");
    let square = w.get("square_emb.weight");
    let budget = w.get("budget_emb.weight");
    let att = w.get("att_emb.weight");
    let sp = lin(w, "scalar_proj", &f.scalars.iter().map(|x| *x as f64).collect::<Vec<_>>(), 1, 8, D);
    let mut x = vec![0.0; 64 * D];
    for s in 0..64 {
        for j in 0..D {
            x[s * D + j] = piece[f.piece[s] as usize * D + j] + square[s * D + j] + sp[j] + budget[f.budget as usize * D + j] + att[f.attacker_to_move as usize * D + j];
        }
    }
    enc_block(w, "enc0", &mut x);
    enc_block(w, "enc1", &mut x);
    let e = layer_norm(w, "enc_ln", &x, 64);
    let mut pooled = vec![0.0; D];
    for s in 0..64 {
        for j in 0..D {
            pooled[j] += e[s * D + j] / 64.0;
        }
    }
    let pl = layer_norm(w, "ws_pool_ln", &pooled, 1);
    let p = lin(w, "ws_pool_proj", &pl, 1, D, D);
    let slot = w.get("slot_emb.weight");
    let mut ws = vec![0.0; N_WS * D];
    for s in 0..N_WS {
        for j in 0..D {
            ws[s * D + j] = slot[s * D + j] + p[j];
        }
    }
    let mut board = e.clone();
    let mut outs = Vec::new();
    for op in arm.schedule() {
        match op {
            Op::Fast => {
                let ws_c = ws.clone();
                clock_block(w, "fast", &mut board, 64, &ws_c, N_WS, Some(&e));
            }
            Op::Slow => {
                let b_c = board.clone();
                clock_block(w, "slow", &mut ws, N_WS, &b_c, 64, None);
            }
            Op::Read => {
                let mut m = vec![0.0; D];
                for s in 0..N_WS {
                    for j in 0..D {
                        m[j] += ws[s * D + j] / N_WS as f64;
                    }
                }
                let hn = layer_norm(w, "head_ln", &m, 1);
                let h: Vec<f64> = lin(w, "head1", &hn, 1, D, D).into_iter().map(gelu).collect();
                outs.push(lin(w, "head2", &h, 1, D, 1)[0]);
            }
        }
    }
    outs
}

pub fn bce64(z: f64, y: bool) -> f64 {
    z.max(0.0) - z * (y as u8 as f64) + (-z.abs()).exp().ln_1p()
}

/// Mean over examples of the mean-over-readouts BCE.
pub fn ref_loss(w: &RefWeights, feats: &[&Features], labels: &[bool], arm: Arm) -> f64 {
    let mut tot = 0.0;
    for (f, y) in feats.iter().zip(labels) {
        let outs = ref_forward(w, f, arm);
        tot += outs.iter().map(|z| bce64(*z, *y)).sum::<f64>() / outs.len() as f64;
    }
    tot / feats.len() as f64
}

/// f64 host reference of the D1 direct-board MLP (842 -> 128 -> 64 -> 1, GELU).
pub fn mlp_ref_forward(w: &RefWeights, f: &Features) -> f64 {
    let x: Vec<f64> = crate::d1::mlp_input_vec(f).into_iter().map(|v| v as f64).collect();
    let h1: Vec<f64> = lin(w, "fc1", &x, 1, crate::d1::MLP_IN, 128).into_iter().map(gelu).collect();
    let h2: Vec<f64> = lin(w, "fc2", &h1, 1, 128, 64).into_iter().map(gelu).collect();
    lin(w, "fc3", &h2, 1, 64, 1)[0]
}

pub fn mlp_ref_loss(w: &RefWeights, feats: &[&Features], labels: &[bool]) -> f64 {
    feats.iter().zip(labels).map(|(f, y)| bce64(mlp_ref_forward(w, f), *y)).sum::<f64>() / feats.len() as f64
}
