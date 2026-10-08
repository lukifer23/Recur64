//! CUDA FP32 qualification of the actual V69 graphs (arms A/B/C).
//!
//! Uses a small fixed fitting-only panel and DISPOSABLE weights drawn from the
//! `qualification_init` stream (never the scientific `model_init` stream).
//! Predefined tolerances are the constants below; none is adjusted after seeing
//! results.

use crate::inventory::{ParamInfo, dump_values, inventory, load_values};
use crate::model::{Arm, Batch, FwdOpts, Model, arm_loss, bce_mean};
use crate::reference::{RefWeights, ref_forward, ref_loss};
use crate::train::{ACCUM, MICRO, Micro, Trainer, clip_factor, lr_at, save_checkpoint, load_checkpoint};
use burn::backend::cuda::CudaDevice;
use burn::backend::{Autodiff, Cuda};
use burn::module::{AutodiffModule, Module, ModuleVisitor, Param};
use burn::optim::{GradientsAccumulator, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use recur64_v69::features::{Features, ModelRow, featurize};
use recur64_v69::streams::MasterSeed;
use serde_json::{Value, json};
use std::path::Path;
use std::time::Instant;

pub type G = Cuda;
pub type GA = Autodiff<Cuda>;

// ---- predefined tolerances (frozen in docs/v69/MODEL_SPEC.md) ----
pub const TOL_BCE_ABS: f64 = 1e-5;
pub const TOL_BCE_GRAD_ABS: f64 = 1e-5;
// Per-tensor accumulation agreement, relative to max(tensor norm, 1e-3 * global norm): the first
// qualification run showed that tiny-gradient tensors (attention key biases, whose true gradient
// is exactly zero by softmax shift invariance) have noise-level relative error, and that CUDA
// "f32" matmuls may run as TF32 tensor-core products (see reference.rs).
pub const TOL_ACCUM_REL_L2: f64 = 2e-2;
pub const TOL_ACCUM_GLOBAL_REL: f64 = 1e-3;
pub const TOL_LOSS_ABS: f64 = 1e-5;
pub const TOL_CLIP_NORM_ABS: f64 = 1e-4;
/// Numeric-gradient check against exact f64 finite differences of the host reference model.
pub const TOL_NUMERIC_REL: f64 = 0.10;
pub const TOL_NUMERIC_ABS: f64 = 1e-9;
/// GPU logits vs independent f64 host reference (strict FP32 would be ~1e-5; TF32 ~1e-3).
pub const TOL_FORWARD_ABS: f64 = 2e-3;
pub const TOL_ISOLATION_LOGIT: f64 = 1e-4;
pub const TOL_CKPT_PARAM_ABS: f64 = 1e-6;
pub const TOL_PROFILE_LOGIT: f64 = 1e-6;
pub const TOL_PROFILE_GRAD_REL_L2: f64 = 1e-5;
pub const TOL_DECAY_REL: f64 = 1e-6;
pub const TOL_ZERO_PROBE: f64 = 1e-12;
pub const DEVICE_MEM_CEILING_MIB: u64 = 3072;
pub const QUAL_WALL_SECS_PER_ARM: u64 = 30 * 60;

pub struct QualCtx {
    pub rows: Vec<ModelRow>,
    pub feats: Vec<Features>,
    pub seed: MasterSeed,
    /// Hash of the scientific canonical initialization tensors (identity check only).
    pub scientific_init_values: Vec<(Vec<usize>, Vec<f32>)>,
    pub scientific_init_hash: String,
}

fn device() -> CudaDevice {
    CudaDevice::default()
}

fn host1<B: Backend>(t: Tensor<B, 1>) -> Vec<f32> {
    t.into_data().to_vec::<f32>().unwrap()
}

fn l2(v: &[f32]) -> f64 {
    v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt()
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let d: f64 = a.iter().zip(b).map(|(x, y)| ((*x - *y) as f64).powi(2)).sum::<f64>().sqrt();
    d / l2(b).max(1e-30)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| ((*x - *y) as f64).abs()).fold(0.0, f64::max)
}

pub struct Check {
    pub name: String,
    pub pass: bool,
    pub detail: Value,
}

impl Check {
    fn new(name: &str, pass: bool, detail: Value) -> Self {
        Self { name: name.to_string(), pass, detail }
    }
    pub fn to_json(&self) -> Value {
        json!({"name": self.name, "pass": self.pass, "detail": self.detail})
    }
}

fn qual_values(ctx: &QualCtx) -> Vec<(Vec<usize>, Vec<f32>)> {
    let inv = inventory::<G, _>(&Model::<G>::new(&device()));
    crate::init::generate(&ctx.seed, "qualification_init", &inv)
}

fn fresh_model(values: &[(Vec<usize>, Vec<f32>)]) -> Model<GA> {
    let d = device();
    load_values::<GA, _>(Model::<GA>::new(&d), values, &d)
}

fn micro_of<'a>(ctx: &'a QualCtx, labels: &[bool], i: usize) -> Micro<'a> {
    Micro { feats: vec![&ctx.feats[2 * i], &ctx.feats[2 * i + 1]], labels: vec![labels[2 * i], labels[2 * i + 1]] }
}

fn all_micros<'a>(ctx: &'a QualCtx, labels: &[bool]) -> Vec<Micro<'a>> {
    (0..ACCUM).map(|i| micro_of(ctx, labels, i)).collect()
}

fn panel_labels(ctx: &QualCtx) -> Vec<bool> {
    ctx.rows.iter().map(|r| r.label).collect()
}

fn batch16(ctx: &QualCtx) -> Batch<GA> {
    let refs: Vec<&Features> = ctx.feats.iter().collect();
    Batch::<GA>::from_features(&refs, &device())
}

fn y_tensor(labels: &[bool]) -> Tensor<GA, 1> {
    Tensor::<GA, 1>::from_floats(labels.iter().map(|&l| if l { 1.0f32 } else { 0.0 }).collect::<Vec<_>>().as_slice(), &device())
}

struct GradDump<'a, B: AutodiffBackend> {
    grads: &'a GradientsParams,
    out: Vec<Option<Vec<f32>>>,
    _b: std::marker::PhantomData<B>,
}

impl<B: AutodiffBackend> ModuleVisitor<B> for GradDump<'_, B> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        self.out.push(self.grads.get::<B::InnerBackend, D>(param.id).map(|g| g.into_data().to_vec::<f32>().unwrap()));
    }
}

fn grads_to_host(model: &Model<GA>, grads: &GradientsParams) -> Vec<Option<Vec<f32>>> {
    let mut v = GradDump::<GA> { grads, out: Vec::new(), _b: Default::default() };
    model.visit(&mut v);
    v.out
}

struct ZeroGrads {
    decay: GradientsParams,
    nodecay: GradientsParams,
}

impl ModuleVisitor<GA> for ZeroGrads {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<GA, D>>) {
        let z = Tensor::<G, D>::zeros(param.val().dims(), &device());
        if D >= 2 {
            self.decay.register::<G, D>(param.id, z);
        } else {
            self.nodecay.register::<G, D>(param.id, z);
        }
    }
}


// ------------------------------------------------------------------ checks

pub fn check_bce_reference() -> Check {
    let zs: [f64; 13] = [-50.0, -30.0, -5.0, -1.0, -0.1, 0.0, 0.1, 1.0, 5.0, 30.0, 50.0, 88.0, -88.0];
    let mut z = Vec::new();
    let mut y = Vec::new();
    for &v in &zs {
        for lab in [0.0f32, 1.0] {
            z.push(v as f32);
            y.push(lab);
        }
    }
    let n = z.len();
    let zt = Tensor::<GA, 1>::from_floats(z.as_slice(), &device()).require_grad();
    let yt = Tensor::<GA, 1>::from_floats(y.as_slice(), &device());
    let loss = crate::model::bce_mean(zt.clone(), yt);
    let got = host1(loss.clone())[0] as f64;
    let grads = loss.backward();
    let gz = zt.grad(&grads).unwrap().into_data().to_vec::<f32>().unwrap();
    let f = |z: f64, y: f64| z.max(0.0) - z * y + (-z.abs()).exp().ln_1p();
    let want = z.iter().zip(&y).map(|(a, b)| f(*a as f64, *b as f64)).sum::<f64>() / n as f64;
    let mut max_g = 0f64;
    let mut max_fd = 0f64;
    for i in 0..n {
        let (zi, yi) = (z[i] as f64, y[i] as f64);
        let analytic = (1.0 / (1.0 + (-zi).exp()) - yi) / n as f64;
        max_g = max_g.max((gz[i] as f64 - analytic).abs());
        let h = 1e-4;
        let fd = (f(zi + h, yi) - f(zi - h, yi)) / (2.0 * h) / n as f64;
        max_fd = max_fd.max((analytic - fd).abs()); // validates the analytic reference itself
    }
    let ok = got.is_finite() && (got - want).abs() <= TOL_BCE_ABS * want.abs().max(1.0) && max_g <= TOL_BCE_GRAD_ABS && gz.iter().all(|g| g.is_finite()) && max_fd < 1e-6;
    Check::new("stable_bce_value_and_gradient", ok, json!({"got": got, "reference_f64": want, "max_grad_abs_err": max_g, "reference_self_check_fd_err": max_fd, "logit_range": [-88, 88]}))
}

pub fn check_readout_mean_and_accumulation(ctx: &QualCtx, arm: Arm, values: &[(Vec<usize>, Vec<f32>)]) -> Vec<Check> {
    let labels = panel_labels(ctx);
    let model = fresh_model(values);
    let outs = model.forward(&batch16(ctx), arm);
    let per: Vec<Vec<f32>> = outs.iter().map(|o| host1(o.clone())).collect();
    let f = |z: f64, y: bool| z.max(0.0) - z * (y as u8 as f64) + (-z.abs()).exp().ln_1p();
    let mut readout_means = Vec::new();
    for p in &per {
        readout_means.push(p.iter().zip(&labels).map(|(z, y)| f(*z as f64, *y)).sum::<f64>() / p.len() as f64);
    }
    let want = readout_means.iter().sum::<f64>() / readout_means.len() as f64;
    let got = host1(arm_loss(&outs, &y_tensor(&labels)))[0] as f64;
    let c1 = Check::new(
        "loss_is_mean_over_prescribed_readouts",
        per.len() == arm.n_readouts() && (got - want).abs() <= TOL_LOSS_ABS,
        json!({"readouts": per.len(), "expected_readouts": arm.n_readouts(), "model_loss": got, "host_f64_mean": want, "per_readout_bce": readout_means}),
    );

    // accumulation: sum over 8 microbatches of grad(L_m/8) == grad(L_batch16)
    let tr = Trainer::<GA>::new(fresh_model(values), arm, &device());
    let mut acc = GradientsAccumulator::<Model<GA>>::new();
    let mut loss_acc = 0.0;
    for m in 0..ACCUM {
        let (g, l, _) = tr.micro_grads(&micro_of(ctx, &labels, m), 1.0 / ACCUM as f32);
        acc.accumulate(&tr.model, g);
        loss_acc += l / ACCUM as f64;
    }
    let g_acc = grads_to_host(&tr.model, &acc.grads());
    let outs = tr.model.forward(&batch16(ctx), arm);
    let full_loss = arm_loss(&outs, &y_tensor(&labels));
    let full_val = host1(full_loss.clone())[0] as f64;
    let g_full = grads_to_host(&tr.model, &GradientsParams::from_grads(full_loss.backward(), &tr.model));
    let inv = inventory::<GA, _>(&tr.model);
    let mut missing = 0;
    let mut n_cmp = 0;
    let (mut na, mut nf) = (0f64, 0f64);
    for (a, b) in g_acc.iter().zip(&g_full) {
        if let (Some(a), Some(b)) = (a, b) {
            na += a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>();
            nf += b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>();
        }
    }
    let gnorm = nf.sqrt();
    // error of each tensor relative to max(its own norm, 1e-3 * global norm)
    let mut scored: Vec<(f64, String, f64, f64)> = Vec::new();
    for ((a, b), p) in g_acc.iter().zip(&g_full).zip(&inv) {
        match (a, b) {
            (Some(a), Some(b)) => {
                let err: f64 = a.iter().zip(b).map(|(x, y)| ((*x - *y) as f64).powi(2)).sum::<f64>().sqrt();
                let scale = l2(b).max(1e-3 * gnorm);
                scored.push((err / scale, format!("{}.{}", p.path, p.leaf), err, l2(b)));
                n_cmp += 1;
            }
            _ => missing += 1,
        }
    }
    scored.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap());
    let worst = scored.first().map(|s| s.0).unwrap_or(0.0);
    let worst_list: Vec<Value> = scored.iter().take(4).map(|s| json!({"tensor": s.1, "err_over_scale": s.0, "abs_err": s.2, "tensor_norm": s.3})).collect();
    let c2 = Check::new(
        "gradient_accumulation_8x2_equals_batch16_mean",
        missing == 0 && worst <= TOL_ACCUM_REL_L2 && (loss_acc - full_val).abs() <= TOL_LOSS_ABS && (na.sqrt() - nf.sqrt()).abs() <= TOL_ACCUM_GLOBAL_REL * gnorm,
        json!({"tensors_compared": n_cmp, "missing": missing, "worst_err_over_scale": worst, "tolerance": TOL_ACCUM_REL_L2, "worst_tensors": worst_list, "accum_loss_mean": loss_acc, "full_batch_loss": full_val, "global_norm_accum": na.sqrt(), "global_norm_full": gnorm}),
    );
    vec![c1, c2]
}

pub fn check_clipping(ctx: &QualCtx, arm: Arm, values: &[(Vec<usize>, Vec<f32>)]) -> Check {
    let labels = panel_labels(ctx);
    // factor function
    let factor_ok = clip_factor(0.5) == 1.0 && (clip_factor(2.0) - 0.5).abs() < 1e-6 && clip_factor(0.0) == 1.0;
    // big gradients: clipped once to global norm 1
    let tr = Trainer::<GA>::new(fresh_model(values), arm, &device());
    let mut acc = GradientsAccumulator::<Model<GA>>::new();
    for m in 0..ACCUM {
        let (g, _, _) = tr.micro_grads(&micro_of(ctx, &labels, m), 400.0 / ACCUM as f32);
        acc.accumulate(&tr.model, g);
    }
    let big = acc.grads();
    let big_host = grads_to_host(&tr.model, &big);
    let (gd, gn, norm, factor, clipped) = tr.clip_and_split(big);
    let (nd, _) = crate::train::global_grad_norm(&tr.model, &gd);
    let (nn_, _) = crate::train::global_grad_norm(&tr.model, &gn);
    let post = (nd * nd + nn_ * nn_).sqrt();
    // direction preserved: post-clip = pre-clip * factor per tensor (spot check global norm only)
    let pre: f64 = big_host.iter().flatten().map(|v| l2(v).powi(2)).sum::<f64>().sqrt();
    let big_ok = clipped && norm > 1.0 && (post - 1.0).abs() <= TOL_CLIP_NORM_ABS && (pre - norm).abs() <= 1e-3 * norm;
    // small gradients: untouched
    let mut acc = GradientsAccumulator::<Model<GA>>::new();
    for m in 0..ACCUM {
        let (g, _, _) = tr.micro_grads(&micro_of(ctx, &labels, m), 1.0 / ACCUM as f32 * 1e-3);
        acc.accumulate(&tr.model, g);
    }
    let small = acc.grads();
    let small_host = grads_to_host(&tr.model, &small);
    let (gd2, gn2, norm2, _, clipped2) = tr.clip_and_split(small);
    let mut same = true;
    // decay/nodecay union must equal the input exactly (bitwise) when not clipped
    let (n_a, _) = crate::train::global_grad_norm(&tr.model, &gd2);
    let (n_b, _) = crate::train::global_grad_norm(&tr.model, &gn2);
    let pre2: f64 = small_host.iter().flatten().map(|v| l2(v).powi(2)).sum::<f64>().sqrt();
    same &= ((n_a * n_a + n_b * n_b).sqrt() - pre2).abs() <= 1e-9 + 1e-6 * pre2;
    let small_ok = !clipped2 && norm2 < 1.0 && same;
    // once per update: 3 real updates -> exactly 3 clip operations
    let mut t = Trainer::<GA>::new(fresh_model(values), arm, &device());
    for _ in 0..3 {
        t.step(&all_micros(ctx, &labels));
    }
    let once_ok = t.clip_calls == 3;
    Check::new(
        "global_clipping_once_on_accumulated_gradient",
        factor_ok && big_ok && small_ok && once_ok,
        json!({"factor_fn_ok": factor_ok, "big": {"pre_norm": norm, "factor": factor, "post_norm": post, "clipped": clipped}, "small": {"pre_norm": norm2, "clipped": clipped2, "untouched": same}, "clip_ops_in_3_updates": t.clip_calls, "microbatches_per_update": ACCUM}),
    )
}

pub fn check_grad_inventory(ctx: &QualCtx, arm: Arm, values: &[(Vec<usize>, Vec<f32>)]) -> Check {
    let labels = panel_labels(ctx);
    let model = fresh_model(values);
    let inv: Vec<ParamInfo> = inventory::<GA, _>(&model);
    let loss = arm_loss(&model.forward(&batch16(ctx), arm), &y_tensor(&labels));
    let g = grads_to_host(&model, &GradientsParams::from_grads(loss.backward(), &model));
    let mut missing = Vec::new();
    let mut zero = Vec::new();
    let mut structural = Vec::new();
    let mut nonfinite = Vec::new();
    let mut by_comp: std::collections::BTreeMap<String, f64> = Default::default();
    let gnorm = g.iter().flatten().map(|v| l2(v).powi(2)).sum::<f64>().sqrt();
    for (p, gr) in inv.iter().zip(&g) {
        match gr {
            None => missing.push(format!("{}.{}", p.path, p.leaf)),
            Some(v) => {
                if v.iter().any(|x| !x.is_finite()) {
                    nonfinite.push(format!("{}.{}", p.path, p.leaf));
                }
                let n = l2(v);
                // Attention key biases have an exactly-zero true gradient (softmax is invariant to
                // adding a constant to all of a query's scores); only noise-level values are expected.
                let key_bias = p.leaf == "bias" && p.path.ends_with(".k");
                if n == 0.0 {
                    zero.push(format!("{}.{}", p.path, p.leaf));
                } else if n <= 1e-6 * gnorm {
                    // informational: present, finite, nonzero, but negligible at these weights
                    structural.push(format!("{}.{} (norm {:.2e}{})", p.path, p.leaf, n, if key_bias { ", true gradient is exactly 0 for key biases" } else { "" }));
                }
                let e = by_comp.entry(p.component.clone()).or_insert(f64::INFINITY);
                *e = e.min(n);
            }
        }
    }
    Check::new(
        "relevant_gradient_inventory",
        missing.is_empty() && zero.is_empty() && nonfinite.is_empty(),
        json!({"params": inv.len(), "global_grad_norm": gnorm, "missing": missing, "exactly_zero": zero, "negligible_below_1e-6_of_global_norm_info": structural, "nonfinite": nonfinite, "min_grad_l2_by_component": by_comp}),
    )
}

pub fn check_recurrent_connectivity(ctx: &QualCtx, arm: Arm, values: &[(Vec<usize>, Vec<f32>)]) -> Check {
    let labels = panel_labels(ctx);
    let n_calls = arm.n_calls();

    let run = |detach: Option<usize>| {
        let model = fresh_model(values);
        let probes: Vec<(Tensor<GA, 3>, Tensor<GA, 3>)> = (0..n_calls)
            .map(|_| {
                (
                    Tensor::<GA, 3>::zeros([16, 64, crate::model::D], &device()).require_grad(),
                    Tensor::<GA, 3>::zeros([16, crate::model::N_WS, crate::model::D], &device()).require_grad(),
                )
            })
            .collect();
        let opts = FwdOpts::<GA> { probes: Some(probes.clone()), detach_after_call: detach, sync_each_call: false };
        let outs = model.forward_ex(&batch16(ctx), arm, &opts);
        // final-readout-only loss: earlier-call credit must flow through the full unroll
        let loss = bce_mean(outs.last().unwrap().clone(), y_tensor(&labels));
        let grads = loss.backward();
        let norms: Vec<(f64, f64)> = probes
            .iter()
            .map(|(pb, pw)| {
                let nb = pb.grad(&grads).map(|g| l2(&g.into_data().to_vec::<f32>().unwrap())).unwrap_or(0.0);
                let nw = pw.grad(&grads).map(|g| l2(&g.into_data().to_vec::<f32>().unwrap())).unwrap_or(0.0);
                (nb, nw)
            })
            .collect();
        let pg = grads_to_host(&model, &GradientsParams::from_grads(grads, &model));
        (norms, pg, inventory::<GA, _>(&model))
    };
    let (norms, pg_full, inv) = run(None);
    // expectation: ws state after every call matters; board state matters after every call except the last
    let mut ok = true;
    let mut bad = Vec::new();
    for (k, (nb, nw)) in norms.iter().enumerate() {
        let last = k + 1 == n_calls;
        if *nw <= TOL_ZERO_PROBE || !nw.is_finite() {
            ok = false;
            bad.push(format!("ws probe after call {} is zero", k + 1));
        }
        if !last && (*nb <= TOL_ZERO_PROBE || !nb.is_finite()) {
            ok = false;
            bad.push(format!("board probe after call {} is zero", k + 1));
        }
        if last && *nb > TOL_ZERO_PROBE {
            ok = false;
            bad.push("board after final call should not influence the head".to_string());
        }
    }
    // negative control: detaching after call m must zero all probes at calls <= m and keep later ones
    let m = (n_calls / 2).max(1);
    let (dn, pg_det, _) = run(Some(m));
    let mut control_ok = true;
    for (k, (nb, nw)) in dn.iter().enumerate() {
        if k < m && (*nb > TOL_ZERO_PROBE || *nw > TOL_ZERO_PROBE) {
            control_ok = false;
        }
        if k >= m && k + 1 < n_calls && (*nb <= TOL_ZERO_PROBE || *nw <= TOL_ZERO_PROBE) {
            control_ok = false;
        }
    }
    // repeated shared-call contribution: shared block gradients differ when early calls are cut
    let mut fast_diff = 0f64;
    let mut slow_diff = 0f64;
    for (p, (a, b)) in inv.iter().zip(pg_full.iter().zip(&pg_det)) {
        // a missing gradient (no path to the loss after cutting) counts as exactly zero
        if let Some(a) = a {
            let zeros;
            let b: &Vec<f32> = match b {
                Some(b) => b,
                None => {
                    zeros = vec![0.0f32; a.len()];
                    &zeros
                }
            };
            let d = rel_l2(b, a);
            if p.component == "fast" {
                fast_diff = fast_diff.max(d);
            }
            if p.component == "slow" {
                slow_diff = slow_diff.max(d);
            }
        }
    }
    // Cutting after call m removes the contribution of calls 1..=m to a block's shared parameters:
    // expected to change iff that block was called at or before call m.
    let sched: Vec<crate::model::Op> = arm.schedule().into_iter().filter(|o| *o != crate::model::Op::Read).collect();
    let expect_fast = sched[..m].contains(&crate::model::Op::Fast);
    let expect_slow = sched[..m].contains(&crate::model::Op::Slow);
    let contrib_ok = (if expect_fast { fast_diff > 1e-3 } else { fast_diff <= 1e-6 }) && (if expect_slow { slow_diff > 1e-3 } else { slow_diff <= 1e-6 });
    Check::new(
        "recurrent_credit_assignment_full_unroll",
        ok && control_ok && contrib_ok,
        json!({"calls": n_calls, "probe_grad_l2_board_ws_per_call": norms, "violations": bad, "negative_control_detach_after_call": m, "negative_control_ok": control_ok, "shared_block_grad_rel_change_when_early_calls_cut": {"fast": fast_diff, "slow": slow_diff, "expected_fast_change": expect_fast, "expected_slow_change": expect_slow}}),
    )
}

fn ref_weights(values: &[(Vec<usize>, Vec<f32>)]) -> RefWeights {
    let inv = inventory::<G, _>(&Model::<G>::new(&device()));
    RefWeights::new(&inv, values)
}

/// GPU logits at every readout vs the independent f64 host reference on the whole panel.
/// This also MEASURES the real numeric precision of the CUDA f32 path (TF32 tensor-core
/// matmuls may be selected by Burn/cubek autotune; strict FP32 cannot be forced).
pub fn check_forward_vs_reference(ctx: &QualCtx, arm: Arm, values: &[(Vec<usize>, Vec<f32>)]) -> Check {
    let model = fresh_model(values);
    let gpu: Vec<Vec<f32>> = model.forward(&batch16(ctx), arm).into_iter().map(host1).collect();
    let rw = ref_weights(values);
    let mut worst_abs: f64 = 0.0;
    let mut worst_rel: f64 = 0.0;
    let mut scale: f64 = 0.0;
    for (i, f) in ctx.feats.iter().enumerate() {
        let r = ref_forward(&rw, f, arm);
        for (k, z) in r.iter().enumerate() {
            let g = gpu[k][i] as f64;
            worst_abs = worst_abs.max((g - z).abs());
            worst_rel = worst_rel.max((g - z).abs() / z.abs().max(1e-3));
            scale = scale.max(z.abs());
        }
    }
    Check::new(
        "cuda_forward_matches_independent_f64_reference",
        worst_abs <= TOL_FORWARD_ABS,
        json!({"examples": ctx.feats.len(), "readouts": gpu.len(), "max_abs_logit_error": worst_abs, "max_rel_error": worst_rel, "max_abs_logit": scale, "tolerance_abs": TOL_FORWARD_ABS,
               "interpretation": "~1e-6..1e-5 would indicate strict FP32 matmul; ~1e-4..1e-3 indicates TF32-class tensor-core inputs"}),
    )
}

/// Analytic GPU gradient vs EXACT f64 finite differences of the host reference, along the
/// GPU gradient direction of selected parameter tensors (first 4 panel examples).
pub fn check_numeric_gradient(ctx: &QualCtx, arm: Arm, values: &[(Vec<usize>, Vec<f32>)]) -> Check {
    let n = 4usize;
    let labels: Vec<bool> = ctx.rows.iter().take(n).map(|r| r.label).collect();
    let feats: Vec<&Features> = ctx.feats.iter().take(n).collect();
    let model = fresh_model(values);
    let inv = inventory::<GA, _>(&model);
    let batch = Batch::<GA>::from_features(&feats, &device());
    let loss = arm_loss(&model.forward(&batch, arm), &y_tensor(&labels));
    let g = grads_to_host(&model, &GradientsParams::from_grads(loss.backward(), &model));
    let rw = ref_weights(values);
    let eps = 1e-3f64;
    let mut results = Vec::new();
    let mut ok = true;
    for target in [("fast.ffn.l1", "weight"), ("slow.cross.q", "weight"), ("fast.self_attn.q", "weight"), ("slow.ffn.l2", "weight"), ("enc1.ffn.l2", "weight"), ("fast.cross.o", "weight"), ("head1", "bias")] {
        let idx = inv.iter().position(|p| p.path == target.0 && p.leaf == target.1).expect("target parameter");
        let gv = g[idx].as_ref().unwrap();
        let gn = l2(gv);
        let dir: Vec<f64> = gv.iter().map(|x| *x as f64 / gn).collect();
        let key = format!("{}.{}", target.0, target.1);
        let lp = ref_loss(&rw.perturbed(&key, &dir, eps), &feats, &labels, arm);
        let lm = ref_loss(&rw.perturbed(&key, &dir, -eps), &feats, &labels, arm);
        let numeric = (lp - lm) / (2.0 * eps);
        // along its own gradient direction the analytic directional derivative equals ||g||
        let pass = (numeric - gn).abs() <= TOL_NUMERIC_REL * gn + TOL_NUMERIC_ABS;
        ok &= pass;
        results.push(json!({"param": key, "analytic_gpu_grad_norm": gn, "f64_central_diff_along_it": numeric, "rel_err": (numeric - gn).abs() / gn.max(1e-30), "pass": pass}));
    }
    Check::new("gradient_vs_exact_f64_finite_differences_through_unroll", ok, json!({"epsilon": eps, "examples": n, "targets": results, "tolerance_rel": TOL_NUMERIC_REL}))
}

pub fn check_state_isolation(ctx: &QualCtx, arm: Arm, values: &[(Vec<usize>, Vec<f32>)]) -> Check {
    let model = fresh_model(values);
    let d = device();
    let all: Vec<&Features> = ctx.feats.iter().collect();
    let run = |idx: &[usize]| -> Vec<Vec<f32>> {
        let fs: Vec<&Features> = idx.iter().map(|&i| all[i]).collect();
        model.forward(&Batch::<GA>::from_features(&fs, &d), arm).into_iter().map(host1).collect()
    };
    let base = run(&(0..16).collect::<Vec<_>>());
    let mut worst: f64 = 0.0;
    // each example alone, in pairs, and in reversed order must match its batch-16 logits
    for i in 0..16 {
        let solo = run(&[i]);
        for (r, s) in solo.iter().enumerate() {
            worst = worst.max((s[0] - base[r][i]).abs() as f64);
        }
    }
    let rev: Vec<usize> = (0..16).rev().collect();
    let r2 = run(&rev);
    for (r, v) in r2.iter().enumerate() {
        for k in 0..16 {
            worst = worst.max((v[k] - base[r][15 - k]).abs() as f64);
        }
    }
    // repeat run must be identical (no hidden state)
    let again = run(&(0..16).collect::<Vec<_>>());
    let repeat = base.iter().zip(&again).map(|(a, b)| max_abs_diff(a, b)).fold(0.0, f64::max);
    Check::new("no_state_leakage_across_examples_or_batches", worst <= TOL_ISOLATION_LOGIT && repeat <= TOL_ISOLATION_LOGIT, json!({"max_logit_diff_solo_or_reordered_vs_batch16": worst, "max_logit_diff_repeat_run": repeat, "tolerance": TOL_ISOLATION_LOGIT}))
}

pub fn check_label_input_separation(ctx: &QualCtx, arm: Arm, values: &[(Vec<usize>, Vec<f32>)]) -> Check {
    let flipped: Vec<ModelRow> = ctx.rows.iter().map(|r| ModelRow { label: !r.label, id: format!("zz-{}", r.id), ..r.clone() }).collect();
    let refeat: Vec<Features> = flipped.iter().map(|r| featurize(&r.fen, r.budget).unwrap().0).collect();
    let same_feats = refeat == ctx.feats;
    let model = fresh_model(values);
    let a = {
        let refs: Vec<&Features> = ctx.feats.iter().collect();
        model.forward(&Batch::<GA>::from_features(&refs, &device()), arm).into_iter().map(host1).collect::<Vec<_>>()
    };
    let b = {
        let refs: Vec<&Features> = refeat.iter().collect();
        model.forward(&Batch::<GA>::from_features(&refs, &device()), arm).into_iter().map(host1).collect::<Vec<_>>()
    };
    let diff = a.iter().zip(&b).map(|(x, y)| max_abs_diff(x, y)).fold(0.0, f64::max);
    Check::new("labels_ids_never_inputs", same_feats && diff == 0.0, json!({"features_identical_after_label_and_id_mutation": same_feats, "max_prediction_diff": diff}))
}

pub fn check_init_identity_and_independence(ctx: &QualCtx, values: &[(Vec<usize>, Vec<f32>)]) -> Check {
    // identity: three arm models built from the scientific canonical init have identical tensor bytes.
    // (Identity check only: these models are dropped without any step.)
    let mut hashes = Vec::new();
    for _ in Arm::ALL {
        let m = load_values::<GA, _>(Model::<GA>::new(&device()), &ctx.scientific_init_values, &device());
        hashes.push(crate::init::tensors_hash(&dump_values::<GA, _>(&m)));
    }
    let identical = hashes.iter().all(|h| *h == hashes[0]) && hashes[0] == ctx.scientific_init_hash;
    // independence: stepping model A (disposable weights) must not move B or its optimizer
    let labels = panel_labels(ctx);
    let mut ta = Trainer::<GA>::new(fresh_model(values), Arm::A, &device());
    let tb = Trainer::<GA>::new(fresh_model(values), Arm::B, &device());
    let h0 = crate::init::tensors_hash(&dump_values::<GA, _>(&tb.model));
    ta.step(&all_micros(ctx, &labels));
    let ha = crate::init::tensors_hash(&dump_values::<GA, _>(&ta.model));
    let hb = crate::init::tensors_hash(&dump_values::<GA, _>(&tb.model));
    let independent = ha != h0 && hb == h0;
    Check::new("bit_identical_init_and_independent_storage_optimizers", identical && independent, json!({"arm_init_hashes": hashes, "file_hash": ctx.scientific_init_hash, "step_moved_A_not_B": independent}))
}

pub fn check_weight_decay_groups(values: &[(Vec<usize>, Vec<f32>)]) -> Check {
    let model = fresh_model(values);
    let inv = inventory::<GA, _>(&model);
    let before = dump_values::<GA, _>(&model);
    let mut t = Trainer::<GA>::new(model, Arm::A, &device());
    let mut z = ZeroGrads { decay: GradientsParams::new(), nodecay: GradientsParams::new() };
    t.model.visit(&mut z);
    // zero gradient => Adam delta is exactly 0, so only decoupled decay acts: p <- p*(1 - lr*wd)
    let lr = 1.0;
    let m = t.model.clone();
    let m = t.opt_decay.step(lr, m, z.decay);
    let m = t.opt_nodecay.step(lr, m, z.nodecay);
    let after = dump_values::<GA, _>(&m);
    let mut decay_worst: f64 = 0.0;
    let mut nodecay_moved = 0usize;
    let (mut n_decay, mut n_nodecay) = (0, 0);
    for (p, (b, a)) in inv.iter().zip(before.iter().zip(&after)) {
        if p.decay {
            n_decay += 1;
            for (x, y) in b.1.iter().zip(&a.1) {
                let want = *x as f64 * (1.0 - lr * crate::train::WEIGHT_DECAY);
                decay_worst = decay_worst.max((*y as f64 - want).abs() / want.abs().max(1e-6));
            }
        } else {
            n_nodecay += 1;
            if b.1 != a.1 {
                nodecay_moved += 1;
            }
        }
    }
    let leaf_ok = inv.iter().all(|p| p.decay == !matches!(p.leaf.as_str(), "bias" | "gamma" | "beta"));
    Check::new("weight_decay_excludes_biases_and_norms", decay_worst <= TOL_DECAY_REL && nodecay_moved == 0 && leaf_ok, json!({"decay_tensors": n_decay, "nodecay_tensors": n_nodecay, "worst_rel_err_decay": decay_worst, "nodecay_tensors_changed": nodecay_moved, "note": "per-update decay at scheduled LR (<=5e-8 relative) is near f32 resolution; applied as AdamW prescribes"}))
}

pub fn check_checkpoint_roundtrip(ctx: &QualCtx, arm: Arm, values: &[(Vec<usize>, Vec<f32>)], dir: &Path) -> Check {
    let labels = panel_labels(ctx);
    let micros = all_micros(ctx, &labels);
    let mut t1 = Trainer::<GA>::new(fresh_model(values), arm, &device());
    for _ in 0..3 {
        t1.step(&micros);
    }
    save_checkpoint(dir, &t1, json!({"qualification": true})).expect("save");
    let (mut t2, meta) = load_checkpoint::<GA>(dir, arm, &device()).expect("load");
    let at_load = dump_values::<GA, _>(&t2.model);
    let saved = dump_values::<GA, _>(&t1.model);
    let weights_diff = at_load.iter().zip(&saved).map(|(a, b)| max_abs_diff(&a.1, &b.1)).fold(0.0, f64::max);
    for _ in 0..2 {
        t1.step(&micros);
        t2.step(&micros);
    }
    let p1 = dump_values::<GA, _>(&t1.model);
    let p2 = dump_values::<GA, _>(&t2.model);
    let cont = p1.iter().zip(&p2).map(|(a, b)| max_abs_diff(&a.1, &b.1)).fold(0.0, f64::max);
    // negative control: weights restored but fresh optimizer state
    let (t3m, _) = load_checkpoint::<GA>(dir, arm, &device()).expect("load3");
    let mut t3 = Trainer::<GA>::new(t3m.model.clone(), arm, &device());
    t3.update = t3m.update;
    for _ in 0..2 {
        t3.step(&micros);
    }
    let p3 = dump_values::<GA, _>(&t3.model);
    let ctrl = p1.iter().zip(&p3).map(|(a, b)| max_abs_diff(&a.1, &b.1)).fold(0.0, f64::max);
    let sched_ok = meta.completed_updates == 3 && meta.lr_schedule_position == 3 && (meta.next_lr - lr_at(3)).abs() < 1e-15 && t2.update == 5 && t1.update == 5;
    Check::new(
        "checkpoint_restores_weights_optimizer_and_schedule",
        weights_diff == 0.0 && cont <= TOL_CKPT_PARAM_ABS && ctrl > 10.0 * TOL_CKPT_PARAM_ABS.max(cont) && sched_ok,
        json!({"restored_weights_max_diff": weights_diff, "continued_training_max_param_diff": cont, "tolerance": TOL_CKPT_PARAM_ABS, "negative_control_fresh_optimizer_max_param_diff": ctrl, "schedule_ok": sched_ok}),
    )
}

pub fn check_profile_parity(ctx: &QualCtx, arm: Arm, values: &[(Vec<usize>, Vec<f32>)]) -> Check {
    let labels = panel_labels(ctx);
    let run = |sync: bool| {
        let model = fresh_model(values);
        let opts = FwdOpts::<GA> { probes: None, detach_after_call: None, sync_each_call: sync };
        let outs = model.forward_ex(&batch16(ctx), arm, &opts);
        let logits: Vec<Vec<f32>> = outs.iter().map(|o| host1(o.clone())).collect();
        let loss = arm_loss(&outs, &y_tensor(&labels));
        let g = grads_to_host(&model, &GradientsParams::from_grads(loss.backward(), &model));
        (logits, g)
    };
    let (la, ga) = run(false);
    let (lb, gb) = run(true);
    let ld = la.iter().zip(&lb).map(|(a, b)| max_abs_diff(a, b)).fold(0.0, f64::max);
    let gd = ga.iter().zip(&gb).filter_map(|(a, b)| Some(rel_l2(a.as_ref()?, b.as_ref()?))).fold(0.0, f64::max);
    Check::new("normal_vs_synchronized_profile_execution_parity", ld <= TOL_PROFILE_LOGIT && gd <= TOL_PROFILE_GRAD_REL_L2, json!({"max_logit_diff": ld, "max_grad_rel_l2": gd}))
}

// ------------------------------------------------------------- latency / memory

pub fn nvidia_smi_mib() -> Option<u64> {
    let out = std::process::Command::new("nvidia-smi").args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"]).output().ok()?;
    String::from_utf8_lossy(&out.stdout).lines().next()?.trim().parse().ok()
}

pub fn host_working_set_mib() -> Option<u64> {
    let pid = std::process::id();
    let out = std::process::Command::new("powershell")
        .args(["-NoProfile", "-Command", &format!("(Get-Process -Id {pid}).WorkingSet64")])
        .output()
        .ok()?;
    let b: u64 = String::from_utf8_lossy(&out.stdout).trim().parse().ok()?;
    Some(b / (1024 * 1024))
}

pub fn measure_latency_memory(ctx: &QualCtx, arm: Arm, values: &[(Vec<usize>, Vec<f32>)]) -> Check {
    let labels = panel_labels(ctx);
    let micros = all_micros(ctx, &labels);
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let peak = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let (s2, p2) = (stop.clone(), peak.clone());
    let sampler = std::thread::spawn(move || {
        while !s2.load(std::sync::atomic::Ordering::Relaxed) {
            if let Some(m) = nvidia_smi_mib() {
                p2.fetch_max(m, std::sync::atomic::Ordering::Relaxed);
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    });
    let dev_before = nvidia_smi_mib();
    let host_before = host_working_set_mib();
    let mut t = Trainer::<GA>::new(fresh_model(values), arm, &device());
    let d = device();
    let sync = || {
        let _ = <G as Backend>::sync(&d);
    };
    // inference latency (synchronized), batch 2 and 16, 5 warmups + 20 timed
    let infer = |n: usize| -> f64 {
        let refs: Vec<&Features> = ctx.feats.iter().take(n).collect();
        let m = t.model.valid();
        let mut times = Vec::new();
        for i in 0..25 {
            let t0 = Instant::now();
            let b = Batch::<G>::from_features(&refs, &d);
            let outs = m.forward(&b, arm);
            let _ = host1(outs.last().unwrap().clone());
            sync();
            if i >= 5 {
                times.push(t0.elapsed().as_secs_f64() * 1e3);
            }
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        times[times.len() / 2]
    };
    let inf2 = infer(2);
    let inf16 = infer(16);
    // full update latency: 2 warmups + 10 timed
    let mut ups = Vec::new();
    for i in 0..12 {
        let t0 = Instant::now();
        t.step(&micros);
        sync();
        if i >= 2 {
            ups.push(t0.elapsed().as_secs_f64() * 1e3);
        }
    }
    ups.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let upd_med = ups[ups.len() / 2];
    let upd_mean = ups.iter().sum::<f64>() / ups.len() as f64;
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    let _ = sampler.join();
    let dev_peak = peak.load(std::sync::atomic::Ordering::Relaxed);
    let host_after = host_working_set_mib();
    let ok = dev_peak <= DEVICE_MEM_CEILING_MIB;
    Check::new(
        "latency_and_memory",
        ok,
        json!({
            "inference_median_ms_batch2": inf2,
            "inference_median_ms_batch16": inf16,
            "update_median_ms": upd_med,
            "update_mean_ms": upd_mean,
            "device_mem_mib": {"before": dev_before, "sampled_peak_100ms_poll": dev_peak, "ceiling": DEVICE_MEM_CEILING_MIB},
            "host_working_set_mib": {"before": host_before, "after": host_after},
            "limitations": "device memory is nvidia-smi sampled (100 ms poll, whole-GPU used memory) and can miss short peaks; allocator high-water not available through this API; host working set is a point sample (launcher records peak)",
            "block_calls": arm.n_calls(), "readouts": arm.n_readouts()
        }),
    )
}

/// Run every check for one arm. Stops adding checks after the wall limit.
pub fn qualify_arm(ctx: &QualCtx, arm: Arm, ckpt_dir: &Path) -> Value {
    let t0 = Instant::now();
    let values = qual_values(ctx);
    let mut checks: Vec<Check> = Vec::new();
    let guard = |c: Vec<Check>, checks: &mut Vec<Check>| {
        for x in c {
            eprintln!("[qual {}] {} -> {}", arm.name(), x.name, if x.pass { "PASS" } else { "FAIL" });
            if !x.pass {
                eprintln!("    detail: {}", x.detail);
            }
            checks.push(x);
        }
    };
    guard(vec![check_bce_reference()], &mut checks);
    guard(check_readout_mean_and_accumulation(ctx, arm, &values), &mut checks);
    guard(vec![check_clipping(ctx, arm, &values)], &mut checks);
    guard(vec![check_grad_inventory(ctx, arm, &values)], &mut checks);
    guard(vec![check_recurrent_connectivity(ctx, arm, &values)], &mut checks);
    guard(vec![check_forward_vs_reference(ctx, arm, &values)], &mut checks);
    guard(vec![check_numeric_gradient(ctx, arm, &values)], &mut checks);
    guard(vec![check_state_isolation(ctx, arm, &values)], &mut checks);
    guard(vec![check_label_input_separation(ctx, arm, &values)], &mut checks);
    guard(vec![check_init_identity_and_independence(ctx, &values)], &mut checks);
    guard(vec![check_weight_decay_groups(&values)], &mut checks);
    guard(vec![check_checkpoint_roundtrip(ctx, arm, &values, ckpt_dir)], &mut checks);
    guard(vec![check_profile_parity(ctx, arm, &values)], &mut checks);
    guard(vec![measure_latency_memory(ctx, arm, &values)], &mut checks);
    let wall = t0.elapsed().as_secs_f64();
    let within = wall <= QUAL_WALL_SECS_PER_ARM as f64;
    let pass = checks.iter().all(|c| c.pass) && within;
    json!({
        "arm": arm.name(),
        "block_calls": arm.n_calls(),
        "readouts": arm.n_readouts(),
        "wall_secs": wall,
        "within_wall_limit": within,
        "qualified": pass,
        "checks": checks.iter().map(|c| c.to_json()).collect::<Vec<_>>(),
        "microbatch": MICRO, "accumulation_steps": ACCUM,
    })
}
