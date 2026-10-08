//! D1 qualification (CUDA, RTX 2050): the direct-board MLP and D1-A through the generic D1
//! trainer. Disposable weights from the `d1_qualification_init` stream; never used for a fit.
//! Tolerances are those of the E1 qualification (see qualify.rs); backend precision is adopted
//! as measured: f32 storage/accumulation, matmul inputs possibly TF32.

use crate::d1::{D1_ACCUM, DTrainer, DiagModel, Mlp, load_ckpt, mlp_input_vec, save_ckpt};
use crate::init;
use crate::inventory::{ParamInfo, dump_values, inventory, load_values};
use crate::model::{Arm, Batch, Model};
use crate::qualify::*;
use crate::reference::{RefWeights, mlp_ref_forward, mlp_ref_loss, ref_forward};
use burn::module::{AutodiffModule, Module, ModuleVisitor, Param};
use burn::optim::{GradientsAccumulator, GradientsParams, Optimizer};
use burn::prelude::*;
use recur64_v69::features::{Features, ModelRow, featurize};
use recur64_v69::streams::MasterSeed;
use serde_json::{Value, json};
use std::path::Path;

pub struct D1QualCtx {
    pub rows: Vec<ModelRow>,
    pub feats: Vec<Features>,
    pub seed: MasterSeed,
    /// E1 canonical untrained init tensors (forward-equivalence check only; no training).
    pub e1_init_values: Vec<(Vec<usize>, Vec<f32>)>,
}

type Vals = Vec<(Vec<usize>, Vec<f32>)>;

struct GDump<'a> {
    grads: &'a GradientsParams,
    out: Vec<Option<Vec<f32>>>,
}

impl ModuleVisitor<GA> for GDump<'_> {
    fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<GA, D>>) {
        self.out.push(self.grads.get::<G, D>(p.id).map(|g| g.into_data().to_vec::<f32>().unwrap()));
    }
}

fn grads_host<M: Module<GA>>(m: &M, g: &GradientsParams) -> Vec<Option<Vec<f32>>> {
    let mut v = GDump { grads: g, out: Vec::new() };
    m.visit(&mut v);
    v.out
}

struct Zero {
    decay: GradientsParams,
    nodecay: GradientsParams,
}

impl ModuleVisitor<GA> for Zero {
    fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<GA, D>>) {
        let z = Tensor::<G, D>::zeros(p.val().dims(), &device());
        if D >= 2 {
            self.decay.register::<G, D>(p.id, z);
        } else {
            self.nodecay.register::<G, D>(p.id, z);
        }
    }
}

fn labels(ctx: &D1QualCtx) -> Vec<bool> {
    ctx.rows.iter().map(|r| r.label).collect()
}

fn micro<'a>(ctx: &'a D1QualCtx, lab: &[bool], i: usize) -> (Vec<&'a Features>, Vec<bool>) {
    (vec![&ctx.feats[2 * i], &ctx.feats[2 * i + 1]], vec![lab[2 * i], lab[2 * i + 1]])
}

fn micros<'a>(ctx: &'a D1QualCtx, lab: &[bool]) -> Vec<(Vec<&'a Features>, Vec<bool>)> {
    (0..D1_ACCUM).map(|i| micro(ctx, lab, i)).collect()
}

fn logits_host<M: DiagModel<GA>>(m: &M, feats: &[&Features]) -> Vec<f32> {
    host1(m.logits(feats, &device()))
}

/// Checks shared by both models, through the generic D1 trainer.
fn machinery<M>(name: &str, ctx: &D1QualCtx, build: &dyn Fn(&Vals) -> M, values: &Vals) -> Vec<Check>
where
    M: DiagModel<GA> + AutodiffModule<GA>,
    M::InnerModule: DiagModel<G>,
{
    let lab = labels(ctx);
    let all: Vec<&Features> = ctx.feats.iter().take(16).collect();
    let y16 = Tensor::<GA, 1>::from_floats(lab.iter().take(16).map(|&l| l as u8 as f32).collect::<Vec<_>>().as_slice(), &device());
    let mut out = Vec::new();
    let tag = |s: &str| format!("{name}:{s}");

    // --- accumulation 8x2 == batch 16 ; gradient inventory ---
    let tr = DTrainer::<GA, M>::new(build(values), &device());
    let mut acc = GradientsAccumulator::<M>::new();
    let mut loss_acc = 0.0;
    for m in 0..D1_ACCUM {
        let (f, l) = micro(ctx, &lab, m);
        let (g, lv) = tr.micro_grads(&f, &l, 1.0 / D1_ACCUM as f32);
        acc.accumulate(&tr.model, g);
        loss_acc += lv / D1_ACCUM as f64;
    }
    let g_acc = grads_host(&tr.model, &acc.grads());
    let full = crate::model::bce_mean(tr.model.logits(&all, &device()), y16);
    let full_val = host1(full.clone())[0] as f64;
    let g_full = grads_host(&tr.model, &GradientsParams::from_grads(full.backward(), &tr.model));
    let inv: Vec<ParamInfo> = inventory::<GA, _>(&tr.model);
    let gnorm = g_full.iter().flatten().map(|v| l2(v).powi(2)).sum::<f64>().sqrt();
    let anorm = g_acc.iter().flatten().map(|v| l2(v).powi(2)).sum::<f64>().sqrt();
    let mut worst = 0f64;
    let (mut missing, mut zero, mut nonfinite) = (0, Vec::new(), 0);
    for ((a, b), p) in g_acc.iter().zip(&g_full).zip(&inv) {
        match (a, b) {
            (Some(a), Some(b)) => {
                let err: f64 = a.iter().zip(b).map(|(x, y)| ((*x - *y) as f64).powi(2)).sum::<f64>().sqrt();
                worst = worst.max(err / l2(b).max(1e-3 * gnorm));
                if b.iter().any(|x| !x.is_finite()) {
                    nonfinite += 1;
                }
                if l2(b) == 0.0 {
                    zero.push(format!("{}.{}", p.path, p.leaf));
                }
            }
            _ => missing += 1,
        }
    }
    out.push(Check::new(
        &tag("accumulation_8x2_equals_batch16"),
        missing == 0 && worst <= TOL_ACCUM_REL_L2 && (loss_acc - full_val).abs() <= TOL_LOSS_ABS && (anorm - gnorm).abs() <= TOL_ACCUM_GLOBAL_REL * gnorm,
        json!({"worst_err_over_scale": worst, "loss_accum": loss_acc, "loss_full": full_val, "global_norm_accum": anorm, "global_norm_full": gnorm}),
    ));
    out.push(Check::new(&tag("gradient_inventory"), missing == 0 && zero.is_empty() && nonfinite == 0, json!({"params": inv.len(), "missing": missing, "exactly_zero": zero, "nonfinite": nonfinite})));

    // --- clipping: big -> norm 1 once; small -> untouched; one clip op per update ---
    let mut acc = GradientsAccumulator::<M>::new();
    for m in 0..D1_ACCUM {
        let (f, l) = micro(ctx, &lab, m);
        let (g, _) = tr.micro_grads(&f, &l, 400.0 / D1_ACCUM as f32);
        acc.accumulate(&tr.model, g);
    }
    let (gd, gn, norm, _factor, clipped) = tr.clip_and_split(acc.grads());
    let (nd, _) = crate::d1::global_norm::<GA, M>(&tr.model, &gd);
    let (nn, _) = crate::d1::global_norm::<GA, M>(&tr.model, &gn);
    let post = (nd * nd + nn * nn).sqrt();
    let mut acc = GradientsAccumulator::<M>::new();
    for m in 0..D1_ACCUM {
        let (f, l) = micro(ctx, &lab, m);
        let (g, _) = tr.micro_grads(&f, &l, 1e-3 / D1_ACCUM as f32);
        acc.accumulate(&tr.model, g);
    }
    let (_, _, norm2, _, clipped2) = tr.clip_and_split(acc.grads());
    let mut t3 = DTrainer::<GA, M>::new(build(values), &device());
    for _ in 0..3 {
        t3.step(&micros(ctx, &lab));
    }
    out.push(Check::new(
        &tag("global_clip_once_per_update"),
        clipped && norm > 1.0 && (post - 1.0).abs() <= TOL_CLIP_NORM_ABS && !clipped2 && norm2 < 1.0 && t3.clip_calls == 3,
        json!({"big_pre": norm, "big_post": post, "small_pre": norm2, "small_clipped": clipped2, "clip_ops_in_3_updates": t3.clip_calls}),
    ));

    // --- parameter groups: zero-gradient step at lr=1 shrinks decay tensors by (1-1e-4), others unchanged ---
    let mut t = DTrainer::<GA, M>::new(build(values), &device());
    let before = dump_values::<GA, _>(&t.model);
    let mut z = Zero { decay: GradientsParams::new(), nodecay: GradientsParams::new() };
    t.model.visit(&mut z);
    let m = t.opt_decay.step(1.0, t.model.clone(), z.decay);
    let m = t.opt_nodecay.step(1.0, m, z.nodecay);
    let after = dump_values::<GA, _>(&m);
    let (mut dworst, mut moved) = (0f64, 0);
    for (p, (b, a)) in inv.iter().zip(before.iter().zip(&after)) {
        if p.decay {
            for (x, y) in b.1.iter().zip(&a.1) {
                let want = *x as f64 * (1.0 - crate::train::WEIGHT_DECAY);
                dworst = dworst.max((*y as f64 - want).abs() / want.abs().max(1e-6));
            }
        } else if b.1 != a.1 {
            moved += 1;
        }
    }
    let leaf_ok = inv.iter().all(|p| p.decay == !matches!(p.leaf.as_str(), "bias" | "gamma" | "beta"));
    out.push(Check::new(&tag("weight_decay_groups"), dworst <= TOL_DECAY_REL && moved == 0 && leaf_ok, json!({"decay_tensors": inv.iter().filter(|p| p.decay).count(), "nodecay_tensors": inv.iter().filter(|p| !p.decay).count(), "worst_rel_err": dworst, "nodecay_changed": moved})));

    // --- checkpoint: weights + both optimizers restored; continued training agrees; fresh-optimizer control differs ---
    let dir = std::env::temp_dir().join(format!("v69-d1-qual-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let mut a = DTrainer::<GA, M>::new(build(values), &device());
    for _ in 0..3 {
        a.step(&micros(ctx, &lab));
    }
    save_ckpt(&dir, &a).expect("save");
    let mut b = load_ckpt::<GA, M>(&dir, build(values), &device()).expect("load");
    let restored = dump_values::<GA, _>(&b.model).iter().zip(&dump_values::<GA, _>(&a.model)).map(|(x, y)| max_abs_diff(&x.1, &y.1)).fold(0.0, f64::max);
    let mut c = DTrainer::<GA, M>::new(b.model.clone(), &device());
    c.update = b.update;
    for _ in 0..2 {
        a.step(&micros(ctx, &lab));
        b.step(&micros(ctx, &lab));
        c.step(&micros(ctx, &lab));
    }
    let (pa, pb, pc) = (dump_values::<GA, _>(&a.model), dump_values::<GA, _>(&b.model), dump_values::<GA, _>(&c.model));
    let cont = pa.iter().zip(&pb).map(|(x, y)| max_abs_diff(&x.1, &y.1)).fold(0.0, f64::max);
    let ctrl = pa.iter().zip(&pc).map(|(x, y)| max_abs_diff(&x.1, &y.1)).fold(0.0, f64::max);
    let _ = std::fs::remove_dir_all(&dir);
    out.push(Check::new(&tag("checkpoint_restores_weights_optimizers_schedule"), restored == 0.0 && cont <= TOL_CKPT_PARAM_ABS && ctrl > 10.0 * TOL_CKPT_PARAM_ABS.max(cont) && a.update == 5 && b.update == 5, json!({"restored_weights_max_diff": restored, "continued_max_diff": cont, "fresh_optimizer_control_diff": ctrl})));

    // --- state independence: solo and reordered vs batch of 16, repeat run ---
    let model = build(values);
    let base = logits_host(&model, &all);
    let mut worst = 0f64;
    for i in 0..16 {
        let solo = logits_host(&model, &[all[i]]);
        worst = worst.max((solo[0] - base[i]).abs() as f64);
    }
    let rev: Vec<&Features> = all.iter().rev().cloned().collect();
    let r = logits_host(&model, &rev);
    for k in 0..16 {
        worst = worst.max((r[k] - base[15 - k]).abs() as f64);
    }
    let again = logits_host(&model, &all);
    let rep = max_abs_diff(&base, &again);
    out.push(Check::new(&tag("state_independence"), worst <= TOL_ISOLATION_LOGIT && rep <= TOL_ISOLATION_LOGIT, json!({"max_diff_solo_or_reordered": worst, "max_diff_repeat": rep})));

    // --- input/label separation ---
    let flipped: Vec<ModelRow> = ctx.rows.iter().map(|r| ModelRow { label: !r.label, id: format!("zz-{}", r.id), ..r.clone() }).collect();
    let refeat: Vec<Features> = flipped.iter().map(|r| featurize(&r.fen, r.budget).unwrap().0).collect();
    let same = refeat == ctx.feats;
    let refs: Vec<&Features> = refeat.iter().take(16).collect();
    let pd = max_abs_diff(&logits_host(&model, &all), &logits_host(&model, &refs));
    out.push(Check::new(&tag("labels_ids_never_inputs"), same && pd == 0.0, json!({"features_identical": same, "max_prediction_diff": pd})));
    out
}

fn mlp_vals(ctx: &D1QualCtx) -> Vals {
    let inv = inventory::<G, _>(&Mlp::<G>::new(&device()));
    init::generate(&ctx.seed, "d1_qualification_init", &inv)
}

fn a_vals(ctx: &D1QualCtx) -> Vals {
    let inv = inventory::<G, _>(&Model::<G>::new(&device()));
    init::generate(&ctx.seed, "d1_qualification_init", &inv)
}

fn build_m(v: &Vals) -> Mlp<GA> {
    load_values::<GA, _>(Mlp::<GA>::new(&device()), v, &device())
}

fn build_a(v: &Vals) -> Model<GA> {
    load_values::<GA, _>(Model::<GA>::new(&device()), v, &device())
}

fn latency<M>(ctx: &D1QualCtx, build: &dyn Fn(&Vals) -> M, values: &Vals, name: &str) -> Check
where
    M: DiagModel<GA> + AutodiffModule<GA>,
    M::InnerModule: DiagModel<G>,
{
    let lab = labels(ctx);
    let d = device();
    let before = nvidia_smi_mib();
    let mut t = DTrainer::<GA, M>::new(build(values), &d);
    let mut ms = Vec::new();
    for i in 0..12 {
        let t0 = std::time::Instant::now();
        t.step(&micros(ctx, &lab));
        let _ = <G as Backend>::sync(&d);
        if i >= 2 {
            ms.push(t0.elapsed().as_secs_f64() * 1e3);
        }
    }
    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let after = nvidia_smi_mib();
    Check::new(&format!("{name}:latency_memory"), after.unwrap_or(0) <= DEVICE_MEM_CEILING_MIB, json!({"update_median_ms": ms[ms.len() / 2], "device_mib_before": before, "device_mib_after": after, "ceiling": DEVICE_MEM_CEILING_MIB}))
}

pub fn qualify_d1(ctx: &D1QualCtx) -> Value {
    let t0 = std::time::Instant::now();
    let mut checks: Vec<Check> = Vec::new();
    let push = |c: Vec<Check>, checks: &mut Vec<Check>| {
        for x in c {
            eprintln!("[d1 qual] {} -> {}", x.name, if x.pass { "PASS" } else { "FAIL" });
            if !x.pass {
                eprintln!("    detail: {}", x.detail);
            }
            checks.push(x);
        }
    };
    push(vec![check_bce_reference()], &mut checks);

    // ---- D1-M specific: forward vs f64 reference; gradient vs exact f64 finite differences ----
    let vm = mlp_vals(ctx);
    let inv_m = inventory::<G, _>(&Mlp::<G>::new(&device()));
    let rw = RefWeights::new(&inv_m, &vm);
    let model = build_m(&vm);
    let refs: Vec<&Features> = ctx.feats.iter().collect();
    let gpu = logits_host(&model, &refs);
    let ferr = ctx.feats.iter().zip(&gpu).map(|(f, g)| (*g as f64 - mlp_ref_forward(&rw, f)).abs()).fold(0.0, f64::max);
    push(vec![Check::new("D1-M:forward_matches_f64_reference", ferr <= TOL_FORWARD_ABS, json!({"max_abs_logit_error": ferr, "tolerance": TOL_FORWARD_ABS, "inputs_width": crate::d1::MLP_IN, "one_hot_check_rowsum": mlp_input_vec(&ctx.feats[0]).iter().take(832).sum::<f32>()}))], &mut checks);
    {
        let lab: Vec<bool> = ctx.rows.iter().take(8).map(|r| r.label).collect();
        let f8: Vec<&Features> = ctx.feats.iter().take(8).collect();
        let y = Tensor::<GA, 1>::from_floats(lab.iter().map(|&l| l as u8 as f32).collect::<Vec<_>>().as_slice(), &device());
        let loss = crate::model::bce_mean(model.logits(&f8, &device()), y);
        let g = grads_host(&model, &GradientsParams::from_grads(loss.backward(), &model));
        let mut res = Vec::new();
        let mut ok = true;
        for (path, leaf) in [("fc1", "weight"), ("fc2", "weight"), ("fc3", "weight"), ("fc1", "bias"), ("fc2", "bias")] {
            let idx = inv_m.iter().position(|p| p.path == path && p.leaf == leaf).unwrap();
            let gv = g[idx].as_ref().unwrap();
            let gn = l2(gv);
            let dir: Vec<f64> = gv.iter().map(|x| *x as f64 / gn).collect();
            let key = format!("{path}.{leaf}");
            let eps = 1e-3;
            let num = (mlp_ref_loss(&rw.perturbed(&key, &dir, eps), &f8, &lab) - mlp_ref_loss(&rw.perturbed(&key, &dir, -eps), &f8, &lab)) / (2.0 * eps);
            let pass = (num - gn).abs() <= TOL_NUMERIC_REL * gn + TOL_NUMERIC_ABS;
            ok &= pass;
            res.push(json!({"param": key, "gpu_grad_norm": gn, "f64_fd": num, "rel_err": (num - gn).abs() / gn.max(1e-30), "pass": pass}));
        }
        push(vec![Check::new("D1-M:gradient_matches_f64_finite_differences", ok, json!({"targets": res}))], &mut checks);
    }
    push(machinery("D1-M", ctx, &build_m, &vm), &mut checks);
    push(vec![latency(ctx, &build_m, &vm, "D1-M")], &mut checks);

    // ---- D1-A: forward at the canonical init equals the E1 one-pass forward; machinery via D1 trainer ----
    let inv_a = inventory::<G, _>(&Model::<G>::new(&device()));
    let e1 = build_a(&ctx.e1_init_values);
    let via_d1 = logits_host(&e1, &refs);
    let direct: Vec<f32> = host1(e1.forward(&Batch::<GA>::from_features(&refs, &device()), Arm::A).pop().unwrap());
    let same_path = max_abs_diff(&via_d1, &direct);
    let rwa = RefWeights::new(&inv_a, &ctx.e1_init_values);
    let ref_err = ctx.feats.iter().zip(&via_d1).map(|(f, g)| (*g as f64 - ref_forward(&rwa, f, Arm::A)[0]).abs()).fold(0.0, f64::max);
    push(vec![Check::new("D1-A:forward_at_canonical_init_matches_E1_one_pass", same_path == 0.0 && ref_err <= TOL_FORWARD_ABS, json!({"max_diff_vs_E1_code_path": same_path, "max_abs_error_vs_f64_reference": ref_err, "tolerance": TOL_FORWARD_ABS, "init": "E1 canonical untrained init (forward only, no training)"}))], &mut checks);
    let va = a_vals(ctx);
    push(machinery("D1-A", ctx, &build_a, &va), &mut checks);
    push(vec![latency(ctx, &build_a, &va, "D1-A")], &mut checks);

    let pass = checks.iter().all(|c| c.pass);
    json!({"qualified": pass, "wall_secs": t0.elapsed().as_secs_f64(), "weights": "disposable (d1_qualification_init); never used to initialize a fit", "precision": "f32 storage/accumulation; matmul inputs possibly TF32 (adopted as measured)", "checks": checks.iter().map(|c| c.to_json()).collect::<Vec<_>>()})
}

pub fn ckpt_dir_unused(_: &Path) {}
