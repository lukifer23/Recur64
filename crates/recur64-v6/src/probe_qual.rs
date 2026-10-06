//! Source-bound real-device qualification of the objective probe. A failed gate
//! returns Err (receipt records the failure); fitting is never authorized after one.
use crate::{
    intervene::{Condition, Maps},
    loss::eligible_bce,
    model::{Arm, Reader},
    objective::{self, Objective, Views},
    p0::{Plan, episode},
    qualify::{self, memory, optimizer_digest, parameter_digest},
};
use anyhow::{Result, ensure};
use burn::module::{Module, ModuleVisitor, Param};
use burn::optim::GradientsParams;
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
use burn::tensor::{Bool, backend::AutodiffBackend};
use recur64_v5::data::V5Data;
use serde_json::{Value, json};
use std::{path::Path, time::Instant};

fn host<const D: usize, T: Backend>(t: Tensor<T, D>) -> Result<Vec<f32>> {
    t.into_data()
        .to_vec::<f32>()
        .map_err(|e| anyhow::anyhow!("{e:?}"))
}

/// Closed-form f64 reference of the auxiliary for one row.
fn reference_aux(d: &[f64], eligible: &[bool], correct: &[bool]) -> f64 {
    let sp = |x: f64| x.max(0.) + (-x.abs()).exp().ln_1p();
    let mut total = 0.;
    let mut classes = 0;
    for target in [true, false] {
        let v: Vec<f64> = (0..d.len())
            .filter(|&k| eligible[k] && correct[k] == target)
            .map(|k| if target { sp(-d[k]) } else { sp(d[k]) })
            .collect();
        if !v.is_empty() {
            total += v.iter().sum::<f64>() / v.len() as f64;
            classes += 1;
        }
    }
    if classes == 0 {
        0.
    } else {
        total / classes as f64
    }
}

/// Numeric and analytic BCE value/gradient references, both signed paths of d=a-s.
pub fn loss_references<B: AutodiffBackend>(device: &B::Device) -> Result<Value> {
    let a_vals = [0.9f32, -1.5, 2.25, 0.0, 3.0, -0.5];
    let s_vals = [0.4f32, -0.25, 1.0, 0.0, 0.5, -1.25];
    let eligible = [true, true, true, true, false, true];
    let correct = [true, false, true, false, true, false];
    let t2 = |v: &[f32]| {
        Tensor::<B, 2>::from_data(burn::tensor::TensorData::new(v.to_vec(), [1, 6]), device)
    };
    let tb = |v: &[bool]| {
        Tensor::<B, 2, Bool>::from_data(burn::tensor::TensorData::new(v.to_vec(), [1, 6]), device)
    };
    let a = t2(&a_vals).require_grad();
    let s = t2(&s_vals).require_grad();
    let loss = eligible_bce(a.clone() - s.clone(), tb(&eligible), tb(&correct));
    let value = host(loss.clone())?[0] as f64;
    let grads = loss.backward();
    let ga = host(a.grad(&grads).unwrap())?;
    let gs = host(s.grad(&grads).unwrap())?;
    let d: Vec<f64> = a_vals
        .iter()
        .zip(&s_vals)
        .map(|(x, y)| f64::from(*x) - f64::from(*y))
        .collect();
    let reference = reference_aux(&d, &eligible, &correct);
    ensure!(
        (value - reference).abs() < 1e-6,
        "BCE value vs f64 reference"
    );
    // Analytic gradient: P={0,2}, N={1,3,5}, two available classes, one batch row.
    let sigma = |x: f64| 1. / (1. + (-x).exp());
    let mut max_analytic = 0f64;
    let mut max_numeric = 0f64;
    for k in 0..6 {
        let analytic = if !eligible[k] {
            0.
        } else if correct[k] {
            -sigma(-d[k]) / 2. / 2.
        } else {
            sigma(d[k]) / 3. / 2.
        };
        max_analytic = max_analytic.max((f64::from(ga[k]) - analytic).abs());
        // Central finite difference of the f64 reference with respect to a_k.
        let h = 1e-6;
        let (mut hi, mut lo) = (d.clone(), d.clone());
        hi[k] += h;
        lo[k] -= h;
        let numeric = (reference_aux(&hi, &eligible, &correct)
            - reference_aux(&lo, &eligible, &correct))
            / (2. * h);
        max_numeric = max_numeric.max((f64::from(ga[k]) - numeric).abs());
        ensure!(
            ga[k].to_bits() == (-gs[k]).to_bits() || (ga[k] == 0. && gs[k] == 0.),
            "signed gradient paths are not exact negatives at {k}"
        );
    }
    ensure!(
        max_analytic < 1e-6 && max_numeric < 1e-5,
        "BCE gradient references"
    );
    ensure!(
        ga[4] == 0. && gs[4] == 0.,
        "ineligible candidate received gradient"
    );
    // Empty support contributes exactly zero value and gradient.
    let a2 = t2(&a_vals).require_grad();
    let empty = eligible_bce(a2.clone(), tb(&[false; 6]), tb(&correct));
    let g = host(a2.grad(&empty.clone().backward()).unwrap())?;
    ensure!(
        host(empty)?[0] == 0. && g.iter().all(|v| *v == 0.),
        "empty support must be exactly zero"
    );
    // d == 0 is exactly ln2 per available class (BCE at zero).
    let z = t2(&[0.; 6]);
    let at_zero = host(eligible_bce(z.clone(), tb(&eligible), tb(&correct)))?[0] as f64;
    ensure!((at_zero - 2f64.ln()).abs() < 1e-6, "BCE at zero");
    // Single available class is not averaged with an invented empty class.
    let only_pos = host(eligible_bce(
        z,
        tb(&[true, false, false, false, false, false]),
        tb(&[true; 6]),
    ))?[0] as f64;
    ensure!(
        (only_pos - 2f64.ln()).abs() < 1e-6,
        "single available class averaging"
    );
    Ok(
        json!({"bce_value_vs_f64": (value - reference).abs(), "gradient_max_abs_error_analytic": max_analytic,
              "gradient_max_abs_error_numeric": max_numeric, "positive_and_negative_paths_exact_negatives": true,
              "empty_support_zero": true, "bce_at_zero_ln2": true, "single_class_not_averaged_with_empty": true}),
    )
}

struct BaselineAbsent<'a, B: AutodiffBackend> {
    raw: &'a B::Gradients,
    absent: bool,
    marker: std::marker::PhantomData<B>,
}
impl<B: AutodiffBackend> ModuleVisitor<B> for BaselineAbsent<'_, B> {
    fn visit_float<const N: usize>(&mut self, p: &Param<Tensor<B, N>>) {
        self.absent &= p.val().grad(self.raw).is_none();
    }
}

fn restore<B: AutodiffBackend>(
    dir: &Path,
    device: &B::Device,
) -> Result<(Reader<B>, qualify::Opt<B>)> {
    qualify::restore(Arm::OnePass, dir, device)
}

fn same(a: &Value, b: &Value, keys: &[&str]) -> bool {
    keys.iter().all(|k| a[*k] == b[*k])
}

/// Model-level checks for one objective on a real batch.
#[allow(clippy::too_many_arguments)]
fn objective_checks<B: AutodiffBackend>(
    base: &crate::baseline::FrozenBase<B>,
    objective: Objective,
    views: &[Views<B>; 2],
    snapshot: &Path,
    device: &B::Device,
    backend: &str,
    dir: &Path,
    peak: &mut u64,
) -> Result<Value> {
    let name = objective.name();
    let (initial_model, initial_opt) = {
        let (m, o) = restore::<B>(snapshot, device)?;
        (parameter_digest(&m)?, optimizer_digest(&o)?)
    };
    let (m0, _) = restore::<B>(snapshot, device)?;
    let v = &views[0];
    // Exact frozen baseline: all-null (factual == null) logits equal B0 bitwise.
    let mut null_exact = true;
    for view in [Some(&v.real), v.successor.as_ref()].into_iter().flatten() {
        let out = m0.forward(view, 1, true, false, &mut |_| {});
        null_exact &= host(out.raw_delta)?.iter().all(|x| *x == 0.)
            && host(out.logits)?
                .iter()
                .zip(host(view.z0.clone())?)
                .all(|(a, b)| a.to_bits() == b.to_bits());
    }
    ensure!(null_exact, "all-null is not exact B0 for {name}");
    // Null cancellation / differential structure with the real (initial) reader.
    let g = m0.forward(&v.real, 1, false, false, &mut |_| {});
    let g2 = m0.forward(&v.real, 1, false, false, &mut |_| {});
    let identity_d = host(g.iteration_delta[0].clone() - g2.iteration_delta[0].clone())?;
    ensure!(
        identity_d.iter().all(|x| *x == 0.),
        "identity intervention: d must be exactly zero"
    );
    let mut d_vs_direct = 0f32;
    let mut d_norm = 0f32;
    if let Some(sv) = &v.successor {
        let s = m0.forward(sv, 1, false, false, &mut |_| {});
        let d = host(g.iteration_delta[0].clone() - s.iteration_delta[0].clone())?;
        let direct = host(
            (g.factual[0].clone() - s.factual[0].clone())
                .mask_fill(v.real.legal.clone().bool_not(), 0.),
        )?;
        for (x, y) in d.iter().zip(&direct) {
            d_vs_direct = d_vs_direct.max((x - y).abs());
            d_norm = d_norm.max(x.abs());
        }
        ensure!(
            d_vs_direct <= 1e-4,
            "d != F_G - F_S beyond fp32 tolerance: {d_vs_direct}"
        );
    }
    // Gradient of A(d) vanishes under the identity intervention (both paths cancel).
    // Fresh forwards per backward: a graph is never reused across two backward passes.
    let identity_grads = {
        let ga = m0.forward(&v.real, 1, false, false, &mut |_| {});
        let gb = m0.forward(&v.real, 1, false, false, &mut |_| {});
        let aux = objective::differential_aux(&ga, &gb, v.real.eligible.clone(), v.correct.clone());
        GradientsParams::from_grads(aux.backward(), &m0)
    };
    let reference_grads = {
        let out = m0.forward(&v.real, 1, false, false, &mut |_| {});
        let (p, a) = crate::loss::components(&out, &v.real, v.correct.clone());
        GradientsParams::from_grads((p + a.mul_scalar(0.5)).backward(), &m0)
    };
    let identity_l2 = qualify::gradients(&m0, &identity_grads)?
        .iter()
        .map(|r| r.l2.unwrap_or(0.))
        .fold(0f64, f64::max);
    let reference_l2 = qualify::gradients(&m0, &reference_grads)?
        .iter()
        .map(|r| r.l2.unwrap_or(0.))
        .fold(0f64, f64::max);
    ensure!(
        identity_l2 <= 1e-5 * reference_l2.max(1e-12),
        "null/identity gradient cancellation failed: {identity_l2} vs {reference_l2}"
    );
    // Absent baseline gradients and full reader gradient inventory from the real objective.
    let (policy, auxiliary, _) = objective::microbatch(&m0, objective, v, &mut |_| {});
    let raw = (policy + auxiliary.mul_scalar(0.5)).backward();
    let mut absent = BaselineAbsent::<B> {
        raw: &raw,
        absent: true,
        marker: std::marker::PhantomData,
    };
    base.visit(&mut absent);
    ensure!(absent.absent, "baseline gradient present");
    let inventory = qualify::gradients(&m0, &GradientsParams::from_grads(raw, &m0))?;
    ensure!(
        inventory
            .iter()
            .all(|r| r.gradient_present == Some(true) && r.finite)
            && !inventory.iter().any(|r| r.name == "correction_out.bias"),
        "reader gradient inventory"
    );

    // Normal/profile parity (and exact original-objective behaviour for the control).
    let mut parity = Vec::new();
    for (k, view) in views.iter().enumerate() {
        let (a, mut ao) = restore::<B>(snapshot, device)?;
        let (a, ar) = objective::step(a, &mut ao, objective, view, 1e-3, false)?;
        let (b, mut bo) = restore::<B>(snapshot, device)?;
        let (b, br) = objective::step(b, &mut bo, objective, view, 1e-3, true)?;
        ensure!(
            same(
                &ar,
                &br,
                &[
                    "logits",
                    "raw_delta",
                    "loss",
                    "policy_loss",
                    "auxiliary_loss",
                    "gradient_digest"
                ]
            ) && parameter_digest(&a)? == parameter_digest(&b)?
                && optimizer_digest(&ao)? == optimizer_digest(&bo)?,
            "{name}: exact normal/profile parity failed"
        );
        let mut original_objective = Value::Null;
        if objective == Objective::Control {
            let (c, mut co) = restore::<B>(snapshot, device)?;
            let (c, cr) =
                qualify::step(c, &mut co, &view.real, view.correct.clone(), 1, 1e-3, false)?;
            ensure!(
                same(
                    &ar,
                    &cr,
                    &[
                        "logits",
                        "raw_delta",
                        "loss",
                        "policy_loss",
                        "auxiliary_loss",
                        "gradient_digest"
                    ]
                ) && parameter_digest(&a)? == parameter_digest(&c)?
                    && optimizer_digest(&ao)? == optimizer_digest(&co)?,
                "control is not the exact original objective"
            );
            original_objective = json!({"exact_original_objective_behavior": true});
        }
        std::fs::write(
            dir.join(format!("{name}-batch{k}-parity.json")),
            serde_json::to_vec_pretty(&json!({"normal": ar, "profile": br}))?,
        )?;
        parity.push(json!({"batch": k, "normal_profile_exact": true, "original_objective": original_objective,
                           "numeric": br, "post_adamw_parameter_digest": parameter_digest(&a)?, "moment_digest": optimizer_digest(&ao)?}));
    }
    // Independent replicas and clone purity (empty moments).
    {
        let (r1, mut o1) = restore::<B>(snapshot, device)?;
        let (r2, o2) = restore::<B>(snapshot, device)?;
        let (_, _) = objective::step(r1, &mut o1, objective, v, 1e-3, false)?;
        ensure!(
            parameter_digest(&r2)? == initial_model && optimizer_digest(&o2)? == initial_opt,
            "independent replicas share state"
        );
        let (orig, orig_opt) = restore::<B>(snapshot, device)?;
        let mut cloned_opt = orig_opt.clone();
        let _ = objective::step(orig.clone(), &mut cloned_opt, objective, v, 1e-3, false)?;
        ensure!(
            parameter_digest(&orig)? == initial_model
                && optimizer_digest(&orig_opt)? == initial_opt,
            "clone contaminated original"
        );
    }
    // Fifty resident updates; populated-moment purity; restore and exact continuation.
    let (mut m, mut opt) = restore::<B>(snapshot, device)?;
    let mut seconds = Vec::new();
    for _ in 0..50 {
        let start = Instant::now();
        let (next, _) = objective::step(m, &mut opt, objective, &views[1], 1e-3, false)?;
        m = next;
        B::sync(device).map_err(|e| anyhow::anyhow!("{e:?}"))?;
        seconds.push(start.elapsed().as_secs_f64());
        if backend == "cuda" {
            *peak = (*peak).max(memory()?);
            ensure!(
                *peak as f64 <= 3.2 * 1024.,
                "device-wide memory cap exceeded"
            );
        }
    }
    let (warm_model, warm_opt) = (parameter_digest(&m)?, optimizer_digest(&opt)?);
    let mut disposable = opt.clone();
    let _ = objective::step(m.clone(), &mut disposable, objective, &views[1], 1e-3, true)?;
    ensure!(
        parameter_digest(&m)? == warm_model && optimizer_digest(&opt)? == warm_opt,
        "populated-moment clone contamination"
    );
    let resident = dir.join(format!("{name}-resident"));
    qualify::save(&m, &opt, &resident)?;
    let (loaded, mut lo) = restore::<B>(&resident, device)?;
    ensure!(
        parameter_digest(&m)? == parameter_digest(&loaded)?
            && optimizer_digest(&opt)? == optimizer_digest(&lo)?,
        "model/moment restore failed"
    );
    let (next, _) = objective::step(m, &mut opt, objective, &views[1], 1e-3, false)?;
    let (resumed, _) = objective::step(loaded, &mut lo, objective, &views[1], 1e-3, false)?;
    ensure!(
        parameter_digest(&next)? == parameter_digest(&resumed)?
            && optimizer_digest(&opt)? == optimizer_digest(&lo)?,
        "exact continuation failed"
    );
    let mean = seconds.iter().sum::<f64>() / seconds.len() as f64;
    Ok(json!({
        "objective": name,
        "all_null_exact_b0": true,
        "identity_intervention_d_exactly_zero": true,
        "d_vs_direct_factual_difference_max_abs": d_vs_direct,
        "d_max_abs": d_norm,
        "identity_gradient_max_l2": identity_l2,
        "reference_objective_gradient_max_l2": reference_l2,
        "baseline_gradients_absent": true,
        "reader_gradient_inventory": inventory,
        "intentionally_absent": ["correction_out.bias"],
        "parity": parity,
        "independent_replicas_and_clone_purity": true,
        "populated_moment_clone_purity": true,
        "resident_updates": 50,
        "resident_seconds_mean": mean,
        "resident_seconds_max": seconds.iter().cloned().fold(0., f64::max),
        "resident_seconds": seconds,
        "projected_fit_seconds_per_arm_at_12_microbatches": mean * 12. * 200.,
        "checkpoint_restore_exact": true,
        "moment_restore_exact": true,
        "continuation_exact": true,
    }))
}

/// Inspection-only regression: the new evaluator on the frozen P0 one-pass update-200
/// weights must reproduce the independent frozen diagnostic rows exactly.
fn p0_regression<B: AutodiffBackend>(
    base: &crate::baseline::FrozenBase<B>,
    data: &V5Data,
    p0: &Plan,
    maps: &Maps,
    device: &B::Device,
) -> Result<Value> {
    let at = Path::new("runs/v6-p0/seed-6300/one-pass/update-200");
    let receipt: Value =
        serde_json::from_slice(&std::fs::read("docs/evidence/v6-p0/one-pass-result.json")?)?;
    ensure!(
        recur64_v5::stage::hash_file(&at.join("model.mpk"))?
            == receipt["checkpoint_update200_sha256"]
                .as_str()
                .unwrap_or(""),
        "frozen P0 one-pass update-200 hash mismatch"
    );
    let rec = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let m = Reader::<B>::new(Arm::OnePass, device).load_record(rec.load(at.join("model"), device)?);
    let matrix = crate::probe_eval::evaluate(
        &m,
        base,
        data,
        p0,
        maps,
        &crate::digest(maps)?,
        "inspection",
        200,
        device,
    )?;
    let diag: Value = serde_json::from_slice(&std::fs::read(
        "runs/v6-p0/failure-diagnostics-d34c948/one-pass-200/diagnostics.json",
    )?)?;
    let mut by = std::collections::BTreeMap::new();
    for r in diag["rows"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("diagnostic rows"))?
    {
        by.insert(
            (
                r["id"].as_str().unwrap().to_string(),
                r["policy"].as_u64().unwrap() as usize,
                r["condition"].as_str().unwrap().to_string(),
            ),
            (
                r["set_loss"].as_f64().unwrap(),
                r["selected_action"].as_u64().unwrap(),
                r["correct"].as_bool().unwrap(),
            ),
        );
    }
    let pairs = [
        ("real", "real"),
        ("shuffle_complete_e002", "shuffle_original"),
        ("all_null", "all_null"),
        ("board_flags_zero", "board_flags_zero"),
        ("successor_frames_zero", "successor_frames_zero"),
    ];
    let mut compared = 0;
    for r in &matrix.rows {
        if let Some((_, old)) = pairs.iter().find(|(n, _)| *n == r.condition) {
            let key = (r.id.clone(), r.policy, old.to_string());
            let (loss, action, correct) = by
                .get(&key)
                .ok_or_else(|| anyhow::anyhow!("missing frozen diagnostic row"))?;
            ensure!(
                r.set_loss == *loss
                    && u64::from(r.selected_action) == *action
                    && r.correct == *correct,
                "new evaluator differs from frozen diagnostic at {key:?}"
            );
            compared += 1;
        }
    }
    ensure!(compared == 192 * pairs.len(), "regression coverage");
    Ok(
        json!({"inspection_only": true, "weights_used_for_initialization": false, "rows_compared": compared,
              "exact_set_loss_action_correct": true, "frozen_model_sha256": receipt["checkpoint_update200_sha256"]}),
    )
}

pub fn run<B: AutodiffBackend>(
    base: &crate::baseline::FrozenBase<B>,
    data: &V5Data,
    p0: &Plan,
    device: &B::Device,
    backend: &str,
    dir: &Path,
) -> Result<Value> {
    data.require_role(recur64_v5::native_data_v2::Role::Train)?;
    ensure!(
        !dir.exists(),
        "qualification output directory already exists"
    );
    std::fs::create_dir_all(dir)?;
    let before = base.baseline_parameter_digest()?;
    let maps = crate::intervene::all_maps(p0)?;
    let cells = objective::check_all_interventions(p0, &maps)?;
    // Donor choice must not read correctness bookkeeping.
    let mut flipped = p0.clone();
    for e in &mut flipped.entries {
        e.baseline_correct = !e.baseline_correct;
    }
    ensure!(
        crate::digest(&crate::intervene::all_maps(&flipped)?)? == crate::digest(&maps)?,
        "donor maps read correctness"
    );
    let bank: Vec<Vec<_>> = (0..p0.entries.len())
        .map(|e| {
            (0..2)
                .map(|p| {
                    crate::intervene::intervened(p0, &maps, e, p, Condition::SuccessorOnlyE002)
                })
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<_>>()?;
    let references = loss_references::<B>(device)?;
    // Canonical-style initial reader and fresh optimizer snapshot.
    let initial_dir = dir.join("initial");
    let init_receipt = objective::create_initial::<B>(&initial_dir, device)?;
    B::seed(device, 6300);
    let reader = Reader::<B>::new(Arm::OnePass, device);
    let inventory = qualify::inventory(&reader)?;
    let root_count = base.param_breakdown()[0].1;
    let total = reader.num_params() + root_count;
    ensure!(
        total == objective::PARAMETERS_TOTAL && total <= 8_000_000,
        "parameter inventory {total}"
    );
    let optimizer = recur64_model::train::adamw::<B, Reader<B>>();
    let snapshot = dir.join("snapshot");
    qualify::save(&reader, &optimizer, &snapshot)?;
    ensure!(
        init_receipt["parameter_digest"] == crate::digest(&inventory)?.as_str(),
        "seeded initialization is not reproducible"
    );
    // Two real microbatches (different policies), with real and successor-only views.
    let mk = |k: usize| -> Result<Views<B>> {
        let (i, j, p) = episode(0, k);
        objective::make_views::<B>(base, data, p0, Some(&bank), [i, j], p, device)
    };
    let views = [mk(0)?, mk(1)?];
    ensure!(
        views.iter().all(|v| v.successor.as_ref().is_some_and(|s| {
            s.eligible.clone().into_data() == v.real.eligible.clone().into_data()
                && s.legal.clone().into_data() == v.real.legal.clone().into_data()
                && s.node_structure.clone().into_data() == v.real.node_structure.clone().into_data()
                && s.terminals.clone().into_data() == v.real.terminals.clone().into_data()
                && s.pool_attacker.clone().into_data() == v.real.pool_attacker.clone().into_data()
                && s.local_allow.clone().into_data() == v.real.local_allow.clone().into_data()
                && s.flags.clone().into_data() == v.real.flags.clone().into_data()
        })),
        "successor view changed structure/masks/flags tensors"
    );
    // Targets mutated: predictions and packets unchanged.
    let (m_t, _) = restore::<B>(&snapshot, device)?;
    let p_before = host(
        m_t.forward(&views[0].real, 1, false, false, &mut |_| {})
            .logits,
    )?;
    let digest_before = p0.entries[0].packets[0].content_digest()?;
    let n = views[0].real.legal.dims()[1];
    let mutated = Tensor::<B, 2, Bool>::from_data(
        burn::tensor::TensorData::new((0..2 * n).map(|k| k % 3 == 0).collect::<Vec<_>>(), [2, n]),
        device,
    );
    let _ = crate::loss::components(
        &m_t.forward(&views[0].real, 1, false, false, &mut |_| {}),
        &views[0].real,
        mutated,
    );
    ensure!(
        p_before
            == host(
                m_t.forward(&views[0].real, 1, false, false, &mut |_| {})
                    .logits
            )?
            && digest_before == p0.entries[0].packets[0].content_digest()?,
        "target mutation changed packets/predictions"
    );
    let view_seconds = |with_successor: bool| -> Result<f64> {
        let start = Instant::now();
        for _ in 0..3 {
            let (i, j, p) = episode(0, 0);
            let bank_ref = with_successor.then_some(bank.as_slice());
            let _ = objective::make_views::<B>(base, data, p0, bank_ref, [i, j], p, device)?;
        }
        B::sync(device).map_err(|e| anyhow::anyhow!("{e:?}"))?;
        Ok(start.elapsed().as_secs_f64() / 3.)
    };
    let view_build = json!({"real_only_seconds": view_seconds(false)?, "real_plus_successor_seconds": view_seconds(true)?});
    let mut peak = if backend == "cuda" { memory()? } else { 0 };
    let mut arms = Vec::new();
    for objective in [Objective::Control, Objective::Treatment] {
        arms.push(objective_checks::<B>(
            base, objective, &views, &snapshot, device, backend, dir, &mut peak,
        )?);
    }
    let regression = if backend == "cuda" {
        p0_regression(base, data, p0, &maps, device)?
    } else {
        json!({"run": false, "reason": "frozen weights were produced on CUDA; CPU comparison would invent bitwise equality"})
    };
    ensure!(
        before == base.baseline_parameter_digest()?,
        "frozen baseline mutated"
    );
    let mean = |i: usize| arms[i]["resident_seconds_mean"].as_f64().unwrap();
    let projected = |i: usize| {
        arms[i]["projected_fit_seconds_per_arm_at_12_microbatches"]
            .as_f64()
            .unwrap()
    };
    ensure!(
        projected(1) <= 45. * 60.,
        "treatment projected fit exceeds one 45-minute invocation: STOP before fitting"
    );
    Ok(json!({
        "schema": "v6_objective_probe_qualification_v2",
        "source_sha": crate::SOURCE, "objective": objective::OBJECTIVE,
        "config_digest": crate::packet::config_digest()?,
        "backend": backend, "precision": "fp32", "microbatch": 2,
        "reader_parameters": inventory.iter().map(|x| x.elements).sum::<usize>(),
        "frozen_root_parameters": root_count, "total_parameters": total,
        "intervention_cells_verified": cells,
        "donor_map_summary": objective::maps_summary(&maps),
        "donor_maps_independent_of_correctness": true,
        "successor_view_structure_masks_flags_identical": true,
        "target_mutation_leaves_predictions_and_packets_unchanged": true,
        "seeded_initialization_reproducible": true, "initial": init_receipt,
        "loss_references": references, "view_build": view_build,
        "arms": arms,
        "p0_frozen_regression": regression,
        "resident_seconds_mean": {"control": mean(0), "treatment": mean(1)},
        "treatment_over_control_resident_cost_ratio": mean(1) / mean(0),
        "projected_fit_seconds": {"control": projected(0), "treatment": projected(1)},
        "device_used_peak_sampled_mib": if backend == "cuda" { Some(peak) } else { None },
        "baseline_parameters_exact": true, "baseline_optimizer_registered": false,
        "pass": true,
    }))
}

#[cfg(test)]
mod tests {
    #[test]
    fn bce_value_and_both_signed_gradient_paths_match_references() {
        type B = burn::backend::Autodiff<burn::backend::Flex>;
        let v = super::loss_references::<B>(&Default::default()).unwrap();
        assert_eq!(v["positive_and_negative_paths_exact_negatives"], true);
        assert_eq!(v["empty_support_zero"], true);
    }
}
