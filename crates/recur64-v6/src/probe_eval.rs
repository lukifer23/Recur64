//! Fixed endpoint evaluation (eight frozen conditions), root-level gates and the
//! comparative decision. Everything here is graph-free and TRAIN-only.
use crate::{
    intervene::{Condition, DonorRow, Maps, intervened},
    model::{Arm, Reader, inputs},
    p0::{Plan, choose},
};
use anyhow::{Result, ensure};
use burn::module::AutodiffModule;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use recur64_v5::{data::V5Data, model::RootInputs};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Candidate {
    pub action_id: u16,
    pub minimum_mate_correct: bool,
    pub eligible: bool,
    pub baseline_logit: f32,
    pub final_logit: f32,
    pub factual: f32,
    pub null: f32,
    pub raw_delta: f32,
    pub centered_delta: f32,
}
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct NodeRow {
    pub root_candidate: usize,
    pub path: Vec<u16>,
    pub depth: u8,
    pub root_to_move: bool,
    pub legal_replies: usize,
    pub observed_replies: usize,
    pub unknown_replies: usize,
    pub original_flags: Vec<f32>,
    pub effective_flags: Vec<f32>,
}
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Row {
    pub id: String,
    pub cell: String,
    pub policy: usize,
    pub condition: String,
    pub baseline_correct: bool,
    pub selected_index: usize,
    pub selected_action: u16,
    pub correct: bool,
    pub set_loss: f64,
    pub margin: Option<f64>,
    pub correction_range: f64,
    pub auxiliary_loss: f64,
    pub eligible_positive: usize,
    pub eligible_negative: usize,
    pub packet_digest: String,
    pub donor: Vec<DonorRow>,
    pub nodes: Vec<NodeRow>,
    pub candidates: Vec<Candidate>,
}
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Matrix {
    pub schema: String,
    pub source_sha: String,
    pub arm: String,
    pub update: usize,
    pub plan_digest: String,
    pub maps_digest: String,
    pub baseline_exact_all_null: bool,
    pub objective: String,
    pub probe_plan_digest: String,
    pub model_sha256: String,
    pub rows: Vec<Row>,
}

pub fn bce_aux(raw: &[f32], correct: &[bool], eligible: &[bool]) -> f64 {
    let mut total = 0.;
    let mut classes = 0;
    for target in [true, false] {
        let v: Vec<f64> = raw
            .iter()
            .zip(correct)
            .zip(eligible)
            .filter(|((_, c), e)| **e && **c == target)
            .map(|((v, _), _)| {
                let x = f64::from(*v);
                if target {
                    (-x).max(0.) + (-x.abs()).exp().ln_1p()
                } else {
                    x.max(0.) + (-x.abs()).exp().ln_1p()
                }
            })
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
fn host<const D: usize, T: Backend>(t: Tensor<T, D>) -> Result<Vec<f32>> {
    t.into_data()
        .to_vec::<f32>()
        .map_err(|e| anyhow::anyhow!("{e:?}"))
}

#[allow(clippy::too_many_arguments)]
pub fn evaluate<B: AutodiffBackend>(
    m: &Reader<B>,
    base: &crate::baseline::FrozenBase<B>,
    data: &V5Data,
    plan: &Plan,
    maps: &Maps,
    maps_digest: &str,
    arm: &str,
    update: usize,
    device: &B::Device,
) -> Result<Matrix> {
    data.require_role(recur64_v5::native_data_v2::Role::Train)?;
    ensure!(m.arm() == Arm::OnePass, "one-pass reader only");
    let before = base.baseline_parameter_digest()?;
    let valid = m.valid();
    let mut rows = Vec::new();
    let mut exact = true;
    for first in (0..plan.entries.len()).step_by(2) {
        let entry_ids = [first, first + 1];
        let indices: Vec<usize> = entry_ids.iter().map(|&e| plan.entries[e].index).collect();
        let roots = data.roots(&indices)?;
        let refs: Vec<_> = roots.iter().collect();
        let root = base
            .valid()
            .base_root(&RootInputs::<B::InnerBackend>::from_roots(&refs, device)?);
        for policy in 0..2 {
            for condition in Condition::all() {
                let packets = entry_ids
                    .iter()
                    .map(|&e| intervened(plan, maps, e, policy, condition))
                    .collect::<Result<Vec<_>>>()?;
                let i = inputs::<B::InnerBackend>(
                    &packets,
                    root.hypotheses.clone(),
                    root.z0.clone(),
                    device,
                )?;
                let out = valid.forward(&i, 1, condition == Condition::AllNull, false, &mut |_| {});
                let logits = host(out.logits)?;
                let raw = host(out.raw_delta)?;
                let centered = host(out.centered)?;
                let factual = host(out.factual.last().unwrap().clone())?;
                let null = host(out.null.last().unwrap().clone())?;
                let z0 = host(i.z0.clone())?;
                let w = i.width;
                if condition == Condition::AllNull {
                    exact &= raw.iter().all(|v| *v == 0.)
                        && logits
                            .iter()
                            .zip(&z0)
                            .all(|(a, b)| a.to_bits() == b.to_bits());
                }
                for (j, &ei) in entry_ids.iter().enumerate() {
                    let e = &plan.entries[ei];
                    let p = data.position(e.index);
                    let n = p.legal.len();
                    let at = j * w..j * w + n;
                    let correct: Vec<bool> =
                        (0..n).map(|a| p.correct.contains(&(a as u32))).collect();
                    let original = &e.packets[policy];
                    let eligible = original.eligibility();
                    ensure!(
                        n == original.legal.len()
                            && original
                                .legal
                                .iter()
                                .zip(&p.legal)
                                .all(|(a, b)| *a as u32 == *b as u32),
                        "candidate ActionId alignment"
                    );
                    let zz = &logits[at.clone()];
                    let sel = choose(zz);
                    let best_correct = zz
                        .iter()
                        .zip(&correct)
                        .filter(|(_, c)| **c)
                        .map(|(v, _)| *v as f64)
                        .max_by(f64::total_cmp)
                        .unwrap();
                    let best_other = zz
                        .iter()
                        .zip(&correct)
                        .filter(|(_, c)| !**c)
                        .map(|(v, _)| *v as f64)
                        .max_by(f64::total_cmp);
                    let rr = &raw[at.clone()];
                    let candidates = (0..n)
                        .map(|a| Candidate {
                            action_id: original.legal[a],
                            minimum_mate_correct: correct[a],
                            eligible: eligible[a],
                            baseline_logit: z0[j * w + a],
                            final_logit: zz[a],
                            factual: factual[j * w + a],
                            null: null[j * w + a],
                            raw_delta: rr[a],
                            centered_delta: centered[j * w + a],
                        })
                        .collect();
                    let nodes = original
                        .nodes
                        .iter()
                        .enumerate()
                        .map(|(k, v)| NodeRow {
                            root_candidate: v.root_candidate,
                            path: v.path.clone(),
                            depth: v.depth,
                            root_to_move: v.root_to_move,
                            legal_replies: original.legal_counts[k],
                            observed_replies: original.children(k).len(),
                            unknown_replies: original.unknown(k),
                            original_flags: v.payload.flags.to_vec(),
                            effective_flags: packets[j].nodes[k].payload.flags.to_vec(),
                        })
                        .collect();
                    let donor = condition
                        .map_seed()
                        .map(|s| maps.for_seed(s).rows[ei][policy].clone())
                        .unwrap_or_default();
                    rows.push(Row {
                        id: e.id.clone(),
                        cell: e.cell.clone(),
                        policy,
                        condition: condition.name().into(),
                        baseline_correct: e.baseline_correct,
                        selected_index: sel,
                        selected_action: original.legal[sel],
                        correct: correct[sel],
                        set_loss: recur64_v5::loss::reference_set_loss(
                            zz,
                            &vec![true; n],
                            &correct,
                        ),
                        margin: best_other.map(|v| best_correct - v),
                        correction_range: rr.iter().copied().fold(f32::NEG_INFINITY, f32::max)
                            as f64
                            - rr.iter().copied().fold(f32::INFINITY, f32::min) as f64,
                        auxiliary_loss: bce_aux(rr, &correct, &eligible),
                        eligible_positive: correct
                            .iter()
                            .zip(&eligible)
                            .filter(|(c, e)| **c && **e)
                            .count(),
                        eligible_negative: correct
                            .iter()
                            .zip(&eligible)
                            .filter(|(c, e)| !**c && **e)
                            .count(),
                        packet_digest: original.digest.clone(),
                        donor,
                        nodes,
                        candidates,
                    });
                }
            }
        }
    }
    ensure!(
        before == base.baseline_parameter_digest()?,
        "baseline mutated during evaluation"
    );
    ensure!(exact, "all-null is not exact frozen B0");
    Ok(Matrix {
        schema: "v6_objective_probe_endpoint_v2".into(),
        source_sha: crate::SOURCE.into(),
        arm: arm.into(),
        update,
        plan_digest: plan.digest.clone(),
        maps_digest: maps_digest.into(),
        baseline_exact_all_null: exact,
        objective: crate::objective::OBJECTIVE.into(),
        probe_plan_digest: String::new(),
        model_sha256: String::new(),
        rows,
    })
}

type Key<'a> = (&'a str, usize, &'a str);
struct Index<'a> {
    by: BTreeMap<Key<'a>, &'a Row>,
    ids: Vec<&'a str>,
}
impl<'a> Index<'a> {
    fn new(m: &'a Matrix) -> Result<Self> {
        let mut by = BTreeMap::new();
        let mut ids: Vec<&str> = Vec::new();
        for r in &m.rows {
            ensure!(
                by.insert((r.id.as_str(), r.policy, r.condition.as_str()), r)
                    .is_none(),
                "duplicate endpoint row"
            );
            if r.condition == "real" && r.policy == 0 {
                ids.push(&r.id);
            }
        }
        ensure!(
            ids.len() == 96 && m.rows.len() == 96 * 2 * Condition::all().len(),
            "endpoint matrix must hold 96 roots x 2 policies x 8 conditions"
        );
        for id in &ids {
            for p in 0..2 {
                for c in Condition::all() {
                    ensure!(by.contains_key(&(*id, p, c.name())), "missing endpoint row");
                }
            }
        }
        Ok(Self { by, ids })
    }
    fn row(&self, id: &str, policy: usize, c: &str) -> &'a Row {
        self.by[&(id, policy, c)]
    }
    fn baseline(&self, id: &str) -> bool {
        self.row(id, 0, "real").baseline_correct
    }
    fn corrected(&self, id: &str, c: &str) -> bool {
        !self.baseline(id) && (0..2).all(|p| self.row(id, p, c).correct)
    }
    fn harmed(&self, id: &str, c: &str) -> bool {
        self.baseline(id) && (0..2).any(|p| !self.row(id, p, c).correct)
    }
    fn mean(&self, c: &str, f: impl Fn(&Row) -> f64) -> f64 {
        let v: Vec<f64> = self
            .by
            .iter()
            .filter(|((_, _, cc), _)| *cc == c)
            .map(|(_, r)| f(r))
            .collect();
        v.iter().sum::<f64>() / v.len() as f64
    }
}

fn summary(ix: &Index, c: &str) -> Value {
    let rows: Vec<&Row> = ix
        .by
        .iter()
        .filter(|((_, _, cc), _)| *cc == c)
        .map(|(_, r)| *r)
        .collect();
    let margins: Vec<f64> = rows.iter().filter_map(|r| r.margin).collect();
    json!({
        "mean_set_loss": ix.mean(c, |r| r.set_loss),
        "correct_rows": rows.iter().filter(|r| r.correct).count(),
        "mean_margin": margins.iter().sum::<f64>() / margins.len().max(1) as f64,
        "mean_correction_range": ix.mean(c, |r| r.correction_range),
        "mean_auxiliary_bce_on_delta": ix.mean(c, |r| r.auxiliary_loss),
        "corrected_roots": ix.ids.iter().filter(|i| ix.corrected(i, c)).count(),
        "harmed_roots": ix.ids.iter().filter(|i| ix.harmed(i, c)).count(),
    })
}

fn gate(value: Value, threshold: Value, pass: bool) -> Value {
    json!({"value": value, "threshold": threshold, "pass": pass})
}

/// Absolute (single-arm) root gates and diagnostics.
pub fn analyze(initial: &Matrix, last: &Matrix) -> Result<Value> {
    ensure!(
        initial.update == 0
            && last.update == 200
            && initial.plan_digest == last.plan_digest
            && initial.maps_digest == last.maps_digest
            && initial.arm == last.arm,
        "endpoint pair identity"
    );
    let a = Index::new(initial)?;
    let b = Index::new(last)?;
    ensure!(
        a.ids == b.ids
            && b.ids.iter().all(|i| a.baseline(i) == b.baseline(i))
            && b.ids.iter().filter(|i| !b.baseline(i)).count() == 48,
        "endpoint ID/stratum mismatch"
    );
    let real_corrected: Vec<&str> = b
        .ids
        .iter()
        .copied()
        .filter(|i| b.corrected(i, "real"))
        .collect();
    let real_harmed: Vec<&str> = b
        .ids
        .iter()
        .copied()
        .filter(|i| b.harmed(i, "real"))
        .collect();
    let loss0 = a.mean("real", |r| r.set_loss);
    let loss1 = b.mean("real", |r| r.set_loss);
    let contrast = |c: &str| b.mean(c, |r| r.set_loss) - loss1;
    let reversed = |c: &str| {
        real_corrected
            .iter()
            .filter(|i| (0..2).all(|p| !b.row(i, p, c).correct))
            .count()
    };
    let shuffles = [
        "shuffle_complete_e002",
        "successor_only_e002",
        "successor_only_e402",
        "successor_only_e502",
    ];
    let mut shuffle_report = serde_json::Map::new();
    for c in shuffles {
        shuffle_report.insert(
            c.into(),
            json!({
                "shuffle_minus_real_mean_set_loss": contrast(c),
                "real_corrected_roots_wrong_under_both_policies": reversed(c),
                "corrected_roots_under_condition": b.ids.iter().filter(|i| b.corrected(i, c)).count(),
                "harmed_roots_under_condition": b.ids.iter().filter(|i| b.harmed(i, c)).count(),
                "summary": summary(&b, c),
            }),
        );
    }
    // A(d): host reference of the treatment's auxiliary on d = F_G - F_Ssucc(G).
    let mut differential = serde_json::Map::new();
    for c in [
        "successor_only_e002",
        "successor_only_e402",
        "successor_only_e502",
    ] {
        let mut values = Vec::new();
        for r in b.by.values().filter(|r| r.condition == "real") {
            let s = b.row(&r.id, r.policy, c);
            let d: Vec<f32> = r
                .candidates
                .iter()
                .zip(&s.candidates)
                .map(|(x, y)| x.factual - y.factual)
                .collect();
            let correct: Vec<bool> = r
                .candidates
                .iter()
                .map(|x| x.minimum_mate_correct)
                .collect();
            let eligible: Vec<bool> = r.candidates.iter().map(|x| x.eligible).collect();
            values.push(bce_aux(&d, &correct, &eligible));
        }
        differential.insert(
            c.into(),
            json!(values.iter().sum::<f64>() / values.len() as f64),
        );
    }
    let survival = |c: &str| {
        let hs: Vec<&&str> = real_corrected
            .iter()
            .filter(|i| b.corrected(i, c))
            .collect();
        let hm: Vec<&&str> = real_harmed.iter().filter(|i| b.harmed(i, c)).collect();
        json!({
            "helpful_corrected_roots_surviving_both_policies": hs.len(),
            "harmful_roots_surviving_either_policy": hm.len(),
            "helpful_ids": hs, "harmful_ids": hm,
        })
    };
    // Per-cell and support strata (support = policies with an observed correct branch).
    let mut cells: BTreeMap<String, Value> = BTreeMap::new();
    for cell in b
        .ids
        .iter()
        .map(|i| b.row(i, 0, "real").cell.clone())
        .collect::<BTreeSet<_>>()
    {
        let ids: Vec<&&str> = b
            .ids
            .iter()
            .filter(|i| b.row(i, 0, "real").cell == cell)
            .collect();
        let wrong: Vec<&&&str> = ids.iter().filter(|i| !b.baseline(i)).collect();
        let right: Vec<&&&str> = ids.iter().filter(|i| b.baseline(i)).collect();
        cells.insert(
            cell,
            json!({
                "baseline_wrong": wrong.len(), "corrected": wrong.iter().filter(|i| b.corrected(i, "real")).count(),
                "baseline_right": right.len(), "harmed": right.iter().filter(|i| b.harmed(i, "real")).count(),
                "mean_real_set_loss_initial": ids.iter().map(|i| (0..2).map(|p| a.row(i,p,"real").set_loss).sum::<f64>()/2.).sum::<f64>()/ids.len() as f64,
                "mean_real_set_loss_final": ids.iter().map(|i| (0..2).map(|p| b.row(i,p,"real").set_loss).sum::<f64>()/2.).sum::<f64>()/ids.len() as f64,
            }),
        );
    }
    let support = |i: &str| {
        (0..2)
            .filter(|p| b.row(i, *p, "real").eligible_positive > 0)
            .count()
    };
    let mut strata = json!({});
    for s in 0..=2usize {
        let wrong: Vec<&&str> = b
            .ids
            .iter()
            .filter(|i| !b.baseline(i) && support(i) == s)
            .collect();
        strata[format!("baseline_wrong_with_observed_correct_branch_in_{s}_policies")] = json!({
            "roots": wrong.len(),
            "corrected": wrong.iter().filter(|i| b.corrected(i, "real")).count(),
        });
    }
    let per_policy: Vec<Value> = (0..2)
        .map(|p| {
            let wr: Vec<&&str> = b.ids.iter().filter(|i| !b.baseline(i)).collect();
            let rt: Vec<&&str> = b.ids.iter().filter(|i| b.baseline(i)).collect();
            json!({"policy": p,
                "corrected_rows": wr.iter().filter(|i| b.row(i, p, "real").correct).count(),
                "harmed_rows": rt.iter().filter(|i| !b.row(i, p, "real").correct).count()})
        })
        .collect();
    let succ_independent: Vec<&str> = vec!["successor_only_e402", "successor_only_e502"];
    let mut gates = serde_json::Map::new();
    gates.insert(
        "finite_exact_B0_and_all_null".into(),
        gate(
            json!(last.baseline_exact_all_null && initial.baseline_exact_all_null),
            json!(true),
            last.baseline_exact_all_null
                && initial.baseline_exact_all_null
                && last.rows.iter().all(|r| {
                    r.set_loss.is_finite() && r.candidates.iter().all(|c| c.final_logit.is_finite())
                }),
        ),
    );
    gates.insert(
        "real_policy_loss_reduction".into(),
        gate(
            json!(1. - loss1 / loss0),
            json!(">=0.20 unless initial<0.05"),
            loss0 < 0.05 || loss1 <= loss0 * 0.8,
        ),
    );
    gates.insert(
        "corrected_wrong_roots".into(),
        gate(
            json!(real_corrected.len()),
            json!(">=12 of 48"),
            real_corrected.len() >= 12,
        ),
    );
    gates.insert(
        "harmed_right_roots".into(),
        gate(
            json!(real_harmed.len()),
            json!("<=2 of 48"),
            real_harmed.len() <= 2,
        ),
    );
    gates.insert(
        "complete_shuffle_minus_real_loss".into(),
        gate(
            json!(contrast("shuffle_complete_e002")),
            json!(">=0.05"),
            contrast("shuffle_complete_e002") >= 0.05,
        ),
    );
    gates.insert(
        "corrected_roots_wrong_under_complete_shuffle_both_policies".into(),
        gate(
            json!(reversed("shuffle_complete_e002")),
            json!(">=6"),
            reversed("shuffle_complete_e002") >= 6,
        ),
    );
    for c in &succ_independent {
        gates.insert(
            format!("{c}_shuffle_minus_real_loss"),
            gate(json!(contrast(c)), json!(">=0.05"), contrast(c) >= 0.05),
        );
        gates.insert(
            format!("{c}_corrected_roots_wrong_both_policies"),
            gate(json!(reversed(c)), json!(">=6"), reversed(c) >= 6),
        );
    }
    let absolute_pass = gates.values().all(|g| g["pass"] == true);
    let transitions: Vec<Value> = b
        .ids
        .iter()
        .map(|i| {
            let per = |c: &str| (0..2).map(|p| b.row(i, p, c).correct).collect::<Vec<_>>();
            let sel = |c: &str| (0..2).map(|p| b.row(i, p, c).selected_action).collect::<Vec<_>>();
            json!({
                "id": i, "cell": b.row(i, 0, "real").cell, "baseline_correct": b.baseline(i),
                "baseline_selected_actions": (0..2).map(|p| { let r = b.row(i, p, "real"); r.candidates[choose(&r.candidates.iter().map(|c| c.baseline_logit).collect::<Vec<_>>())].action_id }).collect::<Vec<_>>(),
                "corrected": b.corrected(i, "real"), "harmed": b.harmed(i, "real"),
                "correct_by_policy": per("real"), "selected_actions": sel("real"),
                "correct_by_policy_by_condition": Condition::all().iter().map(|c| (c.name(), per(c.name()))).collect::<BTreeMap<_, _>>(),
                "observed_correct_branch_by_policy": (0..2).map(|p| b.row(i,p,"real").eligible_positive>0).collect::<Vec<_>>(),
            })
        })
        .collect();
    Ok(json!({
        "schema": "v6_objective_probe_arm_analysis_v2",
        "arm": last.arm,
        "unit": "96 roots; corrected under BOTH policies, harmed under EITHER; policy rows are not independent roots",
        "policy_loss": {"initial": loss0, "final": loss1, "relative_reduction": 1. - loss1 / loss0},
        "auxiliary_bce_on_delta_real": {"initial": a.mean("real", |r| r.auxiliary_loss), "final": b.mean("real", |r| r.auxiliary_loss)},
        "auxiliary_bce_on_differential_d_final": differential,
        "conditions_initial": Condition::all().iter().map(|c| (c.name(), summary(&a, c.name()))).collect::<BTreeMap<_, _>>(),
        "conditions_final": Condition::all().iter().map(|c| (c.name(), summary(&b, c.name()))).collect::<BTreeMap<_, _>>(),
        "corrected_ids": real_corrected, "harmed_ids": real_harmed,
        "shuffles": shuffle_report,
        "board_erasure_survival": survival("board_flags_zero"),
        "successor_removal_survival": survival("successor_frames_zero"),
        "per_policy": per_policy, "per_cell": cells, "support_strata": strata,
        "gates": gates, "absolute_pass": absolute_pass,
        "root_transitions": transitions,
        "scope": "TRAIN memorization/learnability on a repeatedly exposed panel; no held-out claim",
    }))
}

/// Refuse any endpoint that is not exactly this launch's: expected schema and
/// scientific source, arm/update identity, frozen panel + donor-map bindings, and the
/// complete objective-probe launch plan (which binds contract, initial reader and
/// panel). The P0 plan digest alone is not accepted as launch identity.
pub fn verify_endpoint(
    m: &Matrix,
    arm: &str,
    update: usize,
    plan: &crate::objective::ProbePlan,
) -> Result<()> {
    ensure!(
        m.schema == "v6_objective_probe_endpoint_v2"
            && m.source_sha == crate::SOURCE
            && plan.source_sha == crate::SOURCE
            && m.objective == crate::objective::OBJECTIVE
            && m.arm == arm
            && m.update == update
            && m.plan_digest == plan.p0_plan_digest
            && m.maps_digest == plan.maps_digest
            && m.probe_plan_digest == plan.digest
            && !m.model_sha256.is_empty()
            && m.baseline_exact_all_null
            && m.rows.len() == 96 * 2 * Condition::all().len(),
        "endpoint provenance mismatch ({arm} update {update}): foreign source, launch, arm or panel"
    );
    Ok(())
}

/// Treatment versus fresh control, computed only after BOTH arms completed 0/200.
pub fn decide(
    plan: &crate::objective::ProbePlan,
    control: &[&Matrix; 2],
    treatment: &[&Matrix; 2],
) -> Result<Value> {
    for (arm, pair) in [("control", control), ("treatment", treatment)] {
        for (m, update) in pair.iter().zip([0, 200]) {
            verify_endpoint(m, arm, update, plan)?;
        }
    }
    ensure!(
        control[0].plan_digest == treatment[0].plan_digest
            && control[0].maps_digest == treatment[0].maps_digest
            && control.iter().all(|m| m.arm == "control")
            && treatment.iter().all(|m| m.arm == "treatment"),
        "arm/plan identity"
    );
    let ca = analyze(control[0], control[1])?;
    let ta = analyze(treatment[0], treatment[1])?;
    let c = Index::new(control[1])?;
    let t = Index::new(treatment[1])?;
    ensure!(c.ids == t.ids, "paired roots differ");
    let loss_c = c.mean("real", |r| r.set_loss);
    let loss_t = t.mean("real", |r| r.set_loss);
    let corrected = |ix: &Index| ix.ids.iter().filter(|i| ix.corrected(i, "real")).count();
    let harmed = |ix: &Index| ix.ids.iter().filter(|i| ix.harmed(i, "real")).count();
    let (cc, tc, ch, th) = (corrected(&c), corrected(&t), harmed(&c), harmed(&t));
    let mut comparative = serde_json::Map::new();
    comparative.insert(
        "treatment_real_mean_set_loss_lower_by".into(),
        gate(
            json!(loss_c - loss_t),
            json!(">=0.05"),
            loss_c - loss_t >= 0.05,
        ),
    );
    comparative.insert(
        "additional_paired_corrected_roots".into(),
        gate(json!(tc as i64 - cc as i64), json!(">=4"), tc >= cc + 4),
    );
    comparative.insert(
        "treatment_harmed_not_more_than_control_and_at_most_2".into(),
        gate(
            json!({"control": ch, "treatment": th}),
            json!("treatment<=control and <=2"),
            th <= ch && th <= 2,
        ),
    );
    let comparative_pass = comparative.values().all(|g| g["pass"] == true);
    let t_abs = ta["absolute_pass"] == true;
    let outcome = if t_abs && comparative_pass {
        "OBJECTIVE_SUPPORTED_TRAIN_ONLY"
    } else if t_abs {
        "TREATMENT_PASSES_NO_COMPARATIVE_ADVANTAGE"
    } else {
        "TREATMENT_FAILS"
    };
    Ok(json!({
        "schema": "v6_objective_probe_decision_v2",
        "objective": crate::objective::OBJECTIVE,
        "outcome": outcome,
        "control": ca, "treatment": ta,
        "comparative_gates": comparative, "comparative_pass": comparative_pass,
        "paired_root_deltas": {
            "treatment_only_corrected": only(&t, &c, Index::corrected),
            "control_only_corrected": only(&c, &t, Index::corrected),
            "treatment_only_harmed": only(&t, &c, Index::harmed),
            "control_only_harmed": only(&c, &t, Index::harmed),
        },
        "bootstrap_or_significance_claim": false,
        "compute_matched": false,
        "scope": "disposable TRAIN learnability/memorization; single trajectory pair; no held-out or semantic-reasoning claim",
    }))
}

fn only<'a>(
    x: &Index<'a>,
    y: &Index<'a>,
    f: impl Fn(&Index<'a>, &str, &str) -> bool,
) -> Vec<&'a str> {
    x.ids
        .iter()
        .copied()
        .filter(|i| f(x, i, "real") && !f(y, i, "real"))
        .collect()
}
#[cfg(test)]
mod tests {
    use super::*;
    fn row(
        id: usize,
        policy: usize,
        cond: Condition,
        baseline: bool,
        correct: bool,
        loss: f64,
    ) -> Row {
        Row {
            id: format!("r{id:03}"),
            cell: format!("c{}", id / 24),
            policy,
            condition: cond.name().into(),
            baseline_correct: baseline,
            selected_index: 0,
            selected_action: 0,
            correct,
            set_loss: loss,
            margin: Some(0.),
            correction_range: 0.,
            auxiliary_loss: 0.,
            eligible_positive: 1,
            eligible_negative: 1,
            packet_digest: String::new(),
            donor: vec![],
            nodes: vec![],
            candidates: vec![Candidate {
                action_id: 0,
                minimum_mate_correct: true,
                eligible: true,
                baseline_logit: 0.,
                final_logit: 0.,
                factual: 0.,
                null: 0.,
                raw_delta: 0.,
                centered_delta: 0.,
            }],
        }
    }
    /// `final_state(root, cond)` => (correct under both policies, loss).
    fn matrix(arm: &str, update: usize, f: impl Fn(usize, Condition) -> (bool, f64)) -> Matrix {
        let mut rows = Vec::new();
        for id in 0..96 {
            for p in 0..2 {
                for c in Condition::all() {
                    let (ok, loss) = f(id, c);
                    // root 12 is wrong under policy 0 only (for the "harm either policy" check).
                    let ok = if id == 60 && p == 0 && c == Condition::Real && update == 200 {
                        false
                    } else {
                        ok
                    };
                    rows.push(row(id, p, c, id >= 48, ok, loss));
                }
            }
        }
        Matrix {
            schema: "v6_objective_probe_endpoint_v2".into(),
            source_sha: crate::SOURCE.into(),
            arm: arm.into(),
            update,
            plan_digest: "p".into(),
            maps_digest: "m".into(),
            baseline_exact_all_null: true,
            objective: crate::objective::OBJECTIVE.into(),
            probe_plan_digest: "probe".into(),
            model_sha256: format!("model-{arm}-{update}"),
            rows,
        }
    }
    #[test]
    fn root_level_counting_and_gates() {
        let initial = matrix("treatment", 0, |id, _| (id >= 48, 1.));
        // 12 roots corrected (ids 0..12). Complete and successor-only shuffles reverse the first
        // six of them, board+flags/successor-frame removal reverse none.
        let last = matrix("treatment", 200, |id, c| {
            let real = !(12..48).contains(&id);
            match c {
                Condition::Real => (real, 0.5),
                Condition::ShuffleCompleteE002
                | Condition::SuccessorOnlyE402
                | Condition::SuccessorOnlyE502 => (if id < 6 { false } else { real }, 0.6),
                _ => (real, 0.5),
            }
        });
        let a = analyze(&initial, &last).unwrap();
        assert_eq!(a["corrected_ids"].as_array().unwrap().len(), 12);
        assert_eq!(
            a["harmed_ids"].as_array().unwrap().len(),
            1,
            "harm counts a root wrong under EITHER policy"
        );
        assert_eq!(a["gates"]["corrected_wrong_roots"]["pass"], true);
        assert_eq!(a["gates"]["harmed_right_roots"]["pass"], true);
        assert_eq!(a["gates"]["complete_shuffle_minus_real_loss"]["pass"], true);
        assert_eq!(
            a["gates"]["successor_only_e402_corrected_roots_wrong_both_policies"]["value"],
            6
        );
        assert_eq!(
            a["board_erasure_survival"]["helpful_corrected_roots_surviving_both_policies"],
            12
        );
        assert_eq!(a["absolute_pass"], true);
        // One policy row of a corrected root failing makes it uncorrected (BOTH required).
        let mut broken = last.clone();
        for r in &mut broken.rows {
            if r.id == "r000" && r.policy == 1 && r.condition == "real" {
                r.correct = false;
            }
        }
        let b = analyze(&initial, &broken).unwrap();
        assert_eq!(b["corrected_ids"].as_array().unwrap().len(), 11);
        assert!(
            analyze(
                &initial,
                &Matrix {
                    rows: last.rows[1..].to_vec(),
                    ..last.clone()
                }
            )
            .is_err()
        );
    }
    fn test_plan() -> crate::objective::ProbePlan {
        crate::objective::ProbePlan {
            schema: crate::objective::PLAN_SCHEMA.into(),
            objective: crate::objective::OBJECTIVE.into(),
            launch_plan: crate::objective::LAUNCH_PLAN.into(),
            source_sha: crate::SOURCE.into(),
            config_digest: String::new(),
            train_digest: String::new(),
            contract_sha256: String::new(),
            p0_producer: String::new(),
            p0_plan_digest: "p".into(),
            p0_plan_raw_sha256: String::new(),
            episode_digest: String::new(),
            panel: vec![],
            maps_digest: "m".into(),
            map_summary: Value::Null,
            seed: 6300,
            updates: 200,
            aux_weight: 0.5,
            conditions: vec![],
            initial_model_sha256: String::new(),
            initial_parameter_digest: String::new(),
            digest: "probe".into(),
        }
    }
    #[test]
    fn foreign_source_or_mismatched_launch_endpoints_are_refused() {
        let plan = test_plan();
        let good = matrix("treatment", 200, |id, _| (id >= 48, 1.));
        verify_endpoint(&good, "treatment", 200, &plan).unwrap();
        type Mutation = (&'static str, fn(&mut Matrix));
        let mutations: [Mutation; 10] = [
            ("foreign source", |m| m.source_sha = "f".repeat(40)),
            ("wrong schema", |m| {
                m.schema = "v6_objective_probe_endpoint_v1".into()
            }),
            ("other launch plan", |m| {
                m.probe_plan_digest = "another-launch".into()
            }),
            ("only the P0 panel digest matches", |m| {
                m.probe_plan_digest.clear()
            }),
            ("foreign panel", |m| m.plan_digest = "other-panel".into()),
            ("foreign donor maps", |m| {
                m.maps_digest = "other-maps".into()
            }),
            ("wrong objective", |m| {
                m.objective = "v6_content_differential_aux_v1".into()
            }),
            ("no checkpoint binding", |m| m.model_sha256.clear()),
            ("inexact baseline", |m| m.baseline_exact_all_null = false),
            ("missing rows", |m| {
                m.rows.pop();
            }),
        ];
        for (name, f) in mutations {
            let mut m = good.clone();
            f(&mut m);
            assert!(
                verify_endpoint(&m, "treatment", 200, &plan).is_err(),
                "{name}"
            );
        }
        assert!(
            verify_endpoint(&good, "control", 200, &plan).is_err(),
            "arm identity"
        );
        assert!(
            verify_endpoint(&good, "treatment", 0, &plan).is_err(),
            "update identity"
        );
        // A foreign endpoint inside a pair blocks the whole decision.
        let (ci, ti) = (
            matrix("control", 0, |id, _| (id >= 48, 1.)),
            matrix("treatment", 0, |id, _| (id >= 48, 1.)),
        );
        let mut foreign = matrix("control", 200, |id, _| (id >= 48, 1.));
        foreign.source_sha = "f".repeat(40);
        assert!(decide(&plan, &[&ci, &foreign], &[&ti, &good]).is_err());
    }
    #[test]
    fn decision_classes_and_comparative_gates() {
        let initial = |arm: &str| matrix(arm, 0, |id, _| (id >= 48, 1.));
        let passing = |arm: &str, loss: f64| {
            matrix(arm, 200, move |id, c| {
                let real = !(12..48).contains(&id);
                match c {
                    Condition::Real => (real, loss),
                    Condition::ShuffleCompleteE002
                    | Condition::SuccessorOnlyE402
                    | Condition::SuccessorOnlyE502
                    | Condition::SuccessorOnlyE002 => {
                        (if id < 6 { false } else { real }, loss + 0.1)
                    }
                    _ => (real, loss),
                }
            })
        };
        let weak = matrix("control", 200, |id, _| (!(4..48).contains(&id), 0.9));
        let (ci, ti) = (initial("control"), initial("treatment"));
        let t = passing("treatment", 0.5);
        let d = decide(&test_plan(), &[&ci, &weak], &[&ti, &t]).unwrap();
        assert_eq!(d["outcome"], "OBJECTIVE_SUPPORTED_TRAIN_ONLY");
        let c2 = passing("control", 0.5);
        let d = decide(&test_plan(), &[&ci, &c2], &[&ti, &t]).unwrap();
        assert_eq!(d["outcome"], "TREATMENT_PASSES_NO_COMPARATIVE_ADVANTAGE");
        let d = decide(
            &test_plan(),
            &[&ci, &weak],
            &[&ti, &matrix("treatment", 200, |id, _| (id >= 48, 1.))],
        )
        .unwrap();
        assert_eq!(d["outcome"], "TREATMENT_FAILS");
    }
    #[test]
    fn auxiliary_reference_matches_available_class_averaging() {
        let v = bce_aux(&[0., 2., -3.], &[true, true, false], &[true, false, false]);
        assert!((v - 2f64.ln()).abs() < 1e-12);
        assert_eq!(bce_aux(&[1.], &[true], &[false]), 0.);
    }
}
