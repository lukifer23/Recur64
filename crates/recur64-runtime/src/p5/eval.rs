//! TUNE evaluation and the offline selector diagnostics of P5.
//!
//! Evaluation runs the same `ActiveSearchModel::run` the training uses, on the
//! inference backend, over every TUNE position. The proof structure is used only
//! OFFLINE to classify what the learned selector did; it never enters a model input.

use std::collections::{BTreeMap, HashSet};

use burn::prelude::*;
use serde::Serialize;

use recur64_core::GameState;
use recur64_model::active::loss::selector_nll_sum;
use recur64_model::active::{ActiveSearchModel, QueryRecord, RunOptions, Selection};

use crate::proof::trace::{Path, PositionTrace};

use super::data::Dataset;
use super::recipe::{BUDGETS, teacher_key_base};
use super::teacher::{SeededProofTeacher, follow_key};

/// Seed of the diagnostic evaluation teacher (fixed; not a run seed).
pub const EVAL_TEACHER_SEED: u64 = 0x7E57_0001;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvalSelection {
    Active,
    Fixed,
    /// The seeded, latched training teacher, with a fixed evaluation seed.
    Teacher,
}

impl EvalSelection {
    pub fn label(self) -> &'static str {
        match self {
            EvalSelection::Active => "active",
            EvalSelection::Fixed => "fixed_bfs_actionid_v1",
            EvalSelection::Teacher => "teacher_proof_teacher_seeded_v1",
        }
    }
}

/// Policy metrics averaged over examples.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Metrics {
    pub n: usize,
    pub top1: f64,
    pub correct_mass: f64,
    pub ce: f64,
    pub entropy: f64,
    pub chance_top1: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ExampleResult {
    pub top1: f64,
    pub mass: f64,
    pub ce: f64,
    pub entropy: f64,
    pub chance: f64,
}

fn mean_of(rows: &[&ExampleResult]) -> Metrics {
    let n = rows.len();
    let f = |g: fn(&ExampleResult) -> f64| rows.iter().map(|r| g(r)).sum::<f64>() / n.max(1) as f64;
    Metrics {
        n,
        top1: f(|r| r.top1),
        correct_mass: f(|r| r.mass),
        ce: f(|r| r.ce),
        entropy: f(|r| r.entropy),
        chance_top1: f(|r| r.chance),
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct EvalSummary {
    pub budget: usize,
    pub selection: String,
    pub pooled: Metrics,
    /// Unweighted mean of the per-cell means.
    pub macro_cell: Metrics,
    pub cells: BTreeMap<String, Metrics>,
    pub families: BTreeMap<String, Metrics>,
    pub depths: BTreeMap<String, Metrics>,
}

pub fn cell_key(p: &crate::proof::targets::ProofPosition) -> String {
    format!("{} M{}", p.family, p.mate_depth)
}

/// The policy metrics of one position from its log-probability row (legal candidates only).
/// Shared by every evaluator so ACTIVE and ALL-INFO are scored by the same function.
pub fn example_result(row: &[f32], p: &crate::proof::targets::ProofPosition) -> ExampleResult {
    let correct = &p.correct;
    let mass: f64 = correct
        .iter()
        .map(|&c| f64::from(row[c as usize]).exp())
        .sum();
    let ce: f64 = -correct
        .iter()
        .map(|&c| f64::from(row[c as usize]))
        .sum::<f64>()
        / correct.len() as f64;
    let entropy: f64 = -row
        .iter()
        .map(|&v| f64::from(v).exp() * f64::from(v))
        .sum::<f64>();
    let mut best = 0usize;
    for (j, &v) in row.iter().enumerate() {
        if v > row[best] {
            best = j;
        }
    }
    ExampleResult {
        top1: f64::from(u8::from(correct.contains(&(best as u32)))),
        mass,
        ce,
        entropy,
        chance: f64::from(p.chance_top1),
    }
}

fn summarise(
    data: &Dataset,
    results: &[ExampleResult],
    budget: usize,
    selection: EvalSelection,
) -> EvalSummary {
    summarise_positions(
        data.positions(),
        results,
        budget,
        selection.label().to_string(),
    )
}

/// Pooled, per-cell, per-family and per-depth metrics of per-position results.
pub fn summarise_positions(
    positions: &[crate::proof::targets::ProofPosition],
    results: &[ExampleResult],
    budget: usize,
    selection_label: String,
) -> EvalSummary {
    let mut cells: BTreeMap<String, Vec<&ExampleResult>> = BTreeMap::new();
    let mut fams: BTreeMap<String, Vec<&ExampleResult>> = BTreeMap::new();
    let mut deps: BTreeMap<String, Vec<&ExampleResult>> = BTreeMap::new();
    for (p, r) in positions.iter().zip(results) {
        cells.entry(cell_key(p)).or_default().push(r);
        fams.entry(p.family.clone()).or_default().push(r);
        deps.entry(format!("M{}", p.mate_depth))
            .or_default()
            .push(r);
    }
    let all: Vec<&ExampleResult> = results.iter().collect();
    let cell_metrics: BTreeMap<String, Metrics> =
        cells.iter().map(|(k, v)| (k.clone(), mean_of(v))).collect();
    let k = cell_metrics.len().max(1) as f64;
    let mac = |g: fn(&Metrics) -> f64| cell_metrics.values().map(g).sum::<f64>() / k;
    EvalSummary {
        budget,
        selection: selection_label,
        pooled: mean_of(&all),
        macro_cell: Metrics {
            n: results.len(),
            top1: mac(|m| m.top1),
            correct_mass: mac(|m| m.correct_mass),
            ce: mac(|m| m.ce),
            entropy: mac(|m| m.entropy),
            chance_top1: mac(|m| m.chance_top1),
        },
        cells: cell_metrics,
        families: fams.iter().map(|(k, v)| (k.clone(), mean_of(v))).collect(),
        depths: deps.iter().map(|(k, v)| (k.clone(), mean_of(v))).collect(),
    }
}

/// How the learned (or any) selector's queries relate to the proof structure,
/// aggregated over a set of examples.
#[derive(Debug, Clone, Default, Serialize)]
pub struct SelectorDiag {
    pub examples: usize,
    pub queries: usize,
    pub proof_admissible: usize,
    pub refute_admissible: usize,
    pub off_target: usize,
    /// Queries made after the proof was already complete (forced filler).
    pub post_completion_queries: usize,
    pub first_query_correct_root: usize,
    pub depth_sum: u64,
    pub depth_histogram: BTreeMap<u32, u64>,
    pub distinct_root_branches_sum: usize,
    /// Sum over queries of (residual before - residual after).
    pub residual_decrease_sum: i64,
    pub queries_reducing_residual: usize,
    pub final_residual_sum: u64,
    pub complete_after_budget: usize,
    /// Positions with `Q* <= budget`: the ideal ceiling.
    pub ideal_ceiling: usize,
}

impl SelectorDiag {
    fn merge(&mut self, o: &SelectorDiag) {
        self.examples += o.examples;
        self.queries += o.queries;
        self.proof_admissible += o.proof_admissible;
        self.refute_admissible += o.refute_admissible;
        self.off_target += o.off_target;
        self.post_completion_queries += o.post_completion_queries;
        self.first_query_correct_root += o.first_query_correct_root;
        self.depth_sum += o.depth_sum;
        for (k, v) in &o.depth_histogram {
            *self.depth_histogram.entry(*k).or_default() += v;
        }
        self.distinct_root_branches_sum += o.distinct_root_branches_sum;
        self.residual_decrease_sum += o.residual_decrease_sum;
        self.queries_reducing_residual += o.queries_reducing_residual;
        self.final_residual_sum += o.final_residual_sum;
        self.complete_after_budget += o.complete_after_budget;
        self.ideal_ceiling += o.ideal_ceiling;
    }
}

/// Offline classification of one example's queries against its proof trace.
pub fn classify_queries(
    trace: &PositionTrace,
    queries: &[QueryRecord],
    budget: usize,
) -> SelectorDiag {
    let mut d = SelectorDiag {
        examples: 1,
        ideal_ceiling: usize::from(trace.q_star <= budget as u64),
        ..Default::default()
    };
    // paths[slot] = root-relative action path of the node in that tree slot.
    let mut paths: Vec<Path> = vec![Vec::new()];
    let mut s: HashSet<Path> = HashSet::new();
    let mut branches: HashSet<usize> = HashSet::new();
    let root_correct: HashSet<u16> = trace.nodes[trace.root as usize]
        .alts
        .iter()
        .map(|a| a.a)
        .collect();
    for (i, q) in queries.iter().enumerate() {
        let mut e = paths[q.parent_slot].clone();
        e.push(q.action);
        let before = trace.residual(&s);
        let adm = trace.admissible(&s);
        d.queries += 1;
        if before == 0 {
            d.post_completion_queries += 1;
        }
        if adm.proof.contains(&e) {
            d.proof_admissible += 1;
        } else if adm.refute.contains(&e) {
            d.refute_admissible += 1;
        } else {
            d.off_target += 1;
        }
        if i == 0 && root_correct.contains(&q.action) {
            d.first_query_correct_root += 1;
        }
        d.depth_sum += u64::from(q.depth);
        *d.depth_histogram.entry(q.depth).or_default() += 1;
        branches.insert(q.branch);
        s.insert(e.clone());
        let after = trace.residual(&s);
        d.residual_decrease_sum += before as i64 - after as i64;
        d.queries_reducing_residual += usize::from(after < before);
        paths.push(e);
    }
    d.distinct_root_branches_sum = branches.len();
    d.final_residual_sum = trace.residual(&s);
    d.complete_after_budget = usize::from(trace.is_complete(&s));
    d
}

/// P5.2 (derived, evaluation-only): the selector diagnostic with the proof-completion
/// boundary made explicit. Once the proof residual is zero, STOP is masked and the
/// forced remaining queries are filler; counting them in the same denominator as
/// pre-completion queries makes the off-target share hard to read. This splits them.
#[derive(Debug, Clone, Default, Serialize)]
pub struct RefinedDiag {
    pub examples: usize,
    pub queries: usize,
    pub pre_completion_queries: usize,
    pub pre_completion_proof_admissible: usize,
    pub pre_completion_refute_admissible: usize,
    pub pre_completion_off_target: usize,
    /// Residual decrease summed over pre-completion queries.
    pub pre_completion_residual_decrease_sum: u64,
    pub pre_completion_depth_histogram: BTreeMap<u32, u64>,
    /// Selector entropy / margin summed over pre-completion queries that recorded them.
    pub pre_completion_selector_entropy_sum: f64,
    pub pre_completion_selector_margin_sum: f64,
    pub pre_completion_selector_stat_count: usize,
    pub post_completion_queries: usize,
    /// Always zero: a proof-admissible edge cannot exist once the residual is zero
    /// (asserted; kept so the evidence shows the invariant held).
    pub post_completion_proof_admissible: usize,
    pub post_completion_refute_admissible: usize,
    pub post_completion_off_target: usize,
    pub post_completion_depth_histogram: BTreeMap<u32, u64>,
    pub positions_with_post_completion_queries: usize,
    /// Number of queries after which the residual first reached zero -> positions.
    pub first_completion_step: BTreeMap<u32, u64>,
    pub never_complete: usize,
    /// For k in {1,2,4,8} <= budget: positions whose proof was complete after k queries.
    pub complete_after_query: BTreeMap<u32, u64>,
}

impl RefinedDiag {
    fn merge(&mut self, o: &RefinedDiag) {
        self.examples += o.examples;
        self.queries += o.queries;
        self.pre_completion_queries += o.pre_completion_queries;
        self.pre_completion_proof_admissible += o.pre_completion_proof_admissible;
        self.pre_completion_refute_admissible += o.pre_completion_refute_admissible;
        self.pre_completion_off_target += o.pre_completion_off_target;
        self.pre_completion_residual_decrease_sum += o.pre_completion_residual_decrease_sum;
        self.pre_completion_selector_entropy_sum += o.pre_completion_selector_entropy_sum;
        self.pre_completion_selector_margin_sum += o.pre_completion_selector_margin_sum;
        self.pre_completion_selector_stat_count += o.pre_completion_selector_stat_count;
        self.post_completion_queries += o.post_completion_queries;
        self.post_completion_proof_admissible += o.post_completion_proof_admissible;
        self.post_completion_refute_admissible += o.post_completion_refute_admissible;
        self.post_completion_off_target += o.post_completion_off_target;
        self.positions_with_post_completion_queries += o.positions_with_post_completion_queries;
        self.never_complete += o.never_complete;
        for (dst, src) in [
            (
                &mut self.pre_completion_depth_histogram,
                &o.pre_completion_depth_histogram,
            ),
            (
                &mut self.post_completion_depth_histogram,
                &o.post_completion_depth_histogram,
            ),
        ] {
            for (k, v) in src {
                *dst.entry(*k).or_default() += v;
            }
        }
        for (dst, src) in [
            (&mut self.first_completion_step, &o.first_completion_step),
            (&mut self.complete_after_query, &o.complete_after_query),
        ] {
            for (k, v) in src {
                *dst.entry(*k).or_default() += v;
            }
        }
    }
}

/// Offline classification of one example's queries with the completion boundary explicit.
pub fn classify_queries_refined(
    trace: &PositionTrace,
    queries: &[QueryRecord],
    budget: usize,
) -> anyhow::Result<RefinedDiag> {
    let mut d = RefinedDiag {
        examples: 1,
        ..Default::default()
    };
    let mut paths: Vec<Path> = vec![Vec::new()];
    let mut s: HashSet<Path> = HashSet::new();
    let mut completed_at: Option<u32> = None;
    let mut residual_after: Vec<u64> = Vec::with_capacity(queries.len());
    for (i, q) in queries.iter().enumerate() {
        let mut e = paths[q.parent_slot].clone();
        e.push(q.action);
        let before = trace.residual(&s);
        let adm = trace.admissible(&s);
        d.queries += 1;
        let (proof, refute) = (adm.proof.contains(&e), adm.refute.contains(&e));
        if before == 0 {
            anyhow::ensure!(
                !proof,
                "a proof-admissible edge exists after the proof completed (query {i})"
            );
            d.post_completion_queries += 1;
            if refute {
                d.post_completion_refute_admissible += 1;
            } else {
                d.post_completion_off_target += 1;
            }
            *d.post_completion_depth_histogram
                .entry(q.depth)
                .or_default() += 1;
        } else {
            d.pre_completion_queries += 1;
            if proof {
                d.pre_completion_proof_admissible += 1;
            } else if refute {
                d.pre_completion_refute_admissible += 1;
            } else {
                d.pre_completion_off_target += 1;
            }
            *d.pre_completion_depth_histogram.entry(q.depth).or_default() += 1;
            if let (Some(ent), Some(mar)) = (q.selector_entropy, q.selector_margin) {
                d.pre_completion_selector_entropy_sum += f64::from(ent);
                d.pre_completion_selector_margin_sum += f64::from(mar);
                d.pre_completion_selector_stat_count += 1;
            }
        }
        s.insert(e.clone());
        let after = trace.residual(&s);
        anyhow::ensure!(
            after <= before,
            "the proof residual increased ({before} -> {after}) at query {i}"
        );
        if before > 0 {
            d.pre_completion_residual_decrease_sum += before - after;
        }
        if after == 0 && completed_at.is_none() {
            completed_at = Some(i as u32 + 1);
        }
        residual_after.push(after);
        paths.push(e);
    }
    d.positions_with_post_completion_queries = usize::from(d.post_completion_queries > 0);
    match completed_at {
        Some(step) => *d.first_completion_step.entry(step).or_default() += 1,
        None => d.never_complete = 1,
    }
    for k in [1u32, 2, 4, 8] {
        if k as usize <= budget && completed_at.is_some_and(|c| c <= k) {
            *d.complete_after_query.entry(k).or_default() += 1;
        }
    }
    Ok(d)
}

/// Everything one evaluation pass produces.
pub struct EvalOutput {
    pub summary: EvalSummary,
    /// Selector NLL under the teacher (sum, supervised decisions); teacher mode only.
    pub selector_nll: Option<(f64, usize)>,
    /// Offline selector diagnostics per cell and pooled (budget > 0 only).
    pub selector_diag: Option<(SelectorDiag, BTreeMap<String, SelectorDiag>)>,
    /// P5.2 refined diagnostic (pre/post completion), same conditions as `selector_diag`.
    pub refined_diag: Option<(RefinedDiag, BTreeMap<String, RefinedDiag>)>,
    /// Per-position policy results in dataset order (for paired bootstraps).
    pub per_position: Vec<ExampleResult>,
}

/// Evaluate `model` on every position of `data` at `budget`.
pub fn evaluate<B: Backend>(
    model: &ActiveSearchModel<B>,
    data: &Dataset,
    budget: usize,
    selection: EvalSelection,
    batch: usize,
    device: &B::Device,
) -> anyhow::Result<EvalOutput> {
    anyhow::ensure!(
        BUDGETS.contains(&budget),
        "evaluation budgets are {{0,2,4,8}}"
    );
    let mut results: Vec<ExampleResult> = Vec::with_capacity(data.positions().len());
    let mut nll = (0.0f64, 0usize);
    let mut diag_all = SelectorDiag::default();
    let mut diag_cells: BTreeMap<String, SelectorDiag> = BTreeMap::new();
    let mut refined_all = RefinedDiag::default();
    let mut refined_cells: BTreeMap<String, RefinedDiag> = BTreeMap::new();
    let base = teacher_key_base(EVAL_TEACHER_SEED);
    let n = data.positions().len();
    let mut start = 0usize;
    while start < n {
        let end = (start + batch).min(n);
        let positions = &data.positions()[start..end];
        let roots: Vec<GameState> = positions
            .iter()
            .map(|p| GameState::from_fen(&p.fen).map_err(|e| anyhow::anyhow!("{}: {e}", p.id)))
            .collect::<anyhow::Result<_>>()?;
        let mut opts = RunOptions::forced(budget);
        opts.health_checks = false;
        let mut teacher;
        let sel = match selection {
            EvalSelection::Active => Selection::Active,
            EvalSelection::Fixed => Selection::Fixed,
            EvalSelection::Teacher => {
                let traces: Vec<&_> = (start..end).map(|i| &data.traces[i]).collect();
                let keys: Vec<u64> = (start..end)
                    .map(|i| follow_key(base, &data.positions()[i].id, i as u64, budget))
                    .collect();
                teacher = SeededProofTeacher::new(traces, keys);
                Selection::Script(&mut teacher)
            }
        };
        let out = model.run(&roots, &opts, sel, device)?;
        let w = out.readout.policy.mask.dims()[1];
        let lp: Vec<f32> = out
            .readout
            .policy
            .log_probs
            .clone()
            .into_data()
            .to_vec()
            .map_err(|e| anyhow::anyhow!("reading policy: {e:?}"))?;
        anyhow::ensure!(
            lp.iter().all(|v| v.is_finite()),
            "non-finite policy output during evaluation"
        );
        for (i, p) in positions.iter().enumerate() {
            let row = &lp[i * w..i * w + p.legal.len()];
            results.push(example_result(row, p));
        }
        if matches!(selection, EvalSelection::Teacher)
            && let Some((sum, count)) = selector_nll_sum(&out.selector_steps)
        {
            let v: f64 = sum
                .into_data()
                .to_vec::<f32>()
                .map_err(|e| anyhow::anyhow!("{e:?}"))?[0]
                .into();
            nll.0 += v;
            nll.1 += count;
        }
        if matches!(selection, EvalSelection::Active) && budget > 0 {
            for (i, p) in positions.iter().enumerate() {
                let d = classify_queries(&data.traces[start + i], &out.traces[i], budget);
                diag_all.merge(&d);
                diag_cells.entry(cell_key(p)).or_default().merge(&d);
                let r = classify_queries_refined(&data.traces[start + i], &out.traces[i], budget)
                    .map_err(|e| anyhow::anyhow!("{}: {e}", p.id))?;
                refined_all.merge(&r);
                refined_cells.entry(cell_key(p)).or_default().merge(&r);
            }
        }
        start = end;
    }
    let selector_nll = (matches!(selection, EvalSelection::Teacher) && nll.1 > 0).then_some(nll);
    let selector_diag = (matches!(selection, EvalSelection::Active) && budget > 0)
        .then_some((diag_all, diag_cells));
    let refined_diag = (matches!(selection, EvalSelection::Active) && budget > 0)
        .then_some((refined_all, refined_cells));
    Ok(EvalOutput {
        summary: summarise(data, &results, budget, selection),
        selector_nll,
        selector_diag,
        refined_diag,
        per_position: results,
    })
}

/// The frozen LR-screen score of one run at update 800: the mean over budgets
/// {0,2,4,8} of the mean over the six TUNE cells of the ACTIVE policy CE, all 24
/// values equally weighted.
pub fn screen_score(active: &[EvalSummary]) -> anyhow::Result<f64> {
    anyhow::ensure!(
        active.len() == BUDGETS.len(),
        "need ACTIVE evaluations at exactly {{0,2,4,8}}"
    );
    let mut total = 0.0;
    for b in BUDGETS {
        let s = active
            .iter()
            .find(|s| s.budget == b && s.selection == EvalSelection::Active.label())
            .ok_or_else(|| anyhow::anyhow!("missing ACTIVE evaluation at B{b}"))?;
        anyhow::ensure!(
            s.cells.len() == 6,
            "B{b}: {} cells, expected 6",
            s.cells.len()
        );
        let m: f64 = s.cells.values().map(|c| c.ce).sum::<f64>() / 6.0;
        anyhow::ensure!(m.is_finite(), "B{b}: non-finite CE");
        total += m;
    }
    Ok(total / BUDGETS.len() as f64)
}
