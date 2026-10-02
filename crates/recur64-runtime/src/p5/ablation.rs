//! `query_content_ablation_v1` evaluation (P5.2, optional, evaluation-only).
//!
//! Question: how much of a policy's use of queries comes from the CONTENT of the queried
//! states versus from WHICH edges were queried? Obtain a query path from ACTIVE or the
//! TEACHER, then replay exactly that path twice: once with normal state content and once
//! with the state-derived neural content removed (see
//! `recur64_model::active::QueryContentAblation` for exactly what is removed and what is
//! preserved). Never part of any gate and never reachable from training.

use std::collections::BTreeMap;

use burn::prelude::*;
use serde::Serialize;

use recur64_core::GameState;
use recur64_model::active::{
    ActiveSearchModel, EdgeRef, QueryScript, RunOptions, ScriptStep, Selection, Tree,
};

use super::data::Dataset;
use super::eval::{
    EVAL_TEACHER_SEED, EvalSelection, EvalSummary, ExampleResult, example_result,
    summarise_positions,
};
use super::recipe::{BUDGETS, teacher_key_base};
use super::teacher::{SeededProofTeacher, follow_key};

/// Replays recorded `(parent_slot, action)` query sequences. Tree slots are assigned in
/// query order, so a recorded `(slot, action)` pair names the same edge in a replay.
pub struct ReplayScript {
    pub seqs: Vec<Vec<(usize, u16)>>,
}

impl QueryScript for ReplayScript {
    fn next(
        &mut self,
        example: usize,
        step: usize,
        frontier: &[EdgeRef],
        _tree: &Tree,
    ) -> anyhow::Result<ScriptStep> {
        let (slot, action) = *self
            .seqs
            .get(example)
            .and_then(|s| s.get(step))
            .ok_or_else(|| anyhow::anyhow!("replay sequence {example} has no step {step}"))?;
        let follow = frontier
            .iter()
            .position(|e| e.node_slot == slot && e.action == action)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "replayed edge (slot {slot}, action {action}) is not on the frontier"
                )
            })?;
        Ok(ScriptStep {
            follow,
            targets: Vec::new(),
        })
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct AblationEval {
    pub budget: usize,
    /// `active` or `teacher`: where the query path came from.
    pub path_source: String,
    /// The source run's own policy (reference).
    pub source: EvalSummary,
    /// The same path replayed with normal state content.
    pub normal_state_content: EvalSummary,
    /// The same path replayed with `query_content_ablation_v1`.
    pub ablated_query_state_content: EvalSummary,
    /// Largest per-position |CE difference| between the source run and the normal replay.
    /// Zero (to float noise) proves the replay reproduces the source path.
    pub replay_vs_source_max_abs_ce_diff: f64,
    /// Positions whose ablated top-1 differs from the normal replay's.
    pub positions_top1_changed_by_ablation: usize,
    /// Per-position results of the normal replay and the ablated replay, in dataset order
    /// (for the paired V3.5 Content-Use estimator; not serialised into evidence summaries).
    #[serde(skip)]
    pub normal_per_position: Vec<ExampleResult>,
    #[serde(skip)]
    pub ablated_per_position: Vec<ExampleResult>,
}

fn rows(
    out: &recur64_model::active::ActiveOutput<impl Backend>,
    positions: &[crate::proof::targets::ProofPosition],
) -> anyhow::Result<Vec<ExampleResult>> {
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
    Ok(positions
        .iter()
        .enumerate()
        .map(|(i, p)| example_result(&lp[i * w..i * w + p.legal.len()], p))
        .collect())
}

/// Evaluate the ablation on every position of `data` at `budget` (> 0) for a query path
/// from `source` (ACTIVE or TEACHER).
pub fn evaluate_query_content_ablation<B: Backend>(
    model: &ActiveSearchModel<B>,
    data: &Dataset,
    budget: usize,
    source: EvalSelection,
    batch: usize,
    device: &B::Device,
) -> anyhow::Result<AblationEval> {
    anyhow::ensure!(
        BUDGETS.contains(&budget) && budget > 0,
        "the ablation needs a query budget in {{2,4,8}}"
    );
    anyhow::ensure!(
        matches!(source, EvalSelection::Active | EvalSelection::Teacher),
        "the query path comes from ACTIVE or the TEACHER"
    );
    let n = data.positions().len();
    let base = teacher_key_base(EVAL_TEACHER_SEED);
    let (mut src, mut norm, mut abl) = (Vec::new(), Vec::new(), Vec::new());
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
        // 1. the source path (and the source policy).
        let mut teacher;
        let sel = match source {
            EvalSelection::Active => Selection::Active,
            _ => {
                let traces: Vec<&_> = (start..end).map(|i| &data.traces[i]).collect();
                let keys: Vec<u64> = (start..end)
                    .map(|i| follow_key(base, &data.positions()[i].id, i as u64, budget))
                    .collect();
                teacher = SeededProofTeacher::new(traces, keys);
                Selection::Script(&mut teacher)
            }
        };
        let out = model.run(&roots, &opts, sel, device)?;
        src.extend(rows(&out, positions)?);
        let seqs: Vec<Vec<(usize, u16)>> = out
            .traces
            .iter()
            .map(|t| t.iter().map(|q| (q.parent_slot, q.action)).collect())
            .collect();
        // 2. the same path, normal content; 3. the same path, ablated content.
        let mut replay = ReplayScript { seqs: seqs.clone() };
        let out_n = model.run(&roots, &opts, Selection::Script(&mut replay), device)?;
        norm.extend(rows(&out_n, positions)?);
        let mut replay = ReplayScript { seqs };
        let abl_opts = RunOptions {
            health_checks: false,
            ..RunOptions::query_content_ablation_v1(budget)
        };
        let out_a = model.run(&roots, &abl_opts, Selection::Script(&mut replay), device)?;
        abl.extend(rows(&out_a, positions)?);
        start = end;
    }
    let label = |s: &str| {
        format!(
            "{}_path_{s}",
            if source == EvalSelection::Active {
                "active"
            } else {
                "teacher"
            }
        )
    };
    let diff = src
        .iter()
        .zip(&norm)
        .map(|(a, b)| (a.ce - b.ce).abs())
        .fold(0.0f64, f64::max);
    let changed = norm
        .iter()
        .zip(&abl)
        .filter(|(a, b)| a.top1 != b.top1)
        .count();
    let pos = data.positions();
    Ok(AblationEval {
        budget,
        path_source: if source == EvalSelection::Active {
            "active"
        } else {
            "teacher"
        }
        .into(),
        source: summarise_positions(pos, &src, budget, label("source")),
        normal_state_content: summarise_positions(
            pos,
            &norm,
            budget,
            label("normal_state_content"),
        ),
        ablated_query_state_content: summarise_positions(
            pos,
            &abl,
            budget,
            label("ablated_query_state_content"),
        ),
        replay_vs_source_max_abs_ce_diff: diff,
        positions_top1_changed_by_ablation: changed,
        normal_per_position: norm,
        ablated_per_position: abl,
    })
}

/// Cell-wise top-1 of an `EvalSummary` (convenience for reports).
pub fn cell_top1(s: &EvalSummary) -> BTreeMap<String, f64> {
    s.cells.iter().map(|(k, m)| (k.clone(), m.top1)).collect()
}
