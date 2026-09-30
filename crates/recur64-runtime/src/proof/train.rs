//! Fixed-data, policy-only training and evaluation on `ProofTargetsV1`.
//!
//! Exact-target training (no WDL loss) with the learner's optimizer and gradient
//! reduction (`accumulated_update`), deterministic seeded epoch shuffles, and a
//! per-position evaluator. Nothing here searches, plays games or reads a network
//! label: the targets are the exact proof targets.

use std::collections::BTreeMap;

use burn::module::AutodiffModule;
use burn::optim::Optimizer;
use burn::prelude::*;
use burn::tensor::TensorData;
use burn::tensor::backend::AutodiffBackend;
use serde::{Deserialize, Serialize};

use recur64_core::{ActionId, CandidateFactsV1, GameState, ObservationV1};
use recur64_model::candidate::CandidateInputs;
use recur64_model::loss::Targets;
use recur64_model::net::NeuralModel;

use super::generator::{Rng, mix};
use super::targets::ProofTargets;
use crate::accum::{LossMode, MicroBatch, UpdateReport, accumulated_update};
use crate::inference::{BatchEvaluator, BatchedModel};
use crate::learner::lr_at;

/// One position, fully prepared for the network.
pub struct PreparedPos {
    pub id: String,
    pub family: String,
    pub depth: u8,
    pub obs: ObservationV1,
    pub legal: Vec<ActionId>,
    pub facts: Vec<CandidateFactsV1>,
    /// Uniform exact target over the correct set, aligned to `legal`.
    pub target: Vec<f32>,
    pub correct: Vec<u32>,
    pub chance: f32,
}

/// Build network inputs for every position (facts always computed; models that
/// do not consume them ignore them).
pub fn prepare(t: &ProofTargets) -> anyhow::Result<Vec<PreparedPos>> {
    t.positions
        .iter()
        .map(|p| {
            let state = GameState::from_fen(&p.fen)?;
            let legal = state.legal_actions();
            anyhow::ensure!(
                legal.iter().map(|a| a.index() as u16).collect::<Vec<_>>() == p.legal,
                "{}: stored legal actions differ from GameState",
                p.id
            );
            Ok(PreparedPos {
                id: p.id.clone(),
                family: p.family.clone(),
                depth: p.mate_depth,
                obs: recur64_core::encode_observation_v1(&state),
                facts: recur64_core::candidate_facts(&state),
                legal,
                target: p.target(),
                correct: p.correct.clone(),
                chance: p.chance_top1,
            })
        })
        .collect()
}

/// Per-position evaluation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PosResult {
    pub id: String,
    pub family: String,
    pub depth: u8,
    /// The policy's argmax move is a correct move.
    pub top1: bool,
    /// Probability mass on the correct set.
    pub mass: f32,
    /// Cross-entropy against the exact uniform target (nats).
    pub ce: f32,
    pub entropy: f32,
    pub chance: f32,
}

/// Aggregate over a group of positions.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Agg {
    pub n: usize,
    pub top1: f64,
    pub mass: f64,
    pub ce: f64,
    pub entropy: f64,
    pub chance_top1: f64,
}

/// Aggregates overall, by depth ("M1".."M5") and by family.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EvalSummary {
    pub overall: Agg,
    pub by_depth: BTreeMap<String, Agg>,
    pub by_family: BTreeMap<String, Agg>,
}

fn agg(rs: &[&PosResult]) -> Agg {
    let n = rs.len().max(1) as f64;
    Agg {
        n: rs.len(),
        top1: rs.iter().filter(|r| r.top1).count() as f64 / n,
        mass: rs.iter().map(|r| r.mass as f64).sum::<f64>() / n,
        ce: rs.iter().map(|r| r.ce as f64).sum::<f64>() / n,
        entropy: rs.iter().map(|r| r.entropy as f64).sum::<f64>() / n,
        chance_top1: rs.iter().map(|r| r.chance as f64).sum::<f64>() / n,
    }
}

pub fn summarize(results: &[PosResult]) -> EvalSummary {
    let all: Vec<&PosResult> = results.iter().collect();
    let mut by_depth: BTreeMap<String, Vec<&PosResult>> = BTreeMap::new();
    let mut by_family: BTreeMap<String, Vec<&PosResult>> = BTreeMap::new();
    for r in results {
        by_depth.entry(format!("M{}", r.depth)).or_default().push(r);
        by_family.entry(r.family.clone()).or_default().push(r);
    }
    EvalSummary {
        overall: agg(&all),
        by_depth: by_depth.into_iter().map(|(k, v)| (k, agg(&v))).collect(),
        by_family: by_family.into_iter().map(|(k, v)| (k, agg(&v))).collect(),
    }
}

/// Evaluate `model` (inference backend) on `data`, one result per position.
pub fn evaluate<B, M>(
    model: &M,
    device: &B::Device,
    data: &[PreparedPos],
) -> anyhow::Result<Vec<PosResult>>
where
    B: Backend,
    M: NeuralModel<B>,
{
    let needs = model.needs_candidate_facts();
    let batched = BatchedModel::new(model.clone(), 1, device.clone());
    let mut out = Vec::with_capacity(data.len());
    for chunk in data.chunks(64) {
        let obs: Vec<ObservationV1> = chunk.iter().map(|p| p.obs.clone()).collect();
        let legal: Vec<Vec<ActionId>> = chunk.iter().map(|p| p.legal.clone()).collect();
        let facts: Vec<Option<Vec<CandidateFactsV1>>> = chunk
            .iter()
            .map(|p| needs.then(|| p.facts.clone()))
            .collect();
        let res = batched
            .evaluate_batch_with_facts(&obs, &legal, &facts)
            .map_err(|e| anyhow::anyhow!("evaluation failed: {e}"))?;
        for (p, r) in chunk.iter().zip(res) {
            let argmax = r
                .policy
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i as u32)
                .unwrap_or(0);
            let mass: f32 = p.correct.iter().map(|&c| r.policy[c as usize]).sum();
            let ce: f32 = p
                .target
                .iter()
                .zip(&r.policy)
                .map(|(t, q)| if *t > 0.0 { -t * q.max(1e-12).ln() } else { 0.0 })
                .sum();
            let entropy: f32 = r
                .policy
                .iter()
                .map(|q| if *q > 0.0 { -q * q.ln() } else { 0.0 })
                .sum();
            anyhow::ensure!(
                mass.is_finite() && ce.is_finite() && entropy.is_finite(),
                "{}: non-finite evaluation",
                p.id
            );
            out.push(PosResult {
                id: p.id.clone(),
                family: p.family.clone(),
                depth: p.depth,
                top1: p.correct.contains(&argmax),
                mass,
                ce,
                entropy,
                chance: p.chance,
            });
        }
    }
    Ok(out)
}

/// Training request.
#[derive(Debug, Clone, Serialize)]
pub struct TrainSpec {
    pub updates: usize,
    pub warmup: usize,
    pub lr: f64,
    /// Physical micro-batch and accumulation steps (effective batch = product).
    pub micro: usize,
    pub accum: usize,
    pub seed: u64,
    /// Evaluate on `tune` every this many updates (0 = only at the end).
    pub eval_every: usize,
}

/// One update's record.
#[derive(Debug, Clone, Serialize)]
pub struct UpdateRecord {
    pub update: usize,
    pub lr: f64,
    #[serde(flatten)]
    pub report: UpdateReport,
}

/// Everything a training run produced besides the model.
pub struct TrainRun {
    pub updates: Vec<UpdateRecord>,
    /// (update, tune summary) at each evaluation point.
    pub tune_curve: Vec<(usize, EvalSummary)>,
    pub epochs_seen: f64,
}

/// Micro-batch tensors for `idx` positions of `data`.
fn micro_batch<B: Backend>(
    data: &[PreparedPos],
    idx: &[usize],
    needs_facts: bool,
    device: &B::Device,
) -> anyhow::Result<MicroBatch<B>> {
    let obs: Vec<&ObservationV1> = idx.iter().map(|&i| &data[i].obs).collect();
    let legal: Vec<Vec<ActionId>> = idx.iter().map(|&i| data[i].legal.clone()).collect();
    let facts: Vec<&[CandidateFactsV1]> = idx.iter().map(|&i| data[i].facts.as_slice()).collect();
    let inp = CandidateInputs::<B>::from_parts(&obs, &legal, &facts, device)?;
    let (b, w) = (idx.len(), inp.cands.width);
    let mut t = vec![0.0f32; b * w];
    for (row, &i) in idx.iter().enumerate() {
        t[row * w..row * w + data[i].target.len()].copy_from_slice(&data[i].target);
    }
    Ok(MicroBatch {
        board: inp.board,
        cands: inp.cands,
        facts: needs_facts.then_some(inp.facts),
        targets: Targets {
            policy_target: Tensor::<B, 2>::from_data(TensorData::new(t, [b, w]), device),
            // Unused under LossMode::PolicyOnly; required by the shared type.
            wdl_target: Tensor::<B, 1, Int>::from_data(TensorData::new(vec![1i32; b], [b]), device),
        },
        examples: b,
    })
}

/// Train `model` for `spec.updates` optimizer updates on `train`, evaluating on
/// `tune` at the configured points. Policy-only exact-target loss.
#[allow(clippy::too_many_arguments)]
pub fn train<B, MT, MI, O>(
    mut model: MT,
    optim: &mut O,
    spec: &TrainSpec,
    train: &[PreparedPos],
    tune: &[PreparedPos],
    device: &B::Device,
    inner_device: &burn::tensor::Device<B::InnerBackend>,
    mut on_update: impl FnMut(&UpdateRecord),
) -> anyhow::Result<(MT, TrainRun)>
where
    B: AutodiffBackend,
    MT: NeuralModel<B> + AutodiffModule<B, InnerModule = MI>,
    MI: NeuralModel<B::InnerBackend>,
    O: Optimizer<MT, B>,
{
    anyhow::ensure!(!train.is_empty(), "no training positions");
    let needs = model.needs_candidate_facts();
    let effective = spec.micro * spec.accum;
    let mut order: Vec<usize> = Vec::new();
    let mut epoch = 0u64;
    let mut cursor = 0usize;
    let next_index = |order: &mut Vec<usize>, cursor: &mut usize, epoch: &mut u64| -> usize {
        if *cursor >= order.len() {
            // A fresh seeded permutation each epoch (deterministic ordering).
            let mut p: Vec<usize> = (0..train.len()).collect();
            let mut rng = Rng(mix(spec.seed ^ mix(*epoch + 0xE90C)));
            for i in (1..p.len()).rev() {
                let j = (rng.next_u64() % (i as u64 + 1)) as usize;
                p.swap(i, j);
            }
            *order = p;
            *cursor = 0;
            *epoch += 1;
        }
        let i = order[*cursor];
        *cursor += 1;
        i
    };

    let mut run = TrainRun {
        updates: Vec::with_capacity(spec.updates),
        tune_curve: Vec::new(),
        epochs_seen: 0.0,
    };
    let eval_now = |model: &MT| -> anyhow::Result<EvalSummary> {
        let inner = model.valid();
        Ok(summarize(&evaluate::<B::InnerBackend, MI>(
            &inner,
            inner_device,
            tune,
        )?))
    };
    for u in 0..spec.updates {
        let lr = lr_at(u as u64, spec.lr, spec.warmup as u64, spec.updates as u64);
        let mut micros = Vec::with_capacity(spec.accum);
        for _ in 0..spec.accum {
            let idx: Vec<usize> = (0..spec.micro)
                .map(|_| next_index(&mut order, &mut cursor, &mut epoch))
                .collect();
            micros.push(micro_batch::<B>(train, &idx, needs, device)?);
        }
        let (m, report) =
            accumulated_update::<B, MT, O, _>(model, optim, micros, lr, LossMode::PolicyOnly)
                .map_err(|e| anyhow::anyhow!("update {u}: {e}"))?;
        model = m;
        let rec = UpdateRecord {
            update: u,
            lr,
            report,
        };
        on_update(&rec);
        run.updates.push(rec);
        if spec.eval_every > 0 && (u + 1) % spec.eval_every == 0 && u + 1 < spec.updates {
            run.tune_curve.push((u + 1, eval_now(&model)?));
        }
    }
    run.tune_curve.push((spec.updates, eval_now(&model)?));
    run.epochs_seen = (spec.updates * effective) as f64 / train.len() as f64;
    Ok((model, run))
}

/// Mean and seeded percentile-bootstrap 95% CI of paired per-position values.
pub fn paired_bootstrap(values: &[f64], resamples: usize, seed: u64) -> (f64, f64, f64) {
    let n = values.len();
    if n == 0 {
        return (0.0, 0.0, 0.0);
    }
    let mean = values.iter().sum::<f64>() / n as f64;
    let mut rng = Rng(mix(seed));
    let mut means: Vec<f64> = (0..resamples)
        .map(|_| {
            (0..n)
                .map(|_| values[(rng.next_u64() % n as u64) as usize])
                .sum::<f64>()
                / n as f64
        })
        .collect();
    means.sort_by(f64::total_cmp);
    let lo = means[((resamples as f64) * 0.025) as usize];
    let hi = means[(((resamples as f64) * 0.975) as usize).min(resamples - 1)];
    (mean, lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bootstrap_ci_brackets_the_mean_and_is_seeded() {
        let v: Vec<f64> = (0..400).map(|i| if i % 4 == 0 { 1.0 } else { 0.0 }).collect();
        let (m, lo, hi) = paired_bootstrap(&v, 2000, 5);
        assert!((m - 0.25).abs() < 1e-9);
        assert!(lo < m && m < hi && hi - lo < 0.12);
        assert_eq!(paired_bootstrap(&v, 2000, 5), (m, lo, hi));
        let zeros = vec![0.0; 50];
        let (m0, lo0, hi0) = paired_bootstrap(&zeros, 500, 1);
        assert_eq!((m0, lo0, hi0), (0.0, 0.0, 0.0));
    }

    #[test]
    fn summary_groups_by_depth_and_family() {
        let mk = |depth, family: &str, top1| PosResult {
            id: "x".into(),
            family: family.into(),
            depth,
            top1,
            mass: 0.5,
            ce: 1.0,
            entropy: 2.0,
            chance: 0.1,
        };
        let s = summarize(&[mk(1, "KQvK", true), mk(1, "KRvK", false), mk(2, "KQvK", true)]);
        assert_eq!(s.overall.n, 3);
        assert!((s.by_depth["M1"].top1 - 0.5).abs() < 1e-12);
        assert!((s.by_depth["M2"].top1 - 1.0).abs() < 1e-12);
        assert_eq!(s.by_family["KQvK"].n, 2);
    }
}
