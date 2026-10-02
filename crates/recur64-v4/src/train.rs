//! TRAIN-only training and evaluation machinery for the V4 mechanism studies (stages A-D).
//!
//! * Stage A trains the base tower on root CE (budget 0, nothing else executes).
//! * Stage B trains the evidence path with the base detached (`Freeze::BASE`), on label-
//!   independent FIXED / RANDOM query schedules and root CE.
//! * Stage C trains the utility head (and its parent/action state encoder) with base AND
//!   evidence detached, from realised counterfactual utilities.
//!
//! Every gradient reaching an optimiser comes from one of these functions, so the stage
//! boundaries (which groups may move) are enforced here, not by convention.

use burn::optim::GradientsParams;
use burn::optim::GradientsAccumulator;
use burn::prelude::*;
use burn::tensor::TensorData;
use burn::tensor::backend::AutodiffBackend;
use recur64_model::active::{EdgeRef, QueryScript, ScriptStep, Tree};

use crate::data::V4Data;
use crate::model::EvidenceBeliefModel;
use crate::session::{
    ContentMode, Freeze, RunOptions, Selection, Session, root_ce,
};
use crate::util::SplitMix;

/// SplitMix-style mixing of two words (frozen: used for every derived seed).
pub fn mix(a: u64, b: u64) -> u64 {
    let mut r = SplitMix(a ^ b.wrapping_mul(0xD6E8_FEB8_6659_FD93));
    r.next()
}

fn host<B: Backend, const D: usize>(t: Tensor<B, D>) -> Vec<f32> {
    t.into_data().to_vec::<f32>().expect("f32 tensor")
}

fn scalar<B: Backend>(t: Tensor<B, 1>) -> f64 {
    f64::from(host(t)[0])
}

fn target_tensor<B: Backend>(
    data: &V4Data,
    idx: &[usize],
    w: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    Tensor::from_data(
        TensorData::new(data.target_rows(idx, w), [idx.len(), w]),
        device,
    )
}

/// Split `idx` into micro-batches of about `micro` examples, none smaller than 2 (the
/// shuffled-content diagnostic needs two examples per step).
pub fn chunks_min2(idx: &[usize], micro: usize) -> Vec<&[usize]> {
    let micro = micro.max(2);
    let mut out: Vec<&[usize]> = idx.chunks(micro).collect();
    if out.len() >= 2 && out.last().is_some_and(|c| c.len() < 2) {
        let last = out.pop().expect("non-empty");
        let prev = out.pop().expect("non-empty");
        let start = idx.len() - prev.len() - last.len();
        out.push(&idx[start..]);
    }
    out
}

/// Label-independent query schedule used while training the evidence path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sel {
    Fixed,
    Random(u64),
    Utility,
}

impl Sel {
    fn selection(self, micro_index: usize) -> Selection<'static> {
        match self {
            Sel::Fixed => Selection::Fixed,
            Sel::Random(s) => Selection::Random(mix(s, micro_index as u64)),
            Sel::Utility => Selection::Utility,
        }
    }
}

/// One accumulated CE update. `opts` carries the budget and the freeze scope.
pub fn ce_update<B: AutodiffBackend>(
    model: &EvidenceBeliefModel<B>,
    data: &V4Data,
    batch: &[usize],
    micro: usize,
    opts: &RunOptions,
    sel: Sel,
    device: &B::Device,
) -> anyhow::Result<(GradientsParams, f64)> {
    let n_total = batch.len() as f32;
    let mut acc = GradientsAccumulator::<EvidenceBeliefModel<B>>::new();
    let mut loss_sum = 0.0;
    for (mi, chunk) in batch.chunks(micro.max(1)).enumerate() {
        let roots = data.roots(chunk)?;
        let out = model.run(&roots, opts, sel.selection(mi), mix(0xB0, mi as u64), device)?;
        let w = out.log_probs.dims()[1];
        let tgt = target_tensor::<B>(data, chunk, w, device);
        let loss = (out.log_probs * tgt).sum().neg().div_scalar(n_total);
        loss_sum += scalar(loss.clone());
        acc.accumulate(model, GradientsParams::from_grads(loss.backward(), model));
    }
    Ok((acc.grads(), loss_sum))
}

// ---------------------------------------------------------------------------------------
// Policy evaluation (no gradient; works on any backend, including `model.valid()`)
// ---------------------------------------------------------------------------------------

pub enum EvalSel<'a> {
    Fixed,
    Random(u64),
    Utility,
    /// Replay recorded frontier choices (one list per evaluated position), optionally with the
    /// evaluation-only content treatments.
    Replay {
        chosen: &'a [Vec<usize>],
        content: ContentMode,
    },
}

struct ReplayScript<'a> {
    chosen: &'a [Vec<usize>],
}

impl QueryScript for ReplayScript<'_> {
    fn next(
        &mut self,
        example: usize,
        step: usize,
        frontier: &[EdgeRef],
        _tree: &Tree,
    ) -> anyhow::Result<ScriptStep> {
        let follow = *self
            .chosen
            .get(example)
            .and_then(|c| c.get(step))
            .ok_or_else(|| anyhow::anyhow!("replay has no recorded choice for example {example} step {step}"))?;
        anyhow::ensure!(follow < frontier.len(), "replayed index outside the frontier");
        Ok(ScriptStep {
            follow,
            targets: Vec::new(),
        })
    }
}

/// Per-position results of one evaluation, aligned with the evaluated indices.
#[derive(Debug, Clone, Default)]
pub struct PosEval {
    pub ce: Vec<f64>,
    pub top1: Vec<f64>,
    pub mass: Vec<f64>,
    pub delta_norm: Vec<f64>,
    pub chosen: Vec<Vec<usize>>,
}

impl PosEval {
    pub fn mean(v: &[f64]) -> f64 {
        v.iter().sum::<f64>() / v.len().max(1) as f64
    }
}

/// Evaluate root CE / top-1 / correct mass of the model on `idx` at `budget`.
pub fn eval_policy<B: Backend>(
    model: &EvidenceBeliefModel<B>,
    data: &V4Data,
    idx: &[usize],
    budget: usize,
    sel: EvalSel<'_>,
    micro: usize,
    device: &B::Device,
) -> anyhow::Result<PosEval> {
    let mut out = PosEval::default();
    let mut offset = 0usize;
    for chunk in chunks_min2(idx, micro) {
        let roots = data.roots(chunk)?;
        let res = match &sel {
            EvalSel::Fixed => model.run(&roots, &RunOptions::new(budget), Selection::Fixed, 0, device)?,
            EvalSel::Random(seed) => model.run(
                &roots,
                &RunOptions::new(budget),
                Selection::Random(mix(*seed, offset as u64)),
                0,
                device,
            )?,
            EvalSel::Utility => {
                model.run(&roots, &RunOptions::new(budget), Selection::Utility, 0, device)?
            }
            EvalSel::Replay { chosen, content } => {
                let opts = match content {
                    ContentMode::Normal => RunOptions::new(budget),
                    ContentMode::Zero => RunOptions::zero_content_eval(budget),
                    ContentMode::Shuffled => RunOptions::shuffled_content_eval(budget),
                };
                let mut script = ReplayScript {
                    chosen: &chosen[offset..offset + chunk.len()],
                };
                model.run(&roots, &opts, Selection::Script(&mut script), 0, device)?
            }
        };
        let w = res.logits.dims()[1];
        let z = host(res.logits.clone());
        let d = host(res.delta.clone());
        for (r, &i) in chunk.iter().enumerate() {
            let p = data.position(i);
            let vw = p.legal.len();
            let row = &z[r * w..r * w + vw];
            let correct = data.correct(i);
            out.ce.push(f64::from(root_ce(row, &correct)));
            let (mut best, mut arg) = (f32::NEG_INFINITY, 0usize);
            for (c, &v) in row.iter().enumerate() {
                if v > best {
                    best = v;
                    arg = c;
                }
            }
            out.top1.push(f64::from(u8::from(correct.contains(&arg))));
            let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let den: f32 = row.iter().map(|v| (v - mx).exp()).sum();
            out.mass.push(f64::from(
                correct.iter().map(|&c| (row[c] - mx).exp()).sum::<f32>() / den,
            ));
            out.delta_norm.push(f64::from(
                d[r * w..r * w + vw].iter().map(|v| v * v).sum::<f32>().sqrt(),
            ));
            out.chosen.push(res.chosen[r].clone());
        }
        offset += chunk.len();
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------
// Counterfactual probes and the utility head
// ---------------------------------------------------------------------------------------

/// Frozen probe-count default (`K` edges per probed state).
pub const PROBE_K: usize = 8;
/// Pairs closer than this in realised utility are not ranked (nats).
pub const RANK_MARGIN: f64 = 0.01;
/// Regression targets are clipped to `[-C, C]` nats.
pub const REGRESSION_CLIP: f64 = 2.0;

/// One probed partial state of one position.
#[derive(Debug, Clone)]
pub struct ProbeSample {
    /// Index into the data's positions.
    pub position: usize,
    /// Real queries executed before probing (label-independent random prefix).
    pub prefix: usize,
    pub frontier: usize,
    /// Frontier index of the head's argmax over the WHOLE frontier (always probed, first).
    pub preferred: usize,
    /// Probed frontier indices (`picks[0] == preferred`).
    pub picks: Vec<usize>,
    /// Head scores of the picks (before the child is seen).
    pub scores: Vec<f64>,
    /// Realised utilities `CE_before - CE_after_e` of the picks.
    pub u: Vec<f64>,
    pub ce_before: f64,
}

fn fnv(s: &str) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in s.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// The frozen diversity rule: the learner's preferred edge, then `k - 1` distinct frontier
/// indices drawn by a partial Fisher-Yates shuffle seeded from (seed, position id, prefix).
pub fn choose_picks(
    frontier: usize,
    preferred: usize,
    k: usize,
    seed: u64,
    position_id: &str,
    prefix: usize,
) -> Vec<usize> {
    let mut rest: Vec<usize> = (0..frontier).filter(|&i| i != preferred).collect();
    let mut rng = SplitMix(mix(seed ^ fnv(position_id), prefix as u64));
    let take = k.saturating_sub(1).min(rest.len());
    for i in 0..take {
        let j = i + (rng.next() % (rest.len() - i) as u64) as usize;
        rest.swap(i, j);
    }
    let mut picks = vec![preferred];
    picks.extend_from_slice(&rest[..take]);
    picks
}

/// Build probe labels for one micro-batch: run `prefix` real random queries (label-
/// independent), score the whole frontier with the head, then probe the preferred edge plus a
/// diverse fill in isolated forks. Returns the head scores `[b, fmax]` (with graph when `B` is
/// an autodiff backend) and the labelled samples. Evidence and base are frozen.
#[allow(clippy::too_many_arguments)]
pub fn probe_batch<B: Backend>(
    model: &EvidenceBeliefModel<B>,
    data: &V4Data,
    chunk: &[usize],
    prefix: usize,
    k: usize,
    seed: u64,
    device: &B::Device,
) -> anyhow::Result<(Tensor<B, 2>, Vec<ProbeSample>)> {
    let roots = data.roots(chunk)?;
    let opts = RunOptions::new(8)
        .with_freeze(Freeze::BASE_AND_EVIDENCE)
        .with_state();
    let mut s = Session::new(model, &roots, opts, mix(seed, 0xAA ^ prefix as u64), device)?;
    let mut rnd = Selection::Random(0);
    for _ in 0..prefix {
        s.advance(&mut rnd)?;
    }
    let fronts = s.frontiers();
    anyhow::ensure!(
        fronts.iter().all(|f| !f.is_empty()),
        "a probed example has an empty frontier"
    );
    let (scores, _view) = s.utilities(&fronts)?;
    let fmax = scores.dims()[1];
    let sh = host(scores.clone());
    let mut samples = Vec::with_capacity(chunk.len());
    for (e, &i) in chunk.iter().enumerate() {
        let row = &sh[e * fmax..e * fmax + fronts[e].len()];
        let mut preferred = 0usize;
        for (c, &v) in row.iter().enumerate() {
            if v > row[preferred] {
                preferred = c;
            }
        }
        let picks = choose_picks(fronts[e].len(), preferred, k, seed, &data.position(i).id, prefix);
        let before = s.belief_logits(e)?;
        let correct = data.correct(i);
        let ce_before = f64::from(root_ce(&before, &correct));
        let probes = s.probe_edges(e, &fronts[e], &picks)?;
        let u: Vec<f64> = probes
            .iter()
            .map(|p| ce_before - f64::from(root_ce(&p.logits, &correct)))
            .collect();
        samples.push(ProbeSample {
            position: i,
            prefix,
            frontier: fronts[e].len(),
            preferred,
            scores: picks.iter().map(|&p| f64::from(row[p])).collect(),
            picks,
            u,
            ce_before,
        });
    }
    Ok((scores, samples))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UtilityLoss {
    /// Pairwise logistic ranking between higher/lower realised utility.
    Ranking,
    /// Huber-free squared error to realised utility clipped to `[-2, 2]` nats.
    Regression,
}

/// The utility loss for one probed micro-batch (scalar, with graph through `scores`).
pub fn utility_loss<B: Backend>(
    scores: Tensor<B, 2>,
    samples: &[ProbeSample],
    kind: UtilityLoss,
) -> Tensor<B, 1> {
    let device = scores.device();
    let [b, fmax] = scores.dims();
    debug_assert_eq!(b, samples.len());
    let flat = scores.clone().reshape([b * fmax]);
    let zero = scores.sum().mul_scalar(0.0);
    match kind {
        UtilityLoss::Ranking => {
            let (mut ii, mut jj) = (Vec::new(), Vec::new());
            for (e, s) in samples.iter().enumerate() {
                for a in 0..s.picks.len() {
                    for c in 0..s.picks.len() {
                        if s.u[a] - s.u[c] > RANK_MARGIN {
                            ii.push((e * fmax + s.picks[a]) as i32);
                            jj.push((e * fmax + s.picks[c]) as i32);
                        }
                    }
                }
            }
            if ii.is_empty() {
                return zero;
            }
            let n = ii.len();
            let ti = Tensor::<B, 1, Int>::from_data(TensorData::new(ii, [n]), &device);
            let tj = Tensor::<B, 1, Int>::from_data(TensorData::new(jj, [n]), &device);
            let d = flat.clone().select(0, ti) - flat.select(0, tj);
            // softplus(-d), numerically stable
            let sp = d.clone().neg().clamp_min(0.0) + d.abs().neg().exp().add_scalar(1.0).log();
            sp.mean()
        }
        UtilityLoss::Regression => {
            let (mut pi, mut tv) = (Vec::new(), Vec::new());
            for (e, s) in samples.iter().enumerate() {
                for (a, &p) in s.picks.iter().enumerate() {
                    pi.push((e * fmax + p) as i32);
                    tv.push(s.u[a].clamp(-REGRESSION_CLIP, REGRESSION_CLIP) as f32);
                }
            }
            if pi.is_empty() {
                return zero;
            }
            let n = pi.len();
            let ti = Tensor::<B, 1, Int>::from_data(TensorData::new(pi, [n]), &device);
            let tt = Tensor::<B, 1>::from_data(TensorData::new(tv, [n]), &device);
            let diff = flat.select(0, ti) - tt;
            (diff.clone() * diff).mean()
        }
    }
}

/// Aggregate answers D / E / F from labelled probe samples.
#[derive(Debug, Clone, serde::Serialize)]
pub struct UtilityReport {
    pub states: usize,
    pub probes: usize,
    // D: does a single acquired state sometimes help and sometimes harm?
    pub frac_positive: f64,
    pub frac_negative: f64,
    pub frac_zero: f64,
    pub mean_u: f64,
    pub std_u: f64,
    // E: does the head rank realised utility above random?
    pub spearman_mean: f64,
    pub spearman_states: usize,
    pub pairwise_ok: usize,
    pub pairwise_n: usize,
    // F: is the best-predicted edge more useful than a random frontier edge?
    pub mean_u_preferred: f64,
    pub mean_u_random_probed: f64,
    pub frac_preferred_positive: f64,
    pub frac_random_positive: f64,
    pub mean_regret_vs_best_probed: f64,
    /// Per state: `U(preferred) - mean U(other probed)` (for the paired bootstrap).
    #[serde(skip)]
    pub f_delta: Vec<f64>,
    /// Per state Spearman (for the paired bootstrap).
    #[serde(skip)]
    pub spearman_per_state: Vec<f64>,
}

pub fn utility_report(samples: &[ProbeSample]) -> UtilityReport {
    let all_u: Vec<f64> = samples.iter().flat_map(|s| s.u.iter().copied()).collect();
    let n = all_u.len().max(1) as f64;
    let mean_u = all_u.iter().sum::<f64>() / n;
    let std_u = (all_u.iter().map(|u| (u - mean_u).powi(2)).sum::<f64>() / n).sqrt();
    let (mut pos, mut neg, mut zero) = (0usize, 0usize, 0usize);
    for &u in &all_u {
        if u > 1e-9 {
            pos += 1;
        } else if u < -1e-9 {
            neg += 1;
        } else {
            zero += 1;
        }
    }
    let (mut sp, mut ok, mut np) = (Vec::new(), 0usize, 0usize);
    let (mut f_delta, mut pref_u, mut rand_u) = (Vec::new(), Vec::new(), Vec::new());
    let (mut pref_pos, mut rand_pos, mut rand_n) = (0usize, 0usize, 0usize);
    let mut regret = Vec::new();
    for s in samples {
        if let Some(r) = crate::stats::spearman(&s.scores, &s.u) {
            sp.push(r);
        }
        let (o, c) = crate::stats::pairwise_agreement(&s.scores, &s.u, RANK_MARGIN);
        ok += o;
        np += c;
        // picks[0] is the preferred edge by construction.
        let others = &s.u[1..];
        if !others.is_empty() {
            let m = others.iter().sum::<f64>() / others.len() as f64;
            f_delta.push(s.u[0] - m);
            rand_u.push(m);
            rand_pos += others.iter().filter(|&&u| u > 1e-9).count();
            rand_n += others.len();
        }
        pref_u.push(s.u[0]);
        pref_pos += usize::from(s.u[0] > 1e-9);
        regret.push(s.u.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - s.u[0]);
    }
    let avg = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    UtilityReport {
        states: samples.len(),
        probes: all_u.len(),
        frac_positive: pos as f64 / n,
        frac_negative: neg as f64 / n,
        frac_zero: zero as f64 / n,
        mean_u,
        std_u,
        spearman_mean: avg(&sp),
        spearman_states: sp.len(),
        pairwise_ok: ok,
        pairwise_n: np,
        mean_u_preferred: avg(&pref_u),
        mean_u_random_probed: avg(&rand_u),
        frac_preferred_positive: pref_pos as f64 / samples.len().max(1) as f64,
        frac_random_positive: rand_pos as f64 / rand_n.max(1) as f64,
        mean_regret_vs_best_probed: avg(&regret),
        f_delta,
        spearman_per_state: sp,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn micro_batches_never_end_with_a_single_example() {
        let idx: Vec<usize> = (0..33).collect();
        for c in chunks_min2(&idx, 16) {
            assert!(c.len() >= 2);
        }
        let total: usize = chunks_min2(&idx, 16).iter().map(|c| c.len()).sum();
        assert_eq!(total, 33);
    }

    #[test]
    fn probe_picks_are_distinct_deterministic_and_start_with_the_preferred_edge() {
        let a = choose_picks(25, 7, 8, 5101, "pos-1", 2);
        assert_eq!(a, choose_picks(25, 7, 8, 5101, "pos-1", 2));
        assert_eq!(a[0], 7);
        assert_eq!(a.len(), 8);
        let set: std::collections::HashSet<_> = a.iter().collect();
        assert_eq!(set.len(), 8);
        assert!(a.iter().all(|&i| i < 25));
        // A small frontier is probed completely.
        assert_eq!(choose_picks(3, 1, 8, 1, "p", 0).len(), 3);
        // The rule depends on the position id and the prefix.
        assert_ne!(a, choose_picks(25, 7, 8, 5101, "pos-2", 2));
    }
}
