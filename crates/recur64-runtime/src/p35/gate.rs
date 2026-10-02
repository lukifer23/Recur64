//! V3.5 pre-registered estimators: Gate II, Gate III and Content-Use.
//!
//! All three are the same three-seed paired estimator as Gate I (`p6::gate`): for each
//! TUNE KQRvK M3 position `i` and seed `s` a paired value `v[s][i]` is formed; `d_i =
//! mean_s v[s][i]`; the point estimate is `mean_i d_i`; the 95% percentile bootstrap CI
//! resamples positions with replacement (20,000 resamples) keeping every seed's pair for a
//! position together, using the fixed generator `SplitMix64` and a per-gate fixed seed.
//!
//! * **Gate II**: `v = top1(ACTIVE_B8) - top1(B0)` (same checkpoint). PASS iff the pooled
//!   delta >= +0.10, CI lower > 0 and every seed's delta > 0.
//! * **Gate III**: `v = top1(ACTIVE_B8) - top1(FIXED_B8)` (`fixed_bfs_actionid_v1`). PASS
//!   iff delta >= +0.05, CI lower > 0 and every seed's delta > 0.
//! * **Content-Use**: `v = CE(ablated) - CE(normal)` on the same ACTIVE B8 query path
//!   (`query_content_ablation_v1`). PASS iff mean > 0, CI lower > 0 and every seed's mean
//!   > 0. No magnitude threshold.

use serde::Serialize;

use crate::p6::gate::{LOWER_RANK, RESAMPLES, SplitMix64, UPPER_RANK};

pub const GATE2_SEED: u64 = 0x7A35_0002;
pub const GATE3_SEED: u64 = 0x7A35_0003;
pub const CONTENT_SEED: u64 = 0x7A35_0004;
pub const GATE2_THRESHOLD: f64 = 0.10;
pub const GATE3_THRESHOLD: f64 = 0.05;
/// Content-Use has no magnitude threshold: strictly positive is the rule.
pub const CONTENT_THRESHOLD: f64 = 0.0;

#[derive(Debug, Clone, Serialize)]
pub struct Paired {
    pub positions: usize,
    pub seeds: usize,
    pub delta: f64,
    pub per_seed_delta: Vec<f64>,
    pub ci_lower: f64,
    pub ci_upper: f64,
    pub threshold: f64,
    /// Whether the point estimate must reach the threshold inclusively (Gates II/III) or
    /// exceed it strictly (Content-Use, threshold 0).
    pub inclusive: bool,
    pub resamples: usize,
    pub bootstrap_seed: u64,
    pub every_seed_positive: bool,
    pub pass: bool,
}

/// The pure paired estimator. `values[s][i]` is the paired value of seed `s` on position `i`.
pub fn paired(
    values: &[Vec<f64>],
    threshold: f64,
    inclusive: bool,
    bootstrap_seed: u64,
) -> anyhow::Result<Paired> {
    anyhow::ensure!(!values.is_empty(), "no seeds");
    let n = values[0].len();
    anyhow::ensure!(n > 0, "no positions");
    for v in values {
        anyhow::ensure!(v.len() == n, "every seed must cover the same {n} positions");
        anyhow::ensure!(v.iter().all(|x| x.is_finite()), "non-finite paired value");
    }
    let seeds = values.len();
    let d: Vec<f64> = (0..n)
        .map(|i| values.iter().map(|v| v[i]).sum::<f64>() / seeds as f64)
        .collect();
    let delta = d.iter().sum::<f64>() / n as f64;
    let per_seed: Vec<f64> = values
        .iter()
        .map(|v| v.iter().sum::<f64>() / n as f64)
        .collect();
    let mut rng = SplitMix64::new(bootstrap_seed);
    let mut means = Vec::with_capacity(RESAMPLES);
    for _ in 0..RESAMPLES {
        let mut sum = 0.0;
        for _ in 0..n {
            sum += d[rng.index(n)];
        }
        means.push(sum / n as f64);
    }
    means.sort_by(|a, b| a.partial_cmp(b).expect("finite means"));
    let (lo, hi) = (means[LOWER_RANK], means[UPPER_RANK]);
    let every_seed_positive = per_seed.iter().all(|&x| x > 0.0);
    let reaches = if inclusive {
        delta >= threshold
    } else {
        delta > threshold
    };
    Ok(Paired {
        positions: n,
        seeds,
        delta,
        per_seed_delta: per_seed,
        ci_lower: lo,
        ci_upper: hi,
        threshold,
        inclusive,
        resamples: RESAMPLES,
        bootstrap_seed,
        every_seed_positive,
        pass: reaches && lo > 0.0 && every_seed_positive,
    })
}

/// Gate II from per-position top-1 correctness (0/1).
pub fn gate2(active_b8: &[Vec<f64>], b0: &[Vec<f64>]) -> anyhow::Result<Paired> {
    paired(&diff01(active_b8, b0)?, GATE2_THRESHOLD, true, GATE2_SEED)
}

/// Gate III from per-position top-1 correctness (0/1).
pub fn gate3(active_b8: &[Vec<f64>], fixed_b8: &[Vec<f64>]) -> anyhow::Result<Paired> {
    paired(
        &diff01(active_b8, fixed_b8)?,
        GATE3_THRESHOLD,
        true,
        GATE3_SEED,
    )
}

/// Content-Use from per-position cross-entropies of the ablated and normal replays.
pub fn content_use(ce_ablated: &[Vec<f64>], ce_normal: &[Vec<f64>]) -> anyhow::Result<Paired> {
    anyhow::ensure!(
        ce_ablated.len() == ce_normal.len(),
        "ablated and normal runs need the same seeds"
    );
    let v: Vec<Vec<f64>> = ce_ablated
        .iter()
        .zip(ce_normal)
        .map(|(a, n)| {
            anyhow::ensure!(
                a.len() == n.len(),
                "ablated and normal need the same positions"
            );
            Ok(a.iter().zip(n).map(|(x, y)| x - y).collect())
        })
        .collect::<anyhow::Result<_>>()?;
    paired(&v, CONTENT_THRESHOLD, false, CONTENT_SEED)
}

fn diff01(a: &[Vec<f64>], b: &[Vec<f64>]) -> anyhow::Result<Vec<Vec<f64>>> {
    anyhow::ensure!(a.len() == b.len(), "the two arms need the same seeds");
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            anyhow::ensure!(x.len() == y.len(), "the two arms need the same positions");
            anyhow::ensure!(
                x.iter().chain(y).all(|&v| v == 0.0 || v == 1.0),
                "top-1 correctness must be 0 or 1"
            );
            Ok(x.iter().zip(y).map(|(p, q)| p - q).collect())
        })
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Outcome {
    FullGo,
    PartialCompute,
    PartialContent,
    PathComputeEffectWithoutInformationUse,
    /// Content, II and III pass but Gate VI (stability) failed: not a GO.
    StabilityFailure,
    NoGo,
}

/// The pre-registered classification. Gate I is historical (PASS) and not an input.
pub fn classify(content: bool, g2: bool, g3: bool, g6: bool) -> Outcome {
    if content && g2 && g3 && g6 {
        Outcome::FullGo
    } else if content && g2 && g3 && !g6 {
        Outcome::StabilityFailure
    } else if !content && (g2 || g3) {
        Outcome::PathComputeEffectWithoutInformationUse
    } else if g2 && !g3 {
        Outcome::PartialCompute
    } else if content && !g2 {
        Outcome::PartialContent
    } else {
        Outcome::NoGo
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flips(n: usize, k: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let b0 = vec![vec![0.0; n]; 3];
        let a: Vec<Vec<f64>> = (0..3)
            .map(|_| (0..n).map(|i| f64::from(u8::from(i < k))).collect())
            .collect();
        (a, b0)
    }

    #[test]
    fn gate2_threshold_is_inclusive_and_needs_every_seed_positive() {
        let (a, b0) = flips(1000, 100);
        let g = gate2(&a, &b0).unwrap();
        assert!((g.delta - 0.10).abs() < 1e-12 && g.pass, "{g:?}");
        let (a, b0) = flips(1000, 99);
        assert!(!gate2(&a, &b0).unwrap().pass);
        // Pooled delta well above the threshold but one seed non-positive: FAIL.
        let (mut a, b0) = flips(1000, 300);
        a[2] = vec![0.0; 1000];
        let g = gate2(&a, &b0).unwrap();
        assert!(g.delta >= 0.10 && g.ci_lower > 0.0 && !g.every_seed_positive && !g.pass);
    }

    #[test]
    fn gate3_uses_its_own_threshold_and_seed() {
        let (a, f) = flips(1000, 50);
        let g = gate3(&a, &f).unwrap();
        assert!(g.pass && g.bootstrap_seed == GATE3_SEED && g.threshold == 0.05);
        assert_ne!(GATE2_SEED, GATE3_SEED);
        assert_ne!(GATE3_SEED, CONTENT_SEED);
    }

    #[test]
    fn content_use_is_strictly_positive_with_a_positive_ci() {
        let normal = vec![vec![1.0; 500]; 3];
        let ablated: Vec<Vec<f64>> = (0..3).map(|_| vec![1.01; 500]).collect();
        let g = content_use(&ablated, &normal).unwrap();
        assert!(g.pass && g.delta > 0.0 && g.threshold == 0.0 && !g.inclusive);
        // Zero effect: delta == 0 is not > 0.
        assert!(!content_use(&normal, &normal).unwrap().pass);
        // One seed negative: FAIL even if pooled positive.
        let mut abl = ablated.clone();
        abl[1] = vec![0.9; 500];
        let g = content_use(&abl, &normal).unwrap();
        assert!(!g.every_seed_positive && !g.pass);
    }

    #[test]
    fn the_bootstrap_is_deterministic_and_inputs_are_validated() {
        let (a, b0) = flips(200, 40);
        let (x, y) = (gate2(&a, &b0).unwrap(), gate2(&a, &b0).unwrap());
        assert_eq!((x.ci_lower, x.ci_upper), (y.ci_lower, y.ci_upper));
        assert!(gate2(&[], &[]).is_err());
        assert!(gate2(&[vec![0.5]], &[vec![0.0]]).is_err());
        assert!(gate2(&[vec![1.0, 0.0]], &[vec![1.0]]).is_err());
        assert!(content_use(&[vec![f64::NAN]], &[vec![1.0]]).is_err());
    }

    #[test]
    fn the_outcome_classification_follows_the_preregistration() {
        use Outcome::*;
        assert_eq!(classify(true, true, true, true), FullGo);
        assert_eq!(classify(true, true, false, true), PartialCompute);
        assert_eq!(classify(true, false, false, true), PartialContent);
        assert_eq!(
            classify(false, true, false, true),
            PathComputeEffectWithoutInformationUse
        );
        assert_eq!(
            classify(false, false, true, true),
            PathComputeEffectWithoutInformationUse
        );
        assert_eq!(classify(false, false, false, true), NoGo);
        assert_eq!(classify(true, true, true, false), StabilityFailure);
    }
}
