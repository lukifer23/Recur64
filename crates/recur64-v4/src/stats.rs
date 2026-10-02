//! Estimators for the pre-registered mechanism rules: a paired percentile bootstrap over
//! positions (seed pairs kept together, SplitMix64, 20,000 resamples, ranks 499 / 19,499 as in
//! V3.5) plus rank statistics for the utility head.

use crate::util::SplitMix;

pub const RESAMPLES: usize = 20_000;
/// Percentile ranks of the 95% interval for `RESAMPLES = 20_000`.
pub const LO_RANK: usize = 499;
pub const HI_RANK: usize = 19_499;

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct Ci {
    /// Pooled mean over seeds and positions.
    pub mean: f64,
    pub lo: f64,
    pub hi: f64,
    /// Mean per seed.
    pub per_seed: Vec<f64>,
}

impl Ci {
    /// CI wholly above zero and every seed above zero (the V3.5 `Paired.pass` shape).
    pub fn wholly_positive(&self) -> bool {
        self.lo > 0.0 && self.per_seed.iter().all(|&s| s > 0.0)
    }
}

/// `values[s][i]`: a paired quantity for seed `s`, position `i`. Positions are resampled with
/// replacement, the same positions for every seed.
pub fn paired_bootstrap(values: &[Vec<f64>], seed: u64) -> anyhow::Result<Ci> {
    anyhow::ensure!(!values.is_empty(), "no seeds");
    let n = values[0].len();
    anyhow::ensure!(
        n > 0 && values.iter().all(|v| v.len() == n),
        "seed arrays differ in length"
    );
    let per_seed: Vec<f64> = values
        .iter()
        .map(|v| v.iter().sum::<f64>() / n as f64)
        .collect();
    let pooled: Vec<f64> = (0..n)
        .map(|i| values.iter().map(|v| v[i]).sum::<f64>() / values.len() as f64)
        .collect();
    let mean = pooled.iter().sum::<f64>() / n as f64;
    let mut rng = SplitMix(seed);
    let mut stats = Vec::with_capacity(RESAMPLES);
    for _ in 0..RESAMPLES {
        let mut s = 0.0;
        for _ in 0..n {
            s += pooled[(rng.next() % n as u64) as usize];
        }
        stats.push(s / n as f64);
    }
    stats.sort_by(|a, b| a.partial_cmp(b).expect("finite bootstrap statistics"));
    Ok(Ci {
        mean,
        lo: stats[LO_RANK],
        hi: stats[HI_RANK],
        per_seed,
    })
}

/// Average ranks (ties share the mean rank).
fn ranks(x: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..x.len()).collect();
    idx.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).expect("finite"));
    let mut r = vec![0.0; x.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && x[idx[j + 1]] == x[idx[i]] {
            j += 1;
        }
        let avg = (i + j) as f64 / 2.0 + 1.0;
        for &k in &idx[i..=j] {
            r[k] = avg;
        }
        i = j + 1;
    }
    r
}

/// Spearman rank correlation; `None` when either side is constant.
pub fn spearman(x: &[f64], y: &[f64]) -> Option<f64> {
    if x.len() != y.len() || x.len() < 3 {
        return None;
    }
    let (rx, ry) = (ranks(x), ranks(y));
    let n = x.len() as f64;
    let (mx, my) = (rx.iter().sum::<f64>() / n, ry.iter().sum::<f64>() / n);
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for i in 0..x.len() {
        sxy += (rx[i] - mx) * (ry[i] - my);
        sxx += (rx[i] - mx).powi(2);
        syy += (ry[i] - my).powi(2);
    }
    if sxx == 0.0 || syy == 0.0 {
        return None;
    }
    Some(sxy / (sxx * syy).sqrt())
}

/// `(agreeing pairs, counted pairs)` over pairs whose realised utilities differ by more than
/// `margin`; a pair agrees when the prediction orders it the same way.
pub fn pairwise_agreement(pred: &[f64], u: &[f64], margin: f64) -> (usize, usize) {
    let (mut ok, mut n) = (0, 0);
    for i in 0..pred.len() {
        for j in 0..pred.len() {
            if u[i] - u[j] > margin {
                n += 1;
                if pred[i] > pred[j] {
                    ok += 1;
                } else if pred[i] == pred[j] {
                    // a tie is half right
                    ok += 0;
                }
            }
        }
    }
    (ok, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spearman_of_a_monotone_map_is_one_and_of_its_reverse_is_minus_one() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y: Vec<f64> = x.iter().map(|v| v * v).collect();
        let z: Vec<f64> = x.iter().map(|v| -v).collect();
        assert!((spearman(&x, &y).unwrap() - 1.0).abs() < 1e-12);
        assert!((spearman(&x, &z).unwrap() + 1.0).abs() < 1e-12);
        assert!(spearman(&x, &[1.0; 5]).is_none());
    }

    #[test]
    fn the_bootstrap_interval_brackets_the_mean_and_is_deterministic() {
        let v: Vec<Vec<f64>> = vec![(0..200).map(|i| (i % 7) as f64 - 2.0).collect(); 2];
        let a = paired_bootstrap(&v, 7).unwrap();
        let b = paired_bootstrap(&v, 7).unwrap();
        assert_eq!(a, b);
        assert!(a.lo < a.mean && a.mean < a.hi);
        let all_pos = paired_bootstrap(&[vec![1.0; 50]], 1).unwrap();
        assert!(all_pos.wholly_positive());
        let all_neg = paired_bootstrap(&[vec![-1.0; 50]], 1).unwrap();
        assert!(!all_neg.wholly_positive());
    }

    #[test]
    fn pairwise_agreement_counts_only_pairs_with_a_real_utility_gap() {
        let (ok, n) = pairwise_agreement(&[3.0, 2.0, 1.0], &[0.5, 0.2, 0.2], 0.01);
        assert_eq!(n, 2); // (0,1) and (0,2); the 1-2 pair is a tie in u
        assert_eq!(ok, 2);
    }
}
