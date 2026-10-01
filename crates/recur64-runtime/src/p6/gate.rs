//! Gate I: the frozen P6 pass rule and its exact three-seed paired estimator.
//!
//! For each KQRvK M3 position `i` and seed `s`: `d[i,s] = 1(AllInfo_s correct on i) -
//! 1(B0_s correct on i)`; `d_i = mean_s d[i,s]`; `Delta = mean_i d_i`. Gate I passes iff
//! `Delta >= 0.20` AND the paired 95% percentile bootstrap CI (20,000 resamples of the
//! positions with replacement, all three seed pairs kept together, fixed seed
//! `0x7A160001`) lies wholly above zero. No other quantity enters pass/fail.

use serde::Serialize;

pub const BOOTSTRAP_SEED: u64 = 0x7A16_0001;
pub const RESAMPLES: usize = 20_000;
pub const THRESHOLD: f64 = 0.20;
/// 95% percentile interval: sorted resample means at these 0-based ranks.
pub const LOWER_RANK: usize = 499; // ceil(0.025 * 20000) - 1
pub const UPPER_RANK: usize = 19_499; // ceil(0.975 * 20000) - 1

/// SplitMix64: the fixed, self-contained generator of the bootstrap.
pub struct SplitMix64(u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform index in `0..n` by multiply-shift.
    pub fn index(&mut self, n: usize) -> usize {
        ((u128::from(self.next_u64()) * n as u128) >> 64) as usize
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Gate1 {
    pub positions: usize,
    pub seeds: usize,
    /// `mean_i d_i`.
    pub delta: f64,
    /// `mean_i d[i,s]` per seed (reported prominently; not part of the pass rule).
    pub per_seed_delta: Vec<f64>,
    pub seed_delta_min: f64,
    pub seed_delta_max: f64,
    pub ci_lower: f64,
    pub ci_upper: f64,
    pub threshold: f64,
    pub resamples: usize,
    pub bootstrap_seed: u64,
    pub pass: bool,
}

/// The pure Gate I computation. `allinfo[s][i]` and `b0[s][i]` are the top-1 correctness
/// (0/1) of seed `s` on gate-cell position `i`.
pub fn gate1(allinfo: &[Vec<f64>], b0: &[Vec<f64>]) -> anyhow::Result<Gate1> {
    anyhow::ensure!(
        allinfo.len() == b0.len() && !allinfo.is_empty(),
        "ALL-INFO and B0 need the same, non-zero number of seeds"
    );
    let n = allinfo[0].len();
    anyhow::ensure!(n > 0, "no gate-cell positions");
    for (a, b) in allinfo.iter().zip(b0) {
        anyhow::ensure!(
            a.len() == n && b.len() == n,
            "every seed must cover the same {n} positions"
        );
        anyhow::ensure!(
            a.iter().chain(b).all(|&v| v == 0.0 || v == 1.0),
            "top-1 correctness must be 0 or 1"
        );
    }
    let seeds = allinfo.len();
    let d: Vec<f64> = (0..n)
        .map(|i| (0..seeds).map(|s| allinfo[s][i] - b0[s][i]).sum::<f64>() / seeds as f64)
        .collect();
    let delta = d.iter().sum::<f64>() / n as f64;
    let per_seed: Vec<f64> = (0..seeds)
        .map(|s| (0..n).map(|i| allinfo[s][i] - b0[s][i]).sum::<f64>() / n as f64)
        .collect();
    let mut rng = SplitMix64::new(BOOTSTRAP_SEED);
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
    Ok(Gate1 {
        positions: n,
        seeds,
        delta,
        seed_delta_min: per_seed.iter().copied().fold(f64::INFINITY, f64::min),
        seed_delta_max: per_seed.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        per_seed_delta: per_seed,
        ci_lower: lo,
        ci_upper: hi,
        threshold: THRESHOLD,
        resamples: RESAMPLES,
        bootstrap_seed: BOOTSTRAP_SEED,
        pass: delta >= THRESHOLD && lo > 0.0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `k` of `n` positions flip from wrong (B0) to correct (ALL-INFO) in every seed.
    fn flips(n: usize, k: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let b0 = vec![vec![0.0; n]; 3];
        let ai: Vec<Vec<f64>> = (0..3)
            .map(|_| (0..n).map(|i| f64::from(u8::from(i < k))).collect())
            .collect();
        (ai, b0)
    }

    #[test]
    fn the_estimator_is_the_mean_of_the_per_seed_paired_deltas() {
        let (ai, b0) = flips(750, 225);
        let g = gate1(&ai, &b0).unwrap();
        assert!((g.delta - 0.30).abs() < 1e-12);
        assert!(g.per_seed_delta.iter().all(|d| (d - 0.30).abs() < 1e-12));
        assert_eq!(g.positions, 750);
        assert!(g.pass && g.ci_lower > 0.0 && g.ci_upper >= g.delta - 0.1);
    }

    #[test]
    fn the_bootstrap_is_deterministic() {
        let (mut ai, b0) = flips(300, 90);
        ai[1][5] = 0.0;
        let (a, b) = (gate1(&ai, &b0).unwrap(), gate1(&ai, &b0).unwrap());
        assert_eq!((a.ci_lower, a.ci_upper), (b.ci_lower, b.ci_upper));
        // The generator itself is fixed (SplitMix64 reference value for seed 0).
        assert_eq!(SplitMix64::new(0).next_u64(), 0xE220_A839_7B1D_CDAF);
    }

    #[test]
    fn the_threshold_boundary_is_inclusive_and_the_ci_must_be_wholly_positive() {
        // Exactly +0.20 with a tight CI (all positions identical effect): passes.
        let n = 750;
        let b0 = vec![vec![0.0; n]; 3];
        let ai: Vec<Vec<f64>> = (0..3)
            .map(|_| (0..n).map(|i| f64::from(u8::from(i % 5 == 0))).collect())
            .collect();
        let g = gate1(&ai, &b0).unwrap();
        assert!((g.delta - 0.20).abs() < 1e-12);
        assert!(
            g.pass,
            "delta exactly at the threshold with CI > 0 passes: {g:?}"
        );
        // Just below the threshold fails even though the CI is above zero.
        let ai: Vec<Vec<f64>> = (0..3)
            .map(|_| {
                (0..n)
                    .map(|i| f64::from(u8::from(i % 5 == 0 && i != 0)))
                    .collect()
            })
            .collect();
        let g = gate1(&ai, &b0).unwrap();
        assert!(g.delta < 0.20 && g.ci_lower > 0.0 && !g.pass);
        // A large mean delta whose CI reaches zero fails: two positions, one flips
        // (a quarter of the resamples contain no flip at all).
        let b0 = vec![vec![0.0; 2]; 3];
        let ai = vec![vec![1.0, 0.0]; 3];
        let g = gate1(&ai, &b0).unwrap();
        assert!(g.delta >= 0.20 && g.ci_lower <= 0.0 && !g.pass, "{g:?}");
    }

    #[test]
    fn a_seed_with_a_negative_delta_is_reported_but_does_not_change_the_rule() {
        let n = 400;
        let b0 = vec![vec![0.0; n]; 3];
        let mut ai: Vec<Vec<f64>> = (0..3)
            .map(|_| (0..n).map(|i| f64::from(u8::from(i % 2 == 0))).collect())
            .collect();
        // Seed 2: ALL-INFO gets worse than B0 on every position it had right.
        let mut b0 = b0;
        b0[2] = vec![1.0; n];
        ai[2] = vec![0.0; n];
        let g = gate1(&ai, &b0).unwrap();
        assert!(g.per_seed_delta[2] < 0.0 && g.seed_delta_min < 0.0);
        assert!((g.delta - (0.5 + 0.5 - 1.0) / 3.0).abs() < 1e-12);
        assert!(!g.pass);
    }

    #[test]
    fn malformed_inputs_are_refused() {
        assert!(gate1(&[], &[]).is_err());
        assert!(gate1(&[vec![1.0]], &[vec![1.0], vec![0.0]]).is_err());
        assert!(gate1(&[vec![1.0, 0.0]], &[vec![1.0]]).is_err());
        assert!(gate1(&[vec![0.5]], &[vec![0.0]]).is_err());
    }
}
