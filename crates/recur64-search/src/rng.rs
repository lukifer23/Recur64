//! A small deterministic RNG (SplitMix64) for reproducible move sampling.

/// SplitMix64 generator.
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
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

    /// Uniform in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    /// Standard normal (Box-Muller; one value per call).
    pub fn normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(f64::MIN_POSITIVE);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// Gamma(shape, 1) by Marsaglia-Tsang, with the `U^(1/shape)` boost for
    /// `shape < 1`.
    pub fn gamma(&mut self, shape: f64) -> f64 {
        assert!(shape > 0.0 && shape.is_finite(), "gamma shape must be > 0");
        if shape < 1.0 {
            let u = self.next_f64().max(f64::MIN_POSITIVE);
            return self.gamma(shape + 1.0) * u.powf(1.0 / shape);
        }
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.normal();
            let v = (1.0 + c * x).powi(3);
            if v <= 0.0 {
                continue;
            }
            let u = self.next_f64().max(f64::MIN_POSITIVE);
            if u.ln() < 0.5 * x * x + d - d * v + d * v.ln() {
                return d * v;
            }
        }
    }

    /// Symmetric Dirichlet(alpha) sample of length `n` (sums to 1).
    pub fn dirichlet(&mut self, alpha: f64, n: usize) -> Vec<f32> {
        let g: Vec<f64> = (0..n).map(|_| self.gamma(alpha)).collect();
        let sum: f64 = g.iter().sum();
        if sum > 0.0 {
            g.iter().map(|x| (x / sum) as f32).collect()
        } else {
            vec![1.0 / n.max(1) as f32; n]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_from_seed() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn dirichlet_is_normalized_deterministic_and_unbiased() {
        let mut a = Rng::new(3);
        let mut b = Rng::new(3);
        assert_eq!(a.dirichlet(0.3, 20), b.dirichlet(0.3, 20));
        let mut r = Rng::new(11);
        let n = 20;
        let mut mean = vec![0.0f64; n];
        let draws = 4000;
        for _ in 0..draws {
            let d = r.dirichlet(0.3, n);
            assert!((d.iter().sum::<f32>() - 1.0).abs() < 1e-4);
            assert!(d.iter().all(|x| *x >= 0.0 && x.is_finite()));
            for (m, x) in mean.iter_mut().zip(&d) {
                *m += *x as f64 / draws as f64;
            }
        }
        // Symmetric Dirichlet: every component has mean 1/n.
        assert!(
            mean.iter().all(|m| (m - 1.0 / n as f64).abs() < 0.01),
            "{mean:?}"
        );
    }

    #[test]
    fn gamma_mean_matches_shape() {
        let mut r = Rng::new(5);
        for shape in [0.3f64, 1.0, 2.5] {
            let m: f64 = (0..20000).map(|_| r.gamma(shape)).sum::<f64>() / 20000.0;
            assert!(
                (m - shape).abs() < 0.05 * shape.max(1.0),
                "shape {shape} mean {m}"
            );
        }
    }

    #[test]
    fn f64_in_unit_interval() {
        let mut r = Rng::new(7);
        for _ in 0..1000 {
            let v = r.next_f64();
            assert!((0.0..1.0).contains(&v));
        }
    }
}
