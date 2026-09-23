//! Deterministic synthetic fixtures for Phase 0 systems tests.
//!
//! These are **not chess data**. They are reproducible tensors shaped like the
//! eventual Recur64 observation and legal-candidate policy so the systems proof
//! exercises the real graph.

use burn::prelude::*;
use burn::tensor::TensorData;

use crate::action::CandidateBatch;
use crate::loss::Targets;
use crate::model::CandidateTensors;

/// Small deterministic PRNG (SplitMix64) so fixtures do not depend on the
/// backend RNG or an external `rand` version.
#[derive(Debug, Clone)]
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
    /// Uniform `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / ((1u64 << 24) as f32)
    }
    /// Standard normal via Box-Muller.
    pub fn normal(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-7);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

/// A fixed synthetic batch: boards, candidate lists, policy and WDL targets.
#[derive(Debug, Clone)]
pub struct SynthFixture {
    pub batch: usize,
    pub width: usize,
    pub in_features: usize,
    pub board: Vec<f32>,
    pub lists: Vec<Vec<(u32, u32, u8)>>,
    pub policy_target: Vec<f32>,
    pub wdl_target: Vec<i64>,
}

impl SynthFixture {
    /// Build a fixture with varied legal-list lengths (including empty/terminal
    /// rows and at least one promotion row).
    pub fn new(batch: usize, in_features: usize, seed: u64) -> Self {
        let mut rng = SplitMix64::new(seed);
        let mut lists: Vec<Vec<(u32, u32, u8)>> = Vec::with_capacity(batch);
        for b in 0..batch {
            // Offset by one so row 0 always has a legal candidate: a batch of
            // one must still contain a non-terminal position.
            let len = match (b + 1) % 5 {
                0 => 0, // terminal / no legal moves
                1 => 1,
                2 => 3,
                3 => 7,
                _ => 5,
            };
            let mut row = Vec::with_capacity(len);
            for k in 0..len {
                let from = ((b * 7 + k * 3) % 64) as u32;
                let to = ((b * 11 + k * 5 + 1) % 64) as u32;
                // Force promotion candidates in some rows.
                let promo = if (b + 1) % 5 == 3 && k == 6 {
                    4
                } else if (b + 1) % 5 == 2 && k == 2 {
                    1
                } else {
                    0
                };
                row.push((from, to, promo));
            }
            lists.push(row);
        }
        let width = lists.iter().map(|l| l.len()).max().unwrap_or(0);

        let board: Vec<f32> = (0..batch * 64 * in_features)
            .map(|_| rng.normal())
            .collect();

        // One-hot policy target on the first legal candidate (zero if terminal).
        let mut policy_target = vec![0.0f32; batch * width];
        for (b, l) in lists.iter().enumerate() {
            if !l.is_empty() {
                policy_target[b * width] = 1.0;
            }
        }
        let wdl_target: Vec<i64> = (0..batch).map(|b| (b % 3) as i64).collect();

        Self {
            batch,
            width,
            in_features,
            board,
            lists,
            policy_target,
            wdl_target,
        }
    }

    pub fn candidate_batch(&self) -> CandidateBatch {
        CandidateBatch::from_lists(&self.lists)
    }

    /// A batch-1 fixture for row `i`, keeping this fixture's padded width so
    /// valid entries can be compared position-for-position.
    pub fn row(&self, i: usize) -> SynthFixture {
        let n = 64 * self.in_features;
        SynthFixture {
            batch: 1,
            width: self.width,
            in_features: self.in_features,
            board: self.board[i * n..(i + 1) * n].to_vec(),
            lists: vec![self.lists[i].clone()],
            policy_target: self.policy_target[i * self.width..(i + 1) * self.width].to_vec(),
            wdl_target: vec![self.wdl_target[i]],
        }
    }

    pub fn tensors<B: Backend>(
        &self,
        device: &B::Device,
    ) -> (Tensor<B, 3>, CandidateTensors<B>, Targets<B>) {
        let board = Tensor::<B, 3>::from_data(
            TensorData::new(self.board.clone(), [self.batch, 64, self.in_features]),
            device,
        );
        let cb = self.candidate_batch();
        let cands = CandidateTensors::from_batch(&cb, device);
        let policy_target = Tensor::<B, 2>::from_data(
            TensorData::new(self.policy_target.clone(), [self.batch, self.width]),
            device,
        );
        let wdl_target = Tensor::<B, 1, Int>::from_data(
            TensorData::new(self.wdl_target.clone(), [self.batch]),
            device,
        );
        (
            board,
            cands,
            Targets {
                policy_target,
                wdl_target,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixture_is_deterministic() {
        let a = SynthFixture::new(8, 119, 42);
        let b = SynthFixture::new(8, 119, 42);
        assert_eq!(a.board, b.board);
        assert_eq!(a.policy_target, b.policy_target);
        assert_eq!(a.lists, b.lists);
    }

    #[test]
    fn fixture_has_terminal_and_promotion_rows() {
        let f = SynthFixture::new(10, 119, 7);
        assert!(f.lists.iter().any(|l| l.is_empty()));
        assert!(f.lists.iter().flatten().any(|&(_, _, p)| p == 4));
    }
}
