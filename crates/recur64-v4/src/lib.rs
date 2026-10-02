//! Recur64 V4 `evidence_belief_v4`: learned active hypothesis testing.
//!
//! One root board is encoded once into a hypothesis token per legal root action. An
//! immutable base belief `z0` is read out of those tokens. Every exact one-edge state query
//! becomes an explicit [`content::EvidenceMessage`](ledger::EvidenceMessage) built only from
//! state CONTENT through a bias-free, content-multiplied path, so zero content gives exactly
//! zero evidence. Messages live in a set-like [`ledger::EvidenceLedger`] and are read by
//! the hypotheses through [`belief::BeliefUpdate`], which only ADDS a bounded delta to `z0`.
//! A [`utility::QueryUtilityHead`] predicts the decision value of each frontier edge before
//! its child is seen.
//!
//! See `docs/V4_RESEARCH_PLAN.md` for the laws, contracts and pre-registered rules.

pub mod accounting;
pub mod base;
pub mod belief;
pub mod content;
pub mod data;
pub mod ledger;
pub mod model;
pub mod session;
pub mod stage;
pub mod stats;
pub mod study;
pub mod train;
pub mod utility;

/// Observation squares / features (Observation V1).
pub const SQUARES: usize = 64;
pub const IN_FEATURES: usize = 119;

/// Finite stand-in for -inf in masked logits (same as `active_search_v3`).
pub const MASKED_LOGIT: f32 = -1.0e9;

/// Small normal initialisation of the final scoring layers (same value as the V3 readouts).
pub const POLICY_INIT_STD: f64 = 0.01;

pub(crate) mod util {
    use burn::nn::Linear;
    use burn::prelude::*;

    /// Apply `linear` over the last dimension of a rank-3 tensor as one 2-D matmul.
    pub fn rows<B: Backend>(linear: &Linear<B>, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, s, d] = x.dims();
        let y = linear.forward(x.reshape([b * s, d]));
        let out = y.dims()[1];
        y.reshape([b, s, out])
    }

    /// Row-major relative-displacement bucket for each `(from, to)` square pair, repeated per
    /// head: `[heads, squares * squares]` (same layout as the V3 model's private helper).
    pub fn rel_index_data(heads: usize, squares: usize) -> Vec<i32> {
        assert_eq!(squares, 64, "V4 assumes an 8x8 board");
        let mut v = Vec::with_capacity(heads * squares * squares);
        for _ in 0..heads {
            for i in 0..squares {
                let (ri, fi) = (i / 8, i % 8);
                for j in 0..squares {
                    let (rj, fj) = (j / 8, j % 8);
                    v.push((rj as i32 - ri as i32 + 7) * 15 + (fj as i32 - fi as i32 + 7));
                }
            }
        }
        v
    }

    /// SplitMix64, the deterministic generator used for every frozen V4 rule.
    pub struct SplitMix(pub u64);
    impl SplitMix {
        pub fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
    }
}
