//! Recur64 V420 `recurrent_depth_v420`: does one shared board-transformer block applied K times
//! improve exactly-labelled endgame accuracy, including at K beyond the training depth?
//!
//! Plan and frozen gates: `docs/V420_RESEARCH_PLAN.md`. This crate reuses only PUBLIC pieces of
//! `recur64-model` (`Block`, `CandidateInputs`, `adamw`) and the V4 TRAIN data and statistics, and
//! changes no V3/V3.5/V4 code. It refuses every sealed set through `V4Data::load`.

pub mod diag;
pub mod gates;
pub mod model;
pub mod train;

/// Finite stand-in for -inf in masked logits.
pub const MASKED_LOGIT: f32 = -1.0e9;
