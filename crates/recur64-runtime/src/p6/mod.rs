//! P6: the ALL-INFO information-sufficiency control (`docs/V3_P6_PLAN.md`).
//!
//! * `recipe`: the frozen training recipe and its identity;
//! * `data`: verified TRAIN/TUNE targets (no traces) and exhaustive depth-2 tree building;
//! * `train`: the resumable ALL-INFO trainer (policy cross-entropy only);
//! * `eval`: TUNE evaluation with per-position results;
//! * `gate`: the frozen Gate I rule and its paired bootstrap;
//! * `census`: the TRAIN-side depth-2 state census.
//!
//! HOLDOUT_C is never loadable here (`load_working_split` refuses it).

pub mod census;
pub mod data;
pub mod eval;
pub mod gate;
pub mod recipe;
pub mod train;

#[cfg(test)]
mod tests;
