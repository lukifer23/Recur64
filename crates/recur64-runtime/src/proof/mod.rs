//! Exact proof curriculum: `ProofTargetsV1`, the exact mate solver, the
//! generator and its independent audit. No network, no PUCT, no engine.

pub mod audit;
pub mod compare;
pub mod custody;
pub mod generator;
pub mod mate;
pub mod sampler;
pub mod targets;
pub mod trace;
pub mod trace_audit;
pub mod trace_store;
pub mod trace_teacher;
pub mod train;

#[cfg(test)]
mod trace_tests;
