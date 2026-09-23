//! Recur64 Phase 0 probe model crate.
//!
//! This crate intentionally contains only what Phase 0 needs: the model-shaped
//! graph (square-token transformer, sparse legal-candidate policy, pooled WDL),
//! recurrence proof, optimizer/overfit proof, checkpoint/resume proof, precision
//! gate, and the benchmark harness. No chess rules, search, replay, or runtime.

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Pinned backend version. Kept in sync with `docs/DECISIONS.md` and the
/// `Cargo.lock` entry for `burn`. See `docs/ARCHITECTURE.md`.
pub const BURN_VERSION: &str = "0.21.0";

pub mod action;
pub mod checkpoint;
pub mod config;
pub mod fixture;
pub mod loss;
pub mod model;
pub mod precision;
pub mod train;

pub use config::{DeviceKind, ModelConfig, Precision, ProbeConfig};
