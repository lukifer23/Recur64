//! Recur64 runtime — the systems that close the self-play/learning loop.
//!
//! Phase 2 provides the single GPU inference owner and batcher, independent
//! self-play workers, replay writing/reading/audit, the learner, the run
//! coordinator, and cancellation. No search tree logic lives here (that is
//! `recur64-search`), and no orchestration lives in `recur64-core`.

pub mod evaluator;

pub use evaluator::SyncEvaluator;
