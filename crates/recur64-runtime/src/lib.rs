//! Recur64 runtime — the systems that close the self-play/learning loop.
//!
//! Phase 2 provides the single GPU inference owner and batcher, independent
//! self-play workers, replay writing/reading/audit, the learner, the run
//! coordinator, and cancellation. No search tree logic lives here (that is
//! `recur64-search`), and no orchestration lives in `recur64-core`.

pub mod evaluator;
pub mod inference;
pub mod replay;
pub mod rng;
pub mod selfplay;

pub use evaluator::SyncEvaluator;
pub use inference::{
    BatchEvaluator, BatchedEvaluator, BatchedModel, InferenceConfig, InferenceMetrics,
    InferenceOwner, MetricsSnapshot,
};
pub use rng::Rng;
pub use selfplay::{
    SelfPlayConfig, SelfPlayGame, SelfPlayPly, TargetEntry, play_game, play_game_from,
    play_game_seeded,
};
