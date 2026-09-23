//! Recur64 runtime — the systems that close the self-play/learning loop.
//!
//! Phase 2 provides the single GPU inference owner and batcher, replay
//! writing/reading/audit, the learner, the bounded run coordinator, and
//! cancellation. Search tree logic lives in `recur64-search`; chess contracts
//! live in `recur64-core`.

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod cancel;
pub mod config;
pub mod coordinator;
pub mod evaluator;
pub mod inference;
pub mod learner;
pub mod model_io;
pub mod replay;
pub mod run_dir;
pub mod sweep;

pub use cancel::CancelToken;
pub use config::RunConfig;
pub use coordinator::{RunReport, collect_only, game_uci_moves, run, write_report};
pub use evaluator::SyncEvaluator;
pub use inference::{
    BatchEvaluator, BatchedEvaluator, BatchedModel, InferenceConfig, InferenceMetrics,
    InferenceOwner, MetricsSnapshot,
};
pub use learner::{LearnerConfig, TrainReport, TrainingExample, build_examples, train_from_games};
pub use run_dir::{LineageRecord, RunDir, RunMetadata, RunStatus, read_metadata, write_metadata};
pub use sweep::{SweepCellResult, SweepCellSpec};

// Re-exported from `recur64-search` for convenience.
pub use recur64_search::{
    Rng, SelfPlayConfig, SelfPlayGame, SelfPlayPly, TargetEntry, play_game, play_game_from,
    play_game_seeded,
};
