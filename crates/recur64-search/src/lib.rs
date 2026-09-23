//! Recur64 search — the transparent PUCT control search and game play.
//!
//! CPU-only and dependent only on `recur64-core`. Positions are evaluated
//! through the narrow [`evaluator::Evaluator`] trait, so search never depends on
//! Burn or on `recur64-model`. Phase 2 implements PUCT only; Gumbel is deferred.

pub mod evaluator;
pub mod game_tree;
pub mod play;
pub mod puct;
pub mod rng;

pub use evaluator::{
    EvalError, EvalRequest, EvalResult, Evaluator, FixedEvaluator, ScriptedEvaluator, value_to_wdl,
};
pub use game_tree::ChessGame;
pub use play::{
    SelfPlayConfig, SelfPlayGame, SelfPlayPly, TargetEntry, play_game, play_game_from,
    play_game_seeded,
};
pub use puct::{PuctConfig, PuctGame, RootEdge, SearchResult, search};
pub use rng::Rng;
