//! Recur64 search — the transparent PUCT control search.
//!
//! Search is CPU-only and depends only on `recur64-core`. It evaluates positions
//! through the narrow [`evaluator::Evaluator`] trait, so it never depends on
//! Burn or on `recur64-model`. Phase 2 implements PUCT only; Gumbel is deferred.

pub mod evaluator;
pub mod game_tree;
pub mod puct;

pub use evaluator::{
    EvalError, EvalRequest, EvalResult, Evaluator, FixedEvaluator, ScriptedEvaluator, value_to_wdl,
};
pub use game_tree::ChessGame;
pub use puct::{PuctConfig, PuctGame, RootEdge, SearchResult, search};
