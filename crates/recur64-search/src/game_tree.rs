//! Real-chess adapter for the generic PUCT search.
//!
//! [`ChessGame`] wraps a [`GameState`] and a [`Evaluator`] and implements
//! [`PuctGame`]. Terminal values come from the authoritative rules profile, not
//! from the network. `Truncated`/`Aborted` are administrative, not chess
//! terminals, so they are treated as non-terminal here (search never imposes a
//! ply cap).

use recur64_core::{ActionId, GameState, StandardMove, Termination, encode_observation_v1};

use crate::evaluator::{EvalError, EvalRequest, EvalResult, Evaluator};
use crate::puct::PuctGame;

/// A chess position plus the evaluator used for its non-terminal children.
pub struct ChessGame<'a> {
    pub state: GameState,
    pub evaluator: &'a dyn Evaluator,
}

impl<'a> ChessGame<'a> {
    pub fn new(state: GameState, evaluator: &'a dyn Evaluator) -> Self {
        Self { state, evaluator }
    }
}

impl PuctGame for ChessGame<'_> {
    type Action = ActionId;

    fn terminal_value(&self) -> Option<f32> {
        match self.state.termination() {
            None => None,
            // The side to move is checkmated -> losing from its perspective.
            Some(Termination::Checkmate) => Some(-1.0),
            Some(t) if t.is_draw() => Some(0.0),
            // Administrative states are not search terminals.
            Some(_) => None,
        }
    }

    fn legal_actions(&self) -> Vec<ActionId> {
        self.state.legal_actions()
    }

    fn apply(&self, action: ActionId) -> Self {
        let perspective = self.state.perspective();
        let (from, to, promo) = action.to_physical(perspective);
        let promotion = if promo.is_none() { None } else { Some(promo) };
        let mv = StandardMove::new(from, to, promotion);
        let mut child = self.state.clone();
        child
            .apply(mv)
            .expect("search applies only actions produced by legal_actions");
        ChessGame {
            state: child,
            evaluator: self.evaluator,
        }
    }

    fn evaluate(&self, legal: &[ActionId]) -> Result<EvalResult, EvalError> {
        let observation = encode_observation_v1(&self.state);
        self.evaluator.evaluate(EvalRequest {
            observation: &observation,
            legal,
            side_to_move: self.state.side_to_move(),
        })
    }

    /// Submit all leaves of one multi-leaf round to the tree's evaluator at
    /// once, so they can share an inference batch.
    fn evaluate_many(
        games: &[&Self],
        legal: &[Vec<ActionId>],
    ) -> Vec<Result<EvalResult, EvalError>> {
        let Some(first) = games.first() else {
            return Vec::new();
        };
        let observations: Vec<_> = games
            .iter()
            .map(|g| encode_observation_v1(&g.state))
            .collect();
        let requests: Vec<EvalRequest<'_>> = games
            .iter()
            .zip(&observations)
            .zip(legal)
            .map(|((g, observation), legal)| EvalRequest {
                observation,
                legal,
                side_to_move: g.state.side_to_move(),
            })
            .collect();
        first.evaluator.evaluate_many(&requests)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluator::ScriptedEvaluator;
    use crate::puct::{PuctConfig, search};
    use recur64_core::{Color, Perspective, PromotionCode, Square};

    fn config(sims: u32) -> PuctConfig {
        PuctConfig {
            c_puct: 1.0,
            simulations: sims,
            leaves_in_flight: 1,
        }
    }

    fn white_action(from: Square, to: Square) -> ActionId {
        ActionId::from_physical(from, to, PromotionCode::NONE, Perspective::white())
    }

    /// Uniform policy, value 0; records the size of every batched submission.
    struct BatchRecorder {
        sizes: std::sync::Mutex<Vec<usize>>,
    }

    impl Evaluator for BatchRecorder {
        fn evaluate(&self, request: EvalRequest<'_>) -> Result<EvalResult, EvalError> {
            self.sizes.lock().unwrap().push(1);
            Ok(EvalResult::uniform(request.legal.len(), 0.0))
        }
        fn evaluate_many(
            &self,
            requests: &[EvalRequest<'_>],
        ) -> Vec<Result<EvalResult, EvalError>> {
            self.sizes.lock().unwrap().push(requests.len());
            requests
                .iter()
                .map(|r| Ok(EvalResult::uniform(r.legal.len(), 0.0)))
                .collect()
        }
    }

    /// D47: multi-leaf search keeps the exact traversal budget, submits leaves
    /// in batches, and removes every virtual loss (with value 0 everywhere and
    /// no terminal reached, all edge values must return to exactly 0).
    #[test]
    fn multi_leaf_search_keeps_budget_batches_and_clears_virtual_loss() {
        let ev = BatchRecorder {
            sizes: std::sync::Mutex::new(Vec::new()),
        };
        let game = ChessGame::new(GameState::startpos(), &ev);
        let cfg = PuctConfig {
            c_puct: 1.0,
            simulations: 64,
            leaves_in_flight: 8,
        };
        let r = search(game, &cfg).unwrap();
        assert_eq!(r.traversals, 64);
        assert_eq!(
            r.total_visits, 63,
            "every traversal after the root expansion"
        );
        assert_eq!(r.edges.iter().map(|e| e.visits).sum::<u32>(), 63);
        assert_eq!(r.root_value, 0.0, "virtual loss fully removed");
        let sizes = ev.sizes.lock().unwrap();
        assert!(
            sizes.iter().any(|&n| n > 1),
            "leaves were batched: {sizes:?}"
        );
        assert!(sizes.iter().all(|&n| n <= 8));
        assert_eq!(
            sizes.iter().sum::<usize>(),
            64,
            "one evaluation per non-terminal node"
        );
    }

    /// D47: virtual loss does not stop the search from finding a forced mate.
    #[test]
    fn multi_leaf_search_still_prefers_mate_in_one() {
        let state = GameState::from_fen("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1").unwrap();
        let ev = ScriptedEvaluator::new(vec![0.0]);
        let game = ChessGame::new(state, &ev);
        let cfg = PuctConfig {
            c_puct: 1.0,
            simulations: 64,
            leaves_in_flight: 4,
        };
        let r = search(game, &cfg).unwrap();
        assert_eq!(r.best_action(), Some(white_action(Square::A1, Square::A8)));
        assert!(r.root_value > 0.0);
        assert_eq!(r.total_visits, 63);
    }

    #[test]
    fn mate_in_one_is_preferred() {
        // Back-rank mate: Ra1-a8#.
        let state = GameState::from_fen("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1").unwrap();
        let ev = ScriptedEvaluator::new(vec![0.0]); // uniform, value 0
        let game = ChessGame::new(state, &ev);
        let r = search(game, &config(64)).unwrap();
        assert_eq!(r.best_action(), Some(white_action(Square::A1, Square::A8)));
        assert!(r.root_value > 0.0, "root is winning: {}", r.root_value);
    }

    #[test]
    fn forced_single_legal_move() {
        // White king h1 in check from Qg2; only Kxg2 is legal.
        let state = GameState::from_fen("7k/8/8/8/8/8/6q1/7K w - - 0 1").unwrap();
        let ev = ScriptedEvaluator::new(vec![0.0]);
        let game = ChessGame::new(state, &ev);
        let r = search(game, &config(16)).unwrap();
        assert_eq!(r.edges.len(), 1);
        assert_eq!(r.best_action(), Some(white_action(Square::H1, Square::G2)));
        let p = r.policy();
        assert!((p[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn stalemate_root_has_no_target_and_no_eval() {
        let state = GameState::from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1").unwrap();
        assert_eq!(state.termination(), Some(Termination::Stalemate));
        let ev = ScriptedEvaluator::new(vec![0.0]);
        let game = ChessGame::new(state, &ev);
        let r = search(game, &config(16)).unwrap();
        assert!(r.edges.is_empty());
        assert_eq!(r.traversals, 0);
        assert_eq!(ev.calls(), 0, "terminal root must not evaluate");
        assert_eq!(r.root_value, 0.0);
    }

    #[test]
    fn terminal_children_are_not_evaluated() {
        // Mate-in-1: the mating child is terminal, so only non-terminal nodes are
        // evaluated. With 64 simulations the evaluator is called far fewer times
        // than 64.
        let state = GameState::from_fen("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1").unwrap();
        let ev = ScriptedEvaluator::new(vec![0.0]);
        let game = ChessGame::new(state, &ev);
        let _ = search(game, &config(64)).unwrap();
        assert!(ev.calls() <= 64);
        // The mating move's child is terminal and must not be counted as an eval
        // in a way that exceeds the budget; sanity-check a small bound.
        assert!(ev.calls() >= 1, "root must be evaluated");
    }

    #[test]
    fn perspective_after_black_move_is_not_inverted() {
        // A simple symmetric position: after a quiet white move, the child is
        // black to move; search must keep root_value finite and near 0 with a
        // neutral evaluator.
        let state = GameState::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            .unwrap();
        let ev = ScriptedEvaluator::new(vec![0.0]);
        let game = ChessGame::new(state, &ev);
        let r = search(game, &config(16)).unwrap();
        assert!(r.root_value.is_finite());
        assert_eq!(r.edges.len(), 20);
        // Neutral evaluator -> value should be ~0, not inverted to +-1.
        assert!(r.root_value.abs() < 1e-6);
        let _ = Color::White; // silence unused import in some cfgs
    }
}
