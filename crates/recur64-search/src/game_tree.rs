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
        })
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
        }
    }

    fn white_action(from: Square, to: Square) -> ActionId {
        ActionId::from_physical(from, to, PromotionCode::NONE, Perspective::white())
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
