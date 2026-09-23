//! Authoritative game state, history, and the single transition path.
//!
//! `GameState::apply` is the only way to advance a game. The environment keeps
//! the full position history needed to adjudicate repetition and the halfmove
//! rules; the network observation only sees a bounded window.

use cozy_chess::Board;

use crate::action::{ActionId, PromotionCode};
use crate::error::CoreError;
use crate::rules::{Outcome, Termination, classify};
use crate::square::{Color, Perspective};
use crate::uci::StandardMove;

/// A complete, authoritative chess game state.
#[derive(Debug, Clone)]
pub struct GameState {
    board: Board,
    /// All positions, oldest first; `history.last()` is the current position.
    history: Vec<Board>,
    /// Moves played, oldest first.
    moves: Vec<StandardMove>,
    /// Optional administrative ply cap (Truncated, not a draw).
    max_plies: Option<u32>,
    repetition_count: u32,
    termination: Option<Termination>,
}

impl GameState {
    /// Start position.
    pub fn startpos() -> Self {
        Self::from_board(Board::default(), None)
    }

    /// Parse a FEN (regular or Shredder). FEN is an interchange/debug format.
    pub fn from_fen(fen: &str) -> Result<Self, CoreError> {
        let board: Board = fen
            .parse()
            .map_err(|e| CoreError::Fen(format!("{fen:?}: {e}")))?;
        Ok(Self::from_board(board, None))
    }

    /// Start position with an administrative ply cap.
    pub fn with_max_plies(mut self, max: u32) -> Self {
        self.max_plies = Some(max);
        self.recompute();
        self
    }

    fn from_board(board: Board, max_plies: Option<u32>) -> Self {
        let mut state = Self {
            history: vec![board.clone()],
            board,
            moves: Vec::new(),
            max_plies,
            repetition_count: 1,
            termination: None,
        };
        state.recompute();
        state
    }

    fn recompute(&mut self) {
        let count = self
            .history
            .iter()
            .filter(|past| self.board.same_position(past))
            .count() as u32;
        self.repetition_count = count.max(1);
        self.termination = classify(
            &self.board,
            self.repetition_count,
            self.moves.len() as u32,
            self.max_plies,
        );
    }

    pub fn board(&self) -> &Board {
        &self.board
    }

    pub fn side_to_move(&self) -> Color {
        self.board.side_to_move()
    }

    pub fn perspective(&self) -> Perspective {
        Perspective::of(self.board.side_to_move())
    }

    pub fn ply(&self) -> u32 {
        self.moves.len() as u32
    }

    /// Full authoritative position history, oldest first.
    pub fn history(&self) -> &[Board] {
        &self.history
    }

    pub fn moves(&self) -> &[StandardMove] {
        &self.moves
    }

    pub fn repetition_count(&self) -> u32 {
        self.repetition_count
    }

    pub fn termination(&self) -> Option<Termination> {
        self.termination
    }

    pub fn outcome(&self) -> Option<Outcome> {
        self.termination
            .and_then(|t| t.outcome(self.board.side_to_move()))
    }

    pub fn is_terminal(&self) -> bool {
        self.termination.is_some()
    }

    pub fn to_fen(&self) -> String {
        self.board.to_string()
    }

    /// All legal moves in the standard/UCI convention.
    pub fn legal_standard_moves(&self) -> Vec<StandardMove> {
        let board = &self.board;
        let mut out = Vec::with_capacity(64);
        board.generate_moves(|moves| {
            for mv in moves {
                out.push(
                    StandardMove::from_cozy(board, mv).expect("cozy move must be well-formed"),
                );
            }
            false
        });
        out
    }

    /// Canonical legal `ActionId`s, sorted ascending for determinism.
    ///
    /// Empty for terminal positions.
    pub fn legal_actions(&self) -> Vec<ActionId> {
        let p = self.perspective();
        let mut ids: Vec<ActionId> = self
            .legal_standard_moves()
            .iter()
            .map(|mv| {
                ActionId::from_physical(
                    mv.from,
                    mv.to,
                    mv.promotion.unwrap_or(PromotionCode::NONE),
                    p,
                )
            })
            .collect();
        ids.sort_unstable();
        ids
    }

    /// Canonical legal `ActionId`s in a fixed-capacity list (no heap
    /// allocation). Sorted ascending. Empty for terminal positions.
    pub fn legal_action_list(&self) -> crate::action::ActionList {
        let p = self.perspective();
        let mut list = crate::action::ActionList::new();
        let board = &self.board;
        board.generate_moves(|moves| {
            for mv in moves {
                let std = StandardMove::from_cozy(board, mv).expect("cozy move well-formed");
                list.push(ActionId::from_physical(
                    std.from,
                    std.to,
                    std.promotion.unwrap_or(PromotionCode::NONE),
                    p,
                ));
            }
            false
        });
        list.sort();
        list
    }

    /// Apply a standard move. Errors visibly if the game is terminal or the
    /// move is illegal.
    pub fn apply(&mut self, mv: StandardMove) -> Result<(), CoreError> {
        if self.is_terminal() {
            return Err(CoreError::IllegalMove(format!(
                "game is already terminal ({})",
                self.termination.map(|t| t.label()).unwrap_or("?")
            )));
        }
        let cozy = mv.to_cozy(&self.board)?;
        self.board
            .try_play(cozy)
            .map_err(|_| CoreError::IllegalMove(mv.to_uci()))?;
        self.moves.push(mv);
        self.history.push(self.board.clone());
        self.recompute();
        Ok(())
    }

    /// Parse and apply a UCI move.
    pub fn apply_uci(&mut self, s: &str) -> Result<StandardMove, CoreError> {
        let mv = StandardMove::from_uci(&self.board, s)?;
        self.apply(mv)?;
        Ok(mv)
    }

    /// Apply a sequence of UCI moves.
    pub fn apply_uci_seq(&mut self, moves: &[&str]) -> Result<(), CoreError> {
        for m in moves {
            self.apply_uci(m)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn startpos_basics() {
        let g = GameState::startpos();
        assert_eq!(g.side_to_move(), Color::White);
        assert_eq!(g.ply(), 0);
        assert_eq!(g.legal_standard_moves().len(), 20);
        assert_eq!(g.legal_actions().len(), 20);
        assert!(!g.is_terminal());
        assert_eq!(g.repetition_count(), 1);
        assert_eq!(g.history().len(), 1);
    }

    #[test]
    fn apply_updates_state_and_history() {
        let mut g = GameState::startpos();
        g.apply_uci("e2e4").unwrap();
        assert_eq!(g.side_to_move(), Color::Black);
        assert_eq!(g.ply(), 1);
        assert_eq!(g.history().len(), 2);
        assert_eq!(
            g.to_fen(),
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        );
    }

    #[test]
    fn illegal_move_errors() {
        let mut g = GameState::startpos();
        assert!(g.apply_uci("e2e5").is_err());
        assert!(g.apply_uci("e1e8").is_err());
        // State unchanged.
        assert_eq!(g.ply(), 0);
    }

    #[test]
    fn cannot_apply_to_terminal_game() {
        let mut g =
            GameState::from_fen("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
                .unwrap();
        assert_eq!(g.termination(), Some(Termination::Checkmate));
        assert!(g.apply_uci("e1e2").is_err());
    }

    #[test]
    fn threefold_by_knight_shuffle() {
        let mut g = GameState::startpos();
        // 1.Nf3 Nf6 2.Ng1 Ng8 3.Nf3 Nf6 4.Ng1 Ng8
        let seq = [
            "g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8",
        ];
        for (i, m) in seq.iter().enumerate() {
            g.apply_uci(m).unwrap();
            if i == 3 {
                assert_eq!(g.repetition_count(), 2, "second occurrence at ply 4");
                assert!(!g.is_terminal());
            }
        }
        assert_eq!(g.repetition_count(), 3);
        assert_eq!(g.termination(), Some(Termination::ThreefoldRepetition));
    }

    #[test]
    fn fifty_move_triggers_at_clock_100() {
        let mut g = GameState::from_fen("8/8/8/4k3/8/8/8/R3K2R w KQ - 99 1").unwrap();
        assert!(!g.is_terminal());
        g.apply_uci("e1e2").unwrap();
        assert_eq!(g.termination(), Some(Termination::FiftyMoveRule));
    }

    #[test]
    fn truncation_cap() {
        let mut g = GameState::startpos().with_max_plies(2);
        g.apply_uci("e2e4").unwrap();
        assert!(!g.is_terminal());
        g.apply_uci("e7e5").unwrap();
        assert_eq!(g.termination(), Some(Termination::Truncated));
        assert_eq!(g.outcome(), None);
    }

    #[test]
    fn legal_actions_are_unique_and_decodable() {
        let g = GameState::startpos();
        let ids = g.legal_actions();
        let mut sorted = ids.clone();
        sorted.dedup();
        assert_eq!(sorted.len(), ids.len(), "no duplicate ActionIds");
        for id in ids {
            let (f, t, p) = id.to_physical(g.perspective());
            let promo = if p.is_none() { None } else { Some(p) };
            // Every decoded physical move must be legal.
            let mv = StandardMove::new(f, t, promo);
            assert!(g.board().is_legal(mv.to_cozy(g.board()).unwrap()));
        }
    }
}
