//! Recur64 Rules Profile V1.
//!
//! Standard chess only (no Chess960). Draw convention: **auto-claim on the
//! current position** for threefold repetition and the 50-move condition. This
//! is an engine-training convention, not a model of optional human claim
//! strategy.
//!
//! Termination precedence is explicit and ordered; see [`classify`].

use cozy_chess::{Board, Color, Piece};

/// Why a game ended (or why it was administratively stopped).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Termination {
    /// Side to move is in check and has no legal moves.
    Checkmate,
    /// Side to move is not in check and has no legal moves.
    Stalemate,
    /// Recognized dead position from our conservative material set.
    InsufficientMaterial,
    /// Auto-claimed threefold repetition (count >= 3).
    ThreefoldRepetition,
    /// Auto-claimed 50-move condition (halfmove clock >= 100).
    FiftyMoveRule,
    /// Administrative ply cap reached. **Not a chess result.**
    Truncated,
    /// Externally cancelled. **Not a chess result.**
    Aborted,
}

/// A game result, from the perspective of the board (winner color / draw).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Outcome {
    Win(Color),
    Draw,
}

impl Termination {
    pub fn is_draw(self) -> bool {
        matches!(
            self,
            Termination::Stalemate
                | Termination::InsufficientMaterial
                | Termination::ThreefoldRepetition
                | Termination::FiftyMoveRule
        )
    }

    pub fn is_decisive(self) -> bool {
        matches!(self, Termination::Checkmate)
    }

    /// `Truncated` and `Aborted` are not results and yield no WDL supervision.
    pub fn outcome(self, side_to_move: Color) -> Option<Outcome> {
        match self {
            Termination::Checkmate => Some(Outcome::Win(!side_to_move)),
            Termination::Stalemate
            | Termination::InsufficientMaterial
            | Termination::ThreefoldRepetition
            | Termination::FiftyMoveRule => Some(Outcome::Draw),
            Termination::Truncated | Termination::Aborted => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Termination::Checkmate => "checkmate",
            Termination::Stalemate => "stalemate",
            Termination::InsufficientMaterial => "insufficient_material",
            Termination::ThreefoldRepetition => "threefold_repetition",
            Termination::FiftyMoveRule => "fifty_move_rule",
            Termination::Truncated => "truncated",
            Termination::Aborted => "aborted",
        }
    }
}

/// True if the position has at least one legal move.
pub fn has_legal_move(board: &Board) -> bool {
    board.generate_moves(|_| true)
}

/// Square color: `false` = dark, `true` = light. a1 is dark.
fn square_is_light(square: cozy_chess::Square) -> bool {
    (square.file() as u8 + square.rank() as u8) % 2 == 1
}

/// Conservative, sound insufficient-material recognition.
///
/// Recognizes only cases that are provably dead:
/// - K vs K;
/// - K + one minor (bishop or knight) vs K;
/// - bishops-only positions where **all** bishops stand on one square color.
///
/// Deliberately does **not** recognize K+N vs K+N, opposite-colored bishops,
/// or unusual blocked positions. Missing a dead position is safe (the game
/// still ends by another rule); falsely declaring a draw is not.
pub fn is_insufficient_material(board: &Board) -> bool {
    if !board.pieces(Piece::Pawn).is_empty()
        || !board.pieces(Piece::Rook).is_empty()
        || !board.pieces(Piece::Queen).is_empty()
    {
        return false;
    }
    let bishops = board.pieces(Piece::Bishop);
    let knights = board.pieces(Piece::Knight);
    let minors = bishops.len() + knights.len();
    if minors == 0 {
        return true; // K vs K
    }
    if minors == 1 {
        return true; // K+minor vs K
    }
    if !knights.is_empty() {
        return false; // not recognized
    }
    // Bishops only: dead iff all bishops share a square color.
    let mut color: Option<bool> = None;
    for sq in bishops {
        let light = square_is_light(sq);
        match color {
            None => color = Some(light),
            Some(c) if c != light => return false,
            _ => {}
        }
    }
    true
}

/// Classify the termination of `board`, in exact precedence order.
///
/// `repetition_count` is the FIDE repetition count of the current position
/// (including the current occurrence; 1 means "seen once").
/// `ply` is the number of plies already played.
pub fn classify(
    board: &Board,
    repetition_count: u32,
    ply: u32,
    max_plies: Option<u32>,
) -> Option<Termination> {
    // 1-2. Checkmate / stalemate take precedence over every draw condition.
    if !has_legal_move(board) {
        let in_check = !board.checkers().is_empty();
        return Some(if in_check {
            Termination::Checkmate
        } else {
            Termination::Stalemate
        });
    }
    // 3. Dead position (automatic under FIDE).
    if is_insufficient_material(board) {
        return Some(Termination::InsufficientMaterial);
    }
    // 4. Auto-claimed threefold repetition.
    if repetition_count >= 3 {
        return Some(Termination::ThreefoldRepetition);
    }
    // 5. Auto-claimed 50-move condition.
    if board.halfmove_clock() >= 100 {
        return Some(Termination::FiftyMoveRule);
    }
    // 6. Administrative cap (NOT a chess result).
    if max_plies.is_some_and(|max| ply >= max) {
        return Some(Termination::Truncated);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insufficient_material_recognized_cases() {
        for fen in [
            "8/8/8/4k3/8/8/8/4K3 w - - 0 1",     // K vs K
            "8/8/8/4k3/8/8/8/2B1K3 w - - 0 1",   // K+B vs K
            "8/8/8/4k3/8/8/8/2N1K3 w - - 0 1",   // K+N vs K
            "8/8/8/4k3/8/8/1b6/2B1K3 w - - 0 1", // bishops b2 and c1: both dark
        ] {
            let b: Board = fen.parse().unwrap();
            assert!(is_insufficient_material(&b), "{fen} should be dead");
        }
    }

    #[test]
    fn insufficient_material_not_recognized() {
        // K+N+N vs K is not in our recognized set.
        let b: Board = "8/8/8/4k3/8/8/8/2N1K1N1 w - - 0 1".parse().unwrap();
        assert!(!is_insufficient_material(&b), "K+N+N vs K not dead");
        // Opposite-colored bishops: c1 (dark) and d5 (light).
        let b: Board = "4k3/8/8/3b4/8/8/8/2B1K3 w - - 0 1".parse().unwrap();
        assert!(!is_insufficient_material(&b), "opposite bishops not dead");
        // Pawn present.
        let b: Board = "8/8/8/4k3/8/8/4P3/4K3 w - - 0 1".parse().unwrap();
        assert!(!is_insufficient_material(&b), "pawn means not dead");
    }

    #[test]
    fn checkmate_precedence_over_draws() {
        // Fool's mate; halfmove clock is irrelevant, checkmate wins.
        let b: Board = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
            .parse()
            .unwrap();
        assert_eq!(classify(&b, 1, 4, None), Some(Termination::Checkmate));
    }

    #[test]
    fn stalemate_classified() {
        let b: Board = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1".parse().unwrap();
        assert_eq!(classify(&b, 1, 1, None), Some(Termination::Stalemate));
    }

    #[test]
    fn threefold_and_fifty_precedence() {
        let b: Board = "8/8/8/4k3/8/8/8/R3K2R w KQ - 0 1".parse().unwrap();
        assert_eq!(
            classify(&b, 3, 10, None),
            Some(Termination::ThreefoldRepetition)
        );
        // halfmove clock 100 with a non-dead position
        let b: Board = "8/8/8/4k3/8/8/8/R3K2R w KQ - 100 1".parse().unwrap();
        assert_eq!(classify(&b, 1, 200, None), Some(Termination::FiftyMoveRule));
    }

    #[test]
    fn truncation_is_not_a_draw() {
        let b = Board::default();
        let t = classify(&b, 1, 512, Some(512)).unwrap();
        assert_eq!(t, Termination::Truncated);
        assert_eq!(t.outcome(b.side_to_move()), None);
        assert!(!t.is_draw());
    }

    #[test]
    fn checkmate_outcome_winner_is_opponent_of_mover() {
        let b: Board = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
            .parse()
            .unwrap();
        assert_eq!(
            Termination::Checkmate.outcome(b.side_to_move()),
            Some(Outcome::Win(Color::Black))
        );
    }
}
