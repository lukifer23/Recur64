//! Standard (UCI-style) move representation and cozy-chess conversion.
//!
//! Recur64 has exactly **one** external move convention: standard chess/UCI, in
//! which castling is written as the **king destination** (`e1g1`, `e1c1`,
//! `e8g8`, `e8c8`). cozy-chess stores castling internally as king-captures-rook
//! (`e1h1`, ...); that internal form never leaks into the action or observation
//! layers.

use std::fmt;

use cozy_chess::{Board, Move as CozyMove, util};

use crate::action::PromotionCode;
use crate::error::CoreError;
use crate::square::{Color, Piece, Square};

/// A move in standard chess/UCI convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StandardMove {
    pub from: Square,
    pub to: Square,
    /// `None` for non-promotions; `Some(N/B/R/Q)` for promotions.
    pub promotion: Option<PromotionCode>,
}

fn king_home(side: Color) -> Square {
    match side {
        Color::White => Square::E1,
        Color::Black => Square::E8,
    }
}

fn rook_home_short(side: Color) -> Square {
    match side {
        Color::White => Square::H1,
        Color::Black => Square::H8,
    }
}

fn rook_home_long(side: Color) -> Square {
    match side {
        Color::White => Square::A1,
        Color::Black => Square::A8,
    }
}

fn castle_king_dest(side: Color, short: bool) -> Square {
    match (side, short) {
        (Color::White, true) => Square::G1,
        (Color::White, false) => Square::C1,
        (Color::Black, true) => Square::G8,
        (Color::Black, false) => Square::C8,
    }
}

impl StandardMove {
    pub const fn new(from: Square, to: Square, promotion: Option<PromotionCode>) -> Self {
        Self {
            from,
            to,
            promotion,
        }
    }

    /// Convert a cozy move (handling king-captures-rook castling) into the
    /// standard convention.
    pub fn from_cozy(board: &Board, mv: CozyMove) -> Result<Self, CoreError> {
        let side = board
            .color_on(mv.from)
            .ok_or_else(|| CoreError::InvalidUci(format!("move from empty square {}", mv.from)))?;
        let piece = board
            .piece_on(mv.from)
            .expect("color_on implies a piece on the square");

        if piece == Piece::King && mv.from == king_home(side) {
            if mv.to == rook_home_short(side) {
                return Ok(StandardMove::new(
                    mv.from,
                    castle_king_dest(side, true),
                    None,
                ));
            }
            if mv.to == rook_home_long(side) {
                return Ok(StandardMove::new(
                    mv.from,
                    castle_king_dest(side, false),
                    None,
                ));
            }
        }

        let promotion = match mv.promotion {
            Some(p) => Some(PromotionCode::from_piece(p)?),
            None => None,
        };
        Ok(StandardMove::new(mv.from, mv.to, promotion))
    }

    /// Convert to a cozy move (expanding castling to king-captures-rook).
    pub fn to_cozy(&self, board: &Board) -> Result<CozyMove, CoreError> {
        let side = board.color_on(self.from).ok_or_else(|| {
            CoreError::InvalidUci(format!("move from empty square {}", self.from))
        })?;
        let piece = board
            .piece_on(self.from)
            .expect("color_on implies a piece on the square");

        if piece == Piece::King && self.from == king_home(side) {
            if self.to == castle_king_dest(side, true) {
                return Ok(CozyMove {
                    from: self.from,
                    to: rook_home_short(side),
                    promotion: None,
                });
            }
            if self.to == castle_king_dest(side, false) {
                return Ok(CozyMove {
                    from: self.from,
                    to: rook_home_long(side),
                    promotion: None,
                });
            }
        }

        Ok(CozyMove {
            from: self.from,
            to: self.to,
            promotion: self.promotion.and_then(|p| p.piece()),
        })
    }

    /// Parse a UCI move string in the context of `board`. Malformed input fails
    /// visibly; no repair.
    pub fn from_uci(board: &Board, s: &str) -> Result<Self, CoreError> {
        let mv = util::parse_uci_move(board, s)
            .map_err(|e| CoreError::InvalidUci(format!("{s:?}: {e}")))?;
        let parsed = Self::from_cozy(board, mv)?;
        // cozy's parser ignores trailing characters; require the exact
        // canonical form so malformed input is never silently repaired.
        if parsed.to_uci() != s {
            return Err(CoreError::InvalidUci(format!(
                "{s:?} is not a canonical UCI move"
            )));
        }
        Ok(parsed)
    }

    /// Format as a UCI move string.
    pub fn to_uci(self) -> String {
        let mut s = format!("{}{}", self.from, self.to);
        if let Some(p) = self.promotion {
            s.push(match p.code() {
                1 => 'n',
                2 => 'b',
                3 => 'r',
                4 => 'q',
                _ => unreachable!("PromotionCode is validated"),
            });
        }
        s
    }
}

impl fmt::Display for StandardMove {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_uci())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn board() -> Board {
        Board::default()
    }

    #[test]
    fn normal_and_capture_roundtrip() {
        let b = board();
        for s in ["e2e4", "g1f3", "b1c3"] {
            let mv = StandardMove::from_uci(&b, s).unwrap();
            assert_eq!(mv.to_uci(), s);
            let cozy = mv.to_cozy(&b).unwrap();
            assert_eq!(StandardMove::from_cozy(&b, cozy).unwrap(), mv);
        }
    }

    #[test]
    fn castling_roundtrips_all_four() {
        // White: e1g1 / e1c1; Black: e8g8 / e8c8.
        let white: Board = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1".parse().unwrap();
        for s in ["e1g1", "e1c1"] {
            let mv = StandardMove::from_uci(&white, s).unwrap();
            assert_eq!(mv.to_uci(), s, "white castling {s}");
            let cozy = mv.to_cozy(&white).unwrap();
            assert_eq!(StandardMove::from_cozy(&white, cozy).unwrap(), mv);
        }
        let black: Board = "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1".parse().unwrap();
        for s in ["e8g8", "e8c8"] {
            let mv = StandardMove::from_uci(&black, s).unwrap();
            assert_eq!(mv.to_uci(), s, "black castling {s}");
            let cozy = mv.to_cozy(&black).unwrap();
            assert_eq!(StandardMove::from_cozy(&black, cozy).unwrap(), mv);
        }
    }

    #[test]
    fn castling_matches_cozy_util() {
        // Cross-check our detection against cozy's own UCI converters.
        let boards = [
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
            "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
        ];
        for fen in boards {
            let b: Board = fen.parse().unwrap();
            b.generate_moves(|moves| {
                for mv in moves {
                    let ours = StandardMove::from_cozy(&b, mv).unwrap();
                    let via_util = util::display_uci_move(&b, mv).to_string();
                    assert_eq!(ours.to_uci(), via_util, "castling mismatch on {fen}");
                }
                false
            });
        }
    }

    #[test]
    fn promotions_all_types_roundtrip() {
        let b: Board = "8/P7/8/8/8/8/8/k6K w - - 0 1".parse().unwrap();
        for s in ["a7a8q", "a7a8r", "a7a8b", "a7a8n"] {
            let mv = StandardMove::from_uci(&b, s).unwrap();
            assert_eq!(mv.to_uci(), s);
            assert!(mv.promotion.is_some());
        }
    }

    #[test]
    fn en_passant_roundtrip() {
        let b: Board = "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3"
            .parse()
            .unwrap();
        let mv = StandardMove::from_uci(&b, "e5d6").unwrap();
        assert_eq!(mv.to_uci(), "e5d6");
        let cozy = mv.to_cozy(&b).unwrap();
        assert_eq!(StandardMove::from_cozy(&b, cozy).unwrap(), mv);
    }

    #[test]
    fn malformed_uci_fails_visibly() {
        let b = board();
        for bad in ["", "e2", "z9z9", "e2e9", "e2e4qq"] {
            assert!(
                StandardMove::from_uci(&b, bad).is_err(),
                "{bad:?} should fail"
            );
        }
    }
}
