//! Squares, colors, pieces, and the canonical (current-side-to-move) perspective.
//!
//! Recur64 uses `a1 = 0`, rank-major indexing, matching cozy-chess exactly.
//! Canonicalization for a Black-to-move position is a **rank reflection**
//! (`square XOR 56`) plus a **color swap**. Files are fixed, which preserves the
//! kingside/queenside meaning. The transform is its own inverse.

use crate::error::CoreError;

pub use cozy_chess::{Color, File, Piece, Rank, Square};

/// Number of squares on the board.
pub const NUM_SQUARES: u8 = 64;

/// Reflect a square vertically (rank reflection), preserving file. `a1 <-> a8`.
pub const fn flip_rank(sq: Square) -> Square {
    sq.flip_rank()
}

/// Canonical square from the perspective of `side`.
///
/// White to move is the identity; Black to move reflects the rank. This is an
/// involution, so it is exactly reversible.
pub const fn canonical_square(sq: Square, side: Color) -> Square {
    sq.relative_to(side)
}

/// Canonical color from the perspective of `side`.
///
/// After canonicalization the side to move is always `White` (channels 0..5 in
/// Observation V1 are "own" pieces).
pub const fn canonical_color(color: Color, side: Color) -> Color {
    match side {
        Color::White => color,
        Color::Black => invert(color),
    }
}

/// `const`-friendly color inversion (`Color` implements `Not`).
pub const fn invert(color: Color) -> Color {
    match color {
        Color::White => Color::Black,
        Color::Black => Color::White,
    }
}

/// Convert a raw index to a square, erroring visibly out of range.
pub fn square_from_index(index: u8) -> Result<Square, CoreError> {
    if index >= NUM_SQUARES {
        return Err(CoreError::InvalidCoordinate(format!(
            "square index {index} out of range 0..64"
        )));
    }
    Ok(Square::index(index as usize))
}

/// The perspective used to canonicalize an observation or action: the current
/// side to move.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Perspective {
    pub side: Color,
}

impl Perspective {
    pub const fn white() -> Self {
        Self { side: Color::White }
    }
    pub const fn black() -> Self {
        Self { side: Color::Black }
    }
    pub const fn of(side: Color) -> Self {
        Self { side }
    }
    /// Canonicalize a square.
    pub const fn square(&self, sq: Square) -> Square {
        canonical_square(sq, self.side)
    }
    /// Canonicalize a color.
    pub const fn color(&self, color: Color) -> Color {
        canonical_color(color, self.side)
    }
    pub const fn is_identity(&self) -> bool {
        matches!(self.side, Color::White)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_is_involution() {
        for i in 0u8..64 {
            let sq = Square::index(i as usize);
            for side in [Color::White, Color::Black] {
                let once = canonical_square(sq, side);
                let twice = canonical_square(once, side);
                assert_eq!(sq, twice, "canonical transform must be an involution");
            }
        }
    }

    #[test]
    fn black_canonical_reflects_rank_and_fixes_files() {
        for i in 0u8..64 {
            let sq = Square::index(i as usize);
            let c = canonical_square(sq, Color::Black);
            assert_eq!(c as usize, (i as usize) ^ 56);
            assert_eq!(c.file(), sq.file(), "files must stay fixed");
        }
    }

    #[test]
    fn canonical_color_swaps_only_for_black() {
        assert_eq!(canonical_color(Color::White, Color::White), Color::White);
        assert_eq!(canonical_color(Color::Black, Color::White), Color::Black);
        assert_eq!(canonical_color(Color::White, Color::Black), Color::Black);
        assert_eq!(canonical_color(Color::Black, Color::Black), Color::White);
    }

    #[test]
    fn hand_fixtures() {
        // e1 -> e8 under black canonicalization; files preserved.
        assert_eq!(canonical_square(Square::E1, Color::Black), Square::E8);
        assert_eq!(canonical_square(Square::E8, Color::Black), Square::E1);
        assert_eq!(canonical_square(Square::A1, Color::Black), Square::A8);
        assert_eq!(canonical_square(Square::H1, Color::Black), Square::H8);
        // White is identity.
        assert_eq!(canonical_square(Square::E1, Color::White), Square::E1);
    }
}
