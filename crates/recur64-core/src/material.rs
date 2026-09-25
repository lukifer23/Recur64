//! Material accounting for diagnostics (standard 1/3/3/5/9 piece values).
//!
//! Diagnostic only: search and training never use material values.

use cozy_chess::{Board, Color, Piece};

const VALUES: [(Piece, i32); 5] = [
    (Piece::Pawn, 1),
    (Piece::Knight, 3),
    (Piece::Bishop, 3),
    (Piece::Rook, 5),
    (Piece::Queen, 9),
];

/// Material of `color` on `board` in pawn units.
pub fn material(board: &Board, color: Color) -> i32 {
    VALUES
        .iter()
        .map(|(piece, value)| board.colored_pieces(color, *piece).len() as i32 * value)
        .sum()
}

/// White material minus Black material.
pub fn material_balance(board: &Board) -> i32 {
    material(board, Color::White) - material(board, Color::Black)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn start_position_is_balanced_with_39_each() {
        let b = Board::default();
        assert_eq!(material(&b, Color::White), 39);
        assert_eq!(material(&b, Color::Black), 39);
        assert_eq!(material_balance(&b), 0);
    }

    #[test]
    fn queen_up_is_plus_nine() {
        let b: Board = "4k3/8/8/8/8/8/8/3QK3 w - - 0 1".parse().unwrap();
        assert_eq!(material_balance(&b), 9);
    }
}
