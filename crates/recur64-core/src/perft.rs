//! Perft traversal using Recur64's own move conversion.
//!
//! Perft counts leaf nodes. Traversal uses cozy-chess legal move generation but
//! routes every move through Recur64's `StandardMove` conversion and back, so a
//! conversion bug shows up as a node-count mismatch against published counts.

use cozy_chess::Board;

use crate::uci::StandardMove;

/// Count leaf nodes at `depth` from `board`.
pub fn perft(board: &Board, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let mut nodes = 0u64;
    board.generate_moves(|moves| {
        for mv in moves {
            let standard = StandardMove::from_cozy(board, mv).expect("cozy move well-formed");
            let back = standard
                .to_cozy(board)
                .expect("standard move converts back");
            debug_assert_eq!(back, mv, "cozy<->standard conversion must round-trip");
            let mut child = board.clone();
            // Only generated (legal) moves are played.
            child.play_unchecked(back);
            nodes += if depth == 1 {
                1
            } else {
                perft(&child, depth - 1)
            };
        }
        false
    });
    nodes
}

/// Divide-style breakdown: legal moves and their subtree node counts at `depth`.
pub fn perft_divide(board: &Board, depth: u32) -> Vec<(String, u64)> {
    let mut out = Vec::new();
    board.generate_moves(|moves| {
        for mv in moves {
            let standard = StandardMove::from_cozy(board, mv).expect("well-formed");
            let mut child = board.clone();
            child.play_unchecked(mv);
            let n = if depth <= 1 {
                1
            } else {
                perft(&child, depth - 1)
            };
            out.push((standard.to_uci(), n));
        }
        false
    });
    out.sort();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn startpos_shallow() {
        let b = Board::default();
        assert_eq!(perft(&b, 0), 1);
        assert_eq!(perft(&b, 1), 20);
        assert_eq!(perft(&b, 2), 400);
        assert_eq!(perft(&b, 3), 8902);
    }

    #[test]
    fn divide_sums_to_total() {
        let b: Board = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
            .parse()
            .unwrap();
        let total: u64 = perft_divide(&b, 2).iter().map(|(_, n)| n).sum();
        assert_eq!(total, perft(&b, 2));
        assert_eq!(total, 2039);
    }
}
