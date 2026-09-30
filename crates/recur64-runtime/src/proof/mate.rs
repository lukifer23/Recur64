//! Exact forced-mate solver for bare-king defender endings.
//!
//! No network, no PUCT, no heuristic evaluation, no external engine: exhaustive
//! adversarial search under the chess rules, with memoization.
//!
//! **Mate in N** means the attacker (the side to move at the root) can force
//! checkmate within N of the attacker's own moves against EVERY legal defender
//! reply. The minimal such N is the position's *mate depth*; the *correct* root
//! moves are those that begin a forced mate of exactly that minimal depth.
//!
//! Contract `fresh_no_history_v1`: positions come from a FEN with a zero
//! half-move clock and no earlier history. A line of at most nine plies then
//! cannot reach a threefold repetition or the fifty-move claim, so the only
//! terminations inside the search are checkmate, stalemate and insufficient
//! material (checkmate and stalemate take precedence over every draw rule, as in
//! `recur64_core::rules::classify`). A defender reply that stalemates, or leaves
//! a dead position, is simply a reply after which no mate follows.

use std::collections::HashMap;

use cozy_chess::{Board, GameStatus, Move};

use recur64_core::is_insufficient_material;

/// Deepest mate this solver is asked for (M5 is the pre-registered maximum).
pub const MAX_DEPTH: u8 = 5;

fn moves(b: &Board) -> Vec<Move> {
    let mut v = Vec::with_capacity(48);
    b.generate_moves(|m| {
        v.extend(m);
        false
    });
    v
}

/// Position still in play: has legal moves and is not a dead position.
fn ongoing(b: &Board) -> bool {
    b.status() == GameStatus::Ongoing && !is_insufficient_material(b)
}

/// Memoizing exact solver. One instance may be reused across positions.
#[derive(Default)]
pub struct MateSolver {
    /// (position hash, attacker moves left) -> attacker (to move) forces mate.
    attacker: HashMap<(u64, u8), bool>,
    /// (position hash, attacker moves left) -> defender (to move) is forced
    /// into mate whatever it plays.
    defender: HashMap<(u64, u8), bool>,
    /// Search nodes visited (for throughput reporting).
    pub nodes: u64,
}

impl MateSolver {
    pub fn new() -> Self {
        Self::default()
    }

    /// Memoized entries currently held (callers reset the solver when this grows).
    pub fn table_size(&self) -> usize {
        self.attacker.len() + self.defender.len()
    }

    /// The side to move in `b` can force checkmate within `n` of its own moves.
    pub fn attacker_forces(&mut self, b: &Board, n: u8) -> bool {
        if n == 0 || !ongoing(b) {
            return false;
        }
        let key = (b.hash(), n);
        if let Some(&v) = self.attacker.get(&key) {
            return v;
        }
        self.nodes += 1;
        let mut result = false;
        for m in moves(b) {
            let mut after = b.clone();
            after.play_unchecked(m);
            // Checkmate ends the game whatever the clocks say.
            if after.status() == GameStatus::Won {
                result = true;
                break;
            }
            if n >= 2 && self.defender_lost(&after, n - 1) {
                result = true;
                break;
            }
        }
        self.attacker.insert(key, result);
        result
    }

    /// `after` has the defender to move: it is still in play, has replies, and
    /// after EVERY reply the attacker forces mate within `n` more moves.
    pub fn defender_lost(&mut self, after: &Board, n: u8) -> bool {
        if n == 0 || !ongoing(after) {
            return false;
        }
        let key = (after.hash(), n);
        if let Some(&v) = self.defender.get(&key) {
            return v;
        }
        self.nodes += 1;
        let mut all = true;
        for r in moves(after) {
            let mut next = after.clone();
            next.play_unchecked(r);
            // A defender reply that ends the game (stalemate, dead position, or
            // mating the attacker) is not a forced loss.
            if !self.attacker_forces(&next, n) {
                all = false;
                break;
            }
        }
        self.defender.insert(key, all);
        all
    }

    /// Minimal N in `1..=max` such that the side to move forces mate within N
    /// of its moves, or `None` if there is none within `max`.
    pub fn mate_depth(&mut self, b: &Board, max: u8) -> Option<u8> {
        (1..=max.min(MAX_DEPTH)).find(|&n| self.attacker_forces(b, n))
    }

    /// For a position of mate depth `depth`: indices (into `moves_in_order`) of
    /// the root moves that begin a forced mate of exactly `depth` attacker moves.
    pub fn correct_moves(&mut self, b: &Board, depth: u8, moves_in_order: &[Move]) -> Vec<usize> {
        let mut out = Vec::new();
        for (i, m) in moves_in_order.iter().enumerate() {
            let mut after = b.clone();
            after.play_unchecked(*m);
            let wins = if after.status() == GameStatus::Won {
                depth == 1
            } else {
                depth >= 2 && self.defender_lost(&after, depth - 1)
            };
            if wins {
                out.push(i);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn board(fen: &str) -> Board {
        fen.parse().unwrap()
    }

    #[test]
    fn back_rank_mate_in_one() {
        let b = board("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1");
        let mut s = MateSolver::new();
        assert_eq!(s.mate_depth(&b, 3), Some(1));
    }

    /// Direct enumeration, written independently of the solver: the side to move
    /// has a checkmate now.
    fn brute_mate_in_1(b: &Board) -> bool {
        moves(b).into_iter().any(|m| {
            let mut a = b.clone();
            a.play_unchecked(m);
            a.status() == GameStatus::Won
        })
    }

    /// Direct enumeration: a forced mate in exactly two (no mate in one).
    fn brute_mate_in_2(b: &Board) -> bool {
        if brute_mate_in_1(b) {
            return false;
        }
        moves(b).into_iter().any(|m| {
            let mut a = b.clone();
            a.play_unchecked(m);
            if a.status() != GameStatus::Ongoing || is_insufficient_material(&a) {
                return false;
            }
            let replies = moves(&a);
            !replies.is_empty()
                && replies.into_iter().all(|r| {
                    let mut n = a.clone();
                    n.play_unchecked(r);
                    n.status() == GameStatus::Ongoing && brute_mate_in_1(&n)
                })
        })
    }

    #[test]
    fn solver_agrees_with_direct_enumeration_on_random_positions() {
        use super::super::generator::{Rng, sample_position};
        let mut rng = Rng(99);
        let mut s = MateSolver::new();
        let (mut d1, mut d2, mut checked) = (0, 0, 0);
        while checked < 600 {
            let white: &[char] = match checked % 3 {
                0 => &['K', 'Q'],
                1 => &['K', 'R'],
                _ => &['K', 'Q', 'R'],
            };
            let Some((_, st)) = sample_position(&mut rng, white) else {
                continue;
            };
            let b = st.board();
            let depth = s.mate_depth(b, 2);
            assert_eq!(depth == Some(1), brute_mate_in_1(b), "{}", st.to_fen());
            assert_eq!(depth == Some(2), brute_mate_in_2(b), "{}", st.to_fen());
            d1 += usize::from(depth == Some(1));
            d2 += usize::from(depth == Some(2));
            checked += 1;
        }
        assert!(
            d1 > 20 && d2 > 20,
            "coverage: {d1} mate-in-1, {d2} mate-in-2"
        );
    }

    #[test]
    fn known_depths_and_a_rook_ladder() {
        let mut s = MateSolver::new();
        // Qb8 is checkmate: the queen covers g8, the king covers g7 and h7.
        let m1 = board("7k/8/6K1/8/8/8/8/1Q6 w - - 0 1");
        assert_eq!(s.mate_depth(&m1, 3), Some(1));
        // A lone rook and a far-away king cannot force mate within two moves.
        let far = board("8/8/8/4k3/8/8/8/R3K3 w - - 0 1");
        assert!(s.mate_depth(&far, 2).is_none());
    }

    #[test]
    fn stalemate_is_not_a_forced_mate() {
        // Several white queen moves here stalemate the bare king. The solver must
        // agree exactly with a direct enumeration of moves that give checkmate:
        // a stalemating move is never counted as a mate.
        let b = board("7k/8/6Q1/8/8/8/8/K7 w - - 0 1");
        let mut s = MateSolver::new();
        let stalemates = moves(&b)
            .into_iter()
            .filter(|m| {
                let mut a = b.clone();
                a.play_unchecked(*m);
                a.status() == GameStatus::Drawn
            })
            .count();
        assert!(stalemates > 0, "the fixture must contain stalemating moves");
        let mates = moves(&b)
            .into_iter()
            .filter(|m| {
                let mut a = b.clone();
                a.play_unchecked(*m);
                a.status() == GameStatus::Won
            })
            .count();
        assert_eq!(s.attacker_forces(&b, 1), mates > 0);
    }

    #[test]
    fn correct_moves_are_exactly_the_shortest_winning_first_moves() {
        let b = board("7k/8/6K1/8/8/8/8/1Q6 w - - 0 1");
        let mut s = MateSolver::new();
        let d = s.mate_depth(&b, 3).unwrap();
        let ms = moves(&b);
        let correct = s.correct_moves(&b, d, &ms);
        assert!(!correct.is_empty());
        for (i, mv) in ms.iter().enumerate() {
            let mut after = b.clone();
            after.play_unchecked(*mv);
            let is_win = after.status() == GameStatus::Won
                || (d >= 2 && MateSolver::new().defender_lost(&after, d - 1));
            assert_eq!(correct.contains(&i), is_win);
        }
        // No correct move wins faster than depth (it would have been depth d-1).
        for &i in &correct {
            let mut after = b.clone();
            after.play_unchecked(ms[i]);
            if d >= 2 {
                assert!(!MateSolver::new().defender_lost(&after, d - 2));
            }
        }
    }
}
