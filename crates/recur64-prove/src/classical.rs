//! A strong tablebase-free classical ordering heuristic (the baseline the learned ordering must beat).
//!
//! Attacker moves are scored on the position they create: mate first, stalemate and hanging pieces last, then
//! fewer defender replies, checks, a defender king pushed to the edge, and a nearby attacking king. Defender
//! replies are ordered refutations-first: capturing the attacking piece, then escaping toward the centre.

use cozy_chess::{Board, Color, GameStatus, Move, Piece};

use crate::search::Orderer;
use crate::tb::moves_of;

fn king_sq(b: &Board, c: Color) -> (i32, i32) {
    let s = b.king(c);
    (s.file() as i32, s.rank() as i32)
}

pub fn edge_distance(f: i32, r: i32) -> i32 {
    f.min(7 - f).min(r).min(7 - r)
}

fn white_heavy(b: &Board) -> usize {
    (b.colored_pieces(Color::White, Piece::Queen) | b.colored_pieces(Color::White, Piece::Rook)).len() as usize
}

pub fn attacker_score(b: &Board, m: Move) -> i32 {
    let mut after = b.clone();
    after.play_unchecked(m);
    match after.status() {
        GameStatus::Won => return 1_000_000,
        GameStatus::Drawn => return -1_000_000,
        GameStatus::Ongoing => {}
    }
    let replies = moves_of(&after);
    let before = white_heavy(b);
    let hanging = replies.iter().any(|r| {
        let mut x = after.clone();
        x.play_unchecked(*r);
        white_heavy(&x) < before
    });
    if hanging {
        return -500_000;
    }
    let (bf, br) = king_sq(&after, Color::Black);
    let (wf, wr) = king_sq(&after, Color::White);
    let cheb = (bf - wf).abs().max((br - wr).abs());
    let check = i32::from(!after.checkers().is_empty());
    -10 * replies.len() as i32 + 6 * check - 6 * edge_distance(bf, br) - 2 * cheb
}

pub struct Classical;

impl Orderer for Classical {
    fn attacker(&mut self, b: &Board, moves: &mut Vec<Move>) {
        let mut scored: Vec<(i32, Move)> = moves.iter().map(|&m| (attacker_score(b, m), m)).collect();
        scored.sort_by_key(|&(s, _)| std::cmp::Reverse(s));
        *moves = scored.into_iter().map(|(_, m)| m).collect();
    }

    fn defender(&mut self, b: &Board, moves: &mut Vec<Move>) {
        let mut scored: Vec<(i32, Move)> = moves
            .iter()
            .map(|&m| {
                let capture = i32::from(b.colors(Color::White).has(m.to)) * 100;
                let centre = edge_distance(m.to.file() as i32, m.to.rank() as i32);
                (capture + centre, m)
            })
            .collect();
        scored.sort_by_key(|&(s, _)| std::cmp::Reverse(s));
        *moves = scored.into_iter().map(|(_, m)| m).collect();
    }
}
