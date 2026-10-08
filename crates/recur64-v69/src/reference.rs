//! Independently structured exhaustive enumerator (verification only).
//!
//! Differences from `oracle`: no cache, no node budget, no move-ordering
//! shortcuts, no use of `recur64_core::rules`; it computes the exact minimax
//! *value* (minimal number of attacker moves forcing mate, `None` if none within
//! the limit) instead of a boolean budgeted predicate. Terminal detection is
//! "no legal moves" (+ check/no check). Repetition/50-move/dead-position rules
//! cannot change a mate verdict in this domain: the horizon is <= 3 attacker
//! moves from a zero-clock no-history root, and the only dead position
//! reachable (bare kings after both attacker pieces are captured) contains no
//! mate and therefore yields `None` here as well.
//! Legal move generation is the shared `cozy-chess` generator; it is
//! cross-checked separately against brute-force `is_legal` (see tests).

use cozy_chess::{Board, Color, Move};

fn legal(b: &Board) -> Vec<Move> {
    let mut v = Vec::new();
    b.generate_moves(|ms| {
        for m in ms {
            v.push(m);
        }
        false
    });
    v
}

fn after(b: &Board, m: Move) -> Board {
    let mut c = b.clone();
    c.try_play(m).expect("generated move must be legal");
    c
}

/// Attacker to move: minimal number of attacker moves (<= `limit`) forcing mate.
pub fn att_value(b: &Board, att: Color, limit: u8) -> Option<u8> {
    let mut limit = limit;
    let mut best: Option<u8> = None;
    for m in legal(b) {
        if limit == 0 {
            break;
        }
        let c = after(b, m);
        let replies = legal(&c);
        let cand = if replies.is_empty() {
            if c.checkers().is_empty() { None } else { Some(1) }
        } else {
            def_value(&c, att, limit - 1).map(|v| v + 1)
        };
        if let Some(v) = cand {
            if best.is_none_or(|x| v < x) {
                best = Some(v);
                limit = v - 1;
            }
        }
    }
    best
}

/// Defender to move at a position with legal moves: worst case (over defender
/// replies) minimal attacker moves needed; `None` if some reply escapes `limit`.
pub fn def_value(b: &Board, att: Color, limit: u8) -> Option<u8> {
    if limit == 0 {
        return None;
    }
    let mut worst = 0u8;
    for m in legal(b) {
        let c = after(b, m);
        if legal(&c).is_empty() {
            return None; // stalemate (or, impossibly here, mate of the attacker)
        }
        let v = att_value(&c, att, limit)?;
        worst = worst.max(v);
    }
    Some(worst)
}

/// Exact: does the attacker force mate within `n` from defender-to-move `child`?
pub fn child_target(child: &Board, att: Color, n: u8) -> bool {
    def_value(child, att, n).is_some()
}

/// Minimal forced-mate depth of an attacker-to-move root up to `max`.
pub fn root_min_depth(root: &Board, att: Color, max: u8) -> Option<u8> {
    att_value(root, att, max)
}
