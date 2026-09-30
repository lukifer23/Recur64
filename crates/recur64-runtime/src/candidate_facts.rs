//! `CandidateFactsV1`: exact, deterministic facts about each LEGAL MOVE.
//!
//! `ComputeBankV1` tells the network whether a mate exists; it cannot say which
//! candidate delivers it. This provider closes that gap with a bounded, exact,
//! one-ply computation per legal move: it plays the move on a copy of the
//! position, and reports what happened. There is no engine evaluation, no
//! search beyond the opponent's immediate replies, no hand-tuned scalar and no
//! external data.
//!
//! Fields per candidate (all in `0..=1`, in this order):
//!
//! | # | field | meaning |
//! |---|---|---|
//! | 0 | `mate` | the move gives checkmate |
//! | 1 | `check` | the move gives check |
//! | 2 | `capture` | the move captures material (incl. en passant) |
//! | 3 | `captured_value` | material captured / 9 (P1 N3 B3 R5 Q9) |
//! | 4 | `attacked_after` | the opponent has a legal move landing on the destination square |
//! | 5 | `promotion` | the move promotes |
//! | 6 | `promotion_gain` | own material gained by promoting / 8 |
//! | 7 | `stalemate` | the move stalemates the opponent |
//!
//! Row order equals `GameState::legal_actions()`, the same order the model's
//! candidate tensors use. Padded rows are all zero. Facts depend only on the
//! position and the legal move, and are computed for the side to move; repetition
//! and fifty-move history only matter through the position's own termination
//! rules (`GameState`), which the move application already enforces.

use recur64_core::{Color, GameState, StandardMove, Termination};

pub use recur64_model::experimental::{CANDIDATE_FACT_FIELDS as FIELDS, CANDIDATE_FACTS_VERSION};
fn value(c: char) -> i32 {
    match c.to_ascii_lowercase() {
        'p' => 1,
        'n' | 'b' => 3,
        'r' => 5,
        'q' => 9,
        _ => 0,
    }
}

/// (white material, black material) from a FEN board field.
fn material(fen: &str) -> (i32, i32) {
    let (mut w, mut b) = (0, 0);
    for c in fen.split(' ').next().unwrap_or("").chars() {
        if c.is_ascii_uppercase() {
            w += value(c);
        } else if c.is_ascii_alphabetic() {
            b += value(c);
        }
    }
    (w, b)
}

/// Facts for one position: `legal.len() * FIELDS` values in legal-action order.
pub fn facts_for(state: &GameState) -> Vec<f32> {
    let mover_white = state.side_to_move() == Color::White;
    let (mw, mb) = material(&state.to_fen());
    let (own_before, opp_before) = if mover_white { (mw, mb) } else { (mb, mw) };
    let perspective = state.perspective();
    let mut out = Vec::new();
    for id in state.legal_actions() {
        let (from, to, promo) = id.to_physical(perspective);
        let promotion = if promo.is_none() { None } else { Some(promo) };
        let mut after = state.clone();
        after
            .apply(StandardMove::new(from, to, promotion))
            .expect("a legal action applies");
        let (aw, ab) = material(&after.to_fen());
        let (own_after, opp_after) = if mover_white { (aw, ab) } else { (ab, aw) };
        let captured = (opp_before - opp_after).max(0);
        let gain = (own_after - own_before).max(0);
        let terminal = after.termination();
        let mate = terminal == Some(Termination::Checkmate);
        let stalemate = terminal == Some(Termination::Stalemate);
        let check = !after.board().checkers().is_empty();
        // The opponent can land a legal move on the destination square. Terminal
        // positions have no replies.
        let attacked = !after.is_terminal()
            && after
                .legal_standard_moves()
                .iter()
                .any(|reply| reply.to == to);
        out.extend_from_slice(&[
            f32::from(u8::from(mate)),
            f32::from(u8::from(check)),
            f32::from(u8::from(captured > 0)),
            captured as f32 / 9.0,
            f32::from(u8::from(attacked)),
            f32::from(u8::from(!promo.is_none())),
            gain as f32 / 8.0,
            f32::from(u8::from(stalemate)),
        ]);
    }
    out
}

/// Facts for a batch, padded to `width` candidates: `[b, width, FIELDS]`
/// row-major. `width` must be at least every position's legal count.
pub fn candidate_facts(states: &[GameState], width: usize) -> anyhow::Result<Vec<f32>> {
    let mut out = vec![0.0f32; states.len() * width * FIELDS];
    for (i, s) in states.iter().enumerate() {
        let f = facts_for(s);
        anyhow::ensure!(
            f.len() <= width * FIELDS,
            "position {i} has {} legal moves, wider than the candidate width {width}",
            f.len() / FIELDS
        );
        out[i * width * FIELDS..i * width * FIELDS + f.len()].copy_from_slice(&f);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The row of the move `from`->`to` (e.g. "e7e8"); among promotions the one
    /// with the largest material gain (the queen).
    fn row(state: &GameState, uci_to: &str) -> Vec<f32> {
        let f = facts_for(state);
        let perspective = state.perspective();
        let mut best: Option<Vec<f32>> = None;
        for (i, id) in state.legal_actions().iter().enumerate() {
            let (from, to, _) = id.to_physical(perspective);
            let name = format!("{from}{to}").to_lowercase();
            if name == uci_to.trim_end_matches('q') {
                let r = f[i * FIELDS..(i + 1) * FIELDS].to_vec();
                if best.as_ref().is_none_or(|b| r[6] > b[6]) {
                    best = Some(r);
                }
            }
        }
        best.unwrap_or_else(|| panic!("move {uci_to} not legal"))
    }

    #[test]
    fn a_back_rank_mate_is_flagged_on_exactly_the_mating_move() {
        let s = GameState::from_fen("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1").unwrap();
        let mating = row(&s, "a1a8");
        assert_eq!(mating[0], 1.0, "mate flag");
        assert_eq!(mating[1], 1.0, "a mate is also check");
        let quiet = row(&s, "h1g1");
        assert_eq!(quiet[0], 0.0);
        // Exactly one candidate mates.
        let f = facts_for(&s);
        let mates = f.chunks(FIELDS).filter(|r| r[0] == 1.0).count();
        assert_eq!(mates, 1);
    }

    #[test]
    fn captures_report_the_material_taken() {
        // White rook takes an undefended black queen on d5.
        let s = GameState::from_fen("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1").unwrap();
        let take = row(&s, "d1d5");
        assert_eq!(take[2], 1.0);
        assert!((take[3] - 1.0).abs() < 1e-6, "queen = 9/9");
        let quiet = row(&s, "e1e2");
        assert_eq!(quiet[2], 0.0);
        assert_eq!(quiet[3], 0.0);
    }

    #[test]
    fn promotion_and_attacked_destination_are_reported() {
        let s = GameState::from_fen("7k/4P3/8/8/8/8/8/K7 w - - 0 1").unwrap();
        let promote = row(&s, "e7e8q");
        assert_eq!(promote[5], 1.0);
        assert!((promote[6] - 1.0).abs() < 1e-6, "pawn to queen = +8 / 8");
        assert_eq!(promote[1], 1.0, "e8=Q checks the king on h8");
        // A knight stepping next to a rook that can take it is 'attacked_after'.
        let s = GameState::from_fen("4k3/8/8/8/8/8/r7/N3K3 w - - 0 1").unwrap();
        let hangs = row(&s, "a1b3");
        assert_eq!(hangs[4], 0.0, "b3 is not attacked by the a2 rook");
        let s = GameState::from_fen("4k3/8/8/8/8/8/1r6/N3K3 w - - 0 1").unwrap();
        let hangs = row(&s, "a1b3");
        assert_eq!(hangs[4], 1.0, "the b2 rook can take on b3");
    }

    #[test]
    fn stalemating_moves_are_flagged_and_padding_is_zero() {
        // Qg6-f7 stalemates the black king on h8 (Kh8 has no moves, not in check).
        let s = GameState::from_fen("7k/8/6Q1/8/8/8/8/K7 w - - 0 1").unwrap();
        let f = facts_for(&s);
        assert!(
            f.chunks(FIELDS).any(|r| r[7] == 1.0),
            "some move stalemates"
        );
        let width = s.legal_actions().len() + 5;
        let batch = candidate_facts(&[s], width).unwrap();
        assert!(batch[(width - 5) * FIELDS..].iter().all(|v| *v == 0.0));
        assert!(candidate_facts(&[GameState::startpos()], 3).is_err());
    }

    #[test]
    fn facts_are_deterministic() {
        let s = GameState::startpos();
        assert_eq!(facts_for(&s), facts_for(&s));
        assert_eq!(facts_for(&s).len(), 20 * FIELDS);
    }
}
