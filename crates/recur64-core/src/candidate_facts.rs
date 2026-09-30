//! `CandidateFactsV1`: exact, deterministic one-ply facts about each LEGAL MOVE.
//!
//! Each fact is computed by playing the move on a copy of the authoritative
//! position and inspecting the result. There is no engine evaluation, no search
//! beyond the opponent's immediate legal replies, and no external data. Facts are
//! computed from the [`GameState`] (never from a lossy observation).
//!
//! Fields per candidate, all in `0..=1`, in this order:
//!
//! | # | field | meaning |
//! |---|---|---|
//! | 0 | `mate` | the move gives checkmate |
//! | 1 | `check` | the move gives check |
//! | 2 | `capture` | the move captures material (incl. en passant) |
//! | 3 | `captured_value` | material captured / 9 (P1 N3 B3 R5 Q9) |
//! | 4 | `attacked_after` | the opponent has a legal move landing on the destination square |
//! | 5 | `promotion` | the move promotes |
//! | 6 | `promotion_gain` | own material gained by the move / 8 |
//! | 7 | `stalemate` | the move stalemates the opponent |
//!
//! Row order always equals [`GameState::legal_actions`]: a checkmated or
//! stalemated position has no legal moves and so no rows. (States that are
//! terminal only administratively, e.g. by the ply cap, still list their legal
//! moves; such states bypass the network.) `attacked_after` is false when the
//! position after the move is terminal for ANY reason (the same gating
//! `GameState::apply` applies), so repetition history and the ply cap enter only
//! through that terminal test.

use crate::rules::classify;
use crate::{Board, GameState, StandardMove, Termination, material};

/// Fields per candidate.
pub const CANDIDATE_FACT_FIELDS: usize = 8;
/// Version of the field layout and semantics above.
pub const CANDIDATE_FACTS_VERSION: u32 = 1;

/// One candidate's facts.
pub type CandidateFactsV1 = [f32; CANDIDATE_FACT_FIELDS];

/// Facts for every legal move of `state`, in legal-action order.
pub fn candidate_facts(state: &GameState) -> Vec<CandidateFactsV1> {
    let mover = state.side_to_move();
    let before: &Board = state.board();
    let own_before = material(before, mover);
    let opp_before = material(before, !mover);
    let perspective = state.perspective();
    let ply_after = state.ply() + 1;
    let history = state.history();

    let actions = state.legal_actions();
    let mut out = Vec::with_capacity(actions.len());
    for id in actions {
        let (from, to, promo) = id.to_physical(perspective);
        let promotion = if promo.is_none() { None } else { Some(promo) };
        let cozy = StandardMove::new(from, to, promotion)
            .to_cozy(before)
            .expect("a legal action converts to a cozy move");
        let mut after = before.clone();
        after.play_unchecked(cozy);

        let own_after = material(&after, mover);
        let opp_after = material(&after, !mover);
        let captured = (opp_before - opp_after).max(0);
        let gain = (own_after - own_before).max(0);

        // The same terminal classification GameState::apply performs; the
        // repetition count includes the new occurrence.
        let repetition = 1 + history.iter().filter(|p| after.same_position(p)).count() as u32;
        let termination = classify(&after, repetition, ply_after, state.max_plies());
        let mate = termination == Some(Termination::Checkmate);
        let stalemate = termination == Some(Termination::Stalemate);
        let check = !after.checkers().is_empty();

        let attacked = termination.is_none() && {
            let mut hit = false;
            after.generate_moves(|moves| {
                for reply in moves {
                    let std = StandardMove::from_cozy(&after, reply)
                        .expect("cozy move must be well-formed");
                    if std.to == to {
                        hit = true;
                        return true;
                    }
                }
                false
            });
            hit
        };

        out.push([
            f32::from(u8::from(mate)),
            f32::from(u8::from(check)),
            f32::from(u8::from(captured > 0)),
            captured as f32 / 9.0,
            f32::from(u8::from(attacked)),
            f32::from(u8::from(promotion.is_some())),
            gain as f32 / 8.0,
            f32::from(u8::from(stalemate)),
        ]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Row of the move `from`->`to`; among promotions the largest-gain one.
    fn row(state: &GameState, uci: &str) -> CandidateFactsV1 {
        let f = candidate_facts(state);
        let p = state.perspective();
        let want = uci.trim_end_matches('q');
        let mut best: Option<CandidateFactsV1> = None;
        for (i, id) in state.legal_actions().iter().enumerate() {
            let (from, to, _) = id.to_physical(p);
            if format!("{from}{to}").to_lowercase() == want && best.is_none_or(|b| f[i][6] > b[6]) {
                best = Some(f[i]);
            }
        }
        best.unwrap_or_else(|| panic!("move {uci} not legal"))
    }

    #[test]
    fn back_rank_mate_is_flagged_on_exactly_the_mating_move() {
        let s = GameState::from_fen("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1").unwrap();
        let mating = row(&s, "a1a8");
        assert_eq!((mating[0], mating[1]), (1.0, 1.0));
        assert_eq!(row(&s, "h1g1")[0], 0.0);
        assert_eq!(
            candidate_facts(&s).iter().filter(|r| r[0] == 1.0).count(),
            1
        );
    }

    #[test]
    fn captures_report_the_material_taken() {
        let s = GameState::from_fen("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1").unwrap();
        let take = row(&s, "d1d5");
        assert_eq!(take[2], 1.0);
        assert!((take[3] - 1.0).abs() < 1e-6);
        let quiet = row(&s, "e1e2");
        assert_eq!((quiet[2], quiet[3]), (0.0, 0.0));
    }

    #[test]
    fn en_passant_counts_as_a_pawn_capture() {
        let s = GameState::from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1").unwrap();
        let ep = row(&s, "e5d6");
        assert_eq!(ep[2], 1.0);
        assert!((ep[3] - 1.0 / 9.0).abs() < 1e-6);
    }

    #[test]
    fn promotion_and_attacked_destination_are_reported() {
        let s = GameState::from_fen("7k/4P3/8/8/8/8/8/K7 w - - 0 1").unwrap();
        let promote = row(&s, "e7e8q");
        assert_eq!(promote[5], 1.0);
        assert!((promote[6] - 1.0).abs() < 1e-6, "pawn to queen = +8 / 8");
        assert_eq!(promote[1], 1.0);
        let s = GameState::from_fen("4k3/8/8/8/8/8/r7/N3K3 w - - 0 1").unwrap();
        assert_eq!(row(&s, "a1b3")[4], 0.0);
        let s = GameState::from_fen("4k3/8/8/8/8/8/1r6/N3K3 w - - 0 1").unwrap();
        assert_eq!(row(&s, "a1b3")[4], 1.0);
    }

    #[test]
    fn stalemating_moves_are_flagged() {
        let s = GameState::from_fen("7k/8/6Q1/8/8/8/8/K7 w - - 0 1").unwrap();
        assert!(candidate_facts(&s).iter().any(|r| r[7] == 1.0));
    }

    #[test]
    fn rows_follow_legal_action_order_and_terminals_have_none() {
        let s = GameState::startpos();
        assert_eq!(candidate_facts(&s).len(), s.legal_actions().len());
        assert_eq!(candidate_facts(&s), candidate_facts(&s));
        let mated = GameState::from_fen("R5k1/5ppp/8/8/8/8/8/7K b - - 0 1").unwrap();
        assert!(mated.is_terminal());
        assert!(candidate_facts(&mated).is_empty());
    }

    #[test]
    fn all_values_are_in_unit_range() {
        for fen in [
            "r3k2r/pppq1ppp/2n1bn2/3pp3/3PP3/2N1BN2/PPPQ1PPP/R3K2R w KQkq - 0 1",
            "4k3/1P6/8/8/8/8/8/4K3 w - - 0 1",
        ] {
            let s = GameState::from_fen(fen).unwrap();
            for r in candidate_facts(&s) {
                assert!(r.iter().all(|v| (0.0..=1.0).contains(v)), "{r:?}");
            }
        }
    }
}
