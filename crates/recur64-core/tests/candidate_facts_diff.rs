//! Differential test for `CandidateFactsV1`.
//!
//! The production implementation plays each move on a cloned `Board` and
//! re-derives the terminal classification. This test compares it, field by
//! field, against a deliberately slow reference that uses the authoritative
//! `GameState::apply` path and FEN-string material counting (the semantics the
//! HP branch documents), over seeded random games and the perft/edge fixtures.

use recur64_core::fixtures::{EDGE_CASES, PERFT_FIXTURES};
use recur64_core::{
    CANDIDATE_FACT_FIELDS, Color, GameState, StandardMove, Termination, candidate_facts,
};

struct SplitMix64(u64);
impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

fn value(c: char) -> i32 {
    match c.to_ascii_lowercase() {
        'p' => 1,
        'n' | 'b' => 3,
        'r' => 5,
        'q' => 9,
        _ => 0,
    }
}

fn fen_material(fen: &str) -> (i32, i32) {
    let (mut w, mut b) = (0, 0);
    for c in fen.split(' ').next().unwrap().chars() {
        if c.is_ascii_uppercase() {
            w += value(c);
        } else if c.is_ascii_alphabetic() {
            b += value(c);
        }
    }
    (w, b)
}

fn reference(state: &GameState) -> Vec<[f32; CANDIDATE_FACT_FIELDS]> {
    let white = state.side_to_move() == Color::White;
    let (mw, mb) = fen_material(&state.to_fen());
    let (own_before, opp_before) = if white { (mw, mb) } else { (mb, mw) };
    let p = state.perspective();
    let mut out = Vec::new();
    for id in state.legal_actions() {
        let (from, to, promo) = id.to_physical(p);
        let promotion = if promo.is_none() { None } else { Some(promo) };
        let mut after = state.clone();
        after.apply(StandardMove::new(from, to, promotion)).unwrap();
        let (aw, ab) = fen_material(&after.to_fen());
        let (own_after, opp_after) = if white { (aw, ab) } else { (ab, aw) };
        let captured = (opp_before - opp_after).max(0);
        let gain = (own_after - own_before).max(0);
        let t = after.termination();
        let attacked = !after.is_terminal()
            && after
                .legal_standard_moves()
                .iter()
                .any(|reply| reply.to == to);
        out.push([
            f32::from(u8::from(t == Some(Termination::Checkmate))),
            f32::from(u8::from(!after.board().checkers().is_empty())),
            f32::from(u8::from(captured > 0)),
            captured as f32 / 9.0,
            f32::from(u8::from(attacked)),
            f32::from(u8::from(!promo.is_none())),
            gain as f32 / 8.0,
            f32::from(u8::from(t == Some(Termination::Stalemate))),
        ]);
    }
    out
}

fn check(state: &GameState, what: &str) {
    assert_eq!(
        candidate_facts(state),
        reference(state),
        "{what}: {}",
        state.to_fen()
    );
}

#[test]
fn matches_the_reference_over_seeded_random_games() {
    let mut rng = SplitMix64(0xC0FF_EE25);
    let mut positions = 0usize;
    let (mut mates, mut promos, mut caps, mut stales) = (0, 0, 0, 0);
    for game in 0..300 {
        let mut g = GameState::startpos().with_max_plies(if game % 7 == 0 { 40 } else { 300 });
        for _ in 0..160 {
            let actions = g.legal_actions();
            if actions.is_empty() || g.is_terminal() {
                break;
            }
            check(&g, "random game");
            let facts = candidate_facts(&g);
            positions += 1;
            mates += facts.iter().filter(|r| r[0] == 1.0).count();
            promos += facts.iter().filter(|r| r[5] == 1.0).count();
            caps += facts.iter().filter(|r| r[2] == 1.0).count();
            stales += facts.iter().filter(|r| r[7] == 1.0).count();
            let id = actions[(rng.next_u64() as usize) % actions.len()];
            let (f, t, pr) = id.to_physical(g.perspective());
            g.apply(StandardMove::new(f, t, (!pr.is_none()).then_some(pr)))
                .unwrap();
        }
    }
    assert!(positions > 10_000, "fixture too small: {positions}");
    assert!(caps > 1_000, "captures under-covered: {caps}");
    eprintln!(
        "compared {positions} positions; mates {mates} promos {promos} caps {caps} stalemates {stales}"
    );
}

#[test]
fn matches_the_reference_on_perft_and_edge_fixtures() {
    for f in PERFT_FIXTURES {
        check(&GameState::from_fen(f.fen).unwrap(), "perft fixture");
    }
    for e in EDGE_CASES {
        if let Ok(s) = GameState::from_fen(e.fen) {
            check(&s, "edge case");
        }
    }
}

#[test]
fn repetition_history_gates_attacked_after_like_apply() {
    // Shuffle so the third occurrence is reached by a candidate move.
    let mut g = GameState::from_fen("4k3/8/8/8/8/8/R7/4K3 w - - 0 1").unwrap();
    for m in ["a2a3", "e8d8", "a3a2", "d8e8", "a2a3", "e8d8", "a3a2"] {
        check(&g, "shuffle");
        g.apply_uci(m).unwrap();
    }
    check(&g, "shuffle final");
}
