//! Differential testing.
//!
//! Mandatory: seeded random games checked against Recur64's own invariants and
//! (via `tests/perft.rs`) published CPW counts.
//!
//! Optional: an independent oracle (`shakmaty`, GPL-3.0, dev/test only) behind
//! the `oracle` feature. It compares legal move sets and mate/stalemate only;
//! draw adjudication is intentionally excluded because Recur64's auto-claim
//! profile differs from a library default.

use recur64_core::{GameState, StandardMove};

struct SplitMix64(u64);
impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

fn step(g: &mut GameState, rng: &mut SplitMix64) -> bool {
    let actions = g.legal_actions();
    if actions.is_empty() {
        return false;
    }
    let id = actions[(rng.next_u64() as usize) % actions.len()];
    let (f, t, pr) = id.to_physical(g.perspective());
    let promo = if pr.is_none() { None } else { Some(pr) };
    g.apply(StandardMove::new(f, t, promo)).unwrap();
    true
}

#[cfg(feature = "oracle")]
mod oracle {
    use super::*;
    use std::collections::BTreeSet;

    use shakmaty::fen::Fen;
    use shakmaty::uci::UciMove;
    use shakmaty::{CastlingMode, Chess, Position};

    fn shakmaty_legals(fen: &str) -> Option<BTreeSet<String>> {
        let setup: Fen = fen.parse().ok()?;
        let pos: Chess = setup.into_position(CastlingMode::Standard).ok()?;
        Some(
            pos.legal_moves()
                .iter()
                .map(|m| UciMove::from_standard(*m).to_string())
                .collect(),
        )
    }

    fn shakmaty_mate_stalemate(fen: &str) -> Option<(bool, bool)> {
        let setup: Fen = fen.parse().ok()?;
        let pos: Chess = setup.into_position(CastlingMode::Standard).ok()?;
        Some((pos.is_checkmate(), pos.is_stalemate()))
    }

    #[test]
    fn legal_moves_match_shakmaty() {
        for seed in 0..200u64 {
            let mut rng = SplitMix64::new(seed ^ 0x9E37_79B9_7F4A_7C15);
            let mut g = GameState::startpos();
            for _ in 0..60 {
                if g.is_terminal() {
                    break;
                }
                let fen = g.to_fen();
                let ours: BTreeSet<String> = g
                    .legal_standard_moves()
                    .iter()
                    .map(|m| m.to_uci())
                    .collect();
                if let Some(theirs) = shakmaty_legals(&fen) {
                    assert_eq!(ours, theirs, "legal move mismatch: seed={seed} fen={fen}");
                }
                if let Some((mate, stale)) = shakmaty_mate_stalemate(&fen) {
                    let ours_mate = g.termination() == Some(recur64_core::Termination::Checkmate);
                    let ours_stale = g.termination() == Some(recur64_core::Termination::Stalemate);
                    assert_eq!(ours_mate, mate, "mate mismatch: seed={seed} fen={fen}");
                    assert_eq!(
                        ours_stale, stale,
                        "stalemate mismatch: seed={seed} fen={fen}"
                    );
                }
                if !step(&mut g, &mut rng) {
                    break;
                }
            }
        }
    }
}

#[test]
fn random_game_replay_is_reproducible() {
    // Same seed must reproduce the same game and final FEN.
    let play = |seed: u64| {
        let mut rng = SplitMix64::new(seed);
        let mut g = GameState::startpos();
        for _ in 0..50 {
            if !step(&mut g, &mut rng) {
                break;
            }
        }
        g.to_fen()
    };
    for seed in 0..50u64 {
        assert_eq!(play(seed), play(seed), "seed {seed} not reproducible");
    }
}
