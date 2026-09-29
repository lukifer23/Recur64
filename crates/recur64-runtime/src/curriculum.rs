//! Endgame curriculum (D51): a fixed share of self-play games starts from a
//! generated, simplified position in which one side has decisive material.
//!
//! Only the *start position* changes. Moves, search targets and outcome
//! labels still come from the network's own play under the rules profile: no
//! tablebase, engine label, material reward or adjudication is involved. A
//! won-material start whose game is drawn is labelled a draw, exactly as in
//! a standard-start game.
//!
//! Positions are generated deterministically from the run seed and the global
//! game id, so every curriculum game is reproducible and its start FEN is
//! recorded in the replay like any other game.

use serde::{Deserialize, Serialize};

use recur64_core::{GameState, Square};
use recur64_search::Rng;

/// Recorded in the scientific identity when the curriculum is enabled.
pub const ENDGAME_CURRICULUM_VERSION: &str = "endgame_curriculum_v1";

/// Mixed into the game seed so start-position generation uses its own stream.
const CURRICULUM_SEED_SALT: u64 = 0xD51E_0D6A_3E5C_0001;

/// Rejection-sampling bound per position; a valid family succeeds in a
/// handful of attempts, so hitting it indicates a configuration error.
const MAX_ATTEMPTS: u32 = 10_000;

/// Curriculum configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EndgameCurriculum {
    /// Share of self-play games started from a generated endgame, in (0, 1].
    /// Curriculum games are interleaved evenly by global game id.
    pub fraction: f64,
    /// Material families `K<pieces>vK<pieces>`, stronger side first, e.g.
    /// `KQvK`, `KRvK`, `KRPvKP`. One is drawn uniformly per game.
    pub families: Vec<String>,
}

/// Non-king pieces of each side, as FEN letters (uppercase).
#[derive(Debug, Clone, PartialEq, Eq)]
struct Family {
    strong: Vec<char>,
    weak: Vec<char>,
}

fn parse_side(s: &str, family: &str) -> anyhow::Result<Vec<char>> {
    let rest = s
        .strip_prefix('K')
        .ok_or_else(|| anyhow::anyhow!("family {family:?}: each side starts with K"))?;
    rest.chars()
        .map(|c| match c {
            'Q' | 'R' | 'B' | 'N' | 'P' => Ok(c),
            _ => anyhow::bail!("family {family:?}: unknown piece {c:?} (use QRBNP)"),
        })
        .collect()
}

fn parse_family(family: &str) -> anyhow::Result<Family> {
    let (strong, weak) = family
        .split_once('v')
        .ok_or_else(|| anyhow::anyhow!("family {family:?}: expected K<pieces>vK<pieces>"))?;
    let f = Family {
        strong: parse_side(strong, family)?,
        weak: parse_side(weak, family)?,
    };
    let value = |pieces: &[char]| -> i32 {
        pieces
            .iter()
            .map(|c| match c {
                'Q' => 9,
                'R' => 5,
                'B' | 'N' => 3,
                _ => 1,
            })
            .sum()
    };
    // Mating material the stronger side can force with: a queen, a rook or a
    // pawn (promotion). Lone minors cannot, so they are not a curriculum.
    anyhow::ensure!(
        f.strong.iter().any(|c| matches!(c, 'Q' | 'R' | 'P')),
        "family {family:?}: the stronger side needs a queen, rook or pawn"
    );
    anyhow::ensure!(
        value(&f.strong) > value(&f.weak),
        "family {family:?}: the first side must have more material"
    );
    Ok(f)
}

impl EndgameCurriculum {
    /// Refuse an unusable configuration before any game is played.
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.fraction > 0.0 && self.fraction <= 1.0,
            "endgame_curriculum.fraction must be in (0, 1]"
        );
        anyhow::ensure!(
            !self.families.is_empty(),
            "endgame_curriculum.families must not be empty"
        );
        for f in &self.families {
            parse_family(f)?;
        }
        Ok(())
    }

    /// Whether global game `game_id` starts from a generated endgame. Exactly
    /// `floor(n * fraction)` of the first `n` games do, spread evenly.
    pub fn is_curriculum_game(&self, game_id: u64) -> bool {
        let f = self.fraction;
        ((game_id + 1) as f64 * f).floor() > (game_id as f64 * f).floor()
    }

    /// The generated start position for curriculum game `game_id`.
    pub fn start_position(&self, seed: u64, game_id: u64) -> anyhow::Result<GameState> {
        let families = self
            .families
            .iter()
            .map(|f| parse_family(f))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let mut rng = Rng::new((seed ^ CURRICULUM_SEED_SALT).wrapping_add(game_id));
        for _ in 0..MAX_ATTEMPTS {
            let family = &families[below(&mut rng, families.len())];
            if let Some(state) = try_position(family, &mut rng) {
                return Ok(state);
            }
        }
        anyhow::bail!("no valid curriculum position after {MAX_ATTEMPTS} attempts")
    }
}

fn below(rng: &mut Rng, n: usize) -> usize {
    (rng.next_u64() % n as u64) as usize
}

/// One random placement; `None` if it is not a legal, quiet, ongoing start.
fn try_position(family: &Family, rng: &mut Rng) -> Option<GameState> {
    let strong_is_white = rng.next_u64() & 1 == 0;
    let white_to_move = rng.next_u64() & 1 == 0;
    let mut board: [Option<char>; 64] = [None; 64];
    let mut place = |piece: char, white: bool, rng: &mut Rng| -> usize {
        // Pawns never stand on the first or last rank.
        let sq = loop {
            let sq = below(rng, 64);
            let rank = sq / 8;
            if board[sq].is_none() && (piece != 'P' || (1..=6).contains(&rank)) {
                break sq;
            }
        };
        board[sq] = Some(if white {
            piece
        } else {
            piece.to_ascii_lowercase()
        });
        sq
    };
    let strong_king = place('K', strong_is_white, rng);
    let weak_king = place('K', !strong_is_white, rng);
    let (sf, sr) = (strong_king % 8, strong_king / 8);
    let (wf, wr) = (weak_king % 8, weak_king / 8);
    if sf.abs_diff(wf) <= 1 && sr.abs_diff(wr) <= 1 {
        return None;
    }
    for &p in &family.strong {
        place(p, strong_is_white, rng);
    }
    for &p in &family.weak {
        place(p, !strong_is_white, rng);
    }
    let fen = format!(
        "{} {} - - 0 1",
        placement_fen(&board),
        if white_to_move { 'w' } else { 'b' }
    );
    // Parsing rejects illegal placements (e.g. the side not to move in check).
    let state = GameState::from_fen(&fen).ok()?;
    if state.termination().is_some() || !state.board().checkers().is_empty() {
        return None;
    }
    // A quiet start: the side to move cannot capture, so no game begins by
    // winning or losing material for free.
    let occupied = |sq: Square| board[sq as usize].is_some();
    if state
        .legal_standard_moves()
        .iter()
        .any(|mv| occupied(mv.to))
    {
        return None;
    }
    Some(state)
}

/// FEN piece placement for a board indexed a1 = 0 .. h8 = 63.
fn placement_fen(board: &[Option<char>; 64]) -> String {
    let mut out = String::new();
    for rank in (0..8).rev() {
        let mut empty = 0;
        for file in 0..8 {
            match board[rank * 8 + file] {
                Some(c) => {
                    if empty > 0 {
                        out.push_str(&empty.to_string());
                        empty = 0;
                    }
                    out.push(c);
                }
                None => empty += 1,
            }
        }
        if empty > 0 {
            out.push_str(&empty.to_string());
        }
        if rank > 0 {
            out.push('/');
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use recur64_core::material_balance;

    fn curriculum(fraction: f64, families: &[&str]) -> EndgameCurriculum {
        EndgameCurriculum {
            fraction,
            families: families.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn families_parse_and_invalid_ones_are_refused() {
        assert!(
            curriculum(0.25, &["KQvK", "KRvK", "KRPvKP", "KQvKP"])
                .validate()
                .is_ok()
        );
        for bad in ["KBvK", "KNvK", "QvK", "KQK", "KXvK", "KRvKQ", "KPvKP"] {
            assert!(curriculum(0.25, &[bad]).validate().is_err(), "{bad}");
        }
        assert!(curriculum(0.0, &["KQvK"]).validate().is_err());
        assert!(curriculum(1.5, &["KQvK"]).validate().is_err());
        assert!(curriculum(0.5, &[]).validate().is_err());
    }

    #[test]
    fn curriculum_games_are_an_exact_even_share() {
        let c = curriculum(0.25, &["KQvK"]);
        let picked: Vec<u64> = (0..64).filter(|&i| c.is_curriculum_game(i)).collect();
        assert_eq!(picked.len(), 16);
        assert_eq!(&picked[..4], &[3, 7, 11, 15]);
        let all = curriculum(1.0, &["KQvK"]);
        assert!((0..64).all(|i| all.is_curriculum_game(i)));
    }

    #[test]
    fn generated_positions_are_legal_quiet_decisive_and_reproducible() {
        let c = curriculum(1.0, &["KQvK", "KRvK", "KRPvKP"]);
        let mut white_strong = 0;
        let mut white_to_move = 0;
        for id in 0..200 {
            let s = c.start_position(1, id).unwrap();
            assert_eq!(s.to_fen(), c.start_position(1, id).unwrap().to_fen());
            assert!(s.termination().is_none(), "{}", s.to_fen());
            assert!(s.board().checkers().is_empty(), "{}", s.to_fen());
            let balance = material_balance(s.board());
            assert!(balance.abs() >= 5, "decisive material: {}", s.to_fen());
            white_strong += usize::from(balance > 0);
            white_to_move += usize::from(s.side_to_move() == recur64_core::Color::White);
            // Round-trips through FEN (the replay stores the start FEN).
            assert_eq!(
                GameState::from_fen(&s.to_fen()).unwrap().to_fen(),
                s.to_fen()
            );
        }
        // Both colors and both sides to move occur.
        assert!((60..140).contains(&white_strong), "{white_strong}");
        assert!((60..140).contains(&white_to_move), "{white_to_move}");
        assert_ne!(
            c.start_position(1, 0).unwrap().to_fen(),
            c.start_position(2, 0).unwrap().to_fen(),
            "the run seed changes the positions"
        );
    }
}
