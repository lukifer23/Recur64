//! Replay audit: reconstruct every game and verify integrity, legality,
//! probability alignment, outcome perspective, and provenance.

use std::collections::{BTreeSet, HashSet};
use std::path::Path;

use recur64_core::{Color, GameState, StandardMove, Termination};

use super::reader::ReplayReader;
use super::schema::{GameRecord, action_from_index, termination_from_label};

/// Result of an audit. `errors` is empty iff the replay is clean.
#[derive(Debug, Clone, serde::Serialize)]
pub struct AuditReport {
    pub shards: usize,
    pub games: usize,
    pub plies: u64,
    pub errors: Vec<String>,
}

impl AuditReport {
    pub fn ok(&self) -> bool {
        self.errors.is_empty()
    }
}

fn is_result(term: Termination) -> bool {
    matches!(
        term,
        Termination::Checkmate
            | Termination::Stalemate
            | Termination::InsufficientMaterial
            | Termination::ThreefoldRepetition
            | Termination::FiftyMoveRule
    )
}

fn audit_game(g: &GameRecord, errors: &mut Vec<String>, plies: &mut u64) {
    let tag = |m: String| format!("game {}: {m}", g.game_id);

    if g.search.simulations == 0 {
        errors.push(tag("search budget is zero".into()));
    }
    if g.plies.is_empty() {
        errors.push(tag("game has no plies".into()));
        return;
    }

    let Some(term) = termination_from_label(&g.termination) else {
        errors.push(tag(format!("unknown termination {:?}", g.termination)));
        return;
    };
    if is_result(term) && g.outcome.is_none() {
        errors.push(tag("result game has no outcome".into()));
    }
    if !is_result(term) && g.outcome.is_some() {
        errors.push(tag("truncated/aborted game must not have an outcome".into()));
    }

    let mut state = match GameState::from_fen(&g.start_fen) {
        Ok(s) => s,
        Err(e) => {
            errors.push(tag(format!("bad start FEN: {e}")));
            return;
        }
    };

    for (i, ply) in g.plies.iter().enumerate() {
        *plies += 1;
        let legal = state.legal_actions();
        let legal_set: BTreeSet<u32> = legal.iter().map(|a| a.index()).collect();

        if !legal_set.contains(&(ply.selected as u32)) {
            errors.push(tag(format!("ply {i}: selected action not legal")));
            return;
        }
        if ply.target.is_empty() {
            errors.push(tag(format!("ply {i}: empty target")));
        }
        let mut sum = 0.0f32;
        for (a, p) in &ply.target {
            if !legal_set.contains(&(*a as u32)) {
                errors.push(tag(format!("ply {i}: target action {a} not legal")));
            }
            if !p.is_finite() || *p < 0.0 {
                errors.push(tag(format!("ply {i}: bad target prob {p}")));
            }
            sum += *p;
        }
        if (sum - 1.0).abs() > 1e-3 {
            errors.push(tag(format!("ply {i}: target sum {sum} != 1")));
        }

        let stm = if state.side_to_move() == Color::White {
            0u8
        } else {
            1u8
        };
        if stm != ply.side_to_move {
            errors.push(tag(format!("ply {i}: side_to_move mismatch")));
        }

        let perspective = state.perspective();
        let Ok(id) = action_from_index(ply.selected) else {
            errors.push(tag(format!("ply {i}: invalid action index")));
            return;
        };
        let (from, to, promo) = id.to_physical(perspective);
        let promotion = if promo.is_none() { None } else { Some(promo) };
        if state.apply(StandardMove::new(from, to, promotion)).is_err() {
            errors.push(tag(format!("ply {i}: recorded move illegal")));
            return;
        }
    }

    if is_result(term) {
        let actual = state.termination();
        if actual != Some(term) {
            errors.push(tag(format!("termination {actual:?} != recorded {term:?}")));
        }
        // Outcome perspective: checkmate loser is the side to move at the end.
        if term == Termination::Checkmate {
            let expected = if state.side_to_move() == Color::White {
                2u8 // black won
            } else {
                0u8 // white won
            };
            if g.outcome != Some(expected) {
                errors.push(tag(format!(
                    "checkmate outcome {:?} != expected {expected}",
                    g.outcome
                )));
            }
        }
        if term != Termination::Checkmate && g.outcome != Some(1) && !is_draw_exception(term) {
            errors.push(tag("non-checkmate result should be a draw".into()));
        }
    }
}

fn is_draw_exception(_t: Termination) -> bool {
    false
}

/// Audit a set of in-memory game records.
pub fn audit_games(games: &[GameRecord]) -> AuditReport {
    let mut errors = Vec::new();
    let mut plies = 0u64;
    let mut seen = HashSet::new();
    for g in games {
        if !seen.insert(g.game_id) {
            errors.push(format!("duplicate game_id {}", g.game_id));
        }
        audit_game(g, &mut errors, &mut plies);
    }
    AuditReport {
        shards: 0,
        games: games.len(),
        plies,
        errors,
    }
}

/// Audit an on-disk replay directory.
pub fn audit_dir(dir: &Path) -> anyhow::Result<AuditReport> {
    let reader = ReplayReader::open(dir)?;
    let mut errors = Vec::new();
    let mut plies = 0u64;
    let mut games = 0usize;
    let mut shards = 0usize;
    let mut seen = HashSet::new();

    for info in &reader.manifest().shards {
        shards += 1;
        match reader.read_shard(&info.file) {
            Ok(shard) => {
                if shard.games.len() as u64 != info.games {
                    errors.push(format!(
                        "{}: shard has {} games, manifest says {}",
                        info.file,
                        shard.games.len(),
                        info.games
                    ));
                }
                for g in &shard.games {
                    if !seen.insert(g.game_id) {
                        errors.push(format!("duplicate game_id {}", g.game_id));
                    }
                    audit_game(g, &mut errors, &mut plies);
                    games += 1;
                }
            }
            Err(e) => errors.push(format!("{}: {e}", info.file)),
        }
    }

    Ok(AuditReport {
        shards,
        games,
        plies,
        errors,
    })
}
