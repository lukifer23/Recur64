//! Internal systems arena: candidate vs reference, paired colors.
//!
//! This is a systems comparison, not an Elo claim. Both sides use the same rules
//! profile, search budget, and recurrence; only the evaluator differs. Moves are
//! chosen deterministically (temperature 0) so results are reproducible from a
//! recorded seed.

use std::collections::BTreeMap;

use recur64_core::{Color, Outcome};
use recur64_search::{
    EvalError, EvalRequest, EvalResult, Evaluator, SelfPlayConfig, play_game_seeded,
};

/// Arena configuration.
#[derive(Debug, Clone)]
pub struct ArenaConfig {
    pub games: u32,
    pub simulations: u32,
    pub c_puct: f32,
    pub recurrence: usize,
    pub ply_cap: u32,
    pub seed: u64,
}

impl Default for ArenaConfig {
    fn default() -> Self {
        Self {
            games: 4,
            simulations: 16,
            c_puct: 1.0,
            recurrence: 1,
            ply_cap: 256,
            seed: 0,
        }
    }
}

/// Arena result from the candidate's perspective.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ArenaResult {
    pub games: u32,
    pub candidate_wins: u32,
    pub reference_wins: u32,
    pub draws: u32,
    pub truncated: u32,
    /// `(candidate_wins + 0.5 * draws) / decided` over non-truncated games.
    pub candidate_score: f64,
    pub terminations: BTreeMap<String, u32>,
    pub model_reference: String,
    pub model_candidate: String,
}

/// Routes each ply to the correct model by side to move.
struct SideRouter<'a> {
    white: &'a dyn Evaluator,
    black: &'a dyn Evaluator,
}

impl Evaluator for SideRouter<'_> {
    fn evaluate(&self, request: EvalRequest<'_>) -> Result<EvalResult, EvalError> {
        match request.side_to_move {
            Color::White => self.white.evaluate(request),
            Color::Black => self.black.evaluate(request),
        }
    }
}

/// Run a paired-color arena between `candidate` and `reference`.
pub fn run_arena(
    reference: &dyn Evaluator,
    candidate: &dyn Evaluator,
    reference_id: &str,
    candidate_id: &str,
    cfg: &ArenaConfig,
) -> Result<ArenaResult, EvalError> {
    let sp = SelfPlayConfig {
        simulations_per_move: cfg.simulations,
        c_puct: cfg.c_puct,
        temperature: 0.0,
        ply_cap: cfg.ply_cap,
        recurrence: cfg.recurrence,
    };

    let mut candidate_wins = 0u32;
    let mut reference_wins = 0u32;
    let mut draws = 0u32;
    let mut truncated = 0u32;
    let mut terminations: BTreeMap<String, u32> = BTreeMap::new();

    for i in 0..cfg.games {
        let candidate_is_white = i % 2 == 0;
        let router = if candidate_is_white {
            SideRouter {
                white: candidate,
                black: reference,
            }
        } else {
            SideRouter {
                white: reference,
                black: candidate,
            }
        };
        let seed = cfg.seed.wrapping_add(i as u64);
        let game = play_game_seeded(&router, &sp, seed)?;
        *terminations
            .entry(game.termination.label().to_string())
            .or_insert(0) += 1;

        match game.outcome {
            None => truncated += 1,
            Some(Outcome::Draw) => draws += 1,
            Some(Outcome::Win(winner)) => {
                let candidate_won = (winner == Color::White) == candidate_is_white;
                if candidate_won {
                    candidate_wins += 1;
                } else {
                    reference_wins += 1;
                }
            }
        }
    }

    let decided = cfg.games.saturating_sub(truncated);
    let candidate_score = if decided == 0 {
        0.5
    } else {
        (candidate_wins as f64 + 0.5 * draws as f64) / decided as f64
    };

    Ok(ArenaResult {
        games: cfg.games,
        candidate_wins,
        reference_wins,
        draws,
        truncated,
        candidate_score,
        terminations,
        model_reference: reference_id.to_string(),
        model_candidate: candidate_id.to_string(),
    })
}
