//! Internal systems arena: candidate vs reference, paired colors.
//!
//! This is a systems comparison, not an Elo claim. Both sides use the same rules
//! profile, search budget, and recurrence; only the evaluator differs. Moves are
//! chosen deterministically (temperature 0) so results are reproducible from a
//! recorded seed.

use std::collections::BTreeMap;

use recur64_core::{Color, GameState, Outcome};
use recur64_search::{
    EvalError, EvalRequest, EvalResult, Evaluator, Rng, SelfPlayConfig, play_game_from,
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
    /// Frozen opening FENs. Empty means the standard start only.
    pub openings: Vec<String>,
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
            openings: Vec::new(),
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
    /// 95% confidence interval on `candidate_score` (per-game outcomes 1/0.5/0).
    pub score_ci_low: f64,
    pub score_ci_high: f64,
    pub opening_count: usize,
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

    let openings: Vec<String> = if cfg.openings.is_empty() {
        vec![GameState::startpos().to_fen()]
    } else {
        cfg.openings.clone()
    };

    let mut candidate_wins = 0u32;
    let mut reference_wins = 0u32;
    let mut draws = 0u32;
    let mut truncated = 0u32;
    let mut scores: Vec<f64> = Vec::new();
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
        let opening = &openings[(i as usize / 2) % openings.len()];
        let start = GameState::from_fen(opening).unwrap_or_else(|_| GameState::startpos());
        let game = play_game_from(&router, &sp, &mut Rng::new(seed), start)?;
        *terminations
            .entry(game.termination.label().to_string())
            .or_insert(0) += 1;

        match game.outcome {
            None => truncated += 1,
            Some(Outcome::Draw) => {
                draws += 1;
                scores.push(0.5);
            }
            Some(Outcome::Win(winner)) => {
                let candidate_won = (winner == Color::White) == candidate_is_white;
                if candidate_won {
                    candidate_wins += 1;
                    scores.push(1.0);
                } else {
                    reference_wins += 1;
                    scores.push(0.0);
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
    // 95% CI from the per-game score variance.
    let (ci_low, ci_high) = if scores.len() < 2 {
        (candidate_score, candidate_score)
    } else {
        let n = scores.len() as f64;
        let mean = scores.iter().sum::<f64>() / n;
        let var = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let se = (var / n).sqrt();
        ((mean - 1.96 * se).max(0.0), (mean + 1.96 * se).min(1.0))
    };

    Ok(ArenaResult {
        games: cfg.games,
        candidate_wins,
        reference_wins,
        draws,
        truncated,
        candidate_score,
        score_ci_low: ci_low,
        score_ci_high: ci_high,
        opening_count: openings.len(),
        terminations,
        model_reference: reference_id.to_string(),
        model_candidate: candidate_id.to_string(),
    })
}
