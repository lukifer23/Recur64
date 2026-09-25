//! Internal systems arena: candidate vs reference, paired colors.
//!
//! This is a systems comparison, not an Elo claim. Both sides use the same rules
//! profile, search budget, and recurrence; only the evaluator differs. By
//! default moves are chosen deterministically (temperature 0). An optional
//! sampled opening phase (`sample_plies`) and root noise diversify games that
//! would otherwise collapse into repetition between near-identical networks;
//! every game is still reproducible from the recorded seed.

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
    /// Games played at once. Results are aggregated in game-index order, so
    /// the report does not depend on it (1 = sequential).
    pub concurrency: usize,
    /// Sample moves from visit counts (temperature 1) for this many plies
    /// after the opening position, then play argmax. `None` = argmax from
    /// the first ply (the original contract).
    pub sample_plies: Option<u32>,
    /// Root Dirichlet noise for arena search (`0.0` = none, the original
    /// contract).
    pub root_dirichlet_alpha: f32,
    pub root_dirichlet_epsilon: f32,
    /// Soft wall-clock deadline (D38): no game starts after it.
    pub deadline: Option<std::time::Instant>,
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
            concurrency: 1,
            sample_plies: None,
            root_dirichlet_alpha: 0.3,
            root_dirichlet_epsilon: 0.0,
            deadline: None,
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
    /// Reported as 0.5 when nothing was decided; read `informative` first.
    pub candidate_score: f64,
    /// Games won by either side.
    pub decisive_games: u32,
    /// `decisive_games > 0`. A 0.5 from all draws or all truncations is not a
    /// measured tie.
    pub informative: bool,
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
        // Default contract: argmax from ply 0, no noise. With `sample_plies`
        // the first plies sample at temperature 1, then argmax.
        temperature: if cfg.sample_plies.is_some() { 1.0 } else { 0.0 },
        ply_cap: cfg.ply_cap,
        recurrence: cfg.recurrence,
        argmax_after_ply: cfg.sample_plies,
        root_dirichlet_alpha: cfg.root_dirichlet_alpha,
        root_dirichlet_epsilon: cfg.root_dirichlet_epsilon,
    };

    let openings: Vec<String> = if cfg.openings.is_empty() {
        vec![GameState::startpos().to_fen()]
    } else {
        cfg.openings.clone()
    };

    let play_one = |i: u32| -> Result<recur64_search::SelfPlayGame, EvalError> {
        let candidate_is_white = i.is_multiple_of(2);
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
        let start = GameState::from_fen(opening)
            .map_err(|e| EvalError::Invalid(format!("invalid opening FEN: {e}")))?;
        play_game_from(&router, &sp, &mut Rng::new(seed), start)
    };
    let games = play_indexed_until(cfg.games, cfg.concurrency, cfg.deadline, play_one)?;

    let mut candidate_wins = 0u32;
    let mut reference_wins = 0u32;
    let mut draws = 0u32;
    let mut truncated = 0u32;
    let mut scores: Vec<f64> = Vec::new();
    let mut terminations: BTreeMap<String, u32> = BTreeMap::new();

    for (i, game) in games.into_iter().enumerate() {
        let candidate_is_white = i % 2 == 0;
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
        decisive_games: candidate_wins + reference_wins,
        informative: candidate_wins + reference_wins > 0,
        score_ci_low: ci_low,
        score_ci_high: ci_high,
        opening_count: openings.len(),
        terminations,
        model_reference: reference_id.to_string(),
        model_candidate: candidate_id.to_string(),
    })
}

/// Play `games` independent games on up to `concurrency` threads and return
/// them in game-index order. The first error aborts the evaluation.
pub fn play_indexed<T, F>(games: u32, concurrency: usize, play: F) -> Result<Vec<T>, EvalError>
where
    T: Send,
    F: Fn(u32) -> Result<T, EvalError> + Sync,
{
    play_indexed_until(games, concurrency, None, play)
}

/// [`play_indexed`] with a soft deadline (D38): once `deadline` passes, no
/// new game starts; games already in flight finish. If any game did not run,
/// the whole evaluation is [`EvalError::DeadlineExceeded`] (no partial result).
pub fn play_indexed_until<T, F>(
    games: u32,
    concurrency: usize,
    deadline: Option<std::time::Instant>,
    play: F,
) -> Result<Vec<T>, EvalError>
where
    T: Send,
    F: Fn(u32) -> Result<T, EvalError> + Sync,
{
    let expired = || deadline.is_some_and(|d| std::time::Instant::now() > d);
    let incomplete = |done: usize| {
        EvalError::DeadlineExceeded(format!(
            "{done} of {games} evaluation games completed before the deadline"
        ))
    };
    let threads = concurrency.clamp(1, games.max(1) as usize);
    if threads == 1 {
        let mut out = Vec::with_capacity(games as usize);
        for i in 0..games {
            if expired() {
                return Err(incomplete(out.len()));
            }
            out.push(play(i)?);
        }
        return Ok(out);
    }
    let next = std::sync::atomic::AtomicU32::new(0);
    let mut slots: Vec<Option<T>> = (0..games).map(|_| None).collect();
    let mut first_error = None;
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(|| {
                    let mut out = Vec::new();
                    loop {
                        if expired() {
                            break;
                        }
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        if i >= games {
                            break;
                        }
                        let result = play(i);
                        let failed = result.is_err();
                        out.push((i, result));
                        if failed {
                            next.store(games, std::sync::atomic::Ordering::SeqCst);
                            break;
                        }
                    }
                    out
                })
            })
            .collect();
        for h in handles {
            match h.join() {
                Ok(results) => {
                    for (i, r) in results {
                        match r {
                            Ok(v) => slots[i as usize] = Some(v),
                            Err(e) => {
                                first_error.get_or_insert(e);
                            }
                        }
                    }
                }
                Err(_) => {
                    first_error
                        .get_or_insert(EvalError::Backend("evaluation worker panicked".into()));
                }
            }
        }
    });
    if let Some(e) = first_error {
        return Err(e);
    }
    let done = slots.iter().filter(|s| s.is_some()).count();
    if done < games as usize && deadline.is_some() {
        return Err(incomplete(done));
    }
    slots
        .into_iter()
        .enumerate()
        .map(|(i, v)| {
            v.ok_or_else(|| EvalError::Backend(format!("evaluation game {i} did not run")))
        })
        .collect()
}
