//! Internal systems arena: candidate vs reference, paired colors.
//!
//! This is a systems comparison, not an Elo claim. Both sides use the same rules
//! profile, search budget, and recurrence; only the evaluator differs. By
//! default moves are chosen deterministically (temperature 0). An optional
//! sampled opening phase (`sample_plies`) and root noise diversify games that
//! would otherwise collapse into repetition between near-identical networks;
//! every game is still reproducible from the recorded seed. The seed policy
//! ([`ArenaRngPolicy`]) decides whether a color-swapped pair shares a stream.

use std::collections::BTreeMap;

use recur64_core::{Color, GameState, Outcome};
use recur64_search::{
    EvalError, EvalRequest, EvalResult, Evaluator, Rng, SelfPlayConfig, play_game_from,
};

/// How each arena game's RNG seed is derived (H3.5B).
///
/// Game `i` plays opening `i / 2`, with colors swapped between `2p` and
/// `2p + 1`. Under a stochastic arena (D45: sampling + root noise) the policy
/// decides whether the two color-swapped games of an opening share one
/// random stream.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArenaRngPolicy {
    /// `seed = base + i`: every game has its own stream (the historical
    /// contract; the pair shares only the opening).
    #[default]
    PerGameV1,
    /// `seed = base + i / 2`: both games of an opening pair share one stream
    /// (common random numbers). Identical deterministic evaluators then play
    /// the same game twice with colors swapped, so the pair scores exactly 0.5.
    PairedCommonV1,
}

impl ArenaRngPolicy {
    pub fn is_default(&self) -> bool {
        *self == Self::PerGameV1
    }
}

/// RNG seed of arena game `index` under `policy`.
pub fn arena_game_seed(policy: ArenaRngPolicy, base: u64, index: u32) -> u64 {
    match policy {
        ArenaRngPolicy::PerGameV1 => base.wrapping_add(index as u64),
        ArenaRngPolicy::PairedCommonV1 => base.wrapping_add((index / 2) as u64),
    }
}

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
    /// Leaves per search round (D47); `1` = original search.
    pub leaves_in_flight: u32,
    /// Per-game seed derivation (H3.5B); the default is the historical one.
    pub rng_policy: ArenaRngPolicy,
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
            leaves_in_flight: 1,
            rng_policy: ArenaRngPolicy::PerGameV1,
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
    /// Seed derivation used (H3.5B). This and the fields below are additive
    /// diagnostics; the per-game fields above are unchanged.
    pub rng_policy: ArenaRngPolicy,
    /// Color-pair diagnostics (the pair is the independent unit).
    pub pairs: ArenaPairDiagnostics,
    /// Per-game records in game-index order.
    pub game_records: Vec<ArenaGameRecord>,
}

/// One arena game, from the candidate's perspective.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct ArenaGameRecord {
    pub index: u32,
    pub pair: u32,
    pub candidate_white: bool,
    pub seed: u64,
    /// 1 / 0.5 / 0, or `None` when truncated.
    pub candidate_score: Option<f64>,
    /// `"white"`, `"black"` or `"draw"`; `None` when truncated.
    pub winner: Option<String>,
    pub termination: String,
    pub plies: usize,
    /// SHA-256 prefix of the start FEN and the selected action sequence.
    pub moves_digest: String,
    /// Final position (diagnostic; e.g. what a truncated game looked like).
    pub final_fen: String,
}

/// Pair-level arena diagnostics (H3.5B). Diagnostic only: promotion still
/// reads the per-game score.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize)]
pub struct ArenaPairDiagnostics {
    /// Opening pairs `(2p, 2p + 1)`, including an unpaired last game.
    pub pairs: u32,
    /// Both games present and neither truncated.
    pub complete_pairs: u32,
    pub pairs_with_truncation: u32,
    /// Complete pairs where both games were draws.
    pub all_draw_pairs: u32,
    /// Complete pairs with one candidate win and one candidate loss.
    pub split_pairs: u32,
    /// Complete pairs with the same board result in both games (same winner
    /// color, or both drawn), i.e. pair score exactly 0.5.
    pub mirrored_pairs: u32,
    /// Pairs whose two games played the identical move sequence.
    pub identical_move_pairs: u32,
    /// Complete-pair candidate score (mean of its two games) -> count, keys
    /// `"0.00"`, `"0.25"`, `"0.50"`, `"0.75"`, `"1.00"`.
    pub pair_score_histogram: BTreeMap<String, u32>,
    /// Mean complete-pair score (0.5 when there are none).
    pub mean_pair_score: f64,
    /// 95% Wald interval over complete-pair scores.
    pub pair_score_ci_low: f64,
    pub pair_score_ci_high: f64,
}

/// `(mean, ci_low, ci_high)`: 95% Wald interval, clamped to `[0, 1]`.
fn mean_ci(scores: &[f64], fallback: f64) -> (f64, f64, f64) {
    if scores.is_empty() {
        return (fallback, fallback, fallback);
    }
    let n = scores.len() as f64;
    let mean = scores.iter().sum::<f64>() / n;
    if scores.len() < 2 {
        return (mean, mean, mean);
    }
    let var = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let se = (var / n).sqrt();
    (
        mean,
        (mean - 1.96 * se).max(0.0),
        (mean + 1.96 * se).min(1.0),
    )
}

/// Pair diagnostics from per-game records in index order.
pub fn pair_diagnostics(records: &[ArenaGameRecord]) -> ArenaPairDiagnostics {
    let mut d = ArenaPairDiagnostics::default();
    let mut pair_scores = Vec::new();
    for chunk in records.chunks(2) {
        d.pairs += 1;
        if chunk.iter().any(|g| g.candidate_score.is_none()) {
            d.pairs_with_truncation += 1;
        }
        let [a, b] = chunk else { continue };
        if a.moves_digest == b.moves_digest {
            d.identical_move_pairs += 1;
        }
        let (Some(sa), Some(sb)) = (a.candidate_score, b.candidate_score) else {
            continue;
        };
        d.complete_pairs += 1;
        if sa == 0.5 && sb == 0.5 {
            d.all_draw_pairs += 1;
        }
        if (sa - sb).abs() == 1.0 {
            d.split_pairs += 1;
        }
        if a.winner == b.winner {
            d.mirrored_pairs += 1;
        }
        let ps = (sa + sb) / 2.0;
        *d.pair_score_histogram
            .entry(format!("{ps:.2}"))
            .or_insert(0) += 1;
        pair_scores.push(ps);
    }
    let (mean, lo, hi) = mean_ci(&pair_scores, 0.5);
    d.mean_pair_score = mean;
    d.pair_score_ci_low = lo;
    d.pair_score_ci_high = hi;
    d
}

/// Replay the selected actions from the start FEN to the final position.
fn final_fen(game: &recur64_search::SelfPlayGame) -> String {
    let replay = || -> Result<String, recur64_core::CoreError> {
        let mut state = GameState::from_fen(&game.start_fen)?;
        for p in &game.plies {
            let (from, to, promo) = p.selected.to_physical(state.perspective());
            let promotion = (!promo.is_none()).then_some(promo);
            state.apply(recur64_core::StandardMove::new(from, to, promotion))?;
        }
        Ok(state.to_fen())
    };
    replay().unwrap_or_else(|e| format!("unreplayable: {e}"))
}

fn moves_digest(game: &recur64_search::SelfPlayGame) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update((game.start_fen.len() as u64).to_le_bytes());
    h.update(game.start_fen.as_bytes());
    for p in &game.plies {
        h.update(p.selected.index().to_le_bytes());
    }
    format!("{:x}", h.finalize())[..16].to_string()
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

    /// Split a multi-leaf round by side so each model still receives its
    /// leaves as one submission; results are returned in request order.
    fn evaluate_many(&self, requests: &[EvalRequest<'_>]) -> Vec<Result<EvalResult, EvalError>> {
        let (white, black): (Vec<_>, Vec<_>) = requests
            .iter()
            .enumerate()
            .partition(|(_, r)| r.side_to_move == Color::White);
        let white_req: Vec<EvalRequest<'_>> = white.iter().map(|(_, r)| **r).collect();
        let black_req: Vec<EvalRequest<'_>> = black.iter().map(|(_, r)| **r).collect();
        let mut out: Vec<Option<Result<EvalResult, EvalError>>> =
            (0..requests.len()).map(|_| None).collect();
        for ((i, _), r) in white.iter().zip(self.white.evaluate_many(&white_req)) {
            out[*i] = Some(r);
        }
        for ((i, _), r) in black.iter().zip(self.black.evaluate_many(&black_req)) {
            out[*i] = Some(r);
        }
        out.into_iter()
            .map(|r| r.unwrap_or_else(|| Err(EvalError::Backend("unrouted request".into()))))
            .collect()
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
        search_leaves_in_flight: cfg.leaves_in_flight,
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
        let seed = arena_game_seed(cfg.rng_policy, cfg.seed, i);
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
    let mut game_records = Vec::with_capacity(games.len());

    for (i, game) in games.into_iter().enumerate() {
        let candidate_is_white = i % 2 == 0;
        *terminations
            .entry(game.termination.label().to_string())
            .or_insert(0) += 1;

        let (score, winner) = match game.outcome {
            None => {
                truncated += 1;
                (None, None)
            }
            Some(Outcome::Draw) => {
                draws += 1;
                scores.push(0.5);
                (Some(0.5), Some("draw"))
            }
            Some(Outcome::Win(winner)) => {
                let candidate_won = (winner == Color::White) == candidate_is_white;
                let s = if candidate_won {
                    candidate_wins += 1;
                    1.0
                } else {
                    reference_wins += 1;
                    0.0
                };
                scores.push(s);
                let color = if winner == Color::White {
                    "white"
                } else {
                    "black"
                };
                (Some(s), Some(color))
            }
        };
        game_records.push(ArenaGameRecord {
            index: i as u32,
            pair: i as u32 / 2,
            candidate_white: candidate_is_white,
            // play_game_from does not stamp the seed; derive it from the policy.
            seed: arena_game_seed(cfg.rng_policy, cfg.seed, i as u32),
            candidate_score: score,
            winner: winner.map(str::to_string),
            termination: game.termination.label().to_string(),
            plies: game.plies.len(),
            moves_digest: moves_digest(&game),
            final_fen: final_fen(&game),
        });
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
        let (_, lo, hi) = mean_ci(&scores, candidate_score);
        (lo, hi)
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
        rng_policy: cfg.rng_policy,
        pairs: pair_diagnostics(&game_records),
        game_records,
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
