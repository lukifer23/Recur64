//! Independent game play using PUCT and a single evaluator.
//!
//! This lives in `recur64-search` (not the runtime) so that the arena can drive
//! games without depending on the Burn-backed runtime. It uses only
//! `recur64-core` and the search itself.

use recur64_core::{ActionId, Color, GameState, Outcome, StandardMove, Termination};

use crate::evaluator::Evaluator;
use crate::game_tree::ChessGame;
use crate::puct::{PuctConfig, RootEdge, search};
use crate::rng::Rng;

/// Game-play configuration.
#[derive(Debug, Clone, Copy)]
pub struct SelfPlayConfig {
    pub simulations_per_move: u32,
    pub c_puct: f32,
    /// `0.0` = deterministic argmax over visits; otherwise sample from visits^(1/T).
    pub temperature: f32,
    pub ply_cap: u32,
    pub recurrence: usize,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        Self {
            simulations_per_move: 16,
            c_puct: 1.0,
            temperature: 1.0,
            ply_cap: 256,
            recurrence: 1,
        }
    }
}

/// One sparse `(action, probability)` entry of a search target.
#[derive(Debug, Clone, PartialEq)]
pub struct TargetEntry {
    pub action: ActionId,
    pub prob: f32,
}

/// One played ply.
#[derive(Debug, Clone)]
pub struct SelfPlayPly {
    pub selected: ActionId,
    pub target: Vec<TargetEntry>,
    pub visits_total: u32,
    pub side_to_move: Color,
}

/// A completed game (pre-serialization).
#[derive(Debug, Clone)]
pub struct SelfPlayGame {
    pub start_fen: String,
    pub plies: Vec<SelfPlayPly>,
    pub termination: Termination,
    /// `None` for truncated/aborted games.
    pub outcome: Option<Outcome>,
    pub seed: u64,
}

fn sparse_target(edges: &[RootEdge<ActionId>], total_visits: u32) -> Vec<TargetEntry> {
    if total_visits > 0 {
        edges
            .iter()
            .filter(|e| e.visits > 0)
            .map(|e| TargetEntry {
                action: e.action,
                prob: e.visits as f32 / total_visits as f32,
            })
            .collect()
    } else {
        let prior_sum: f32 = edges.iter().map(|e| e.prior).sum();
        edges
            .iter()
            .map(|e| TargetEntry {
                action: e.action,
                prob: if prior_sum > 0.0 {
                    e.prior / prior_sum
                } else {
                    1.0 / edges.len().max(1) as f32
                },
            })
            .collect()
    }
}

fn sample_action(edges: &[RootEdge<ActionId>], temperature: f32, rng: &mut Rng) -> ActionId {
    let best = edges
        .iter()
        .max_by(|a, b| {
            a.visits
                .cmp(&b.visits)
                .then_with(|| b.action.cmp(&a.action))
        })
        .expect("non-empty edges");
    if temperature <= 0.0 {
        return best.action;
    }
    let inv_t = 1.0 / temperature as f64;
    let weights: Vec<f64> = edges
        .iter()
        .map(|e| (e.visits as f64).powf(inv_t))
        .collect();
    let sum: f64 = weights.iter().sum();
    if sum.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        return best.action;
    }
    let mut x = rng.next_f64() * sum;
    for (e, w) in edges.iter().zip(weights.iter()) {
        x -= w;
        if x <= 0.0 {
            return e.action;
        }
    }
    best.action
}

/// Play one complete game from the standard start position.
pub fn play_game(
    evaluator: &dyn Evaluator,
    cfg: &SelfPlayConfig,
    rng: &mut Rng,
) -> Result<SelfPlayGame, crate::EvalError> {
    play_game_from(evaluator, cfg, rng, GameState::startpos())
}

/// Play one complete game from an arbitrary start state.
pub fn play_game_from(
    evaluator: &dyn Evaluator,
    cfg: &SelfPlayConfig,
    rng: &mut Rng,
    mut state: GameState,
) -> Result<SelfPlayGame, crate::EvalError> {
    let start_fen = state.to_fen();
    let mut plies = Vec::new();
    let termination;

    loop {
        if let Some(t) = state.termination() {
            termination = t;
            break;
        }
        if state.ply() >= cfg.ply_cap {
            termination = Termination::Truncated;
            break;
        }

        let side_to_move = state.side_to_move();
        let game = ChessGame::new(state.clone(), evaluator);
        let result = search(
            game,
            &PuctConfig {
                c_puct: cfg.c_puct,
                simulations: cfg.simulations_per_move,
            },
        )?;
        if result.edges.is_empty() {
            termination = Termination::Aborted;
            break;
        }
        let target = sparse_target(&result.edges, result.total_visits);
        let selected = sample_action(&result.edges, cfg.temperature, rng);
        plies.push(SelfPlayPly {
            selected,
            target,
            visits_total: result.total_visits,
            side_to_move,
        });

        let perspective = state.perspective();
        let (from, to, promo) = selected.to_physical(perspective);
        let promotion = if promo.is_none() { None } else { Some(promo) };
        state
            .apply(StandardMove::new(from, to, promotion))
            .expect("selected action is legal at this position");
    }

    let outcome = termination.outcome(state.side_to_move());
    Ok(SelfPlayGame {
        start_fen,
        plies,
        termination,
        outcome,
        seed: 0,
    })
}

/// Play one game and attach a seed to the record.
pub fn play_game_seeded(
    evaluator: &dyn Evaluator,
    cfg: &SelfPlayConfig,
    seed: u64,
) -> Result<SelfPlayGame, crate::EvalError> {
    let mut rng = Rng::new(seed);
    let mut g = play_game(evaluator, cfg, &mut rng)?;
    g.seed = seed;
    Ok(g)
}
