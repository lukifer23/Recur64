//! Raw-policy evaluation: games with **no search**.
//!
//! This isolates what the network itself has learned from what search repairs.
//! One side plays the raw policy (sampled or argmax); the other plays uniformly
//! random legal moves. Colors are paired across the opening suite.

use std::collections::BTreeMap;

use recur64_core::{
    ActionId, Color, GameState, Outcome, StandardMove, Termination, encode_observation_v1,
};
use recur64_search::{EvalError, EvalRequest, Evaluator, Rng};

/// Result of a raw-policy match, from the policy's perspective.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RawMatchResult {
    pub games: u32,
    pub policy_wins: u32,
    pub random_wins: u32,
    pub draws: u32,
    pub truncated: u32,
    pub policy_score: f64,
    pub terminations: BTreeMap<String, u32>,
}

fn sample_policy(policy: &[f32], legal: &[ActionId], temperature: f32, rng: &mut Rng) -> ActionId {
    let best = policy
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    if temperature <= 0.0 {
        return legal[best];
    }
    let inv_t = 1.0 / temperature as f64;
    let weights: Vec<f64> = policy
        .iter()
        .map(|p| (*p as f64).max(0.0).powf(inv_t))
        .collect();
    let sum: f64 = weights.iter().sum();
    if sum.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        return legal[best];
    }
    let mut x = rng.next_f64() * sum;
    for (i, w) in weights.iter().enumerate() {
        x -= w;
        if x <= 0.0 {
            return legal[i];
        }
    }
    legal[best]
}

fn play_raw(
    ev: &dyn Evaluator,
    policy_is_white: bool,
    temperature: f32,
    ply_cap: u32,
    rng: &mut Rng,
    mut state: GameState,
) -> Result<(Termination, Option<Outcome>), EvalError> {
    loop {
        if let Some(t) = state.termination() {
            return Ok((t, t.outcome(state.side_to_move())));
        }
        if state.ply() >= ply_cap {
            return Ok((Termination::Truncated, None));
        }
        let legal = state.legal_actions();
        if legal.is_empty() {
            return Ok((Termination::Aborted, None));
        }
        let policy_turn = (state.side_to_move() == Color::White) == policy_is_white;
        let chosen = if policy_turn {
            let obs = encode_observation_v1(&state);
            let r = ev.evaluate(EvalRequest {
                observation: &obs,
                legal: &legal,
                side_to_move: state.side_to_move(),
            })?;
            sample_policy(&r.policy, &legal, temperature, rng)
        } else {
            legal[(rng.next_u64() as usize) % legal.len()]
        };
        let perspective = state.perspective();
        let (from, to, promo) = chosen.to_physical(perspective);
        let promotion = if promo.is_none() { None } else { Some(promo) };
        state
            .apply(StandardMove::new(from, to, promotion))
            .expect("selected action is legal");
    }
}

/// Play `games` raw-policy-vs-random games over the opening suite, paired colors.
pub fn raw_policy_vs_random(
    ev: &dyn Evaluator,
    games: u32,
    temperature: f32,
    ply_cap: u32,
    seed: u64,
    openings: &[String],
) -> Result<RawMatchResult, EvalError> {
    let openings: Vec<String> = if openings.is_empty() {
        vec![GameState::startpos().to_fen()]
    } else {
        openings.to_vec()
    };

    let mut policy_wins = 0u32;
    let mut random_wins = 0u32;
    let mut draws = 0u32;
    let mut truncated = 0u32;
    let mut terminations: BTreeMap<String, u32> = BTreeMap::new();

    for i in 0..games {
        let policy_is_white = i % 2 == 0;
        let opening = &openings[(i as usize / 2) % openings.len()];
        let start = GameState::from_fen(opening).unwrap_or_else(|_| GameState::startpos());
        let mut rng = Rng::new(seed.wrapping_add(i as u64));
        let (term, outcome) = play_raw(ev, policy_is_white, temperature, ply_cap, &mut rng, start)?;
        *terminations.entry(term.label().to_string()).or_insert(0) += 1;
        match outcome {
            None => truncated += 1,
            Some(Outcome::Draw) => draws += 1,
            Some(Outcome::Win(winner)) => {
                if (winner == Color::White) == policy_is_white {
                    policy_wins += 1;
                } else {
                    random_wins += 1;
                }
            }
        }
    }

    let decided = games.saturating_sub(truncated);
    let policy_score = if decided == 0 {
        0.5
    } else {
        (policy_wins as f64 + 0.5 * draws as f64) / decided as f64
    };

    Ok(RawMatchResult {
        games,
        policy_wins,
        random_wins,
        draws,
        truncated,
        policy_score,
        terminations,
    })
}
