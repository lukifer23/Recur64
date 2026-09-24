//! Independent game play using PUCT and a single evaluator.
//!
//! This lives in `recur64-search` (not the runtime) so that the arena can drive
//! games without depending on the Burn-backed runtime. It uses only
//! `recur64-core` and the search itself.

use recur64_core::{ActionId, Color, GameState, Outcome, StandardMove, Termination};

use crate::evaluator::Evaluator;
use crate::game_tree::ChessGame;
use crate::puct::{PuctConfig, RootEdge, RootNoise, search_with_root_noise};
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
    /// Play the highest-visit move (temperature 0) from this ply on (counted
    /// from the game's start position). `None` samples at `temperature` on
    /// every ply.
    pub argmax_after_ply: Option<u32>,
    /// Root Dirichlet noise concentration; used only when
    /// `root_dirichlet_epsilon > 0`.
    pub root_dirichlet_alpha: f32,
    /// Root noise mixing weight; `0.0` disables root noise.
    pub root_dirichlet_epsilon: f32,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        Self {
            simulations_per_move: 16,
            c_puct: 1.0,
            temperature: 1.0,
            ply_cap: 256,
            recurrence: 1,
            argmax_after_ply: None,
            root_dirichlet_alpha: 0.3,
            root_dirichlet_epsilon: 0.0,
        }
    }
}

impl SelfPlayConfig {
    /// Move-selection temperature at `ply` (plies played since the start).
    pub fn temperature_at(&self, ply: u32) -> f32 {
        match self.argmax_after_ply {
            Some(n) if ply >= n => 0.0,
            _ => self.temperature,
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

/// Per-game sums of root search diagnostics (not serialized into replay).
///
/// Three priors are distinguished at every searched root: the network policy,
/// the noisy root prior PUCT actually used (network mixed with root Dirichlet
/// noise; identical to the network policy when noise is off), and the final
/// visit target. `target_vs_noisy` isolates tree-search movement after
/// exploration noise; `noisy_vs_network` is the noise alone.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RootSearchDiag {
    pub plies: u64,
    pub kl_noisy_vs_network: f64,
    pub argmax_changed_by_noise: u64,
    pub kl_target_vs_noisy: f64,
    pub argmax_changed_by_search: u64,
    pub kl_target_vs_network: f64,
    pub argmax_changed_total: u64,
    pub abs_network_value: f64,
    pub abs_root_value: f64,
}

impl RootSearchDiag {
    pub fn add(&mut self, o: &RootSearchDiag) {
        self.plies += o.plies;
        self.kl_noisy_vs_network += o.kl_noisy_vs_network;
        self.argmax_changed_by_noise += o.argmax_changed_by_noise;
        self.kl_target_vs_noisy += o.kl_target_vs_noisy;
        self.argmax_changed_by_search += o.argmax_changed_by_search;
        self.kl_target_vs_network += o.kl_target_vs_network;
        self.argmax_changed_total += o.argmax_changed_total;
        self.abs_network_value += o.abs_network_value;
        self.abs_root_value += o.abs_root_value;
    }

    fn push(&mut self, network: &[f32], noisy: &[f32], target: &[f32], net_v: f32, root_v: f32) {
        self.plies += 1;
        self.kl_noisy_vs_network += kl(noisy, network);
        self.kl_target_vs_noisy += kl(target, noisy);
        self.kl_target_vs_network += kl(target, network);
        let (an, ao, at) = (argmax(network), argmax(noisy), argmax(target));
        self.argmax_changed_by_noise += u64::from(an != ao);
        self.argmax_changed_by_search += u64::from(ao != at);
        self.argmax_changed_total += u64::from(an != at);
        self.abs_network_value += net_v.abs() as f64;
        self.abs_root_value += root_v.abs() as f64;
    }
}

/// KL(p || q) in nats over the support of `p`.
fn kl(p: &[f32], q: &[f32]) -> f64 {
    p.iter()
        .zip(q)
        .filter(|(p, _)| **p > 0.0)
        .map(|(&p, &q)| p as f64 * (p as f64 / (q as f64).max(1e-12)).ln())
        .sum()
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |b, (i, &x)| {
            if x > b.1 { (i, x) } else { b }
        })
        .0
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
    /// Root search diagnostics summed over the game's searched plies.
    pub root_diag: RootSearchDiag,
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
    let start_ply = state.ply();
    let mut plies = Vec::new();
    let mut root_diag = RootSearchDiag::default();
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
        let root_noise = (cfg.root_dirichlet_epsilon > 0.0).then(|| RootNoise {
            epsilon: cfg.root_dirichlet_epsilon,
            noise: rng.dirichlet(cfg.root_dirichlet_alpha as f64, state.legal_actions().len()),
        });
        let game = ChessGame::new(state.clone(), evaluator);
        let result = search_with_root_noise(
            game,
            &PuctConfig {
                c_puct: cfg.c_puct,
                simulations: cfg.simulations_per_move,
            },
            root_noise.as_ref(),
        )?;
        if result.edges.is_empty() {
            termination = Termination::Aborted;
            break;
        }
        if result.total_visits > 0 {
            let noisy: Vec<f32> = result.edges.iter().map(|e| e.prior).collect();
            let network: Vec<f32> = match &root_noise {
                // Invert the mix: prior' = (1-eps)*p + eps*noise.
                Some(n) if n.epsilon < 1.0 => noisy
                    .iter()
                    .zip(&n.noise)
                    .map(|(m, x)| ((m - n.epsilon * x) / (1.0 - n.epsilon)).max(0.0))
                    .collect(),
                _ => noisy.clone(),
            };
            let total = result.total_visits as f32;
            let visits: Vec<f32> = result
                .edges
                .iter()
                .map(|e| e.visits as f32 / total)
                .collect();
            root_diag.push(
                &network,
                &noisy,
                &visits,
                result.root_network_value,
                result.root_value,
            );
        }
        let target = sparse_target(&result.edges, result.total_visits);
        let temperature = cfg.temperature_at(state.ply() - start_ply);
        let selected = sample_action(&result.edges, temperature, rng);
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
        root_diag,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_after_ply_switches_temperature_to_zero() {
        let cfg = SelfPlayConfig {
            temperature: 1.0,
            argmax_after_ply: Some(30),
            ..SelfPlayConfig::default()
        };
        assert_eq!(cfg.temperature_at(0), 1.0);
        assert_eq!(cfg.temperature_at(29), 1.0);
        assert_eq!(cfg.temperature_at(30), 0.0);
        assert_eq!(cfg.temperature_at(200), 0.0);
        let always = SelfPlayConfig::default();
        assert_eq!(always.temperature_at(500), always.temperature);
    }

    #[test]
    fn root_diagnostics_separate_noise_from_search() {
        let ev = crate::evaluator::FixedEvaluator::uniform(0.0);
        let base = SelfPlayConfig {
            simulations_per_move: 16,
            ply_cap: 12,
            ..SelfPlayConfig::default()
        };
        let clean = play_game(&ev, &base, &mut Rng::new(9)).unwrap().root_diag;
        assert!(clean.plies > 0);
        assert!(
            clean.kl_noisy_vs_network.abs() < 1e-9,
            "no noise: noisy == network"
        );
        assert_eq!(clean.argmax_changed_by_noise, 0);
        assert!((clean.kl_target_vs_noisy - clean.kl_target_vs_network).abs() < 1e-9);
        assert_eq!(clean.argmax_changed_by_search, clean.argmax_changed_total);

        let noisy_cfg = SelfPlayConfig {
            root_dirichlet_epsilon: 0.25,
            root_dirichlet_alpha: 0.3,
            ..base
        };
        let noisy = play_game(&ev, &noisy_cfg, &mut Rng::new(9))
            .unwrap()
            .root_diag;
        assert!(
            noisy.kl_noisy_vs_network > 0.0,
            "noise must move the root prior"
        );
        assert!(noisy.kl_target_vs_noisy.is_finite() && noisy.kl_target_vs_network.is_finite());
    }
}
