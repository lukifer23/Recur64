//! Network-to-target divergence: how far the stored visit target moved away
//! from the generating network's own policy.
//!
//! Re-evaluating replay positions with the generating (frozen) network
//! recovers its raw policy offline, without storing it in Replay V1.
//!
//! SEMANTICS (D41/D42): self-play mixes root Dirichlet noise into the prior
//! PUCT uses, so this metric measures `visit target` vs `raw network policy`
//! and therefore includes BOTH exploration noise and tree-search movement.
//! It is not pure search gain. The exact decomposition (network → noisy root
//! prior, noisy root prior → target) is measured during self-play by
//! `coordinator::RootSearchSummary`. Before D41 (no root noise) the two
//! coincide. The historical field names (`mean_kl_target_prior`, …) are kept
//! for comparability with recorded evidence.
//!
//! The same pass also reports the network's value/WDL on generated positions
//! and cheap repetition-attractor indicators (halfmove clock, repetition
//! share), all from the reconstructed positions.

use recur64_core::{ActionId, GameState, ObservationV1, StandardMove};

use crate::inference::BatchEvaluator;
use crate::replay::sampler::example_for_ply;
use crate::replay::schema::GameRecord;

/// Prior-vs-target statistics over a set of plies.
#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct GainStats {
    pub positions: u64,
    /// Mean KL(target || prior) in nats.
    pub mean_kl_target_prior: f64,
    /// Share of positions where the target's argmax differs from the prior's.
    pub argmax_changed_fraction: f64,
    pub mean_prior_entropy: f64,
    pub mean_target_entropy: f64,
    pub mean_prior_top1: f64,
    pub mean_target_top1: f64,
}

/// (observation, legal, target, from a result game)
type Pending = (ObservationV1, Vec<ActionId>, Vec<f32>, bool);

/// Network value and repetition indicators over generated positions.
#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct PositionStats {
    pub positions: u64,
    /// Mean predicted [win, draw, loss] (side to move).
    pub mean_wdl: [f64; 3],
    pub mean_abs_value: f64,
    /// Mean halfmove clock (plies since the last capture or pawn move).
    pub mean_halfmove_clock: f64,
    /// Share of positions that already occurred earlier in the game.
    pub repeated_position_share: f64,
}

/// Network-to-target divergence over every ply and over trainable
/// (result-game) plies, plus value / repetition indicators over all plies.
#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct SearchGain {
    pub all: GainStats,
    pub trainable: GainStats,
    pub positions: PositionStats,
}

#[derive(Default)]
struct PosAcc {
    n: u64,
    wdl: [f64; 3],
    abs_v: f64,
    halfmove: f64,
    repeated: u64,
}

#[derive(Default)]
struct Acc {
    n: u64,
    kl: f64,
    changed: u64,
    hp: f64,
    ht: f64,
    p1: f64,
    t1: f64,
}

impl Acc {
    fn push(&mut self, prior: &[f32], target: &[f32]) {
        let argmax = |v: &[f32]| {
            v.iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |best, (i, &x)| {
                    if x > best.1 { (i, x) } else { best }
                })
        };
        let entropy = |v: &[f32]| {
            -v.iter()
                .filter(|&&x| x > 0.0)
                .map(|&x| x as f64 * (x as f64).ln())
                .sum::<f64>()
        };
        let kl: f64 = target
            .iter()
            .zip(prior)
            .filter(|(t, _)| **t > 0.0)
            .map(|(&t, &p)| t as f64 * (t as f64 / (p as f64).max(1e-12)).ln())
            .sum();
        let (pa, pmax) = argmax(prior);
        let (ta, tmax) = argmax(target);
        self.n += 1;
        self.kl += kl;
        self.changed += u64::from(pa != ta);
        self.hp += entropy(prior);
        self.ht += entropy(target);
        self.p1 += pmax as f64;
        self.t1 += tmax as f64;
    }

    fn stats(&self) -> GainStats {
        let n = self.n.max(1) as f64;
        GainStats {
            positions: self.n,
            mean_kl_target_prior: self.kl / n,
            argmax_changed_fraction: self.changed as f64 / n,
            mean_prior_entropy: self.hp / n,
            mean_target_entropy: self.ht / n,
            mean_prior_top1: self.p1 / n,
            mean_target_top1: self.t1 / n,
        }
    }
}

/// Reconstruct every played position, evaluate it with `model` in batches of
/// `batch`, and compare the prior with the stored visit target.
pub fn search_gain(
    model: &dyn BatchEvaluator,
    games: &[GameRecord],
    batch: usize,
) -> anyhow::Result<SearchGain> {
    let mut all = Acc::default();
    let mut trainable = Acc::default();
    let mut pos = PosAcc::default();
    let mut pending: Vec<Pending> = Vec::new();
    let flush = |pending: &mut Vec<Pending>,
                 all: &mut Acc,
                 trainable: &mut Acc,
                 pos: &mut PosAcc|
     -> anyhow::Result<()> {
        if pending.is_empty() {
            return Ok(());
        }
        let obs: Vec<_> = pending.iter().map(|p| p.0.clone()).collect();
        let legal: Vec<_> = pending.iter().map(|p| p.1.clone()).collect();
        let out = model
            .evaluate_batch(&obs, &legal)
            .map_err(|e| anyhow::anyhow!("search-gain evaluation failed: {e}"))?;
        for ((_, _, target, is_trainable), r) in pending.drain(..).zip(out) {
            pos.n += 1;
            for (acc, p) in pos.wdl.iter_mut().zip(r.wdl) {
                *acc += p as f64;
            }
            pos.abs_v += r.value.abs() as f64;
            all.push(&r.policy, &target);
            if is_trainable {
                trainable.push(&r.policy, &target);
            }
        }
        Ok(())
    };
    for game in games {
        let mut state = GameState::from_fen(&game.start_fen)
            .map_err(|e| anyhow::anyhow!("game {} start FEN: {e}", game.game_id))?;
        for (i, ply) in game.plies.iter().enumerate() {
            let ex = example_for_ply(&state, game.outcome.unwrap_or(1), ply)
                .map_err(|e| anyhow::anyhow!("game {} ply {i}: {e}", game.game_id))?;
            pos.halfmove += state.board().halfmove_clock() as f64;
            pos.repeated += u64::from(state.repetition_count() > 1);
            pending.push((ex.observation, ex.legal, ex.policy, game.outcome.is_some()));
            if pending.len() >= batch.max(1) {
                flush(&mut pending, &mut all, &mut trainable, &mut pos)?;
            }
            let id = ActionId::from_index(ply.selected as u32)
                .map_err(|e| anyhow::anyhow!("game {} ply {i}: {e}", game.game_id))?;
            let (from, to, promo) = id.to_physical(state.perspective());
            let promotion = if promo.is_none() { None } else { Some(promo) };
            state
                .apply(StandardMove::new(from, to, promotion))
                .map_err(|e| anyhow::anyhow!("game {} ply {i}: {e}", game.game_id))?;
        }
    }
    flush(&mut pending, &mut all, &mut trainable, &mut pos)?;
    let n = pos.n.max(1) as f64;
    Ok(SearchGain {
        all: all.stats(),
        trainable: trainable.stats(),
        positions: PositionStats {
            positions: pos.n,
            mean_wdl: pos.wdl.map(|x| x / n),
            mean_abs_value: pos.abs_v / n,
            mean_halfmove_clock: pos.halfmove / n,
            repeated_position_share: pos.repeated as f64 / n,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_equal_to_prior_has_zero_gain() {
        let mut a = Acc::default();
        a.push(&[0.5, 0.3, 0.2], &[0.5, 0.3, 0.2]);
        let s = a.stats();
        assert!(s.mean_kl_target_prior.abs() < 1e-9);
        assert_eq!(s.argmax_changed_fraction, 0.0);
    }

    #[test]
    fn moved_target_reports_kl_and_argmax_change() {
        let mut a = Acc::default();
        // Uniform prior over 4, target all on the last move.
        a.push(&[0.25; 4], &[0.0, 0.0, 0.0, 1.0]);
        // Same argmax, sharper target.
        a.push(&[0.6, 0.4], &[0.9, 0.1]);
        let s = a.stats();
        let kl2 = 0.9f64 * (0.9f64 / 0.6).ln() + 0.1 * (0.1f64 / 0.4).ln();
        assert!((s.mean_kl_target_prior - (4f64.ln() + kl2) / 2.0).abs() < 1e-6);
        assert_eq!(s.argmax_changed_fraction, 0.5);
        assert!((s.mean_target_top1 - 0.95).abs() < 1e-6);
        assert!((s.mean_prior_entropy - (4f64.ln() + 0.6730116670092565) / 2.0).abs() < 1e-6);
    }
}
