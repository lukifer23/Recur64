//! `ReasoningTargetsV1`: a fixed, exact, teacher-labelled position set for the
//! X15 reasoning screen.
//!
//! The teacher is Recur64's own network plus deterministic PUCT (no root noise,
//! one leaf in flight, no external engine, tablebase, book or human data).
//! Each position is stored as `start_fen` plus the exact prefix of action ids,
//! so the repetition and fifty-move state is reconstructed move for move
//! rather than lost in a FEN. Each position carries one target per rung of a
//! simulation ladder, so "distance to a deeper search" is a stored fact.
//!
//! Scientific identity = teacher contract + seed + positions (`digest`).
//! Provenance (git revision, wall-clock time) is recorded but excluded from it.

use std::path::Path;

use recur64_core::{ActionId, GameState, StandardMove, encode_observation_v1};
use recur64_search::Evaluator;
use recur64_search::game_tree::ChessGame;
use recur64_search::puct::{PuctConfig, search};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const SCHEMA: &str = "reasoning_targets_v1";

/// Position categories, in a fixed order (also the selection order).
pub const CATEGORIES: [&str; 5] = [
    "opening",
    "middlegame",
    "endgame",
    "tactical",
    "material_advantage",
];

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TeacherContract {
    /// Path the checkpoint was loaded from (informational).
    pub checkpoint: String,
    /// `model_id` from the checkpoint's `meta.json`.
    pub model_id: String,
    pub architecture: String,
    pub recurrence: usize,
    pub c_puct: f32,
    pub leaves_in_flight: u32,
    pub root_noise: bool,
    /// Simulation ladder, shallowest first; the last rung is the deep teacher.
    pub ladder: Vec<u32>,
    /// How the teacher network was evaluated. Empty for files written before
    /// this field existed (batch-1 `SyncEvaluator`); skipped when empty so those
    /// files keep their digest.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub evaluator: String,
}

/// Recorded but NOT part of the scientific digest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    pub git_rev: String,
    pub created_unix_s: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RungTarget {
    pub simulations: u32,
    /// Visit distribution over `legal`, in `legal` order (sums to 1).
    pub policy: Vec<f32>,
    /// Search root value, side-to-move perspective, in [-1, 1].
    pub root_value: f32,
    /// The raw network's value at the root (pre-search).
    pub root_network_value: f32,
    pub total_visits: u32,
    /// Index into `legal` of the most-visited action (ties: lower action id).
    pub best: usize,
    /// Entropy of `policy` in nats.
    pub entropy: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PositionTarget {
    pub id: String,
    pub category: String,
    /// `train` or `val`; assigned per source game so the two never share one.
    pub split: String,
    pub source_game_id: u64,
    pub ply: u32,
    pub start_fen: String,
    /// `ActionId::index()` of every move played from `start_fen`.
    pub prefix: Vec<u32>,
    /// FEN of the position (audit only; history lives in `prefix`).
    pub fen: String,
    /// SHA-256 over the ObservationV1 floats (little-endian).
    pub observation_sha256: String,
    /// Legal action ids in engine order (`GameState::legal_actions`).
    pub legal: Vec<u32>,
    pub rungs: Vec<RungTarget>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReasoningTargetsV1 {
    pub schema: String,
    pub teacher: TeacherContract,
    pub seed: u64,
    /// SHA-256 over (teacher, seed, positions).
    pub digest: String,
    pub provenance: Provenance,
    pub positions: Vec<PositionTarget>,
}

impl ReasoningTargetsV1 {
    fn compute_digest(
        teacher: &TeacherContract,
        seed: u64,
        positions: &[PositionTarget],
    ) -> String {
        let canonical = serde_json::to_vec(&(teacher, seed, positions)).expect("serialize");
        format!("{:x}", Sha256::digest(&canonical))
    }

    pub fn new(
        teacher: TeacherContract,
        seed: u64,
        provenance: Provenance,
        positions: Vec<PositionTarget>,
    ) -> Self {
        let digest = Self::compute_digest(&teacher, seed, &positions);
        Self {
            schema: SCHEMA.into(),
            teacher,
            seed,
            digest,
            provenance,
            positions,
        }
    }

    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(path, serde_json::to_vec(self)?)?;
        Ok(())
    }

    /// Load and verify the schema and the digest.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let me: Self = serde_json::from_slice(&std::fs::read(path)?)?;
        anyhow::ensure!(
            me.schema == SCHEMA,
            "unknown targets schema {:?}",
            me.schema
        );
        let d = Self::compute_digest(&me.teacher, me.seed, &me.positions);
        anyhow::ensure!(
            d == me.digest,
            "targets digest mismatch: file says {}, contents hash to {d}",
            me.digest
        );
        Ok(me)
    }
}

/// Rebuild the exact `GameState` (history included) for a stored position.
pub fn rebuild_state(start_fen: &str, prefix: &[u32]) -> anyhow::Result<GameState> {
    let mut state = GameState::from_fen(start_fen).map_err(|e| anyhow::anyhow!("{e}"))?;
    for &idx in prefix {
        let id = ActionId::from_index(idx).map_err(|e| anyhow::anyhow!("{e}"))?;
        let (from, to, promo) = id.to_physical(state.perspective());
        let promotion = if promo.is_none() { None } else { Some(promo) };
        state
            .apply(StandardMove::new(from, to, promotion))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
    }
    Ok(state)
}

pub fn observation_digest(state: &GameState) -> String {
    let obs = encode_observation_v1(state);
    let mut h = Sha256::new();
    for v in obs.as_slice() {
        h.update(v.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}

/// Reconstruct every stored position move for move and check that the FEN,
/// legal list and observation digest all reproduce. Returns the position count.
pub fn audit(targets: &ReasoningTargetsV1) -> anyhow::Result<usize> {
    for p in &targets.positions {
        let state = rebuild_state(&p.start_fen, &p.prefix)?;
        anyhow::ensure!(state.to_fen() == p.fen, "{}: FEN does not reproduce", p.id);
        let legal: Vec<u32> = state.legal_actions().iter().map(|a| a.index()).collect();
        anyhow::ensure!(legal == p.legal, "{}: legal list does not reproduce", p.id);
        anyhow::ensure!(
            observation_digest(&state) == p.observation_sha256,
            "{}: observation digest does not reproduce",
            p.id
        );
        for r in &p.rungs {
            anyhow::ensure!(r.policy.len() == p.legal.len(), "{}: policy length", p.id);
            let s: f32 = r.policy.iter().sum();
            anyhow::ensure!((s - 1.0).abs() < 1e-4, "{}: policy sums to {s}", p.id);
        }
    }
    Ok(targets.positions.len())
}

// --- selection -------------------------------------------------------------------

/// SplitMix64 finalizer: a stable, seedable hash.
fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn piece_value(c: char) -> i32 {
    match c.to_ascii_lowercase() {
        'p' => 1,
        'n' | 'b' => 3,
        'r' => 5,
        'q' => 9,
        _ => 0,
    }
}

/// (white material, black material, non-king piece count) from a FEN.
fn material(fen: &str) -> (i32, i32, i32) {
    let (mut w, mut b, mut n) = (0, 0, 0);
    for c in fen.split(' ').next().unwrap_or("").chars() {
        if c.is_ascii_alphabetic() && !c.eq_ignore_ascii_case(&'k') {
            n += 1;
            if c.is_ascii_uppercase() {
                w += piece_value(c);
            } else {
                b += piece_value(c);
            }
        }
    }
    (w, b, n)
}

/// Classify a position by simple, transparent features.
fn categorize(state: &GameState, ply: u32) -> &'static str {
    let fen = state.to_fen();
    let (w, b, pieces) = material(&fen);
    if !state.board().checkers().is_empty() {
        "tactical"
    } else if ply <= 14 {
        "opening"
    } else if pieces <= 10 {
        "endgame"
    } else if (w - b).abs() >= 3 {
        "material_advantage"
    } else {
        "middlegame"
    }
}

/// A position candidate before teacher labelling.
pub struct Candidate {
    pub game_id: u64,
    pub ply: u32,
    pub start_fen: String,
    pub prefix: Vec<u32>,
    pub category: &'static str,
}

/// Deterministically pick `n` positions from replay games with a fixed quota
/// per category (remainders go to `middlegame`), by seeded hash order. Terminal
/// positions are skipped. `val_denominator = 4` puts about a quarter of the
/// source games in `val`.
pub fn select_positions(
    games: &[crate::replay::schema::GameRecord],
    n: usize,
    seed: u64,
) -> anyhow::Result<Vec<Candidate>> {
    let mut by_cat: Vec<Vec<(u64, Candidate)>> = CATEGORIES.iter().map(|_| Vec::new()).collect();
    for g in games {
        let mut state = GameState::from_fen(&g.start_fen).map_err(|e| anyhow::anyhow!("{e}"))?;
        let mut prefix: Vec<u32> = Vec::new();
        for (ply, rec) in g.plies.iter().enumerate() {
            if !state.is_terminal() && ply >= 2 {
                let cat = categorize(&state, ply as u32);
                let key = mix(seed ^ mix(g.game_id) ^ mix(ply as u64 + 0x5151));
                let idx = CATEGORIES.iter().position(|c| *c == cat).expect("category");
                by_cat[idx].push((
                    key,
                    Candidate {
                        game_id: g.game_id,
                        ply: ply as u32,
                        start_fen: g.start_fen.clone(),
                        prefix: prefix.clone(),
                        category: cat,
                    },
                ));
            }
            let id =
                ActionId::from_index(rec.selected as u32).map_err(|e| anyhow::anyhow!("{e}"))?;
            let (from, to, promo) = id.to_physical(state.perspective());
            let promotion = if promo.is_none() { None } else { Some(promo) };
            state
                .apply(StandardMove::new(from, to, promotion))
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            prefix.push(rec.selected as u32);
        }
    }
    for v in &mut by_cat {
        v.sort_by_key(|(k, _)| *k);
    }
    // Equal quotas; leftover slots (rounding or an exhausted category) go to
    // whichever categories still have candidates, in category order.
    let mut take = vec![n / CATEGORIES.len(); CATEGORIES.len()];
    let mut short = n - take.iter().sum::<usize>();
    let mut i = 1; // rounding remainder starts at middlegame
    while short > 0 {
        take[i % CATEGORIES.len()] += 1;
        short -= 1;
        i += 1;
    }
    let mut out = Vec::new();
    let mut deficit = 0usize;
    for (c, quota) in take.iter().enumerate() {
        let got = (*quota).min(by_cat[c].len());
        deficit += quota - got;
        out.extend(by_cat[c].drain(..got).map(|(_, c)| c));
    }
    while deficit > 0 {
        let mut progressed = false;
        for v in &mut by_cat {
            if deficit > 0 && !v.is_empty() {
                out.push(v.remove(0).1);
                deficit -= 1;
                progressed = true;
            }
        }
        anyhow::ensure!(progressed, "not enough replay positions for {n} targets");
    }
    Ok(out)
}

/// Whether a source game belongs to the validation split.
pub fn split_for_game(game_id: u64, seed: u64) -> &'static str {
    if mix(seed ^ mix(game_id ^ 0xA5A5)).is_multiple_of(4) {
        "val"
    } else {
        "train"
    }
}

// --- teacher labelling ---------------------------------------------------------

/// Label candidates with the teacher at every rung of the ladder, using
/// `threads` worker threads over contiguous chunks. Each search is independent
/// and deterministic (one leaf in flight, no noise, batch-1 forwards), so the
/// result is identical for any thread count; threads only overlap GPU
/// submission latency. Output order equals input order.
pub fn label(
    candidates: Vec<Candidate>,
    evaluator: &(dyn Evaluator + Sync),
    teacher: &TeacherContract,
    seed: u64,
    threads: usize,
    progress: &(dyn Fn(usize, usize) + Sync),
) -> anyhow::Result<Vec<PositionTarget>> {
    let total = candidates.len();
    let threads = threads.clamp(1, 16).min(total.max(1));
    let chunk = total.div_ceil(threads).max(1);
    let done = std::sync::atomic::AtomicUsize::new(0);
    let mut chunks: Vec<Vec<Candidate>> = Vec::new();
    let mut it = candidates.into_iter().peekable();
    while it.peek().is_some() {
        chunks.push(it.by_ref().take(chunk).collect());
    }
    let results: Vec<anyhow::Result<Vec<PositionTarget>>> = std::thread::scope(|scope| {
        let handles: Vec<_> = chunks
            .into_iter()
            .map(|cs| {
                let done = &done;
                scope.spawn(move || {
                    cs.into_iter()
                        .map(|c| {
                            let p = label_one(c, evaluator, teacher, seed)?;
                            let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                            progress(n, total);
                            Ok(p)
                        })
                        .collect::<anyhow::Result<Vec<_>>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| {
                h.join()
                    .unwrap_or_else(|_| Err(anyhow::anyhow!("labelling thread panicked")))
            })
            .collect()
    });
    let mut out = Vec::with_capacity(total);
    for r in results {
        out.extend(r?);
    }
    Ok(out)
}

fn label_one(
    c: Candidate,
    evaluator: &dyn Evaluator,
    teacher: &TeacherContract,
    seed: u64,
) -> anyhow::Result<PositionTarget> {
    {
        let state = rebuild_state(&c.start_fen, &c.prefix)?;
        let legal_ids = state.legal_actions();
        anyhow::ensure!(!legal_ids.is_empty(), "terminal candidate slipped through");
        let mut rungs = Vec::new();
        for &sims in &teacher.ladder {
            let cfg = PuctConfig {
                c_puct: teacher.c_puct,
                simulations: sims,
                leaves_in_flight: teacher.leaves_in_flight,
            };
            let res = search(ChessGame::new(state.clone(), evaluator), &cfg)
                .map_err(|e| anyhow::anyhow!("search failed: {e:?}"))?;
            let total_visits = res.total_visits.max(1) as f32;
            let mut policy = vec![0.0f32; legal_ids.len()];
            for e in &res.edges {
                let pos = legal_ids
                    .iter()
                    .position(|a| *a == e.action)
                    .ok_or_else(|| anyhow::anyhow!("search returned an illegal action"))?;
                policy[pos] = e.visits as f32 / total_visits;
            }
            let sum: f32 = policy.iter().sum();
            anyhow::ensure!(
                sum > 0.0,
                "teacher produced no visits at {sims} simulations"
            );
            for p in &mut policy {
                *p /= sum;
            }
            let best = policy
                .iter()
                .enumerate()
                .fold(0usize, |b, (k, v)| if *v > policy[b] { k } else { b });
            let entropy = -policy
                .iter()
                .filter(|p| **p > 0.0)
                .map(|p| p * p.ln())
                .sum::<f32>();
            rungs.push(RungTarget {
                simulations: sims,
                policy,
                root_value: res.root_value,
                root_network_value: res.root_network_value,
                total_visits: res.total_visits,
                best,
                entropy,
            });
        }
        Ok(PositionTarget {
            id: format!("g{}p{}", c.game_id, c.ply),
            category: c.category.into(),
            split: split_for_game(c.game_id, seed).into(),
            source_game_id: c.game_id,
            ply: c.ply,
            start_fen: c.start_fen,
            prefix: c.prefix,
            fen: state.to_fen(),
            observation_sha256: observation_digest(&state),
            legal: legal_ids.iter().map(|a| a.index()).collect(),
            rungs,
        })
    }
}

// --- disjointness and ladder informativeness ------------------------------------

/// The set of source game ids a targets file draws from.
pub fn source_game_ids(t: &ReasoningTargetsV1) -> std::collections::BTreeSet<u64> {
    t.positions.iter().map(|p| p.source_game_id).collect()
}

/// Drop replay games whose id is in `exclude` (whole games, never just plies).
pub fn exclude_games(
    games: Vec<crate::replay::schema::GameRecord>,
    exclude: &std::collections::BTreeSet<u64>,
) -> Vec<crate::replay::schema::GameRecord> {
    games
        .into_iter()
        .filter(|g| !exclude.contains(&g.game_id))
        .collect()
}

/// Error unless `a` and `b` share no source game.
pub fn ensure_disjoint(a: &ReasoningTargetsV1, b: &ReasoningTargetsV1) -> anyhow::Result<usize> {
    let (ia, ib) = (source_game_ids(a), source_game_ids(b));
    let shared: Vec<_> = ia.intersection(&ib).collect();
    anyhow::ensure!(
        shared.is_empty(),
        "targets share {} source games (e.g. {:?})",
        shared.len(),
        shared.iter().take(5).collect::<Vec<_>>()
    );
    Ok(ia.len().min(ib.len()))
}

/// Jensen-Shannon divergence (nats) between two distributions over the same support.
pub fn js_divergence(p: &[f32], q: &[f32]) -> f32 {
    let kl = |a: &[f32], m: &[f32]| -> f32 {
        a.iter()
            .zip(m)
            .filter(|(x, _)| **x > 0.0)
            .map(|(x, y)| x * (x / y).ln())
            .sum()
    };
    let m: Vec<f32> = p.iter().zip(q).map(|(a, b)| 0.5 * (a + b)).collect();
    0.5 * kl(p, &m) + 0.5 * kl(q, &m)
}

/// Descriptive ladder-informativeness statistics (not a gate): do deeper rungs
/// carry different information?
pub fn ladder_report(positions: &[&PositionTarget]) -> serde_json::Value {
    let n = positions.len().max(1) as f32;
    let rungs = positions.first().map_or(0, |p| p.rungs.len());
    let sims: Vec<u32> = positions
        .first()
        .map(|p| p.rungs.iter().map(|r| r.simulations).collect())
        .unwrap_or_default();
    let mean = |f: &dyn Fn(&PositionTarget) -> f32| positions.iter().map(|p| f(p)).sum::<f32>() / n;
    let entropy: Vec<f32> = (0..rungs).map(|r| mean(&|p| p.rungs[r].entropy)).collect();
    let adjacent: Vec<f32> = (1..rungs)
        .map(|r| mean(&|p| js_divergence(&p.rungs[r - 1].policy, &p.rungs[r].policy)))
        .collect();
    let deep = rungs.saturating_sub(1);
    let vs_deep: Vec<f32> = (0..deep)
        .map(|r| mean(&|p| js_divergence(&p.rungs[r].policy, &p.rungs[deep].policy)))
        .collect();
    let flips: Vec<usize> = (0..deep)
        .map(|r| {
            positions
                .iter()
                .filter(|p| p.rungs[r].best != p.rungs[deep].best)
                .count()
        })
        .collect();
    let value_change: Vec<f32> = (0..deep)
        .map(|r| mean(&|p| (p.rungs[r].root_value - p.rungs[deep].root_value).abs()))
        .collect();
    serde_json::json!({
        "positions": positions.len(),
        "simulations": sims,
        "mean_target_entropy_by_rung": entropy,
        "mean_js_adjacent_rungs": adjacent,
        "mean_js_rung_vs_deepest": vs_deep,
        "best_move_differs_from_deepest": flips,
        "mean_abs_root_value_change_vs_deepest": value_change,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn material_counts_the_fen_board_only() {
        assert_eq!(material("4k3/8/8/8/8/8/4P3/4K2R w - - 0 1"), (6, 0, 2));
        assert_eq!(
            material("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").2,
            30
        );
    }

    #[test]
    fn selection_hash_is_seeded_and_stable() {
        assert_eq!(mix(1), mix(1));
        assert_ne!(mix(1), mix(2));
        assert_eq!(split_for_game(7, 1), split_for_game(7, 1));
    }
}

#[cfg(test)]
mod disjoint_tests {
    use super::*;

    fn tgt(ids: &[u64]) -> ReasoningTargetsV1 {
        let positions = ids
            .iter()
            .map(|&g| PositionTarget {
                id: format!("g{g}"),
                category: "opening".into(),
                split: "val".into(),
                source_game_id: g,
                ply: 4,
                start_fen: String::new(),
                prefix: vec![],
                fen: String::new(),
                observation_sha256: String::new(),
                legal: vec![],
                rungs: vec![],
            })
            .collect();
        ReasoningTargetsV1::new(
            TeacherContract {
                checkpoint: String::new(),
                model_id: String::new(),
                architecture: "probe_v1".into(),
                recurrence: 1,
                c_puct: 1.0,
                leaves_in_flight: 1,
                root_noise: false,
                ladder: vec![16],
                evaluator: String::new(),
            },
            1,
            Provenance {
                git_rev: String::new(),
                created_unix_s: 0,
            },
            positions,
        )
    }

    #[test]
    fn disjointness_is_enforced_at_the_game_level() {
        assert!(ensure_disjoint(&tgt(&[1, 2, 3]), &tgt(&[4, 5])).is_ok());
        assert!(ensure_disjoint(&tgt(&[1, 2, 3]), &tgt(&[3, 9])).is_err());
    }

    #[test]
    fn js_divergence_is_zero_for_equal_and_positive_for_different() {
        let p = [0.5, 0.5, 0.0];
        assert!(js_divergence(&p, &p).abs() < 1e-7);
        let q = [0.0, 0.5, 0.5];
        assert!(js_divergence(&p, &q) > 0.1);
        // symmetric and bounded by ln 2
        assert!((js_divergence(&p, &q) - js_divergence(&q, &p)).abs() < 1e-7);
        assert!(js_divergence(&[1.0, 0.0], &[0.0, 1.0]) <= std::f32::consts::LN_2 + 1e-6);
    }
}
