//! Replay store and streaming sampler.
//!
//! The Phase 2 learner materialized every training example (with its ~30 KB
//! observation) in memory, which does not scale to a bounded replay capacity.
//! This store keeps only the compact `GameRecord`s (no observations) and builds
//! each example on demand by replaying the game to the sampled ply.

use std::path::{Path, PathBuf};

use recur64_core::{
    ActionId, Color, GameState, ObservationV1, StandardMove, encode_observation_v1,
};
use recur64_search::Rng;

use super::reader::ReplayReader;
use super::schema::{GameRecord, Manifest, PlyRecord, Shard, ShardInfo};
use super::writer::write_manifest_atomic;

/// One reconstructed training example.
#[derive(Debug, Clone)]
pub struct TrainingExample {
    pub observation: ObservationV1,
    pub legal: Vec<ActionId>,
    /// Target distribution aligned to `legal`.
    pub policy: Vec<f32>,
    /// WDL class from the side-to-move perspective: 0 win, 1 draw, 2 loss.
    pub wdl: i64,
    /// Replay game id this example came from (in-memory provenance only; not
    /// part of the replay schema).
    pub source_game_id: u64,
}

/// WDL class from the side-to-move perspective.
pub fn wdl_class(outcome: u8, side: Color) -> i64 {
    match outcome {
        1 => 1,
        0 => {
            if side == Color::White {
                0
            } else {
                2
            }
        }
        2 => {
            if side == Color::Black {
                0
            } else {
                2
            }
        }
        _ => 1,
    }
}

/// Build the training example for one ply of a reconstructed position.
pub fn example_for_ply(
    state: &GameState,
    outcome: u8,
    ply: &PlyRecord,
) -> Result<TrainingExample, String> {
    let legal = state.legal_actions();
    let mut policy = vec![0.0f32; legal.len()];
    for (idx, prob) in &ply.target {
        let pos = legal
            .iter()
            .position(|a| a.index() == *idx as u32)
            .ok_or_else(|| format!("target action {idx} not legal"))?;
        policy[pos] += *prob;
    }
    Ok(TrainingExample {
        observation: encode_observation_v1(state),
        legal,
        policy,
        wdl: wdl_class(outcome, state.side_to_move()),
        source_game_id: 0,
    })
}

/// Result of enforcing replay capacity.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CapacityReport {
    pub kept_shards: usize,
    pub archived_shards: usize,
    pub positions: u64,
}

/// Archive oldest shards until the active replay is within `max_positions`.
///
/// Archived shards are moved to `replay/archive/` (never deleted), and the
/// manifest is rewritten atomically to reference only the kept shards.
pub fn enforce_capacity(dir: &Path, max_positions: u64) -> anyhow::Result<CapacityReport> {
    let reader = ReplayReader::open(dir)?;
    let manifest = reader.manifest().clone();

    let mut kept_rev: Vec<ShardInfo> = Vec::new();
    let mut positions = 0u64;
    for info in manifest.shards.iter().rev() {
        if kept_rev.is_empty() || positions + info.positions <= max_positions {
            positions += info.positions;
            kept_rev.push(info.clone());
        } else {
            break;
        }
    }
    kept_rev.reverse();
    let archived = manifest.shards.len() - kept_rev.len();

    if archived > 0 {
        let archive = dir.join("archive");
        std::fs::create_dir_all(&archive)?;
        for info in &manifest.shards {
            if !kept_rev.iter().any(|k| k.file == info.file) {
                let from = dir.join(&info.file);
                if from.exists() {
                    std::fs::rename(from, archive.join(&info.file))?;
                }
            }
        }
    }

    let new_manifest = Manifest {
        header: manifest.header.clone(),
        games: kept_rev.iter().map(|s| s.games).sum(),
        bytes: kept_rev.iter().map(|s| s.bytes).sum(),
        shards: kept_rev.clone(),
    };
    write_manifest_atomic(dir, &new_manifest)?;

    Ok(CapacityReport {
        kept_shards: kept_rev.len(),
        archived_shards: archived,
        positions,
    })
}

/// A loaded replay store with on-demand sampling.
pub struct ReplayStore {
    dir: PathBuf,
    /// `(shard_index, game)` in insertion order.
    games: Vec<(usize, GameRecord)>,
    /// `(game_index, ply_index)` per shard, for result games only.
    coords_by_shard: Vec<Vec<(usize, usize)>>,
    total_positions: u64,
}

impl ReplayStore {
    /// Open a replay directory and index it.
    pub fn open(dir: &Path) -> anyhow::Result<Self> {
        let reader = ReplayReader::open(dir)?;
        let mut games = Vec::new();
        for (shard_index, info) in reader.manifest().shards.iter().enumerate() {
            let shard: Shard = reader.read_shard(&info.file)?;
            for g in shard.games {
                games.push((shard_index, g));
            }
        }
        let shard_count = games
            .iter()
            .map(|(s, _)| *s)
            .max()
            .map(|m| m + 1)
            .unwrap_or(0);
        let mut coords_by_shard: Vec<Vec<(usize, usize)>> = vec![Vec::new(); shard_count];
        let mut total_positions = 0u64;
        for (gi, (si, g)) in games.iter().enumerate() {
            if g.outcome.is_none() {
                continue; // truncated/aborted: no WDL supervision
            }
            total_positions += g.plies.len() as u64;
            for pi in 0..g.plies.len() {
                coords_by_shard[*si].push((gi, pi));
            }
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            games,
            coords_by_shard,
            total_positions,
        })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn total_positions(&self) -> u64 {
        self.total_positions
    }

    pub fn total_games(&self) -> usize {
        self.games.len()
    }

    /// Games with a result (their plies are sampleable).
    pub fn trainable_games(&self) -> usize {
        self.games
            .iter()
            .filter(|(_, g)| g.outcome.is_some())
            .count()
    }

    /// Positions available for sampling (result games only).
    pub fn sampleable(&self) -> usize {
        self.coords_by_shard.iter().map(|v| v.len()).sum()
    }

    /// Number of shards (for age reporting).
    pub fn shard_count(&self) -> usize {
        self.coords_by_shard.len()
    }

    /// Reconstruct one example from a coordinate.
    fn example_at(&self, coord: (usize, usize)) -> Result<TrainingExample, String> {
        let (gi, pi) = coord;
        let (_, game) = &self.games[gi];
        let outcome = game.outcome.ok_or("no outcome")?;
        let mut state = GameState::from_fen(&game.start_fen).map_err(|e| e.to_string())?;
        for (i, ply) in game.plies.iter().enumerate() {
            if i == pi {
                let mut example = example_for_ply(&state, outcome, ply)?;
                example.source_game_id = game.game_id;
                return Ok(example);
            }
            let id = ActionId::from_index(ply.selected as u32).map_err(|e| e.to_string())?;
            let perspective = state.perspective();
            let (from, to, promo) = id.to_physical(perspective);
            let promotion = if promo.is_none() { None } else { Some(promo) };
            state
                .apply(StandardMove::new(from, to, promotion))
                .map_err(|e| e.to_string())?;
        }
        Err(format!("ply {pi} out of range for game {gi}"))
    }

    /// Sample `n` examples. Newer shards are favored (weight = shard index + 1),
    /// so replay freshness is respected within the reuse budget.
    pub fn sample_batch(&self, n: usize, rng: &mut Rng) -> Result<Vec<TrainingExample>, String> {
        if self.sampleable() == 0 {
            return Err("replay has no sampleable positions".into());
        }
        // Shard weights (recency-biased): shard s has weight s+1.
        let shards = self.shard_count().max(1);
        let weight_total: u64 = (1..=shards as u64).sum();
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let mut pick = rng.next_u64() % weight_total;
            let mut shard = shards - 1;
            for s in 0..shards {
                let w = (s + 1) as u64;
                if pick < w {
                    shard = s;
                    break;
                }
                pick -= w;
            }
            // Fall back to any non-empty shard if the picked one has no results.
            let pool = if self.coords_by_shard[shard].is_empty() {
                self.coords_by_shard
                    .iter()
                    .find(|v| !v.is_empty())
                    .expect("sampleable > 0")
            } else {
                &self.coords_by_shard[shard]
            };
            let coord = pool[(rng.next_u64() as usize) % pool.len()];
            out.push(self.example_at(coord)?);
        }
        Ok(out)
    }
}
