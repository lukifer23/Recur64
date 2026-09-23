//! Replay V1 schema.
//!
//! Compact and versioned. No dense observations and no 20,480-wide policy
//! vectors are stored: a game stores its start FEN and selected moves, so every
//! position (and its Observation V1) is reconstructed by replaying the game.
//! Legal actions are regenerated on read and the sparse target is mapped onto
//! them, which makes an illegal or misaligned target a hard error.

use serde::{Deserialize, Serialize};

use recur64_core::{ActionId, ContractVersions, Outcome, Termination};

/// Replay schema version.
pub const REPLAY_SCHEMA_VERSION: u32 = 1;
/// Shard file magic.
pub const SHARD_MAGIC: [u8; 4] = *b"R64S";

/// Provenance for a shard.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayHeader {
    pub replay_schema_version: u32,
    pub observation_version: u32,
    pub action_version: u32,
    pub rules_profile_version: u32,
    pub run_id: String,
    pub model_id: String,
    pub git_revision: Option<String>,
    pub backend: String,
    pub precision: String,
}

impl ReplayHeader {
    pub fn new(
        run_id: impl Into<String>,
        model_id: impl Into<String>,
        backend: impl Into<String>,
        precision: impl Into<String>,
    ) -> Self {
        let v = ContractVersions::V1;
        Self {
            replay_schema_version: REPLAY_SCHEMA_VERSION,
            observation_version: v.observation,
            action_version: v.action,
            rules_profile_version: v.rules_profile,
            run_id: run_id.into(),
            model_id: model_id.into(),
            git_revision: None,
            backend: backend.into(),
            precision: precision.into(),
        }
    }

    /// Check that the recorded contract versions match the current ones.
    pub fn check_contracts(&self) -> Result<(), String> {
        let v = ContractVersions::V1;
        if self.replay_schema_version != REPLAY_SCHEMA_VERSION {
            return Err(format!(
                "replay schema {} != {REPLAY_SCHEMA_VERSION}",
                self.replay_schema_version
            ));
        }
        if self.observation_version != v.observation
            || self.action_version != v.action
            || self.rules_profile_version != v.rules_profile
        {
            return Err(format!(
                "contract mismatch: obs {} action {} rules {} (expected {}/{}/{})",
                self.observation_version,
                self.action_version,
                self.rules_profile_version,
                v.observation,
                v.action,
                v.rules_profile
            ));
        }
        Ok(())
    }
}

/// Search settings that produced a game.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchRecord {
    pub simulations: u32,
    pub c_puct: f32,
    pub temperature: f32,
    pub recurrence: usize,
}

/// One played ply.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlyRecord {
    /// Canonical `ActionId` index of the selected move.
    pub selected: u16,
    /// Sparse search target: `(ActionId index, probability)`.
    pub target: Vec<(u16, f32)>,
    pub visits_total: u32,
    /// 0 = White to move, 1 = Black to move (for perspective audit).
    pub side_to_move: u8,
}

/// One complete game.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GameRecord {
    pub game_id: u64,
    pub start_fen: String,
    pub seed: u64,
    pub search: SearchRecord,
    pub plies: Vec<PlyRecord>,
    pub termination: String,
    /// `0` white win, `1` draw, `2` black win; `None` for truncated/aborted.
    pub outcome: Option<u8>,
}

impl GameRecord {
    /// Convert a completed self-play game into a replay record.
    pub fn from_selfplay(
        game_id: u64,
        game: &recur64_search::SelfPlayGame,
        search: SearchRecord,
    ) -> Self {
        let plies = game
            .plies
            .iter()
            .map(|p| PlyRecord {
                selected: p.selected.index() as u16,
                target: p
                    .target
                    .iter()
                    .map(|t| (t.action.index() as u16, t.prob))
                    .collect(),
                visits_total: p.visits_total,
                side_to_move: if p.side_to_move == recur64_core::Color::White {
                    0
                } else {
                    1
                },
            })
            .collect();
        Self {
            game_id,
            start_fen: game.start_fen.clone(),
            seed: game.seed,
            search,
            plies,
            termination: game.termination.label().to_string(),
            outcome: Self::encode_outcome(game.outcome),
        }
    }

    /// Encode an outcome from the board's perspective.
    pub fn encode_outcome(outcome: Option<Outcome>) -> Option<u8> {
        outcome.map(|o| match o {
            Outcome::Win(recur64_core::Color::White) => 0,
            Outcome::Draw => 1,
            Outcome::Win(recur64_core::Color::Black) => 2,
        })
    }

    /// True if this game carries a terminal WDL result.
    pub fn has_result(&self) -> bool {
        self.outcome.is_some()
    }
}

/// A shard: a header plus a bounded batch of games.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Shard {
    pub header: ReplayHeader,
    pub games: Vec<GameRecord>,
}

/// One entry in the replay manifest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShardInfo {
    pub file: String,
    pub games: u64,
    pub bytes: u64,
    pub crc32: u32,
}

/// The replay manifest: an atomic index of committed shards.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub header: ReplayHeader,
    pub shards: Vec<ShardInfo>,
    pub games: u64,
    pub bytes: u64,
}

/// Decode an `ActionId` index into a typed id, failing visibly out of range.
pub fn action_from_index(index: u16) -> Result<ActionId, String> {
    ActionId::from_index(index as u32).map_err(|e| e.to_string())
}

/// Parse a termination label back to the enum.
pub fn termination_from_label(label: &str) -> Option<Termination> {
    Some(match label {
        "checkmate" => Termination::Checkmate,
        "stalemate" => Termination::Stalemate,
        "insufficient_material" => Termination::InsufficientMaterial,
        "threefold_repetition" => Termination::ThreefoldRepetition,
        "fifty_move_rule" => Termination::FiftyMoveRule,
        "truncated" => Termination::Truncated,
        "aborted" => Termination::Aborted,
        _ => return None,
    })
}
