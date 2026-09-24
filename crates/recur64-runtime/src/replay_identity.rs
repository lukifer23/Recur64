//! `replay-identity.json`: a sidecar that makes a replay directory
//! self-describing without changing the Replay V1 binary schema.
//!
//! Replay V1's `SearchRecord` records simulations, c_puct, temperature and
//! recurrence, but self-play also depends on the argmax ply, root noise, the
//! readout head and the seed policy. Each write batch (one sweep cell, one
//! pilot cycle) appends one entry naming the full generation contract. A
//! replay without the sidecar is legacy: its search contract is unverified.

use std::path::Path;

use crate::config::{RunConfig, SELFPLAY_SEED_POLICY};

pub const FILE: &str = "replay-identity.json";
pub const KIND: &str = "recur64-replay-identity-v1";

/// The generation contract of one contiguous range of replay games.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ReplayIdentityEntry {
    pub first_game_id: u64,
    pub games: u64,
    pub model_id: String,
    pub head_version: u32,
    pub scientific_config_hash: String,
    pub resolved_config_hash: String,
    pub seed_policy: String,
    pub seed: u64,
    pub simulations: u32,
    pub c_puct: f32,
    pub temperature: f32,
    pub argmax_after_ply: Option<u32>,
    pub root_dirichlet_alpha: f32,
    pub root_dirichlet_epsilon: f32,
    pub recurrence: usize,
    pub ply_cap: u32,
    pub start_fen: Option<String>,
    pub git_revision: Option<String>,
}

impl ReplayIdentityEntry {
    pub fn from_config(
        cfg: &RunConfig,
        model_id: &str,
        first_game_id: u64,
        games: u64,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            first_game_id,
            games,
            model_id: model_id.to_string(),
            head_version: recur64_model::model::HEAD_VERSION,
            scientific_config_hash: cfg.scientific_config_hash()?,
            resolved_config_hash: cfg.resolved_config_hash(),
            seed_policy: SELFPLAY_SEED_POLICY.to_string(),
            seed: cfg.seed,
            simulations: cfg.simulations_per_move,
            c_puct: cfg.c_puct,
            temperature: cfg.temperature,
            argmax_after_ply: cfg.argmax_after_ply,
            root_dirichlet_alpha: cfg.root_dirichlet_alpha,
            root_dirichlet_epsilon: cfg.root_dirichlet_epsilon,
            recurrence: cfg.recurrence,
            ply_cap: cfg.ply_cap,
            start_fen: cfg.start_fen.clone(),
            git_revision: option_env!("RECUR64_GIT_SHA").map(str::to_owned),
        })
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ReplayIdentity {
    pub kind: String,
    pub entries: Vec<ReplayIdentityEntry>,
}

/// Read the sidecar; `None` for a legacy replay without one.
pub fn read(dir: &Path) -> anyhow::Result<Option<ReplayIdentity>> {
    let path = dir.join(FILE);
    if !path.exists() {
        return Ok(None);
    }
    let id: ReplayIdentity = serde_json::from_slice(&std::fs::read(&path)?)?;
    anyhow::ensure!(id.kind == KIND, "unknown replay identity kind {}", id.kind);
    Ok(Some(id))
}

/// Append one entry (atomic replace of the sidecar).
pub fn append(dir: &Path, entry: ReplayIdentityEntry) -> anyhow::Result<()> {
    let mut id = read(dir)?.unwrap_or(ReplayIdentity {
        kind: KIND.to_string(),
        entries: Vec::new(),
    });
    id.entries.push(entry);
    std::fs::create_dir_all(dir)?;
    let tmp = dir.join(format!("{FILE}.tmp"));
    std::fs::write(&tmp, serde_json::to_vec_pretty(&id)?)?;
    std::fs::rename(&tmp, dir.join(FILE))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sidecar_appends_and_reads_back_and_legacy_is_none() {
        let dir = std::env::temp_dir().join(format!("recur64-replay-id-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        assert!(read(&dir).unwrap().is_none(), "no sidecar = legacy");
        let cfg = RunConfig::from_toml_str(include_str!("../../../configs/smoke.toml")).unwrap();
        let a = ReplayIdentityEntry::from_config(&cfg, "m0", 0, 8).unwrap();
        let b = ReplayIdentityEntry::from_config(&cfg, "m1", 8, 8).unwrap();
        append(&dir, a.clone()).unwrap();
        append(&dir, b.clone()).unwrap();
        let id = read(&dir).unwrap().unwrap();
        assert_eq!(id.entries, vec![a, b]);
        assert_eq!(
            id.entries[0].head_version,
            recur64_model::model::HEAD_VERSION
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
