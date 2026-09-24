//! Run directory layout and metadata.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::config::RunConfig;

/// Run status recorded in `metadata.json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RunStatus {
    Running,
    Completed,
    Interrupted,
    Failed,
}

/// Run metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunMetadata {
    pub run_id: String,
    pub status: RunStatus,
    pub git_revision: Option<String>,
    /// Branch the binary was built from (HP experiment provenance).
    #[serde(default)]
    pub git_branch: Option<String>,
    /// Self-play / learner seed recorded for this run.
    #[serde(default)]
    pub seed: u64,
    /// Hardware scheduling profile label (e.g. "hp-home").
    #[serde(default)]
    pub hardware_profile: Option<String>,
    /// Model profile label (e.g. "f15", "r15").
    #[serde(default)]
    pub model_profile: Option<String>,
    pub recur64_version: String,
    pub observation_version: u32,
    pub action_version: u32,
    pub rules_profile_version: u32,
    pub replay_schema_version: u32,
    pub device: String,
    pub precision: String,
    pub note: Option<String>,
}

impl RunMetadata {
    pub fn new(cfg: &RunConfig) -> Self {
        let v = recur64_core::ContractVersions::V1;
        Self {
            run_id: cfg.run_id.clone(),
            status: RunStatus::Running,
            git_revision: option_env!("RECUR64_GIT_SHA").map(|s| s.to_string()),
            git_branch: option_env!("RECUR64_GIT_BRANCH").map(|s| s.to_string()),
            seed: cfg.seed,
            hardware_profile: cfg.hardware_profile.clone(),
            model_profile: cfg.model_profile.clone(),
            recur64_version: crate::VERSION.to_string(),
            observation_version: v.observation,
            action_version: v.action,
            rules_profile_version: v.rules_profile,
            replay_schema_version: crate::replay::REPLAY_SCHEMA_VERSION,
            device: cfg.device.clone(),
            precision: cfg.precision.clone(),
            note: None,
        }
    }
}

/// The run directory layout.
pub struct RunDir {
    pub root: PathBuf,
}

impl RunDir {
    /// Create the run directory structure. Refuses to reuse an existing run
    /// unless `force` is set.
    pub fn create(root: &Path, force: bool) -> anyhow::Result<Self> {
        if root.exists() {
            anyhow::ensure!(
                force,
                "run directory already exists: {} (pass --force to reuse)",
                root.display()
            );
            if force {
                std::fs::remove_dir_all(root)?;
            }
        }
        std::fs::create_dir_all(root)?;
        let dir = Self {
            root: root.to_path_buf(),
        };
        for sub in ["logs", "replay", "checkpoints", "eval", "report"] {
            std::fs::create_dir_all(dir.root.join(sub))?;
        }
        Ok(dir)
    }

    pub fn replay(&self) -> PathBuf {
        self.root.join("replay")
    }
    pub fn checkpoints(&self) -> PathBuf {
        self.root.join("checkpoints")
    }
    pub fn reference_ckpt(&self) -> PathBuf {
        self.checkpoints().join("reference")
    }
    pub fn candidate_ckpt(&self) -> PathBuf {
        self.checkpoints().join("candidate")
    }
    pub fn eval(&self) -> PathBuf {
        self.root.join("eval")
    }
    pub fn report(&self) -> PathBuf {
        self.root.join("report")
    }
    pub fn logs(&self) -> PathBuf {
        self.root.join("logs")
    }

    /// Write the resolved config and initial metadata.
    pub fn write_config_and_metadata(&self, cfg: &RunConfig) -> anyhow::Result<()> {
        std::fs::write(self.root.join("config.toml"), toml::to_string_pretty(cfg)?)?;
        write_metadata(&self.root, &RunMetadata::new(cfg))?;
        Ok(())
    }

    pub fn update_status(
        &self,
        cfg: &RunConfig,
        status: RunStatus,
        note: Option<String>,
    ) -> anyhow::Result<()> {
        let mut meta = RunMetadata::new(cfg);
        meta.status = status;
        meta.note = note;
        write_metadata(&self.root, &meta)
    }
}

/// Atomically write `metadata.json`.
pub fn write_metadata(root: &Path, meta: &RunMetadata) -> anyhow::Result<()> {
    let tmp = root.join("metadata.json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(meta)?)?;
    std::fs::rename(&tmp, root.join("metadata.json"))?;
    Ok(())
}

/// Read `metadata.json`.
pub fn read_metadata(root: &Path) -> anyhow::Result<RunMetadata> {
    Ok(serde_json::from_slice(&std::fs::read(
        root.join("metadata.json"),
    )?)?)
}

/// One cycle's provenance record, appended to `lineage.jsonl`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageRecord {
    pub cycle: u32,
    pub run_id: String,
    pub parent_model_id: String,
    pub candidate_model_id: String,
    pub promoted_model_id: String,
    pub replay_model_ids: Vec<String>,
    pub new_positions: u64,
    pub new_trainable_positions: u64,
    pub examples_consumed: u64,
    pub optimizer_step_start: u64,
    pub optimizer_step_end: u64,
    pub wall_clock_secs: f64,
    pub arena_candidate_score: Option<f64>,
    pub snapshot_decision: String,
    pub config_hash: String,
    pub scientific_config_hash: String,
    pub resolved_config_hash: String,
    pub git_revision: Option<String>,
    pub git_branch: Option<String>,
    pub seed: u64,
}

impl RunDir {
    /// Append a lineage record (JSON Lines, one record per cycle).
    pub fn append_lineage(&self, record: &LineageRecord) -> anyhow::Result<()> {
        use std::io::Write;
        let path = self.root.join("lineage.jsonl");
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        writeln!(f, "{}", serde_json::to_string(record)?)?;
        f.sync_all()?;
        Ok(())
    }

    /// Read all lineage records.
    pub fn read_lineage(&self) -> anyhow::Result<Vec<LineageRecord>> {
        let path = self.root.join("lineage.jsonl");
        if !path.exists() {
            return Ok(Vec::new());
        }
        let text = std::fs::read_to_string(path)?;
        let mut out = Vec::new();
        for line in text.lines() {
            if line.trim().is_empty() {
                continue;
            }
            out.push(serde_json::from_str(line)?);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> RunConfig {
        RunConfig::from_toml_str(
            r#"
run_id = "meta-test"
device = "cpu"
precision = "fp32"
seed = 7
hardware_profile = "hp-home"
model_profile = "f15"
[model]
width = 32
heads = 4
ffn = 64
input_blocks = 0
core_blocks = 1
output_blocks = 0
"#,
        )
        .unwrap()
    }

    #[test]
    fn metadata_records_seed_and_profiles() {
        let meta = RunMetadata::new(&cfg());
        assert_eq!(meta.seed, 7);
        assert_eq!(meta.hardware_profile.as_deref(), Some("hp-home"));
        assert_eq!(meta.model_profile.as_deref(), Some("f15"));
    }

    #[test]
    fn metadata_records_git_provenance_when_built_in_repo() {
        let meta = RunMetadata::new(&cfg());
        if option_env!("RECUR64_GIT_SHA").is_some() {
            assert!(
                meta.git_revision.is_some(),
                "git revision must be recorded when built inside the repo"
            );
        }
        if option_env!("RECUR64_GIT_BRANCH").is_some() {
            assert!(meta.git_branch.is_some());
        }
    }

    #[test]
    fn metadata_round_trips_through_json() {
        let meta = RunMetadata::new(&cfg());
        let bytes = serde_json::to_vec(&meta).unwrap();
        let back: RunMetadata = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(back.run_id, meta.run_id);
        assert_eq!(back.seed, meta.seed);
        assert_eq!(back.git_revision, meta.git_revision);
    }
}
