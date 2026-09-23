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
            git_revision: None,
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
    pub replay_model_ids: Vec<String>,
    pub new_positions: u64,
    pub examples_consumed: u64,
    pub optimizer_step_start: u64,
    pub optimizer_step_end: u64,
    pub wall_clock_secs: f64,
    pub arena_candidate_score: Option<f64>,
    pub snapshot_decision: String,
    pub config_hash: String,
    pub git_revision: Option<String>,
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
