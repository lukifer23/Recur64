//! Training checkpoints for the Phase 0 probe.
//!
//! A *training checkpoint* preserves enough state to genuinely resume: model
//! parameters, optimizer moments/step, recurrence configuration, RNG state, and
//! metadata. A weights-only file is an *inference export*, not a resumable
//! checkpoint, and must not be presented as one.

use std::path::{Path, PathBuf};

use burn::module::Module;
use burn::optim::Optimizer;
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
use burn::tensor::backend::AutodiffBackend;

use crate::config::ModelConfig;
use crate::model::ProbeModel;

/// Current checkpoint schema version. Bump on any breaking change.
///
/// * v1 — Phase 0 probe checkpoints (no chess contract metadata).
/// * v2 — Phase 2: records observation/action/rules/replay contract versions,
///   model identity, and run identity.
pub const SCHEMA_VERSION: u32 = 2;

/// Metadata stored alongside a training checkpoint.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CheckpointMeta {
    pub schema_version: u32,
    pub recur64_version: String,
    pub git_revision: Option<String>,
    pub backend: String,
    pub precision: String,
    pub model: ModelConfig,
    pub recurrence: usize,
    pub deep_supervision: bool,
    pub step: u64,
    pub lr: f64,
    pub seed: u64,
    /// Sampler / fixture RNG state needed to resume the exact data sequence.
    pub rng_state: u64,
    // --- Phase 2 chess contract + identity metadata (defaulted for reading
    //     older metadata; a schema mismatch is refused before use). ---
    #[serde(default)]
    pub observation_version: u32,
    #[serde(default)]
    pub action_version: u32,
    #[serde(default)]
    pub rules_profile_version: u32,
    #[serde(default)]
    pub replay_schema_version: u32,
    /// Stable content-derived identity of the saved model.
    #[serde(default)]
    pub model_id: String,
    #[serde(default)]
    pub run_id: String,
    /// Number of optimizer updates applied.
    #[serde(default)]
    pub update_counter: u64,
    /// Position of the LR schedule.
    #[serde(default)]
    pub lr_schedule_step: u64,
    /// Readout-head function version ([`crate::model::HEAD_VERSION`]).
    /// Metadata written before the field existed is head v1.
    #[serde(default = "legacy_head_version")]
    pub head_version: u32,
}

fn legacy_head_version() -> u32 {
    1
}

impl CheckpointMeta {
    /// Build metadata with the current contract versions and empty identity.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: ModelConfig,
        recurrence: usize,
        deep_supervision: bool,
        step: u64,
        lr: f64,
        seed: u64,
        rng_state: u64,
        backend: impl Into<String>,
        precision: impl Into<String>,
    ) -> Self {
        let v = recur64_core::ContractVersions::V1;
        Self {
            schema_version: SCHEMA_VERSION,
            recur64_version: crate::VERSION.to_string(),
            git_revision: None,
            backend: backend.into(),
            precision: precision.into(),
            model,
            recurrence,
            deep_supervision,
            step,
            lr,
            seed,
            rng_state,
            observation_version: v.observation,
            action_version: v.action,
            rules_profile_version: v.rules_profile,
            replay_schema_version: 1,
            model_id: String::new(),
            run_id: String::new(),
            update_counter: step,
            lr_schedule_step: step,
            head_version: crate::model::HEAD_VERSION,
        }
    }

    /// Refuse a checkpoint whose recorded model configuration differs from
    /// the configuration it is being loaded into. Tensor shapes alone do not
    /// prove compatibility: same-shape settings such as `rms_eps` change the
    /// function. Recurrence is deliberately NOT checked: it is a runtime
    /// choice, and the same R10 weights are evaluated at R1/R2/R4.
    pub fn check_model(&self, requested: &ModelConfig) -> anyhow::Result<()> {
        let recorded = serde_json::to_value(&self.model)?;
        let wanted = serde_json::to_value(requested)?;
        anyhow::ensure!(
            recorded == wanted,
            "checkpoint model config {recorded} differs from the requested model config {wanted}"
        );
        Ok(())
    }

    /// Verify the model-function contracts: head version and the chess
    /// observation / action / rules versions. `replay_schema_version` is
    /// training-data provenance, not part of the network's function, so it
    /// is recorded but does not block loading.
    pub fn check_contracts(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.head_version == crate::model::HEAD_VERSION,
            "checkpoint head version {} is not the current head version {}: its weights were trained for a different readout function and are refused",
            self.head_version,
            crate::model::HEAD_VERSION
        );
        let v = recur64_core::ContractVersions::V1;
        anyhow::ensure!(
            self.observation_version == v.observation
                && self.action_version == v.action
                && self.rules_profile_version == v.rules_profile,
            "checkpoint contract mismatch: obs {} action {} rules {} (expected {}/{}/{})",
            self.observation_version,
            self.action_version,
            self.rules_profile_version,
            v.observation,
            v.action,
            v.rules_profile
        );
        Ok(())
    }
}

fn paths(dir: &Path) -> (PathBuf, PathBuf, PathBuf) {
    (
        dir.join("model"),
        dir.join("optimizer"),
        dir.join("meta.json"),
    )
}

/// Content hash of a file (used for stable model identity).
pub fn hash_file(path: &Path) -> anyhow::Result<String> {
    use sha2::{Digest, Sha256};
    let bytes = std::fs::read(path)?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

/// Save a full training checkpoint.
pub fn save_training<B, O>(
    dir: &Path,
    model: &ProbeModel<B>,
    optim: &O,
    meta: &CheckpointMeta,
) -> anyhow::Result<()>
where
    B: AutodiffBackend,
    O: Optimizer<ProbeModel<B>, B>,
{
    std::fs::create_dir_all(dir)?;
    let (model_path, optim_path, meta_path) = paths(dir);
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    model
        .clone()
        .save_file(model_path.clone(), &recorder)
        .map_err(|e| anyhow::anyhow!("save model to {}: {e}", model_path.display()))?;
    recorder
        .record(optim.to_record(), optim_path.clone())
        .map_err(|e| anyhow::anyhow!("save optimizer to {}: {e}", optim_path.display()))?;
    // Model identity is the content hash of the saved weights. The MPK recorder
    // writes `<path>.mpk`; fall back to the bare path if that convention changes.
    let mut meta = meta.clone();
    let with_mpk = model_path.with_extension("mpk");
    let model_file = if with_mpk.exists() {
        with_mpk
    } else {
        model_path.clone()
    };
    meta.model_id = hash_file(&model_file)
        .map_err(|e| anyhow::anyhow!("hash {}: {e}", model_file.display()))?;
    std::fs::write(&meta_path, serde_json::to_vec_pretty(&meta)?)
        .map_err(|e| anyhow::anyhow!("write {}: {e}", meta_path.display()))?;
    Ok(())
}

/// Load a full training checkpoint into a freshly built model and optimizer.
///
/// Refuses to load a checkpoint whose schema version does not match.
pub fn load_training<B, O>(
    dir: &Path,
    template: ProbeModel<B>,
    optim: O,
    device: &B::Device,
) -> anyhow::Result<(ProbeModel<B>, O, CheckpointMeta)>
where
    B: AutodiffBackend,
    O: Optimizer<ProbeModel<B>, B>,
{
    let (model_path, optim_path, meta_path) = paths(dir);
    let meta: CheckpointMeta = serde_json::from_slice(&std::fs::read(&meta_path)?)?;
    anyhow::ensure!(
        meta.schema_version == SCHEMA_VERSION,
        "checkpoint schema mismatch: found {} expected {}",
        meta.schema_version,
        SCHEMA_VERSION
    );
    meta.check_contracts()?;
    meta.check_model(template.config())?;
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let model = template.load_file(model_path, &recorder, device)?;
    let optim_record = recorder.load(optim_path, device)?;
    let optim = optim.load_record(optim_record);
    Ok((model, optim, meta))
}

/// Save a weights-only inference export (NOT a resumable checkpoint).
pub fn save_inference_export<B: Backend>(
    dir: &Path,
    model: &ProbeModel<B>,
) -> anyhow::Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let path = dir.join("weights");
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    model.clone().save_file(path.clone(), &recorder)?;
    Ok(path)
}
