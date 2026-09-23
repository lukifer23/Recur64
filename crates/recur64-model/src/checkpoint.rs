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
pub const SCHEMA_VERSION: u32 = 1;

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
}

fn paths(dir: &Path) -> (PathBuf, PathBuf, PathBuf) {
    (
        dir.join("model"),
        dir.join("optimizer"),
        dir.join("meta.json"),
    )
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
    model.clone().save_file(model_path, &recorder)?;
    recorder.record(optim.to_record(), optim_path)?;
    std::fs::write(meta_path, serde_json::to_vec_pretty(meta)?)?;
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
