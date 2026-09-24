//! Build and load the probe model for a concrete Burn backend.

use std::path::Path;

use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};

use recur64_model::config::ModelConfig;
use recur64_model::model::ProbeModel;

/// Build a fresh model (random init, eagerly initialized).
pub fn build<B: Backend>(cfg: &ModelConfig, device: &B::Device) -> ProbeModel<B> {
    ProbeModel::<B>::new(cfg.clone(), device)
}

/// Load model weights from a checkpoint directory (expects `<dir>/model[.mpk]`).
pub fn load<B: Backend>(
    dir: &Path,
    cfg: &ModelConfig,
    device: &B::Device,
) -> anyhow::Result<ProbeModel<B>> {
    // Every load path checks the checkpoint contracts (chess contracts and
    // head version), not only full training loads.
    let meta: recur64_model::checkpoint::CheckpointMeta =
        serde_json::from_slice(&std::fs::read(dir.join("meta.json")).map_err(|e| {
            anyhow::anyhow!(
                "checkpoint {} has no readable meta.json: {e}",
                dir.display()
            )
        })?)?;
    meta.check_contracts()?;
    let template = ProbeModel::<B>::new(cfg.clone(), device);
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let model = template.load_file(dir.join("model"), &recorder, device)?;
    Ok(model)
}
