//! Build and load the probe model for a concrete Burn backend.

use std::path::Path;

use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::ElementConversion;

use recur64_model::chimera::ChimeraModel;
use recur64_model::config::ModelConfig;
use recur64_model::experimental::{Architecture, ExperimentalConfig};
use recur64_model::model::ProbeModel;

thread_local! {
    static DEVICE_CHECKS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// How many device checks the *calling thread* has started. A test seam that
/// proves a build/load path goes through [`verify_device`].
pub fn device_checks_run() -> usize {
    DEVICE_CHECKS.with(|c| c.get())
}

/// How long the device check may take (first-use JIT compilation included).
const DEVICE_CHECK_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);

/// Run a small computation with a known result on `device` (elementwise,
/// reduction and matmul kernels) and refuse the device unless it is exact.
///
/// A device whose kernels cannot run must fail here, visibly, before any real
/// work: when CUDA's NVRTC cannot be loaded, the panic happens only on the
/// device's worker thread and JIT kernels silently do nothing. The check runs
/// on a helper thread with a timeout, so a panic or a hang becomes an error.
pub fn verify_device<B: Backend>(device: &B::Device) -> anyhow::Result<()> {
    DEVICE_CHECKS.with(|c| c.set(c.get() + 1));
    let device = device.clone();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let x = Tensor::<B, 1>::from_floats([1.0, 2.0, 3.0, 4.0], &device);
            let affine: f32 = (x.clone() * 2.0 + 1.0).sum().into_scalar().elem();
            let m = x.clone().reshape([2, 2]);
            let product: Vec<f32> = m.clone().matmul(m).into_data().to_vec().unwrap_or_default();
            (affine, product)
        }));
        let _ = tx.send(result);
    });
    let (affine, product) = match rx.recv_timeout(DEVICE_CHECK_TIMEOUT) {
        Ok(Ok(values)) => values,
        Ok(Err(_)) => anyhow::bail!(
            "device check panicked: the backend cannot run kernels on this device. \
             For CUDA, check that NVRTC for the pinned CUDA 12.9.1 runtime is on PATH \
             (docs/HARDWARE.md, DECISIONS.md D3/D57). Refusing to run."
        ),
        Err(_) => anyhow::bail!(
            "device check did not finish within {DEVICE_CHECK_TIMEOUT:?}: the backend \
             cannot run kernels on this device. Refusing to run."
        ),
    };
    // 2 * (1 + 2 + 3 + 4) + 4 = 24; [[1, 2], [3, 4]]^2 = [[7, 10], [15, 22]].
    anyhow::ensure!(
        affine == 24.0 && product == [7.0, 10.0, 15.0, 22.0],
        "device check computed wrong results (sum {affine}, matmul {product:?}; \
         expected 24 and [7, 10, 15, 22]): kernels are not running correctly on \
         this device. Refusing to run."
    );
    Ok(())
}

/// Build a fresh model (random init, eagerly initialized) after verifying that
/// the device runs kernels correctly.
pub fn build<B: Backend>(cfg: &ModelConfig, device: &B::Device) -> anyhow::Result<ProbeModel<B>> {
    verify_device::<B>(device)?;
    Ok(ProbeModel::<B>::new(cfg.clone(), device))
}

/// Load model weights from a checkpoint directory (expects `<dir>/model[.mpk]`).
pub fn load<B: Backend>(
    dir: &Path,
    cfg: &ModelConfig,
    device: &B::Device,
) -> anyhow::Result<ProbeModel<B>> {
    verify_device::<B>(device)?;
    load_unverified(dir, cfg, device)
}

/// [`load`] without the behavioural device check, for callers that already ran
/// it (a multi-owner pool instantiates several copies of one network, and each
/// copy must not pay the check again).
pub fn load_unverified<B: Backend>(
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
    meta.check_model(cfg)?;
    meta.check_architecture(recur64_model::experimental::Architecture::ProbeV1)?;
    let template = ProbeModel::<B>::new(cfg.clone(), device);
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let model = template.load_file(dir.join("model"), &recorder, device)?;
    Ok(model)
}

/// Build a fresh X15 / Chimera model after validating its experimental
/// contract and verifying that the device runs kernels correctly.
pub fn build_chimera<B: Backend>(
    cfg: &ModelConfig,
    exp: &ExperimentalConfig,
    device: &B::Device,
) -> anyhow::Result<ChimeraModel<B>> {
    verify_device::<B>(device)?;
    build_chimera_unverified(cfg, exp, device)
}

/// [`build_chimera`] without the behavioural device check, for callers that
/// already ran it (an owner pool builds several copies of one network).
pub fn build_chimera_unverified<B: Backend>(
    cfg: &ModelConfig,
    exp: &ExperimentalConfig,
    device: &B::Device,
) -> anyhow::Result<ChimeraModel<B>> {
    anyhow::ensure!(
        exp.is_chimera(),
        "build_chimera requires architecture = chimera_v1 (got {:?})",
        exp.architecture
    );
    exp.validate(cfg.width)?;
    Ok(ChimeraModel::<B>::new(cfg.clone(), exp.clone(), device))
}

/// Load X15 weights (no optimizer) from a checkpoint directory after the
/// device check. Refuses probe checkpoints and any model or experimental
/// contract mismatch.
pub fn load_chimera<B: Backend>(
    dir: &Path,
    cfg: &ModelConfig,
    exp: &ExperimentalConfig,
    device: &B::Device,
) -> anyhow::Result<ChimeraModel<B>> {
    verify_device::<B>(device)?;
    load_chimera_unverified(dir, cfg, exp, device)
}

/// [`load_chimera`] without the behavioural device check.
pub fn load_chimera_unverified<B: Backend>(
    dir: &Path,
    cfg: &ModelConfig,
    exp: &ExperimentalConfig,
    device: &B::Device,
) -> anyhow::Result<ChimeraModel<B>> {
    let meta = read_meta(dir)?;
    meta.check_contracts()?;
    meta.check_model(cfg)?;
    meta.check_architecture(Architecture::ChimeraV1)?;
    meta.check_experimental(exp)?;
    let template = build_chimera_unverified::<B>(cfg, exp, device)?;
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    Ok(template.load_file(dir.join("model"), &recorder, device)?)
}

/// Resume an X15 training checkpoint (weights, optimizer state and metadata)
/// after the device check.
pub fn load_chimera_training<B, O>(
    dir: &Path,
    cfg: &ModelConfig,
    exp: &ExperimentalConfig,
    optim: O,
    device: &B::Device,
) -> anyhow::Result<(
    ChimeraModel<B>,
    O,
    recur64_model::checkpoint::CheckpointMeta,
)>
where
    B: burn::tensor::backend::AutodiffBackend,
    O: burn::optim::Optimizer<ChimeraModel<B>, B>,
{
    verify_device::<B>(device)?;
    let template = build_chimera_unverified::<B>(cfg, exp, device)?;
    recur64_model::checkpoint::load_training_chimera(dir, template, optim, device)
}

fn read_meta(dir: &Path) -> anyhow::Result<recur64_model::checkpoint::CheckpointMeta> {
    let bytes = std::fs::read(dir.join("meta.json")).map_err(|e| {
        anyhow::anyhow!(
            "checkpoint {} has no readable meta.json: {e}",
            dir.display()
        )
    })?;
    Ok(serde_json::from_slice(&bytes)?)
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default, clippy::items_after_test_module)]
mod tests {
    use super::*;
    use recur64_model::experimental::{Architecture, ExperimentalConfig};

    type Cpu = recur64_model::train::CpuTrainBackend;

    fn tiny_model_cfg() -> ModelConfig {
        ModelConfig {
            width: 32,
            heads: 4,
            ffn: 48,
            input_blocks: 1,
            core_blocks: 1,
            output_blocks: 1,
            squares: 64,
            in_features: 119,
            policy_dim: 8,
            wdl_classes: 3,
            promo_codes: 5,
            rms_eps: 1e-5,
        }
    }

    fn tiny_exp() -> ExperimentalConfig {
        let mut e = ExperimentalConfig::default();
        e.architecture = Architecture::ChimeraV1;
        e.thought_steps = 2;
        e.reasoning_tokens = 2;
        e.reasoning.aux_width = 16;
        e.reasoning.aux_heads = 4;
        e.reasoning.latent_ffn = 32;
        e.visual.channels = 4;
        e.visual.blocks = 1;
        e
    }

    #[test]
    fn device_check_passes_on_a_working_backend() {
        let device = Default::default();
        verify_device::<Cpu>(&device).unwrap();
    }

    #[test]
    fn chimera_build_and_load_run_the_device_check_at_the_runtime_boundary() {
        let device = Default::default();
        let (cfg, exp) = (tiny_model_cfg(), tiny_exp());
        let before = device_checks_run();
        build_chimera::<Cpu>(&cfg, &exp, &device).unwrap();
        assert_eq!(device_checks_run(), before + 1, "build_chimera must verify");
        build_chimera_unverified::<Cpu>(&cfg, &exp, &device).unwrap();
        assert_eq!(
            device_checks_run(),
            before + 1,
            "unverified must not repeat"
        );
    }

    #[test]
    fn a_probe_config_is_refused_by_the_chimera_builder() {
        let device = Default::default();
        let err = build_chimera::<Cpu>(&tiny_model_cfg(), &ExperimentalConfig::default(), &device);
        assert!(err.is_err());
    }
}

// --- Chimera V2 ---------------------------------------------------------------------

use recur64_model::chimera2::ChimeraV2Model;

/// Build a fresh Chimera V2 model after validating its contract and verifying the device.
pub fn build_chimera_v2<B: Backend>(
    cfg: &ModelConfig,
    exp: &ExperimentalConfig,
    device: &B::Device,
) -> anyhow::Result<ChimeraV2Model<B>> {
    verify_device::<B>(device)?;
    build_chimera_v2_unverified(cfg, exp, device)
}

/// [`build_chimera_v2`] without the behavioural device check (callers that already ran it).
pub fn build_chimera_v2_unverified<B: Backend>(
    cfg: &ModelConfig,
    exp: &ExperimentalConfig,
    device: &B::Device,
) -> anyhow::Result<ChimeraV2Model<B>> {
    anyhow::ensure!(
        exp.is_chimera_v2(),
        "build_chimera_v2 requires architecture = chimera_v2 (got {:?})",
        exp.architecture
    );
    exp.validate(cfg.width)?;
    exp.validate_v2_model(cfg)?;
    Ok(ChimeraV2Model::<B>::new(cfg.clone(), exp.clone(), device))
}

/// Load V2 weights (no optimizer) after the device check. Refuses probe and V1
/// checkpoints and any model or experimental contract mismatch.
pub fn load_chimera_v2<B: Backend>(
    dir: &Path,
    cfg: &ModelConfig,
    exp: &ExperimentalConfig,
    device: &B::Device,
) -> anyhow::Result<ChimeraV2Model<B>> {
    verify_device::<B>(device)?;
    let meta = read_meta(dir)?;
    meta.check_contracts()?;
    meta.check_model(cfg)?;
    meta.check_architecture(Architecture::ChimeraV2)?;
    meta.check_experimental(exp)?;
    let template = build_chimera_v2_unverified::<B>(cfg, exp, device)?;
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    Ok(template.load_file(dir.join("model"), &recorder, device)?)
}

/// Resume a V2 training checkpoint (weights, optimizer state, metadata) after the device check.
pub fn load_chimera_v2_training<B, O>(
    dir: &Path,
    cfg: &ModelConfig,
    exp: &ExperimentalConfig,
    optim: O,
    device: &B::Device,
) -> anyhow::Result<(
    ChimeraV2Model<B>,
    O,
    recur64_model::checkpoint::CheckpointMeta,
)>
where
    B: burn::tensor::backend::AutodiffBackend,
    O: burn::optim::Optimizer<ChimeraV2Model<B>, B>,
{
    verify_device::<B>(device)?;
    let template = build_chimera_v2_unverified::<B>(cfg, exp, device)?;
    recur64_model::checkpoint::load_training_v2(dir, template, optim, device)
}
