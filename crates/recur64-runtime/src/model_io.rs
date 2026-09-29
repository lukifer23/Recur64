//! Build and load the probe model for a concrete Burn backend.

use std::path::Path;

use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::ElementConversion;

use recur64_model::config::ModelConfig;
use recur64_model::model::ProbeModel;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_check_passes_on_a_working_backend() {
        let device = Default::default();
        verify_device::<recur64_model::train::CpuTrainBackend>(&device).unwrap();
    }
}
