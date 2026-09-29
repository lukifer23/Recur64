//! Precision support gate.
//!
//! FP32 is the correctness baseline. Non-FP32 precision is only accepted when
//! the *entire* graph (forward, backward, optimizer, normalization, attention,
//! softmax/log-softmax, masking, gather, checkpoint restore) has been tested on
//! the selected backend. Until then, requesting a non-FP32 precision fails
//! visibly rather than silently falling back.

use crate::config::{DeviceKind, Precision};

/// DETECTED vs TESTED distinction used by `recur64 doctor`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SupportStatus {
    /// Runs end-to-end and has been measured.
    Tested,
    /// Hardware/backend reports capability, but the full graph has not run.
    Detected,
    /// Not yet attempted.
    NotTested,
    /// Known unsupported.
    Unsupported,
}

impl SupportStatus {
    pub fn label(&self) -> &'static str {
        match self {
            SupportStatus::Tested => "TESTED",
            SupportStatus::Detected => "detected",
            SupportStatus::NotTested => "NOT YET TESTED",
            SupportStatus::Unsupported => "unsupported",
        }
    }
}

/// Whether this binary was built with TF32 tensor-core matmuls enabled (the
/// `tf32` feature turns on Burn autotune, whose candidates include TF32
/// `cmma`/`mma` kernels for f32 inputs).
pub const TF32_BUILD: bool = cfg!(feature = "tf32");

/// Refuse non-FP32 precision until it is genuinely supported end-to-end.
///
/// TF32 (T5) is refused in both directions unless build and request agree:
/// an FP32 request on a TF32 build could silently use TF32 matmuls, and a
/// TF32 request on an FP32 build would silently run FP32.
pub fn ensure_supported(precision: Precision, device: DeviceKind) -> anyhow::Result<()> {
    match (precision, device) {
        (Precision::Fp32, DeviceKind::Cuda) if TF32_BUILD => anyhow::bail!(
            "this binary is built with the `tf32` feature, whose matmuls may use TF32 \
             tensor cores; refusing an FP32 request rather than silently reducing \
             precision. Use a build without `tf32`, or request precision = \"tf32\"."
        ),
        (Precision::Fp32, _) => Ok(()),
        (Precision::Tf32, DeviceKind::Cuda) if TF32_BUILD => Ok(()),
        (Precision::Tf32, DeviceKind::Cuda) => anyhow::bail!(
            "precision tf32 requires a binary built with `--features tf32`; this build \
             would run FP32. Refusing rather than silently substituting."
        ),
        (Precision::Tf32, DeviceKind::Cpu) => anyhow::bail!(
            "precision tf32 is a CUDA tensor-core mode; the CPU backend has no TF32 path."
        ),
        (Precision::Bf16, DeviceKind::Cuda) => anyhow::bail!(
            "BF16 full-graph support on the CUDA backend is NOT YET TESTED. \
             Refusing to run rather than silently falling back to FP32. \
             Re-run with --precision fp32."
        ),
        (Precision::Bf16, DeviceKind::Cpu) => anyhow::bail!(
            "BF16 full-graph support on the CPU (Flex) backend is NOT YET IMPLEMENTED. \
             Refusing to run rather than silently falling back to FP32."
        ),
        (Precision::Fp16, _) => anyhow::bail!(
            "FP16 requires an explicitly tested overflow/loss-scaling strategy and is \
             not supported in Phase 0. Refusing to run."
        ),
    }
}

/// Report status for a precision on a device, for `doctor`/`model-info`.
pub fn status(precision: Precision, device: DeviceKind) -> (SupportStatus, &'static str) {
    match (precision, device) {
        (Precision::Fp32, DeviceKind::Cpu) => {
            (SupportStatus::Tested, "CPU FP32 correctness baseline")
        }
        (Precision::Fp32, DeviceKind::Cuda) => (
            SupportStatus::NotTested,
            "CUDA FP32 graph not yet run on this workstation",
        ),
        (Precision::Bf16, DeviceKind::Cuda) => (
            SupportStatus::NotTested,
            "BF16 hardware capability may be detected, but the full graph is NOT YET TESTED",
        ),
        (Precision::Tf32, DeviceKind::Cuda) => (
            SupportStatus::NotTested,
            "TF32 tensor-core matmuls (T5); validated only by the T5 parity/training checks",
        ),
        (Precision::Tf32, DeviceKind::Cpu) => (SupportStatus::Unsupported, "CUDA-only mode"),
        (Precision::Bf16, DeviceKind::Cpu) => (SupportStatus::NotTested, "not implemented"),
        (Precision::Fp16, _) => (SupportStatus::Unsupported, "requires explicit loss scaling"),
    }
}
