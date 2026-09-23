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

/// Refuse non-FP32 precision until it is genuinely supported end-to-end.
pub fn ensure_supported(precision: Precision, device: DeviceKind) -> anyhow::Result<()> {
    match (precision, device) {
        (Precision::Fp32, _) => Ok(()),
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
        (Precision::Bf16, DeviceKind::Cpu) => (SupportStatus::NotTested, "not implemented"),
        (Precision::Fp16, _) => (SupportStatus::Unsupported, "requires explicit loss scaling"),
    }
}
