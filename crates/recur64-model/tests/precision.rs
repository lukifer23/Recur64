//! Phase 0 precision gate: unsupported requests must fail visibly.

use recur64_model::config::{DeviceKind, Precision};
use recur64_model::precision::ensure_supported;

#[test]
fn fp32_is_accepted() {
    ensure_supported(Precision::Fp32, DeviceKind::Cpu).expect("fp32 cpu must be supported");
    ensure_supported(Precision::Fp32, DeviceKind::Cuda).expect("fp32 cuda request is allowed");
}

#[test]
fn bf16_fails_visibly_until_tested() {
    let cpu = ensure_supported(Precision::Bf16, DeviceKind::Cpu).unwrap_err();
    assert!(cpu.to_string().contains("NOT YET"));
    let cuda = ensure_supported(Precision::Bf16, DeviceKind::Cuda).unwrap_err();
    assert!(cuda.to_string().contains("NOT YET TESTED"));
    assert!(cuda.to_string().contains("silently falling back"));
}

#[test]
fn fp16_fails_visibly() {
    let err = ensure_supported(Precision::Fp16, DeviceKind::Cuda).unwrap_err();
    assert!(err.to_string().contains("loss-scaling"));
}

/// T5: TF32 runs only when build and request agree.
#[test]
fn tf32_requires_a_matching_build_in_both_directions() {
    use recur64_model::precision::TF32_BUILD;
    let tf32 = ensure_supported(Precision::Tf32, DeviceKind::Cuda);
    let fp32 = ensure_supported(Precision::Fp32, DeviceKind::Cuda);
    if TF32_BUILD {
        assert!(tf32.is_ok());
        assert!(fp32.unwrap_err().to_string().contains("tf32"));
    } else {
        assert!(tf32.unwrap_err().to_string().contains("--features tf32"));
        assert!(fp32.is_ok());
    }
    assert!(ensure_supported(Precision::Tf32, DeviceKind::Cpu).is_err());
}
