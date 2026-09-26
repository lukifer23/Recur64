//! Mainline F10/R10 matched-parameter guarantees.
//!
//! F10 is the feed-forward control (`0 input + 8 core + 0 output`); R10 is the
//! shared recurrent core (`2 input + 4 core + 2 output`). Both store exactly
//! eight unique transformer blocks of identical geometry, so their unique
//! parameter counts must match exactly. These are *contract* tests: a mismatch
//! must fail loudly rather than silently unbalance the recurrence comparison.
//!
//! F10 and R10 R1 are **not** function-class equivalent: raw input is
//! reinjected at a different graph location, so the clean within-family
//! comparison is R10 R1 vs R2 vs R4. F10 is the matched-parameter
//! architecture control.

use burn::backend::Flex;
use recur64_model::config::{ModelConfig, ProbeConfig};
use recur64_model::model::ProbeModel;

/// Exact unique parameter count for the mainline F10/R10 geometry
/// (width 384, heads 12, ffn 768, 8 unique blocks, policy_dim 128).
/// 9,805,288 under head v1; head v2 adds the final pre-head RMSNorm
/// (width 384 scale parameters) to both architectures.
const MAINLINE_UNIQUE_PARAMS: usize = 9_805_672;

fn mainline_model(input: usize, core: usize, output: usize) -> ModelConfig {
    ModelConfig {
        width: 384,
        heads: 12,
        ffn: 768,
        input_blocks: input,
        core_blocks: core,
        output_blocks: output,
        squares: 64,
        in_features: 119,
        policy_dim: 128,
        wdl_classes: 3,
        promo_codes: 5,
        rms_eps: 1e-5,
    }
}

#[test]
fn f10_and_r10_have_identical_unique_parameter_counts() {
    let device = Default::default();
    let f10 = ProbeModel::<Flex>::new(mainline_model(0, 8, 0), &device);
    let r10 = ProbeModel::<Flex>::new(mainline_model(2, 4, 2), &device);

    let n_f10 = f10.num_params();
    let n_r10 = r10.num_params();

    assert_eq!(
        n_f10, MAINLINE_UNIQUE_PARAMS,
        "F10 unique parameter count drifted from the frozen mainline geometry"
    );
    assert_eq!(
        n_r10, MAINLINE_UNIQUE_PARAMS,
        "R10 unique parameter count drifted from the frozen mainline geometry"
    );
    assert_eq!(
        n_f10, n_r10,
        "F10 and R10 must be matched in unique parameter count"
    );
    assert_eq!(f10.core_block_count(), 8);
    assert_eq!(
        r10.core_block_count(),
        4,
        "R10 shares exactly four core blocks"
    );
}

#[test]
fn f10_and_r10_block_accounting_matches_spec() {
    let f10 = mainline_model(0, 8, 0);
    let r10 = mainline_model(2, 4, 2);

    // Unique stored blocks (shared core counted once).
    assert_eq!(f10.unique_blocks(), 8);
    assert_eq!(r10.unique_blocks(), 8);

    // Final-output inference: F10 is 0 + 8R + 0; R10 is 2 + 4R + 2.
    assert_eq!(f10.executed_blocks_final(1), 8);
    assert_eq!(r10.executed_blocks_final(1), 8);
    assert_eq!(r10.executed_blocks_final(2), 12);
    assert_eq!(r10.executed_blocks_final(4), 20);

    // Deep-supervision training: F10 is 0 + (8+0)R; R10 is 2 + (4+2)R.
    assert_eq!(f10.executed_blocks_deep_supervision(1), 8);
    assert_eq!(r10.executed_blocks_deep_supervision(1), 8);
    assert_eq!(r10.executed_blocks_deep_supervision(2), 14);
    assert_eq!(r10.executed_blocks_deep_supervision(4), 26);
}

#[test]
fn param_breakdown_sums_to_total() {
    let device = Default::default();
    let r10 = ProbeModel::<Flex>::new(mainline_model(2, 4, 2), &device);
    let sum: usize = r10.param_breakdown().iter().map(|(_, n)| *n).sum();
    assert_eq!(sum, r10.num_params());
}

/// The topology used by R10 (2 input / 4 core / 2 output) must satisfy R=1
/// parity between the recurrent loop and the explicit straight-line control.
/// Uses small dimensions so the check is fast; the topology, not the width, is
/// what this test pins.
#[test]
fn r10_topology_r1_parity() {
    use recur64_model::fixture::SynthFixture;

    let cfg = ModelConfig {
        width: 32,
        heads: 4,
        ffn: 64,
        input_blocks: 2,
        core_blocks: 4,
        output_blocks: 2,
        squares: 64,
        in_features: 119,
        policy_dim: 16,
        wdl_classes: 3,
        promo_codes: 5,
        rms_eps: 1e-5,
    };
    let device = Default::default();
    let model = ProbeModel::<Flex>::new(cfg.clone(), &device);
    let fx = SynthFixture::new(4, 119, 1);
    let (board, cands, _t) = fx.tensors::<Flex>(&device);

    let a = model.forward_r(board.clone(), &cands, 1, false);
    let b = model.forward_control(board, &cands);
    let la = a.readouts[0]
        .policy
        .log_probs
        .clone()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let lb = b.readouts[0]
        .policy
        .log_probs
        .clone()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    assert_eq!(la.len(), lb.len());
    for (x, y) in la.iter().zip(lb.iter()) {
        assert!((x - y).abs() < 1e-5, "R=1 parity mismatch: {x} vs {y}");
    }
    assert_eq!(a.executed_blocks, 8); // 2 + 4*1 + 2
}

/// The committed config files must parse and describe the frozen geometry.
#[test]
fn mainline_config_files_parse_and_match() {
    let f10_text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../configs/f10.toml"
    ))
    .expect("configs/f10.toml must exist");
    let r10_text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../configs/r10-probe.toml"
    ))
    .expect("configs/r10-probe.toml must exist");

    let f10 = ProbeConfig::from_toml_str(&f10_text).unwrap();
    let r10 = ProbeConfig::from_toml_str(&r10_text).unwrap();

    assert_eq!(f10.model.unique_blocks(), 8);
    assert_eq!(r10.model.unique_blocks(), 8);
    assert_eq!(f10.model.width, r10.model.width);
    assert_eq!(f10.model.heads, r10.model.heads);
    assert_eq!(f10.model.ffn, r10.model.ffn);
    assert_eq!(f10.model.head_dim(), 32);
    assert_eq!(r10.recurrence, vec![1, 2, 4]);
}
