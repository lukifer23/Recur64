//! HP experiment F15/R15 matched-parameter guarantees (H0.4).
//!
//! F15 is the feed-forward control (0 input + 8 core + 0 output); R15 is the
//! shared recurrent core (2 input + 4 core + 2 output). Both store exactly eight
//! unique transformer blocks of identical geometry, so their unique parameter
//! counts must match exactly. These are *contract* tests: a mismatch must fail
//! loudly rather than silently unbalance the recurrence comparison.

use burn::backend::Flex;
use recur64_model::config::{ModelConfig, ProbeConfig};
use recur64_model::model::ProbeModel;

/// Exact unique parameter count for the HP F15/R15 geometry
/// (width 512, heads 8, ffn 768, 8 unique blocks, policy_dim 128).
const HP_UNIQUE_PARAMS: usize = 15_154_120;

fn hp_model(input: usize, core: usize, output: usize) -> ModelConfig {
    ModelConfig {
        width: 512,
        heads: 8,
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
fn f15_and_r15_have_identical_unique_parameter_counts() {
    let device = Default::default();
    let f15 = ProbeModel::<Flex>::new(hp_model(0, 8, 0), &device);
    let r15 = ProbeModel::<Flex>::new(hp_model(2, 4, 2), &device);

    let n_f15 = f15.num_params();
    let n_r15 = r15.num_params();

    assert_eq!(
        n_f15, HP_UNIQUE_PARAMS,
        "F15 unique parameter count drifted from the frozen HP geometry"
    );
    assert_eq!(
        n_r15, HP_UNIQUE_PARAMS,
        "R15 unique parameter count drifted from the frozen HP geometry"
    );
    assert_eq!(
        n_f15, n_r15,
        "F15 and R15 must be matched in unique parameter count"
    );
    assert_eq!(f15.core_block_count(), 8);
    assert_eq!(
        r15.core_block_count(),
        4,
        "R15 shares exactly four core blocks"
    );
}

#[test]
fn f15_and_r15_block_accounting_matches_spec() {
    let f15 = hp_model(0, 8, 0);
    let r15 = hp_model(2, 4, 2);

    // Unique stored blocks (shared core counted once).
    assert_eq!(f15.unique_blocks(), 8);
    assert_eq!(r15.unique_blocks(), 8);

    // Final-output inference: F15 is 0 + 8R + 0; R15 is 2 + 4R + 2.
    assert_eq!(f15.executed_blocks_final(1), 8);
    assert_eq!(r15.executed_blocks_final(1), 8);
    assert_eq!(r15.executed_blocks_final(2), 12);
    assert_eq!(r15.executed_blocks_final(4), 20);

    // Deep-supervision training: F15 is 0 + (8+0)R; R15 is 2 + (4+2)R.
    assert_eq!(f15.executed_blocks_deep_supervision(1), 8);
    assert_eq!(r15.executed_blocks_deep_supervision(1), 8);
    assert_eq!(r15.executed_blocks_deep_supervision(2), 14);
    assert_eq!(r15.executed_blocks_deep_supervision(4), 26);
}

#[test]
fn param_breakdown_sums_to_total() {
    let device = Default::default();
    let r15 = ProbeModel::<Flex>::new(hp_model(2, 4, 2), &device);
    let sum: usize = r15.param_breakdown().iter().map(|(_, n)| *n).sum();
    assert_eq!(sum, r15.num_params());
}

/// The topology used by R15 (2 input / 4 core / 2 output) must satisfy R=1
/// parity between the recurrent loop and the explicit straight-line control.
/// Uses small dimensions so the check is fast; the topology, not the width, is
/// what this test pins.
#[test]
fn r15_topology_r1_parity() {
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
fn hp_config_files_parse_and_match() {
    let f15_text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../configs/f15.toml"
    ))
    .expect("configs/f15.toml must exist");
    let r15_text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../configs/r15.toml"
    ))
    .expect("configs/r15.toml must exist");

    let f15 = ProbeConfig::from_toml_str(&f15_text).unwrap();
    let r15 = ProbeConfig::from_toml_str(&r15_text).unwrap();

    assert_eq!(f15.model.unique_blocks(), 8);
    assert_eq!(r15.model.unique_blocks(), 8);
    assert_eq!(f15.model.width, r15.model.width);
    assert_eq!(f15.model.heads, r15.model.heads);
    assert_eq!(f15.model.ffn, r15.model.ffn);
    assert_eq!(f15.model.head_dim(), 64);
    assert_eq!(r15.recurrence, vec![1, 2, 4]);
}
