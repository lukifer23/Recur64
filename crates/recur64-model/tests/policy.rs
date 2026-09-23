//! Phase 0 policy-path correctness: joint normalization, padding, promotions,
//! terminal bypass, and promotion-path gradients.

use burn::optim::GradientsParams;
use burn::prelude::*;
use burn::tensor::Distribution;

use recur64_model::action::{CandidateBatch, PROMO_NONE, PROMO_Q};
use recur64_model::config::ModelConfig;
use recur64_model::fixture::SynthFixture;
use recur64_model::loss::model_loss;
use recur64_model::model::{CandidateTensors, ProbeModel};
use recur64_model::train::CpuTrainBackend;

type B = burn::backend::Flex;

fn cfg() -> ModelConfig {
    ModelConfig {
        width: 32,
        heads: 4,
        ffn: 64,
        input_blocks: 1,
        core_blocks: 2,
        output_blocks: 1,
        squares: 64,
        in_features: 119,
        policy_dim: 16,
        wdl_classes: 3,
        promo_codes: 5,
        rms_eps: 1e-5,
    }
}

fn log_probs_vec(model: &ProbeModel<B>, fx: &SynthFixture) -> (Vec<f32>, usize) {
    let device = Default::default();
    let (board, cands, _t) = fx.tensors::<B>(&device);
    let out = model.forward_r(board, &cands, 1, false);
    let w = cands.width;
    (
        out.readouts[0]
            .policy
            .log_probs
            .clone()
            .into_data()
            .to_vec::<f32>()
            .unwrap(),
        w,
    )
}

#[test]
fn legal_probabilities_normalize_to_one() {
    let device = Default::default();
    let model = ProbeModel::<B>::new(cfg(), &device);
    let fx = SynthFixture::new(10, 119, 5);
    let (lp, w) = log_probs_vec(&model, &fx);
    for (b, list) in fx.lists.iter().enumerate() {
        if list.is_empty() {
            continue;
        }
        let sum: f32 = (0..list.len()).map(|k| lp[b * w + k].exp()).sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "row {b} probabilities sum to {sum}, expected 1"
        );
    }
}

#[test]
fn padded_candidates_are_exactly_zero() {
    let device = Default::default();
    let model = ProbeModel::<B>::new(cfg(), &device);
    let fx = SynthFixture::new(10, 119, 6);
    let (lp, w) = log_probs_vec(&model, &fx);
    for (b, list) in fx.lists.iter().enumerate() {
        for k in list.len()..w {
            assert_eq!(lp[b * w + k], 0.0, "row {b} padding {k} must be exactly 0");
        }
    }
}

#[test]
fn terminal_rows_bypass_softmax_without_nan() {
    let device = Default::default();
    let model = ProbeModel::<B>::new(cfg(), &device);
    // Row 0 is terminal (empty list); other rows have candidates.
    let lists = vec![vec![], vec![(0, 8, PROMO_NONE), (8, 16, PROMO_Q)]];
    let cb = CandidateBatch::from_lists(&lists);
    let cands = CandidateTensors::from_batch(&cb, &device);
    let board = Tensor::<B, 3>::random([2, 64, 119], Distribution::Default, &device);
    let out = model.forward_r(board, &cands, 1, false);
    let lp = out.readouts[0]
        .policy
        .log_probs
        .clone()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    assert!(
        lp.iter().all(|v| v.is_finite()),
        "terminal bypass must not produce NaN"
    );
    let valid = out.readouts[0]
        .policy
        .valid
        .clone()
        .into_data()
        .to_vec::<bool>()
        .unwrap();
    assert!(!valid[0], "row 0 must be flagged invalid/terminal");
    assert!(valid[1], "row 1 must be valid");
}

#[test]
fn promotion_path_receives_gradient() {
    type TB = CpuTrainBackend;
    type Inner = burn::backend::Flex;
    let device = Default::default();
    let model = ProbeModel::<TB>::new(cfg(), &device);
    // Fixture with promotion candidates present.
    let fx = SynthFixture::new(10, 119, 7);
    assert!(fx.lists.iter().flatten().any(|&(_, _, p)| p > 0));
    let (board, cands, targets) = fx.tensors::<TB>(&device);
    let out = model.forward_r(board, &cands, 1, false);
    let loss = model_loss(&out, &targets);
    let grads = GradientsParams::from_grads(loss.backward(), &model);
    let g = grads
        .get::<Inner, 2>(model.promo_weight_id())
        .expect("promotion head gradient");
    let nonzero = g
        .into_data()
        .to_vec::<f32>()
        .unwrap()
        .iter()
        .any(|v| v.abs() > 1e-12);
    assert!(nonzero, "promotion path must receive a nonzero gradient");
}
