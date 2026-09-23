//! Phase 0 recurrence correctness: shared weights, aggregated gradients,
//! R=1 parity, state reset, and batch/individual agreement.

use burn::optim::GradientsParams;
use burn::prelude::*;

use recur64_model::config::ModelConfig;
use recur64_model::fixture::SynthFixture;
use recur64_model::loss::model_loss;
use recur64_model::model::ProbeModel;
use recur64_model::train::CpuTrainBackend;

fn small_cfg() -> ModelConfig {
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

fn log_probs<B: Backend>(out: &recur64_model::model::ModelOutput<B>) -> Vec<f32> {
    out.readouts[0]
        .policy
        .log_probs
        .clone()
        .into_data()
        .to_vec::<f32>()
        .unwrap()
}

/// The recurrent model executed at R=1 must match the explicit straight-line
/// control graph exactly.
#[test]
fn r1_parity() {
    type B = burn::backend::Flex;
    let device = Default::default();
    let model = ProbeModel::<B>::new(small_cfg(), &device);
    let fx = SynthFixture::new(4, 119, 1);
    let (board, cands, _t) = fx.tensors::<B>(&device);

    let a = model.forward_r(board.clone(), &cands, 1, false);
    let b = model.forward_control(board, &cands);
    let la = log_probs(&a);
    let lb = log_probs(&b);
    assert_eq!(la.len(), lb.len());
    for (x, y) in la.iter().zip(lb.iter()) {
        assert!((x - y).abs() < 1e-5, "R=1 parity mismatch: {x} vs {y}");
    }
    assert_eq!(a.executed_blocks, 4); // input 1 + core 2*1 + output 1
}

/// Gradients must reach the shared core weight and later recurrent executions
/// must change that gradient. Verified against finite differences.
#[test]
fn shared_core_gradients_aggregate() {
    type B = CpuTrainBackend;
    type Inner = burn::backend::Flex;
    let device = Default::default();
    let model = ProbeModel::<B>::new(small_cfg(), &device);
    let fx = SynthFixture::new(4, 119, 2);
    let (board, cands, targets) = fx.tensors::<B>(&device);
    let id = model.core_weight_id();

    let grad_at = |r: usize| -> f32 {
        let out = model.forward_r(board.clone(), &cands, r, false);
        let loss = model_loss(&out, &targets);
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        let g = grads
            .get::<Inner, 2>(id)
            .expect("gradient for shared core weight");
        g.into_data().to_vec::<f32>().unwrap()[0]
    };

    let g1 = grad_at(1);
    let g4 = grad_at(4);
    assert!(g1.is_finite() && g4.is_finite(), "grads must be finite");
    assert!(g1.abs() > 1e-8, "R=1 shared-core gradient must be nonzero");
    assert!(g4.abs() > 1e-8, "R=4 shared-core gradient must be nonzero");
    assert!(
        (g1 - g4).abs() > 1e-7,
        "later recurrent uses must change the shared-core gradient (g1={g1}, g4={g4})"
    );

    // Finite-difference check of the R=4 analytic gradient.
    let w0 = model.core_weight_scalar();
    let eps = 1e-2f32;
    let loss_val = |w: f32| -> f32 {
        let m = model.with_core_weight_scalar(w);
        let out = m.forward_r(board.clone(), &cands, 4, false);
        model_loss(&out, &targets)
            .into_data()
            .to_vec::<f32>()
            .unwrap()[0]
    };
    let numeric = (loss_val(w0 + eps) - loss_val(w0 - eps)) / (2.0 * eps);
    let denom = numeric.abs().max(g4.abs()).max(1e-6);
    let rel = (numeric - g4).abs() / denom;
    assert!(
        rel < 0.25,
        "analytic {g4} vs numeric {numeric} (relative error {rel})"
    );
}

/// Evaluating position A must not depend on other positions in the batch:
/// there is no persistent hidden state between independent positions.
#[test]
fn batch_matches_single_item() {
    type B = burn::backend::Flex;
    let device = Default::default();
    let model = ProbeModel::<B>::new(small_cfg(), &device);
    let fx = SynthFixture::new(4, 119, 3);
    let (board, cands, _t) = fx.tensors::<B>(&device);
    let w = cands.width;

    let full = log_probs(&model.forward_r(board, &cands, 2, false));

    for i in 0..fx.batch {
        if fx.lists[i].is_empty() {
            // Terminal positions carry no policy path; nothing to compare.
            continue;
        }
        let len = fx.lists[i].len();
        let row = fx.row(i);
        let (b1, c1, _t) = row.tensors::<B>(&device);
        let single = log_probs(&model.forward_r(b1, &c1, 2, false));
        assert_eq!(c1.width, len);
        for k in 0..len {
            let x = full[i * w + k];
            let y = single[k];
            assert!(
                (x - y).abs() < 1e-5,
                "row {i} candidate {k}: batch {x} vs single {y}"
            );
        }
    }
}

/// Shared parameters are counted once regardless of recurrence count.
#[test]
fn parameter_count_independent_of_recurrence() {
    type B = burn::backend::Flex;
    let device = Default::default();
    let model = ProbeModel::<B>::new(small_cfg(), &device);
    let n = model.num_params();
    assert_eq!(model.core_block_count(), 2);
    assert!(n > 0);
    // Executing more loops must not change the stored parameter count.
    let fx = SynthFixture::new(2, 119, 9);
    let (board, cands, _t) = fx.tensors::<B>(&device);
    let _ = model.forward_r(board, &cands, 4, false);
    assert_eq!(model.num_params(), n);
}
