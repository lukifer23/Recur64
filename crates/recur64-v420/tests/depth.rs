//! V420 P1 CPU correctness (no result, no training): the recurrence invariants the AGENTS.md rules
//! and the plan require before any run. Tiny geometry; the real geometry is exercised by the CUDA smoke.

use burn::module::AutodiffModule;
use burn::optim::GradientsParams;
use burn::prelude::*;
use recur64_core::GameState;
use recur64_model::candidate::CandidateInputs;
use recur64_model::train::CpuTrainBackend;
use recur64_v420::diag::group_norms;
use recur64_v420::model::{DepthModel, Geometry};

type Ad = CpuTrainBackend;
type Inner = <CpuTrainBackend as burn::tensor::backend::AutodiffBackend>::InnerBackend;

fn states() -> Vec<GameState> {
    [
        "4k3/8/8/8/8/8/8/R3K3 w - - 0 1",
        "8/8/8/3k4/8/8/4K3/Q7 w - - 0 1",
        "7k/8/6K1/8/8/8/8/R7 w - - 0 1",
    ]
    .iter()
    .map(|f| GameState::from_fen(f).expect("valid fen"))
    .collect()
}

fn loss(model: &DepthModel<Ad>, k: usize) -> Tensor<Ad, 1> {
    let dev = Default::default();
    let inputs = CandidateInputs::<Ad>::from_states(&states(), &dev).unwrap();
    let z = model.logits(&inputs, k).unwrap();
    // A fixed non-degenerate scalar of the masked logits (pad entries are -1e9 and excluded).
    let valid = inputs.cands.mask.clone().float();
    (burn::tensor::activation::log_softmax(z, 1).mul(valid)).sum().neg()
}

#[test]
fn parameter_count_is_independent_of_k_and_wide_has_more() {
    let dev = Default::default();
    let shared = DepthModel::<Ad>::new(&Geometry::tiny(1), &dev);
    let wide = DepthModel::<Ad>::new(&Geometry::tiny(4), &dev);
    assert!(wide.num_params_total() > shared.num_params_total());
    // K is not a parameter: the same model evaluates at any K and its count never changes.
    let n = shared.num_params_total();
    for k in [1, 2, 4, 8] {
        let _ = loss(&shared, k);
        assert_eq!(shared.num_params_total(), n);
    }
}

#[test]
fn depth_changes_the_function_of_a_shared_block() {
    let dev = Default::default();
    let m = DepthModel::<Inner>::new(&Geometry::tiny(1), &dev);
    let inputs = CandidateInputs::<Inner>::from_states(&states(), &dev).unwrap();
    let z1: Vec<f32> = m.logits(&inputs, 1).unwrap().into_data().to_vec().unwrap();
    let z2: Vec<f32> = m.logits(&inputs, 2).unwrap().into_data().to_vec().unwrap();
    let z1b: Vec<f32> = m.logits(&inputs, 1).unwrap().into_data().to_vec().unwrap();
    assert_eq!(z1, z1b, "K=1 is deterministic");
    assert!(
        z1.iter().zip(&z2).any(|(a, b)| (a - b).abs() > 1e-9),
        "K=2 must differ from K=1 (the block is actually applied twice)"
    );
    assert!(z1.iter().chain(&z2).all(|v| v.is_finite()));
}

#[test]
fn the_shared_block_receives_gradient_at_every_k_and_the_gradient_depends_on_k() {
    let dev = Default::default();
    let model = DepthModel::<Ad>::new(&Geometry::tiny(1), &dev);
    let mut norms = Vec::new();
    for k in [1usize, 2, 4] {
        let l = loss(&model, k);
        let grads = GradientsParams::from_grads(l.backward(), &model);
        let g = group_norms(&model, &grads);
        let blocks = g.get("blocks").expect("blocks group");
        assert_eq!(
            blocks.tensors_with_grad, blocks.tensors,
            "K={k}: every shared-block tensor must receive a gradient"
        );
        assert!(blocks.grad_norm.is_finite() && blocks.grad_norm > 0.0, "K={k}: grad {}", blocks.grad_norm);
        norms.push(blocks.grad_norm);
    }
    assert!(
        (norms[0] - norms[1]).abs() > 1e-9 && (norms[1] - norms[2]).abs() > 1e-9,
        "the shared-block gradient must change with recurrence depth: {norms:?}"
    );
}

#[test]
fn unshared_models_execute_exactly_their_block_count() {
    let dev = Default::default();
    let wide = DepthModel::<Inner>::new(&Geometry::tiny(4), &dev);
    let inputs = CandidateInputs::<Inner>::from_states(&states(), &dev).unwrap();
    assert!(wide.logits(&inputs, 4).is_ok());
    assert!(wide.logits(&inputs, 2).is_err(), "an unshared 4-block model refuses K=2");
    assert!(wide.logits(&inputs, 8).is_err(), "an unshared 4-block model refuses K=8");
}

#[test]
fn the_inference_copy_matches_the_autodiff_model() {
    let dev = Default::default();
    let model = DepthModel::<Ad>::new(&Geometry::tiny(1), &dev);
    let inputs_ad = CandidateInputs::<Ad>::from_states(&states(), &dev).unwrap();
    let a: Vec<f32> = model.logits(&inputs_ad, 3).unwrap().into_data().to_vec().unwrap();
    let inner = model.valid();
    let inputs_in = CandidateInputs::<Inner>::from_states(&states(), &dev).unwrap();
    let b: Vec<f32> = inner.logits(&inputs_in, 3).unwrap().into_data().to_vec().unwrap();
    assert_eq!(a.len(), b.len());
    assert!(a.iter().zip(&b).all(|(x, y)| (x - y).abs() < 1e-5));
}
