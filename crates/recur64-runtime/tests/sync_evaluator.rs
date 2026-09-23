//! T2.4: the synchronous neural evaluator returns aligned policy/value.

use burn::backend::Flex;

use recur64_core::{GameState, encode_observation_v1};
use recur64_model::config::ModelConfig;
use recur64_model::model::ProbeModel;
use recur64_runtime::SyncEvaluator;
use recur64_search::{EvalRequest, Evaluator};

fn micro() -> ModelConfig {
    ModelConfig {
        width: 192,
        heads: 6,
        ffn: 384,
        input_blocks: 0,
        core_blocks: 4,
        output_blocks: 0,
        squares: 64,
        in_features: 119,
        policy_dim: 128,
        wdl_classes: 3,
        promo_codes: 5,
        rms_eps: 1e-5,
    }
}

#[test]
fn policy_is_aligned_and_normalized() {
    let device = Default::default();
    let model = ProbeModel::<Flex>::new(micro(), &device);
    let ev = SyncEvaluator::new(model, 1, device);

    let state = GameState::startpos();
    let legal = state.legal_actions();
    let obs = encode_observation_v1(&state);
    let r = ev
        .evaluate(EvalRequest {
            observation: &obs,
            legal: &legal,
            side_to_move: recur64_core::Color::White,
        })
        .unwrap();

    assert_eq!(r.policy.len(), legal.len());
    assert!((r.policy.iter().sum::<f32>() - 1.0).abs() < 1e-4);
    assert!(r.policy.iter().all(|p| *p >= 0.0 && p.is_finite()));
    assert!(r.value.is_finite() && (-1.0..=1.0).contains(&r.value));
    assert!((r.wdl.iter().sum::<f32>() - 1.0).abs() < 1e-4);
    // value = P(win) - P(loss)
    assert!((r.value - (r.wdl[0] - r.wdl[2])).abs() < 1e-5);
}

#[test]
fn evaluation_is_deterministic() {
    let device = Default::default();
    let model = ProbeModel::<Flex>::new(micro(), &device);
    let ev = SyncEvaluator::new(model, 1, device);
    let state = GameState::startpos();
    let legal = state.legal_actions();
    let obs = encode_observation_v1(&state);
    let a = ev
        .evaluate(EvalRequest {
            observation: &obs,
            legal: &legal,
            side_to_move: recur64_core::Color::White,
        })
        .unwrap();
    let b = ev
        .evaluate(EvalRequest {
            observation: &obs,
            legal: &legal,
            side_to_move: recur64_core::Color::White,
        })
        .unwrap();
    assert_eq!(a.policy, b.policy);
    assert_eq!(a.value, b.value);
}

#[test]
fn empty_legal_is_rejected() {
    let device = Default::default();
    let model = ProbeModel::<Flex>::new(micro(), &device);
    let ev = SyncEvaluator::new(model, 1, device);
    let obs = encode_observation_v1(&GameState::startpos());
    let err = ev.evaluate(EvalRequest {
        observation: &obs,
        legal: &[],
        side_to_move: recur64_core::Color::White,
    });
    assert!(err.is_err());
}
