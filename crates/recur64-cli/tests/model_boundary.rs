//! Phase 1 neural boundary: real chess data feeds the existing Phase 0 model
//! API without any model change and without training.
//!
//! CPU FP32 only. This proves contract compatibility, not move quality.

use burn::backend::Flex;
use burn::prelude::*;
use burn::tensor::TensorData;

use recur64_core::{GameState, OBS_LEN, encode_observation_v1};
use recur64_model::action::CandidateBatch;
use recur64_model::config::ModelConfig;
use recur64_model::model::{CandidateTensors, ProbeModel};

fn micro_cfg() -> ModelConfig {
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

/// Canonical `(from, to, promo_code)` tuples for the sparse policy path.
fn canonical_tuples(g: &GameState) -> Vec<(u32, u32, u8)> {
    let p = g.perspective();
    g.legal_actions()
        .iter()
        .map(|id| {
            let (f, t, promo) = id.to_physical(p);
            (f as u32, t as u32, promo.code())
        })
        .collect()
}

fn forward_real_position(fen: &str) {
    let device = Default::default();
    let g = GameState::from_fen(fen).unwrap();
    let model = ProbeModel::<Flex>::new(micro_cfg(), &device);

    // Observation -> [1, 64, 119] tensor.
    let obs = encode_observation_v1(&g);
    assert_eq!(obs.as_slice().len(), OBS_LEN);
    let board = Tensor::<Flex, 3>::from_data(
        TensorData::new(obs.as_slice().to_vec(), [1, 64, 119]),
        &device,
    );

    // Legal actions -> padded candidate batch (canonical orientation).
    let tuples = canonical_tuples(&g);
    assert!(
        !tuples.is_empty(),
        "non-terminal position must have candidates"
    );
    let cb = CandidateBatch::from_lists(std::slice::from_ref(&tuples));
    assert_eq!(cb.width, tuples.len());
    let cands = CandidateTensors::from_batch(&cb, &device);

    let out = model.forward_r(board, &cands, 1, false);
    assert_eq!(out.readouts.len(), 1);

    // Policy probabilities over the legal candidates sum to 1.
    let log_probs = out.readouts[0]
        .policy
        .log_probs
        .clone()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let sum: f32 = (0..tuples.len()).map(|k| log_probs[k].exp()).sum();
    assert!((sum - 1.0).abs() < 1e-4, "policy must normalize, got {sum}");

    // WDL logits finite.
    let wdl = out.readouts[0]
        .wdl_logits
        .clone()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    assert_eq!(wdl.len(), 3);
    assert!(wdl.iter().all(|v| v.is_finite()));
}

#[test]
fn real_positions_feed_the_phase0_model() {
    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/P6k/8/8/8/8/8/K7 w - - 0 1", // promotion position
    ] {
        forward_real_position(fen);
    }
}

#[test]
fn terminal_positions_bypass_the_policy_softmax() {
    let g = GameState::from_fen("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
        .unwrap();
    assert!(g.is_terminal());
    let tuples = canonical_tuples(&g);
    assert!(tuples.is_empty());
    let cb = CandidateBatch::from_lists(&[tuples]);
    assert_eq!(cb.width, 0);
    assert!(cb.terminal[0], "terminal row must be flagged for bypass");
    // The caller must not feed a terminal row through the policy path; the
    // model asserts on width 0 rather than computing an all-masked softmax.
}
