//! A freshly initialized F10 must start from a near-uninformative prior.
//!
//! PUCT with an uninformative value head allocates visits roughly in
//! proportion to the prior, so a confident-but-arbitrary initial policy is
//! reproduced as the search target and self-play distills random
//! preferences (measured in Phase 4 P4.4: KL(target || prior) fell as the
//! budget rose). The initial policy must therefore be close to uniform over
//! legal moves and the initial value close to neutral.

use burn::backend::Flex;
use burn::prelude::*;

use recur64_core::{GameState, encode_observation_v1};
use recur64_eval::OpeningSuite;
use recur64_model::config::ModelConfig;
use recur64_model::model::ProbeModel;
use recur64_runtime::SyncEvaluator;
use recur64_search::{EvalRequest, Evaluator};

fn f10() -> ModelConfig {
    ModelConfig {
        width: 384,
        heads: 12,
        ffn: 768,
        input_blocks: 0,
        core_blocks: 8,
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
fn fresh_f10_prior_is_near_uniform_and_value_near_neutral() {
    let suite = OpeningSuite::load(std::path::Path::new("../../configs/openings-v1.toml"))
        .expect("openings-v1");
    let mut fens = suite.openings.clone();
    fens.push(GameState::startpos().to_fen());
    let device = Default::default();
    let (mut ratio_sum, mut abs_value_sum, mut abs_value_max, mut n) = (0.0f64, 0.0f64, 0.0f64, 0);
    for seed in [1u64, 2, 3] {
        <Flex as Backend>::seed(&device, seed);
        let ev = SyncEvaluator::new(ProbeModel::<Flex>::new(f10(), &device), 1, device);
        for fen in &fens {
            let state = GameState::from_fen(fen).expect("opening fen");
            let legal = state.legal_actions();
            let obs = encode_observation_v1(&state);
            let r = ev
                .evaluate(EvalRequest {
                    observation: &obs,
                    legal: &legal,
                    side_to_move: state.side_to_move(),
                })
                .expect("evaluate");
            let h: f64 = -r
                .policy
                .iter()
                .filter(|&&p| p > 0.0)
                .map(|&p| p as f64 * (p as f64).ln())
                .sum::<f64>();
            ratio_sum += h / (legal.len() as f64).ln();
            abs_value_sum += r.value.abs() as f64;
            abs_value_max = abs_value_max.max(r.value.abs() as f64);
            n += 1;
        }
    }
    let ratio = ratio_sum / n as f64;
    let abs_value = abs_value_sum / n as f64;
    println!(
        "fresh F10 over {n} positions: entropy/uniform = {ratio:.3}, mean |value| = {abs_value:.3}, max |value| = {abs_value_max:.3}"
    );
    assert!(
        ratio >= 0.9,
        "initial policy too confident: entropy/uniform = {ratio:.3}"
    );
    assert!(
        abs_value <= 0.1,
        "initial value not neutral: mean |value| = {abs_value:.3}"
    );
}
