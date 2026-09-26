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

fn r10() -> ModelConfig {
    ModelConfig {
        input_blocks: 2,
        core_blocks: 4,
        output_blocks: 2,
        ..f10()
    }
}

/// HP feed-forward control: 512/8/768, `0 + 8 + 0`.
fn f15() -> ModelConfig {
    ModelConfig {
        width: 512,
        heads: 8,
        ffn: 768,
        ..f10()
    }
}

/// HP shared recurrent core: 512/8/768, `2 + 4 + 2`.
fn r15() -> ModelConfig {
    ModelConfig {
        input_blocks: 2,
        core_blocks: 4,
        output_blocks: 2,
        ..f15()
    }
}

/// (mean entropy / uniform, mean |value|, max |value|, positions) of fresh
/// networks over three seeds and the openings-v1 suite plus startpos.
fn fresh_stats(cfg: ModelConfig, recurrence: usize) -> (f64, f64, f64, usize) {
    let suite = OpeningSuite::load(std::path::Path::new("../../configs/openings-v1.toml"))
        .expect("openings-v1");
    let mut fens = suite.openings.clone();
    fens.push(GameState::startpos().to_fen());
    let device = Default::default();
    let (mut ratio_sum, mut abs_value_sum, mut abs_value_max, mut n) = (0.0f64, 0.0f64, 0.0f64, 0);
    for seed in [1u64, 2, 3] {
        <Flex as Backend>::seed(&device, seed);
        let ev = SyncEvaluator::new(
            ProbeModel::<Flex>::new(cfg.clone(), &device),
            recurrence,
            device,
        );
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
    (
        ratio_sum / n as f64,
        abs_value_sum / n as f64,
        abs_value_max,
        n,
    )
}

#[test]
fn fresh_f10_prior_is_near_uniform_and_value_near_neutral() {
    let (ratio, abs_value, abs_value_max, n) = fresh_stats(f10(), 1);
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

/// Head v2 also removes the layout-dependent head-input scale, so a fresh
/// R10 must start sane as well. R1 is asserted; R2/R4 are diagnostics.
#[test]
fn fresh_r10_prior_is_near_uniform_and_value_near_neutral() {
    for r in [1usize, 2, 4] {
        let (ratio, abs_value, abs_value_max, n) = fresh_stats(r10(), r);
        println!(
            "fresh R10 R{r} over {n} positions: entropy/uniform = {ratio:.3}, mean |value| = {abs_value:.3}, max |value| = {abs_value_max:.3}"
        );
        if r == 1 {
            assert!(ratio >= 0.9, "R10 R1 prior too confident: {ratio:.3}");
            assert!(abs_value <= 0.1, "R10 R1 value not neutral: {abs_value:.3}");
        }
    }
}

/// H3 H3.1: the F15 head-v2 reference must start from a near-uniform policy and
/// a neutral value. This is a sanity gate, not a strength claim.
#[test]
fn fresh_f15_prior_is_near_uniform_and_value_near_neutral() {
    let (ratio, abs_value, abs_value_max, n) = fresh_stats(f15(), 1);
    println!(
        "fresh F15 R1 over {n} positions: entropy/uniform = {ratio:.3}, mean |value| = {abs_value:.3}, max |value| = {abs_value_max:.3}"
    );
    assert!(
        ratio >= 0.9,
        "F15 R1 prior too confident: entropy/uniform = {ratio:.3}"
    );
    assert!(
        abs_value <= 0.1,
        "F15 R1 value not neutral: mean |value| = {abs_value:.3}"
    );
}

/// H3 H3.1: head-v2 sanity must hold across the R15 recurrence ladder. R1 is
/// the structural baseline; R2/R4 must also stay sane, since a layout-dependent
/// head-input scale (the head-v1 defect) would have destabilised deeper R.
#[test]
fn fresh_r15_prior_is_near_uniform_and_value_near_neutral_across_recurrence() {
    for r in [1usize, 2, 4] {
        let (ratio, abs_value, abs_value_max, n) = fresh_stats(r15(), r);
        println!(
            "fresh R15 R{r} over {n} positions: entropy/uniform = {ratio:.3}, mean |value| = {abs_value:.3}, max |value| = {abs_value_max:.3}"
        );
        assert!(ratio >= 0.9, "R15 R{r} prior too confident: {ratio:.3}");
        assert!(
            abs_value <= 0.1,
            "R15 R{r} value not neutral: {abs_value:.3}"
        );
    }
}
