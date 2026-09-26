//! H3.5B: common random numbers for the color-swapped arena pair.
//!
//! Under the D45 stochastic arena (sampled opening plies + root noise) the
//! historical per-game seeds give the two games of an opening pair different
//! random streams, so even a model against itself does not mirror. Under
//! `paired_common_v1`, one deterministic evaluator on both sides plays the
//! same game twice with colors swapped, so every pair scores exactly 0.5.
//!
//! Deterministic here: CPU Flex, one game at a time, one leaf per round, so
//! every forward pass sees a batch of one. No CUDA-bitwise claim is made.

use burn::backend::Flex;
use burn::prelude::Backend;

use recur64_eval::{ArenaConfig, ArenaRngPolicy, arena_game_seed, run_arena};
use recur64_model::config::ModelConfig;
use recur64_model::model::ProbeModel;
use recur64_runtime::SyncEvaluator;
use recur64_search::FixedEvaluator;

fn tiny() -> ModelConfig {
    ModelConfig {
        width: 32,
        heads: 4,
        ffn: 64,
        input_blocks: 0,
        core_blocks: 1,
        output_blocks: 0,
        squares: 64,
        in_features: 119,
        policy_dim: 16,
        wdl_classes: 3,
        promo_codes: 5,
        rms_eps: 1e-5,
    }
}

fn stochastic(policy: ArenaRngPolicy, concurrency: usize) -> ArenaConfig {
    ArenaConfig {
        games: 8,
        simulations: 64,
        // Simple endgames so random-ish play still ends (mate, stalemate,
        // insufficient material or the fifty-move rule) before the cap.
        ply_cap: 240,
        seed: 11,
        openings: vec![
            "8/8/8/4k3/8/8/8/4KQ2 w - - 0 1".into(),
            "8/8/8/4k3/8/8/8/R3K3 w - - 0 1".into(),
            // Back-rank mate in one for White: a decisive, mirrored pair.
            "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1".into(),
            "r5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1".into(),
        ],
        concurrency,
        sample_plies: Some(30),
        root_dirichlet_alpha: 0.3,
        root_dirichlet_epsilon: 0.25,
        rng_policy: policy,
        ..ArenaConfig::default()
    }
}

#[test]
fn seed_assignment_is_explicit_per_policy() {
    let seeds = |p| {
        (0..6)
            .map(|i| arena_game_seed(p, 100, i))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        seeds(ArenaRngPolicy::PerGameV1),
        [100, 101, 102, 103, 104, 105]
    );
    assert_eq!(
        seeds(ArenaRngPolicy::PairedCommonV1),
        [100, 100, 101, 101, 102, 102]
    );
    assert_eq!(ArenaRngPolicy::default(), ArenaRngPolicy::PerGameV1);
}

#[test]
fn paired_common_rng_mirrors_identical_evaluators() {
    let device = Default::default();
    <Flex as Backend>::seed(&device, 5);
    let model = ProbeModel::<Flex>::new(tiny(), &device);
    let ev = SyncEvaluator::new(model, 1, device);

    let paired = run_arena(
        &ev,
        &ev,
        "m",
        "m",
        &stochastic(ArenaRngPolicy::PairedCommonV1, 1),
    )
    .unwrap();
    for (a, b) in paired.game_records.chunks(2).map(|c| (&c[0], &c[1])) {
        assert_eq!(a.seed, b.seed, "pair {} shares one stream", a.pair);
        assert_eq!(a.moves_digest, b.moves_digest, "pair {} replays", a.pair);
        assert_eq!(a.winner, b.winner);
    }
    println!(
        "paired_common_v1: terminations {:?}, decisive {}",
        paired.terminations, paired.decisive_games
    );
    let d = &paired.pairs;
    assert_eq!(d.pairs, 4);
    assert_eq!(d.identical_move_pairs, 4);
    assert_eq!(d.mirrored_pairs, d.complete_pairs);
    assert!(d.complete_pairs > 0, "fixture must produce completed pairs");
    assert!(
        paired.decisive_games > 0,
        "fixture must produce decisive pairs"
    );
    assert_eq!(
        d.split_pairs * 2,
        paired.decisive_games,
        "decisive games mirror"
    );
    assert_eq!(d.mean_pair_score, 0.5);
    if paired.truncated == 0 {
        assert_eq!(paired.candidate_score, 0.5);
    }

    // The historical policy under the same model: record, don't assert.
    let per_game = run_arena(
        &ev,
        &ev,
        "m",
        "m",
        &stochastic(ArenaRngPolicy::PerGameV1, 1),
    )
    .unwrap();
    println!(
        "per_game_v1 self-play: identical_move_pairs {}/{}, mirrored {}/{} complete, score {:.3}",
        per_game.pairs.identical_move_pairs,
        per_game.pairs.pairs,
        per_game.pairs.mirrored_pairs,
        per_game.pairs.complete_pairs,
        per_game.candidate_score
    );
    assert_ne!(per_game.game_records[0].seed, per_game.game_records[1].seed);
}

#[test]
fn paired_policy_is_concurrency_independent() {
    let reference = FixedEvaluator::uniform(0.0);
    let candidate = FixedEvaluator::uniform(0.0);
    let one = run_arena(
        &reference,
        &candidate,
        "r",
        "c",
        &stochastic(ArenaRngPolicy::PairedCommonV1, 1),
    )
    .unwrap();
    let four = run_arena(
        &reference,
        &candidate,
        "r",
        "c",
        &stochastic(ArenaRngPolicy::PairedCommonV1, 4),
    )
    .unwrap();
    assert_eq!(one.game_records, four.game_records);
    assert_eq!(one.pairs, four.pairs);
    assert_eq!(one.pairs.identical_move_pairs, one.pairs.pairs);
}
