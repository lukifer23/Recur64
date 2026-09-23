//! T2.11: the arena runs paired-color games with fixed budget.

use burn::backend::Flex;

use recur64_eval::{ArenaConfig, run_arena};
use recur64_model::config::ModelConfig;
use recur64_model::model::ProbeModel;
use recur64_runtime::SyncEvaluator;

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
fn arena_runs_paired_games() {
    let device = Default::default();
    let reference =
        SyncEvaluator::new(ProbeModel::<Flex>::new(micro(), &device), 1, device);
    let candidate = SyncEvaluator::new(ProbeModel::<Flex>::new(micro(), &device), 1, device);
    let cfg = ArenaConfig {
        games: 2,
        simulations: 2,
        c_puct: 1.0,
        recurrence: 1,
        ply_cap: 4,
        seed: 1,
    };
    let r = run_arena(&reference, &candidate, "ref", "cand", &cfg).unwrap();
    assert_eq!(r.games, 2);
    assert_eq!(
        r.candidate_wins + r.reference_wins + r.draws + r.truncated,
        2
    );
    assert!((0.0..=1.0).contains(&r.candidate_score));
    assert!(!r.terminations.is_empty());
    assert_eq!(r.model_reference, "ref");
    assert_eq!(r.model_candidate, "cand");
}

#[test]
fn arena_is_reproducible_from_seed() {
    let device = Default::default();
    let reference =
        SyncEvaluator::new(ProbeModel::<Flex>::new(micro(), &device), 1, device);
    let candidate = SyncEvaluator::new(ProbeModel::<Flex>::new(micro(), &device), 1, device);
    let cfg = ArenaConfig {
        games: 2,
        simulations: 2,
        c_puct: 1.0,
        recurrence: 1,
        ply_cap: 4,
        seed: 99,
    };
    let a = run_arena(&reference, &candidate, "ref", "cand", &cfg).unwrap();
    let b = run_arena(&reference, &candidate, "ref", "cand", &cfg).unwrap();
    assert_eq!(a.candidate_wins, b.candidate_wins);
    assert_eq!(a.reference_wins, b.reference_wins);
    assert_eq!(a.draws, b.draws);
    assert_eq!(a.truncated, b.truncated);
}

