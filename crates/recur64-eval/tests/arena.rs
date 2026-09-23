//! T2.11: the arena runs paired-color games with a fixed budget.

use recur64_eval::{ArenaConfig, run_arena};
use recur64_search::FixedEvaluator;

fn cfg() -> ArenaConfig {
    ArenaConfig {
        games: 4,
        simulations: 2,
        c_puct: 1.0,
        recurrence: 1,
        ply_cap: 4,
        seed: 7,
    }
}

#[test]
fn arena_runs_paired_games() {
    // Two deterministic evaluators; with a 4-ply cap the games truncate, which
    // still exercises the full paired-color accounting path.
    let reference = FixedEvaluator::uniform(0.0);
    let candidate = FixedEvaluator::uniform(0.0);
    let r = run_arena(&reference, &candidate, "ref", "cand", &cfg()).unwrap();
    assert_eq!(r.games, 4);
    assert_eq!(
        r.candidate_wins + r.reference_wins + r.draws + r.truncated,
        4
    );
    assert!((0.0..=1.0).contains(&r.candidate_score));
    assert!(!r.terminations.is_empty());
    assert_eq!(r.model_reference, "ref");
    assert_eq!(r.model_candidate, "cand");
}

#[test]
fn arena_is_reproducible_from_seed() {
    let reference = FixedEvaluator::uniform(0.0);
    let candidate = FixedEvaluator::uniform(0.0);
    let a = run_arena(&reference, &candidate, "ref", "cand", &cfg()).unwrap();
    let b = run_arena(&reference, &candidate, "ref", "cand", &cfg()).unwrap();
    assert_eq!(a.candidate_wins, b.candidate_wins);
    assert_eq!(a.reference_wins, b.reference_wins);
    assert_eq!(a.draws, b.draws);
    assert_eq!(a.truncated, b.truncated);
    assert_eq!(a.terminations, b.terminations);
}
