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
        openings: Vec::new(),
        concurrency: 1,
        ..ArenaConfig::default()
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

#[test]
fn concurrent_arena_matches_sequential_and_flags_uninformative() {
    let reference = FixedEvaluator::uniform(0.0);
    let candidate = FixedEvaluator::uniform(0.0);
    let mut c = cfg();
    c.games = 6;
    c.ply_cap = 60;
    c.openings = vec![
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".into(),
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1".into(),
    ];
    let seq = run_arena(&reference, &candidate, "ref", "cand", &c).unwrap();
    c.concurrency = 4;
    let par = run_arena(&reference, &candidate, "ref", "cand", &c).unwrap();
    assert_eq!(
        serde_json_like(&seq),
        serde_json_like(&par),
        "concurrency must not change the report"
    );
    assert_eq!(seq.decisive_games, seq.candidate_wins + seq.reference_wins);
    assert_eq!(seq.informative, seq.decisive_games > 0);

    // Every game truncated at ply 4: numerically 0.5, but not informative.
    let r = run_arena(&reference, &candidate, "ref", "cand", &cfg()).unwrap();
    assert_eq!(r.truncated, 4);
    assert_eq!(r.candidate_score, 0.5);
    assert!(!r.informative);
    assert_eq!(r.decisive_games, 0);
}

fn serde_json_like(r: &recur64_eval::ArenaResult) -> String {
    format!(
        "{} {} {} {} {:?} {} {} {}",
        r.candidate_wins,
        r.reference_wins,
        r.draws,
        r.truncated,
        r.terminations,
        r.candidate_score,
        r.score_ci_low,
        r.score_ci_high
    )
}

/// D45: a sampled arena opening phase is reproducible from the seed and
/// independent of concurrency, like the deterministic arena.
#[test]
fn sampled_arena_is_reproducible_and_concurrency_independent() {
    let reference = FixedEvaluator::uniform(0.0);
    let candidate = FixedEvaluator::uniform(0.0);
    let mut c = cfg();
    c.games = 6;
    c.ply_cap = 60;
    c.sample_plies = Some(8);
    let a = run_arena(&reference, &candidate, "ref", "cand", &c).unwrap();
    let b = run_arena(&reference, &candidate, "ref", "cand", &c).unwrap();
    assert_eq!(serde_json_like(&a), serde_json_like(&b));
    c.concurrency = 3;
    let par = run_arena(&reference, &candidate, "ref", "cand", &c).unwrap();
    assert_eq!(serde_json_like(&a), serde_json_like(&par));
}

/// D38: no evaluation game starts after the deadline, and an incomplete
/// evaluation is an explicit error, never a partial (misleading) result.
#[test]
fn arena_past_deadline_is_an_explicit_incomplete_error() {
    let reference = FixedEvaluator::uniform(0.0);
    let candidate = FixedEvaluator::uniform(0.0);
    for concurrency in [1usize, 3] {
        let mut c = cfg();
        c.concurrency = concurrency;
        c.deadline = Some(std::time::Instant::now() - std::time::Duration::from_secs(1));
        let err = run_arena(&reference, &candidate, "ref", "cand", &c).unwrap_err();
        assert!(
            matches!(err, recur64_search::EvalError::DeadlineExceeded(_)),
            "{err}"
        );
    }
    // A future deadline changes nothing.
    let mut c = cfg();
    c.deadline = Some(std::time::Instant::now() + std::time::Duration::from_secs(3600));
    let with = run_arena(&reference, &candidate, "ref", "cand", &c).unwrap();
    let without = run_arena(&reference, &candidate, "ref", "cand", &cfg()).unwrap();
    assert_eq!(serde_json_like(&with), serde_json_like(&without));
}
