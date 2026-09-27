//! D56 `early_material_v1`: +5 material for 40 consecutive plies.

use recur64_eval::{ArenaConfig, ArenaEarlyAdjudication, run_arena};
use recur64_search::FixedEvaluator;

/// White has K+Q vs a bare king from ply 0, so the rule fires at ply 40 for
/// White unless the game ends first.
fn cfg(mode: ArenaEarlyAdjudication) -> ArenaConfig {
    ArenaConfig {
        games: 2,
        simulations: 16,
        ply_cap: 160,
        // Sampled play: deterministic argmax with a uniform evaluator
        // repeats within ~14 plies, before the 40-ply window can fill.
        sample_plies: Some(400),
        root_dirichlet_epsilon: 0.5,
        seed: 3,
        openings: vec!["8/8/8/4k3/8/8/8/3QK3 w - - 0 1".into()],
        early_adjudication: mode,
        ..ArenaConfig::default()
    }
}

#[test]
fn off_by_default_and_off_records_nothing() {
    assert_eq!(
        ArenaConfig::default().early_adjudication,
        ArenaEarlyAdjudication::Off
    );
    let (r, c) = (FixedEvaluator::uniform(0.0), FixedEvaluator::uniform(0.0));
    let a = run_arena(&r, &c, "r", "c", &cfg(ArenaEarlyAdjudication::Off)).unwrap();
    assert!(a.early_adjudication_summary.is_none());
    assert!(
        a.game_records
            .iter()
            .all(|g| g.early_adjudication.is_none())
    );
}

#[test]
fn shadow_observes_without_changing_games() {
    let (r, c) = (FixedEvaluator::uniform(0.0), FixedEvaluator::uniform(0.0));
    let off = run_arena(&r, &c, "r", "c", &cfg(ArenaEarlyAdjudication::Off)).unwrap();
    let shadow = run_arena(&r, &c, "r", "c", &cfg(ArenaEarlyAdjudication::Shadow)).unwrap();
    for (a, b) in off.game_records.iter().zip(&shadow.game_records) {
        assert_eq!(
            a.moves_digest, b.moves_digest,
            "shadow must not change play"
        );
        assert_eq!(a.candidate_score, b.candidate_score);
    }
    let s = shadow.early_adjudication_summary.unwrap();
    assert!(s.fired_games > 0, "fixture must make the rule fire");
    assert_eq!(
        s.total_plies,
        off.game_records.iter().map(|g| g.plies as u64).sum()
    );
    for g in &shadow.game_records {
        if let Some(e) = g.early_adjudication {
            assert!(!e.enforced);
            assert!(e.fired_at_ply >= 40, "needs 40 consecutive plies");
            // White (the queen side) leads: candidate is White in even games.
            assert_eq!(e.leader_is_candidate, g.candidate_white);
        }
    }
}

#[test]
fn enforce_ends_games_at_the_firing_ply_with_the_leader_winning() {
    let (r, c) = (FixedEvaluator::uniform(0.0), FixedEvaluator::uniform(0.0));
    let a = run_arena(&r, &c, "r", "c", &cfg(ArenaEarlyAdjudication::Enforce)).unwrap();
    let s = a.early_adjudication_summary.unwrap();
    assert!(s.fired_games > 0, "fixture must make the rule fire");
    for g in &a.game_records {
        if let Some(e) = g.early_adjudication {
            assert!(e.enforced);
            assert_eq!(g.plies as u32, e.fired_at_ply, "game ends at firing");
            let leader = if e.leader_is_candidate { 1.0 } else { 0.0 };
            assert_eq!(g.adjudicated_score, leader);
            assert_eq!(g.termination, "truncated");
        }
    }
    assert_eq!(
        a.terminations
            .get("early_adjudicated")
            .copied()
            .unwrap_or(0),
        s.fired_games
    );
    assert_eq!(s.plies_after_firing, 0);
}
