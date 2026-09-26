//! R15-P0.1: `material_v1` truncation adjudication.
//!
//! The pre-registered validation re-scores the H3.6 cycle-2 parent-arena
//! replay (the only arena with recorded final positions): 12 / 13 / 7, score
//! (12 + 6.5) / 32 = 0.578, versus 0.643 as played.

use recur64_eval::{adjudicate_truncated, material_balance_white};

#[test]
fn material_balance_and_thresholds() {
    assert_eq!(
        material_balance_white("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        Some(0)
    );
    // K+R vs K: +5 wins; K+B+N vs K: +6 wins; K+R vs K+B: +2 draws.
    assert_eq!(
        adjudicate_truncated("8/8/8/4k3/8/8/8/R3K3 w - - 0 1", true),
        1.0
    );
    assert_eq!(
        adjudicate_truncated("8/8/8/4k3/8/8/8/R3K3 w - - 0 1", false),
        0.0
    );
    assert_eq!(
        adjudicate_truncated("8/8/8/4k3/8/8/8/BN2K3 w - - 0 1", true),
        1.0
    );
    assert_eq!(
        adjudicate_truncated("8/8/4b3/4k3/8/8/8/R3K3 w - - 0 1", true),
        0.5
    );
    // Q vs R (+4) is conservatively a draw.
    assert_eq!(
        adjudicate_truncated("8/8/4r3/4k3/8/8/8/Q3K3 w - - 0 1", true),
        0.5
    );
    assert_eq!(material_balance_white("unreplayable: x"), None);
    assert_eq!(adjudicate_truncated("unreplayable: x", true), 0.5);
}

#[test]
fn h36_cycle2_replay_rescores_to_the_preregistered_value() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../docs/evidence/hp-h3/smoke-h36/rootcause/c2-parent-arena-replay/eval-arena.json"
    );
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let games = v["result"]["game_records"].as_array().unwrap();
    assert_eq!(games.len(), 32);
    let (mut w, mut d, mut l) = (0, 0, 0);
    for g in games {
        let s = match g["candidate_score"].as_f64() {
            Some(s) => s,
            None => adjudicate_truncated(
                g["final_fen"].as_str().unwrap(),
                g["candidate_white"].as_bool().unwrap(),
            ),
        };
        match s {
            1.0 => w += 1,
            0.5 => d += 1,
            _ => l += 1,
        }
    }
    assert_eq!((w, d, l), (12, 13, 7));
    let score = (w as f64 + 0.5 * d as f64) / 32.0;
    assert!((score - 0.578125).abs() < 1e-12, "{score}");
}
