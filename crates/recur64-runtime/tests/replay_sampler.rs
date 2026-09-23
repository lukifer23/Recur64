//! T3.5: on-demand sampling, truncation exclusion, and capacity archiving.

use std::path::PathBuf;

use recur64_core::{ActionId, Color, GameState, PromotionCode, StandardMove};
use recur64_runtime::replay::{
    GameRecord, PlyRecord, ReplayHeader, ReplayStore, ReplayWriter, SearchRecord, enforce_capacity,
};
use recur64_search::Rng;

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("recur64_sampler_{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn record_from_uci(game_id: u64, moves: &[&str], outcome: Option<u8>, term: &str) -> GameRecord {
    let mut state = GameState::startpos();
    let start_fen = state.to_fen();
    let mut plies = Vec::new();
    for m in moves {
        let perspective = state.perspective();
        let mv = StandardMove::from_uci(state.board(), m).unwrap();
        let id = ActionId::from_physical(
            mv.from,
            mv.to,
            mv.promotion.unwrap_or(PromotionCode::NONE),
            perspective,
        );
        plies.push(PlyRecord {
            selected: id.index() as u16,
            target: vec![(id.index() as u16, 1.0)],
            visits_total: 1,
            side_to_move: if state.side_to_move() == Color::White {
                0
            } else {
                1
            },
        });
        state.apply(mv).unwrap();
    }
    GameRecord {
        game_id,
        start_fen,
        seed: 0,
        search: SearchRecord {
            simulations: 1,
            c_puct: 1.0,
            temperature: 0.0,
            recurrence: 1,
        },
        plies,
        termination: term.to_string(),
        outcome,
    }
}

fn header() -> ReplayHeader {
    ReplayHeader::new("t", "m", "cpu", "fp32")
}

#[test]
fn store_excludes_truncated_and_samples_valid() {
    let dir = tmp("store");
    let mut w = ReplayWriter::new(&dir, header(), 8).unwrap();
    w.push(record_from_uci(
        0,
        &["f2f3", "e7e5", "g2g4", "d8h4"],
        Some(2),
        "checkmate",
    ))
    .unwrap();
    // Truncated game: excluded from sampling.
    w.push(record_from_uci(1, &["e2e4", "e7e5"], None, "truncated"))
        .unwrap();
    w.finish().unwrap();

    let store = ReplayStore::open(&dir).unwrap();
    assert_eq!(store.total_games(), 2);
    assert_eq!(
        store.sampleable(),
        4,
        "only the 4 result plies are sampleable"
    );

    let mut rng = Rng::new(1);
    let batch = store.sample_batch(8, &mut rng).unwrap();
    assert_eq!(batch.len(), 8);
    for ex in &batch {
        assert!(!ex.legal.is_empty());
        assert_eq!(ex.policy.len(), ex.legal.len());
        let sum: f32 = ex.policy.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!((0..3).contains(&ex.wdl));
    }
}

#[test]
fn capacity_archives_oldest_shards() {
    let dir = tmp("capacity");
    // shard_max_games = 1 -> one game per shard, 4 positions each.
    let mut w = ReplayWriter::new(&dir, header(), 1).unwrap();
    for id in 0..3 {
        w.push(record_from_uci(
            id,
            &["f2f3", "e7e5", "g2g4", "d8h4"],
            Some(2),
            "checkmate",
        ))
        .unwrap();
    }
    w.finish().unwrap();

    let report = enforce_capacity(&dir, 4).unwrap();
    assert_eq!(report.kept_shards, 1);
    assert_eq!(report.archived_shards, 2);
    assert!(dir.join("archive").exists(), "archived shards preserved");

    let store = ReplayStore::open(&dir).unwrap();
    assert_eq!(store.sampleable(), 4);
    assert_eq!(store.shard_count(), 1);
}
