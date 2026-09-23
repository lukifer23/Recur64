//! T2.7/T2.8: replay writer/reader integrity and audit rejection.

use std::path::PathBuf;

use burn::backend::Flex;

use recur64_core::Termination;
use recur64_model::config::ModelConfig;
use recur64_model::model::ProbeModel;
use recur64_runtime::SyncEvaluator;
use recur64_runtime::replay::{
    GameRecord, ReplayHeader, ReplayReader, ReplayWriter, SearchRecord, audit_dir, audit_games,
    parse_shard_bytes,
};
use recur64_runtime::{SelfPlayConfig, play_game_seeded};

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

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("recur64_replay_test_{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn real_game(id: u64) -> GameRecord {
    let device = Default::default();
    let model = ProbeModel::<Flex>::new(micro(), &device);
    let ev = SyncEvaluator::new(model, 1, device);
    let cfg = SelfPlayConfig {
        simulations_per_move: 2,
        c_puct: 1.0,
        temperature: 1.0,
        ply_cap: 4,
        recurrence: 1,
    };
    let game = play_game_seeded(&ev, &cfg, id).unwrap();
    GameRecord::from_selfplay(
        id,
        &game,
        SearchRecord {
            simulations: cfg.simulations_per_move,
            c_puct: cfg.c_puct,
            temperature: cfg.temperature,
            recurrence: cfg.recurrence,
        },
    )
}

fn header() -> ReplayHeader {
    ReplayHeader::new("test-run", "model-abc", "cpu (Flex)", "fp32")
}

#[test]
fn write_read_roundtrip_and_audit() {
    let dir = tmp("roundtrip");
    let mut w = ReplayWriter::new(&dir, header(), 2).unwrap();
    for id in 0..5 {
        w.push(real_game(id)).unwrap();
    }
    let manifest = w.finish().unwrap();
    assert_eq!(manifest.games, 5);
    assert_eq!(manifest.shards.len(), 3); // 2 + 2 + 1

    let reader = ReplayReader::open(&dir).unwrap();
    let games = reader.read_all_games().unwrap();
    assert_eq!(games.len(), 5);

    let report = audit_dir(&dir).unwrap();
    assert!(report.ok(), "audit errors: {:?}", report.errors);
    assert_eq!(report.games, 5);
    assert!(report.plies > 0);
}

#[test]
fn checksum_detects_corruption() {
    let dir = tmp("corrupt");
    let mut w = ReplayWriter::new(&dir, header(), 8).unwrap();
    w.push(real_game(0)).unwrap();
    w.finish().unwrap();

    // Flip a byte inside the shard payload.
    let shard_path = dir.join("shard-000000.r64shard");
    let mut bytes = std::fs::read(&shard_path).unwrap();
    let n = bytes.len();
    bytes[n - 1] ^= 0xFF;
    std::fs::write(&shard_path, &bytes).unwrap();

    let err = parse_shard_bytes(&bytes).unwrap_err();
    assert!(err.contains("checksum") || err.contains("decode"), "{err}");

    let report = audit_dir(&dir).unwrap();
    assert!(!report.ok(), "audit must reject a corrupted shard");
}

#[test]
fn partial_tmp_is_ignored() {
    let dir = tmp("partial");
    let mut w = ReplayWriter::new(&dir, header(), 8).unwrap();
    w.push(real_game(0)).unwrap();
    w.finish().unwrap();

    // A leftover .tmp from an interrupted write must not affect the replay.
    std::fs::write(dir.join("shard-000001.r64shard.tmp"), b"garbage").unwrap();
    let report = audit_dir(&dir).unwrap();
    assert!(report.ok(), "{:?}", report.errors);
    assert_eq!(report.games, 1);
}

#[test]
fn audit_rejects_truncated_with_outcome() {
    let mut g = real_game(0);
    // Force a truncated game with a (wrong) outcome.
    g.termination = Termination::Truncated.label().to_string();
    g.outcome = Some(1);
    let report = audit_games(&[g]);
    assert!(!report.ok());
    assert!(
        report
            .errors
            .iter()
            .any(|e| e.contains("must not have an outcome"))
    );
}

#[test]
fn audit_rejects_illegal_selected_action() {
    let mut g = real_game(0);
    // 20479 is a valid index but almost never legal here.
    g.plies[0].selected = 20479;
    let report = audit_games(&[g]);
    assert!(!report.ok());
    assert!(report.errors.iter().any(|e| e.contains("not legal")));
}

#[test]
fn audit_rejects_bad_target_sum() {
    let mut g = real_game(0);
    for t in g.plies[0].target.iter_mut() {
        t.1 = 0.0;
    }
    let report = audit_games(&[g]);
    assert!(!report.ok());
    assert!(report.errors.iter().any(|e| e.contains("target sum")));
}
