//! H1: the pilot runs end to end from a frozen reference on CPU, writes its
//! identity and T0 baseline, and keeps lineage and the optimizer trajectory
//! consistent with the promotion decision.

use recur64_model::checkpoint::{CheckpointMeta, save_training};
use recur64_model::train::{CpuTrainBackend, adamw};
use recur64_runtime::replay::ReplayReader;
use recur64_runtime::{CancelToken, RunConfig, RunDir, model_io, run_pilot};

fn cfg(root: &std::path::Path) -> RunConfig {
    let mut c = RunConfig::from_toml_str(
        r#"
run_id = "pilot-e2e"
device = "cpu"
precision = "fp32"
simulations_per_move = 2
games_per_cycle = 4
concurrent_games = 2
cpu_workers = 2
ply_cap = 40
start_fen = "8/8/8/3k4/3r4/3R4/8/3K4 w - - 0 1"
train_batch = 4
accumulation_steps = 2
max_updates = 4
planned_updates = 8
warmup_updates = 1
arena_games = 2
cycles = 2
run_budget_minutes = 10
[model]
width = 32
heads = 4
ffn = 64
input_blocks = 0
core_blocks = 1
output_blocks = 0
"#,
    )
    .unwrap();
    c.reference_checkpoint = Some(root.join("frozen").to_string_lossy().into_owned());
    c
}

fn freeze(c: &RunConfig, dir: &std::path::Path) -> String {
    let device = Default::default();
    let model = model_io::build::<CpuTrainBackend>(&c.model, &device);
    let meta = CheckpointMeta::new(c.model.clone(), 1, false, 0, c.lr, c.seed, 0, "cpu", "fp32");
    save_training(dir, &model, &adamw::<CpuTrainBackend, _>(), &meta).unwrap();
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("meta.json")).unwrap()).unwrap();
    meta["model_id"].as_str().unwrap().to_string()
}

#[test]
fn pilot_from_frozen_reference_keeps_identity_and_lineage() {
    let root = std::env::temp_dir().join(format!("recur64-pilot-e2e-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let mut c = cfg(&root);
    let reference_id = freeze(&c, &root.join("frozen"));

    // A wrong expected id must refuse to start.
    c.reference_model_id = Some("not-the-reference".into());
    let dir = RunDir::create(&root.join("bad"), false).unwrap();
    let err = run_pilot::<CpuTrainBackend>(&c, &dir, &CancelToken::new()).unwrap_err();
    assert!(err.to_string().contains("does not match"), "{err}");

    c.reference_model_id = Some(reference_id.clone());
    let dir = RunDir::create(&root.join("run"), false).unwrap();
    let report = run_pilot::<CpuTrainBackend>(&c, &dir, &CancelToken::new()).unwrap();
    assert_eq!(report.status, "completed");
    assert_eq!(report.identity.reference_model_id, reference_id);
    assert_eq!(
        report.identity.scientific_config_hash,
        c.scientific_config_hash().unwrap()
    );
    assert!(dir.root.join("identity.json").exists());
    assert!(dir.eval().join("baseline-t0.json").exists());
    let baseline = report.baseline.as_ref().unwrap();
    assert_eq!(baseline.raw_vs_random.games, 4);
    assert!(baseline.policy.mean_entropy.is_finite());

    let lineage: Vec<serde_json::Value> = std::fs::read_to_string(dir.root.join("lineage.jsonl"))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(lineage.len(), 2);
    assert_eq!(report.cycles.len(), 2);
    let games = ReplayReader::open(&dir.replay())
        .unwrap()
        .read_all_games()
        .unwrap();
    assert_eq!(games.len(), 8);
    for game in &games {
        assert_eq!(game.seed, c.seed.wrapping_add(game.game_id));
    }
    assert_ne!(games[0].seed, games[4].seed, "cycle seeds must advance");
    assert_eq!(lineage[0]["parent_model_id"], reference_id.as_str());
    let mut parent = reference_id.clone();
    let mut accepted = 0u64;
    for (row, cycle) in lineage.iter().zip(&report.cycles) {
        assert_eq!(row["parent_model_id"], parent.as_str(), "parent chain");
        assert_eq!(
            row["replay_model_ids"][0],
            parent.as_str(),
            "replay generator"
        );
        assert_eq!(row["optimizer_step_start"], accepted, "accepted trajectory");
        assert_eq!(cycle.selfplay.games_completed, 4);
        assert_eq!(cycle.first_game_id, 4 * cycle.cycle as u64);
        let arena = cycle.arena.as_ref().unwrap();
        assert_eq!(
            cycle.reference_arena_is_parent_arena,
            parent == reference_id,
            "reference arena is reused only while the parent is the reference"
        );
        assert_eq!(arena.informative, arena.decisive_games > 0);
        match cycle.decision.as_str() {
            "promote" => {
                assert!(cycle.hold_reasons.is_empty());
                assert!(arena.decisive_games >= c.promotion_min_decisive_games);
                assert!(arena.candidate_score > 0.5);
                assert_ne!(cycle.candidate_model_id, parent);
                parent = cycle.candidate_model_id.clone();
                accepted = row["optimizer_step_end"].as_u64().unwrap();
            }
            "hold" => assert!(!cycle.hold_reasons.is_empty()),
            other => panic!("unexpected decision {other}"),
        }
        assert_eq!(cycle.accepted_optimizer_step_after, accepted);
        assert_eq!(row["promoted_model_id"], parent.as_str());
        if let Some(t) = &cycle.train {
            assert_eq!(t.replay_total_games as u64, cycle.replay_total_games);
            assert!(t.current_cycle_sample_fraction.is_some());
        }
    }
    let _ = std::fs::remove_dir_all(&root);
}
