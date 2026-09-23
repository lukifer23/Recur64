//! T2.13/T2.14: bounded end-to-end cycle and interruption recovery.

use std::time::Duration;

use burn::backend::Autodiff;
use burn::backend::Flex;

use recur64_runtime::replay::audit_dir;
use recur64_runtime::{
    CancelToken, RunConfig, RunDir, RunStatus, read_metadata, run as run_coordinator,
};

type B = Autodiff<Flex>;

fn smoke_cfg() -> RunConfig {
    RunConfig::from_toml_str(
        r#"
run_id = "test"
device = "cpu"
precision = "fp32"
recurrence = 1
start_fen = "8/8/8/4k3/8/8/3Q4/4K3 w - - 0 1"
simulations_per_move = 2
c_puct = 1.0
temperature = 1.0
active_games = 2
cpu_workers = 1
ply_cap = 120
seed = 1
max_inference_batch = 4
batch_timeout_us = 200
shard_max_games = 16
train_batch = 8
max_updates = 2
lr = 3.0e-4
arena_games = 2
run_budget_minutes = 5

[model]
width = 192
heads = 6
ffn = 384
input_blocks = 0
core_blocks = 4
output_blocks = 0
squares = 64
in_features = 119
policy_dim = 128
wdl_classes = 3
promo_codes = 5
rms_eps = 1e-5
"#,
    )
    .unwrap()
}

fn tmp(name: &str) -> std::path::PathBuf {
    let root = std::env::temp_dir().join(format!("recur64_phase2_{name}"));
    let _ = std::fs::remove_dir_all(&root);
    root
}

#[test]
fn full_cycle_completes() {
    let cfg = smoke_cfg();
    let root = tmp("e2e");
    let dir = RunDir::create(&root, true).unwrap();
    let cancel = CancelToken::new();
    let report = run_coordinator::<B>(&cfg, &dir, &cancel).unwrap();

    assert_eq!(report.status, "completed");
    assert!(report.games_collected >= 1);
    assert!(report.audit_ok, "{:?}", report.audit_errors);
    let train = report.train.as_ref().expect("training should have run");
    assert_eq!(train.updates, 2);
    assert!(train.first_loss.is_finite() && train.last_loss.is_finite());
    assert!(report.arena.is_some());

    assert!(root.join("report/report.json").exists());
    assert!(root.join("report/report.md").exists());
    assert!(root.join("checkpoints/reference/meta.json").exists());
    assert!(root.join("checkpoints/candidate/meta.json").exists());
    assert!(root.join("replay/manifest.json").exists());

    let meta = read_metadata(&root).unwrap();
    assert_eq!(meta.status, RunStatus::Completed);
    assert_eq!(meta.observation_version, 1);
    assert_eq!(meta.replay_schema_version, 1);

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn pre_cancelled_run_is_interrupted_and_recoverable() {
    let cfg = smoke_cfg();
    let root = tmp("cancel_pre");
    let dir = RunDir::create(&root, true).unwrap();
    let cancel = CancelToken::new();
    cancel.cancel();
    let report = run_coordinator::<B>(&cfg, &dir, &cancel).unwrap();

    assert_eq!(report.status, "interrupted");
    assert!(root.join("checkpoints/reference/meta.json").exists());
    assert!(!root.join("checkpoints/candidate").exists());
    let meta = read_metadata(&root).unwrap();
    assert_eq!(meta.status, RunStatus::Interrupted);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn cancel_during_collect_never_corrupts_replay() {
    let mut cfg = smoke_cfg();
    cfg.active_games = 64;
    cfg.ply_cap = 150;
    let root = tmp("cancel_mid");
    let dir = RunDir::create(&root, true).unwrap();
    let cancel = CancelToken::new();
    let c2 = cancel.clone();
    let handle = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(200));
        c2.cancel();
    });
    let report = run_coordinator::<B>(&cfg, &dir, &cancel).unwrap();
    handle.join().unwrap();

    assert!(report.status == "interrupted" || report.status == "completed");
    // Any published replay must be valid; no partial shard may be treated as real.
    if root.join("replay/manifest.json").exists() {
        let a = audit_dir(&root.join("replay")).unwrap();
        assert!(a.ok(), "corrupt replay after cancel: {:?}", a.errors);
    }
    let meta = read_metadata(&root).unwrap();
    assert!(matches!(
        meta.status,
        RunStatus::Interrupted | RunStatus::Completed
    ));
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn run_dir_refuses_overwrite_without_force() {
    let root = tmp("nooverwrite");
    let _ = RunDir::create(&root, true).unwrap();
    let res = RunDir::create(&root, false);
    assert!(res.is_err());
    let err = res.err().unwrap();
    assert!(err.to_string().contains("already exists"));
    let _ = std::fs::remove_dir_all(&root);
}
