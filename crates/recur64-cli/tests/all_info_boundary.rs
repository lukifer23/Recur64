//! Command-boundary tests for `all_info_v1`: no historical command may interpret it as
//! another architecture, and `model-info` / `v3-p6` must describe the real graph.
//!
//! Every test runs the real `recur64` binary.

use std::path::PathBuf;
use std::process::Command;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn all_info_config() -> String {
    root()
        .join("configs/v3/all-info-v1-cpu.toml")
        .to_str()
        .unwrap()
        .to_string()
}

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("recur64_all_info_boundary_{name}"));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

struct Out {
    ok: bool,
    stdout: String,
    stderr: String,
}

fn run(args: &[&str]) -> Out {
    let o = Command::new(env!("CARGO_BIN_EXE_recur64"))
        .args(args)
        .output()
        .expect("run recur64");
    Out {
        ok: o.status.success(),
        stdout: String::from_utf8_lossy(&o.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&o.stderr).into_owned(),
    }
}

#[test]
fn model_info_describes_the_real_all_info_graph_and_its_parameter_match() {
    let o = run(&["model-info", "--config", &all_info_config()]);
    assert!(o.ok, "{}", o.stderr);
    assert!(o.stdout.contains("all_info_v1"), "{}", o.stdout);
    assert!(o.stdout.contains("30842524"), "{}", o.stdout);
    assert!(o.stdout.contains("within 0.5%: true"), "{}", o.stdout);
    assert!(o.stdout.contains("all_info_depth2_v1"), "{}", o.stdout);
    // It is not the probe, candidate or active report.
    assert!(!o.stdout.contains("STOP head"), "{}", o.stdout);
}

#[test]
fn every_historical_command_refuses_all_info_v1_before_any_work() {
    let dir = tmp("refuse");
    let d = dir.to_str().unwrap().to_string();
    let cfg = all_info_config();
    let missing = format!("{d}/does-not-exist");
    let x = |n: &str| -> &'static str { Box::leak(format!("{d}/{n}").into_boxed_str()) };
    let cases: Vec<(&str, Vec<&str>)> = vec![
        ("bench", vec!["bench", "--config", &cfg, "--output", x("b")]),
        (
            "v25-qual",
            vec!["v25-qual", "--config", &cfg, "--output", x("q")],
        ),
        (
            "proof train",
            vec![
                "proof",
                "train",
                "--config",
                &cfg,
                "--data",
                &missing,
                "--output",
                x("t"),
                "--seed",
                "1",
                "--lr",
                "1e-4",
                "--updates",
                "1",
            ],
        ),
        (
            "proof eval",
            vec![
                "proof",
                "eval",
                "--config",
                &cfg,
                "--checkpoint",
                &missing,
                "--data",
                &missing,
                "--split",
                "holdout_a",
                "--output",
                x("e.json"),
            ],
        ),
        (
            "v3-qual",
            vec!["v3-qual", "--config", &cfg, "--output", x("v")],
        ),
    ];
    for (name, args) in &cases {
        let o = run(args);
        assert!(!o.ok, "{name} accepted an all_info_v1 config: {}", o.stdout);
        let refused = o.stderr.contains("all_info_v1")
            || o.stderr.contains("qualifies active_search_v3 only");
        assert!(refused, "{name} did not refuse explicitly: {}", o.stderr);
    }
    let created: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    assert!(
        created.is_empty(),
        "refused commands left outputs behind: {created:?}"
    );
}

#[test]
fn v3_p6_refuses_a_layout_off_the_frozen_ladder_and_a_missing_selected_seed_set() {
    let dir = tmp("p6");
    let out = dir.join("r.json");
    let o = run(&[
        "v3-p6",
        "recipe",
        "--layout",
        "32x4",
        "--output",
        out.to_str().unwrap(),
    ]);
    assert!(!o.ok && o.stderr.contains("ladder"), "{}", o.stderr);
    assert!(!out.exists());
    let o = run(&[
        "v3-p6",
        "recipe",
        "--layout",
        "16x8",
        "--output",
        out.to_str().unwrap(),
    ]);
    assert!(o.ok, "{}", o.stderr);
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&out).unwrap()).unwrap();
    assert_eq!(v["selected_lr"], 3.0e-4);
    assert_eq!(v["seeds"], serde_json::json!([5101, 5102, 5103]));
    // Training refuses a seed outside the paired set, before reading any data.
    let o = run(&[
        "v3-p6",
        "train",
        "--train",
        "x",
        "--tune",
        "y",
        "--recipe",
        out.to_str().unwrap(),
        "--seed",
        "9999",
        "--device",
        "cpu",
        "--run-dir",
        dir.join("run").to_str().unwrap(),
        "--summary",
        dir.join("s.json").to_str().unwrap(),
    ]);
    assert!(!o.ok && o.stderr.contains("paired seeds"), "{}", o.stderr);
}
