//! Real command-boundary checks for the distinct V5 namespace.

use std::path::PathBuf;
use std::process::Command;

struct Out {
    ok: bool,
    stdout: String,
    stderr: String,
}

fn run(args: &[&str]) -> Out {
    let out = Command::new(env!("CARGO_BIN_EXE_recur64"))
        .args(args)
        .output()
        .expect("run recur64");
    Out {
        ok: out.status.success(),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("recur64_v5_boundary_{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn model_info_reports_the_frozen_identity_and_measured_parameter_total() {
    let dir = tmp("model");
    let json = dir.join("model.json");
    let out = run(&["v5", "model-info", "--json", json.to_str().unwrap()]);
    assert!(out.ok, "{}", out.stderr);
    assert!(out.stdout.contains("counterfactual_relational_loop_v1"));
    assert!(out.stdout.contains("7160080"));
    assert!(out.stdout.contains("4R"));
    let doc: serde_json::Value = serde_json::from_slice(&std::fs::read(json).unwrap()).unwrap();
    assert_eq!(doc["parameters"], 7_160_080);
    assert_eq!(doc["architecture"], "counterfactual_relational_loop_v1");
    assert_eq!(
        doc["contracts"]["paired_null_readout"],
        "v5_paired_null_readout_v1"
    );
}

#[test]
fn graph_generate_and_audit_execute_the_real_statequery_path() {
    let dir = tmp("graph");
    let graph = dir.join("q4.json");
    let generated = run(&[
        "v5",
        "graph",
        "generate",
        "--fen",
        "6k1/8/8/8/8/8/4Q3/3RK3 w - - 0 1",
        "--position-id",
        "boundary-fixture",
        "--schedule",
        "uniform-frontier",
        "--q",
        "4",
        "--output",
        graph.to_str().unwrap(),
    ]);
    assert!(generated.ok, "{}", generated.stderr);
    let audited = run(&["v5", "graph", "audit", "--graph", graph.to_str().unwrap()]);
    assert!(audited.ok, "{}", audited.stderr);
    assert!(audited.stdout.contains("exact transitions 4"));
}

#[test]
fn custody_refuses_noncanonical_or_missing_data() {
    let dir = tmp("custody");
    let missing = dir.join("not-p25.json");
    std::fs::write(&missing, b"{}").unwrap();
    let out = run(&["v5", "custody", "--data", missing.to_str().unwrap()]);
    assert!(!out.ok);
    assert!(!out.stderr.is_empty());
}

#[test]
fn v5_help_exposes_the_real_drill_evaluation_and_reporting_surface() {
    let out = run(&["v5", "--help"]);
    assert!(out.ok, "{}", out.stderr);
    for command in [
        "drill",
        "evaluate",
        "evaluate-baseline",
        "eval-merge",
        "ablation",
        "pilot-report",
        "extra-loops",
    ] {
        assert!(
            out.stdout.contains(command),
            "missing {command} in {}",
            out.stdout
        );
    }
}

#[test]
fn drill_refuses_before_custody_or_qualification_can_be_bypassed() {
    let dir = tmp("drill-refusal");
    let output = dir.join("drill.json");
    let out = run(&[
        "v5",
        "drill",
        "--device",
        "cpu",
        "--data",
        dir.join("missing-p25.json").to_str().unwrap(),
        "--qualification",
        dir.join("missing-qualification.json").to_str().unwrap(),
        "--output",
        output.to_str().unwrap(),
    ]);
    assert!(!out.ok);
    assert!(!output.exists());
}

#[test]
fn extra_loops_refuses_without_a_pilot_candidate_report() {
    let dir = tmp("r8-refusal");
    let gate = dir.join("gate.json");
    std::fs::write(
        &gate,
        br#"{"schema":"v5_reader_pilot_report_v1","classification":"NO_SIGNAL"}"#,
    )
    .unwrap();
    let out = run(&[
        "v5",
        "extra-loops",
        "--device",
        "cpu",
        "--data",
        dir.join("missing.json").to_str().unwrap(),
        "--stage-b",
        dir.join("stage-b").to_str().unwrap(),
        "--evaluation",
        dir.join("eval.json").to_str().unwrap(),
        "--pilot-report",
        gate.to_str().unwrap(),
        "--qualification",
        dir.join("qual.json").to_str().unwrap(),
        "--output",
        dir.join("r8.json").to_str().unwrap(),
    ]);
    assert!(!out.ok);
    assert!(out.stderr.contains("missing field") || out.stderr.contains("R8 is authorized"));
}

#[test]
fn baseline_evaluation_and_training_require_their_real_prerequisites() {
    let dir = tmp("baseline-prerequisites");
    let output = dir.join("baseline.json");
    let out = run(&[
        "v5",
        "evaluate-baseline",
        "--device",
        "cpu",
        "--data",
        dir.join("missing-p25.json").to_str().unwrap(),
        "--stage-a",
        dir.join("stage-a").to_str().unwrap(),
        "--qualification",
        dir.join("qualification.json").to_str().unwrap(),
        "--output",
        output.to_str().unwrap(),
    ]);
    assert!(!out.ok);
    assert!(!output.exists());
    let training = run(&[
        "v5",
        "train",
        "--stage",
        "a",
        "--device",
        "cpu",
        "--data",
        dir.join("missing-p25.json").to_str().unwrap(),
        "--run-dir",
        dir.join("stage-a").to_str().unwrap(),
        "--qualification",
        dir.join("qualification.json").to_str().unwrap(),
    ]);
    assert!(!training.ok);
    assert!(training.stderr.contains("--drill"));
    assert!(!dir.join("stage-a").exists());
}
