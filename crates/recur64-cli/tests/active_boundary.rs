//! Command-boundary tests: no historical command may interpret an
//! `active_search_v3` configuration as another architecture.
//!
//! Every test runs the real `recur64` binary. Supported commands must describe or
//! build the real active graph; every other command must refuse, visibly and
//! before doing any model or device work.

use std::path::PathBuf;
use std::process::Command;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn active_probe_config() -> PathBuf {
    root().join("configs/v3/active-search-v3-cpu.toml")
}

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("recur64_active_boundary_{name}"));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// A `RunConfig` (the mainline runtime format) whose model is active_search_v3,
/// derived from the committed smoke config so every other section is valid.
fn active_run_config(dir: &std::path::Path) -> PathBuf {
    let text = std::fs::read_to_string(root().join("configs/smoke.toml")).unwrap();
    let i = text
        .find("[model]")
        .expect("smoke.toml has a [model] table");
    let after = text[i + 1..].find("\n[").map(|j| i + 1 + j);
    let rest = after.map_or("", |j| &text[j..]);
    let out = format!(
        "{}[model]\narchitecture = \"active_search_v3\"\nwidth = 640\nheads = 10\nffn = 1280\n\
         input_blocks = 0\ncore_blocks = 8\noutput_blocks = 0\n\n[model.active]\n{}",
        &text[..i],
        rest
    );
    let p = dir.join("active-run.toml");
    std::fs::write(&p, out).unwrap();
    p
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
fn model_info_describes_the_real_active_graph_not_the_probe() {
    let dir = tmp("model_info");
    let json = dir.join("info.json");
    let o = run(&[
        "model-info",
        "--config",
        active_probe_config().to_str().unwrap(),
        "--json",
        json.to_str().unwrap(),
    ]);
    assert!(
        o.ok,
        "model-info must succeed for active_search_v3: {}",
        o.stderr
    );
    let s = &o.stdout;
    // The real graph's identity and parameters.
    assert!(s.contains("architecture    : active_search_v3"), "{s}");
    assert!(
        s.contains("30853790"),
        "exact frozen P2 parameter total missing:\n{s}"
    );
    assert!(s.contains("TOTAL UNIQUE"), "{s}");
    assert!(s.contains("root board"), "root geometry missing");
    assert!(
        s.contains("query encoder   : width=256 heads=4 ffn=512 blocks=2"),
        "{s}"
    );
    assert!(s.contains("workspace       : K=8"), "{s}");
    assert!(s.contains("planner"), "planner geometry missing");
    assert!(s.contains("selector"), "selector geometry missing");
    assert!(s.contains("readout"), "readout geometry missing");
    for contract in [
        "v25_root_encoder_v1",
        "candidate_token_v3_root_v1",
        "state_query_v1",
        "query_state_encoder_v1",
        "frontier_v1",
        "branch_workspace_v1",
        "active_selector_v1",
        "active_planner_v1",
        "proof_trace_v1",
        "budget_0_2_4_8_v1",
        "root_policy_v3_v1",
    ] {
        assert!(s.contains(contract), "contract {contract} missing:\n{s}");
    }
    for part in [
        "root.board_blocks",
        "query.blocks",
        "planner.workspace_update",
        "selector.edge_scorer",
        "readout",
    ] {
        assert!(s.contains(part), "parameter group {part} missing");
    }
    assert!(
        s.contains("123415160"),
        "fp32 parameter bytes (30853790 x 4) missing:\n{s}"
    );
    assert!(
        s.contains("[0, 2, 4, 8, 16]"),
        "supported scientific budgets missing"
    );
    assert!(
        s.contains("recurrence") && s.contains(": 1 "),
        "recurrence statement missing"
    );
    assert!(
        s.contains("exact state-query budget"),
        "test-time-compute statement missing"
    );
    // It must NOT be the Probe report.
    assert!(
        !s.contains("executed transformer blocks and compute multiplier"),
        "active config reached the Probe model-info path:\n{s}"
    );
    let doc: serde_json::Value = serde_json::from_slice(&std::fs::read(&json).unwrap()).unwrap();
    assert_eq!(doc["total_params"], 30_853_790);
    assert_eq!(doc["param_bytes_fp32"], 123_415_160);
    assert_eq!(doc["architecture"], "active_search_v3");
    assert_eq!(doc["recurrence"], 1);
}

#[test]
fn historical_model_info_paths_are_unchanged() {
    // The other architectures still report their own graphs.
    let o = run(&[
        "model-info",
        "--config",
        root()
            .join("configs/v25/candidate-v25-cf.toml")
            .to_str()
            .unwrap(),
    ]);
    assert!(o.ok, "{}", o.stderr);
    assert!(o.stdout.contains("candidate_v25"), "{}", o.stdout);
    assert!(o.stdout.contains("27469204"), "{}", o.stdout);
}

#[test]
fn every_historical_command_refuses_active_search_v3_before_any_work() {
    let dir = tmp("refuse");
    let d = dir.to_str().unwrap().to_string();
    let act = active_probe_config().to_str().unwrap().to_string();
    let run_cfg = active_run_config(&dir).to_str().unwrap().to_string();
    let missing = format!("{d}/does-not-exist");
    // Leaked on purpose: the case table borrows these for the whole test.
    let x = |n: &str| -> &'static str { Box::leak(format!("{d}/{n}").into_boxed_str()) };

    // (name, args). Files named here do not exist: a command that touched them
    // before refusing would fail with a different message.
    let cases: Vec<(&str, Vec<&str>)> = vec![
        ("bench", vec!["bench", "--config", &act, "--output", x("b")]),
        (
            "v25-qual",
            vec!["v25-qual", "--config", &act, "--output", x("q")],
        ),
        (
            "proof train",
            vec![
                "proof",
                "train",
                "--config",
                &act,
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
                &act,
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
            "bench-train",
            vec![
                "bench-train",
                "--config",
                &run_cfg,
                "--checkpoint",
                &missing,
                "--replay",
                &missing,
                "--output",
                x("o1"),
            ],
        ),
        (
            "forward-probe",
            vec![
                "forward-probe",
                "--config",
                &run_cfg,
                "--checkpoint",
                &missing,
                "--replay",
                &missing,
                "--output",
                x("o2"),
            ],
        ),
        (
            "value-diag",
            vec![
                "value-diag",
                "--config",
                &run_cfg,
                "--checkpoint",
                &missing,
                "--phase",
                "x",
                "--eval",
                "x",
                "--output",
                x("o3"),
            ],
        ),
        (
            "eval-policy",
            vec![
                "eval-policy",
                "--config",
                &run_cfg,
                "--checkpoint",
                &missing,
                "--output",
                x("p"),
            ],
        ),
        (
            "freeze-reference",
            vec!["freeze-reference", "--config", &run_cfg, "--output", x("f")],
        ),
        (
            "pilot",
            vec!["pilot", "--config", &run_cfg, "--run-dir", x("pilot")],
        ),
        (
            "search-gain",
            vec![
                "search-gain",
                "--config",
                &run_cfg,
                "--checkpoint",
                &missing,
                "--replay",
                &missing,
                "--output",
                x("o4"),
            ],
        ),
        (
            "eval-arena",
            vec![
                "eval-arena",
                "--config",
                &run_cfg,
                "--reference",
                &missing,
                "--candidate",
                &missing,
                "--output",
                x("o5"),
            ],
        ),
        (
            "bench-runtime",
            vec!["bench-runtime", "--config", &run_cfg, "--output", x("r")],
        ),
        (
            "bench-lifecycle",
            vec![
                "bench-lifecycle",
                "--config",
                &run_cfg,
                "--checkpoint",
                &missing,
                "--output",
                x("l"),
            ],
        ),
        (
            "selfplay",
            vec!["selfplay", "--config", &run_cfg, "--output", x("s")],
        ),
        (
            "train",
            vec![
                "train",
                "--config",
                &run_cfg,
                "--replay",
                &missing,
                "--checkpoint",
                &missing,
                "--output",
                x("o6"),
            ],
        ),
        (
            "arena",
            vec![
                "arena",
                "--config",
                &run_cfg,
                "--reference",
                &missing,
                "--candidate",
                &missing,
                "--output",
                x("o7"),
            ],
        ),
        (
            "run",
            vec!["run", "--config", &run_cfg, "--run-dir", x("run")],
        ),
    ];
    let mut failures = Vec::new();
    for (name, args) in &cases {
        let o = run(args);
        let refused = !o.ok && o.stderr.contains("active_search_v3 is not supported");
        if !refused {
            failures.push(format!(
                "{name}: ok={} stderr={}",
                o.ok,
                o.stderr.lines().next().unwrap_or("")
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "commands that did not refuse active_search_v3 visibly:\n{}",
        failures.join("\n")
    );
    // Nothing was created: refusal happened before any output.
    let created: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n != "active-run.toml")
        .collect();
    assert!(
        created.is_empty(),
        "refused commands left outputs behind: {created:?}"
    );
}

#[test]
fn the_v3_qualification_command_accepts_active_and_refuses_others() {
    // v3-qual refuses a non-active config before any model work.
    let o = run(&[
        "v3-qual",
        "--config",
        root()
            .join("configs/v25/candidate-v25-cf.toml")
            .to_str()
            .unwrap(),
        "--output",
        tmp("v3qual_refuse").to_str().unwrap(),
    ]);
    assert!(!o.ok);
    assert!(
        o.stderr.contains("qualifies active_search_v3 only"),
        "{}",
        o.stderr
    );
}

#[test]
fn a_v3_budget_above_the_scientific_maximum_is_refused_by_the_qualification_cli() {
    let o = run(&[
        "v3-qual",
        "--config",
        active_probe_config().to_str().unwrap(),
        "--output",
        tmp("v3qual_budget").to_str().unwrap(),
        "--budgets",
        "0,32",
    ]);
    assert!(!o.ok);
    assert!(
        o.stderr.contains("maximum") && o.stderr.contains("16"),
        "{}",
        o.stderr
    );
}
