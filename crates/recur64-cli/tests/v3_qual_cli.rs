//! The `v3-qual-verdict` command carries the hardened verdict in its exit status
//! and never modifies the report it reads.

use std::path::PathBuf;
use std::process::Command;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("recur64_v3_qual_cli_{name}"));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn verdict(report: &std::path::Path, out: &std::path::Path) -> (bool, String, String) {
    let o = Command::new(env!("CARGO_BIN_EXE_recur64"))
        .args([
            "v3-qual-verdict",
            "--report",
            report.to_str().unwrap(),
            "--output",
            out.to_str().unwrap(),
        ])
        .output()
        .expect("run recur64");
    (
        o.status.success(),
        String::from_utf8_lossy(&o.stdout).into_owned(),
        String::from_utf8_lossy(&o.stderr).into_owned(),
    )
}

#[test]
fn the_committed_p3_evidence_passes_the_hardened_verdict_and_is_not_modified() {
    let src = root().join("docs/evidence/v3/v3-qual-cuda-fp32.json");
    let before = std::fs::read(&src).unwrap();
    let dir = tmp("pass");
    let out = dir.join("summary.json");
    let (ok, _stdout, stderr) = verdict(&src, &out);
    assert!(ok, "hardened verdict must pass the P3 evidence: {stderr}");
    assert_eq!(
        before,
        std::fs::read(&src).unwrap(),
        "the source report must be untouched"
    );
    let s: serde_json::Value = serde_json::from_slice(&std::fs::read(&out).unwrap()).unwrap();
    assert_eq!(s["qualification_gates_ok"], true);
    assert!(
        s["provenance"]
            .as_str()
            .unwrap()
            .starts_with("DERIVED FROM EXISTING MEASURED")
    );
    assert_eq!(s["source_report"]["sha256"].as_str().unwrap().len(), 64);
    // The unrecorded sub-condition is stated, not hidden.
    assert_eq!(s["recorded_limitations"].as_array().unwrap().len(), 1);
    // The repeated-build slope is a non-gating finding.
    let findings = s["diagnostic_findings"].as_array().unwrap();
    assert!(
        findings
            .iter()
            .any(|f| f["name"] == "repeated_model_build_drop_vram_slope" && f["gating"] == false)
    );
}

#[test]
fn a_failing_report_exits_non_zero_but_still_writes_the_summary() {
    let src = root().join("docs/evidence/v3/v3-qual-cuda-fp32.json");
    let mut r: serde_json::Value = serde_json::from_slice(&std::fs::read(&src).unwrap()).unwrap();
    r["training"][1]["gradient_coverage_update_one"]["stop_head_gradient_nonzero"] = true.into();
    let dir = tmp("fail");
    let bad = dir.join("bad.json");
    std::fs::write(&bad, serde_json::to_vec(&r).unwrap()).unwrap();
    let out = dir.join("summary.json");
    let (ok, _stdout, stderr) = verdict(&bad, &out);
    assert!(!ok, "a failing qualification must exit non-zero");
    assert!(
        stderr.contains("stop_head_gradient_exactly_zero"),
        "{stderr}"
    );
    let s: serde_json::Value = serde_json::from_slice(&std::fs::read(&out).unwrap()).unwrap();
    assert_eq!(s["qualification_gates_ok"], false);
}

#[test]
fn the_original_all_sections_ok_is_reported_but_not_trusted() {
    // The original P3 field did not include every gate. The summary records its
    // value separately from the hardened verdict.
    let src = root().join("docs/evidence/v3/v3-qual-cuda-fp32.json");
    let dir = tmp("orig");
    let out = dir.join("summary.json");
    let _ = verdict(&src, &out);
    let s: serde_json::Value = serde_json::from_slice(&std::fs::read(&out).unwrap()).unwrap();
    assert_eq!(s["source_report"]["original_all_sections_ok"], true);
    assert!(s.get("all_sections_ok").is_none());
}
