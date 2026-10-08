//! Source-level boundary scan: model-side code must not name, and must not hold the
//! roles needed to reach, withheld or root-bearing dataset parts. (The runtime
//! enforcement is tested in recur64-v69/tests/audit_corruption.rs.)

use std::path::Path;

fn sources(dir: &Path, out: &mut Vec<(String, String)>) {
    for e in std::fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        if p.is_dir() {
            sources(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push((p.display().to_string(), std::fs::read_to_string(&p).unwrap()));
        }
    }
}

#[test]
fn model_crate_never_names_withheld_dataset_parts_or_audit_roles() {
    let mut files = Vec::new();
    sources(&Path::new(env!("CARGO_MANIFEST_DIR")).join("src"), &mut files);
    assert!(files.len() >= 6);
    let forbidden = [
        "sealed",
        "pool/",
        "meta/",
        ".meta.jsonl",
        "roots.jsonl",
        "test.jsonl",
        "root_fen",
        "root_depth",
        "Role::DataAudit",
        "Role::MetricAggregator",
        "generation_report",
    ];
    for (name, text) in &files {
        for f in forbidden {
            assert!(!text.contains(f), "{name} mentions forbidden token {f:?}");
        }
    }
}

#[test]
fn model_crate_has_no_cpu_backend_or_unsafe_fallback() {
    let mut files = Vec::new();
    sources(&Path::new(env!("CARGO_MANIFEST_DIR")).join("src"), &mut files);
    for (name, text) in &files {
        for f in ["Flex", "NdArray", "burn::backend::Wgpu", "unsafe "] {
            assert!(!text.contains(f), "{name} mentions {f:?}");
        }
    }
    let manifest = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml")).unwrap();
    assert!(manifest.contains("\"cuda\"") && !manifest.contains("flex") && !manifest.contains("ndarray"));
}
