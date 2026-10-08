//! Corruption tests for the extended data audit and the role access policy.
//! A small dataset is generated with a TEST-ONLY seed into a temp V69-style
//! namespace; each test corrupts a private copy (re-pinning the manifest so only
//! the targeted check can fire) and requires the audit to detect it.

use recur64_v69::access::{Access, Role};
use recur64_v69::audit2::run_audit_v2;
use recur64_v69::custody::Custody;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

const TEST_SEED_HEX: &str = "5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed";

fn base() -> &'static PathBuf {
    static BASE: OnceLock<PathBuf> = OnceLock::new();
    BASE.get_or_init(|| {
        let dir = std::env::temp_dir().join(format!("v69-audit-test-{}", std::process::id())).join("artifacts").join("v69");
        let _ = std::fs::remove_dir_all(dir.parent().unwrap());
        std::fs::create_dir_all(dir.join("seed")).unwrap();
        std::fs::write(dir.join("seed/test_seed.hex"), TEST_SEED_HEX).unwrap();
        let st = std::process::Command::new(env!("CARGO_BIN_EXE_v69-data"))
            .args(["generate", "--artifacts"])
            .arg(&dir)
            .args(["--seed-file", "seed/test_seed.hex", "--run", "base", "--round-size", "1000", "--max-rounds", "30", "--threads", "4"])
            .status()
            .unwrap();
        assert!(st.success(), "test dataset generation failed");
        dir
    })
}

fn copy_dir(src: &Path, dst: &Path) {
    std::fs::create_dir_all(dst).unwrap();
    for e in std::fs::read_dir(src).unwrap() {
        let e = e.unwrap();
        let (s, d) = (e.path(), dst.join(e.file_name()));
        if s.is_dir() {
            copy_dir(&s, &d)
        } else {
            std::fs::copy(&s, &d).unwrap();
            let mut p = std::fs::metadata(&d).unwrap().permissions();
            p.set_readonly(false);
            std::fs::set_permissions(&d, p).unwrap();
        }
    }
}

struct Fixture {
    root: PathBuf,
    name: String,
}

impl Fixture {
    fn new(name: &str) -> Self {
        copy_dir(&base().join("base"), &base().join(name));
        Self { root: base().clone(), name: name.to_string() }
    }
    fn path(&self, rel: &str) -> PathBuf {
        self.root.join(&self.name).join(rel)
    }
    /// Rewrite a jsonl file line-wise and re-pin its manifest hash.
    fn mutate(&self, rel: &str, f: impl FnOnce(&mut Vec<Value>)) {
        let p = self.path(rel);
        let mut rows: Vec<Value> = std::fs::read_to_string(&p).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
        f(&mut rows);
        let text: String = rows.iter().map(|r| serde_json::to_string(r).unwrap() + "\n").collect();
        std::fs::write(&p, &text).unwrap();
        self.repin(rel);
    }
    fn repin(&self, rel: &str) {
        use sha2::{Digest, Sha256};
        let mp = self.path("MANIFEST.sha256.json");
        let mut m: serde_json::Map<String, Value> = serde_json::from_str(&std::fs::read_to_string(&mp).unwrap()).unwrap();
        let h: String = Sha256::digest(std::fs::read(self.path(rel)).unwrap()).iter().map(|b| format!("{b:02x}")).collect();
        m.insert(rel.to_string(), Value::String(h));
        std::fs::write(&mp, serde_json::to_string(&m).unwrap()).unwrap();
    }
    fn audit(&self) -> anyhow::Result<recur64_v69::audit2::AuditV2> {
        let c = Custody::new(&self.root).unwrap();
        let acc = Access::new(&c, Role::DataAudit, Path::new(&self.name), Path::new("seed/test_seed.hex")).unwrap();
        run_audit_v2(&acc, Path::new(&self.name), Path::new("seed/test_seed.hex"), 5_000_000, 4)
    }
    fn expect_fail(&self, tag: &str) {
        match self.audit() {
            Ok(r) => {
                assert!(!r.audit_pass, "corruption not detected");
                assert!(r.failures.iter().any(|f| f.contains(tag)), "expected [{tag}] failure, got {:?}", &r.failures[..r.failures.len().min(5)]);
            }
            Err(e) => assert!(format!("{e:#}").contains(tag), "audit errored without [{tag}]: {e:#}"),
        }
    }
}

#[test]
fn baseline_passes() {
    let f = Fixture::new("c_baseline");
    let r = f.audit().unwrap();
    assert!(r.audit_pass, "{:?}", r.failures);
    assert_eq!(r.examples, 1536);
}

#[test]
fn detects_label_mismatch_between_row_and_metadata() {
    let f = Fixture::new("c_label");
    f.mutate("data/fit.jsonl", |r| {
        let b = r[3]["label"].as_bool().unwrap();
        r[3]["label"] = Value::Bool(!b);
    });
    f.expect_fail("rows-vs-meta");
}

#[test]
fn detects_fen_mismatch_between_row_and_metadata() {
    let f = Fixture::new("c_fen");
    f.mutate("meta/val.meta.jsonl", |r| {
        let other = r[1]["fen"].clone();
        r[0]["fen"] = other;
    });
    f.expect_fail("rows-vs-meta");
}

#[test]
fn detects_budget_mismatch() {
    let f = Fixture::new("c_budget");
    f.mutate("data/val.jsonl", |r| {
        let b = r[0]["budget"].as_u64().unwrap();
        r[0]["budget"] = Value::from(if b == 1 { 2 } else { 1 });
    });
    f.expect_fail("rows-vs-meta");
}

#[test]
fn detects_duplicate_row() {
    let f = Fixture::new("c_dup");
    f.mutate("data/fit.jsonl", |r| {
        let x = r[5].clone();
        r.push(x);
    });
    f.expect_fail("duplicate row id");
}

#[test]
fn detects_missing_row() {
    let f = Fixture::new("c_missing");
    f.mutate("data/val.jsonl", |r| {
        r.remove(7);
    });
    f.expect_fail("has no model row");
}

#[test]
fn detects_extra_row() {
    let f = Fixture::new("c_extra");
    f.mutate("data/fit.jsonl", |r| {
        let mut x = r[0].clone();
        x["id"] = Value::String("fit-extra-row".into());
        r.push(x);
    });
    f.expect_fail("has no metadata");
}

#[test]
fn detects_misassigned_partition() {
    let f = Fixture::new("c_part");
    f.mutate("meta/fit.meta.jsonl", |r| {
        r[0]["partition"] = Value::String("Val".into());
    });
    f.expect_fail("partition");
}

#[test]
fn detects_wrong_group_association() {
    let f = Fixture::new("c_group");
    f.mutate("meta/fit.meta.jsonl", |r| {
        r[0]["group_id"] = Value::String("0".repeat(24));
    });
    f.expect_fail("group");
}

#[test]
fn detects_wrong_target_even_when_row_and_metadata_agree() {
    let f = Fixture::new("c_target");
    for rel in ["data/fit.jsonl", "meta/fit.meta.jsonl"] {
        f.mutate(rel, |r| {
            let b = r[2]["label"].as_bool().unwrap();
            r[2]["label"] = Value::Bool(!b);
        });
    }
    f.expect_fail("exact-target");
}

#[test]
fn detects_corrupted_canonical_identity() {
    let f = Fixture::new("c_canon");
    f.mutate("pool/roots.jsonl", |r| {
        let k = r[0]["key"].as_str().unwrap().to_string();
        let mut k = k.into_bytes();
        k[130 - 2] = if k[130 - 2] == b'0' { b'1' } else { b'0' };
        r[0]["key"] = Value::String(String::from_utf8(k).unwrap());
    });
    f.expect_fail("canon");
}

#[test]
fn detects_example_not_a_child_of_claimed_root() {
    let f = Fixture::new("c_child");
    f.mutate("meta/fit.meta.jsonl", |r| {
        r[0]["mv"] = Value::String("a1a2".into());
    });
    f.expect_fail("child");
}

#[test]
fn rejects_unknown_fields_in_model_rows() {
    let f = Fixture::new("c_unknown");
    f.mutate("data/fit.jsonl", |r| {
        r[0]["root_depth"] = Value::from(2);
    });
    assert!(f.audit().is_err(), "unknown field must fail visibly");
}

#[test]
fn manifest_hash_mismatch_detected() {
    let f = Fixture::new("c_hash");
    let p = f.path("data/fit.jsonl");
    let mut t = std::fs::read_to_string(&p).unwrap();
    t.push('\n');
    std::fs::write(&p, t).unwrap(); // not re-pinned
    f.expect_fail("manifest");
}

// ------------------------------------------------------------ role access policy

fn acc(role: Role) -> Access {
    let c = Custody::new(base()).unwrap();
    Access::new(&c, role, Path::new("base"), Path::new("seed/test_seed.hex")).unwrap()
}

#[test]
fn learner_reads_only_fit_and_seed() {
    let a = acc(Role::Learner);
    assert!(a.read(Path::new("base/data/fit.jsonl")).is_ok());
    assert!(a.read(Path::new("seed/test_seed.hex")).is_ok());
    for bad in ["base/data/val.jsonl", "base/sealed/test.jsonl", "base/sealed/test.meta.jsonl", "base/pool/roots.jsonl", "base/meta/fit.meta.jsonl", "base/generation_report.json"] {
        let e = a.read(Path::new(bad)).unwrap_err().to_string();
        assert!(e.contains("ACCESS VIOLATION"), "{bad}: {e}");
    }
}

#[test]
fn evaluator_reads_fit_and_val_only() {
    let a = acc(Role::Evaluator);
    assert!(a.read(Path::new("base/data/fit.jsonl")).is_ok());
    assert!(a.read(Path::new("base/data/val.jsonl")).is_ok());
    for bad in ["base/sealed/test.jsonl", "base/pool/roots.jsonl", "base/meta/val.meta.jsonl", "base/sealed/test.meta.jsonl"] {
        assert!(a.read(Path::new(bad)).unwrap_err().to_string().contains("ACCESS VIOLATION"), "{bad}");
    }
}

#[test]
fn aggregator_reads_metadata_not_model_rows_pool_or_sealed() {
    let a = acc(Role::MetricAggregator);
    assert!(a.read(Path::new("base/meta/fit.meta.jsonl")).is_ok());
    assert!(a.read(Path::new("base/meta/val.meta.jsonl")).is_ok());
    for bad in ["base/data/fit.jsonl", "base/sealed/test.meta.jsonl", "base/pool/roots.jsonl"] {
        assert!(a.read(Path::new(bad)).unwrap_err().to_string().contains("ACCESS VIOLATION"), "{bad}");
    }
}

#[test]
fn nobody_but_audit_writes_into_dataset_and_paths_outside_namespace_refused() {
    for role in [Role::Learner, Role::Evaluator, Role::MetricAggregator] {
        let a = acc(role);
        assert!(a.write(Path::new("base/data/fit.jsonl"), b"x").is_err());
        assert!(a.read(Path::new("../outside.txt")).is_err());
    }
    // learner may write only under fits/ and qual/
    let a = acc(Role::Learner);
    assert!(a.write(Path::new("fits/A/x.txt"), b"ok").is_ok());
    assert!(a.write(Path::new("eval/A/x.txt"), b"no").is_err());
    // dot-dot traversal into the sealed directory is refused
    assert!(a.read(Path::new("fits/../base/sealed/test.jsonl")).is_err());
}
