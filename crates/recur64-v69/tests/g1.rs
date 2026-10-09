//! G1 tests: role restrictions, exclusion logic, label-independent intervention map.

use recur64_v69::access::{Access, Role};
use recur64_v69::custody::Custody;
use recur64_v69::dataset::{ChildRec, Example, Partition, RootRec};
use recur64_v69::g1::*;
use recur64_v69::streams::MasterSeed;
use std::path::Path;

fn seed() -> MasterSeed {
    MasterSeed::from_hex(&"3c".repeat(32)).unwrap()
}

fn hexkey(s: &str) -> String {
    format!("{:0<130}", s.bytes().map(|b| format!("{b:02x}")).collect::<String>())
}

fn root(id: &str, rk: &str, children: &[&str]) -> RootRec {
    RootRec {
        id: id.into(),
        family: "KQQvK".into(),
        attacker: "white".into(),
        fen: String::new(),
        key: hexkey(rk),
        depth: 2,
        legal_moves: children.len(),
        correct_moves: 1,
        terminal_children: 0,
        children: children.iter().map(|c| ChildRec { mv: format!("m{c}"), fen: String::new(), key: hexkey(c), status: "neg".into() }).collect(),
        oracle_nodes: 0,
        oracle_micros: 0,
    }
}

#[test]
fn exclusion_removes_whole_connected_components_by_full_key() {
    let gen001 = vec![root("g0", "G0", &["x1", "x2"])];
    let idx = ExclusionIndex::from_pool(&gen001);
    // r1 shares child x2 with gen-001; r2 is linked to r1 through child y1; r3 has the same ROOT key as a gen-001 root; r4 is clean
    let pool = vec![root("r1", "R1", &["x2", "y1"]), root("r2", "R2", &["y1", "y2"]), root("r3", "G0", &["z1"]), root("r4", "R4", &["w1"])];
    let (kept, groups, st) = apply_exclusion(&seed(), pool, &idx);
    let ids: Vec<&str> = kept.iter().map(|r| r.id.as_str()).collect();
    assert_eq!(ids, vec!["r4"], "r1+r2 (connected component) and r3 (root overlap) must be removed");
    assert_eq!(st.groups_removed, 2);
    assert_eq!(st.roots_removed, 3);
    assert_eq!(groups.len(), 1);
    // no kept root/child key occurs in the index
    for r in &kept {
        assert!(!idx.roots.contains(&r.key));
        assert!(r.children.iter().all(|c| !idx.children.contains(&c.key)));
    }
}

fn ex(id: &str, f: &str, b: u8, label: bool) -> Example {
    Example { id: id.into(), partition: Partition::Val, family: f.into(), budget: b, label, fen: String::new(), key: String::new(), root_id: String::new(), group_id: String::new(), root_fen: String::new(), root_depth: b + 1, mv: String::new() }
}

#[test]
fn intervention_map_is_a_label_independent_derangement_within_cells() {
    let mut v = Vec::new();
    for f in ["KQQvK", "KRRvK"] {
        for b in [1u8, 2] {
            for i in 0..40 {
                v.push(ex(&format!("{f}-{b}-{i:03}"), f, b, i % 2 == 0));
            }
        }
    }
    let m1 = intervention_map(&seed(), &v).unwrap();
    let flipped: Vec<Example> = v.iter().map(|e| ex(&e.id, &e.family, e.budget, !e.label)).collect();
    assert_eq!(m1, intervention_map(&seed(), &flipped).unwrap(), "labels must not influence the map");
    assert!(m1.iter().all(|(k, d)| k != d));
    let donors: std::collections::HashSet<_> = m1.values().collect();
    assert_eq!(donors.len(), m1.len());
    for (k, d) in &m1 {
        let (a, b) = (v.iter().find(|e| &e.id == k).unwrap(), v.iter().find(|e| &e.id == d).unwrap());
        assert_eq!((&a.family, a.budget), (&b.family, b.budget), "donor stays within the family/budget cell");
    }
}

#[test]
fn g1_roles_are_restricted() {
    let dir = std::env::temp_dir().join(format!("v69-g1-test-{}", std::process::id())).join("artifacts").join("v69");
    let _ = std::fs::remove_dir_all(dir.parent().unwrap());
    for p in ["g1r1/rows", "g1r1/meta", "g1r1/index", "g1r1/seed", "g1r1/eval", "d3/fits/A/final", "gen-001/data", "gen-001/pool", "gen-001/sealed", "d1/baseline"] {
        std::fs::create_dir_all(dir.join(p)).unwrap();
    }
    for p in ["g1r1/rows/g1_rows.jsonl", "g1r1/meta/g1_meta.jsonl", "g1r1/index/i.json", "g1r1/seed/g1_master_seed.hex", "d3/fits/A/final/model.mpk", "d3/fits/A/final/opt_decay.mpk", "gen-001/data/val.jsonl", "gen-001/pool/roots.jsonl", "gen-001/sealed/test.jsonl", "d1/baseline/model.json"] {
        std::fs::write(dir.join(p), b"x").unwrap();
    }
    let c = Custody::new(&dir).unwrap();
    let acc = |r| Access::new(&c, r, Path::new("gen-001"), Path::new("g1r1/seed/g1_master_seed.hex")).unwrap();
    let ev = acc(Role::G1Evaluator);
    for ok in ["g1r1/rows/g1_rows.jsonl", "d3/fits/A/final/model.mpk", "d1/baseline/model.json"] {
        assert!(ev.read(Path::new(ok)).is_ok(), "{ok}");
    }
    for bad in ["g1r1/meta/g1_meta.jsonl", "g1r1/index/i.json", "d3/fits/A/final/opt_decay.mpk", "gen-001/data/val.jsonl", "gen-001/pool/roots.jsonl", "gen-001/sealed/test.jsonl"] {
        assert!(ev.read(Path::new(bad)).unwrap_err().to_string().contains("ACCESS VIOLATION"), "{bad}");
    }
    assert!(ev.write(Path::new("g1r1/eval/A/p.jsonl"), b"ok").is_ok());
    assert!(ev.write(Path::new("g1r1/rows/g1_rows.jsonl"), b"no").is_err());
    let ag = acc(Role::G1Aggregator);
    assert!(ag.read(Path::new("g1r1/meta/g1_meta.jsonl")).is_ok());
    assert!(ag.read(&Path::new("gen-001").join("sealed/test.jsonl")).is_err());
    let b = acc(Role::G1Builder);
    assert!(b.read(&Path::new("gen-001").join("pool/roots.jsonl")).is_ok());
    assert!(b.read(&Path::new("gen-001").join("sealed/test.jsonl")).is_err());
    assert!(b.write(Path::new("g1r1/meta/x.json"), b"ok").is_ok());
}

#[test]
fn r1_excludes_roots_by_identity_then_groups_survivors() {
    let gen001 = vec![root("g0", "G0", &["x1", "x2"])];
    let idx = ExclusionIndex::from_pool(&gen001);
    // r1 shares child x2 with gen-001 (removed); r2 is connected to r1 only through y1 but has no direct overlap (kept);
    // r3 has gen-001's root key (removed); r4 clean (kept)
    let pool = vec![root("r1", "R1", &["x2", "y1"]), root("r2", "R2", &["y1", "y2"]), root("r3", "G0", &["z1"]), root("r4", "R4", &["w1"])];
    let (kept, groups, st) = apply_exclusion_r1(&seed(), pool, &idx);
    let ids: Vec<&str> = kept.iter().map(|r| r.id.as_str()).collect();
    assert_eq!(ids, vec!["r2", "r4"], "only directly overlapping roots are removed");
    assert_eq!((st.roots_removed, st.root_key_overlaps, st.child_key_overlaps), (2, 1, 1));
    assert_eq!(groups.len(), 2, "groups are formed after exclusion");
    for r in &kept {
        assert!(!idx.roots.contains(&r.key));
        assert!(r.children.iter().all(|c| !idx.children.contains(&c.key)), "no kept child identity occurs in gen-001");
    }
}
