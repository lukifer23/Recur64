//! D2 tests on synthetic fixtures: nested subsets, quotas, determinism, example streams,
//! append-only receipts, hash enforcement and role restrictions.

use recur64_v69::access::{Access, Role};
use recur64_v69::custody::Custody;
use recur64_v69::d1::{FrozenD1, verify_group};
use recur64_v69::d2::*;
use recur64_v69::dataset::{Example, Partition};
use recur64_v69::provenance::sha256_hex;
use recur64_v69::streams::MasterSeed;
use std::collections::{BTreeMap, HashSet};
use std::path::Path;

fn seed() -> MasterSeed {
    MasterSeed::from_hex(&"7d".repeat(32)).unwrap()
}

/// 768 synthetic fitting rows: 12 cells x 64; groups shared in pairs for some rows.
fn fit_meta() -> Vec<Example> {
    let mut v = Vec::new();
    let mut k = 0;
    for f in ["KQQvK", "KQRvK", "KRRvK"] {
        for b in [1u8, 2] {
            for label in [false, true] {
                for i in 0..64 {
                    k += 1;
                    v.push(Example {
                        id: format!("fit-{k:04}"),
                        partition: Partition::Fit,
                        family: f.into(),
                        budget: b,
                        label,
                        fen: String::new(),
                        key: String::new(),
                        root_id: format!("r{k}"),
                        group_id: format!("g{}", if i % 8 == 0 { k - 1 } else { k }), // some rows share a group with the previous row
                        root_fen: String::new(),
                        root_depth: b + 1,
                        mv: String::new(),
                    });
                }
            }
        }
    }
    v
}

#[test]
fn nesting_quota_balance_group_preference() {
    let meta = fit_meta();
    // build a 32-id balanced panel: 2 per cell (24) + 1 pos/1 neg in four strata (8)
    let mut ids: HashSet<String> = HashSet::new();
    let mut taken: BTreeMap<String, usize> = BTreeMap::new();
    let extra_strata = ["KQQvK/1", "KQQvK/2", "KRRvK/2", "KQRvK/2"];
    for e in &meta {
        let cell = format!("{}/{}/{}", e.family, e.budget, e.label);
        let stratum = format!("{}/{}", e.family, e.budget);
        let limit = 2 + extra_strata.contains(&stratum.as_str()) as usize;
        let n = taken.entry(cell).or_default();
        if *n < limit {
            ids.insert(e.id.clone());
            *n += 1;
        }
    }
    assert_eq!(ids.len(), 32);
    let a = select_nested(&seed(), &meta, &ids).unwrap();
    let b = select_nested(&seed(), &meta, &ids).unwrap();
    assert_eq!(a.len(), 3);
    let mut prev: HashSet<String> = HashSet::new();
    for (i, ((n, rows, sum), (_, rows2, _))) in a.iter().zip(&b).enumerate() {
        assert_eq!(rows.iter().map(|e| &e.id).collect::<Vec<_>>(), rows2.iter().map(|e| &e.id).collect::<Vec<_>>(), "determinism");
        let s: HashSet<String> = rows.iter().map(|e| e.id.clone()).collect();
        assert_eq!(s.len(), *n);
        assert!(prev.is_subset(&s), "nesting");
        if i == 0 {
            assert!(ids.is_subset(&s), "panel in 64");
        }
        assert_eq!(sum.positives, n / 2);
        assert!(sum.per_stratum_balanced);
        let q = n / 12;
        for c in sum.per_cell.values() {
            assert!(*c == q || *c == q + 1);
        }
        assert_eq!(sum.size, *n);
        prev = s;
    }
    assert_eq!(a[2].1.len(), 768);
    assert_eq!(a[0].2.q_per_cell, 5);
    assert_eq!(a[0].2.extra_pair_strata.len(), 2);
    assert_eq!(a[1].2.q_per_cell, 21);
    assert_eq!(a[2].2.extra_pair_strata.len(), 0);
    // group preference: fewer repeated memberships than a random pick would typically produce is not guaranteed, but
    // the report must be internally consistent
    for (n, _, sum) in &a {
        assert_eq!(sum.repeated_group_memberships, n - sum.distinct_groups);
    }
}

#[test]
fn infeasible_panel_is_refused_not_silently_changed() {
    let meta = fit_meta();
    // a "panel" with 6 positives in one cell cannot nest into the size-64 quota (5 or 6 per cell)
    let mut ids: HashSet<String> = meta.iter().filter(|e| e.family == "KQQvK" && e.budget == 1 && e.label).take(7).map(|e| e.id.clone()).collect();
    for e in meta.iter().filter(|e| !(e.family == "KQQvK" && e.budget == 1 && e.label)).take(25) {
        ids.insert(e.id.clone());
    }
    assert_eq!(ids.len(), 32);
    assert!(select_nested(&seed(), &meta, &ids).is_err());
}

#[test]
fn order_is_deterministic_epoch_permutations_with_exact_exposure() {
    for n in [64usize, 256, 768] {
        let o = d2_order(&seed(), n, n);
        assert_eq!(o.len(), D2_UPDATES * D2_BATCH);
        assert_eq!(o, d2_order(&seed(), n, n));
        for ep in 0..o.len() / n {
            let s: HashSet<usize> = o[ep * n..(ep + 1) * n].iter().copied().collect();
            assert_eq!(s.len(), n);
        }
        let mut expo = vec![0u32; n];
        for i in &o {
            expo[*i] += 1;
        }
        let (mn, mx) = (expo.iter().min().unwrap(), expo.iter().max().unwrap());
        assert!(mx - mn <= 1, "exposure spread {mn}..{mx}");
        // average exposure = 38400 / N
        assert_eq!(expo.iter().sum::<u32>() as usize, D2_UPDATES * D2_BATCH);
    }
    assert_ne!(d2_order(&seed(), 64, 64), d2_order(&seed(), 256, 256)[..38400].to_vec());
    // equal-exposure mapping used in the report: exposures at the named updates
    assert_eq!(50 * 16 / 64, 12);
    assert_eq!(200 * 16 / 256, 12);
    assert_eq!(600 * 16 / 768, 12);
}

fn temp_root(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("v69-d2-test-{}-{tag}", std::process::id())).join("artifacts").join("v69");
    let _ = std::fs::remove_dir_all(d.parent().unwrap());
    std::fs::create_dir_all(d.join("seed")).unwrap();
    std::fs::create_dir_all(d.join("gen-001/data")).unwrap();
    std::fs::create_dir_all(d.join("gen-001/meta")).unwrap();
    std::fs::write(d.join("seed/s.hex"), "7d".repeat(32)).unwrap();
    d
}

#[test]
fn receipts_are_append_only_hashes_enforced_and_roles_restricted() {
    let root = temp_root("a");
    let c = Custody::new(&root).unwrap();
    std::fs::create_dir_all(root.join("d2")).unwrap();
    std::fs::write(root.join("d2/config.json"), b"{}").unwrap();
    std::fs::write(root.join("gen-001/data/fit.jsonl"), b"rows").unwrap();
    std::fs::write(root.join("gen-001/meta/fit.meta.jsonl"), b"meta").unwrap();
    let access = |r| Access::new(&c, r, Path::new("gen-001"), Path::new("seed/s.hex")).unwrap();
    let audit = access(Role::DataAudit);
    write_new(&audit, "d2/receipts/r1.json", b"one").unwrap();
    assert!(write_new(&audit, "d2/receipts/r1.json", b"two").is_err(), "append-only");
    assert_eq!(std::fs::read(root.join("d2/receipts/r1.json")).unwrap(), b"one");
    // frozen enforcement
    let mut g = BTreeMap::new();
    g.insert("d2/config.json".to_string(), sha256_hex(b"{}"));
    g.insert("dataset:data/fit.jsonl".to_string(), sha256_hex(b"rows"));
    let mut groups = BTreeMap::new();
    groups.insert("learner".to_string(), g);
    let fz = FrozenD1 { created_utc: "t".into(), run: "gen-001".into(), seed_fingerprint: "x".into(), groups };
    let learner = access(Role::Learner);
    assert_eq!(verify_group(&learner, Path::new("gen-001"), &fz, "learner").unwrap(), 2);
    std::fs::write(root.join("d2/config.json"), b"{ }").unwrap();
    let e = verify_group(&learner, Path::new("gen-001"), &fz, "learner").unwrap_err().to_string();
    assert!(e.contains("FROZEN HASH MISMATCH"), "{e}");
    // roles: learner reads d2/ and fit rows, never fit metadata or validation
    assert!(learner.read(Path::new("d2/config.json")).is_ok());
    assert!(learner.read(&Path::new("gen-001").join("data/fit.jsonl")).is_ok());
    assert!(learner.read(&Path::new("gen-001").join("meta/fit.meta.jsonl")).unwrap_err().to_string().contains("ACCESS VIOLATION"));
    assert!(learner.read(&Path::new("gen-001").join("data/val.jsonl")).unwrap_err().to_string().contains("ACCESS VIOLATION"));
    assert!(learner.write(Path::new("audit/x.json"), b"no").is_err());
    let agg = access(Role::MetricAggregator);
    assert!(agg.read(&Path::new("gen-001").join("meta/fit.meta.jsonl")).is_ok());
    assert!(agg.read(&Path::new("gen-001").join("data/fit.jsonl")).is_err());
}

#[test]
fn d3_order_extends_d2_exactly_with_250_exposures() {
    let s = seed();
    let d2 = d2_order(&s, 768, 768);
    let d3 = d3_order(&s, 768, 768, D3_UPDATES);
    assert_eq!(d3.len(), D3_UPDATES * D2_BATCH);
    assert_eq!(&d3[..d2.len()], &d2[..], "D3 prefix must equal the D2 stream");
    let mut expo = vec![0u32; 768];
    for i in &d3 {
        expo[*i] += 1;
    }
    assert!(expo.iter().all(|e| *e == 250));
    assert_eq!(d3, d3_order(&s, 768, 768, D3_UPDATES), "deterministic");
    assert_eq!(D3_SNAPSHOTS.len(), 21);
    assert!(D2_SNAPSHOTS.iter().all(|u| D3_SNAPSHOTS.contains(u)));
}
