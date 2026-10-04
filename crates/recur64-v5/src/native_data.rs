//! HP-native data contracts and exact feasibility gate. Emits no accepted split.
use recur64_runtime::proof::audit::{AuditMemo, audit_position};
use recur64_runtime::proof::generator::{
    Candidate, canonical_key, enumerate_pool, label_exact_pool,
};
use recur64_runtime::proof::targets::{FAMILIES, ProofPosition, Split};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

/// No production V5 data identity can be bound before complete measured DATA-B.
/// P25 is retired, and no replacement digest is fabricated.
pub fn require_measured_binding() -> anyhow::Result<()> {
    anyhow::bail!(
        "V5 HP-native lineage is not completely generated/audited/bound; all scientific data loaders are locked. P25 is no longer a V5 prerequisite or accepted replacement"
    )
}

pub const FAMILY: &str = "v5_hp_exact_endgames_v1";
pub const SELECT: &str = "v5_hp_dataset_select_v1";
pub const AUDIT: &str = "v5_hp_independent_audit_v1";
pub const TRAIN_SEED: u64 = 0x7A50_1001;
pub const DEV_SEED: u64 = 0x7A50_1002;
pub const CONFIRM_SEED: u64 = 0x7A50_1003;

pub fn selection_key(seed: u64, canon: &str) -> String {
    let mut h = Sha256::new();
    h.update(SELECT.as_bytes());
    h.update([0]);
    h.update(seed.to_le_bytes());
    h.update(canon.as_bytes());
    format!("{:x}", h.finalize())
}
/// Count is strict. Deduplication/exclusion never changes the requested quota.
pub fn select(
    pool: &[Candidate],
    seed: u64,
    need: usize,
    exclude_fen: &BTreeSet<String>,
    exclude_canon: &BTreeSet<String>,
) -> anyhow::Result<Vec<Candidate>> {
    anyhow::ensure!(need > 0, "empty requested cell");
    let mut unique = BTreeMap::new();
    let mut fens = BTreeSet::new();
    for c in pool {
        anyhow::ensure!(
            canonical_key(&c.fen) == c.canon,
            "candidate canonical identity mismatch"
        );
        anyhow::ensure!(
            fens.insert(c.fen.clone()),
            "duplicate FEN in candidate pool"
        );
        anyhow::ensure!(
            unique.insert(c.canon.clone(), c.clone()).is_none(),
            "duplicate canonical class in pool"
        );
    }
    let mut eligible: Vec<_> = unique
        .into_values()
        .filter(|c| !exclude_canon.contains(&c.canon) && !exclude_fen.contains(&c.fen))
        .collect();
    eligible.sort_by_cached_key(|c| (selection_key(seed, &c.canon), c.canon.clone()));
    anyhow::ensure!(
        eligible.len() >= need,
        "STOP: only {} unique eligible classes, need {need}",
        eligible.len()
    );
    eligible.truncate(need);
    Ok(eligible)
}
pub fn audit_records(records: &[ProofPosition]) -> anyhow::Result<usize> {
    let mut fens = BTreeSet::new();
    let mut canons = BTreeSet::new();
    let mut ids = BTreeSet::new();
    let mut memo = AuditMemo::default();
    for (i, p) in records.iter().enumerate() {
        anyhow::ensure!(
            fens.insert(&p.fen) && canons.insert(&p.canon) && ids.insert(&p.id),
            "duplicate FEN/canonical/ID"
        );
        anyhow::ensure!(canonical_key(&p.fen) == p.canon, "canonical mismatch");
        let pieces = FAMILIES
            .iter()
            .find(|f| f.0 == p.family)
            .ok_or_else(|| anyhow::anyhow!("unknown family"))?
            .1;
        let mut expected = pieces.to_vec();
        expected.push('k');
        expected.sort();
        let mut actual: Vec<_> = p.canon.chars().filter(|c| *c != '.').collect();
        actual.sort();
        anyhow::ensure!(expected == actual, "material/family mismatch");
        if i % 16 == 0 {
            memo = AuditMemo::default();
        } // bounded independent GameState memo
        audit_position(p, &mut memo).map_err(|e| anyhow::anyhow!(e))?;
    }
    Ok(records.len())
}
fn set_digest<'a>(values: impl Iterator<Item = &'a str>) -> String {
    let sorted: BTreeSet<_> = values.collect();
    let mut h = Sha256::new();
    for v in sorted {
        h.update(v.as_bytes());
        h.update(b"\n");
    }
    format!("{:x}", h.finalize())
}
/// Exhaustively verifies the requested 2000-class cell against the ENTIRE space.
/// Two full enumerations, different thread counts, plus independent label audit.
pub fn capacity(source: &str, threads: usize) -> anyhow::Result<Value> {
    anyhow::ensure!(
        threads > 0 && threads <= 16,
        "capacity threads must be 1..16"
    );
    let (a, pool) = enumerate_pool(1, 1, threads)?;
    let (b, again) = enumerate_pool(1, 1, 1)?;
    anyhow::ensure!(
        pool.len() == again.len()
            && pool
                .iter()
                .zip(&again)
                .all(|(x, y)| (&x.canon, &x.fen, x.depth) == (&y.canon, &y.fen, y.depth)),
        "full capacity regeneration mismatch"
    );
    anyhow::ensure!(
        a.raw_placements == 64 * 63 * 62
            && a.canonical_classes == b.canonical_classes
            && a.eligible_by_depth == b.eligible_by_depth,
        "exhaustive census/reproduction failed"
    );
    let records = label_exact_pool(&pool, "KRvK", Split::Train, TRAIN_SEED)?;
    let checked = audit_records(&records)?;
    let pass = pool.len() >= 2000;
    let quota_check = select(&pool, TRAIN_SEED, 2000, &BTreeSet::new(), &BTreeSet::new());
    anyhow::ensure!(
        quota_check.is_ok() == pass,
        "quota enforcement inconsistent"
    );
    Ok(
        json!({"schema":"v5_hp_dataset_capacity_v1","source_sha":source,"architecture":crate::config::ARCHITECTURE,"config_digest":crate::config::V5Config::default().scientific_digest()?,"dataset_family":FAMILY,"selection_contract":SELECT,"audit_contract":AUDIT,"requested_dataset":"V5_HP_TRAIN_V1","generation_seed":TRAIN_SEED,"family":"KRvK","mate_depth":1,"required_count":2000,"available_canonical_classes":pool.len(),"quota_pass":pass,"exclusions_applied":0,"capacity_is_upper_bound_before_exclusions":true,"audit_checked":checked,"audit_failures":0,"canonical_set_digest":set_digest(pool.iter().map(|p|p.canon.as_str())),"exact_fen_set_digest":set_digest(pool.iter().map(|p|p.fen.as_str())),"record_content_digest":format!("{:x}",Sha256::digest(serde_json::to_vec(&records)?)),"enumeration":a,"regeneration":b,"regeneration_identical":true,"complete_dataset_generated":false,"accepted_train_records":0,"accepted_dev_records":0,"accepted_confirm_records":0,"training_authorized":false,"confirm_evaluated":false}),
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn selection_is_stable_strict_and_rejects_duplicates() {
        let fens = [
            "7k/8/6K1/8/8/8/8/1Q6 w - - 0 1",
            "7k/8/6K1/8/8/8/8/2Q5 w - - 0 1",
        ];
        let mut pool: Vec<_> = fens
            .iter()
            .map(|f| Candidate {
                fen: f.to_string(),
                canon: canonical_key(f),
                depth: 1,
            })
            .collect();
        let empty = BTreeSet::new();
        let a = select(&pool, TRAIN_SEED, 2, &empty, &empty).unwrap();
        pool.reverse();
        let b = select(&pool, TRAIN_SEED, 2, &empty, &empty).unwrap();
        assert_eq!(
            a.iter().map(|p| &p.canon).collect::<Vec<_>>(),
            b.iter().map(|p| &p.canon).collect::<Vec<_>>()
        );
        assert_ne!(
            selection_key(TRAIN_SEED, &pool[0].canon),
            selection_key(DEV_SEED, &pool[0].canon)
        );
        assert_ne!(
            selection_key(DEV_SEED, &pool[0].canon),
            selection_key(CONFIRM_SEED, &pool[0].canon)
        );
        assert!(select(&pool, TRAIN_SEED, 3, &empty, &empty).is_err());
        let excluded = BTreeSet::from([pool[0].canon.clone()]);
        assert!(select(&pool, TRAIN_SEED, 2, &empty, &excluded).is_err());
        let mirror = "k7/8/1K6/8/8/8/8/6Q1 w - - 0 1";
        let duplicate_class = vec![
            Candidate {
                fen: fens[0].into(),
                canon: canonical_key(fens[0]),
                depth: 1,
            },
            Candidate {
                fen: mirror.into(),
                canon: canonical_key(mirror),
                depth: 1,
            },
        ];
        assert!(
            select(&duplicate_class, TRAIN_SEED, 1, &empty, &empty)
                .err()
                .unwrap()
                .to_string()
                .contains("duplicate canonical")
        );
        pool.push(pool[0].clone());
        assert!(select(&pool, TRAIN_SEED, 2, &empty, &empty).is_err());
    }
    #[test]
    fn production_lineage_lock_refuses_every_split_before_measured_binding() {
        let (_, pool) = enumerate_pool(1, 1, 1).unwrap();
        let dir =
            std::env::temp_dir().join(format!("recur64-v5-native-lock-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for split in [Split::Train, Split::Tune, Split::Confirm] {
            let positions = label_exact_pool(&pool[..1], "KRvK", split, TRAIN_SEED).unwrap();
            let t = recur64_runtime::proof::targets::ProofTargets::new(
                split,
                TRAIN_SEED,
                json!({"identity":"unbound"}),
                positions,
            );
            let path = dir.join(format!("{}.json", split.label()));
            t.save(&path).unwrap();
            assert!(
                crate::data::V5Data::load(&path)
                    .err()
                    .unwrap()
                    .to_string()
                    .contains("loaders are locked")
            );
            std::fs::remove_file(path).unwrap();
        }
        std::fs::remove_dir(dir).unwrap();
    }

    #[test]
    fn independent_audit_catches_wrong_depth_move_family_canon_and_terminal() {
        let (_, pool) = enumerate_pool(1, 1, 1).unwrap();
        let records = label_exact_pool(&pool[..1], "KRvK", Split::Train, TRAIN_SEED).unwrap();
        audit_records(&records).unwrap();
        let mut bad = records.clone();
        bad[0].mate_depth = 2;
        assert!(audit_records(&bad).is_err());
        let mut bad = records.clone();
        bad[0].correct = vec![
            (0..bad[0].legal.len() as u32)
                .find(|i| !bad[0].correct.contains(i))
                .unwrap(),
        ];
        assert!(audit_records(&bad).is_err());
        let mut bad = records.clone();
        bad[0].family = "KQvK".into();
        assert!(audit_records(&bad).is_err());
        let mut bad = records.clone();
        bad[0].canon = "bad".into();
        assert!(audit_records(&bad).is_err());
        let mut bad = records.clone();
        bad[0].fen = "7k/8/5KQ1/8/8/8/8/8 b - - 0 1".into();
        assert!(
            recur64_core::GameState::from_fen(&bad[0].fen)
                .unwrap()
                .is_terminal()
        );
        bad[0].canon = canonical_key(&bad[0].fen);
        bad[0].family = "KQvK".into();
        assert!(audit_records(&bad).is_err());
        let mut bad = records.clone();
        bad.push(bad[0].clone());
        assert!(audit_records(&bad).is_err());
    }
}
