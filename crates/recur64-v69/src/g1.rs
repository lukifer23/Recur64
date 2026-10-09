//! G1 evaluation-only panel: exclusion against gen-001, keyed selection, label-independent
//! intervention map. Narrow adapter over the audited generator/teacher/grouping code; it does
//! not alter legality, targets, sampling distribution, class quotas or symmetry definitions.

use crate::dataset::{ChildRec, Example, Group, Partition, RootRec, build_groups, CAP_PER_ROOT_PER_CLASS};
use crate::generate::Family;
use crate::streams::{MasterSeed, keyed_u64};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};

pub const G1_QUOTA: usize = 128;
pub const G1_TOTAL: usize = 1536;
pub const STREAM_G1_INTERVENTION: &str = "g1/intervention";
pub const STREAM_G1_BOOTSTRAP: &str = "g1/bootstrap";

/// Full canonical identities (hex of the 65-byte key) of gen-001 roots and ALL their children.
#[derive(Serialize, Deserialize, Default, Debug)]
pub struct ExclusionIndex {
    pub roots: Vec<String>,
    pub children: Vec<String>,
}

impl ExclusionIndex {
    pub fn from_pool(pool: &[RootRec]) -> Self {
        let mut roots: Vec<String> = pool.iter().map(|r| r.key.clone()).collect();
        let mut children: Vec<String> = pool.iter().flat_map(|r| r.children.iter().map(|c| c.key.clone())).collect();
        roots.sort();
        roots.dedup();
        children.sort();
        children.dedup();
        Self { roots, children }
    }
}

#[derive(Serialize, Debug, Default, Clone)]
pub struct ExclusionStats {
    pub roots_before: usize,
    pub groups_before: usize,
    pub roots_with_gen001_overlap: usize,
    pub root_key_overlaps: usize,
    pub child_key_overlaps: usize,
    pub groups_removed: usize,
    pub roots_removed: usize,
    pub roots_after: usize,
    pub groups_after: usize,
}

/// Remove every connected component that contains a root whose canonical root key or any child key
/// is present in the gen-001 index. Full-key set membership (never a hash).
pub fn apply_exclusion(seed: &MasterSeed, pool: Vec<RootRec>, idx: &ExclusionIndex) -> (Vec<RootRec>, Vec<Group>, ExclusionStats) {
    let rset: HashSet<&str> = idx.roots.iter().map(String::as_str).collect();
    let cset: HashSet<&str> = idx.children.iter().map(String::as_str).collect();
    let groups = build_groups(seed, &pool);
    let mut st = ExclusionStats { roots_before: pool.len(), groups_before: groups.len(), ..Default::default() };
    let mut bad_root: Vec<bool> = vec![false; pool.len()];
    for (i, r) in pool.iter().enumerate() {
        let rk = rset.contains(r.key.as_str());
        let ck = r.children.iter().any(|c| cset.contains(c.key.as_str()));
        st.root_key_overlaps += rk as usize;
        st.child_key_overlaps += ck as usize;
        bad_root[i] = rk || ck;
        st.roots_with_gen001_overlap += (rk || ck) as usize;
    }
    let mut remove = vec![false; pool.len()];
    for g in &groups {
        if g.roots.iter().any(|&i| bad_root[i]) {
            st.groups_removed += 1;
            for &i in &g.roots {
                remove[i] = true;
            }
        }
    }
    st.roots_removed = remove.iter().filter(|x| **x).count();
    let kept: Vec<RootRec> = pool.into_iter().enumerate().filter(|(i, _)| !remove[*i]).map(|(_, r)| r).collect();
    let groups = build_groups(seed, &kept);
    st.roots_after = kept.len();
    st.groups_after = groups.len();
    (kept, groups, st)
}

fn cell_name(f: &str, b: u8, label: bool) -> String {
    format!("{f}/n{b}/{}", if label { "pos" } else { "neg" })
}

/// Deterministic keyed selection of exactly 128 per family x budget x class cell.
/// Roots of a cell are visited in keyed order; at most 2 positive and 2 negative children per root;
/// a canonical child already used at the same budget is skipped.
pub fn select_g1(seed: &MasterSeed, roots: &[RootRec], groups: &[Group]) -> (Vec<Example>, BTreeMap<String, usize>) {
    let mut root_group = vec![0usize; roots.len()];
    for (gi, g) in groups.iter().enumerate() {
        for &r in &g.roots {
            root_group[r] = gi;
        }
    }
    let mut examples = Vec::new();
    let mut counts = BTreeMap::new();
    let mut taken: HashSet<(String, u8)> = HashSet::new();
    for f in Family::ALL {
        for b in [1u8, 2] {
            let mut rs: Vec<usize> = (0..roots.len()).filter(|&i| roots[i].family == f.name() && roots[i].depth == b + 1).collect();
            rs.sort_by_key(|&i| keyed_u64(seed, "selection/root", roots[i].key.as_bytes()));
            for class in [false, true] {
                let want = if class { "pos" } else { "neg" };
                let mut count = 0usize;
                for &ri in &rs {
                    if count >= G1_QUOTA {
                        break;
                    }
                    let r = &roots[ri];
                    let mut cand: Vec<&ChildRec> = r.children.iter().filter(|c| c.status == want).collect();
                    cand.sort_by_key(|c| keyed_u64(seed, "selection/child", format!("{}|{}", r.key, c.mv).as_bytes()));
                    let mut got = 0;
                    for c in cand {
                        if got >= CAP_PER_ROOT_PER_CLASS || count >= G1_QUOTA {
                            break;
                        }
                        if !taken.insert((c.key.clone(), b)) {
                            continue;
                        }
                        examples.push(Example {
                            id: format!("g1-{}-{}", crate::canon::key_id(&c.key), b),
                            partition: Partition::Val, // placeholder: G1 has no partitions
                            family: f.name().to_string(),
                            budget: b,
                            label: class,
                            fen: c.fen.clone(),
                            key: c.key.clone(),
                            root_id: r.id.clone(),
                            group_id: groups[root_group[ri]].id.clone(),
                            root_fen: r.fen.clone(),
                            root_depth: r.depth,
                            mv: c.mv.clone(),
                        });
                        got += 1;
                        count += 1;
                    }
                }
                counts.insert(cell_name(f.name(), b, class), count);
            }
        }
    }
    (examples, counts)
}

pub fn feasible(counts: &BTreeMap<String, usize>) -> bool {
    counts.len() == 12 && counts.values().all(|c| *c >= G1_QUOTA)
}

/// Label-independent derangement within family x budget cells: ids sorted, shuffled by the keyed
/// stream `g1/intervention/<family>/<budget>`, mapped to the next id cyclically.
pub fn intervention_map(seed: &MasterSeed, examples: &[Example]) -> Result<BTreeMap<String, String>> {
    let mut cells: BTreeMap<(String, u8), Vec<String>> = BTreeMap::new();
    for e in examples {
        cells.entry((e.family.clone(), e.budget)).or_default().push(e.id.clone());
    }
    let mut map = BTreeMap::new();
    for ((f, b), mut ids) in cells {
        ids.sort();
        let mut rng = seed.stream(&format!("{STREAM_G1_INTERVENTION}/{f}/{b}"), 0);
        for i in (1..ids.len()).rev() {
            ids.swap(i, rng.below(i as u64 + 1) as usize);
        }
        let n = ids.len();
        ensure!(n >= 2, "cell too small");
        for i in 0..n {
            map.insert(ids[i].clone(), ids[(i + 1) % n].clone());
        }
    }
    let donors: HashSet<&String> = map.values().collect();
    ensure!(map.len() == examples.len() && donors.len() == map.len() && map.iter().all(|(k, v)| k != v), "not a derangement");
    Ok(map)
}

/// R1 (owner-approved amendment): identity-level exclusion first, grouping afterwards.
/// A new root is removed iff its canonical root key, or any of its immediate-child keys, occurs in the
/// gen-001 index (full 65-byte keys). Connected groups are then formed among the survivors (for
/// deduplication and the cluster bootstrap). Guarantee: no kept root/child canonical identity occurs
/// in gen-001. Not guaranteed: transitive component-level separation from gen-001.
pub fn apply_exclusion_r1(seed: &MasterSeed, pool: Vec<RootRec>, idx: &ExclusionIndex) -> (Vec<RootRec>, Vec<Group>, ExclusionStats) {
    let rset: HashSet<&str> = idx.roots.iter().map(String::as_str).collect();
    let cset: HashSet<&str> = idx.children.iter().map(String::as_str).collect();
    let mut st = ExclusionStats { roots_before: pool.len(), groups_before: build_groups(seed, &pool).len(), ..Default::default() };
    let mut kept = Vec::with_capacity(pool.len());
    for r in pool {
        let rk = rset.contains(r.key.as_str());
        let ck = r.children.iter().any(|c| cset.contains(c.key.as_str()));
        st.root_key_overlaps += rk as usize;
        st.child_key_overlaps += ck as usize;
        if rk || ck {
            st.roots_with_gen001_overlap += 1;
        } else {
            kept.push(r);
        }
    }
    st.roots_removed = st.roots_with_gen001_overlap;
    let groups = build_groups(seed, &kept);
    st.roots_after = kept.len();
    st.groups_after = groups.len();
    (kept, groups, st)
}
