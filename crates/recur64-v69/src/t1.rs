//! T1 dataset: train/val/test partitions from one fresh draw, identity-level exclusion against all prior
//! V69 data (gen-001 and the G1 pool), root-level partitioning (amendment A1), nested training subsets.

use crate::dataset::{ChildRec, Example, Partition, RootRec, build_groups, CAP_PER_ROOT_PER_CLASS};
use crate::generate::Family;
use crate::streams::{MasterSeed, keyed_u64};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};

pub const TRAIN_PER_CLASS_CELL: usize = 2000;
pub const VAL_PER_CLASS_CELL: usize = 128;
pub const TEST_PER_CLASS_CELL: usize = 256;
pub const SCALES: [usize; 3] = [250, 1000, 2000];

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct T1Meta {
    #[serde(flatten)]
    pub ex: Example,
    /// 0-based position within its (family, budget, class) cell in keyed order (nested subsets: rank < k).
    pub rank: usize,
}

pub fn quota(p: Partition) -> usize {
    match p {
        Partition::Fit => TRAIN_PER_CLASS_CELL,
        Partition::Val => VAL_PER_CLASS_CELL,
        Partition::Test => TEST_PER_CLASS_CELL,
    }
}

pub fn pname(p: Partition) -> &'static str {
    match p {
        Partition::Fit => "train",
        Partition::Val => "val",
        Partition::Test => "test",
    }
}

pub fn cell_name(f: &str, b: u8, label: bool) -> String {
    format!("{f}/n{b}/{}", if label { "pos" } else { "neg" })
}

/// Amendment A1: label-blind root-level assignment by keyed hash of the canonical root key, 80/10/10.
pub fn assign_roots(seed: &MasterSeed, roots: &[RootRec]) -> Vec<Partition> {
    roots
        .iter()
        .map(|r| match keyed_u64(seed, "t1/partition", r.key.as_bytes()) % 100 {
            0..=79 => Partition::Fit,
            80..=89 => Partition::Val,
            _ => Partition::Test,
        })
        .collect()
}

/// Deterministic keyed selection per partition and family x budget x class cell; per-root cap 2 per class.
/// Partitions are filled test, val, train with ONE global taken-set of (canonical child, budget), so no
/// identical child position appears in two partitions. `group_id` is attached afterwards (`attach_groups`).
pub fn select_t1(seed: &MasterSeed, roots: &[RootRec], root_part: &[Partition]) -> (Vec<T1Meta>, BTreeMap<String, usize>) {
    let mut out = Vec::new();
    let mut counts = BTreeMap::new();
    let mut taken: HashSet<String> = HashSet::new();
    for p in [Partition::Test, Partition::Val, Partition::Fit] {
        for f in Family::ALL {
            for b in [1u8, 2] {
                let mut rs: Vec<usize> = (0..roots.len()).filter(|&i| roots[i].family == f.name() && roots[i].depth == b + 1 && root_part[i] == p).collect();
                rs.sort_by_key(|&i| keyed_u64(seed, &format!("t1/sel/root/{}", pname(p)), roots[i].key.as_bytes()));
                for class in [false, true] {
                    let want = if class { "pos" } else { "neg" };
                    let mut count = 0usize;
                    for &ri in &rs {
                        if count >= quota(p) {
                            break;
                        }
                        let r = &roots[ri];
                        let mut cand: Vec<&ChildRec> = r.children.iter().filter(|c| c.status == want).collect();
                        cand.sort_by_key(|c| keyed_u64(seed, "t1/sel/child", format!("{}|{}", r.key, c.mv).as_bytes()));
                        let mut got = 0;
                        for c in cand {
                            if got >= CAP_PER_ROOT_PER_CLASS || count >= quota(p) {
                                break;
                            }
                            if !taken.insert(c.key.clone()) {
                                continue;
                            }
                            out.push(T1Meta {
                                ex: Example {
                                    id: format!("t1-{}-{}-{}", pname(p), crate::canon::key_id(&c.key), b),
                                    partition: p,
                                    family: f.name().to_string(),
                                    budget: b,
                                    label: class,
                                    fen: c.fen.clone(),
                                    key: c.key.clone(),
                                    root_id: r.id.clone(),
                                    group_id: String::new(),
                                    root_fen: r.fen.clone(),
                                    root_depth: r.depth,
                                    mv: c.mv.clone(),
                                },
                                rank: count,
                            });
                            got += 1;
                            count += 1;
                        }
                    }
                    counts.insert(format!("{}/{}", pname(p), cell_name(f.name(), b, class)), count);
                }
            }
        }
    }
    (out, counts)
}

/// Reporting/bootstrap groups: connected components (shared canonical children) among the CONTRIBUTING roots of
/// each partition, computed separately per partition. Reproducible from the stored contributing pool.
pub fn attach_groups(seed: &MasterSeed, metas: &mut [T1Meta], roots: &[RootRec]) -> HashMap<String, usize> {
    let used: HashSet<&str> = metas.iter().map(|m| m.ex.root_id.as_str()).collect();
    let mut part_of_root: HashMap<&str, Partition> = HashMap::new();
    for m in metas.iter() {
        part_of_root.insert(m.ex.root_id.as_str(), m.ex.partition);
    }
    let mut group_of: HashMap<String, String> = HashMap::new();
    let mut counts = HashMap::new();
    for p in [Partition::Fit, Partition::Val, Partition::Test] {
        let sub: Vec<RootRec> = roots.iter().filter(|r| used.contains(r.id.as_str()) && part_of_root.get(r.id.as_str()) == Some(&p)).cloned().collect();
        let groups = build_groups(seed, &sub);
        counts.insert(pname(p).to_string(), groups.len());
        for g in &groups {
            for &i in &g.roots {
                group_of.insert(sub[i].id.clone(), format!("{}-{}", pname(p), g.id));
            }
        }
    }
    for m in metas.iter_mut() {
        m.ex.group_id = group_of[&m.ex.root_id].clone();
    }
    counts
}

pub fn feasible(counts: &BTreeMap<String, usize>) -> bool {
    counts.iter().all(|(k, c)| {
        let q = if k.starts_with("train/") {
            TRAIN_PER_CLASS_CELL
        } else if k.starts_with("val/") {
            VAL_PER_CLASS_CELL
        } else {
            TEST_PER_CLASS_CELL
        };
        *c >= q
    }) && counts.len() == 36
}
