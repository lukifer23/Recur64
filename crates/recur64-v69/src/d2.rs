//! D2 (fitting-set scaling diagnostic): nested fitting subsets, example streams,
//! append-only receipts. Host-only. See docs/v69/D2_CONTRACT.md.

use crate::access::Access;
use crate::dataset::Example;
use crate::streams::{MasterSeed, keyed_u64};
use anyhow::{Result, bail, ensure};
use serde::Serialize;
use std::collections::{BTreeMap, HashSet};
use std::path::Path;

pub const D2_SIZES: [usize; 3] = [64, 256, 768];
pub const D2_UPDATES: usize = 2400;
pub const D2_BATCH: usize = 16;
pub const D2_SNAPSHOTS: [usize; 11] = [0, 50, 100, 200, 400, 600, 800, 1000, 1600, 2000, 2400];
pub const LABEL_SUBSET: &str = "d2/subset";
pub const LABEL_STRATA: &str = "d2/strata";
pub const LABEL_ORDER: &str = "d2_train_order";

const FAMS: [&str; 3] = ["KQQvK", "KQRvK", "KRRvK"];

#[derive(Serialize, Debug, Clone)]
pub struct SubsetSummary {
    pub size: usize,
    pub q_per_cell: usize,
    pub extra_pair_strata: Vec<String>,
    pub distinct_groups: usize,
    pub repeated_group_memberships: usize,
    pub distinct_roots: usize,
    pub positives: usize,
    pub negatives: usize,
    pub per_cell: BTreeMap<String, usize>,
    pub per_stratum_balanced: bool,
    pub contains_d1_panel: bool,
}

fn cell_key(e: &Example) -> String {
    format!("{}/n{}/{}", e.family, e.budget, if e.label { "pos" } else { "neg" })
}

/// Nested subsets 64 ⊂ 256 ⊂ 768 with panel ⊂ 64. For size N: q = floor(N/12) per
/// family x budget x class cell; the remaining N-12q examples are distributed as
/// positive/negative pairs to strata in a keyed order (label `d2/strata/<N>`). Candidates are
/// ordered by (previously-unused group first, keyed hash of the id); no model output is used.
pub fn select_nested(seed: &MasterSeed, fit_meta: &[Example], panel_ids: &HashSet<String>) -> Result<Vec<(usize, Vec<Example>, SubsetSummary)>> {
    ensure!(fit_meta.len() == 768, "expected the complete 768-row fitting partition");
    let by_id: BTreeMap<&str, &Example> = fit_meta.iter().map(|e| (e.id.as_str(), e)).collect();
    let mut selected: Vec<&Example> = Vec::new();
    for id in panel_ids {
        selected.push(by_id.get(id.as_str()).ok_or_else(|| anyhow::anyhow!("panel id {id} not in fitting partition"))?);
    }
    ensure!(selected.len() == 32, "D1 panel must have 32 examples");
    let mut used_groups: HashSet<String> = selected.iter().map(|e| e.group_id.clone()).collect();
    let mut out = Vec::new();
    for &n in &D2_SIZES {
        let q = n / 12;
        let rem = n - 12 * q;
        ensure!(rem % 2 == 0 && rem / 2 <= 6, "size {n}: remainder {rem} not distributable as pairs over 6 strata");
        let mut strata: Vec<(String, u8)> = FAMS.iter().flat_map(|f| [1u8, 2].map(|b| (f.to_string(), b))).collect();
        strata.sort_by_key(|(f, b)| keyed_u64(seed, &format!("{LABEL_STRATA}/{n}"), format!("{f}/{b}").as_bytes()));
        let extra: Vec<(String, u8)> = strata.into_iter().take(rem / 2).collect();
        if n < 768 {
            let have: HashSet<&str> = selected.iter().map(|e| e.id.as_str()).collect();
            let mut add: Vec<&Example> = Vec::new();
            for f in FAMS {
                for b in [1u8, 2] {
                    for label in [false, true] {
                        let target = q + extra.iter().any(|(ef, eb)| ef == f && *eb == b) as usize;
                        let cur = selected.iter().filter(|e| e.family == f && e.budget == b && e.label == label).count();
                        ensure!(cur <= target, "infeasible nesting: cell {f}/n{b}/{label} already has {cur} > quota {target} at size {n}");
                        let mut cand: Vec<&Example> = fit_meta.iter().filter(|e| e.family == f && e.budget == b && e.label == label && !have.contains(e.id.as_str()) && !add.iter().any(|a| a.id == e.id)).collect();
                        cand.sort_by_key(|e| (used_groups.contains(&e.group_id) as u8, keyed_u64(seed, LABEL_SUBSET, e.id.as_bytes())));
                        ensure!(cand.len() >= target - cur, "capacity: cell {f}/n{b}/{label} cannot supply quota at size {n}");
                        for e in cand.into_iter().take(target - cur) {
                            used_groups.insert(e.group_id.clone());
                            add.push(e);
                        }
                    }
                }
            }
            selected.extend(add);
        } else {
            selected = fit_meta.iter().collect();
        }
        ensure!(selected.len() == n, "size {n}: got {}", selected.len());
        let mut rows: Vec<Example> = selected.iter().map(|e| (*e).clone()).collect();
        rows.sort_by(|a, b| a.id.cmp(&b.id));
        out.push((n, rows));
    }
    // summaries + nesting verification
    let mut res = Vec::new();
    let mut prev: Option<HashSet<String>> = None;
    for (n, rows) in out {
        let ids: HashSet<String> = rows.iter().map(|e| e.id.clone()).collect();
        if let Some(p) = &prev {
            ensure!(p.is_subset(&ids), "nesting violated at size {n}");
        } else {
            ensure!(panel_ids.is_subset(&ids), "D1 panel not contained in the smallest subset");
        }
        let mut per_cell: BTreeMap<String, usize> = BTreeMap::new();
        for e in &rows {
            *per_cell.entry(cell_key(e)).or_default() += 1;
        }
        let mut balanced = true;
        for f in FAMS {
            for b in [1u8, 2] {
                let p = per_cell.get(&format!("{f}/n{b}/pos")).copied().unwrap_or(0);
                let ng = per_cell.get(&format!("{f}/n{b}/neg")).copied().unwrap_or(0);
                balanced &= p == ng;
            }
        }
        ensure!(balanced, "stratum imbalance at size {n}");
        let q = n / 12;
        for (cell, c) in &per_cell {
            ensure!(*c == q || *c == q + 1, "cell {cell} count {c} outside quota at size {n}");
        }
        let groups: HashSet<&str> = rows.iter().map(|e| e.group_id.as_str()).collect();
        let roots: HashSet<&str> = rows.iter().map(|e| e.root_id.as_str()).collect();
        let positives = rows.iter().filter(|e| e.label).count();
        let rem = n - 12 * q;
        let mut strata: Vec<(String, u8)> = FAMS.iter().flat_map(|f| [1u8, 2].map(|b| (f.to_string(), b))).collect();
        strata.sort_by_key(|(f, b)| keyed_u64(seed, &format!("{LABEL_STRATA}/{n}"), format!("{f}/{b}").as_bytes()));
        let extra_names: Vec<String> = strata.into_iter().take(rem / 2).map(|(f, b)| format!("{f}/n{b}")).collect();
        res.push((
            n,
            rows.clone(),
            SubsetSummary { size: n, q_per_cell: q, extra_pair_strata: extra_names, distinct_groups: groups.len(), repeated_group_memberships: n - groups.len(), distinct_roots: roots.len(), positives, negatives: n - positives, per_cell, per_stratum_balanced: balanced, contains_d1_panel: panel_ids.is_subset(&ids) },
        ));
        prev = Some(ids);
    }
    Ok(res)
}

/// Shuffled epoch stream for subset size `n` (index space: rows sorted by id):
/// epoch e = Fisher-Yates permutation driven by stream `d2_train_order/<n>` index e.
pub fn d2_order(seed: &MasterSeed, rows: usize, n: usize) -> Vec<usize> {
    let total = D2_UPDATES * D2_BATCH;
    let mut out = Vec::with_capacity(total);
    let mut e = 0u64;
    while out.len() < total {
        let mut p: Vec<usize> = (0..rows).collect();
        let mut rng = seed.stream(&format!("{LABEL_ORDER}/{n}"), e);
        for i in (1..rows).rev() {
            p.swap(i, rng.below(i as u64 + 1) as usize);
        }
        out.extend(p);
        e += 1;
    }
    out.truncate(total);
    out
}

/// Append-only write: refuses to overwrite an existing receipt.
pub fn write_new(access: &Access, rel: &str, bytes: &[u8]) -> Result<()> {
    let p = access.check_write(Path::new(rel))?;
    if p.exists() {
        bail!("APPEND-ONLY: receipt already exists: {rel}");
    }
    access.write(Path::new(rel), bytes)?;
    Ok(())
}
