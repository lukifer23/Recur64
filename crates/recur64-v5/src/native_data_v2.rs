//! Frozen heavy-family lineage. Generation and custody never evaluate a model.
use crate::native_data::{AUDIT, SELECT, audit_records, select};
use recur64_core::GameState;
use recur64_runtime::proof::generator::{
    Candidate, PoolReport, enumerate_pool_range, fen_from_canon, label_exact_pool,
};
use recur64_runtime::proof::targets::{MATE_DEFINITION, ProofPosition, ProofTargets, Split};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

pub const CONTRACT: &str = "v5_hp_data_v2";
pub const FAMILY: &str = "v5_hp_heavy_endgames_v2";
pub const SCHEMA: &str = "v5_hp_dataset_manifest_v2";
pub const RAW_SCHEMA: &str = "v5_hp_dataset_artifact_v2";
pub const FAMILIES: [&str; 3] = ["KQQvK", "KQRvK", "KRRvK"];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    Train,
    Dev,
    Confirm,
}
impl Role {
    pub fn identity(self) -> &'static str {
        match self {
            Self::Train => "V5_HP_TRAIN_V2",
            Self::Dev => "V5_HP_DEV_V2",
            Self::Confirm => "V5_HP_CONFIRM_V2",
        }
    }
    pub fn seed(self) -> u64 {
        match self {
            Self::Train => 0x7A50_2001,
            Self::Dev => 0x7A50_2002,
            Self::Confirm => 0x7A50_2003,
        }
    }
    pub fn split(self) -> Split {
        match self {
            Self::Train => Split::Train,
            Self::Dev => Split::Tune,
            Self::Confirm => Split::Confirm,
        }
    }
    pub fn quota(self) -> usize {
        if self == Self::Train { 3000 } else { 750 }
    }
    pub fn families(self) -> &'static [&'static str] {
        if self == Self::Train {
            &FAMILIES
        } else {
            &FAMILIES[1..]
        }
    }
    pub fn parse(s: &str) -> anyhow::Result<Self> {
        match s {
            "train" => Ok(Self::Train),
            "dev" => Ok(Self::Dev),
            "confirm" => Ok(Self::Confirm),
            _ => anyhow::bail!("unknown V2 role"),
        }
    }
}
pub fn digest<T: Serialize>(v: &T) -> anyhow::Result<String> {
    Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(v)?)))
}
pub fn set_digest<'a>(values: impl Iterator<Item = &'a str>) -> String {
    let mut h = Sha256::new();
    for v in values.collect::<BTreeSet<_>>() {
        h.update(v.as_bytes());
        h.update(b"\n");
    }
    format!("{:x}", h.finalize())
}
fn config() -> anyhow::Result<String> {
    crate::config::V5Config::default().scientific_digest()
}
fn read<T: for<'a> Deserialize<'a>>(path: &Path) -> anyhow::Result<T> {
    Ok(serde_json::from_slice(&std::fs::read(path)?)?)
}
fn write_new(path: &Path, value: &impl Serialize) -> anyhow::Result<()> {
    use std::io::Write;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    f.write_all(&serde_json::to_vec(value)?)?;
    f.sync_all()?;
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolShard {
    pub schema: String,
    pub source_sha: String,
    pub config_digest: String,
    pub data_contract: String,
    pub family: String,
    pub depth_bounds: [u8; 2],
    pub generation_seeds: [u64; 3],
    pub first_start: usize,
    pub first_end: usize,
    pub report: PoolReport,
    pub candidates: Vec<Candidate>,
    pub candidate_digest: String,
}
impl PoolShard {
    fn scientific_digest(&self) -> anyhow::Result<String> {
        let mut scientific = self.clone();
        scientific.candidate_digest.clear();
        scientific.report.wall_s = 0.0;
        digest(&scientific)
    }
    fn validate(&self, source: &str) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.schema == "v5_hp_pool_shard_v2"
                && self.source_sha == source
                && self.config_digest == config()?
                && self.data_contract == CONTRACT
                && self.depth_bounds == [1, 3]
                && self.generation_seeds
                    == [Role::Train.seed(), Role::Dev.seed(), Role::Confirm.seed()],
            "stale shard source/config/contract"
        );
        anyhow::ensure!(
            FAMILIES.contains(&self.family.as_str())
                && self.report.family == self.family
                && self.first_start.is_multiple_of(8)
                && self.first_end == self.first_start + 8
                && self.first_end <= 64,
            "invalid shard range/family"
        );
        anyhow::ensure!(
            self.candidate_digest == self.scientific_digest()?,
            "tampered shard"
        );
        let mut previous: Option<&str> = None;
        for c in &self.candidates {
            anyhow::ensure!(
                c.canon.len() == 64 && c.canon.is_ascii(),
                "invalid canonical encoding"
            );
            anyhow::ensure!(
                (1..=3).contains(&c.depth)
                    && previous.is_none_or(|p| p < c.canon.as_str())
                    && c.fen == fen_from_canon(&c.canon),
                "invalid candidate order/identity"
            );
            GameState::from_fen(&c.fen)?;
            previous = Some(&c.canon);
        }
        Ok(())
    }
}
pub fn pool_shard(
    source: &str,
    family: &str,
    start: usize,
    threads: usize,
    path: &Path,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        start.is_multiple_of(8) && start < 64 && (1..=16).contains(&threads),
        "invalid bounded shard request"
    );
    if path.exists() {
        let old: anyhow::Result<PoolShard> = read(path);
        if let Ok(old) = old
            && old.validate(source).is_ok()
            && old.family == family
            && old.first_start == start
        {
            return Ok(());
        }
        let quarantine = path.with_extension("invalid-quarantine.json");
        anyhow::ensure!(
            !quarantine.exists(),
            "existing quarantine; refusing overwrite"
        );
        std::fs::rename(path, &quarantine)?;
        anyhow::bail!("invalid shard quarantined; rerun to regenerate");
    }
    let partial = path.with_extension("partial.json");
    if partial.exists() {
        let quarantine = path.with_extension("interrupted-quarantine.json");
        anyhow::ensure!(!quarantine.exists(), "existing interrupted quarantine");
        std::fs::rename(&partial, &quarantine)?;
    }
    let fi = recur64_runtime::proof::targets::FAMILIES
        .iter()
        .position(|f| f.0 == family)
        .ok_or_else(|| anyhow::anyhow!("unknown family"))?;
    anyhow::ensure!(
        FAMILIES.contains(&family),
        "light-family generation prohibited"
    );
    let (report, candidates) = enumerate_pool_range(fi, 3, threads, start, start + 8)?;
    let mut shard = PoolShard {
        schema: "v5_hp_pool_shard_v2".into(),
        source_sha: source.into(),
        config_digest: config()?,
        data_contract: CONTRACT.into(),
        family: family.into(),
        depth_bounds: [1, 3],
        generation_seeds: [Role::Train.seed(), Role::Dev.seed(), Role::Confirm.seed()],
        first_start: start,
        first_end: start + 8,
        candidate_digest: String::new(),
        candidates,
        report,
    };
    shard.candidate_digest = shard.scientific_digest()?;
    shard.validate(source)?;
    write_new(&partial, &shard)?;
    std::fs::rename(partial, path)?;
    Ok(())
}
pub fn merge_shards(
    shards: &[PoolShard],
    source: &str,
    family: &str,
) -> anyhow::Result<Vec<Candidate>> {
    anyhow::ensure!(
        shards.len() == 8,
        "complete eight-shard family coverage required"
    );
    let mut ranges = BTreeSet::new();
    let mut merged: BTreeMap<String, Candidate> = BTreeMap::new();
    for s in shards {
        s.validate(source)?;
        anyhow::ensure!(
            s.family == family && ranges.insert(s.first_start),
            "duplicate/wrong family shard"
        );
        for c in &s.candidates {
            if let Some(prior) = merged.get(&c.canon) {
                anyhow::ensure!(prior == c, "inconsistent overlapping canonical label");
            } else {
                merged.insert(c.canon.clone(), c.clone());
            }
        }
    }
    anyhow::ensure!(
        ranges == (0..64).step_by(8).collect(),
        "incomplete range coverage"
    );
    Ok(merged.into_values().collect())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Exclusion {
    pub identity: String,
    pub content_digest: String,
    pub fen_set_digest: String,
    pub canonical_set_digest: String,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Overlap {
    pub left: String,
    pub right: String,
    pub exact_fen: usize,
    pub canonical: usize,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    pub schema: String,
    pub identity: String,
    pub role: Role,
    pub source_sha: String,
    pub config_digest: String,
    pub dataset_family: String,
    pub data_contract: String,
    pub selection_contract: String,
    pub audit_contract: String,
    pub generation_seed: u64,
    pub cell_counts: BTreeMap<String, usize>,
    pub records: usize,
    pub content_digest: String,
    pub target_digest: String,
    pub fen_set_digest: String,
    pub canonical_set_digest: String,
    pub exclusions: Vec<Exclusion>,
    pub overlaps: Vec<Overlap>,
    pub audit_count: usize,
    pub audit_failures: usize,
    pub generated: bool,
    pub audited: bool,
    pub sealed: bool,
    pub evaluated: bool,
    pub historical_cross_disjointness: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    pub schema: String,
    pub manifest: Manifest,
    pub targets: ProofTargets,
}
pub fn cell_counts(positions: &[ProofPosition]) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for p in positions {
        *counts
            .entry(format!("{} M{}", p.family, p.mate_depth))
            .or_insert(0) += 1;
    }
    counts
}
fn expected_cells(role: Role) -> BTreeMap<String, usize> {
    role.families()
        .iter()
        .flat_map(|f| (1..=3).map(move |d| (format!("{f} M{d}"), role.quota())))
        .collect()
}
fn validate_quotas(
    role: Role,
    counts: &BTreeMap<String, usize>,
    total: usize,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        *counts == expected_cells(role) && total == role.families().len() * 3 * role.quota(),
        "wrong total/cell quota"
    );
    Ok(())
}
pub fn overlap(a: &Dataset, b: &Dataset) -> Overlap {
    let af: BTreeSet<_> = a.targets.positions.iter().map(|p| &p.fen).collect();
    let ac: BTreeSet<_> = a.targets.positions.iter().map(|p| &p.canon).collect();
    Overlap {
        left: a.manifest.identity.clone(),
        right: b.manifest.identity.clone(),
        exact_fen: b
            .targets
            .positions
            .iter()
            .filter(|p| af.contains(&p.fen))
            .count(),
        canonical: b
            .targets
            .positions
            .iter()
            .filter(|p| ac.contains(&p.canon))
            .count(),
    }
}
impl Dataset {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let d: Self = read(path)?;
        d.validate()?;
        Ok(d)
    }
    pub fn validate(&self) -> anyhow::Result<()> {
        let m = &self.manifest;
        let role = m.role;
        anyhow::ensure!(
            self.schema == RAW_SCHEMA
                && m.schema == SCHEMA
                && m.identity == role.identity()
                && m.dataset_family == FAMILY
                && m.data_contract == CONTRACT
                && m.selection_contract == SELECT
                && m.audit_contract == AUDIT
                && m.config_digest == config()?
                && m.source_sha.len() == 40
                && m.source_sha.bytes().all(|c| c.is_ascii_hexdigit()),
            "dataset contract/source/config mismatch"
        );
        self.targets.validate()?;
        anyhow::ensure!(
            self.targets.mate_definition == MATE_DEFINITION
                && self.targets.split == role.split()
                && self.targets.seed == role.seed()
                && m.generation_seed == role.seed(),
            "target role/history/mate contract mismatch"
        );
        anyhow::ensure!(
            m.generated
                && m.audited
                && m.audit_failures == 0
                && m.audit_count == m.records
                && !m.evaluated
                && m.sealed == (role == Role::Confirm),
            "incomplete audit/seal or evaluated artifact"
        );
        validate_quotas(role, &m.cell_counts, m.records)?;
        anyhow::ensure!(
            cell_counts(&self.targets.positions) == m.cell_counts
                && self.targets.positions.len() == m.records,
            "records disagree with measured quotas"
        );
        anyhow::ensure!(
            digest(&self.targets.positions)? == m.content_digest
                && self.targets.digest == m.target_digest
                && set_digest(self.targets.positions.iter().map(|p| p.fen.as_str()))
                    == m.fen_set_digest
                && set_digest(self.targets.positions.iter().map(|p| p.canon.as_str()))
                    == m.canonical_set_digest,
            "scientific content/set digest mismatch"
        );
        let mut fens = BTreeSet::new();
        let mut canons = BTreeSet::new();
        let mut ids = BTreeSet::new();
        let mut prior = None;
        for p in &self.targets.positions {
            GameState::from_fen(&p.fen)?;
            anyhow::ensure!(
                p.canon.len() == 64 && p.canon.is_ascii(),
                "invalid canonical encoding"
            );
            let order = (&p.family, p.mate_depth, &p.canon);
            anyhow::ensure!(
                prior.is_none_or(|o| o < order)
                    && p.fen == fen_from_canon(&p.canon)
                    && p.generator_seed == role.seed()
                    && fens.insert(&p.fen)
                    && canons.insert(&p.canon)
                    && ids.insert(&p.id),
                "duplicate/order/fresh-root identity failure"
            );
            prior = Some(order);
        }
        let expected_exclusions = match role {
            Role::Train => vec![],
            Role::Dev => vec![Role::Train.identity()],
            Role::Confirm => vec![Role::Train.identity(), Role::Dev.identity()],
        };
        anyhow::ensure!(
            m.exclusions
                .iter()
                .map(|e| e.identity.as_str())
                .collect::<Vec<_>>()
                == expected_exclusions
                && m.overlaps.len() == m.exclusions.len()
                && m.overlaps
                    .iter()
                    .all(|o| o.exact_fen == 0 && o.canonical == 0),
            "exclusion/disjointness contract failure"
        );
        Ok(())
    }
    pub fn custody(&self, expected: &Manifest, role: Role) -> anyhow::Result<()> {
        self.validate()?;
        anyhow::ensure!(
            &self.manifest == expected && self.manifest.role == role,
            "custody role/committed manifest mismatch"
        );
        Ok(())
    }
    pub fn scientific_access(&self, role: Role) -> anyhow::Result<()> {
        anyhow::ensure!(
            role != Role::Confirm && self.manifest.role == role,
            "CONFIRM inaccessible; TRAIN/DEV role mismatch"
        );
        self.validate()
    }
}
fn audit_parallel(records: &[ProofPosition], threads: usize) -> anyhow::Result<usize> {
    std::thread::scope(|s| {
        let workers: Vec<_> = records
            .chunks(records.len().div_ceil(threads))
            .map(|part| s.spawn(move || audit_records(part)))
            .collect();
        workers.into_iter().try_fold(0, |sum, h| {
            Ok(sum
                + h.join()
                    .map_err(|_| anyhow::anyhow!("audit worker panic"))??)
        })
    })
}
#[derive(Serialize, Deserialize)]
struct CellCheckpoint {
    schema: String,
    source_sha: String,
    config_digest: String,
    data_contract: String,
    role: Role,
    family: String,
    depth: u8,
    chosen_digest: String,
    exclusion_digest: String,
    record_digest: String,
    records: Vec<ProofPosition>,
}
pub fn generate(
    source: &str,
    role: Role,
    pools: &Path,
    exclusions: &[PathBuf],
    output: &Path,
    threads: usize,
) -> anyhow::Result<Dataset> {
    anyhow::ensure!(
        !output.exists() && (1..=16).contains(&threads),
        "existing output/invalid threads"
    );
    let excluded: Vec<_> = exclusions
        .iter()
        .map(|p| Dataset::load(p))
        .collect::<anyhow::Result<_>>()?;
    let expected = match role {
        Role::Train => vec![],
        Role::Dev => vec![Role::Train],
        Role::Confirm => vec![Role::Train, Role::Dev],
    };
    anyhow::ensure!(
        excluded.iter().map(|d| d.manifest.role).collect::<Vec<_>>() == expected
            && excluded.iter().all(|d| d.manifest.source_sha == source),
        "wrong exclusion order/source"
    );
    let ef = excluded
        .iter()
        .flat_map(|d| d.targets.positions.iter().map(|p| p.fen.clone()))
        .collect();
    let ec = excluded
        .iter()
        .flat_map(|d| d.targets.positions.iter().map(|p| p.canon.clone()))
        .collect();
    let mut records = Vec::new();
    for family in role.families() {
        let shards: Vec<PoolShard> = (0..64)
            .step_by(8)
            .map(|start| read(&pools.join(format!("{family}-{start:02}.json"))))
            .collect::<anyhow::Result<_>>()?;
        let pool = merge_shards(&shards, source, family)?;
        for depth in 1..=3 {
            let cell: Vec<_> = pool.iter().filter(|c| c.depth == depth).cloned().collect();
            let mut chosen = select(&cell, role.seed(), role.quota(), &ef, &ec)?;
            chosen.sort_by(|a, b| a.canon.cmp(&b.canon));
            let checkpoint = output
                .with_extension("cells")
                .join(format!("{family}-m{depth}.json"));
            let chosen_digest = digest(&chosen)?;
            let exclusion_digest =
                digest(&excluded.iter().map(|d| &d.manifest).collect::<Vec<_>>())?;
            let mut labelled = if checkpoint.exists() {
                let saved: CellCheckpoint = read(&checkpoint)?;
                anyhow::ensure!(
                    saved.schema == "v5_hp_cell_checkpoint_v2"
                        && saved.source_sha == source
                        && saved.config_digest == config()?
                        && saved.data_contract == CONTRACT
                        && saved.role == role
                        && saved.family == *family
                        && saved.depth == depth
                        && saved.chosen_digest == chosen_digest
                        && saved.exclusion_digest == exclusion_digest
                        && saved.record_digest == digest(&saved.records)?
                        && saved.records.len() == role.quota()
                        && saved
                            .records
                            .iter()
                            .zip(&chosen)
                            .all(|(p, c)| p.fen == c.fen
                                && p.canon == c.canon
                                && p.mate_depth == depth
                                && p.family == *family
                                && p.split == role.split()
                                && p.generator_seed == role.seed()),
                    "invalid checkpoint; preserved for quarantine/review"
                );
                saved.records
            } else {
                label_exact_pool(&chosen, family, role.split(), role.seed())?
            };
            for p in &mut labelled {
                p.id = format!("{}-{}-m{}-{}", role.identity(), family, depth, p.canon);
            }
            let audited = audit_parallel(&labelled, threads)?;
            anyhow::ensure!(audited == role.quota(), "incomplete audit");
            if !checkpoint.exists() {
                let saved = CellCheckpoint {
                    schema: "v5_hp_cell_checkpoint_v2".into(),
                    source_sha: source.into(),
                    config_digest: config()?,
                    data_contract: CONTRACT.into(),
                    role,
                    family: (*family).into(),
                    depth,
                    chosen_digest,
                    exclusion_digest,
                    record_digest: digest(&labelled)?,
                    records: labelled.clone(),
                };
                write_new(&checkpoint, &saved)?;
            }
            println!(
                "{} {family} M{depth}: selected {}, independently audited {}",
                role.identity(),
                labelled.len(),
                audited
            );
            records.extend(labelled);
        }
    }
    records.sort_by(|a, b| {
        (&a.family, a.mate_depth, &a.canon).cmp(&(&b.family, b.mate_depth, &b.canon))
    });
    let targets = ProofTargets::new(
        role.split(),
        role.seed(),
        serde_json::json!({"dataset_family":FAMILY,"data_contract":CONTRACT,"selection_contract":SELECT,"audit_contract":AUDIT,"max_correct_fraction":0.15,"depth_ge_2_fact_ambiguity":true}),
        records,
    );
    let manifest = Manifest {
        schema: SCHEMA.into(),
        identity: role.identity().into(),
        role,
        source_sha: source.into(),
        config_digest: config()?,
        dataset_family: FAMILY.into(),
        data_contract: CONTRACT.into(),
        selection_contract: SELECT.into(),
        audit_contract: AUDIT.into(),
        generation_seed: role.seed(),
        cell_counts: cell_counts(&targets.positions),
        records: targets.positions.len(),
        content_digest: digest(&targets.positions)?,
        target_digest: targets.digest.clone(),
        fen_set_digest: set_digest(targets.positions.iter().map(|p| p.fen.as_str())),
        canonical_set_digest: set_digest(targets.positions.iter().map(|p| p.canon.as_str())),
        exclusions: excluded
            .iter()
            .map(|d| Exclusion {
                identity: d.manifest.identity.clone(),
                content_digest: d.manifest.content_digest.clone(),
                fen_set_digest: d.manifest.fen_set_digest.clone(),
                canonical_set_digest: d.manifest.canonical_set_digest.clone(),
            })
            .collect(),
        overlaps: vec![],
        audit_count: targets.positions.len(),
        audit_failures: 0,
        generated: true,
        audited: true,
        sealed: role == Role::Confirm,
        evaluated: false,
        historical_cross_disjointness:
            "Cross-disjointness from unavailable workstation-only raw datasets was NOT verified."
                .into(),
    };
    let mut data = Dataset {
        schema: RAW_SCHEMA.into(),
        manifest,
        targets,
    };
    data.manifest.overlaps = excluded.iter().map(|d| overlap(d, &data)).collect();
    data.validate()?;
    write_new(output, &data)?;
    write_new(&output.with_extension("manifest.json"), &data.manifest)?;
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn frozen_roles_counts_seeds() {
        assert_eq!(expected_cells(Role::Train).values().sum::<usize>(), 27000);
        assert_eq!(expected_cells(Role::Dev).values().sum::<usize>(), 4500);
        assert_eq!(expected_cells(Role::Confirm).values().sum::<usize>(), 4500);
        assert_ne!(Role::Train.seed(), Role::Dev.seed());
        assert_ne!(Role::Dev.seed(), Role::Confirm.seed());
        assert_eq!(Role::Train.families().len(), 9 / 3);
    }
    #[test]
    fn set_and_content_hashes_detect_mutation() {
        assert_eq!(
            set_digest(["b", "a"].into_iter()),
            set_digest(["a", "b"].into_iter())
        );
        assert_ne!(set_digest(["a"].into_iter()), set_digest(["b"].into_iter()));
        assert_ne!(digest(&[1, 2]).unwrap(), digest(&[1, 3]).unwrap());
    }
    #[test]
    fn incomplete_shards_refused() {
        assert!(merge_shards(&[], "source", "KQRvK").is_err());
    }
    fn shards() -> Vec<PoolShard> {
        let mut candidates: Vec<_> = [
            "7k/5Q2/5K2/6R1/8/8/8/8 w - - 0 1",
            "7k/5Q2/5K2/1R6/8/8/8/8 w - - 0 1",
        ]
        .into_iter()
        .map(|fen| {
            let canon = recur64_runtime::proof::generator::canonical_key(fen);
            Candidate {
                fen: fen_from_canon(&canon),
                canon,
                depth: 1,
            }
        })
        .collect();
        candidates.sort_by(|a, b| a.canon.cmp(&b.canon));
        (0..64)
            .step_by(8)
            .map(|start| {
                let mut s = PoolShard {
                    schema: "v5_hp_pool_shard_v2".into(),
                    source_sha: "a".repeat(40),
                    config_digest: config().unwrap(),
                    data_contract: CONTRACT.into(),
                    family: "KQRvK".into(),
                    depth_bounds: [1, 3],
                    generation_seeds: [Role::Train.seed(), Role::Dev.seed(), Role::Confirm.seed()],
                    first_start: start,
                    first_end: start + 8,
                    report: PoolReport {
                        family: "KQRvK".into(),
                        ..Default::default()
                    },
                    candidates: candidates.clone(),
                    candidate_digest: String::new(),
                };
                s.candidate_digest = s.scientific_digest().unwrap();
                s
            })
            .collect()
    }
    #[test]
    fn shard_merge_completion_order_and_resume_identity() {
        let a = shards();
        let mut b = a.clone();
        b.reverse();
        assert_eq!(
            merge_shards(&a, &"a".repeat(40), "KQRvK").unwrap(),
            merge_shards(&b, &"a".repeat(40), "KQRvK").unwrap()
        );
        assert_eq!(merge_shards(&a, &"a".repeat(40), "KQRvK").unwrap().len(), 2);
        let restored: PoolShard =
            serde_json::from_slice(&serde_json::to_vec(&a[0]).unwrap()).unwrap();
        assert_eq!(restored.scientific_digest().unwrap(), a[0].candidate_digest);
        let mut duplicate = a.clone();
        duplicate[1] = duplicate[0].clone();
        assert!(merge_shards(&duplicate, &"a".repeat(40), "KQRvK").is_err());
        assert!(merge_shards(&a, &"b".repeat(40), "KQRvK").is_err());
        let mut bad = a.clone();
        bad[0].config_digest = "stale".into();
        assert!(merge_shards(&bad, &"a".repeat(40), "KQRvK").is_err());
        let mut bad = a.clone();
        bad[0].report.raw_placements = 1;
        assert!(merge_shards(&bad, &"a".repeat(40), "KQRvK").is_err());
    }
    #[test]
    fn independent_audit_rejects_legal_action_and_index_tampering() {
        let fen = "7k/5Q2/5K2/6R1/8/8/8/8 w - - 0 1";
        let canon = recur64_runtime::proof::generator::canonical_key(fen);
        let c = Candidate {
            fen: fen_from_canon(&canon),
            canon,
            depth: 1,
        };
        let good = label_exact_pool(&[c], "KQRvK", Split::Train, Role::Train.seed()).unwrap();
        audit_records(&good).unwrap();
        let mut bad = good.clone();
        bad[0].legal.reverse();
        assert!(audit_records(&bad).is_err());
        let mut bad = good.clone();
        bad[0].correct = vec![u32::MAX];
        assert!(audit_records(&bad).is_err());
        let mut bad = good.clone();
        bad[0].fen = bad[0].fen.replace("0 1", "1 1");
        assert!(audit_records(&bad).is_err());
        let mut bad = good.clone();
        bad[0].family = "KRRvK".into();
        assert!(audit_records(&bad).is_err());
        let mut bad = good.clone();
        bad[0].mate_depth = 2;
        assert!(audit_records(&bad).is_err());
        let mut bad = good.clone();
        bad.push(bad[0].clone());
        bad[1].id = "other".into();
        assert!(audit_records(&bad).is_err());
    }
    #[test]
    fn strict_total_and_cell_quota_enforcement() {
        let counts = expected_cells(Role::Train);
        validate_quotas(Role::Train, &counts, 27000).unwrap();
        assert!(validate_quotas(Role::Train, &counts, 26999).is_err());
        let mut bad = counts;
        bad.insert("KQQvK M1".into(), 2999);
        bad.insert("KQQvK M2".into(), 3001);
        assert!(validate_quotas(Role::Train, &bad, 27000).is_err());
        assert!(validate_quotas(Role::Dev, &bad, 27000).is_err());
    }
}
