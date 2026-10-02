//! Data custody for V3: the exclusion inventory, the HOLDOUT_C seal, and the
//! guards that keep sealed data out of working paths.
//!
//! Verifying HOLDOUT_C's integrity (schema, content digest) and reading its
//! canonical classes for exclusion is custody, not evaluation. It is NOT recorded as
//! an exposure. Anything that would evaluate it must go through
//! [`load_sealed_confirmation`], which needs an explicit [`ConfirmAuthorization`]
//! and does record an exposure.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::generator::exclusion_digest;
use super::targets::{ProofTargets, Split};

/// Frozen content digest of HOLDOUT_C (V2.5 record; never evaluated).
pub const HOLDOUT_C_DIGEST: &str =
    "4ab951c6edd8dd4f531bb87d2f4373895d1fddb70efdf052b24c09a1b71d87d5";

/// Frozen `v3_tune_v1` specification (docs/V3_P4_PLAN.md).
pub const V3_TUNE_IDENTITY: &str = "v3_tune_v1";
pub const V3_TUNE_SEED: u64 = 0x7A13_0004; // 2048065540
pub const V3_TUNE_PER_CELL: usize = 750;

pub const SEAL_SCHEMA: &str = "v3_confirm_seal_v1";

/// One dataset in the exclusion inventory.
#[derive(Debug, Clone, Serialize)]
pub struct DatasetRecord {
    pub path: String,
    pub split: String,
    pub positions: usize,
    pub digest: String,
    /// Distinct symmetry-canonical classes the dataset contributes.
    pub canon_classes: usize,
}

/// Every dataset whose canonical classes a new set must avoid.
#[derive(Debug, Clone, Serialize)]
pub struct Inventory {
    pub datasets: Vec<DatasetRecord>,
    #[serde(skip)]
    pub canons: HashSet<String>,
    #[serde(skip)]
    pub fens: HashSet<String>,
    pub excluded_canonical_classes: usize,
    pub excluded_exact_fens: usize,
    pub exclusion_manifest_digest: String,
}

/// Load every `proof-*.json` under each directory (validating schema, contracts and
/// content digest) and collect their canonical classes and exact FENs.
pub fn inventory(dirs: &[PathBuf]) -> anyhow::Result<Inventory> {
    let mut files: Vec<PathBuf> = Vec::new();
    for d in dirs {
        let rd = std::fs::read_dir(d)
            .map_err(|e| anyhow::anyhow!("inventory directory {}: {e}", d.display()))?;
        for e in rd {
            let p = e?.path();
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name.starts_with("proof-") && name.ends_with(".json") && !name.contains("report") {
                files.push(p);
            }
        }
    }
    files.sort();
    anyhow::ensure!(
        !files.is_empty(),
        "no proof-*.json datasets found to exclude"
    );
    let mut datasets = Vec::new();
    let mut canons = HashSet::new();
    let mut fens = HashSet::new();
    let mut digests = Vec::new();
    for p in &files {
        let t = ProofTargets::load(p)?;
        let own: HashSet<&str> = t.positions.iter().map(|x| x.canon.as_str()).collect();
        for x in &t.positions {
            canons.insert(x.canon.clone());
            fens.insert(x.fen.clone());
        }
        digests.push(t.digest.clone());
        datasets.push(DatasetRecord {
            path: p.display().to_string().replace('\\', "/"),
            split: t.split.label().to_string(),
            positions: t.positions.len(),
            digest: t.digest.clone(),
            canon_classes: own.len(),
        });
    }
    // Identical datasets stored in two places count once in the identity.
    digests.sort();
    digests.dedup();
    let manifest_digest = exclusion_digest(&canons, &digests);
    Ok(Inventory {
        excluded_canonical_classes: canons.len(),
        excluded_exact_fens: fens.len(),
        datasets,
        canons,
        fens,
        exclusion_manifest_digest: manifest_digest,
    })
}

/// The recorded state of the sealed confirmation set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Seal {
    pub schema: String,
    pub split: String,
    pub file: String,
    pub expected_digest: String,
    pub actual_digest: String,
    pub positions: usize,
    pub verified: bool,
    pub sealed: bool,
    /// Always false at P4: no model has seen this set.
    pub evaluated: bool,
    pub permitted_p4_use: String,
    pub future_access: String,
}

/// Verify HOLDOUT_C's custody: schema, contracts, content digest equal to the
/// frozen value. Performs no analysis of its content. A different digest is a
/// hard stop: the set is never repaired or regenerated under this experiment.
pub fn verify_holdout_c(path: &Path) -> anyhow::Result<Seal> {
    let t = ProofTargets::load(path)?;
    anyhow::ensure!(
        t.split == Split::HoldoutC,
        "{}: split is not holdout_c",
        path.display()
    );
    anyhow::ensure!(
        t.digest == HOLDOUT_C_DIGEST,
        "STOP: HOLDOUT_C digest {} differs from the frozen {HOLDOUT_C_DIGEST}; it is not repaired \
         or regenerated under this experiment",
        t.digest
    );
    Ok(Seal {
        schema: SEAL_SCHEMA.into(),
        split: "holdout_c".into(),
        file: path.display().to_string().replace('\\', "/"),
        expected_digest: HOLDOUT_C_DIGEST.into(),
        actual_digest: t.digest,
        positions: t.positions.len(),
        verified: true,
        sealed: true,
        evaluated: false,
        permitted_p4_use: "custody and disjointness only: integrity check, canonical-class and exact-FEN exclusion"
            .into(),
        future_access: "only through proof::custody::load_sealed_confirmation with a ConfirmAuthorization for phase V3-P8"
            .into(),
    })
}

/// Load a dataset for ordinary V3 work (tracing, tuning, training). Refuses every
/// holdout and confirmation split and the frozen HOLDOUT_C digest, so a sealed set
/// cannot be substituted for TUNE by accident.
pub fn load_working_split(path: &Path, allowed: &[Split]) -> anyhow::Result<ProofTargets> {
    let t = ProofTargets::load(path)?;
    anyhow::ensure!(
        t.digest != HOLDOUT_C_DIGEST && t.split != Split::HoldoutC,
        "{}: HOLDOUT_C is sealed and cannot be loaded through a working path; the confirmation \
         path requires an explicit ConfirmAuthorization",
        path.display()
    );
    anyhow::ensure!(
        allowed.contains(&t.split),
        "{}: split {} is not allowed here (allowed: {:?})",
        path.display(),
        t.split.label(),
        allowed.iter().map(|s| s.label()).collect::<Vec<_>>()
    );
    Ok(t)
}

/// Proof that the owner authorized a confirmation evaluation. It can only be
/// constructed for phase `V3-P8` with a non-empty reference to the approval, and its
/// use is logged.
#[derive(Debug)]
pub struct ConfirmAuthorization {
    phase: String,
    approval_reference: String,
}

impl ConfirmAuthorization {
    pub fn request(phase: &str, approval_reference: &str) -> anyhow::Result<Self> {
        anyhow::ensure!(
            phase == "V3-P8",
            "confirmation access is only authorized for phase V3-P8, not '{phase}'"
        );
        anyhow::ensure!(
            !approval_reference.trim().is_empty(),
            "confirmation access needs a recorded owner approval reference"
        );
        Ok(Self {
            phase: phase.to_string(),
            approval_reference: approval_reference.to_string(),
        })
    }
}

/// The only way to obtain the sealed confirmation set for evaluation. Checks the
/// seal (verified, sealed, not yet evaluated), re-verifies the digest, and appends an
/// exposure line to `exposure_log` BEFORE returning the data.
pub fn load_sealed_confirmation(
    path: &Path,
    seal: &Path,
    auth: &ConfirmAuthorization,
    exposure_log: &Path,
) -> anyhow::Result<ProofTargets> {
    let s: Seal = serde_json::from_slice(&std::fs::read(seal)?)?;
    anyhow::ensure!(
        s.schema == SEAL_SCHEMA && s.verified && s.sealed && s.expected_digest == HOLDOUT_C_DIGEST,
        "the confirmation seal is not valid"
    );
    anyhow::ensure!(
        !s.evaluated,
        "the seal records that this set was already evaluated; a second confirmation is refused"
    );
    let t = ProofTargets::load(path)?;
    anyhow::ensure!(
        t.split == Split::HoldoutC && t.digest == HOLDOUT_C_DIGEST,
        "{}: not the sealed HOLDOUT_C",
        path.display()
    );
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(exposure_log)?;
    writeln!(
        f,
        "V3 CONFIRM EXPOSURE phase={} approval={} digest={}",
        auth.phase, auth.approval_reference, t.digest
    )?;
    Ok(t)
}

/// Per (family, depth) counts of a dataset (metadata only).
pub fn cell_counts(t: &ProofTargets) -> BTreeMap<String, usize> {
    let mut m = BTreeMap::new();
    for p in &t.positions {
        *m.entry(format!("{} M{}", p.family, p.mate_depth))
            .or_insert(0) += 1;
    }
    m
}

// ---------------------------------------------------------------------------
// V4: `v4_tune_v1` custody. The rule, seed and size were committed in
// docs/V4_RESEARCH_PLAN.md before the set was generated.
// ---------------------------------------------------------------------------

/// Frozen `v4_tune_v1` specification.
pub const V4_TUNE_IDENTITY: &str = "v4_tune_v1";
pub const V4_TUNE_SEED: u64 = 0x7A40_0001;
pub const V4_TUNE_PER_CELL: usize = 1000;
/// 2 families x M1..M3 x `V4_TUNE_PER_CELL`.
pub const V4_TUNE_POSITIONS: usize = 6 * V4_TUNE_PER_CELL;
pub const V4_TUNE_SEAL_SCHEMA: &str = "v4_tune_seal_v1";
/// The only phase that may open the sealed V4 set.
pub const V4_TUNE_AUTH_PHASE: &str = "V4-FINAL";

/// The recorded state of the sealed V4 primary evaluation set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct V4TuneSeal {
    pub schema: String,
    pub identity: String,
    pub file: String,
    pub digest: String,
    pub positions: usize,
    pub verified: bool,
    pub sealed: bool,
    /// Always false during P0/P1: no checkpoint has been evaluated on this set.
    pub evaluated: bool,
    pub permitted_p0p1_use: String,
    pub future_access: String,
}

/// Verify a stored `v4_tune_v1` and produce its seal: schema, contracts and content digest are
/// validated by `ProofTargets::load`; identity, size and cell structure are checked here.
pub fn seal_v4_tune(path: &Path) -> anyhow::Result<V4TuneSeal> {
    let t = ProofTargets::load(path)?;
    anyhow::ensure!(
        t.split == Split::Tune
            && t.filters.get("identity").and_then(|v| v.as_str()) == Some(V4_TUNE_IDENTITY),
        "{}: not a {V4_TUNE_IDENTITY} dataset",
        path.display()
    );
    anyhow::ensure!(
        t.digest != HOLDOUT_C_DIGEST,
        "{}: carries the HOLDOUT_C digest",
        path.display()
    );
    anyhow::ensure!(
        t.positions.len() == V4_TUNE_POSITIONS,
        "{V4_TUNE_IDENTITY} has {} positions, expected {V4_TUNE_POSITIONS}",
        t.positions.len()
    );
    for (cell, n) in cell_counts(&t) {
        anyhow::ensure!(
            n == V4_TUNE_PER_CELL,
            "{V4_TUNE_IDENTITY} cell {cell} has {n} positions, expected {V4_TUNE_PER_CELL}"
        );
    }
    Ok(V4TuneSeal {
        schema: V4_TUNE_SEAL_SCHEMA.into(),
        identity: V4_TUNE_IDENTITY.into(),
        file: path.display().to_string().replace('\\', "/"),
        digest: t.digest,
        positions: t.positions.len(),
        verified: true,
        sealed: true,
        evaluated: false,
        permitted_p0p1_use: "custody, generation determinism and disjointness only. No developmental \
             checkpoint, mechanism study or hyperparameter choice may evaluate against it"
            .into(),
        future_access: "only through proof::custody::load_sealed_v4_tune with a V4TuneAuthorization for phase V4-FINAL"
            .into(),
    })
}

/// Proof that the owner authorized the V4 final evaluation. Constructible only for phase
/// `V4-FINAL` with a non-empty reference to the approval; its use is logged.
#[derive(Debug)]
pub struct V4TuneAuthorization {
    phase: String,
    approval_reference: String,
}

impl V4TuneAuthorization {
    pub fn request(phase: &str, approval_reference: &str) -> anyhow::Result<Self> {
        anyhow::ensure!(
            phase == V4_TUNE_AUTH_PHASE,
            "v4_tune_v1 access is only authorized for phase {V4_TUNE_AUTH_PHASE}, not '{phase}'"
        );
        anyhow::ensure!(
            !approval_reference.trim().is_empty(),
            "v4_tune_v1 access needs a recorded owner approval reference"
        );
        Ok(Self {
            phase: phase.to_string(),
            approval_reference: approval_reference.to_string(),
        })
    }
}

/// The only way to obtain `v4_tune_v1`. Checks the seal (valid, sealed, not evaluated),
/// re-verifies the digest, and appends an exposure line to `exposure_log` BEFORE returning data.
pub fn load_sealed_v4_tune(
    path: &Path,
    seal: &Path,
    auth: &V4TuneAuthorization,
    exposure_log: &Path,
) -> anyhow::Result<ProofTargets> {
    let s: V4TuneSeal = serde_json::from_slice(&std::fs::read(seal)?)?;
    anyhow::ensure!(
        s.schema == V4_TUNE_SEAL_SCHEMA
            && s.identity == V4_TUNE_IDENTITY
            && s.verified
            && s.sealed,
        "the v4_tune_v1 seal is not valid"
    );
    anyhow::ensure!(
        !s.evaluated,
        "the seal records that this set was already evaluated; a second evaluation is refused"
    );
    let t = ProofTargets::load(path)?;
    anyhow::ensure!(
        t.digest == s.digest && t.positions.len() == V4_TUNE_POSITIONS,
        "{}: not the sealed v4_tune_v1",
        path.display()
    );
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(exposure_log)?;
    writeln!(
        f,
        "V4 TUNE EXPOSURE phase={} approval={} digest={}",
        auth.phase, auth.approval_reference, t.digest
    )?;
    Ok(t)
}

#[cfg(test)]
mod v4_tests {
    use super::*;

    #[test]
    fn the_v4_tune_authorization_needs_the_final_phase_and_an_approval_reference() {
        assert!(V4TuneAuthorization::request("V4-P1", "ref").is_err());
        assert!(V4TuneAuthorization::request("V3-P8", "ref").is_err());
        assert!(V4TuneAuthorization::request("V4-FINAL", "  ").is_err());
        assert!(V4TuneAuthorization::request("V4-FINAL", "owner-2026-xx").is_ok());
    }

    #[test]
    fn an_evaluated_or_foreign_v4_seal_is_refused_before_any_data_is_read() {
        let dir = std::env::temp_dir().join(format!("recur64-v4-seal-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let auth = V4TuneAuthorization::request("V4-FINAL", "owner-test").unwrap();
        let mk = |evaluated: bool, schema: &str| V4TuneSeal {
            schema: schema.into(),
            identity: V4_TUNE_IDENTITY.into(),
            file: "x".into(),
            digest: "d".into(),
            positions: V4_TUNE_POSITIONS,
            verified: true,
            sealed: true,
            evaluated,
            permitted_p0p1_use: String::new(),
            future_access: String::new(),
        };
        for (i, seal) in [mk(true, V4_TUNE_SEAL_SCHEMA), mk(false, SEAL_SCHEMA)]
            .iter()
            .enumerate()
        {
            let p = dir.join(format!("seal{i}.json"));
            std::fs::write(&p, serde_json::to_vec(seal).unwrap()).unwrap();
            let missing = dir.join("does-not-exist.json");
            let log = dir.join("exposure.log");
            let e = load_sealed_v4_tune(&missing, &p, &auth, &log)
                .err()
                .expect("must refuse")
                .to_string();
            // A refusal that names the seal, not the missing data file, proves the order.
            assert!(e.contains("seal"), "{e}");
            assert!(!log.exists(), "an exposure was logged for a refused load");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
