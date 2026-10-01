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
