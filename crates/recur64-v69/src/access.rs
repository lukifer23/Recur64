//! Role-based access policy for the accepted gen-001 dataset.
//!
//! Custody (custody.rs) confines paths to the V69 namespace; this layer adds ROLE
//! restrictions inside the dataset directory. Model-side code (learner, endpoint
//! evaluator) must open dataset files only through `Access`, which refuses
//! sealed/, pool/, and root-bearing metadata. Read-only file attributes are not
//! treated as a barrier.

use crate::custody::Custody;
use anyhow::{Result, bail};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// Reads the master seed (stream derivation) and data/fit.jsonl only.
    Learner,
    /// Reads data/fit.jsonl and data/val.jsonl only.
    Evaluator,
    /// Reads fit/val metadata, predictions and intervention maps.
    MetricAggregator,
    /// Data-only integrity audit: may read the whole dataset directory.
    DataAudit,
    /// D1 panel selection (data side): fit rows + fit metadata, writes d1/ only.
    D1Panel,
    /// G1 data builder/auditor: gen-001 pool (exclusion index / verification), G1 seed; writes g1/.
    G1Builder,
    /// G1 frozen-candidate evaluator: G1 model rows + frozen candidate files only; no metadata, no gen-001 data.
    G1Evaluator,
    /// G1 metric aggregator: G1 metadata + predictions + fitting reports.
    G1Aggregator,
    /// T1 data builder/auditor (gen-001 + G1 pools for identity exclusion; writes t1/).
    T1Builder,
    /// T1 trainer: T1 train/val rows and canonical inits; writes t1/runs/. Never test rows or metadata.
    T1Trainer,
    /// T1 final evaluator: frozen candidates + T1 test rows (+ G1 panel rows); no metadata, no train rows.
    T1Evaluator,
    /// T1 aggregator: metadata + predictions.
    T1Aggregator,
}

#[derive(Clone)]
pub struct Access {
    custody: Custody,
    role: Role,
    /// Dataset directory (e.g. artifacts/v69/gen-001), already resolved.
    dataset_dir: PathBuf,
    /// Seed file path (resolved).
    seed_file: PathBuf,
}

fn norm(p: &Path) -> String {
    let s = p.to_string_lossy().into_owned();
    s.strip_prefix(r"\\?\").map(str::to_owned).unwrap_or(s).replace('\\', "/").to_lowercase()
}

impl Access {
    pub fn new(custody: &Custody, role: Role, dataset_rel: &Path, seed_rel: &Path) -> Result<Self> {
        Ok(Self {
            custody: custody.clone(),
            role,
            dataset_dir: custody.resolve(dataset_rel)?,
            seed_file: custody.resolve(seed_rel)?,
        })
    }

    pub fn role(&self) -> Role {
        self.role
    }

    pub fn custody(&self) -> &Custody {
        &self.custody
    }

    pub fn dataset_dir(&self) -> &Path {
        &self.dataset_dir
    }

    /// Allowed dataset-relative reads for the role.
    fn allowed_dataset_reads(role: Role) -> &'static [&'static str] {
        match role {
            Role::Learner => &["data/fit.jsonl", "MANIFEST.sha256.json"],
            Role::Evaluator => &["data/fit.jsonl", "data/val.jsonl", "MANIFEST.sha256.json"],
            Role::MetricAggregator => &["meta/fit.meta.jsonl", "meta/val.meta.jsonl", "MANIFEST.sha256.json"],
            Role::DataAudit => &[],
            Role::D1Panel => &["data/fit.jsonl", "meta/fit.meta.jsonl", "MANIFEST.sha256.json"],
            Role::G1Builder | Role::T1Builder => &["pool/roots.jsonl", "MANIFEST.sha256.json"],
            Role::G1Evaluator | Role::G1Aggregator | Role::T1Trainer | Role::T1Evaluator | Role::T1Aggregator => &[],
        }
    }

    /// Output prefixes (relative to the V69 artifact root) the role may write.
    fn allowed_write_prefixes(role: Role) -> &'static [&'static str] {
        match role {
            Role::Learner => &["fits/", "qual/", "init/", "spec/", "d1/", "d2/", "d3/"],
            Role::Evaluator => &["eval/", "intervention/", "d1/"],
            Role::MetricAggregator => &["report/", "d1/", "d2/", "d3/"],
            Role::DataAudit => &["audit/", "d1/", "d2/", "d3/", "g1/", "g1r1/", "t1/"],
            Role::D1Panel => &["d1/", "d2/", "d3/"],
            Role::G1Builder => &["g1r1/"],
            Role::G1Evaluator => &["g1r1/eval/", "g1r1/receipts/"],
            Role::G1Aggregator => &["g1r1/report/"],
            Role::T1Builder => &["t1/"],
            Role::T1Trainer => &["t1/runs/"],
            Role::T1Evaluator => &["t1/final/", "t1/receipts/"],
            Role::T1Aggregator => &["t1/report/"],
        }
    }

    /// Non-dataset V69 files readable by role (relative prefixes under the root).
    fn allowed_other_read_prefixes(role: Role) -> &'static [&'static str] {
        match role {
            Role::Learner => &["fits/", "qual/", "init/", "spec/", "d1/", "d2/", "d3/"],
            Role::Evaluator => &["fits/", "init/", "spec/", "intervention/", "eval/", "d1/"],
            Role::MetricAggregator => &["eval/", "intervention/", "spec/", "report/", "fits/", "d1/", "d2/", "d3/"],
            Role::DataAudit => &["audit/", "spec/", "d1/", "d2/", "d3/", "g1/", "g1r1/", "t1/", "fits/", "eval/", "report/", "init/", "intervention/", "qual/"],
            Role::D1Panel => &["d1/", "d2/", "d3/", "spec/", "init/"],
            Role::G1Builder => &["g1r1/", "spec/"],
            Role::G1Evaluator => &["g1r1/", "d3/fits/", "d1/baseline/", "d2/subsets/s768_rows.jsonl"],
            Role::G1Aggregator => &["g1r1/", "d3/report/", "d1/report/"],
            Role::T1Builder => &["t1/", "g1r1/pool/", "g1r1/index/", "spec/"],
            Role::T1Trainer => &["t1/", "init/canonical_init.bin", "d1/mlp_init.bin"],
            Role::T1Evaluator => &["t1/", "g1r1/rows/", "g1r1/intervention/", "d1/baseline/"],
            Role::T1Aggregator => &["t1/", "g1r1/meta/", "g1r1/intervention/", "g1r1/eval/", "d3/report/", "d1/report/"],
        }
    }

    /// Parts a role must never read even inside an otherwise allowed prefix: root-bearing
    /// metadata, exclusion indexes, pools, held-out rows and optimizer states.
    fn denied(role: Role, rel: &str) -> bool {
        let g1 = ["g1r1/meta/", "g1r1/pool/", "g1r1/index/", "g1/", "d3/fits/a/final/opt", "d3/fits/m/final/opt"];
        match role {
            Role::G1Evaluator => g1.iter().any(|d| rel.starts_with(d)),
            Role::T1Trainer => ["t1/meta/", "t1/pool/", "t1/index/", "t1/rows/test", "t1/final/"].iter().any(|d| rel.starts_with(d)),
            Role::T1Evaluator => ["t1/meta/", "t1/pool/", "t1/index/", "t1/rows/train", "t1/rows/val"].iter().any(|d| rel.starts_with(d)) || (rel.starts_with("t1/runs/") && rel.contains("/opt_")),
            _ => false,
        }
    }

    fn rel_to_root(&self, resolved: &Path) -> Option<String> {
        let root = norm(self.custody.root());
        let p = norm(resolved);
        p.strip_prefix(&format!("{root}/")).map(|s| s.to_string())
    }

    /// Validate (and resolve) a path for reading under this role.
    pub fn check_read(&self, p: &Path) -> Result<PathBuf> {
        let resolved = self.custody.resolve(p)?;
        if norm(&resolved) == norm(&self.seed_file) {
            // All roles derive streams from the seed (learner: train_order; evaluator:
            // intervention; aggregator: bootstrap). The seed carries no data.
            return Ok(resolved);
        }
        let ds = norm(&self.dataset_dir);
        let rp = norm(&resolved);
        if rp == ds || rp.starts_with(&format!("{ds}/")) {
            if self.role == Role::DataAudit {
                return Ok(resolved);
            }
            let rel = rp.strip_prefix(&format!("{ds}/")).unwrap_or("");
            if Self::allowed_dataset_reads(self.role).iter().any(|a| a.to_lowercase() == rel) {
                return Ok(resolved);
            }
            bail!("ACCESS VIOLATION: role {:?} may not read dataset file '{rel}'", self.role);
        }
        if let Some(rel) = self.rel_to_root(&resolved) {
            if Self::denied(self.role, &rel) {
                bail!("ACCESS VIOLATION: role {:?} may not read '{rel}'", self.role);
            }
            if Self::allowed_other_read_prefixes(self.role).iter().any(|a| rel.starts_with(a)) {
                return Ok(resolved);
            }
            bail!("ACCESS VIOLATION: role {:?} may not read '{rel}'", self.role);
        }
        bail!("ACCESS VIOLATION: {} outside namespace", resolved.display())
    }

    pub fn check_write(&self, p: &Path) -> Result<PathBuf> {
        let resolved = self.custody.resolve(p)?;
        let rp = norm(&resolved);
        let ds = norm(&self.dataset_dir);
        if rp == ds || rp.starts_with(&format!("{ds}/")) {
            if self.role == Role::DataAudit {
                let rel = rp.strip_prefix(&format!("{ds}/")).unwrap_or("");
                if rel == "audit_receipt_v2.json" {
                    return Ok(resolved);
                }
            }
            bail!("ACCESS VIOLATION: role {:?} may not write inside the dataset directory", self.role);
        }
        if let Some(rel) = self.rel_to_root(&resolved) {
            if Self::allowed_write_prefixes(self.role).iter().any(|a| rel.starts_with(a)) {
                return Ok(resolved);
            }
        }
        bail!("ACCESS VIOLATION: role {:?} may not write {}", self.role, resolved.display())
    }

    pub fn read_to_string(&self, p: &Path) -> Result<String> {
        Ok(std::fs::read_to_string(self.check_read(p)?)?)
    }

    pub fn read(&self, p: &Path) -> Result<Vec<u8>> {
        Ok(std::fs::read(self.check_read(p)?)?)
    }

    pub fn write(&self, p: &Path, bytes: &[u8]) -> Result<PathBuf> {
        let r = self.check_write(p)?;
        if let Some(parent) = r.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&r, bytes)?;
        Ok(r)
    }

    /// Resolve a path for directory-style outputs (checkpoint recorders); write
    /// permission is validated, the caller must not read dataset files with it.
    pub fn output_path(&self, p: &Path) -> Result<PathBuf> {
        let r = self.check_write(p)?;
        if let Some(parent) = r.parent() {
            std::fs::create_dir_all(parent)?;
        }
        Ok(r)
    }

    /// Resolve a previously written output for reading (checkpoints etc.).
    pub fn input_path(&self, p: &Path) -> Result<PathBuf> {
        self.check_read(p)
    }
}
