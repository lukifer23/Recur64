//! Model-side data loading. Opens files ONLY through the role-restricted
//! `Access` (learner: data/fit.jsonl; evaluator: fit + val). Each file's SHA-256
//! is checked against the dataset manifest before use.

use anyhow::{Context, Result, ensure};
use recur64_v69::access::Access;
use recur64_v69::features::{FamilyKind, Features, ModelRow, featurize, read_rows};
use recur64_v69::provenance::sha256_hex;
use std::collections::BTreeMap;
use std::path::Path;

pub struct Loaded {
    pub rel: String,
    pub sha256: String,
    pub rows: Vec<ModelRow>,
    pub feats: Vec<Features>,
    pub fams: Vec<FamilyKind>,
}

/// Load `data/<name>.jsonl` from the dataset directory `run_rel`.
pub fn load_split(access: &Access, run_rel: &Path, name: &str) -> Result<Loaded> {
    let rel = format!("data/{name}.jsonl");
    let manifest: BTreeMap<String, String> = serde_json::from_slice(&access.read(&run_rel.join("MANIFEST.sha256.json"))?).context("manifest")?;
    let bytes = access.read(&run_rel.join(&rel))?;
    let sha = sha256_hex(&bytes);
    let want = manifest.get(&rel).with_context(|| format!("manifest lacks {rel}"))?;
    ensure!(&sha == want, "hash mismatch for {rel}: {sha} != {want}");
    let rows = read_rows(std::str::from_utf8(&bytes)?)?;
    let mut feats = Vec::with_capacity(rows.len());
    let mut fams = Vec::with_capacity(rows.len());
    for r in &rows {
        let (f, fam) = featurize(&r.fen, r.budget).with_context(|| format!("row {}", r.id))?;
        feats.push(f);
        fams.push(fam);
    }
    Ok(Loaded { rel, sha256: sha, rows, feats, fams })
}

/// Manifest hash of a dataset-relative file (no content read).
pub fn manifest_hash(access: &Access, run_rel: &Path, rel: &str) -> Result<String> {
    let manifest: BTreeMap<String, String> = serde_json::from_slice(&access.read(&run_rel.join("MANIFEST.sha256.json"))?)?;
    manifest.get(rel).cloned().with_context(|| format!("manifest lacks {rel}"))
}
