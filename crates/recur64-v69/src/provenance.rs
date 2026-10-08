//! Source/provenance helpers shared by all V69 binaries.

use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

pub fn sha256_hex(bytes: &[u8]) -> String {
    crate::streams::hex(&Sha256::digest(bytes))
}

pub fn git(args: &[&str]) -> String {
    std::process::Command::new("git")
        .args(args)
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

/// Digest of the executable V69 sources (data crate, model crate, core rules),
/// the lock file and toolchain pin. Run from the workspace root.
pub fn source_digest() -> String {
    fn walk(d: &Path, out: &mut Vec<PathBuf>) {
        if let Ok(rd) = std::fs::read_dir(d) {
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    walk(&p, out)
                } else {
                    out.push(p)
                }
            }
        }
    }
    let mut files: Vec<PathBuf> = Vec::new();
    for d in ["crates/recur64-v69", "crates/recur64-v69-model", "crates/recur64-core/src"] {
        walk(Path::new(d), &mut files);
    }
    files.push("Cargo.lock".into());
    files.push("rust-toolchain.toml".into());
    files.sort();
    let mut h = Sha256::new();
    for f in files {
        if let Ok(b) = std::fs::read(&f) {
            h.update(f.to_string_lossy().replace('\\', "/").as_bytes());
            h.update(Sha256::digest(&b));
        }
    }
    crate::streams::hex(&h.finalize())
}

#[derive(serde::Serialize, Clone, Debug)]
pub struct SourceId {
    pub git_head: String,
    pub git_dirty_files: usize,
    pub source_digest: String,
}

pub fn source_id() -> SourceId {
    SourceId {
        git_head: git(&["rev-parse", "HEAD"]),
        git_dirty_files: git(&["status", "--porcelain"]).lines().count(),
        source_digest: source_digest(),
    }
}
