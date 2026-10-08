//! Artifact-namespace custody guard.
//!
//! Every V69 path (input or output) must resolve inside the V69 artifact root
//! and must not resolve under a historical scientific-artifact directory.

use anyhow::{Result, bail};
use std::path::{Component, Path, PathBuf};

/// Directory names that denote historical scientific artifacts in this
/// repository family. A V69 path containing any of these components is refused
/// even if (mis)placed inside the namespace.
pub const HISTORICAL_COMPONENTS: &[&str] = &[
    "runs", "evidence", "checkpoints", "datasets", "proof_targets", "prooftargets",
    "acquisition", "donor", "oracle_cache", "x1", "x15", "x2", "h1", "h3", "r15",
];

/// Sibling worktree directory names that hold historical experiments.
pub const HISTORICAL_WORKTREES: &[&str] = &["Recur64", "Recur64-v5", "Recur64-v6"];

#[derive(Debug, Clone)]
pub struct Custody {
    root: PathBuf,
}

fn normalize(p: &Path) -> PathBuf {
    // Lexical normalization (the path may not exist yet): resolve `.`/`..`.
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

fn canon_existing_prefix(p: &Path) -> PathBuf {
    // Canonicalize the longest existing ancestor (resolves symlinks/junctions),
    // then re-append the non-existing tail.
    let p = normalize(p);
    let mut tail = Vec::new();
    let mut cur = p.clone();
    loop {
        if let Ok(c) = std::fs::canonicalize(&cur) {
            let mut r = c;
            for t in tail.iter().rev() {
                r.push(t);
            }
            return r;
        }
        match (cur.file_name().map(|s| s.to_owned()), cur.parent().map(|x| x.to_owned())) {
            (Some(f), Some(par)) => {
                tail.push(f);
                cur = par;
            }
            _ => return p,
        }
    }
}

fn strip_verbatim(s: String) -> String {
    s.strip_prefix(r"\\?\").map(str::to_owned).unwrap_or(s).replace('\\', "/").to_lowercase()
}

impl Custody {
    /// `root` is the V69 artifact root (created if missing).
    pub fn new(root: &Path) -> Result<Self> {
        std::fs::create_dir_all(root)?;
        Ok(Self { root: std::fs::canonicalize(root)? })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Resolve `p` (relative paths are relative to the artifact root). Fails
    /// visibly if the result escapes the namespace or touches a historical
    /// scientific-artifact location.
    pub fn resolve(&self, p: &Path) -> Result<PathBuf> {
        let joined = if p.is_absolute() { p.to_path_buf() } else { self.root.join(p) };
        let resolved = canon_existing_prefix(&joined);
        let r = strip_verbatim(resolved.to_string_lossy().into_owned());
        let root = strip_verbatim(self.root.to_string_lossy().into_owned());
        if !(r == root || r.starts_with(&format!("{root}/"))) {
            bail!(
                "CUSTODY VIOLATION: {} resolves to {} which is outside the V69 artifact root {}",
                p.display(),
                resolved.display(),
                self.root.display()
            );
        }
        let rel = &r[root.len()..];
        for comp in rel.split('/').filter(|c| !c.is_empty()) {
            if HISTORICAL_COMPONENTS.iter().any(|h| comp == *h) {
                bail!(
                    "CUSTODY VIOLATION: {} contains historical-artifact component '{comp}'",
                    resolved.display()
                );
            }
        }
        // Belt and braces: the root itself must not be inside a historical
        // worktree's `runs` tree or any historical worktree directory.
        let components: Vec<String> = root.split('/').map(|s| s.to_string()).collect();
        for w in HISTORICAL_WORKTREES {
            if components.iter().any(|c| c.eq_ignore_ascii_case(w)) {
                bail!("CUSTODY VIOLATION: artifact root {} is inside historical worktree '{w}'", root);
            }
        }
        Ok(resolved)
    }
}
