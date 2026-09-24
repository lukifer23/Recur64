//! Capture git provenance at build time so every run's metadata records the
//! exact revision and branch it was built from (HP experiment requirement).
//! Falls back to "unknown" rather than failing the build when git is absent.

use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() { None } else { Some(s) }
}

fn main() {
    let mut sha = git(&["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".to_string());
    let branch =
        git(&["rev-parse", "--abbrev-ref", "HEAD"]).unwrap_or_else(|| "unknown".to_string());
    // A binary built from uncommitted source must not claim a clean revision.
    // Only inputs to the binary count (crates, manifests, lockfile).
    let dirty = Command::new("git")
        .args([
            "diff",
            "--quiet",
            "HEAD",
            "--",
            "crates",
            "Cargo.toml",
            "Cargo.lock",
        ])
        .current_dir("../..")
        .status()
        .map(|s| !s.success())
        .unwrap_or(false);
    if dirty && sha != "unknown" {
        sha.push_str("-dirty");
    }
    println!("cargo:rustc-env=RECUR64_GIT_SHA={sha}");
    println!("cargo:rustc-env=RECUR64_GIT_BRANCH={branch}");
    println!("cargo:rerun-if-changed=build.rs");
    // HEAD only changes on checkout; commits move the branch ref. Watch both,
    // plus the sources so the dirty flag cannot go stale.
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    println!("cargo:rerun-if-changed=../../.git/refs/heads");
    println!("cargo:rerun-if-changed=../../.git/packed-refs");
    println!("cargo:rerun-if-changed=../../crates");
    println!("cargo:rerun-if-changed=../../Cargo.lock");
}
