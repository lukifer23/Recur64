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
    let sha = git(&["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".to_string());
    let branch =
        git(&["rev-parse", "--abbrev-ref", "HEAD"]).unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=RECUR64_GIT_SHA={sha}");
    println!("cargo:rustc-env=RECUR64_GIT_BRANCH={branch}");
    println!("cargo:rerun-if-changed=build.rs");
    // Re-run when the checked-out revision changes.
    println!("cargo:rerun-if-changed=../../.git/HEAD");
}
