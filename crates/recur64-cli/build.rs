//! Bind V5 scientific commands to the code revision used to build this binary.
//! Historical CLI commands do not depend on this identity.
use std::path::{Path, PathBuf};
use std::process::Command;

fn git(root: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .args(args)
        .current_dir(root)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn main() {
    let root = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("../..");
    // Worktrees keep HEAD under a worktree-specific Git directory and branch
    // references under the common directory. Watch both so committing source
    // triggers a rebuild even when no Rust file changed after the last build.
    for git_path in ["HEAD", "refs", "packed-refs"] {
        if let Some(path) = git(&root, &["rev-parse", "--git-path", git_path]) {
            let path = root.join(path);
            if path.exists() {
                println!("cargo:rerun-if-changed={}", path.display());
            }
        }
    }
    println!("cargo:rerun-if-changed=build.rs");
    let source = git(
        &root,
        &[
            "log",
            "-1",
            "--format=%H",
            "--",
            "crates",
            "Cargo.toml",
            "Cargo.lock",
            "configs",
        ],
    )
    .filter(|sha| sha.len() == 40)
    .unwrap_or_else(|| "unavailable".into());
    println!("cargo:rustc-env=RECUR64_V5_BUILD_SOURCE_SHA={source}");
}
