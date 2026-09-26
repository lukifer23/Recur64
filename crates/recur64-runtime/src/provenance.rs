//! Build-time git provenance, baked by this crate's `build.rs`.
//!
//! `cargo:rustc-env` from a build script is visible only to the crate that
//! owns the script, so `option_env!("RECUR64_GIT_SHA")` inside another crate
//! (e.g. `recur64-cli`) always compiles to `None`. Every crate must read the
//! revision through these functions instead.

/// Commit SHA the binary was built from; `-dirty` is appended when the
/// crates, manifests or lockfile differed from HEAD at build time.
pub fn git_revision() -> Option<&'static str> {
    option_env!("RECUR64_GIT_SHA")
}

/// Branch name at build time.
pub fn git_branch() -> Option<&'static str> {
    option_env!("RECUR64_GIT_BRANCH")
}
