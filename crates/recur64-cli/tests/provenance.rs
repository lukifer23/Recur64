//! H3.5B P6: CLI evidence must carry the build revision.
//!
//! `recur64-runtime/build.rs` bakes `RECUR64_GIT_SHA`, but build-script env
//! vars are crate-scoped: an `option_env!` in this crate silently compiles to
//! `None` (committed H3 eval-arena JSON had `"git_revision": null`). The CLI
//! must read provenance through `recur64_runtime::provenance`.

use std::path::Path;

fn repo_root() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."))
}

#[test]
fn runtime_provenance_is_baked_in_a_git_checkout() {
    if repo_root().join(".git").exists() {
        let sha = recur64_runtime::provenance::git_revision().expect("baked SHA");
        assert_ne!(sha, "unknown");
        assert!(recur64_runtime::provenance::git_branch().is_some());
    }
}

#[test]
fn cli_sources_do_not_read_build_env_directly() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let needle = ["option_env!(\"RECUR64_", "GIT"].concat();
    for entry in std::fs::read_dir(&src).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|e| e == "rs") {
            let text = std::fs::read_to_string(&path).unwrap();
            assert!(
                !text.contains(&needle),
                "{} reads RECUR64_GIT_* directly; use recur64_runtime::provenance",
                path.display()
            );
        }
    }
}
