//! The query tool must not be able to import the solver, model or search.

const FORBIDDEN: [&str; 6] = [
    "recur64-runtime",
    "recur64-model",
    "recur64-search",
    "recur64-eval",
    "recur64-cli",
    "burn",
];

#[test]
fn manifest_depends_only_on_core_serde_sha2() {
    let manifest = include_str!("../Cargo.toml");
    let names: Vec<&str> = manifest
        .lines()
        .skip_while(|l| !l.trim_start().starts_with("[dependencies]"))
        .skip(1)
        .take_while(|l| !l.trim_start().starts_with('['))
        .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#'))
        .map(|l| l.split('=').next().unwrap().trim())
        .collect();
    assert_eq!(
        names,
        ["recur64-core", "serde", "sha2"],
        "unexpected dependencies: {names:?}"
    );
    for f in FORBIDDEN {
        assert!(!names.contains(&f), "forbidden dependency {f}");
    }
}

#[test]
fn source_never_names_a_forbidden_crate() {
    let src = include_str!("../src/lib.rs");
    for f in [
        "recur64_runtime",
        "recur64_model",
        "recur64_search",
        "recur64_eval",
        "proof::",
        "MateSolver",
    ] {
        assert!(!src.contains(f), "lib.rs references {f}");
    }
}
