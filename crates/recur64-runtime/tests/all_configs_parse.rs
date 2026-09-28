//! Every tracked run config parses under the strict (deny-unknown-fields)
//! schema. Probe configs (`ProbeConfig`: configs/f*.toml, r*.toml) and the
//! opening suite are a different schema and are skipped.

use recur64_runtime::RunConfig;

#[test]
fn every_tracked_run_config_parses_strictly() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
    let mut checked = 0;
    let mut stack = vec![root.clone()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if path.extension().is_none_or(|e| e != "toml") {
                continue;
            }
            let text = std::fs::read_to_string(&path).unwrap();
            // Only RunConfig files carry run_id at top level.
            let is_run = text.lines().any(|l| l.trim_start().starts_with("run_id"));
            if !is_run {
                continue;
            }
            RunConfig::from_toml_str(&text)
                .unwrap_or_else(|e| panic!("{} does not parse strictly: {e}", path.display()));
            checked += 1;
        }
    }
    assert!(
        checked >= 20,
        "expected many run configs, checked {checked}"
    );
}
