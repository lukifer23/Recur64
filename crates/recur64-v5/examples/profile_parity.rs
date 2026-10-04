//! Focused engineering executable. No qualification or training authority.
//! Build with RECUR64_DIAGNOSTIC_SOURCE_SHA set to the clean code revision.
#[cfg(not(feature = "cuda"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("CUDA feature required; no fallback")
}

#[cfg(feature = "cuda")]
fn main() -> anyhow::Result<()> {
    std::thread::Builder::new()
        .name("v5-profile-parity".into())
        .stack_size(64 * 1024 * 1024)
        .spawn(run)?
        .join()
        .map_err(|_| anyhow::anyhow!("diagnostic worker panicked"))?
}

#[cfg(feature = "cuda")]
fn run() -> anyhow::Result<()> {
    use std::path::PathBuf;
    use std::process::Command;
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let git = |args: &[&str]| -> anyhow::Result<String> {
        let result = Command::new("git").args(args).current_dir(&root).output()?;
        anyhow::ensure!(result.status.success(), "git identity check failed");
        Ok(String::from_utf8(result.stdout)?.trim().to_owned())
    };
    anyhow::ensure!(
        git(&[
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            "crates",
            "Cargo.toml",
            "Cargo.lock",
            "configs"
        ])?
        .is_empty(),
        "uncommitted scientific source; diagnostic refused"
    );
    let source = git(&[
        "log",
        "-1",
        "--format=%H",
        "--",
        "crates",
        "Cargo.toml",
        "Cargo.lock",
        "configs",
    ])?;
    anyhow::ensure!(
        option_env!("RECUR64_DIAGNOSTIC_SOURCE_SHA") == Some(source.as_str()),
        "diagnostic build/source mismatch; rebuild with RECUR64_DIAGNOSTIC_SOURCE_SHA={source}"
    );
    let mut args = std::env::args_os().skip(1);
    let output = PathBuf::from(
        args.next()
            .ok_or_else(|| anyhow::anyhow!("expected fresh output path"))?,
    );
    anyhow::ensure!(
        args.next().is_none() && !output.exists(),
        "expected one fresh output path"
    );
    let report = recur64_v5::qualification::diagnostic::run::<
        burn::backend::Autodiff<burn::backend::Cuda>,
    >(&source, "cuda", &Default::default())?;
    std::fs::write(output, serde_json::to_vec_pretty(&report)?)?;
    println!(
        "Engineering diagnostic completed: {} (training_authorized=false)",
        report["classification"]
    );
    Ok(())
}
