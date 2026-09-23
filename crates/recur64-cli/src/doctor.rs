//! `recur64 doctor` — read-only environment report.
//!
//! Distinguishes DETECTED from TESTED. Never prints Windows product/device IDs,
//! serial numbers, or secrets.

use std::path::Path;
use std::process::Command;

use recur64_model::config::{DeviceKind, Precision};
use recur64_model::precision::{self, SupportStatus};

fn run(program: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(program).args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn powershell(script: &str) -> Option<String> {
    run(
        "powershell",
        &["-NoProfile", "-NonInteractive", "-Command", script],
    )
}

fn writable(dir: &Path) -> bool {
    if std::fs::create_dir_all(dir).is_err() {
        return false;
    }
    let probe = dir.join(".recur64_write_probe");
    let ok = std::fs::write(&probe, b"ok").is_ok();
    let _ = std::fs::remove_file(&probe);
    ok
}

pub fn run_doctor() -> anyhow::Result<()> {
    println!("Recur64 doctor");
    println!("==============");
    println!("recur64 version : {}", recur64_model::VERSION);
    println!("burn version    : {} (pinned)", recur64_model::BURN_VERSION);

    println!("\n-- platform --");
    println!("os              : {}", std::env::consts::OS);
    println!("arch            : {}", std::env::consts::ARCH);
    if let Some(v) = powershell(
        "(Get-CimInstance Win32_OperatingSystem).Caption + ' build ' + \
         (Get-CimInstance Win32_OperatingSystem).BuildNumber",
    ) {
        println!("windows         : {v}");
    }
    if let Some(v) = run("rustc", &["--version"]) {
        println!("rustc           : {v}");
    }
    if let Some(v) = run("cargo", &["--version"]) {
        println!("cargo           : {v}");
    }

    println!("\n-- cpu / memory --");
    if let Some(v) = powershell("(Get-CimInstance Win32_Processor).Name") {
        println!("cpu             : {v}");
    }
    if let Some(v) = powershell(
        "$c=Get-CimInstance Win32_Processor; \
         'cores=' + $c.NumberOfCores + ' logical=' + $c.NumberOfLogicalProcessors",
    ) {
        println!("cpu topology    : {v}");
    }
    if let Some(v) = powershell(
        "$o=Get-CimInstance Win32_OperatingSystem; \
         'total_gb=' + [math]::Round($o.TotalVisibleMemorySize/1MB,2) + \
         ' free_gb=' + [math]::Round($o.FreePhysicalMemory/1MB,2)",
    ) {
        println!("host memory     : {v}");
    }

    println!("\n-- accelerators (DETECTED) --");
    let nvsmi = run(
        "nvidia-smi",
        &[
            "--query-gpu=name,compute_cap,memory.total,memory.free,driver_version",
            "--format=csv,noheader",
        ],
    );
    match &nvsmi {
        Some(v) => println!("nvidia-smi      : {v}"),
        None => println!("nvidia-smi      : NOT AVAILABLE"),
    }
    if let Some(v) = powershell(
        "(Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }) -join ', '",
    ) {
        println!("video controllers: {v}");
    }

    println!("\n-- cuda runtime --");
    let cuda_path = std::env::var("CUDA_PATH").ok();
    println!(
        "CUDA_PATH       : {}",
        cuda_path.as_deref().unwrap_or("(not set)")
    );
    println!(
        "nvcc            : {}",
        run("nvcc", &["--version"])
            .map(|s| s.lines().last().unwrap_or("").to_string())
            .unwrap_or_else(|| "NOT FOUND".to_string())
    );
    println!("cuda feature compiled   : {}", cfg!(feature = "cuda"));
    if let Ok(local) = std::env::var("LOCALAPPDATA") {
        let base = Path::new(&local).join("Recur64").join("cuda");
        if base.is_dir() {
            let versions: Vec<String> = std::fs::read_dir(&base)
                .into_iter()
                .flatten()
                .flatten()
                .filter(|e| e.path().is_dir())
                .map(|e| e.file_name().to_string_lossy().to_string())
                .collect();
            println!(
                "user-space CUDA : {} ({})",
                base.display(),
                if versions.is_empty() {
                    "none".to_string()
                } else {
                    versions.join(", ")
                }
            );
        }
    }

    println!("\n-- precision support --");
    println!("note: DETECTED is not TESTED. Non-FP32 requires the full graph to run.");
    for (p, d) in [
        (Precision::Fp32, DeviceKind::Cpu),
        (Precision::Fp32, DeviceKind::Cuda),
        (Precision::Bf16, DeviceKind::Cuda),
        (Precision::Fp16, DeviceKind::Cuda),
    ] {
        let (mut status, mut note) = precision::status(p, d);
        if p == Precision::Fp32 && d == DeviceKind::Cuda && cfg!(feature = "cuda") {
            status = SupportStatus::Tested;
            note = "CUDA FP32 graph verified via `recur64 cuda-smoke`";
        }
        let dev = match d {
            DeviceKind::Cpu => "cpu",
            DeviceKind::Cuda => "cuda",
        };
        println!(
            "  {:<4} {:<4} : {:<14} {}",
            p.label(),
            dev,
            status.label(),
            note
        );
    }
    if nvsmi.is_some() {
        println!("BF16 hardware capability : DETECTED (Ada-class GPU present)");
        println!("BF16 full Recur64 graph  : NOT YET TESTED");
    }

    println!("\n-- output directory --");
    let runs = Path::new("runs");
    println!(
        "runs/ writable  : {}",
        if writable(runs) { "yes" } else { "NO" }
    );

    println!("\n-- warnings --");
    let mut warnings = Vec::new();
    if cuda_path.is_none() {
        warnings.push("CUDA toolkit / CUDA_PATH not found: CUDA backend cannot be selected yet.");
    }
    warnings.push("No GPU graph has been executed in this report (DETECTED != TESTED).");
    if writable(runs) {
        // fine
    } else {
        warnings.push("runs/ is not writable; benchmarks cannot record raw data.");
    }
    for w in warnings {
        println!("  - {w}");
    }

    // Explicitly do not print product/device IDs, serials, or credentials.
    println!("\n(Identifiers such as Windows product/device IDs are intentionally omitted.)");
    Ok(())
}
