//! P0-only CLI. No scientific campaign or DEV model capability exists.
use burn::tensor::backend::AutodiffBackend;
use clap::{Parser, Subcommand, ValueEnum};
use recur64_v6::{model::Arm, p0, packet, qualify};
use std::path::{Path, PathBuf};
#[derive(Clone, Copy, Debug, ValueEnum)]
enum Device {
    Cpu,
    Cuda,
}
#[derive(Clone, Copy, Debug, ValueEnum)]
enum ArmArg {
    Principal,
    OnePass,
}
impl From<ArmArg> for Arm {
    fn from(a: ArmArg) -> Self {
        match a {
            ArmArg::Principal => Arm::SharedBackup,
            ArmArg::OnePass => Arm::OnePass,
        }
    }
}
#[derive(Parser)]
struct Cli {
    #[arg(long)]
    data: PathBuf,
    #[arg(long)]
    dev_custody: PathBuf,
    #[arg(long)]
    confirm_custody: PathBuf,
    #[arg(long)]
    stage_a: PathBuf,
    #[arg(long, value_enum, default_value = "cuda")]
    device: Device,
    #[arg(long)]
    output: PathBuf,
    #[command(subcommand)]
    command: Cmd,
}
#[derive(Subcommand)]
enum Cmd {
    Identity,
    Custody,
    Qualify {
        #[arg(long)]
        run_dir: PathBuf,
    },
    Coverage,
    Freeze {
        #[arg(long)]
        cpu_qualification: PathBuf,
        #[arg(long)]
        cuda_qualification: PathBuf,
    },
    Learn {
        #[arg(long, value_enum)]
        arm: ArmArg,
        #[arg(long)]
        plan: PathBuf,
        #[arg(long)]
        plan_binding: PathBuf,
        #[arg(long)]
        run_dir: PathBuf,
        #[arg(long)]
        cpu_qualification: PathBuf,
        #[arg(long)]
        cuda_qualification: PathBuf,
        #[arg(long, default_value = "45")]
        max_minutes: f64,
        #[arg(long)]
        resume: bool,
    },
}
fn validate_qualification(path: &Path, backend: &str) -> anyhow::Result<()> {
    let r: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
    anyhow::ensure!(
        r["schema"] == "v6_p0_qualification_v1"
            && r["source_sha"] == recur64_v6::SOURCE
            && r["config_digest"] == packet::config_digest()?
            && r["backend"] == backend
            && r["precision"] == "fp32"
            && r["microbatch"] == 2
            && r["pass"] == true
            && r["baseline_parameters_exact"] == true,
        "qualification source/config/layout/pass mismatch"
    );
    let arms = r["arms"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("missing arm qualification"))?;
    anyhow::ensure!(
        arms.len() == 2
            && arms.iter().all(|a| a["resident_updates"] == 50
                && a["checkpoint_restore_exact"] == true
                && a["moment_restore_exact"] == true
                && a["continuation_exact"] == true
                && a["checks"].as_array().is_some_and(
                    |v| !v.is_empty() && v.iter().all(|c| c["normal_profile_exact"] == true)
                )),
        "incomplete qualification gates"
    );
    Ok(())
}
fn execute<B: AutodiffBackend>(c: &Cli, backend: &str) -> anyhow::Result<serde_json::Value> {
    let device = Default::default();
    if let Cmd::Freeze {
        cpu_qualification,
        cuda_qualification,
    }
    | Cmd::Learn {
        cpu_qualification,
        cuda_qualification,
        ..
    } = &c.command
    {
        validate_qualification(cpu_qualification, "cpu")?;
        validate_qualification(cuda_qualification, "cuda")?;
    }
    // Custody BEFORE model construction, actual raw bytes and role refusals.
    let custody =
        recur64_v5::data::verify_local_boundaries(&c.data, &c.dev_custody, &c.confirm_custody)?;
    if matches!(c.command, Cmd::Custody) {
        return Ok(custody);
    }
    if matches!(c.command, Cmd::Identity) {
        return Ok(
            serde_json::json!({"source_sha":recur64_v6::SOURCE,"config":packet::config(),"config_digest":packet::config_digest()?,"capability":"P0 TRAIN only"}),
        );
    }
    let data = recur64_v5::data::V5Data::load_train(&c.data)?;
    let base = packet::import_base::<B>(&c.stage_a, &device, backend)?;
    match &c.command {
        Cmd::Qualify { run_dir } => qualify::run(&base, &data, &device, backend, run_dir),
        Cmd::Coverage => p0::coverage(&base, &data, &device),
        Cmd::Freeze { .. } => Ok(serde_json::to_value(p0::freeze(&base, &data, &device)?)?),
        Cmd::Learn {
            arm,
            plan,
            plan_binding,
            run_dir,
            resume,
            max_minutes,
            cpu_qualification,
            cuda_qualification,
        } => {
            anyhow::ensure!(
                backend == "cuda",
                "learnability requires CUDA, no CPU substitute"
            );
            let bytes = std::fs::read(plan)?;
            let binding: serde_json::Value = serde_json::from_slice(&std::fs::read(plan_binding)?)?;
            anyhow::ensure!(
                binding["schema"] == "v6_p0_measured_plan_binding_v1"
                    && binding["source_sha"] == recur64_v6::SOURCE
                    && binding["raw_sha256"] == recur64_v5::stage::hash_file(plan)?
                    && binding["cpu_qualification_sha256"]
                        == recur64_v5::stage::hash_file(cpu_qualification)?
                    && binding["cuda_qualification_sha256"]
                        == recur64_v5::stage::hash_file(cuda_qualification)?,
                "measured plan/raw/qualification binding mismatch"
            );
            let p: p0::Plan = serde_json::from_slice(&bytes)?;
            anyhow::ensure!(
                binding["plan_digest"] == p.digest,
                "scientific plan binding mismatch"
            );
            p0::learn(
                &base,
                &data,
                &p,
                (*arm).into(),
                run_dir,
                &device,
                p0::LearnOptions {
                    resume: *resume,
                    max_minutes: *max_minutes,
                },
            )
        }
        _ => unreachable!(),
    }
}
fn require_committed_source() -> anyhow::Result<()> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let git = |args: &[&str]| -> anyhow::Result<String> {
        let result = std::process::Command::new("git")
            .args(args)
            .current_dir(&root)
            .output()?;
        anyhow::ensure!(result.status.success(), "source verification Git failure");
        Ok(String::from_utf8(result.stdout)?.trim().to_owned())
    };
    anyhow::ensure!(
        git(&["branch", "--show-current"])? == "experiment/hp-v6-branch-backup",
        "P0 requires its isolated V6 branch"
    );
    anyhow::ensure!(
        git(&[
            "log",
            "-1",
            "--format=%H",
            "--",
            "crates",
            "Cargo.toml",
            "Cargo.lock",
            "configs"
        ])? == recur64_v6::SOURCE,
        "binary/current scientific source mismatch"
    );
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
        "uncommitted scientific source refused"
    );
    Ok(())
}
fn main() -> anyhow::Result<()> {
    require_committed_source()?;
    let c = Cli::parse();
    anyhow::ensure!(!c.output.exists(), "refuse overwriting receipt");
    let device = c.device;
    std::thread::Builder::new().stack_size(64*1024*1024).spawn(move||{let result=match device{Device::Cpu=>execute::<recur64_model::train::CpuTrainBackend>(&c,"cpu"),Device::Cuda=>{#[cfg(feature="cuda")]{execute::<burn::backend::Autodiff<burn::backend::Cuda>>(&c,"cuda")}#[cfg(not(feature="cuda"))]{anyhow::bail!("CUDA binary unavailable, no CPU substitution")}}};let value=match &result{Ok(v)=>v.clone(),Err(e)=>serde_json::json!({"schema":"v6_p0_failure_v1","source_sha":recur64_v6::SOURCE,"pass":false,"error":format!("{e:#}"),"fitting_authorized":false})};if let Some(parent)=c.output.parent(){std::fs::create_dir_all(parent)?;}std::fs::write(c.output,serde_json::to_vec_pretty(&value)?)?;result.map(|_|())})?.join().map_err(|_|anyhow::anyhow!("worker panic, preserve native logs; no next invocation"))?
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn no_dev_campaign_resume_escape_or_extra_step_cli() {
        for forbidden in [
            "train",
            "evaluate",
            "self-play",
            "controller",
            "extra-loops",
        ] {
            assert!(Cli::try_parse_from(["recur64-v6", forbidden]).is_err());
        }
        assert!(Cli::try_parse_from(["recur64-v6", "learn", "--updates", "801"]).is_err());
    }
}
