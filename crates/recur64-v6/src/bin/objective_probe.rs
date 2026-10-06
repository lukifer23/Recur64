//! Objective-probe CLI: TRAIN-only. No DEV/CONFIRM model capability, no campaign,
//! no extra-update or seed escape. Every command refuses uncommitted scientific source.
use burn::tensor::backend::AutodiffBackend;
use clap::{Parser, Subcommand, ValueEnum};
use recur64_v6::{objective, packet, probe_eval, probe_qual};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Device {
    Cpu,
    Cuda,
}
#[derive(Clone, Copy, Debug, ValueEnum)]
enum ObjectiveArg {
    Control,
    Treatment,
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
        p0_plan: PathBuf,
        #[arg(long)]
        p0_binding: PathBuf,
        #[arg(long)]
        run_dir: PathBuf,
    },
    /// CUDA only: create the ONE canonical initial reader and the sealed plan.
    Preregister {
        #[arg(long)]
        p0_plan: PathBuf,
        #[arg(long)]
        p0_binding: PathBuf,
        #[arg(long)]
        contract: PathBuf,
        #[arg(long)]
        initial_dir: PathBuf,
        #[arg(long)]
        plan_out: PathBuf,
        #[arg(long)]
        cpu_qualification: PathBuf,
        #[arg(long)]
        cuda_qualification: PathBuf,
    },
    Learn {
        #[arg(long, value_enum)]
        objective: ObjectiveArg,
        #[arg(long)]
        p0_plan: PathBuf,
        #[arg(long)]
        p0_binding: PathBuf,
        #[arg(long)]
        contract: PathBuf,
        #[arg(long)]
        plan: PathBuf,
        #[arg(long)]
        plan_binding: PathBuf,
        #[arg(long)]
        initial_dir: PathBuf,
        #[arg(long)]
        run_dir: PathBuf,
        #[arg(long)]
        cpu_qualification: PathBuf,
        #[arg(long)]
        cuda_qualification: PathBuf,
        #[arg(long, default_value = "40")]
        max_minutes: f64,
        #[arg(long)]
        resume: bool,
    },
    /// No model: compares the two completed arms' endpoint matrices.
    Decide {
        #[arg(long)]
        plan: PathBuf,
        #[arg(long)]
        plan_binding: PathBuf,
        #[arg(long)]
        contract: PathBuf,
        #[arg(long)]
        control_dir: PathBuf,
        #[arg(long)]
        control_result: PathBuf,
        #[arg(long)]
        treatment_dir: PathBuf,
        #[arg(long)]
        treatment_result: PathBuf,
    },
}

fn validate_qualification(path: &Path, backend: &str) -> anyhow::Result<()> {
    let r: Value = serde_json::from_slice(&std::fs::read(path)?)?;
    let arms = r["arms"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("missing arm qualification"))?;
    anyhow::ensure!(
        r["schema"] == "v6_objective_probe_qualification_v2"
            && r["source_sha"] == recur64_v6::SOURCE
            && r["config_digest"] == packet::config_digest()?
            && r["backend"] == backend
            && r["precision"] == "fp32"
            && r["microbatch"] == 2
            && r["pass"] == true
            && r["baseline_parameters_exact"] == true
            && r["total_parameters"] == objective::PARAMETERS_TOTAL
            && arms.len() == 2
            && arms.iter().all(|a| a["resident_updates"] == 50
                && a["checkpoint_restore_exact"] == true
                && a["moment_restore_exact"] == true
                && a["continuation_exact"] == true
                && a["parity"].as_array().is_some_and(
                    |v| !v.is_empty() && v.iter().all(|c| c["normal_profile_exact"] == true)
                )),
        "qualification source/config/pass mismatch for {backend}"
    );
    Ok(())
}
fn hash(p: &Path) -> anyhow::Result<String> {
    recur64_v5::stage::hash_file(p)
}

struct ArmFiles<'a> {
    name: &'static str,
    dir: &'a Path,
    result: &'a Path,
}
/// File-level provenance for one completed arm: every hash in the arm receipt must
/// equal the actual file, the checkpoint metadata must name this launch plan, and the
/// endpoint matrices must name the evaluated checkpoints.
fn verify_arm(
    a: &ArmFiles,
    plan: &objective::ProbePlan,
    m0: &probe_eval::Matrix,
    m200: &probe_eval::Matrix,
) -> anyhow::Result<()> {
    let r: Value = serde_json::from_slice(&std::fs::read(a.result)?)?;
    let meta: Value = serde_json::from_slice(&std::fs::read(a.dir.join("latest.json"))?)?;
    let h = |p: &str| hash(&a.dir.join(p));
    let hist = r["history"].as_array().map_or(0, |v| v.len());
    anyhow::ensure!(
        r["schema"] == "v6_objective_probe_arm_result_v2"
            && r["source_sha"] == recur64_v6::SOURCE
            && r["objective"] == objective::OBJECTIVE
            && r["arm"] == a.name
            && r["plan_digest"] == plan.digest
            && r["update"] == 200
            && r["baseline_exact"] == true
            && r["weights_reused"] == false
            && hist == 200
            && r["history"]
                .as_array()
                .unwrap()
                .iter()
                .all(|u| u["examples"] == 24)
            && r["endpoint_000_sha256"] == h("endpoint-000.json")?
            && r["endpoint_200_sha256"] == h("endpoint-200.json")?
            && r["model_update0_sha256"] == h("update-000/model.mpk")?
            && r["model_update200_sha256"] == h("update-200/model.mpk")?
            && r["optimizer_update200_sha256"] == h("update-200/optimizer.mpk")?
            && r["checkpoint_metadata_sha256"] == h("latest.json")?
            && m0.model_sha256 == r["model_update0_sha256"]
            && m200.model_sha256 == r["model_update200_sha256"]
            && meta["source"] == recur64_v6::SOURCE
            && meta["objective"] == objective::OBJECTIVE
            && meta["arm"] == a.name
            && meta["plan"] == plan.digest
            && meta["update"] == 200
            && meta["model_sha"] == r["model_update200_sha256"],
        "{} arm receipt/checkpoint/endpoint provenance mismatch",
        a.name
    );
    Ok(())
}
fn decide(
    plan_path: &Path,
    plan_binding: &Path,
    contract: &Path,
    control: ArmFiles,
    treatment: ArmFiles,
) -> anyhow::Result<Value> {
    let plan: objective::ProbePlan = serde_json::from_slice(&std::fs::read(plan_path)?)?;
    plan.validate()?;
    let b: Value = serde_json::from_slice(&std::fs::read(plan_binding)?)?;
    anyhow::ensure!(
        b["schema"] == "v6_objective_probe_plan_binding_v2"
            && b["source_sha"] == recur64_v6::SOURCE
            && b["plan_raw_sha256"] == hash(plan_path)?
            && b["plan_digest"] == plan.digest
            && plan.contract_sha256 == hash(contract)?,
        "launch plan binding/contract mismatch"
    );
    let read = |d: &Path, u: usize| -> anyhow::Result<probe_eval::Matrix> {
        Ok(serde_json::from_slice(&std::fs::read(
            d.join(format!("endpoint-{u:03}.json")),
        )?)?)
    };
    let (c0, c1) = (read(control.dir, 0)?, read(control.dir, 200)?);
    let (t0, t1) = (read(treatment.dir, 0)?, read(treatment.dir, 200)?);
    verify_arm(&control, &plan, &c0, &c1)?;
    verify_arm(&treatment, &plan, &t0, &t1)?;
    anyhow::ensure!(
        c0.model_sha256 == t0.model_sha256,
        "control/treatment update-0 readers are not bit-identical"
    );
    let mut v = probe_eval::decide(&plan, &[&c0, &c1], &[&t0, &t1])?;
    v["source_sha"] = json!(recur64_v6::SOURCE);
    v["launch_plan_digest"] = json!(plan.digest);
    v["initial_equality"] = json!({
        "control_update0_model_sha256": c0.model_sha256, "treatment_update0_model_sha256": t0.model_sha256,
        "equal": true, "equals_canonical_initial_file": c0.model_sha256 == plan.initial_model_sha256,
        "canonical_initial_model_sha256": plan.initial_model_sha256,
    });
    v["endpoint_hashes"] = json!({
        "control_000": hash(&control.dir.join("endpoint-000.json"))?, "control_200": hash(&control.dir.join("endpoint-200.json"))?,
        "treatment_000": hash(&treatment.dir.join("endpoint-000.json"))?, "treatment_200": hash(&treatment.dir.join("endpoint-200.json"))?,
    });
    Ok(v)
}

fn execute<B: AutodiffBackend>(c: &Cli, backend: &str) -> anyhow::Result<Value> {
    let device = Default::default();
    if let Cmd::Decide {
        plan,
        plan_binding,
        contract,
        control_dir,
        control_result,
        treatment_dir,
        treatment_result,
    } = &c.command
    {
        return decide(
            plan,
            plan_binding,
            contract,
            ArmFiles {
                name: "control",
                dir: control_dir,
                result: control_result,
            },
            ArmFiles {
                name: "treatment",
                dir: treatment_dir,
                result: treatment_result,
            },
        );
    }
    if let Cmd::Preregister {
        cpu_qualification,
        cuda_qualification,
        ..
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
    // Custody BEFORE any model construction: actual raw bytes and role refusals.
    let custody =
        recur64_v5::data::verify_local_boundaries(&c.data, &c.dev_custody, &c.confirm_custody)?;
    match &c.command {
        Cmd::Custody => return Ok(custody),
        Cmd::Identity => {
            return Ok(
                json!({"source_sha": recur64_v6::SOURCE, "objective": objective::OBJECTIVE,
                "launch_plan": objective::LAUNCH_PLAN, "config_digest": packet::config_digest()?,
                "capability": "objective probe TRAIN only"}),
            );
        }
        _ => {}
    }
    let data = recur64_v5::data::V5Data::load_train(&c.data)?;
    let base = packet::import_base::<B>(&c.stage_a, &device, backend)?;
    match &c.command {
        Cmd::Qualify {
            p0_plan,
            p0_binding,
            run_dir,
        } => {
            let p0 = objective::load_p0_plan(p0_plan, p0_binding)?;
            objective::verify_packets_against_data(&p0, &data)?;
            probe_qual::run(&base, &data, &p0, &device, backend, run_dir)
        }
        Cmd::Preregister {
            p0_plan,
            p0_binding,
            contract,
            initial_dir,
            plan_out,
            cpu_qualification,
            cuda_qualification,
        } => {
            anyhow::ensure!(
                backend == "cuda",
                "preregistration requires CUDA, no CPU substitute"
            );
            anyhow::ensure!(!plan_out.exists(), "refuse overwriting plan");
            let p0 = objective::load_p0_plan(p0_plan, p0_binding)?;
            objective::verify_packets_against_data(&p0, &data)?;
            let maps = recur64_v6::intervene::all_maps(&p0)?;
            let cells = objective::check_all_interventions(&p0, &maps)?;
            let init = objective::create_initial::<B>(initial_dir, &device)?;
            let plan = objective::build_plan(
                &p0,
                &maps,
                hash(contract)?,
                init["model_sha256"].as_str().unwrap().into(),
                init["parameter_digest"].as_str().unwrap().into(),
            )?;
            std::fs::write(plan_out, serde_json::to_vec_pretty(&plan)?)?;
            Ok(json!({
                "schema": "v6_objective_probe_plan_binding_v2", "source_sha": recur64_v6::SOURCE,
                "objective": objective::OBJECTIVE, "launch_plan": objective::LAUNCH_PLAN,
                "config_digest": packet::config_digest()?, "plan_raw_sha256": hash(plan_out)?,
                "plan_digest": plan.digest, "contract_sha256": plan.contract_sha256,
                "p0_plan_digest": plan.p0_plan_digest, "p0_plan_raw_sha256": plan.p0_plan_raw_sha256,
                "episode_digest": plan.episode_digest, "maps_digest": plan.maps_digest,
                "map_summary": plan.map_summary, "intervention_cells_verified": cells,
                "initial_model_sha256": plan.initial_model_sha256,
                "initial_parameter_digest": plan.initial_parameter_digest, "initial": init,
                "cpu_qualification_sha256": hash(cpu_qualification)?,
                "cuda_qualification_sha256": hash(cuda_qualification)?,
                "custody": custody, "optimizer_steps": 0,
            }))
        }
        Cmd::Learn {
            objective: arm,
            p0_plan,
            p0_binding,
            contract,
            plan,
            plan_binding,
            initial_dir,
            run_dir,
            cpu_qualification,
            cuda_qualification,
            max_minutes,
            resume,
        } => {
            anyhow::ensure!(
                backend == "cuda",
                "learnability requires CUDA, no CPU substitute"
            );
            let frozen =
                objective::load_frozen(p0_plan, p0_binding, plan, plan_binding, contract, &data)?;
            let b: Value = serde_json::from_slice(&std::fs::read(plan_binding)?)?;
            anyhow::ensure!(
                b["cpu_qualification_sha256"] == hash(cpu_qualification)?
                    && b["cuda_qualification_sha256"] == hash(cuda_qualification)?
                    && b["initial_model_sha256"] == frozen.plan.initial_model_sha256,
                "preregistered qualification/initial binding mismatch"
            );
            let objective_kind = match arm {
                ObjectiveArg::Control => objective::Objective::Control,
                ObjectiveArg::Treatment => objective::Objective::Treatment,
            };
            objective::learn(
                &base,
                &data,
                objective::Run {
                    objective: objective_kind,
                    frozen: &frozen,
                    initial: initial_dir,
                    dir: run_dir,
                    resume: *resume,
                    max_minutes: *max_minutes,
                },
                &device,
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
        "objective probe requires its isolated V6 branch"
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
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(move || {
            let result = match device {
                Device::Cpu => execute::<recur64_model::train::CpuTrainBackend>(&c, "cpu"),
                Device::Cuda => {
                    #[cfg(feature = "cuda")]
                    {
                        execute::<burn::backend::Autodiff<burn::backend::Cuda>>(&c, "cuda")
                    }
                    #[cfg(not(feature = "cuda"))]
                    {
                        Err(anyhow::anyhow!("CUDA binary unavailable, no CPU substitution"))
                    }
                }
            };
            let value = match &result {
                Ok(v) => v.clone(),
                Err(e) => json!({"schema": "v6_objective_probe_failure_v2", "source_sha": recur64_v6::SOURCE,
                                 "pass": false, "error": format!("{e:#}"), "fitting_authorized": false}),
            };
            if let Some(parent) = c.output.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&c.output, serde_json::to_vec_pretty(&value)?)?;
            result.map(|_| ())
        })?
        .join()
        .map_err(|_| anyhow::anyhow!("worker panic, preserve native logs; no next invocation"))?
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn no_dev_campaign_extra_update_or_seed_cli() {
        for forbidden in [
            "train",
            "evaluate",
            "self-play",
            "controller",
            "extra-loops",
            "dev",
            "confirm",
        ] {
            assert!(Cli::try_parse_from(["objective_probe", forbidden]).is_err());
        }
        let base = [
            "objective_probe",
            "--data",
            "d",
            "--dev-custody",
            "a",
            "--confirm-custody",
            "b",
            "--stage-a",
            "s",
            "--output",
            "o",
        ];
        for extra in [
            ["learn", "--updates"],
            ["learn", "--seed"],
            ["learn", "--objective"],
        ] {
            let args: Vec<&str> = base.iter().copied().chain(extra).chain(["801"]).collect();
            assert!(Cli::try_parse_from(args).is_err(), "{extra:?}");
        }
        // Only the two fixed objectives exist.
        let args: Vec<&str> = base
            .iter()
            .copied()
            .chain(["learn", "--objective", "baseline-training"])
            .collect();
        assert!(Cli::try_parse_from(args).is_err());
    }
}
