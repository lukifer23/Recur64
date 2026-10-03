//! CLI boundary for the distinct V5 research line.

use std::path::{Path, PathBuf};

use burn::prelude::*;
use clap::{Args, Subcommand, ValueEnum};
use recur64_core::GameState;
use recur64_v5::config::{ARCHITECTURE, V5Config};
use recur64_v5::data::V5Data;
use recur64_v5::graph::{AcquiredGraph, EpisodeKey, Schedule, acquire};
use recur64_v5::model::CounterfactualRelationalLoop;
use recur64_v5::stage::{Recipe, Stage, Trainer, baseline_fingerprint, load_finished_model};
use serde::Serialize;

#[derive(Subcommand)]
pub enum V5Cmd {
    /// Detect the local execution environment; this does not qualify a backend.
    Doctor(DoctorArgs),
    /// Print exact V5 model and contract identity.
    ModelInfo(ModelInfoArgs),
    /// Verify the one permitted P25 TRAIN artifact and inherited FIT/DEV partition.
    Custody(CustodyArgs),
    /// Generate or audit exact acquired-graph manifests.
    #[command(subcommand)]
    Graph(GraphCmd),
    /// Run one bounded, resumable chunk of the frozen Stage A or Stage B recipe.
    Train(TrainArgs),
    /// Exercise the actual paired graph over Q2/Q4/Q8 and R1/R2/R4.
    Qualify(QualifyArgs),
}

#[derive(Args)]
pub struct DoctorArgs {
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Args)]
pub struct ModelInfoArgs {
    #[arg(long)]
    json: Option<PathBuf>,
}

#[derive(Args)]
pub struct CustodyArgs {
    #[arg(long)]
    data: PathBuf,
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Clone, Copy, ValueEnum)]
enum ScheduleArg {
    UniformFrontier,
    BaseRankedDepth,
}

impl From<ScheduleArg> for Schedule {
    fn from(value: ScheduleArg) -> Self {
        match value {
            ScheduleArg::UniformFrontier => Schedule::UniformFrontierV1,
            ScheduleArg::BaseRankedDepth => Schedule::BaseRankedDepthV1,
        }
    }
}

#[derive(Subcommand)]
pub enum GraphCmd {
    Generate(GraphGenerateArgs),
    Audit(GraphAuditArgs),
}

#[derive(Args)]
pub struct GraphGenerateArgs {
    #[arg(long)]
    fen: String,
    #[arg(long)]
    position_id: String,
    #[arg(long, value_enum)]
    schedule: ScheduleArg,
    #[arg(long)]
    q: usize,
    #[arg(long, default_value_t = 5301)]
    seed: u64,
    #[arg(long, default_value_t = 0)]
    ordinal: u64,
    /// JSON array of frozen B0 logits; required only by base-ranked-depth.
    #[arg(long)]
    base_logits: Option<PathBuf>,
    #[arg(long)]
    output: PathBuf,
}

#[derive(Args)]
pub struct GraphAuditArgs {
    #[arg(long)]
    graph: PathBuf,
}

#[derive(Clone, Copy, ValueEnum)]
enum DeviceArg {
    Cpu,
    Cuda,
}

#[derive(Clone, Copy, ValueEnum)]
enum StageArg {
    A,
    B,
}

#[derive(Args)]
pub struct TrainArgs {
    #[arg(long, value_enum)]
    stage: StageArg,
    #[arg(long, value_enum)]
    device: DeviceArg,
    #[arg(long)]
    data: PathBuf,
    #[arg(long)]
    run_dir: PathBuf,
    /// Completed Stage A run directory; required for Stage B.
    #[arg(long)]
    stage_a: Option<PathBuf>,
    /// Qualified physical layout (only 2, or the pre-authorized fallback 1).
    #[arg(long, default_value_t = 2)]
    microbatch: usize,
    /// Resume the latest complete generation in this run directory.
    #[arg(long)]
    resume: bool,
    /// Deterministic wall-time boundary for this invocation.
    #[arg(long, default_value_t = 45)]
    max_minutes: u64,
    /// JSON qualification report for this device/layout.
    #[arg(long)]
    qualification: PathBuf,
}

#[derive(Args)]
pub struct QualifyArgs {
    #[arg(long, value_enum)]
    device: DeviceArg,
    #[arg(long, default_value_t = 2)]
    microbatch: usize,
    #[arg(long)]
    output: PathBuf,
}

#[derive(Serialize)]
struct ModelInfo {
    architecture: &'static str,
    scientific_config_digest: String,
    parameters: usize,
    parameter_bytes_fp32: usize,
    parameter_breakdown: Vec<(&'static str, usize)>,
    contracts: recur64_v5::config::Contracts,
    paired_core_block_applications_per_example: &'static str,
}

fn write_json(path: &Path, value: &impl Serialize) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(value)?)?;
    std::fs::rename(tmp, path)?;
    Ok(())
}

fn model_info(a: ModelInfoArgs) -> anyhow::Result<()> {
    type B = burn::backend::Flex;
    let device = Default::default();
    <B as Backend>::seed(&device, 5301);
    let cfg = V5Config::default();
    let model = CounterfactualRelationalLoop::<B>::new(cfg.clone(), &device);
    let info = ModelInfo {
        architecture: ARCHITECTURE,
        scientific_config_digest: cfg.scientific_digest()?,
        parameters: model.num_params(),
        parameter_bytes_fp32: model.num_params() * 4,
        parameter_breakdown: model.param_breakdown(),
        contracts: cfg.contracts,
        paired_core_block_applications_per_example: "4R",
    };
    println!("architecture : {}", info.architecture);
    println!("parameters   : {}", info.parameters);
    println!("fp32 bytes   : {}", info.parameter_bytes_fp32);
    println!("config       : {}", info.scientific_config_digest);
    for (name, count) in &info.parameter_breakdown {
        println!("  {name:<24} {count}");
    }
    println!("paired shared-core applications: 4R");
    if let Some(path) = a.json {
        write_json(&path, &info)?;
    }
    Ok(())
}

fn custody(a: CustodyArgs) -> anyhow::Result<()> {
    let data = V5Data::load(&a.data)?;
    let report = serde_json::json!({
        "schema": "v5_custody_report_v1",
        "source": a.data,
        "train_digest": data.targets.digest,
        "positions": data.targets.positions.len(),
        "fit": data.fit.len(),
        "dev": data.dev.len(),
        "fit_digest": recur64_v5::data::FIT_DIGEST,
        "dev_digest": recur64_v5::data::DEV_DIGEST,
        "canonical_disjoint": true,
        "sealed_inputs_evaluated": false
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    if let Some(path) = a.output {
        write_json(&path, &report)?;
    }
    Ok(())
}

fn graph(c: GraphCmd) -> anyhow::Result<()> {
    match c {
        GraphCmd::Generate(a) => {
            let root = GameState::from_fen(&a.fen)
                .map_err(|e| anyhow::anyhow!("invalid root FEN: {e:?}"))?;
            let logits: Option<Vec<f32>> = a
                .base_logits
                .as_deref()
                .map(|p| -> anyhow::Result<_> { Ok(serde_json::from_slice(&std::fs::read(p)?)?) })
                .transpose()?;
            let manifest = acquire(
                &root,
                EpisodeKey {
                    position_id: a.position_id,
                    schedule: a.schedule.into(),
                    run_seed: a.seed,
                    occurrence_ordinal: a.ordinal,
                },
                a.q,
                logits.as_deref(),
            )?;
            write_json(&a.output, &manifest)?;
            println!(
                "graph {}: requested Q{}, actual Q{}, digest {}",
                manifest.episode.schedule.id(),
                manifest.requested_q,
                manifest.actual_q,
                manifest.digest
            );
            Ok(())
        }
        GraphCmd::Audit(a) => {
            let graph: AcquiredGraph = serde_json::from_slice(&std::fs::read(&a.graph)?)?;
            graph.verify()?;
            println!(
                "valid {} Q{} graph {}, exact transitions {}",
                graph.episode.schedule.id(),
                graph.actual_q,
                graph.digest,
                graph.successful_queries
            );
            Ok(())
        }
    }
}

fn doctor(a: DoctorArgs) -> anyhow::Result<()> {
    let report = serde_json::json!({
        "schema": "v5_hp_doctor_v1",
        "status": "detected_not_tested",
        "architecture": ARCHITECTURE,
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "logical_threads": std::thread::available_parallelism()?.get(),
        "cuda_user_space": std::env::var_os("LOCALAPPDATA")
            .map(PathBuf::from)
            .map(|p| p.join("Recur64").join("cuda").join("12.9.1"))
            .filter(|p| p.is_dir()),
        "precision": "fp32",
        "tested": false
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    if let Some(path) = a.output {
        write_json(&path, &report)?;
    }
    Ok(())
}

fn git_sha() -> anyhow::Result<String> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let sha = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(&root)
        .output()?;
    anyhow::ensure!(sha.status.success(), "cannot resolve source Git SHA");
    Ok(String::from_utf8(sha.stdout)?.trim().to_owned())
}

fn source_sha() -> anyhow::Result<String> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let sha = git_sha()?;
    let code_clean = std::process::Command::new("git")
        .args([
            "diff",
            "--quiet",
            "--",
            "crates",
            "Cargo.toml",
            "Cargo.lock",
            "configs",
        ])
        .current_dir(root)
        .status()?;
    anyhow::ensure!(
        code_clean.success(),
        "tracked code/config changes are uncommitted; scientific training is refused"
    );
    Ok(sha)
}

fn require_empty_new_run(path: &Path) -> anyhow::Result<()> {
    if path.exists() {
        anyhow::ensure!(
            std::fs::read_dir(path)?.next().is_none(),
            "{} is not empty; use --resume or a fresh run directory",
            path.display()
        );
    }
    Ok(())
}

fn qualification_seconds(path: &Path, device: DeviceArg, microbatch: usize) -> anyhow::Result<f64> {
    let value: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
    anyhow::ensure!(
        value["schema"] == "v5_qualification_report_v1"
            && value["pass"] == true
            && value["precision"] == "fp32"
            && value["microbatch"] == microbatch as u64,
        "{} is not a passing V5 qualification for microbatch {}",
        path.display(),
        microbatch
    );
    let wanted = match device {
        DeviceArg::Cpu => "cpu",
        DeviceArg::Cuda => "cuda",
    };
    anyhow::ensure!(
        value["device"] == wanted,
        "qualification device is not {wanted}"
    );
    value["worst_update_warm_seconds"]
        .as_f64()
        .filter(|x| x.is_finite() && *x > 0.0)
        .ok_or_else(|| anyhow::anyhow!("qualification has no valid warm update timing"))
}

fn train_backend<B>(a: &TrainArgs, backend: &str) -> anyhow::Result<()>
where
    B: burn::tensor::backend::AutodiffBackend,
    B::Device: Default,
{
    anyhow::ensure!(
        (1..=45).contains(&a.max_minutes),
        "--max-minutes must be in 1..=45"
    );
    let seconds_per_update = qualification_seconds(&a.qualification, a.device, a.microbatch)?;
    let data = V5Data::load(&a.data)?;
    data.verify_custody()?;
    let source = source_sha()?;
    let device = B::Device::default();
    let (recipe, initial_model) = match a.stage {
        StageArg::A => {
            anyhow::ensure!(a.stage_a.is_none(), "--stage-a is invalid for Stage A");
            <B as Backend>::seed(&device, recur64_v5::stage::PILOT_SEED);
            (
                Recipe::stage_a(source, a.microbatch)?,
                Some(CounterfactualRelationalLoop::<B>::new(
                    V5Config::default(),
                    &device,
                )),
            )
        }
        StageArg::B => {
            let stage_a = a
                .stage_a
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("Stage B requires --stage-a"))?;
            let (model, meta) = load_finished_model::<B>(stage_a, Stage::BaselineA, &device)?;
            let fingerprint = baseline_fingerprint(&model, &device)?;
            (
                Recipe::stage_b(source, a.microbatch, meta.model_hash, fingerprint)?,
                Some(model),
            )
        }
    };
    let mut trainer = if a.resume {
        Trainer::<B>::load_latest(&a.run_dir, recipe.clone(), &device)?
    } else {
        require_empty_new_run(&a.run_dir)?;
        Trainer::new(recipe.clone(), initial_model.expect("new run model"))?
    };
    let remaining = recipe.updates - trainer.updates_done;
    println!(
        "projected remaining work: {} updates, {:.2} hours at qualified {:.3}s/update",
        remaining,
        remaining as f64 * seconds_per_update / 3600.0,
        seconds_per_update
    );
    let deadline = std::time::Duration::from_secs(a.max_minutes * 60);
    let started = std::time::Instant::now();
    while trainer.updates_done < recipe.updates {
        if started.elapsed() >= deadline {
            break;
        }
        let record = trainer.step(&data, &device)?;
        println!(
            "{} update {}/{} loss {:.6} lr {:.8} wall {:.2}s",
            recipe.stage.label(),
            trainer.updates_done,
            recipe.updates,
            record.loss,
            record.lr,
            record.wall_seconds
        );
        if trainer.updates_done % 10 == 0 || trainer.updates_done == recipe.updates {
            let hash = trainer.save(&a.run_dir, backend, &device)?;
            println!("checkpoint update {} model {}", trainer.updates_done, hash);
        }
    }
    if trainer.updates_done % 10 != 0 && trainer.updates_done != 0 {
        let hash = trainer.save(&a.run_dir, backend, &device)?;
        println!(
            "chunk checkpoint update {} model {}",
            trainer.updates_done, hash
        );
    }
    println!(
        "{} stopped at update {}/{} after {:.1}s",
        recipe.stage.label(),
        trainer.updates_done,
        recipe.updates,
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

fn train(a: TrainArgs) -> anyhow::Result<()> {
    match a.device {
        DeviceArg::Cpu => train_backend::<recur64_model::train::CpuTrainBackend>(&a, "cpu"),
        DeviceArg::Cuda => {
            #[cfg(feature = "cuda")]
            {
                train_backend::<burn::backend::Autodiff<burn::backend::Cuda>>(&a, "cuda")
            }
            #[cfg(not(feature = "cuda"))]
            {
                anyhow::bail!(
                    "CUDA support is not compiled; rebuild with --features cuda (no CPU substitution)"
                )
            }
        }
    }
}

fn qualify_backend<B>(a: &QualifyArgs, device_label: &str) -> anyhow::Result<()>
where
    B: burn::tensor::backend::AutodiffBackend,
    B::Device: Default,
{
    let device = B::Device::default();
    let source_sha = git_sha()?;
    let report =
        recur64_v5::qualification::run::<B>(&source_sha, device_label, a.microbatch, &device)?;
    write_json(&a.output, &report)?;
    println!(
        "V5 {device_label} qualification: pass={} microbatch={} params={} null_error={:e} warm_worst={:.3}s",
        report.pass,
        report.microbatch,
        report.parameters,
        report.null_centered_max_abs,
        report.worst_update_warm_seconds
    );
    anyhow::ensure!(
        report.pass,
        "V5 qualification failed; see {}",
        a.output.display()
    );
    Ok(())
}

fn qualify(a: QualifyArgs) -> anyhow::Result<()> {
    match a.device {
        DeviceArg::Cpu => qualify_backend::<recur64_model::train::CpuTrainBackend>(&a, "cpu"),
        DeviceArg::Cuda => {
            #[cfg(feature = "cuda")]
            {
                qualify_backend::<burn::backend::Autodiff<burn::backend::Cuda>>(&a, "cuda")
            }
            #[cfg(not(feature = "cuda"))]
            {
                anyhow::bail!(
                    "CUDA support is not compiled; rebuild with --features cuda (no CPU substitution)"
                )
            }
        }
    }
}

pub fn run(cmd: V5Cmd) -> anyhow::Result<()> {
    match cmd {
        V5Cmd::Doctor(a) => doctor(a),
        V5Cmd::ModelInfo(a) => model_info(a),
        V5Cmd::Custody(a) => custody(a),
        V5Cmd::Graph(c) => graph(c),
        V5Cmd::Train(a) => big_stack("recur64-v5-train", move || train(a)),
        V5Cmd::Qualify(a) => big_stack("recur64-v5-qualify", move || qualify(a)),
    }
}

fn big_stack(
    name: &str,
    task: impl FnOnce() -> anyhow::Result<()> + Send + 'static,
) -> anyhow::Result<()> {
    std::thread::Builder::new()
        .name(name.into())
        .stack_size(64 * 1024 * 1024)
        .spawn(task)?
        .join()
        .map_err(|_| anyhow::anyhow!("{name} worker panicked"))?
}
