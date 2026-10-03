//! CLI boundary for the distinct V5 research line.

use std::path::{Path, PathBuf};

use burn::module::AutodiffModule;
use burn::prelude::*;
use clap::{Args, Subcommand, ValueEnum};
use recur64_core::GameState;
use recur64_v5::config::{ARCHITECTURE, V5Config};
use recur64_v5::data::V5Data;
use recur64_v5::evaluation::{
    EvaluationBundle, ablation_report, classify_pilot, merge_cell_bundles,
};
use recur64_v5::graph::{AcquiredGraph, EpisodeKey, Schedule, acquire};
use recur64_v5::model::CounterfactualRelationalLoop;
use recur64_v5::stage::{
    Recipe, Stage, Trainer, baseline_fingerprint, load_finished_model, load_model_at,
};
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
    /// Run the disposable 24-position reader optimization drill.
    Drill(DrillArgs),
    /// Evaluate one inherited DEV family/depth cell at update 0 or 800.
    Evaluate(EvaluateArgs),
    /// Record final Stage A B0 on all inherited DEV positions, at Q0.
    EvaluateBaseline(EvaluateBaselineArgs),
    /// Merge the six deterministic DEV cell shards into the complete matrix.
    EvalMerge(EvalMergeArgs),
    /// Recompute treatment and composition summaries from per-position evidence.
    Ablation(AblationArgs),
    /// Apply the frozen bootstrap gates to the complete update-800 evaluation.
    PilotReport(PilotReportArgs),
    /// Run the conditionally authorized primary-cell Q8/R8 forward diagnostic.
    ExtraLoops(ExtraLoopsArgs),
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
    /// Passing Q8 engineering drill report.
    #[arg(long)]
    drill: PathBuf,
    /// Conditional Q16 diagnostic; permitted only after informative Q8 failure.
    #[arg(long)]
    drill_q16: Option<PathBuf>,
    /// Final all-DEV Stage A B0 report; required for Stage B.
    #[arg(long)]
    stage_a_evaluation: Option<PathBuf>,
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

#[derive(Args)]
pub struct DrillArgs {
    #[arg(long, value_enum)]
    device: DeviceArg,
    #[arg(long)]
    data: PathBuf,
    #[arg(long, default_value_t = 8)]
    q: usize,
    /// Required for Q16 and must be the failed Q8 report.
    #[arg(long)]
    q8_report: Option<PathBuf>,
    #[arg(long, default_value_t = 2)]
    microbatch: usize,
    #[arg(long)]
    qualification: PathBuf,
    #[arg(long)]
    output: PathBuf,
}

#[derive(Args)]
pub struct EvaluateArgs {
    #[arg(long, value_enum)]
    device: DeviceArg,
    #[arg(long)]
    data: PathBuf,
    #[arg(long)]
    stage_b: PathBuf,
    #[arg(long)]
    update: u64,
    #[arg(long)]
    family: String,
    #[arg(long)]
    mate_depth: u8,
    #[arg(long, default_value_t = 2)]
    microbatch: usize,
    #[arg(long)]
    qualification: PathBuf,
    #[arg(long)]
    output: PathBuf,
}

#[derive(Args)]
pub struct EvaluateBaselineArgs {
    #[arg(long, value_enum)]
    device: DeviceArg,
    #[arg(long)]
    data: PathBuf,
    #[arg(long)]
    stage_a: PathBuf,
    #[arg(long, default_value_t = 2)]
    microbatch: usize,
    #[arg(long)]
    qualification: PathBuf,
    #[arg(long)]
    output: PathBuf,
}

#[derive(Args)]
pub struct EvalMergeArgs {
    #[arg(long, required = true, num_args = 6)]
    input: Vec<PathBuf>,
    #[arg(long)]
    output: PathBuf,
}

#[derive(Args)]
pub struct AblationArgs {
    #[arg(long)]
    evaluation: PathBuf,
    #[arg(long)]
    output: PathBuf,
}

#[derive(Args)]
pub struct PilotReportArgs {
    #[arg(long)]
    evaluation: PathBuf,
    #[arg(long)]
    cpu_qualification: PathBuf,
    #[arg(long)]
    cuda_qualification: PathBuf,
    #[arg(long)]
    drill: PathBuf,
    #[arg(long)]
    drill_q16: Option<PathBuf>,
    #[arg(long)]
    output: PathBuf,
}

#[derive(Args)]
pub struct ExtraLoopsArgs {
    #[arg(long, value_enum)]
    device: DeviceArg,
    #[arg(long)]
    data: PathBuf,
    #[arg(long)]
    stage_b: PathBuf,
    #[arg(long)]
    evaluation: PathBuf,
    #[arg(long)]
    pilot_report: PathBuf,
    #[arg(long, default_value_t = 2)]
    microbatch: usize,
    #[arg(long)]
    qualification: PathBuf,
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

fn git_sha(root: &Path) -> anyhow::Result<String> {
    let sha = std::process::Command::new("git")
        .args([
            "log",
            "-1",
            "--format=%H",
            "--",
            "crates",
            "Cargo.toml",
            "Cargo.lock",
            "configs",
        ])
        .current_dir(root)
        .output()?;
    anyhow::ensure!(
        sha.status.success(),
        "cannot resolve scientific source Git SHA"
    );
    Ok(String::from_utf8(sha.stdout)?.trim().to_owned())
}

fn source_sha() -> anyhow::Result<String> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    checked_source_sha(&root, env!("RECUR64_V5_BUILD_SOURCE_SHA"))
}

fn checked_source_sha(root: &Path, built_sha: &str) -> anyhow::Result<String> {
    let sha = git_sha(root)?;
    let code_state = std::process::Command::new("git")
        .args([
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            "crates",
            "Cargo.toml",
            "Cargo.lock",
            "configs",
        ])
        .current_dir(root)
        .output()?;
    anyhow::ensure!(
        code_state.status.success() && code_state.stdout.is_empty(),
        "code/config changes are uncommitted (including staged/untracked files); V5 scientific execution is refused"
    );
    anyhow::ensure!(
        sha == built_sha,
        "V5 executable was built from {}; current scientific source is {sha}; rebuild before scientific execution",
        built_sha
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

fn qualification_seconds(
    path: &Path,
    device: DeviceArg,
    microbatch: usize,
    source_sha: &str,
) -> anyhow::Result<f64> {
    let value: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
    let config_digest = V5Config::default().scientific_digest()?;
    anyhow::ensure!(
        value["schema"] == "v5_qualification_report_v1"
            && value["pass"] == true
            && value["precision"] == "fp32"
            && value["microbatch"] == microbatch as u64
            && value["source_sha"] == source_sha
            && value["architecture"] == ARCHITECTURE
            && value["config_digest"] == config_digest,
        "{} is not a passing V5 qualification for microbatch {}",
        path.display(),
        microbatch
    );
    require_synchronized_qualification(&value)?;
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
    let source = source_sha()?;
    let seconds_per_physical_update =
        qualification_seconds(&a.qualification, a.device, a.microbatch, &source)?;
    let data = V5Data::load(&a.data)?;
    data.verify_custody()?;
    let drill: recur64_v5::drill::DrillReport = serde_json::from_slice(&std::fs::read(&a.drill)?)?;
    let diagnostic: Option<recur64_v5::drill::DrillReport> = a
        .drill_q16
        .as_ref()
        .map(|path| -> anyhow::Result<recur64_v5::drill::DrillReport> {
            Ok(serde_json::from_slice(&std::fs::read(path)?)?)
        })
        .transpose()?;
    recur64_v5::drill::validated_pilot_prerequisite(
        &drill,
        diagnostic.as_ref(),
        &source,
        &V5Config::default().scientific_digest()?,
        a.microbatch,
    )?;
    let expected_drill_ids: Vec<_> = recur64_v5::drill::select_positions(&data)?
        .iter()
        .map(|&index| data.position(index).id.clone())
        .collect();
    anyhow::ensure!(
        drill.selected_position_ids == expected_drill_ids,
        "drill positions differ from the preregistered stable FIT selection"
    );
    let device = B::Device::default();
    let (recipe, initial_model) = match a.stage {
        StageArg::A => {
            anyhow::ensure!(a.stage_a.is_none(), "--stage-a is invalid for Stage A");
            anyhow::ensure!(
                a.stage_a_evaluation.is_none(),
                "--stage-a-evaluation is invalid for Stage A"
            );
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
            anyhow::ensure!(
                meta.recipe.source_sha == source && meta.backend == backend,
                "Stage A source/device differs from Stage B"
            );
            let fingerprint = baseline_fingerprint(&model, &device)?;
            let baseline_path = a.stage_a_evaluation.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "Stage B requires --stage-a-evaluation from the final Stage A checkpoint"
                )
            })?;
            let baseline: recur64_v5::study::BaselineEvaluation =
                serde_json::from_slice(&std::fs::read(baseline_path)?)?;
            baseline.validate_against_data(&data)?;
            anyhow::ensure!(
                baseline.model_hash == meta.model_hash
                    && baseline.source_sha == source
                    && baseline.baseline_fingerprint == fingerprint
                    && baseline.microbatch == a.microbatch
                    && baseline.device == backend,
                "Stage A baseline report differs from its checkpoint/device/layout"
            );
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
    if !a.resume && recipe.stage == Stage::ReaderB {
        let hash = trainer.save(&a.run_dir, backend, &device)?;
        println!("checkpoint update 0 model {hash}");
    }
    let remaining = recipe.updates - trainer.updates_done;
    let projected_seconds_per_update =
        seconds_per_physical_update * recipe.accumulation_steps as f64;
    println!(
        "projected remaining work: {} optimizer updates, {:.2} hours at conservative {:.3}s/update ({} x {:.3}s qualified physical updates)",
        remaining,
        remaining as f64 * projected_seconds_per_update / 3600.0,
        projected_seconds_per_update,
        recipe.accumulation_steps,
        seconds_per_physical_update
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
    let source_sha = source_sha()?;
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

fn drill_backend<B>(a: &DrillArgs, backend: &str) -> anyhow::Result<()>
where
    B: burn::tensor::backend::AutodiffBackend,
    B::Device: Default,
{
    let source = source_sha()?;
    let _ = qualification_seconds(&a.qualification, a.device, a.microbatch, &source)?;
    anyhow::ensure!(matches!(a.q, 8 | 16), "--q must be 8 or the conditional 16");
    if a.q == 16 {
        let q8_path = a
            .q8_report
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("Q16 requires --q8-report"))?;
        let prior: recur64_v5::drill::DrillReport =
            serde_json::from_slice(&std::fs::read(q8_path)?)?;
        anyhow::ensure!(
            prior.schema == recur64_v5::drill::DRILL_SCHEMA
                && prior.q == 8
                && !prior.pass
                && !prior.initial_below_point_zero_five
                && prior.finite_training
                && prior.baseline_exact
                && prior.source_sha == source,
            "Q16 is authorized only after a finite, informative Q8 drill failure from this source"
        );
    } else {
        anyhow::ensure!(a.q8_report.is_none(), "--q8-report is valid only for Q16");
    }
    let data = V5Data::load(&a.data)?;
    let device = B::Device::default();
    let report = recur64_v5::drill::run::<B>(source, &data, a.q, a.microbatch, &device)?;
    write_json(&a.output, &report)?;
    println!(
        "V5 {backend} Q{} drill: {} loss {:.6} -> {:.6} ({:.2}% reduction), action changes {}",
        report.q,
        report.classification,
        report.initial_mean_set_loss,
        report.final_mean_set_loss,
        100.0 * report.relative_loss_reduction,
        report.action_changes
    );
    Ok(())
}

fn drill(a: DrillArgs) -> anyhow::Result<()> {
    match a.device {
        DeviceArg::Cpu => drill_backend::<recur64_model::train::CpuTrainBackend>(&a, "cpu"),
        DeviceArg::Cuda => {
            #[cfg(feature = "cuda")]
            {
                drill_backend::<burn::backend::Autodiff<burn::backend::Cuda>>(&a, "cuda")
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

fn evaluate_backend<B>(a: &EvaluateArgs, backend: &str) -> anyhow::Result<()>
where
    B: burn::tensor::backend::AutodiffBackend,
    B::Device: Default,
{
    anyhow::ensure!(
        matches!(a.update, 0 | 800),
        "reader evaluation is fixed to update 0 or 800"
    );
    anyhow::ensure!(
        matches!(a.family.as_str(), "KQRvK" | "KRRvK") && matches!(a.mate_depth, 1..=3),
        "evaluation cell must be KQRvK/KRRvK and M1/M2/M3"
    );
    let source = source_sha()?;
    let _ = qualification_seconds(&a.qualification, a.device, a.microbatch, &source)?;
    let data = V5Data::load(&a.data)?;
    let indices: Vec<usize> = data
        .dev
        .iter()
        .copied()
        .filter(|&index| {
            data.position(index).family == a.family
                && data.position(index).mate_depth == a.mate_depth
        })
        .collect();
    anyhow::ensure!(!indices.is_empty(), "selected inherited DEV cell is empty");
    if a.family == "KQRvK" && a.mate_depth == 3 {
        anyhow::ensure!(
            indices.len() == 507,
            "primary KQRvK M3 cell must contain 507 positions"
        );
    }
    let device = B::Device::default();
    let (model, meta) = load_model_at::<B>(&a.stage_b, Stage::ReaderB, a.update, &device)?;
    anyhow::ensure!(
        meta.recipe.source_sha == source
            && meta.config_digest == V5Config::default().scientific_digest()?,
        "evaluation source/config differs from the checkpoint recipe"
    );
    let identity = recur64_v5::study::EvaluationIdentity {
        source_sha: source,
        config_digest: meta.config_digest,
        model_hash: meta.model_hash,
        final_update: a.update,
        scope: format!("{}_M{}", a.family, a.mate_depth),
        split: "inherited_v4_dev".into(),
        microbatch: a.microbatch,
        device: backend.into(),
    };
    let bundle =
        recur64_v5::study::evaluate_reader(&model.valid(), &data, &indices, identity, &device)?;
    write_json(&a.output, &bundle)?;
    println!(
        "V5 {backend} evaluation {} update {}: {} positions, {} records, graph {}",
        bundle.scope,
        bundle.final_update,
        indices.len(),
        bundle.records.len(),
        bundle.graph_manifest_hash
    );
    Ok(())
}

fn evaluate(a: EvaluateArgs) -> anyhow::Result<()> {
    match a.device {
        DeviceArg::Cpu => evaluate_backend::<recur64_model::train::CpuTrainBackend>(&a, "cpu"),
        DeviceArg::Cuda => {
            #[cfg(feature = "cuda")]
            {
                evaluate_backend::<burn::backend::Autodiff<burn::backend::Cuda>>(&a, "cuda")
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

fn evaluate_baseline_backend<B>(a: &EvaluateBaselineArgs, backend: &str) -> anyhow::Result<()>
where
    B: burn::tensor::backend::AutodiffBackend,
    B::Device: Default,
{
    let source = source_sha()?;
    let qualified_seconds =
        qualification_seconds(&a.qualification, a.device, a.microbatch, &source)?;
    let data = V5Data::load(&a.data)?;
    let projected = data.dev.len().div_ceil(a.microbatch) as f64 * qualified_seconds;
    println!(
        "projected final B0 evaluation: {} DEV positions, conservative {:.2} minutes",
        data.dev.len(),
        projected / 60.0
    );
    anyhow::ensure!(
        projected <= 45.0 * 60.0,
        "baseline projection exceeds one bounded process; review sharding before launch"
    );
    let device = B::Device::default();
    let (model, meta) = load_finished_model::<B>(&a.stage_a, Stage::BaselineA, &device)?;
    anyhow::ensure!(
        meta.recipe.source_sha == source
            && meta.backend == backend
            && meta.recipe.physical_microbatch == a.microbatch,
        "final baseline source/device/layout mismatch"
    );
    let fingerprint = baseline_fingerprint(&model, &device)?;
    let identity = recur64_v5::study::EvaluationIdentity {
        source_sha: source,
        config_digest: meta.config_digest,
        model_hash: meta.model_hash,
        final_update: meta.update,
        scope: "all_dev_4403".into(),
        split: "inherited_v4_dev".into(),
        microbatch: a.microbatch,
        device: backend.into(),
    };
    let report = recur64_v5::study::evaluate_final_baseline(
        &model.valid(),
        &data,
        identity,
        fingerprint,
        &device,
    )?;
    write_json(&a.output, &report)?;
    for family in ["KQRvK", "KRRvK"] {
        for depth in 1..=3 {
            let rows: Vec<_> = report
                .records
                .iter()
                .filter(|row| row.family == family && row.mate_depth == depth)
                .collect();
            let n = rows.len();
            println!(
                "B0 {family} M{depth} n={n} top1={:.8} set_loss={:.8}",
                rows.iter().map(|row| row.metrics.top1).sum::<f64>() / n as f64,
                rows.iter().map(|row| row.metrics.set_loss).sum::<f64>() / n as f64
            );
        }
    }
    Ok(())
}

fn evaluate_baseline(a: EvaluateBaselineArgs) -> anyhow::Result<()> {
    match a.device {
        DeviceArg::Cpu => {
            evaluate_baseline_backend::<recur64_model::train::CpuTrainBackend>(&a, "cpu")
        }
        DeviceArg::Cuda => {
            #[cfg(feature = "cuda")]
            {
                evaluate_baseline_backend::<burn::backend::Autodiff<burn::backend::Cuda>>(
                    &a, "cuda",
                )
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

fn eval_merge(a: EvalMergeArgs) -> anyhow::Result<()> {
    let bundles: Vec<EvaluationBundle> = a
        .input
        .iter()
        .map(|path| serde_json::from_slice(&std::fs::read(path)?).map_err(Into::into))
        .collect::<anyhow::Result<_>>()?;
    let merged = merge_cell_bundles(bundles)?;
    write_json(&a.output, &merged)?;
    println!(
        "merged {} records over all 4,403 DEV positions; graph {}",
        merged.records.len(),
        merged.graph_manifest_hash
    );
    Ok(())
}

fn ablation(a: AblationArgs) -> anyhow::Result<()> {
    let bundle: EvaluationBundle = serde_json::from_slice(&std::fs::read(&a.evaluation)?)?;
    let report = ablation_report(&bundle)?;
    write_json(&a.output, &report)?;
    println!(
        "wrote {} ablation cells and {} composition contrasts",
        report.summaries.len(),
        report.composition.len()
    );
    Ok(())
}

fn qualifying_report(path: &Path, device: &str, bundle: &EvaluationBundle) -> anyhow::Result<()> {
    let value: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
    anyhow::ensure!(
        value["schema"] == "v5_qualification_report_v1"
            && value["pass"] == true
            && value["device"] == device
            && value["precision"] == "fp32"
            && value["source_sha"] == bundle.source_sha
            && value["config_digest"] == bundle.config_digest
            && value["microbatch"] == bundle.microbatch as u64,
        "{} is not the matching passing {device} qualification",
        path.display()
    );
    require_synchronized_qualification(&value)?;
    Ok(())
}

fn require_synchronized_qualification(value: &serde_json::Value) -> anyhow::Result<()> {
    anyhow::ensure!(
        value["timing_contract"] == recur64_v5::profile::CONTRACT
            && value["timing_contract_digest"] == recur64_v5::profile::contract_digest()
            && value["profile_outputs_and_all_gradients_exact"] == true
            && value["profile_adamw_parameters_and_moments_exact"] == true
            && value["profile_phase_accounting_consistent"] == true,
        "V5 qualification lacks current synchronized accounting and exact profiling parity"
    );
    Ok(())
}

fn pilot_report(a: PilotReportArgs) -> anyhow::Result<()> {
    let bundle: EvaluationBundle = serde_json::from_slice(&std::fs::read(&a.evaluation)?)?;
    anyhow::ensure!(
        bundle.scope == "all_dev_4403" && bundle.final_update == 800,
        "pilot classification requires the complete fixed update-800 DEV evaluation"
    );
    qualifying_report(&a.cpu_qualification, "cpu", &bundle)?;
    qualifying_report(&a.cuda_qualification, "cuda", &bundle)?;
    let drill: recur64_v5::drill::DrillReport = serde_json::from_slice(&std::fs::read(&a.drill)?)?;
    let diagnostic: Option<recur64_v5::drill::DrillReport> = a
        .drill_q16
        .as_ref()
        .map(|path| -> anyhow::Result<recur64_v5::drill::DrillReport> {
            Ok(serde_json::from_slice(&std::fs::read(path)?)?)
        })
        .transpose()?;
    let engineering_pass = recur64_v5::drill::validated_pilot_prerequisite(
        &drill,
        diagnostic.as_ref(),
        &bundle.source_sha,
        &bundle.config_digest,
        bundle.microbatch,
    )
    .is_ok()
        && bundle.normal_replay_exact;
    let report = classify_pilot(&bundle, engineering_pass, true)?;
    write_json(&a.output, &report)?;
    println!("V5 pilot classification: {}", report.classification);
    Ok(())
}

fn extra_loops_backend<B>(a: &ExtraLoopsArgs, backend: &str) -> anyhow::Result<()>
where
    B: burn::tensor::backend::AutodiffBackend,
    B::Device: Default,
{
    let gate: recur64_v5::evaluation::PilotClassification =
        serde_json::from_slice(&std::fs::read(&a.pilot_report)?)?;
    anyhow::ensure!(
        gate.schema == recur64_v5::evaluation::REPORT_SCHEMA
            && gate.classification == "PILOT_CANDIDATE",
        "R8 is authorized only after all primary pilot gates pass"
    );
    let main: EvaluationBundle = serde_json::from_slice(&std::fs::read(&a.evaluation)?)?;
    anyhow::ensure!(
        main.scope == "all_dev_4403" && main.final_update == 800,
        "R8 requires the complete update-800 evaluation"
    );
    let source = source_sha()?;
    let _ = qualification_seconds(&a.qualification, a.device, a.microbatch, &source)?;
    let data = V5Data::load(&a.data)?;
    let indices: Vec<usize> = data
        .dev
        .iter()
        .copied()
        .filter(|&index| {
            data.position(index).family == "KQRvK" && data.position(index).mate_depth == 3
        })
        .collect();
    let device = B::Device::default();
    let (model, meta) = load_model_at::<B>(&a.stage_b, Stage::ReaderB, 800, &device)?;
    anyhow::ensure!(
        meta.model_hash == main.model_hash
            && meta.recipe.source_sha == source
            && main.source_sha == source,
        "R8 checkpoint/main evaluation/source identity mismatch"
    );
    let identity = recur64_v5::study::EvaluationIdentity {
        source_sha: source,
        config_digest: meta.config_digest,
        model_hash: meta.model_hash,
        final_update: 800,
        scope: String::new(),
        split: "inherited_v4_dev".into(),
        microbatch: a.microbatch,
        device: backend.into(),
    };
    let r8 = recur64_v5::study::evaluate_r8(&model.valid(), &data, &indices, identity, &device)?;
    let report = recur64_v5::evaluation::r8_report(&main, &r8)?;
    let combined = serde_json::json!({"report": report, "evaluation": r8});
    write_json(&a.output, &combined)?;
    println!(
        "V5 {backend} R8 diagnostic: mean top1 improvement {:+.8}, mean set-loss improvement {:+.8}",
        report.mean_top1_improvement, report.mean_set_loss_improvement
    );
    Ok(())
}

fn extra_loops(a: ExtraLoopsArgs) -> anyhow::Result<()> {
    match a.device {
        DeviceArg::Cpu => extra_loops_backend::<recur64_model::train::CpuTrainBackend>(&a, "cpu"),
        DeviceArg::Cuda => {
            #[cfg(feature = "cuda")]
            {
                extra_loops_backend::<burn::backend::Autodiff<burn::backend::Cuda>>(&a, "cuda")
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
        V5Cmd::Drill(a) => big_stack("recur64-v5-drill", move || drill(a)),
        V5Cmd::Evaluate(a) => big_stack("recur64-v5-evaluate", move || evaluate(a)),
        V5Cmd::EvaluateBaseline(a) => {
            big_stack("recur64-v5-baseline-evaluate", move || evaluate_baseline(a))
        }
        V5Cmd::EvalMerge(a) => eval_merge(a),
        V5Cmd::Ablation(a) => ablation(a),
        V5Cmd::PilotReport(a) => pilot_report(a),
        V5Cmd::ExtraLoops(a) => big_stack("recur64-v5-r8", move || extra_loops(a)),
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

#[cfg(test)]
mod source_identity_tests {
    use super::*;

    #[test]
    fn qualification_without_synchronized_parity_is_refused() {
        let mut value = serde_json::json!({
            "timing_contract": recur64_v5::profile::CONTRACT,
            "timing_contract_digest": recur64_v5::profile::contract_digest(),
            "profile_outputs_and_all_gradients_exact": true,
            "profile_adamw_parameters_and_moments_exact": true,
            "profile_phase_accounting_consistent": true,
        });
        assert!(require_synchronized_qualification(&value).is_ok());
        for field in [
            "timing_contract",
            "timing_contract_digest",
            "profile_outputs_and_all_gradients_exact",
            "profile_adamw_parameters_and_moments_exact",
            "profile_phase_accounting_consistent",
        ] {
            let saved = value.as_object_mut().unwrap().remove(field).unwrap();
            assert!(
                require_synchronized_qualification(&value).is_err(),
                "missing {field}"
            );
            value[field] = saved;
        }
        value["profile_outputs_and_all_gradients_exact"] = false.into();
        assert!(require_synchronized_qualification(&value).is_err());
    }

    #[test]
    fn source_guard_refuses_staged_untracked_and_stale_builds_but_allows_documentation_commits() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "recur64-v5-source-test-{}-{unique}",
            std::process::id()
        ));
        std::fs::create_dir_all(root.join("crates")).unwrap();
        let git = |args: &[&str]| {
            let output = std::process::Command::new("git")
                .args(args)
                .current_dir(&root)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
        };
        let commit = || {
            git(&[
                "-c",
                "user.name=Recur64 interface test",
                "-c",
                "user.email=test@invalid",
                "commit",
                "-m",
                "interface fixture",
            ])
        };
        git(&["init"]);
        std::fs::write(root.join("crates/source.rs"), "// interface fixture v1\n").unwrap();
        git(&["add", "crates/source.rs"]);
        commit();
        let built = git_sha(&root).unwrap();
        assert_eq!(checked_source_sha(&root, &built).unwrap(), built);
        std::fs::write(root.join("documentation.md"), "documentation only\n").unwrap();
        git(&["add", "documentation.md"]);
        commit();
        assert_eq!(checked_source_sha(&root, &built).unwrap(), built);
        std::fs::write(root.join("crates/untracked.rs"), "// interface fixture\n").unwrap();
        assert!(
            checked_source_sha(&root, &built)
                .unwrap_err()
                .to_string()
                .contains("uncommitted")
        );
        std::fs::remove_file(root.join("crates/untracked.rs")).unwrap();
        std::fs::write(root.join("crates/source.rs"), "// interface fixture v2\n").unwrap();
        assert!(checked_source_sha(&root, &built).is_err());
        git(&["add", "crates/source.rs"]);
        assert!(
            checked_source_sha(&root, &built)
                .unwrap_err()
                .to_string()
                .contains("uncommitted")
        );
        commit();
        assert!(
            checked_source_sha(&root, &built)
                .unwrap_err()
                .to_string()
                .contains("rebuild")
        );
        let rebuilt = git_sha(&root).unwrap();
        assert_eq!(checked_source_sha(&root, &rebuilt).unwrap(), rebuilt);
        std::fs::remove_dir_all(root).unwrap();
    }
}
