//! `recur64 v3-p5 ...`: the bounded LR screen of `active_search_v3`.
//!
//! * `recipe`: write the committed screen contract (resolved layout, health checks);
//! * `preflight`: TRAIN-only real updates at full geometry on the requested device,
//!   measuring wall time and VRAM, plus a projection of one run's wall time;
//! * `train`: one resumable LR x seed run with the preregistered TUNE evaluations;
//! * `select`: apply the frozen selection rule to the six run summaries, once.
//!
//! Only the frozen TRAIN and v3_tune_v1 datasets and their audited traces are
//! accepted (anything else is refused by `load_dataset`). HOLDOUT_C is never opened.
//! A requested device that is unavailable is an error, never a silent substitution.

use std::path::{Path, PathBuf};
use std::time::Instant;

use burn::tensor::backend::AutodiffBackend;
use clap::{Args, Subcommand, ValueEnum};

use recur64_model::active::ActiveSearchModel;
use recur64_runtime::gpu_telemetry::monitor;
use recur64_runtime::model_io;
use recur64_runtime::p5::data::{Dataset, Expected, load_dataset};
use recur64_runtime::p5::eval::{EvalOutput, EvalSelection, evaluate, screen_score};
use recur64_runtime::p5::recipe::{BUDGETS, CANDIDATE_LRS, Layout, Recipe, SCREEN_SEEDS, UPDATES};
use recur64_runtime::p5::train::{P5State, Trainer, draws_per_update};

const STACK_BYTES: usize = 512 * 1024 * 1024;
const EVAL_BATCH: usize = 64;
const CHECKPOINT_EVERY: u64 = 50;
const WALL_LIMIT_S: f64 = 2.0 * 3600.0;

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
pub enum Dev {
    Cpu,
    Cuda,
}

#[derive(Subcommand, Debug)]
pub enum P5Cmd {
    /// Write the screen contract (layout + health checks) to a file.
    Recipe(RecipeArgs),
    /// TRAIN-only preflight at full geometry; measures wall time and VRAM.
    Preflight(PreflightArgs),
    /// One LR x seed run (resumable).
    Train(TrainArgs),
    /// Apply the frozen selection rule to the six run summaries.
    Select(SelectArgs),
    /// P5.2: re-evaluate the selected final checkpoints with the refined selector
    /// diagnostic (evaluation only; TUNE only; never alters P5).
    Rediagnose(RediagArgs),
}

#[derive(Args, Debug)]
pub struct RediagArgs {
    #[command(flatten)]
    pub data: DataArgs,
    #[arg(long)]
    pub tune: PathBuf,
    #[arg(long)]
    pub tune_trace: PathBuf,
    /// The committed screen contract (`v3-p5-recipe.json`).
    #[arg(long)]
    pub recipe: PathBuf,
    #[arg(long, value_enum)]
    pub device: Dev,
    /// Directory holding `v3-p5-run-lr3e-4-seed*/final` (default `runs/v3/p5`).
    #[arg(long)]
    pub runs_root: PathBuf,
    /// Directory holding the committed run summaries (`docs/evidence/v3`).
    #[arg(long)]
    pub evidence: PathBuf,
    #[arg(long)]
    pub output: PathBuf,
}

#[derive(Args, Debug)]
pub struct RecipeArgs {
    /// `16x8` (default) or `8x16` (only after a genuine CUDA OOM at 16x8).
    #[arg(long, default_value = "16x8")]
    pub layout: String,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub health_checks: bool,
    #[arg(long)]
    pub output: PathBuf,
}

#[derive(Args, Debug, Clone)]
pub struct DataArgs {
    #[arg(long)]
    pub train: PathBuf,
    #[arg(long)]
    pub train_trace: PathBuf,
}

#[derive(Args, Debug)]
pub struct PreflightArgs {
    #[command(flatten)]
    pub data: DataArgs,
    #[arg(long, value_enum)]
    pub device: Dev,
    #[arg(long, default_value = "16x8")]
    pub layout: String,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub health_checks: bool,
    /// Real TRAIN updates to run (the first is cold).
    #[arg(long, default_value_t = 3)]
    pub updates: u64,
    /// TRAIN positions used to time evaluation (strided subset).
    #[arg(long, default_value_t = 128)]
    pub eval_positions: usize,
    #[arg(long)]
    pub output: PathBuf,
}

#[derive(Args, Debug)]
pub struct TrainArgs {
    #[command(flatten)]
    pub data: DataArgs,
    #[arg(long)]
    pub tune: PathBuf,
    #[arg(long)]
    pub tune_trace: PathBuf,
    /// The committed screen contract (`v3-p5-recipe.json`).
    #[arg(long)]
    pub recipe: PathBuf,
    #[arg(long)]
    pub lr: f64,
    #[arg(long)]
    pub seed: u64,
    #[arg(long, value_enum)]
    pub device: Dev,
    /// Run directory (checkpoints, per-update evaluations, log).
    #[arg(long)]
    pub run_dir: PathBuf,
    /// Where to write the run summary.
    #[arg(long)]
    pub summary: PathBuf,
    /// P6 baseline replication (`p6_baseline_replication_v1`): train the exact SELECTED P5
    /// recipe (peak LR 3e-4) at the one additional paired seed 5103. Not a screen run;
    /// needs `--selected-recipe` to prove the recipe is the selected one.
    #[arg(long, default_value_t = false)]
    pub p6_baseline_replication: bool,
    /// `v3-p5-selected-recipe.json` (required with `--p6-baseline-replication`).
    #[arg(long)]
    pub selected_recipe: Option<PathBuf>,
}

/// The only seed the P6 baseline replication may use.
pub const P6_REPLICATION_SEED: u64 = 5103;
/// The LR selected by the P5 screen (V3-D21).
pub const P5_SELECTED_LR: f64 = 3.0e-4;
/// Digest (without seed) of the selected P5 recipe.
pub const P5_SELECTED_DIGEST: &str =
    "a069ba9d18befed65f970aca253b47780365fd7019be38283f270d79d6c1db33";

#[derive(Args, Debug)]
pub struct SelectArgs {
    /// Directory holding the six `v3-p5-run-lr*-seed*.json` summaries.
    #[arg(long)]
    pub summaries: PathBuf,
    #[arg(long)]
    pub recipe: PathBuf,
    #[arg(long)]
    pub selection_out: PathBuf,
    #[arg(long)]
    pub selected_recipe_out: PathBuf,
    /// Where to record the same-seed update-0 pairing integrity check.
    #[arg(long)]
    pub pairing_out: PathBuf,
}

pub fn parse_layout(s: &str) -> anyhow::Result<Layout> {
    match s {
        "16x8" => Ok(Layout::DEFAULT),
        "8x16" => Ok(Layout::FALLBACK),
        other => anyhow::bail!("layout {other}: only 16x8 and the pre-authorised 8x16 exist"),
    }
}

/// Stable file stem of one screen run.
pub fn run_stem(lr: f64, seed: u64) -> String {
    format!("v3-p5-run-lr{lr:e}-seed{seed}")
}

fn on_big_stack<T: Send + 'static>(
    name: &str,
    f: impl FnOnce() -> anyhow::Result<T> + Send + 'static,
) -> anyhow::Result<T> {
    std::thread::Builder::new()
        .name(name.into())
        .stack_size(STACK_BYTES)
        .spawn(f)?
        .join()
        .map_err(|_| anyhow::anyhow!("{name} thread panicked"))?
}

pub fn run(cmd: P5Cmd) -> anyhow::Result<()> {
    match cmd {
        P5Cmd::Recipe(a) => run_recipe(&a),
        P5Cmd::Preflight(a) => {
            let dev = a.device;
            dispatch(dev, move |d| match d {
                Dispatch::Cpu => preflight::<recur64_model::train::CpuTrainBackend>(&a, false),
                #[cfg(feature = "cuda")]
                Dispatch::Cuda => {
                    preflight::<burn::backend::Autodiff<burn::backend::Cuda>>(&a, true)
                }
            })
        }
        P5Cmd::Train(a) => {
            let dev = a.device;
            dispatch(dev, move |d| match d {
                Dispatch::Cpu => train::<recur64_model::train::CpuTrainBackend>(&a, false),
                #[cfg(feature = "cuda")]
                Dispatch::Cuda => train::<burn::backend::Autodiff<burn::backend::Cuda>>(&a, true),
            })
        }
        P5Cmd::Select(a) => run_select(&a),
        P5Cmd::Rediagnose(a) => {
            let dev = a.device;
            dispatch(dev, move |d| match d {
                Dispatch::Cpu => rediagnose::<recur64_model::train::CpuTrainBackend>(&a),
                #[cfg(feature = "cuda")]
                Dispatch::Cuda => rediagnose::<burn::backend::Autodiff<burn::backend::Cuda>>(&a),
            })
        }
    }
}

enum Dispatch {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda,
}

fn dispatch(
    dev: Dev,
    f: impl FnOnce(Dispatch) -> anyhow::Result<()> + Send + 'static,
) -> anyhow::Result<()> {
    let d = match dev {
        Dev::Cpu => Dispatch::Cpu,
        #[cfg(feature = "cuda")]
        Dev::Cuda => Dispatch::Cuda,
        #[cfg(not(feature = "cuda"))]
        Dev::Cuda => anyhow::bail!("CUDA support is not compiled; rebuild with --features cuda"),
    };
    on_big_stack("v3-p5", move || f(d))
}

fn write_json(path: &Path, v: &serde_json::Value) -> anyhow::Result<()> {
    if let Some(p) = path.parent() {
        std::fs::create_dir_all(p)?;
    }
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(v)?)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

fn run_recipe(a: &RecipeArgs) -> anyhow::Result<()> {
    let r = Recipe::screen_contract(parse_layout(&a.layout)?, a.health_checks);
    r.validate()?;
    let doc = serde_json::json!({
        "schema": "v3_p5_recipe_file_v1",
        "contract_digest": r.contract_digest(),
        "candidate_lrs": CANDIDATE_LRS,
        "seeds": SCREEN_SEEDS,
        "recipe": r,
    });
    // The file stores the Recipe itself under "recipe"; `train` reads that field.
    write_json(&a.output, &doc)?;
    println!("contract digest {}", r.contract_digest());
    Ok(())
}

fn load_contract(path: &Path) -> anyhow::Result<Recipe> {
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
    let r: Recipe = serde_json::from_value(v["recipe"].clone())?;
    anyhow::ensure!(
        r.peak_lr.is_none() && r.seed.is_none(),
        "{} is not a screen contract",
        path.display()
    );
    anyhow::ensure!(
        v["contract_digest"] == r.contract_digest(),
        "the recipe file's digest does not match its contents"
    );
    r.validate()?;
    Ok(r)
}

fn strided_subset(data: &Dataset, n: usize) -> Dataset {
    let total = data.positions().len();
    let stride = (total / n.max(1)).max(1);
    let idx: Vec<usize> = (0..total).step_by(stride).take(n).collect();
    let mut targets = data.targets.clone();
    targets.positions = idx
        .iter()
        .map(|&i| data.targets.positions[i].clone())
        .collect();
    Dataset {
        targets,
        traces: idx.iter().map(|&i| data.traces[i].clone()).collect(),
        trace_manifest_digest: data.trace_manifest_digest.clone(),
    }
}

fn eval_json(update: u64, out: &EvalOutput) -> serde_json::Value {
    serde_json::json!({
        "update": update,
        "summary": out.summary,
        "selector_nll_sum_and_count": out.selector_nll,
        "selector_diag": out.selector_diag.as_ref().map(|(all, cells)| serde_json::json!({
            "pooled": all, "cells": cells,
        })),
    })
}

fn preflight<TB: AutodiffBackend>(a: &PreflightArgs, gpu: bool) -> anyhow::Result<()> {
    let device: TB::Device = Default::default();
    let inner = Default::default();
    model_io::verify_device::<TB::InnerBackend>(&inner)?;
    let layout = parse_layout(&a.layout)?;
    let recipe = Recipe::screen_contract(layout, a.health_checks).for_run(1.5e-4, 5101);
    let train = load_dataset(&a.data.train, &a.data.train_trace, &Expected::train())?;
    eprintln!("TRAIN verified: {} positions", train.positions().len());

    let (result, gpu_samples) = monitor(gpu, || -> anyhow::Result<serde_json::Value> {
        let mut tr = Trainer::<TB>::new(recipe.clone(), &train, &device)?;
        let mut walls = Vec::new();
        let mut updates = Vec::new();
        for _ in 0..a.updates {
            let rec = tr.step(&train, &device)?;
            eprintln!(
                "update {} wall {:.1}s loss {:.4} (policy {:.4} selector {:.4}) grad {:.3}",
                rec.update,
                rec.wall_s,
                rec.report.total_loss,
                rec.report.policy_loss,
                rec.report.selector_loss,
                rec.report.grad_norm
            );
            walls.push(rec.wall_s);
            updates.push(serde_json::to_value(&rec)?);
        }
        // Evaluation timing on a strided TRAIN subset (TUNE is not touched).
        let sub = strided_subset(&train, a.eval_positions);
        let model = tr.inference_model();
        let mut eval_s = serde_json::Map::new();
        let mut per_pos_active = 0.0;
        for b in BUDGETS {
            let t0 = Instant::now();
            let out = evaluate(&model, &sub, b, EvalSelection::Active, EVAL_BATCH, &inner)?;
            let s = t0.elapsed().as_secs_f64() / sub.positions().len() as f64;
            per_pos_active += s;
            eval_s.insert(format!("active_B{b}_s_per_position"), s.into());
            anyhow::ensure!(out.summary.pooled.ce.is_finite(), "non-finite eval CE");
        }
        let mut per_pos_diag = 0.0;
        for b in [2usize, 4, 8] {
            for sel in [EvalSelection::Teacher, EvalSelection::Fixed] {
                let t0 = Instant::now();
                evaluate(&model, &sub, b, sel, EVAL_BATCH, &inner)?;
                per_pos_diag += t0.elapsed().as_secs_f64() / sub.positions().len() as f64;
            }
        }
        let steady: Vec<f64> = walls.iter().skip(1).copied().collect();
        let steady_mean = if steady.is_empty() {
            walls.iter().sum::<f64>() / walls.len().max(1) as f64
        } else {
            steady.iter().sum::<f64>() / steady.len() as f64
        };
        let tune_n = recur64_runtime::p5::recipe::TUNE_POSITIONS as f64;
        let train_s = steady_mean * UPDATES as f64;
        let eval_s_total = 5.0 * per_pos_active * tune_n + per_pos_diag * tune_n;
        let ckpt_s = (UPDATES / CHECKPOINT_EVERY) as f64 * 5.0;
        let total = train_s + eval_s_total + ckpt_s;
        Ok(serde_json::json!({
            "updates": updates,
            "steady_update_wall_s_mean": steady_mean,
            "cold_update_wall_s": walls.first(),
            "eval_timing_subset_positions": sub.positions().len(),
            "eval_timing": eval_s,
            "projection": {
                "basis": "steady-state mean update wall x 800 + 5 ACTIVE evaluations (B0/2/4/8) x 4500 + one update-800 diagnostic pass, per-position costs measured on the TRAIN subset; checkpoint cost assumed 5 s x 16",
                "train_s": train_s,
                "evaluation_s": eval_s_total,
                "checkpoint_s_assumed": ckpt_s,
                "single_run_total_s": total,
                "single_run_total_h": total / 3600.0,
                "limit_h": WALL_LIMIT_S / 3600.0,
                "within_limit": total < WALL_LIMIT_S,
            },
        }))
    });
    let mut report = match result {
        Ok(v) => v,
        Err(e) => {
            // A visible failure: record it and exit nonzero. A genuine OOM is the only
            // thing that authorises the 8x16 fallback.
            let doc =
                serde_json::json!({"ok": false, "layout": a.layout, "error": format!("{e:#}")});
            write_json(&a.output, &doc)?;
            return Err(e);
        }
    };
    let o = report.as_object_mut().expect("object");
    o.insert("ok".into(), true.into());
    o.insert(
        "tested".into(),
        "real TRAIN updates at full geometry with live StateQuery on the requested device".into(),
    );
    o.insert(
        "device".into(),
        format!("{:?}", if gpu { Dev::Cuda } else { Dev::Cpu }).into(),
    );
    o.insert(
        "layout".into(),
        serde_json::json!({"micro": layout.micro, "accum": layout.accum}),
    );
    o.insert("health_checks".into(), a.health_checks.into());
    o.insert(
        "recipe_contract_digest".into(),
        recipe.contract_digest().into(),
    );
    o.insert("gpu".into(), serde_json::to_value(&gpu_samples)?);
    write_json(&a.output, &report)?;
    println!("{}", serde_json::to_string_pretty(&report["projection"])?);
    Ok(())
}

// ---------------------------------------------------------------------------
// Train
// ---------------------------------------------------------------------------

fn eval_path(dir: &Path, update: u64) -> PathBuf {
    dir.join(format!("eval-u{update:04}.json"))
}

fn state_dirs(dir: &Path) -> [PathBuf; 2] {
    [dir.join("state-0"), dir.join("state-1")]
}

fn latest_state(dir: &Path, digest: &str) -> anyhow::Result<Option<PathBuf>> {
    let mut best: Option<(u64, PathBuf)> = None;
    for d in state_dirs(dir) {
        let side = d.join("p5-state.json");
        if !side.exists() {
            continue;
        }
        let st: P5State = serde_json::from_slice(&std::fs::read(&side)?)?;
        anyhow::ensure!(
            st.recipe_digest == digest,
            "{} belongs to a different recipe: refusing to resume",
            d.display()
        );
        if best.as_ref().is_none_or(|(u, _)| st.updates_done > *u) {
            best = Some((st.updates_done, d));
        }
    }
    Ok(best.map(|(_, d)| d))
}

fn load_eval_summaries(dir: &Path, update: u64) -> anyhow::Result<serde_json::Value> {
    Ok(serde_json::from_slice(&std::fs::read(eval_path(
        dir, update,
    ))?)?)
}

fn run_eval_at<B: burn::tensor::backend::Backend>(
    model: &ActiveSearchModel<B>,
    tune: &Dataset,
    update: u64,
    final_update: bool,
    recipe_digest: &str,
    device: &B::Device,
) -> anyhow::Result<serde_json::Value> {
    let t0 = Instant::now();
    let mut active = Vec::new();
    let mut active_json = Vec::new();
    for b in BUDGETS {
        let out = evaluate(model, tune, b, EvalSelection::Active, EVAL_BATCH, device)?;
        active_json.push(eval_json(update, &out));
        active.push(out.summary);
    }
    let score = screen_score(&active)?;
    let mut doc = serde_json::json!({
        "update": update,
        "recipe_digest": recipe_digest,
        "screen_score_S_run": score,
        "active": active_json,
    });
    if final_update {
        let mut teacher = Vec::new();
        let mut fixed = Vec::new();
        for b in [2usize, 4, 8] {
            teacher.push(eval_json(
                update,
                &evaluate(model, tune, b, EvalSelection::Teacher, EVAL_BATCH, device)?,
            ));
            fixed.push(eval_json(
                update,
                &evaluate(model, tune, b, EvalSelection::Fixed, EVAL_BATCH, device)?,
            ));
        }
        let o = doc.as_object_mut().expect("object");
        o.insert("teacher_forced_diagnostic".into(), teacher.into());
        o.insert("fixed_diagnostic".into(), fixed.into());
    }
    doc.as_object_mut()
        .expect("object")
        .insert("eval_wall_s".into(), t0.elapsed().as_secs_f64().into());
    Ok(doc)
}

/// Operator guard: a fresh model starts only in an empty run directory, so it can
/// never inherit another run's evaluations or partial artifacts.
fn require_empty_run_dir(dir: &Path) -> anyhow::Result<()> {
    let mut found: Vec<String> = Vec::new();
    for e in std::fs::read_dir(dir)? {
        found.push(e?.file_name().to_string_lossy().into_owned());
    }
    anyhow::ensure!(
        found.is_empty(),
        "{} holds P5 artifacts {found:?} but no valid resumable state: quarantine or remove the run \
         explicitly, or use an empty run directory (a fresh model is never started here)",
        dir.display()
    );
    Ok(())
}

/// The preregistered ineligibility classes (plan section 6): non-finite loss/gradient,
/// a health refusal, a query correctness error, a checkpoint/resume error.
pub const INELIGIBLE_CLASSES: [&str; 4] = [
    "non_finite",
    "health_refusal",
    "query_correctness",
    "checkpoint_resume",
];

fn ineligibility_class(msg: &str) -> Option<&'static str> {
    if msg.contains("non-finite") {
        Some("non_finite")
    } else if msg.contains("health") {
        Some("health_refusal")
    } else if msg.contains("query") {
        Some("query_correctness")
    } else if msg.contains("checkpoint") || msg.contains("refus") {
        Some("checkpoint_resume")
    } else {
        None
    }
}

/// Final per-budget sampler exposure, verified against the layout: every budget must
/// have consumed exactly `updates x draws-per-update` examples.
fn exposure_json(
    recipe: &Recipe,
    stats: &[(usize, recur64_runtime::proof::sampler::SamplerStats)],
) -> anyhow::Result<serde_json::Value> {
    let mut out = Vec::new();
    for (b, st) in stats {
        let mut v = serde_json::to_value(st)?;
        v.as_object_mut()
            .expect("object")
            .insert("budget".into(), (*b).into());
        out.push(v);
    }
    let v = serde_json::Value::Array(out);
    let per = draws_per_update(recipe, BUDGETS[0]);
    for b in BUDGETS {
        anyhow::ensure!(
            draws_per_update(recipe, b) == per,
            "unequal per-update exposure across budgets"
        );
    }
    check_exposure(&v, recipe.updates * per)?;
    Ok(v)
}

/// Validate persisted exposure evidence: four budgets in order, each with exactly
/// `expected` examples, per-cell counts summing to it and balanced within one.
pub fn check_exposure(v: &serde_json::Value, expected: u64) -> anyhow::Result<()> {
    let arr = v
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("sampler_exposure is not a list"))?;
    anyhow::ensure!(
        arr.len() == BUDGETS.len(),
        "sampler_exposure needs 4 budgets"
    );
    for (e, &b) in arr.iter().zip(&BUDGETS) {
        anyhow::ensure!(e["budget"] == b, "sampler_exposure budget order");
        anyhow::ensure!(
            e["examples_drawn"] == expected,
            "B{b}: consumed {} examples, expected {expected}",
            e["examples_drawn"]
        );
        let cells = e["cells"]
            .as_array()
            .filter(|c| !c.is_empty())
            .ok_or_else(|| anyhow::anyhow!("B{b}: no sampler cells"))?;
        let counts: Vec<u64> = cells
            .iter()
            .map(|c| c["examples_consumed"].as_u64())
            .collect::<Option<_>>()
            .ok_or_else(|| anyhow::anyhow!("B{b}: malformed cell counts"))?;
        anyhow::ensure!(
            counts.iter().sum::<u64>() == expected,
            "B{b}: cell counts do not sum to {expected}"
        );
        let (lo, hi) = (counts.iter().min().unwrap(), counts.iter().max().unwrap());
        anyhow::ensure!(hi - lo <= 1, "B{b}: cells unbalanced ({lo}..{hi})");
        anyhow::ensure!(
            e["fraction_by_family"].is_object() && e["fraction_by_depth"].is_object(),
            "B{b}: family/depth fractions missing"
        );
    }
    Ok(())
}

/// Largest absolute difference between a re-evaluated `EvalSummary` and the committed one
/// (pooled and every cell; top-1, correct mass, CE, entropy).
fn summary_max_diff(
    fresh: &serde_json::Value,
    committed: &serde_json::Value,
) -> anyhow::Result<f64> {
    let mut worst = 0.0f64;
    let mut groups: Vec<(&serde_json::Value, &serde_json::Value)> =
        vec![(&fresh["pooled"], &committed["pooled"])];
    for (k, c) in committed["cells"]
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("committed summary has no cells"))?
    {
        groups.push((&fresh["cells"][k], c));
    }
    for (f, c) in groups {
        for key in ["top1", "correct_mass", "ce", "entropy"] {
            let (a, b) = (
                f[key]
                    .as_f64()
                    .ok_or_else(|| anyhow::anyhow!("fresh {key} missing"))?,
                c[key]
                    .as_f64()
                    .ok_or_else(|| anyhow::anyhow!("committed {key} missing"))?,
            );
            worst = worst.max((a - b).abs());
        }
    }
    Ok(worst)
}

/// Policy values must reproduce the committed P5 evidence to this absolute tolerance.
const REDIAG_POLICY_TOL: f64 = 1e-4;

fn rediagnose<TB: AutodiffBackend>(a: &RediagArgs) -> anyhow::Result<()> {
    let device: TB::Device = Default::default();
    let inner = Default::default();
    model_io::verify_device::<TB::InnerBackend>(&inner)?;
    let contract = load_contract(&a.recipe)?;
    let train_ds = load_dataset(&a.data.train, &a.data.train_trace, &Expected::train())?;
    let tune = load_dataset(&a.tune, &a.tune_trace, &Expected::tune())?;
    let mut runs = Vec::new();
    let mut worst = 0.0f64;
    for seed in SCREEN_SEEDS {
        let lr = P5_SELECTED_LR;
        let recipe = contract.clone().for_run(lr, seed);
        let digest = recipe.digest();
        let stem = run_stem(lr, seed);
        let committed: serde_json::Value =
            serde_json::from_slice(&std::fs::read(a.evidence.join(format!("{stem}.json")))?)?;
        anyhow::ensure!(
            committed["recipe_digest"] == digest.as_str(),
            "{stem}: committed summary was run under a different recipe"
        );
        let final_dir = a.runs_root.join(&stem).join("final");
        // Trainer::load verifies the sidecar digest, the checkpoint metadata and every
        // consistency invariant before any weight is used.
        let tr = Trainer::<TB>::load(&final_dir, recipe.clone(), &train_ds, &device)?;
        anyhow::ensure!(
            tr.updates_done == recipe.updates,
            "{stem}: final checkpoint is at update {}, expected {}",
            tr.updates_done,
            recipe.updates
        );
        let model = tr.inference_model();
        let mut budgets = Vec::new();
        for (bi, b) in BUDGETS.iter().copied().enumerate() {
            if b == 0 {
                continue;
            }
            let out = evaluate(&model, &tune, b, EvalSelection::Active, EVAL_BATCH, &inner)?;
            let fresh = serde_json::to_value(&out.summary)?;
            let diff = summary_max_diff(
                &fresh,
                &committed["evaluations"]["800"]["active"][bi]["summary"],
            )?;
            worst = worst.max(diff);
            let (refined, refined_cells) = out
                .refined_diag
                .ok_or_else(|| anyhow::anyhow!("no refined diagnostic at B{b}"))?;
            let (_, _) = out.selector_diag.as_ref().expect("same condition");
            eprintln!("{stem} B{b}: policy max |diff| vs committed P5 = {diff:e}");
            budgets.push(serde_json::json!({
                "budget": b,
                "policy_summary": fresh,
                "policy_max_abs_diff_vs_committed_p5": diff,
                "refined_selector_diag": {"pooled": refined, "cells": refined_cells},
            }));
        }
        runs.push(serde_json::json!({
            "lr": lr, "seed": seed, "recipe_digest": digest, "final_updates_done": tr.updates_done,
            "budgets": budgets,
        }));
    }
    let ok = worst <= REDIAG_POLICY_TOL;
    write_json(
        &a.output,
        &serde_json::json!({
            "schema": "v3_p5.2_selector_diagnostics_v1",
            "label": "DERIVED / RE-EVALUATED DIAGNOSTIC - does not alter P5 selection",
            "dataset": "V3_TUNE_V1 only",
            "contract_digest": contract.contract_digest(),
            "policy_tolerance": REDIAG_POLICY_TOL,
            "policy_max_abs_diff_all": worst,
            "policy_reproduces_committed_p5": ok,
            "runs": runs,
        }),
    )?;
    anyhow::ensure!(
        ok,
        "re-evaluated policy values differ from the committed P5 evidence by {worst:e} (> {REDIAG_POLICY_TOL:e}): STOP and investigate before P6"
    );
    println!("P5.2 policy reproduction max |diff| = {worst:e}");
    Ok(())
}

fn train<TB: AutodiffBackend>(a: &TrainArgs, gpu: bool) -> anyhow::Result<()> {
    let device: TB::Device = Default::default();
    let inner = Default::default();
    model_io::verify_device::<TB::InnerBackend>(&inner)?;
    if a.p6_baseline_replication {
        anyhow::ensure!(
            a.lr == P5_SELECTED_LR && a.seed == P6_REPLICATION_SEED,
            "the P6 baseline replication is the selected LR {P5_SELECTED_LR:e} at seed {P6_REPLICATION_SEED} only"
        );
        let path = a
            .selected_recipe
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("--p6-baseline-replication needs --selected-recipe"))?;
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
        anyhow::ensure!(
            v["schema"] == "v3_p5_selected_recipe_v1"
                && v["selected_lr"].as_f64() == Some(P5_SELECTED_LR)
                && v["digest_without_seed"] == P5_SELECTED_DIGEST,
            "{} is not the accepted selected P5 recipe",
            path.display()
        );
        let mut sel = load_contract(&a.recipe)?.for_run(a.lr, 0);
        sel.seed = None;
        anyhow::ensure!(
            sel.digest() == P5_SELECTED_DIGEST,
            "the contract with LR {:e} does not reproduce the selected recipe digest",
            a.lr
        );
    } else {
        anyhow::ensure!(
            CANDIDATE_LRS.contains(&a.lr) && SCREEN_SEEDS.contains(&a.seed),
            "the screen runs only the preregistered LRs {CANDIDATE_LRS:?} and seeds {SCREEN_SEEDS:?}"
        );
    }
    let recipe = load_contract(&a.recipe)?.for_run(a.lr, a.seed);
    let digest = recipe.digest();
    let train_ds = load_dataset(&a.data.train, &a.data.train_trace, &Expected::train())?;
    let tune = load_dataset(&a.tune, &a.tune_trace, &Expected::tune())?;
    eprintln!(
        "TRAIN {} / TUNE {} verified; run digest {digest}",
        train_ds.positions().len(),
        tune.positions().len()
    );
    std::fs::create_dir_all(&a.run_dir)?;
    // A fresh model may start only in a genuinely empty run directory. This is an
    // operator error, not a preregistered ineligibility: it writes no summary.
    let resume_from = latest_state(&a.run_dir, &digest)?;
    if resume_from.is_none() {
        require_empty_run_dir(&a.run_dir)?;
    }
    let started = Instant::now();

    let (result, gpu_samples) = monitor(gpu, || -> anyhow::Result<serde_json::Value> {
        let mut tr = match &resume_from {
            Some(d) => {
                eprintln!("resuming from {}", d.display());
                Trainer::<TB>::load(d, recipe.clone(), &train_ds, &device)?
            }
            None => Trainer::<TB>::new(recipe.clone(), &train_ds, &device)?,
        };
        let start_update = tr.updates_done;
        loop {
            let u = tr.updates_done;
            if recipe.eval_updates.contains(&u) && !eval_path(&a.run_dir, u).exists() {
                let model = tr.inference_model();
                let doc = run_eval_at(&model, &tune, u, u == recipe.updates, &digest, &inner)?;
                eprintln!(
                    "update {u}: TUNE S_run {:.5} (eval {:.0}s)",
                    doc["screen_score_S_run"].as_f64().unwrap_or(f64::NAN),
                    doc["eval_wall_s"].as_f64().unwrap_or(0.0)
                );
                write_json(&eval_path(&a.run_dir, u), &doc)?;
            }
            if u >= recipe.updates {
                break;
            }
            let rec = tr.step(&train_ds, &device)?;
            if rec.update % 10 == 0 || rec.update < 3 {
                eprintln!(
                    "update {} lr {:.3e} wall {:.1}s loss {:.4} (policy {:.4} selector {:.4}) grad {:.3} sup {}",
                    rec.update,
                    rec.lr,
                    rec.wall_s,
                    rec.report.total_loss,
                    rec.report.policy_loss,
                    rec.report.selector_loss,
                    rec.report.grad_norm,
                    rec.report.supervised_decisions
                );
            }
            if tr.updates_done % CHECKPOINT_EVERY == 0
                || recipe.eval_updates.contains(&tr.updates_done)
            {
                let dirs = state_dirs(&a.run_dir);
                let dir = &dirs[((tr.updates_done / CHECKPOINT_EVERY) % 2) as usize];
                std::fs::create_dir_all(dir)?;
                tr.save(dir)?;
            }
        }
        // Final state, for provenance.
        let dir = a.run_dir.join("final");
        std::fs::create_dir_all(&dir)?;
        tr.save(&dir)?;
        write_json(
            &a.run_dir.join("history.json"),
            &serde_json::to_value(&tr.history)?,
        )?;
        let exposure = exposure_json(&recipe, &tr.samplers.stats())?;
        Ok(serde_json::json!({
            "sampler_exposure": exposure,
            "provenance": {
                "start_kind": if resume_from.is_some() { "resumed" } else { "fresh" },
                "start_update": start_update,
                "resumptions_including_this_invocation": tr.resumptions,
                "prior_resumptions": tr.resumptions.saturating_sub(u32::from(resume_from.is_some())),
            },
        }))
    });

    let mut summary = serde_json::json!({
        "schema": "v3_p5_run_summary_v1",
        "identity": if a.p6_baseline_replication { "p6_baseline_replication_v1" } else { "p5_screen_run" },
        "lr": a.lr,
        "seed": a.seed,
        "recipe_digest": digest,
        "contract_digest": recipe.contract_digest(),
        "device": if gpu { "cuda" } else { "cpu" },
        "gpu": gpu_samples,
        "wall_s_this_invocation": started.elapsed().as_secs_f64(),
    });
    let o = summary.as_object_mut().expect("object");
    match result {
        Ok(extra) => {
            let mut evals = serde_json::Map::new();
            for u in &recipe.eval_updates {
                let doc = load_eval_summaries(&a.run_dir, *u)?;
                anyhow::ensure!(
                    doc["recipe_digest"] == digest.as_str() && doc["update"] == *u,
                    "{} does not belong to this run (recipe/update mismatch): quarantine the run directory",
                    eval_path(&a.run_dir, *u).display()
                );
                evals.insert(u.to_string(), doc);
            }
            let final_doc = &evals[&recipe.updates.to_string()];
            let hist: Vec<recur64_runtime::p5::train::UpdateRecord> =
                serde_json::from_slice(&std::fs::read(a.run_dir.join("history.json"))?)?;
            o.insert("eligible".into(), true.into());
            o.insert("ineligible_reason".into(), serde_json::Value::Null);
            o.insert("S_run".into(), final_doc["screen_score_S_run"].clone());
            o.insert("S_run_update".into(), recipe.updates.into());
            anyhow::ensure!(
                hist.len() as u64 == recipe.updates,
                "history holds {} updates, expected {}",
                hist.len(),
                recipe.updates
            );
            o.insert("updates_trained".into(), (hist.len() as u64).into());
            o.insert(
                "train_wall_s".into(),
                hist.iter().map(|h| h.wall_s).sum::<f64>().into(),
            );
            o.insert("sampler_exposure".into(), extra["sampler_exposure"].clone());
            o.insert("provenance".into(), extra["provenance"].clone());
            o.insert(
                "loss_curve".into(),
                serde_json::to_value(
                    hist.iter()
                        .step_by(10)
                        .map(|h| {
                            serde_json::json!({
                                "update": h.update, "lr": h.lr, "policy": h.report.policy_loss,
                                "selector": h.report.selector_loss, "grad_norm": h.report.grad_norm,
                                "supervised": h.report.supervised_decisions,
                            })
                        })
                        .collect::<Vec<_>>(),
                )?,
            );
            o.insert("evaluations".into(), evals.into());
        }
        Err(e) => {
            let msg = format!("{e:#}");
            // Only the preregistered conditions make a run ineligible; anything else is an
            // infrastructure error the operator must resolve (no summary is written).
            let Some(class) = ineligibility_class(&msg) else {
                return Err(e);
            };
            o.insert("eligible".into(), false.into());
            o.insert("ineligible_class".into(), class.into());
            o.insert("ineligible_reason".into(), msg.clone().into());
            write_json(&a.summary, &summary)?;
            return Err(e);
        }
    }
    write_json(&a.summary, &summary)?;
    println!(
        "run lr {:e} seed {}: S_run {}",
        a.lr, a.seed, summary["S_run"]
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Selection
// ---------------------------------------------------------------------------

/// The frozen rule, as a pure function so it can be tested:
/// `S_lr` = mean of the seeds' `S_run` (every seed must be eligible); the lowest wins;
/// an exact tie goes to the lower LR. No tolerance, no override.
pub fn select_lr(runs: &[(f64, u64, Option<f64>)]) -> anyhow::Result<(f64, Vec<(f64, f64)>)> {
    let mut scores: Vec<(f64, f64)> = Vec::new();
    for lr in CANDIDATE_LRS {
        let per: Vec<Option<f64>> = SCREEN_SEEDS
            .iter()
            .map(|s| {
                runs.iter()
                    .find(|(l, seed, _)| *l == lr && seed == s)
                    .and_then(|(_, _, v)| *v)
            })
            .collect();
        if per.iter().all(|v| v.is_some_and(f64::is_finite)) {
            let mean = per.iter().map(|v| v.expect("checked")).sum::<f64>() / per.len() as f64;
            scores.push((lr, mean));
        }
    }
    anyhow::ensure!(
        !scores.is_empty(),
        "no learning rate has all seeds eligible"
    );
    let mut best = scores[0];
    for &(lr, s) in &scores[1..] {
        if s < best.1 || (s == best.1 && lr < best.0) {
            best = (lr, s);
        }
    }
    Ok((best.0, scores))
}

/// Recompute `S_run` from an evaluation document's own per-cell CEs (mean over the
/// budgets {0,2,4,8} of the mean over the six cells), independent of the stored score.
fn s_run_from_eval(doc: &serde_json::Value) -> anyhow::Result<f64> {
    let active = doc["active"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("evaluation has no ACTIVE list"))?;
    anyhow::ensure!(
        active.len() == BUDGETS.len(),
        "evaluation needs ACTIVE results at exactly B0/B2/B4/B8"
    );
    let mut total = 0.0;
    for (a, &b) in active.iter().zip(&BUDGETS) {
        let s = &a["summary"];
        anyhow::ensure!(
            s["budget"] == b && s["selection"] == EvalSelection::Active.label(),
            "evaluation entry is not ACTIVE B{b}"
        );
        let cells = s["cells"]
            .as_object()
            .filter(|c| c.len() == 6)
            .ok_or_else(|| anyhow::anyhow!("B{b}: expected six TUNE cells"))?;
        let mut m = 0.0;
        for c in cells.values() {
            let ce = c["ce"]
                .as_f64()
                .filter(|x| x.is_finite())
                .ok_or_else(|| anyhow::anyhow!("B{b}: non-finite cell CE"))?;
            m += ce;
        }
        total += m / 6.0;
    }
    Ok(total / BUDGETS.len() as f64)
}

/// Check one run summary against everything the selection rule relies on. Returns the
/// run's `S_run` if it is eligible. Never trusts the file name or contract digest alone.
pub fn validate_summary(
    v: &serde_json::Value,
    lr: f64,
    seed: u64,
    contract: &Recipe,
) -> anyhow::Result<Option<f64>> {
    anyhow::ensure!(
        v["schema"] == "v3_p5_run_summary_v1",
        "wrong summary schema"
    );
    anyhow::ensure!(
        v["lr"].as_f64() == Some(lr) && v["seed"].as_u64() == Some(seed),
        "summary lr/seed {} / {} is not the expected {lr:e} / {seed}",
        v["lr"],
        v["seed"]
    );
    anyhow::ensure!(
        v["contract_digest"] == contract.contract_digest(),
        "summary was run under a different contract"
    );
    anyhow::ensure!(
        v["recipe_digest"] == contract.clone().for_run(lr, seed).digest(),
        "summary recipe digest is not the frozen contract's for this lr/seed"
    );
    let text = v.to_string().to_ascii_lowercase();
    anyhow::ensure!(
        !text.contains("holdout") && !text.contains("confirm"),
        "summary references HOLDOUT/CONFIRM data"
    );
    match v["eligible"].as_bool() {
        Some(false) => {
            let reason = v["ineligible_reason"].as_str().unwrap_or("");
            anyhow::ensure!(!reason.is_empty(), "ineligible summary has no reason");
            let class = v["ineligible_class"].as_str().unwrap_or("");
            anyhow::ensure!(
                INELIGIBLE_CLASSES.contains(&class),
                "ineligible class '{class}' is not a preregistered class"
            );
            anyhow::ensure!(
                v.get("S_run").is_none_or(serde_json::Value::is_null),
                "an ineligible summary must not carry an S_run"
            );
            Ok(None)
        }
        Some(true) => {
            anyhow::ensure!(
                v["ineligible_reason"].is_null(),
                "eligible summary has an ineligible reason"
            );
            anyhow::ensure!(
                v["updates_trained"] == contract.updates && v["S_run_update"] == contract.updates,
                "eligible run did not train and score exactly {} updates",
                contract.updates
            );
            let s = v["S_run"]
                .as_f64()
                .filter(|x| x.is_finite())
                .ok_or_else(|| anyhow::anyhow!("S_run is missing or non-finite"))?;
            let evals = v["evaluations"]
                .as_object()
                .ok_or_else(|| anyhow::anyhow!("no evaluations"))?;
            let mut keys: Vec<u64> = evals
                .keys()
                .map(|k| k.parse::<u64>())
                .collect::<Result<_, _>>()?;
            keys.sort_unstable();
            anyhow::ensure!(
                keys == contract.eval_updates,
                "evaluations at {keys:?}, expected {:?}",
                contract.eval_updates
            );
            for (k, doc) in evals {
                anyhow::ensure!(
                    doc["update"].as_u64().map(|u| u.to_string()).as_deref() == Some(k.as_str()),
                    "evaluation {k} is labelled update {}",
                    doc["update"]
                );
                let again = s_run_from_eval(doc)?;
                let stored = doc["screen_score_S_run"].as_f64().unwrap_or(f64::NAN);
                anyhow::ensure!(
                    (again - stored).abs() <= 1e-9,
                    "evaluation {k}: stored S_run {stored} != recomputed {again}"
                );
            }
            let last = contract.updates.to_string();
            anyhow::ensure!(
                (evals[&last]["screen_score_S_run"]
                    .as_f64()
                    .unwrap_or(f64::NAN)
                    - s)
                    .abs()
                    <= 1e-12,
                "S_run is not the update-{last} evaluation's score"
            );
            check_exposure(
                &v["sampler_exposure"],
                contract.updates * draws_per_update(contract, BUDGETS[0]),
            )?;
            anyhow::ensure!(
                v["provenance"]["start_kind"].is_string(),
                "run provenance is missing"
            );
            Ok(Some(s))
        }
        None => anyhow::bail!("summary has no eligible flag"),
    }
}

/// Same-seed update-0 pairing integrity check. All learning rates of one seed start
/// from identical weights, so their update-0 ACTIVE evaluations must agree. This is
/// an integrity check only; it never ranks learning rates.
pub fn pairing_check(
    summaries: &[(f64, u64, serde_json::Value)],
    tol: f64,
) -> anyhow::Result<serde_json::Value> {
    let mut seeds = Vec::new();
    let mut worst = 0.0f64;
    for seed in SCREEN_SEEDS {
        let docs: Vec<(f64, &serde_json::Value)> = summaries
            .iter()
            .filter(|(_, s, v)| *s == seed && v["evaluations"]["0"].is_object())
            .map(|(lr, _, v)| (*lr, &v["evaluations"]["0"]))
            .collect();
        if docs.len() < 2 {
            seeds.push(
                serde_json::json!({"seed": seed, "skipped": "fewer than two update-0 evaluations"}),
            );
            continue;
        }
        let flat = |d: &serde_json::Value| -> anyhow::Result<Vec<f64>> {
            let mut v = vec![s_run_from_eval(d)?];
            for a in d["active"].as_array().into_iter().flatten() {
                for c in a["summary"]["cells"].as_object().into_iter().flatten() {
                    v.push(c.1["ce"].as_f64().unwrap_or(f64::NAN));
                }
            }
            Ok(v)
        };
        let base = flat(docs[0].1)?;
        let mut max_diff = 0.0f64;
        for (_, d) in &docs[1..] {
            let other = flat(d)?;
            anyhow::ensure!(other.len() == base.len(), "seed {seed}: shape mismatch");
            for (x, y) in base.iter().zip(&other) {
                let diff = (x - y).abs();
                anyhow::ensure!(diff.is_finite(), "seed {seed}: non-finite update-0 CE");
                max_diff = max_diff.max(diff);
            }
        }
        worst = worst.max(max_diff);
        seeds.push(serde_json::json!({
            "seed": seed,
            "lrs": docs.iter().map(|(lr, _)| *lr).collect::<Vec<_>>(),
            "S_run_update0": docs.iter().map(|(_, d)| d["screen_score_S_run"].clone()).collect::<Vec<_>>(),
            "values_compared": base.len(),
            "max_abs_diff": max_diff,
            "bitwise_identical": max_diff == 0.0,
        }));
    }
    anyhow::ensure!(
        worst <= tol,
        "same-seed update-0 evaluations disagree by {worst:e} (> {tol:e}): STOP and investigate pairing/initialization"
    );
    Ok(serde_json::json!({
        "schema": "v3_p5_pairing_check_v1",
        "purpose": "integrity only: same-seed LRs share an initialization; update-0 performance is never used to choose an LR",
        "tolerance": tol,
        "max_abs_diff_all_seeds": worst,
        "seeds": seeds,
    }))
}

const PAIRING_TOL: f64 = 1e-6;

fn run_select(a: &SelectArgs) -> anyhow::Result<()> {
    for out in [&a.selection_out, &a.selected_recipe_out] {
        anyhow::ensure!(
            !out.exists(),
            "{} already exists: the selection rule is applied once",
            out.display()
        );
    }
    let contract = load_contract(&a.recipe)?;
    let mut runs = Vec::new();
    let mut rows = Vec::new();
    let mut docs = Vec::new();
    for lr in CANDIDATE_LRS {
        for seed in SCREEN_SEEDS {
            let p = a.summaries.join(format!("{}.json", run_stem(lr, seed)));
            let v: serde_json::Value = serde_json::from_slice(
                &std::fs::read(&p).map_err(|e| anyhow::anyhow!("{}: {e}", p.display()))?,
            )?;
            let s = validate_summary(&v, lr, seed, &contract)
                .map_err(|e| anyhow::anyhow!("{}: {e}", p.display()))?;
            rows.push(
                serde_json::json!({"lr": lr, "seed": seed, "eligible": s.is_some(), "S_run": s,
                "ineligible_reason": v["ineligible_reason"]}),
            );
            runs.push((lr, seed, s));
            docs.push((lr, seed, v));
        }
    }
    // Integrity first: a pairing disagreement stops the process before the rule runs.
    let pairing = pairing_check(&docs, PAIRING_TOL);
    match &pairing {
        Ok(doc) => write_json(&a.pairing_out, doc)?,
        Err(e) => {
            write_json(
                &a.pairing_out,
                &serde_json::json!({"schema": "v3_p5_pairing_check_v1", "passed": false, "error": format!("{e:#}")}),
            )?;
        }
    }
    pairing?;
    let (winner, scores) = select_lr(&runs)?;
    let doc = serde_json::json!({
        "schema": "v3_p5_lr_selection_v1",
        "rule": "S_run = mean over budgets {0,2,4,8} of mean over 6 TUNE cells of ACTIVE policy CE at update 800 (24 equal values); S_lr = mean of the two seed S_run; lowest S_lr wins; exact tie -> lower LR; no tolerance, no override; FIXED/oracle/selector rates/B16/Gates are diagnostics only",
        "contract_digest": contract.contract_digest(),
        "runs": rows,
        "S_lr": scores.iter().map(|(lr, s)| serde_json::json!({"lr": lr, "S_lr": s})).collect::<Vec<_>>(),
        "selected_lr": winner,
    });
    let mut sel = contract.for_run(winner, 0);
    sel.seed = None;
    // The recipe is written first; the selection file (the other "applied once" gate)
    // last, so a crash between them leaves a leftover that blocks a silent re-run.
    write_json(
        &a.selected_recipe_out,
        &serde_json::json!({
            "schema": "v3_p5_selected_recipe_v1",
            "selected_lr": winner,
            "recipe_without_seed": sel,
            "digest_without_seed": sel.digest(),
            "note": "selected by the preregistered rule on TUNE; P6 is not authorised by this file",
        }),
    )?;
    write_json(&a.selection_out, &doc)?;
    println!("selected peak LR {winner:e}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn runs(v: &[(f64, f64, f64)]) -> Vec<(f64, u64, Option<f64>)> {
        v.iter()
            .flat_map(|&(lr, a, b)| [(lr, 5101, Some(a)), (lr, 5102, Some(b))])
            .collect()
    }

    #[test]
    fn the_lowest_mean_wins_and_ties_go_to_the_lower_lr() {
        let r = runs(&[(7.5e-5, 2.0, 2.2), (1.5e-4, 1.9, 2.1), (3.0e-4, 2.5, 1.4)]);
        let (w, s) = select_lr(&r).unwrap();
        assert_eq!(w, 3.0e-4);
        assert_eq!(s.len(), 3);
        // Exact tie: lower LR.
        let r = runs(&[(7.5e-5, 2.0, 2.0), (1.5e-4, 1.0, 3.0), (3.0e-4, 5.0, 5.0)]);
        assert_eq!(select_lr(&r).unwrap().0, 7.5e-5);
        // A seed-level win cannot rescue a worse mean.
        let r = runs(&[(7.5e-5, 1.0, 9.0), (1.5e-4, 2.0, 2.0), (3.0e-4, 3.0, 3.0)]);
        assert_eq!(select_lr(&r).unwrap().0, 1.5e-4);
    }

    #[test]
    fn an_ineligible_seed_removes_the_whole_lr_and_nothing_eligible_is_an_error() {
        let mut r = runs(&[(7.5e-5, 1.0, 1.0), (1.5e-4, 2.0, 2.0), (3.0e-4, 3.0, 3.0)]);
        r[0].2 = None;
        assert_eq!(select_lr(&r).unwrap().0, 1.5e-4);
        let none: Vec<_> = r.iter().map(|&(l, s, _)| (l, s, None)).collect();
        assert!(select_lr(&none).is_err());
        let nan = vec![(7.5e-5, 5101, Some(f64::NAN)); 2];
        assert!(select_lr(&nan).is_err());
    }

    #[test]
    fn layouts_and_run_names_are_stable() {
        assert_eq!(parse_layout("16x8").unwrap(), Layout::DEFAULT);
        assert_eq!(parse_layout("8x16").unwrap(), Layout::FALLBACK);
        assert!(parse_layout("32x4").is_err());
        assert_eq!(run_stem(1.5e-4, 5101), "v3-p5-run-lr1.5e-4-seed5101");
    }

    // ----- P5.1 summary validation, pairing, fail-closed directories -----

    type Case = (&'static str, fn(&mut serde_json::Value));

    fn contract() -> Recipe {
        Recipe::screen_contract(Layout::DEFAULT, true)
    }

    fn eval_doc(update: u64, ce: f64) -> serde_json::Value {
        let cells: serde_json::Map<String, serde_json::Value> = (0..6)
            .map(|i| {
                (
                    format!("c{i}"),
                    serde_json::json!({"ce": ce + i as f64 * 0.01}),
                )
            })
            .collect();
        let active: Vec<_> = BUDGETS
            .iter()
            .map(|&b| {
                serde_json::json!({"update": update, "summary":
                    {"budget": b, "selection": EvalSelection::Active.label(), "cells": cells}})
            })
            .collect();
        let mut d = serde_json::json!({"update": update, "active": active});
        let s = s_run_from_eval(&d).unwrap();
        d["screen_score_S_run"] = s.into();
        d
    }

    fn exposure(per: u64) -> serde_json::Value {
        serde_json::Value::Array(
            BUDGETS
                .iter()
                .map(|&b| {
                    let cells: Vec<_> = (0..15u64)
                        .map(|i| {
                            serde_json::json!({"cell": format!("x{i}"),
                                "examples_consumed": per / 15 + u64::from(i < per % 15)})
                        })
                        .collect();
                    serde_json::json!({"budget": b, "examples_drawn": per, "cells": cells,
                        "fraction_by_family": {}, "fraction_by_depth": {}})
                })
                .collect(),
        )
    }

    fn good_summary(lr: f64, seed: u64, ce: f64) -> serde_json::Value {
        let c = contract();
        let mut evals = serde_json::Map::new();
        for u in &c.eval_updates {
            evals.insert(u.to_string(), eval_doc(*u, ce + *u as f64 * -1e-4));
        }
        let s = evals["800"]["screen_score_S_run"].clone();
        serde_json::json!({
            "schema": "v3_p5_run_summary_v1", "lr": lr, "seed": seed,
            "recipe_digest": c.clone().for_run(lr, seed).digest(),
            "contract_digest": c.contract_digest(),
            "eligible": true, "ineligible_reason": null, "S_run": s,
            "S_run_update": 800, "updates_trained": 800,
            "evaluations": evals,
            "sampler_exposure": exposure(800 * 32),
            "provenance": {"start_kind": "fresh"},
        })
    }

    #[test]
    fn a_complete_summary_validates_and_every_defect_is_refused() {
        let c = contract();
        let good = good_summary(1.5e-4, 5101, 2.0);
        assert!(validate_summary(&good, 1.5e-4, 5101, &c).unwrap().is_some());
        // Wrong expectation (filename mix-up).
        assert!(validate_summary(&good, 3.0e-4, 5101, &c).is_err());
        assert!(validate_summary(&good, 1.5e-4, 5102, &c).is_err());
        let cases: Vec<Case> = vec![
            ("schema", |v| v["schema"] = "x".into()),
            ("lr field", |v| v["lr"] = 3.0e-4.into()),
            ("seed field", |v| v["seed"] = 5102.into()),
            ("contract digest", |v| v["contract_digest"] = "0".into()),
            ("recipe digest", |v| v["recipe_digest"] = "0".into()),
            ("updates trained", |v| v["updates_trained"] = 799.into()),
            ("S_run update", |v| v["S_run_update"] = 600.into()),
            ("S_run non-finite", |v| v["S_run"] = serde_json::Value::Null),
            ("S_run differs from eval", |v| v["S_run"] = 1.0.into()),
            ("missing eval", |v| {
                v["evaluations"].as_object_mut().unwrap().remove("400");
            }),
            ("extra eval", |v| {
                v["evaluations"]["900"] = v["evaluations"]["800"].clone();
            }),
            ("mislabelled eval", |v| {
                v["evaluations"]["600"]["update"] = 200.into()
            }),
            ("missing budget", |v| {
                v["evaluations"]["800"]["active"]
                    .as_array_mut()
                    .unwrap()
                    .pop();
            }),
            ("B16 appears", |v| {
                v["evaluations"]["800"]["active"][3]["summary"]["budget"] = 16.into();
            }),
            ("stored score tampered", |v| {
                v["evaluations"]["800"]["screen_score_S_run"] = 0.1.into();
            }),
            ("exposure missing", |v| {
                v["sampler_exposure"] = serde_json::Value::Null
            }),
            ("exposure short", |v| {
                v["sampler_exposure"][2]["examples_drawn"] = 25599.into()
            }),
            ("exposure unbalanced", |v| {
                v["sampler_exposure"][1]["cells"][0]["examples_consumed"] = 1.into();
            }),
            ("holdout reference", |v| v["note"] = "uses HOLDOUT_C".into()),
            ("provenance missing", |v| {
                v["provenance"] = serde_json::Value::Null
            }),
        ];
        for (name, mutate) in cases {
            let mut v = good.clone();
            mutate(&mut v);
            assert!(
                validate_summary(&v, 1.5e-4, 5101, &c).is_err(),
                "summary defect not refused: {name}"
            );
        }
    }

    #[test]
    fn an_ineligible_summary_needs_a_preregistered_class_and_no_score() {
        let c = contract();
        let mut v = serde_json::json!({
            "schema": "v3_p5_run_summary_v1", "lr": 7.5e-5, "seed": 5101,
            "recipe_digest": c.clone().for_run(7.5e-5, 5101).digest(),
            "contract_digest": c.contract_digest(),
            "eligible": false, "ineligible_reason": "non-finite loss at update 3",
            "ineligible_class": "non_finite",
        });
        assert_eq!(validate_summary(&v, 7.5e-5, 5101, &c).unwrap(), None);
        for (k, bad) in [
            ("ineligible_reason", serde_json::json!("")),
            ("ineligible_class", serde_json::json!("it was worse")),
            ("S_run", serde_json::json!(1.0)),
        ] {
            let mut w = v.clone();
            w[k] = bad;
            assert!(validate_summary(&w, 7.5e-5, 5101, &c).is_err(), "{k}");
        }
        v.as_object_mut().unwrap().remove("ineligible_class");
        assert!(validate_summary(&v, 7.5e-5, 5101, &c).is_err());
        assert_eq!(
            ineligibility_class("non-finite gradient norm"),
            Some("non_finite")
        );
        assert_eq!(ineligibility_class("disk full"), None);
    }

    #[test]
    fn same_seed_update0_evaluations_must_agree_and_never_rank() {
        let mk = |ce: f64| {
            let mut v = good_summary(1.5e-4, 5101, 2.0);
            v["evaluations"]["0"] = eval_doc(0, ce);
            v
        };
        let docs = |d: [f64; 6]| -> Vec<(f64, u64, serde_json::Value)> {
            CANDIDATE_LRS
                .iter()
                .flat_map(|&l| SCREEN_SEEDS.iter().map(move |&s| (l, s)))
                .enumerate()
                .map(|(i, (lr, seed))| (lr, seed, mk(d[i])))
                .collect()
        };
        // Seed 5101 runs share 3.5, seed 5102 runs share 3.7 (different inits are fine).
        let ok = docs([3.5, 3.7, 3.5, 3.7, 3.5, 3.7]);
        let r = pairing_check(&ok, PAIRING_TOL).unwrap();
        assert_eq!(r["max_abs_diff_all_seeds"], 0.0);
        let bad = docs([3.5, 3.7, 3.5, 3.7, 3.6, 3.7]);
        assert!(pairing_check(&bad, PAIRING_TOL).is_err());
    }

    #[test]
    fn a_fresh_model_is_never_started_in_a_nonempty_run_directory() {
        let d = std::env::temp_dir().join(format!("recur64-p5-empty-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        require_empty_run_dir(&d).unwrap();
        std::fs::write(d.join("eval-u0000.json"), b"{}").unwrap();
        let e = require_empty_run_dir(&d).unwrap_err().to_string();
        assert!(e.contains("eval-u0000.json") && e.contains("quarantine"));
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn select_refuses_when_either_output_already_exists() {
        let d = std::env::temp_dir().join(format!("recur64-p5-once-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        for existing in ["sel.json", "rec.json"] {
            std::fs::write(d.join(existing), b"{}").unwrap();
            let a = SelectArgs {
                summaries: d.clone(),
                recipe: d.join("none.json"),
                selection_out: d.join("sel.json"),
                selected_recipe_out: d.join("rec.json"),
                pairing_out: d.join("pair.json"),
            };
            let e = run_select(&a).unwrap_err().to_string();
            assert!(e.contains("already exists"), "{e}");
            std::fs::remove_file(d.join(existing)).unwrap();
        }
        std::fs::remove_dir_all(&d).unwrap();
    }
}
