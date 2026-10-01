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
use recur64_runtime::p5::train::{P5State, Trainer};

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
}

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

fn train<TB: AutodiffBackend>(a: &TrainArgs, gpu: bool) -> anyhow::Result<()> {
    let device: TB::Device = Default::default();
    let inner = Default::default();
    model_io::verify_device::<TB::InnerBackend>(&inner)?;
    anyhow::ensure!(
        CANDIDATE_LRS.contains(&a.lr) && SCREEN_SEEDS.contains(&a.seed),
        "the screen runs only the preregistered LRs {CANDIDATE_LRS:?} and seeds {SCREEN_SEEDS:?}"
    );
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
    let started = Instant::now();

    let (result, gpu_samples) = monitor(gpu, || -> anyhow::Result<()> {
        let mut tr = match latest_state(&a.run_dir, &digest)? {
            Some(d) => {
                eprintln!("resuming from {}", d.display());
                Trainer::<TB>::load(&d, recipe.clone(), &train_ds, &device)?
            }
            None => Trainer::<TB>::new(recipe.clone(), &train_ds, &device)?,
        };
        loop {
            let u = tr.updates_done;
            if recipe.eval_updates.contains(&u) && !eval_path(&a.run_dir, u).exists() {
                let model = tr.inference_model();
                let doc = run_eval_at(&model, &tune, u, u == recipe.updates, &inner)?;
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
        Ok(())
    });

    let mut summary = serde_json::json!({
        "schema": "v3_p5_run_summary_v1",
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
        Ok(()) => {
            let mut evals = serde_json::Map::new();
            for u in &recipe.eval_updates {
                evals.insert(u.to_string(), load_eval_summaries(&a.run_dir, *u)?);
            }
            let final_doc = &evals[&recipe.updates.to_string()];
            let hist: Vec<recur64_runtime::p5::train::UpdateRecord> =
                serde_json::from_slice(&std::fs::read(a.run_dir.join("history.json"))?)?;
            o.insert("eligible".into(), true.into());
            o.insert("ineligible_reason".into(), serde_json::Value::Null);
            o.insert("S_run".into(), final_doc["screen_score_S_run"].clone());
            o.insert("S_run_update".into(), recipe.updates.into());
            o.insert("updates_trained".into(), (hist.len() as u64).into());
            o.insert(
                "train_wall_s".into(),
                hist.iter().map(|h| h.wall_s).sum::<f64>().into(),
            );
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
            let ineligible = msg.contains("non-finite")
                || msg.contains("health")
                || msg.contains("query")
                || msg.contains("checkpoint")
                || msg.contains("refus");
            if !ineligible {
                return Err(e);
            }
            o.insert("eligible".into(), false.into());
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

fn run_select(a: &SelectArgs) -> anyhow::Result<()> {
    anyhow::ensure!(
        !a.selection_out.exists(),
        "{} already exists: the selection rule is applied once",
        a.selection_out.display()
    );
    let contract = load_contract(&a.recipe)?;
    let mut runs = Vec::new();
    let mut rows = Vec::new();
    for lr in CANDIDATE_LRS {
        for seed in SCREEN_SEEDS {
            let p = a.summaries.join(format!("{}.json", run_stem(lr, seed)));
            let v: serde_json::Value = serde_json::from_slice(
                &std::fs::read(&p).map_err(|e| anyhow::anyhow!("{}: {e}", p.display()))?,
            )?;
            anyhow::ensure!(
                v["contract_digest"] == contract.contract_digest(),
                "{} was run under a different recipe",
                p.display()
            );
            let eligible = v["eligible"] == true;
            let s = eligible.then(|| v["S_run"].as_f64()).flatten();
            rows.push(
                serde_json::json!({"lr": lr, "seed": seed, "eligible": eligible, "S_run": s,
                "ineligible_reason": v["ineligible_reason"]}),
            );
            runs.push((lr, seed, s));
        }
    }
    let (winner, scores) = select_lr(&runs)?;
    let doc = serde_json::json!({
        "schema": "v3_p5_lr_selection_v1",
        "rule": "S_run = mean over budgets {0,2,4,8} of mean over 6 TUNE cells of ACTIVE policy CE at update 800 (24 equal values); S_lr = mean of the two seed S_run; lowest S_lr wins; exact tie -> lower LR; no tolerance, no override; FIXED/oracle/selector rates/B16/Gates are diagnostics only",
        "contract_digest": contract.contract_digest(),
        "runs": rows,
        "S_lr": scores.iter().map(|(lr, s)| serde_json::json!({"lr": lr, "S_lr": s})).collect::<Vec<_>>(),
        "selected_lr": winner,
    });
    write_json(&a.selection_out, &doc)?;
    let selected = contract.for_run(winner, 0);
    let mut sel = selected;
    sel.seed = None;
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
}
