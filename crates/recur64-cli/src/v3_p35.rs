//! `recur64 v3-p35 ...`: the V3.5 on-policy information-acquisition rescue.
//!
//! * `preflight`: TRAIN-only real V3.5 updates at full geometry (init from a P5 final
//!   checkpoint), timing the detached rollout and the autodiff replay;
//! * `train`: one resumable seed run (800 updates) with the preregistered TUNE
//!   evaluations; update 800 runs the full final evaluation;
//! * `eval-final`: the full update-800 evaluation of an existing final checkpoint;
//! * `gate`: apply Gate II / Gate III / Content-Use / Gate VI exactly once.
//!
//! Only the frozen TRAIN and v3_tune_v1 datasets are accepted. HOLDOUT_C is never
//! opened. A requested device that is unavailable is an error, never a substitution.

use std::path::{Path, PathBuf};
use std::time::Instant;

use burn::tensor::backend::AutodiffBackend;
use clap::{Args, Subcommand};

use recur64_model::active::ActiveSearchModel;
use recur64_runtime::gpu_telemetry::{monitor, sample_gpu};
use recur64_runtime::model_io;
use recur64_runtime::p5::ablation::evaluate_query_content_ablation;
use recur64_runtime::p5::data::{Dataset, Expected, load_dataset};
use recur64_runtime::p5::eval::{EvalOutput, EvalSelection, cell_key, evaluate};
use recur64_runtime::p5::recipe::BUDGETS;
use recur64_runtime::p35::gate::{self, Outcome};
use recur64_runtime::p35::recipe::{INIT_CHECKPOINTS, Recipe35};
use recur64_runtime::p35::train::{P35State, Trainer35, UpdateRecord35};

use crate::v3_p5::{
    CHECKPOINT_EVERY, DataArgs, Dev, EVAL_BATCH, dispatch, parse_layout, require_empty_run_dir,
    state_dirs, write_json,
};

/// The primary cell of every gate.
pub const GATE_CELL: &str = "KQRvK M3";
pub const SEEDS: [u64; 3] = [5101, 5102, 5103];
const REPLAY_VS_SOURCE_TOL: f64 = 1e-4;
/// Gate VI VRAM stability: peak after update 100 within 5% of the peak up to update 100.
const VRAM_PLATEAU_FRACTION: f64 = 0.05;

#[derive(Subcommand, Debug)]
pub enum P35Cmd {
    /// TRAIN-only preflight at full geometry (no TUNE).
    Preflight(PreflightArgs),
    /// One seed run (resumable) with the preregistered TUNE evaluations.
    Train(TrainArgs),
    /// Full update-800 evaluation of an existing final checkpoint.
    EvalFinal(EvalFinalArgs),
    /// Apply the pre-registered gates, once.
    Gate(GateArgs),
}

#[derive(Args, Debug)]
pub struct PreflightArgs {
    #[command(flatten)]
    pub data: DataArgs,
    #[arg(long, value_enum)]
    pub device: Dev,
    #[arg(long)]
    pub seed: u64,
    /// The seed's selected P5 final directory (holds `checkpoint/` and `p5-state.json`).
    #[arg(long)]
    pub init_dir: PathBuf,
    #[arg(long, default_value = "16x8")]
    pub layout: String,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub health_checks: bool,
    #[arg(long, default_value_t = 3)]
    pub updates: u64,
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
    #[arg(long)]
    pub seed: u64,
    #[arg(long)]
    pub init_dir: PathBuf,
    #[arg(long, value_enum)]
    pub device: Dev,
    /// The resolved layout frozen by the preflight commit (`16x8` or `8x16`).
    #[arg(long)]
    pub layout: String,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub health_checks: bool,
    #[arg(long)]
    pub run_dir: PathBuf,
}

#[derive(Args, Debug)]
pub struct EvalFinalArgs {
    #[command(flatten)]
    pub data: DataArgs,
    #[arg(long)]
    pub tune: PathBuf,
    #[arg(long)]
    pub tune_trace: PathBuf,
    #[arg(long)]
    pub seed: u64,
    #[arg(long, value_enum)]
    pub device: Dev,
    #[arg(long)]
    pub layout: String,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub health_checks: bool,
    #[arg(long)]
    pub run_dir: PathBuf,
}

#[derive(Args, Debug)]
pub struct GateArgs {
    /// Directory holding `v35-run-seed{5101,5102,5103}/`.
    #[arg(long)]
    pub runs_root: PathBuf,
    #[arg(long)]
    pub output: PathBuf,
}

pub fn run_stem(seed: u64) -> String {
    format!("v35-run-seed{seed}")
}

pub fn run(cmd: P35Cmd) -> anyhow::Result<()> {
    match cmd {
        P35Cmd::Preflight(a) => {
            let dev = a.device;
            dispatch(dev, move |d| match d {
                crate::v3_p5::Dispatch::Cpu => {
                    preflight::<recur64_model::train::CpuTrainBackend>(&a, false)
                }
                #[cfg(feature = "cuda")]
                crate::v3_p5::Dispatch::Cuda => {
                    preflight::<burn::backend::Autodiff<burn::backend::Cuda>>(&a, true)
                }
            })
        }
        P35Cmd::Train(a) => {
            let dev = a.device;
            dispatch(dev, move |d| match d {
                crate::v3_p5::Dispatch::Cpu => {
                    train::<recur64_model::train::CpuTrainBackend>(&a, false)
                }
                #[cfg(feature = "cuda")]
                crate::v3_p5::Dispatch::Cuda => {
                    train::<burn::backend::Autodiff<burn::backend::Cuda>>(&a, true)
                }
            })
        }
        P35Cmd::EvalFinal(a) => {
            let dev = a.device;
            dispatch(dev, move |d| match d {
                crate::v3_p5::Dispatch::Cpu => {
                    eval_final::<recur64_model::train::CpuTrainBackend>(&a)
                }
                #[cfg(feature = "cuda")]
                crate::v3_p5::Dispatch::Cuda => {
                    eval_final::<burn::backend::Autodiff<burn::backend::Cuda>>(&a)
                }
            })
        }
        P35Cmd::Gate(a) => run_gate(&a),
    }
}

fn recipe_for(layout: &str, health_checks: bool, seed: u64) -> anyhow::Result<Recipe35> {
    let r = Recipe35::contract(parse_layout(layout)?, health_checks).for_seed(seed)?;
    r.validate_for_training()?;
    Ok(r)
}

fn eval_path(dir: &Path, update: u64) -> PathBuf {
    dir.join(format!("eval-u{update:04}.json"))
}

fn latest_state(dir: &Path, digest: &str) -> anyhow::Result<Option<PathBuf>> {
    let mut best: Option<(u64, PathBuf)> = None;
    for d in state_dirs(dir) {
        let side = d.join("p35-state.json");
        if !side.exists() {
            continue;
        }
        let st: P35State = serde_json::from_slice(&std::fs::read(&side)?)?;
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

fn eval_json(out: &EvalOutput) -> serde_json::Value {
    serde_json::json!({
        "summary": out.summary,
        "selector_diag": out.selector_diag.as_ref().map(|(all, cells)| serde_json::json!({
            "pooled": all, "cells": cells,
        })),
        "refined_diag": out.refined_diag.as_ref().map(|(all, cells)| serde_json::json!({
            "pooled": all, "cells": cells,
        })),
    })
}

/// Intermediate (diagnostic-only) evaluation: ACTIVE at every training budget.
fn mid_eval<B: burn::tensor::backend::Backend>(
    model: &ActiveSearchModel<B>,
    tune: &Dataset,
    update: u64,
    digest: &str,
    device: &B::Device,
) -> anyhow::Result<serde_json::Value> {
    let t0 = Instant::now();
    let mut active = Vec::new();
    for b in BUDGETS {
        let out = evaluate(model, tune, b, EvalSelection::Active, EVAL_BATCH, device)?;
        active.push(eval_json(&out));
    }
    Ok(serde_json::json!({
        "update": update,
        "recipe_digest": digest,
        "kind": "diagnostic_only",
        "active": active,
        "eval_wall_s": t0.elapsed().as_secs_f64(),
    }))
}

/// The update-800 evaluation: ACTIVE B0/2/4/8, FIXED B2/4/8, `query_content_ablation_v1`
/// on the ACTIVE paths at B2/4/8, selector diagnostics, and the gate-cell per-position
/// arrays the paired estimators consume.
fn final_eval<B: burn::tensor::backend::Backend>(
    model: &ActiveSearchModel<B>,
    tune: &Dataset,
    update: u64,
    digest: &str,
    device: &B::Device,
) -> anyhow::Result<(serde_json::Value, serde_json::Value)> {
    let t0 = Instant::now();
    let cell: Vec<usize> = tune
        .positions()
        .iter()
        .enumerate()
        .filter(|(_, p)| cell_key(p) == GATE_CELL)
        .map(|(i, _)| i)
        .collect();
    anyhow::ensure!(!cell.is_empty(), "no {GATE_CELL} positions in TUNE");
    let pick = |v: &[recur64_runtime::p5::eval::ExampleResult],
                f: fn(&recur64_runtime::p5::eval::ExampleResult) -> f64| {
        cell.iter().map(|&i| f(&v[i])).collect::<Vec<f64>>()
    };
    let mut active = Vec::new();
    let (mut b0_top1, mut a8_top1) = (Vec::new(), Vec::new());
    for b in BUDGETS {
        let out = evaluate(model, tune, b, EvalSelection::Active, EVAL_BATCH, device)?;
        if b == 0 {
            b0_top1 = pick(&out.per_position, |r| r.top1);
        }
        if b == 8 {
            a8_top1 = pick(&out.per_position, |r| r.top1);
        }
        active.push(eval_json(&out));
    }
    let mut fixed = Vec::new();
    let mut f8_top1 = Vec::new();
    for b in [2usize, 4, 8] {
        let out = evaluate(model, tune, b, EvalSelection::Fixed, EVAL_BATCH, device)?;
        if b == 8 {
            f8_top1 = pick(&out.per_position, |r| r.top1);
        }
        fixed.push(eval_json(&out));
    }
    let mut ablation = Vec::new();
    let (mut ce_n8, mut ce_a8) = (Vec::new(), Vec::new());
    for b in [2usize, 4, 8] {
        let r = evaluate_query_content_ablation(
            model,
            tune,
            b,
            EvalSelection::Active,
            EVAL_BATCH,
            device,
        )?;
        anyhow::ensure!(
            r.replay_vs_source_max_abs_ce_diff < REPLAY_VS_SOURCE_TOL,
            "B{b}: the ablation replay does not reproduce the ACTIVE source path ({:e})",
            r.replay_vs_source_max_abs_ce_diff
        );
        if b == 8 {
            ce_n8 = pick(&r.normal_per_position, |x| x.ce);
            ce_a8 = pick(&r.ablated_per_position, |x| x.ce);
        }
        ablation.push(serde_json::to_value(&r)?);
    }
    let doc = serde_json::json!({
        "update": update,
        "recipe_digest": digest,
        "kind": "final",
        "active": active,
        "fixed_B2_B4_B8": fixed,
        "query_content_ablation_active_paths": ablation,
        "eval_wall_s": t0.elapsed().as_secs_f64(),
    });
    let ids: Vec<&str> = cell
        .iter()
        .map(|&i| tune.positions()[i].id.as_str())
        .collect();
    let perpos = serde_json::json!({
        "schema": "v35_gate_cell_per_position_v1",
        "cell": GATE_CELL,
        "update": update,
        "recipe_digest": digest,
        "position_ids": ids,
        "b0_top1": b0_top1,
        "active_b8_top1": a8_top1,
        "fixed_b8_top1": f8_top1,
        "ce_normal_b8": ce_n8,
        "ce_ablated_b8": ce_a8,
    });
    Ok((doc, perpos))
}

fn preflight<TB: AutodiffBackend>(a: &PreflightArgs, gpu: bool) -> anyhow::Result<()> {
    let device: TB::Device = Default::default();
    let inner = Default::default();
    model_io::verify_device::<TB::InnerBackend>(&inner)?;
    let recipe = recipe_for(&a.layout, a.health_checks, a.seed)?;
    let train = load_dataset(&a.data.train, &a.data.train_trace, &Expected::train())?;
    eprintln!("TRAIN verified: {} positions", train.positions().len());
    let (result, gpu_samples) = monitor(gpu, || -> anyhow::Result<serde_json::Value> {
        let mut tr = Trainer35::<TB>::new(recipe.clone(), &train, &a.init_dir, &device)?;
        let mut recs: Vec<UpdateRecord35> = Vec::new();
        for _ in 0..a.updates {
            let rec = tr.step(&train, &device)?;
            eprintln!(
                "update {} wall {:.1}s (rollout {:.1}s replay+backward {:.1}s) loss {:.4} sup {} replay-diff {:.2e}",
                rec.update,
                rec.wall_s,
                rec.report.rollout_s,
                rec.report.replay_backward_s,
                rec.report.total_loss,
                rec.report.supervised_decisions,
                rec.report.max_replay_policy_diff
            );
            recs.push(rec);
        }
        Ok(serde_json::to_value(&recs)?)
    });
    let doc = match result {
        Ok(v) => serde_json::json!({
            "ok": true,
            "tested": "real TRAIN updates at full geometry with live StateQuery from the seed's P5 final weights",
            "device": if gpu { "cuda" } else { "cpu" },
            "layout": a.layout,
            "seed": a.seed,
            "recipe_digest": recipe.digest(),
            "updates": v,
            "gpu": gpu_samples,
        }),
        Err(e) => {
            write_json(
                &a.output,
                &serde_json::json!({"ok": false, "layout": a.layout, "error": format!("{e:#}")}),
            )?;
            return Err(e);
        }
    };
    write_json(&a.output, &doc)?;
    println!("preflight ok: {}", a.output.display());
    Ok(())
}

fn train<TB: AutodiffBackend>(a: &TrainArgs, gpu: bool) -> anyhow::Result<()> {
    let device: TB::Device = Default::default();
    let inner = Default::default();
    model_io::verify_device::<TB::InnerBackend>(&inner)?;
    let recipe = recipe_for(&a.layout, a.health_checks, a.seed)?;
    let digest = recipe.digest();
    let train_ds = load_dataset(&a.data.train, &a.data.train_trace, &Expected::train())?;
    let tune = load_dataset(&a.tune, &a.tune_trace, &Expected::tune())?;
    eprintln!(
        "TRAIN {} / TUNE {} verified; run digest {digest}",
        train_ds.positions().len(),
        tune.positions().len()
    );
    std::fs::create_dir_all(&a.run_dir)?;
    let resume_from = latest_state(&a.run_dir, &digest)?;
    if resume_from.is_none() {
        require_empty_run_dir(&a.run_dir)?;
    }
    let started = Instant::now();
    let eval_updates = recipe.base.eval_updates.clone();
    let final_update = recipe.base.updates;
    let (result, gpu_samples) = monitor(gpu, || -> anyhow::Result<serde_json::Value> {
        let mut tr = match &resume_from {
            Some(d) => {
                eprintln!("resuming from {}", d.display());
                Trainer35::<TB>::load(d, recipe.clone(), &train_ds, &device)?
            }
            None => Trainer35::<TB>::new(recipe.clone(), &train_ds, &a.init_dir, &device)?,
        };
        let mut vram: Vec<(u64, u64)> = Vec::new();
        loop {
            let u = tr.updates_done;
            if eval_updates.contains(&u) && !eval_path(&a.run_dir, u).exists() {
                let model = tr.inference_model();
                if u == final_update {
                    let (doc, perpos) = final_eval(&model, &tune, u, &digest, &inner)?;
                    write_json(&a.run_dir.join("perpos-u0800.json"), &perpos)?;
                    write_json(&eval_path(&a.run_dir, u), &doc)?;
                } else {
                    let doc = mid_eval(&model, &tune, u, &digest, &inner)?;
                    write_json(&eval_path(&a.run_dir, u), &doc)?;
                }
                eprintln!("update {u}: TUNE evaluation written");
            }
            if u >= final_update {
                break;
            }
            let rec = tr.step(&train_ds, &device)?;
            if rec.update % 10 == 0 || rec.update < 3 {
                eprintln!(
                    "update {} lr {:.3e} wall {:.1}s (rollout {:.1}s) loss {:.4} (policy {:.4} selector {:.4}) grad {:.3} sup {} replay-diff {:.1e}",
                    rec.update,
                    rec.lr,
                    rec.wall_s,
                    rec.report.rollout_s,
                    rec.report.total_loss,
                    rec.report.policy_loss,
                    rec.report.selector_loss,
                    rec.report.grad_norm,
                    rec.report.supervised_decisions,
                    rec.report.max_replay_policy_diff
                );
            }
            if tr.updates_done % CHECKPOINT_EVERY == 0 || eval_updates.contains(&tr.updates_done) {
                let dirs = state_dirs(&a.run_dir);
                let dir = &dirs[((tr.updates_done / CHECKPOINT_EVERY) % 2) as usize];
                std::fs::create_dir_all(dir)?;
                tr.save(dir)?;
                if gpu && let Some((mem, _, _)) = sample_gpu() {
                    vram.push((tr.updates_done, mem));
                }
            }
        }
        let dir = a.run_dir.join("final");
        std::fs::create_dir_all(&dir)?;
        tr.save(&dir)?;
        write_json(
            &a.run_dir.join("history.json"),
            &serde_json::to_value(&tr.history)?,
        )?;
        Ok(serde_json::json!({
            "vram_mb_at_checkpoints": vram,
            "resumptions": tr.resumptions,
            "sampler_exposure": tr.samplers.stats().iter().map(|(b, s)| {
                let mut v = serde_json::to_value(s).expect("stats serialise");
                v.as_object_mut().expect("object").insert("budget".into(), (*b).into());
                v
            }).collect::<Vec<_>>(),
        }))
    });
    let extra = result?;
    let hist: Vec<UpdateRecord35> =
        serde_json::from_slice(&std::fs::read(a.run_dir.join("history.json"))?)?;
    anyhow::ensure!(
        hist.len() as u64 == final_update,
        "history holds {} updates, expected {final_update}",
        hist.len()
    );
    let finite = hist.iter().all(|h| {
        let r = &h.report;
        h.lr.is_finite()
            && r.policy_loss.is_finite()
            && r.selector_loss.is_finite()
            && r.total_loss.is_finite()
            && r.grad_norm.is_finite()
    });
    let summary = serde_json::json!({
        "schema": "v35_run_summary_v1",
        "seed": a.seed,
        "recipe_digest": digest,
        "contract_digest": recipe.contract_digest(),
        "device": if gpu { "cuda" } else { "cpu" },
        "layout": a.layout,
        "health_checks": recipe.base.health_checks,
        "updates_trained": hist.len(),
        "train_wall_s": hist.iter().map(|h| h.wall_s).sum::<f64>(),
        "rollout_wall_s": hist.iter().map(|h| h.report.rollout_s).sum::<f64>(),
        "replay_backward_wall_s": hist.iter().map(|h| h.report.replay_backward_s).sum::<f64>(),
        "wall_s_this_invocation": started.elapsed().as_secs_f64(),
        "gate6_inputs": {
            "history_all_finite": finite,
            "max_replay_policy_diff": hist.iter().map(|h| h.report.max_replay_policy_diff).fold(0.0, f64::max),
            "replay_tolerance": recipe.replay_policy_tolerance,
            "supervised_total": hist.iter().map(|h| h.report.supervised_decisions).sum::<usize>(),
            "vram_mb_at_checkpoints": extra["vram_mb_at_checkpoints"],
            "gpu": gpu_samples,
        },
        "sampler_exposure": extra["sampler_exposure"],
        "provenance": {"resumptions": extra["resumptions"]},
        "loss_curve": hist.iter().step_by(10).map(|h| serde_json::json!({
            "update": h.update, "lr": h.lr, "policy": h.report.policy_loss,
            "selector": h.report.selector_loss, "grad_norm": h.report.grad_norm,
            "supervised": h.report.supervised_decisions,
        })).collect::<Vec<_>>(),
    });
    write_json(&a.run_dir.join("summary.json"), &summary)?;
    println!("seed {} done: {} updates", a.seed, hist.len());
    Ok(())
}

fn eval_final<TB: AutodiffBackend>(a: &EvalFinalArgs) -> anyhow::Result<()> {
    let device: TB::Device = Default::default();
    let inner = Default::default();
    model_io::verify_device::<TB::InnerBackend>(&inner)?;
    let recipe = recipe_for(&a.layout, a.health_checks, a.seed)?;
    let digest = recipe.digest();
    let train_ds = load_dataset(&a.data.train, &a.data.train_trace, &Expected::train())?;
    let tune = load_dataset(&a.tune, &a.tune_trace, &Expected::tune())?;
    let tr = Trainer35::<TB>::load(&a.run_dir.join("final"), recipe.clone(), &train_ds, &device)?;
    anyhow::ensure!(
        tr.updates_done == recipe.base.updates,
        "not an update-800 checkpoint"
    );
    for f in ["eval-u0800.json", "perpos-u0800.json"] {
        anyhow::ensure!(
            !a.run_dir.join(f).exists(),
            "{f} already exists: the update-800 evaluation is written once"
        );
    }
    let (doc, perpos) = final_eval(
        &tr.inference_model(),
        &tune,
        tr.updates_done,
        &digest,
        &inner,
    )?;
    write_json(&a.run_dir.join("perpos-u0800.json"), &perpos)?;
    write_json(&a.run_dir.join("eval-u0800.json"), &doc)?;
    println!("final evaluation written for seed {}", a.seed);
    Ok(())
}

// ---------------------------------------------------------------------------
// Gates
// ---------------------------------------------------------------------------

fn column(v: &serde_json::Value, key: &str) -> anyhow::Result<Vec<f64>> {
    v[key]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("per-position file lacks {key}"))?
        .iter()
        .map(|x| {
            x.as_f64()
                .ok_or_else(|| anyhow::anyhow!("{key} holds a non-number"))
        })
        .collect()
}

fn cell_metrics(eval: &serde_json::Value, group: &str, idx: usize) -> serde_json::Value {
    let m = &eval[group][idx]["summary"]["cells"][GATE_CELL];
    serde_json::json!({
        "top1": m["top1"], "correct_mass": m["correct_mass"], "ce": m["ce"], "entropy": m["entropy"],
    })
}

/// Gate VI from a run summary (pure; all inputs recorded by `train`).
pub fn gate6_seed(summary: &serde_json::Value, recipe: &Recipe35) -> (bool, Vec<String>) {
    let g = &summary["gate6_inputs"];
    let mut fails = Vec::new();
    if summary["updates_trained"] != recipe.base.updates {
        fails.push("run did not complete 800 updates".into());
    }
    if g["history_all_finite"] != true {
        fails.push("non-finite loss or gradient in the history".into());
    }
    if g["max_replay_policy_diff"]
        .as_f64()
        .is_none_or(|d| d > recipe.replay_policy_tolerance)
    {
        fails.push("detached-rollout / autodiff-replay parity not within tolerance".into());
    }
    if summary["health_checks"] != true {
        fails.push("health checks were off".into());
    }
    if summary["device"] == "cuda" {
        let v: Vec<(u64, f64)> = g["vram_mb_at_checkpoints"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|p| Some((p[0].as_u64()?, p[1].as_f64()?)))
                    .collect()
            })
            .unwrap_or_default();
        let early = v
            .iter()
            .filter(|(u, _)| *u <= 100)
            .map(|p| p.1)
            .fold(0.0, f64::max);
        let late = v
            .iter()
            .filter(|(u, _)| *u > 100)
            .map(|p| p.1)
            .fold(0.0, f64::max);
        if v.is_empty() || early <= 0.0 {
            fails.push("VRAM was not measured: stability is NOT TESTED".into());
        } else if late > (1.0 + VRAM_PLATEAU_FRACTION) * early {
            fails.push(format!(
                "VRAM grew from {early} to {late} MB after update 100"
            ));
        }
    } else {
        fails.push("not a CUDA run: resident lifecycle / VRAM stability NOT TESTED".into());
    }
    (fails.is_empty(), fails)
}

fn run_gate(a: &GateArgs) -> anyhow::Result<()> {
    anyhow::ensure!(
        !a.output.exists(),
        "{} already exists: the gates are applied exactly once",
        a.output.display()
    );
    let (mut b0, mut a8, mut f8, mut cen, mut cea) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut seeds_json = Vec::new();
    let mut g6_all = true;
    let mut g6_json = Vec::new();
    let mut ids: Option<serde_json::Value> = None;
    for seed in SEEDS {
        let dir = a.runs_root.join(run_stem(seed));
        let summary: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("summary.json"))?)?;
        let perpos: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("perpos-u0800.json"))?)?;
        let eval: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("eval-u0800.json"))?)?;
        let recipe = recipe_for(
            summary["layout"].as_str().unwrap_or("16x8"),
            summary["health_checks"].as_bool().unwrap_or(true),
            seed,
        )?;
        anyhow::ensure!(
            summary["recipe_digest"] == recipe.digest().as_str()
                && perpos["recipe_digest"] == recipe.digest().as_str()
                && eval["recipe_digest"] == recipe.digest().as_str()
                && perpos["update"] == recipe.base.updates,
            "seed {seed}: artifacts do not belong to the frozen recipe at update 800"
        );
        match &ids {
            None => ids = Some(perpos["position_ids"].clone()),
            Some(i) => anyhow::ensure!(
                *i == perpos["position_ids"],
                "seeds cover different positions"
            ),
        }
        b0.push(column(&perpos, "b0_top1")?);
        a8.push(column(&perpos, "active_b8_top1")?);
        f8.push(column(&perpos, "fixed_b8_top1")?);
        cen.push(column(&perpos, "ce_normal_b8")?);
        cea.push(column(&perpos, "ce_ablated_b8")?);
        let (ok6, fails) = gate6_seed(&summary, &recipe);
        g6_all &= ok6;
        g6_json.push(serde_json::json!({"seed": seed, "pass": ok6, "failures": fails}));
        let abl8 = &eval["query_content_ablation_active_paths"][2];
        seeds_json.push(serde_json::json!({
            "seed": seed,
            "gate_cell_active": {
                "B0": cell_metrics(&eval, "active", 0), "B2": cell_metrics(&eval, "active", 1),
                "B4": cell_metrics(&eval, "active", 2), "B8": cell_metrics(&eval, "active", 3),
            },
            "gate_cell_fixed": {
                "B2": cell_metrics(&eval, "fixed_B2_B4_B8", 0), "B4": cell_metrics(&eval, "fixed_B2_B4_B8", 1),
                "B8": cell_metrics(&eval, "fixed_B2_B4_B8", 2),
            },
            "gate_cell_ablation_active_B8": {
                "normal": abl8["normal_state_content"]["cells"][GATE_CELL],
                "ablated": abl8["ablated_query_state_content"]["cells"][GATE_CELL],
                "positions_top1_changed_pooled": abl8["positions_top1_changed_by_ablation"],
            },
            "selector_diag_B8_gate_cell": eval["active"][3]["selector_diag"]["cells"][GATE_CELL],
            "refined_diag_B8_gate_cell": eval["active"][3]["refined_diag"]["cells"][GATE_CELL],
            "ablation_B2_B4_secondary": {
                "B2": {"normal": eval["query_content_ablation_active_paths"][0]["normal_state_content"]["cells"][GATE_CELL],
                       "ablated": eval["query_content_ablation_active_paths"][0]["ablated_query_state_content"]["cells"][GATE_CELL]},
                "B4": {"normal": eval["query_content_ablation_active_paths"][1]["normal_state_content"]["cells"][GATE_CELL],
                       "ablated": eval["query_content_ablation_active_paths"][1]["ablated_query_state_content"]["cells"][GATE_CELL]},
            },
        }));
    }
    let g2 = gate::gate2(&a8, &b0)?;
    let g3 = gate::gate3(&a8, &f8)?;
    let content = gate::content_use(&cea, &cen)?;
    let outcome: Outcome = gate::classify(content.pass, g2.pass, g3.pass, g6_all);

    // Attribution ratio on the pooled seed means: (normal_B8 - ablated_B8) / (normal_B8 - B0).
    // The ablated top-1 is read from the committed ablation summaries (gate cell).
    let mean = |v: &Vec<Vec<f64>>| {
        v.iter().flatten().sum::<f64>() / v.iter().map(Vec::len).sum::<usize>() as f64
    };
    let ablated_top1: f64 = seeds_json
        .iter()
        .map(|s| {
            s["gate_cell_ablation_active_B8"]["ablated"]["top1"]
                .as_f64()
                .unwrap_or(f64::NAN)
        })
        .sum::<f64>()
        / 3.0;
    let (n8, b0m) = (mean(&a8), mean(&b0));
    let attribution = if n8 - b0m > 0.0 {
        serde_json::json!((n8 - ablated_top1) / (n8 - b0m))
    } else {
        serde_json::Value::Null
    };
    let doc = serde_json::json!({
        "schema": "v35_gates_v1",
        "cell": GATE_CELL,
        "dataset": "V3_TUNE_V1 only; HOLDOUT_C NOT EVALUATED",
        "init_checkpoints": INIT_CHECKPOINTS,
        "gate_I_historical": "PASS (immutable; V3-P6)",
        "gate_II": g2,
        "gate_III": g3,
        "content_use": content,
        "gate_VI": {"pass": g6_all, "seeds": g6_json},
        "attribution_ratio": attribution,
        "outcome": outcome,
        "seeds": seeds_json,
    });
    write_json(&a.output, &doc)?;
    println!(
        "Gate II {} (delta {:+.4}, CI [{:+.4},{:+.4}]); Gate III {} (delta {:+.4}); Content-Use {} (mean dCE {:+.5}); Gate VI {}; outcome {:?}",
        if g2.pass { "PASS" } else { "FAIL" },
        g2.delta,
        g2.ci_lower,
        g2.ci_upper,
        if g3.pass { "PASS" } else { "FAIL" },
        g3.delta,
        if content.pass { "PASS" } else { "FAIL" },
        content.delta,
        if g6_all { "PASS" } else { "FAIL" },
        outcome
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn summary(device: &str, vram: serde_json::Value, diff: f64) -> serde_json::Value {
        serde_json::json!({
            "updates_trained": 800, "health_checks": true, "device": device,
            "gate6_inputs": {
                "history_all_finite": true, "max_replay_policy_diff": diff,
                "vram_mb_at_checkpoints": vram,
            },
        })
    }

    fn recipe() -> Recipe35 {
        recipe_for("16x8", true, 5101).unwrap()
    }

    #[test]
    fn gate_vi_requires_measured_stable_vram_and_replay_parity() {
        let stable = serde_json::json!([
            [50, 10000.0],
            [100, 10200.0],
            [150, 10300.0],
            [800, 10400.0]
        ]);
        assert!(gate6_seed(&summary("cuda", stable, 1e-6), &recipe()).0);
        let growing = serde_json::json!([[50, 10000.0], [100, 10100.0], [800, 12000.0]]);
        assert!(!gate6_seed(&summary("cuda", growing, 1e-6), &recipe()).0);
        // Unmeasured VRAM and CPU runs are NOT TESTED, which is not a pass.
        assert!(!gate6_seed(&summary("cuda", serde_json::json!([]), 1e-6), &recipe()).0);
        assert!(!gate6_seed(&summary("cpu", serde_json::json!([]), 1e-6), &recipe()).0);
        let ok = serde_json::json!([[50, 10000.0], [800, 10000.0]]);
        assert!(
            !gate6_seed(&summary("cuda", ok, 1e-2), &recipe()).0,
            "replay parity"
        );
    }

    #[test]
    fn the_gate_is_applied_once_and_a_fresh_model_never_starts_in_a_used_directory() {
        let d = std::env::temp_dir().join("recur64_p35_cli_once");
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        let out = d.join("gates.json");
        std::fs::write(&out, b"{}").unwrap();
        let e = run_gate(&GateArgs {
            runs_root: d.clone(),
            output: out,
        })
        .unwrap_err();
        assert!(e.to_string().contains("exactly once"), "{e}");
        let used = d.join("run");
        std::fs::create_dir_all(&used).unwrap();
        std::fs::write(used.join("eval-u0000.json"), b"{}").unwrap();
        assert!(require_empty_run_dir(&used).is_err());
    }

    #[test]
    fn only_the_three_frozen_seeds_and_layouts_build_a_recipe() {
        assert!(recipe_for("16x8", true, 5101).is_ok());
        assert!(recipe_for("8x16", true, 5103).is_ok());
        assert!(recipe_for("16x8", true, 7).is_err());
        assert!(recipe_for("32x4", true, 5101).is_err());
    }
}
