//! `recur64 v3-p6 ...`: the ALL-INFO information-sufficiency control and Gate I.
//!
//! * `census`: TRAIN-only depth-2 state census;
//! * `recipe`: write the P6 contract (resolved physical layout);
//! * `preflight`: TRAIN-only real updates at full geometry on the requested device;
//! * `train`: one resumable ALL-INFO run (seed), final TUNE evaluation with per-position results;
//! * `b0-reference`: evaluate the paired ACTIVE checkpoints at B0 with per-position results;
//! * `gate`: apply the frozen Gate I rule, once.
//!
//! Only the frozen TRAIN and V3_TUNE_V1 splits are loadable (`load_targets`). HOLDOUT_C is
//! never opened. A requested device that is unavailable is an error, never a substitution.

use std::path::{Path, PathBuf};
use std::time::Instant;

use burn::tensor::backend::AutodiffBackend;
use clap::{Args, Subcommand};

use recur64_model::active::coverage::{
    INERT_NOISE_BOUND, gradient_coverage, is_inherited_inert_key_bias,
};
use recur64_runtime::gpu_telemetry::{monitor, sample_gpu};
use recur64_runtime::model_io;
use recur64_runtime::p5::data::{Expected, load_dataset};
use recur64_runtime::p5::eval::{EvalSelection, ExampleResult, cell_key, evaluate};
use recur64_runtime::p5::recipe::{Recipe, TUNE_POSITIONS};
use recur64_runtime::p5::train::Trainer;
use recur64_runtime::p6::census::census;
use recur64_runtime::p6::data::{Which, load_targets};
use recur64_runtime::p6::eval::evaluate_all_info;
use recur64_runtime::p6::gate::{BOOTSTRAP_SEED, RESAMPLES, THRESHOLD, gate1};
use recur64_runtime::p6::recipe::{GATE_CELL, LAYOUT_LADDER, P6_SEEDS, P6Recipe, SELECTED_LR};
use recur64_runtime::p6::train::{P6Trainer, compute_update};

use crate::v3_p5::{
    CHECKPOINT_EVERY, Dev, Dispatch, dispatch, ineligibility_class, require_empty_run_dir,
    state_dirs, write_json,
};

/// TUNE evaluation batch of the ALL-INFO model (positions per forward).
const P6_EVAL_BATCH: usize = 8;
/// Frozen two-hour per-run projection limit of the systems fallback rule.
const WALL_LIMIT_S: f64 = 2.0 * 3600.0;
/// Frozen VRAM-plateau tolerance and fraction-of-device limit of the layout rule.
const VRAM_PLATEAU_FRACTION: f64 = 0.05;
const VRAM_DEVICE_FRACTION: f64 = 0.95;
/// Absolute tolerance for reproducing committed ACTIVE B0 values.
const B0_TOL: f64 = 1e-4;

#[derive(Subcommand, Debug)]
pub enum P6Cmd {
    /// TRAIN-only depth-2 state census.
    Census(CensusArgs),
    /// Write the P6 contract for a physical layout.
    Recipe(RecipeArgs),
    /// TRAIN-only preflight at full geometry.
    Preflight(PreflightArgs),
    /// One ALL-INFO run (resumable) with the final TUNE evaluation.
    Train(TrainArgs),
    /// Evaluate the paired ACTIVE checkpoints at B0 (per-position results).
    B0Reference(B0Args),
    /// Apply the frozen Gate I rule once.
    Gate(GateArgs),
}

#[derive(Args, Debug)]
pub struct CensusArgs {
    #[arg(long)]
    pub train: PathBuf,
    #[arg(long)]
    pub output: PathBuf,
}

#[derive(Args, Debug)]
pub struct RecipeArgs {
    /// `MxA` on the frozen ladder, e.g. `16x8`.
    #[arg(long, default_value = "16x8")]
    pub layout: String,
    #[arg(long)]
    pub output: PathBuf,
}

#[derive(Args, Debug)]
pub struct PreflightArgs {
    #[arg(long)]
    pub train: PathBuf,
    #[arg(long, value_enum)]
    pub device: Dev,
    #[arg(long)]
    pub layout: String,
    /// Real TRAIN updates to time (the first is cold).
    #[arg(long, default_value_t = 4)]
    pub updates: u64,
    /// TRAIN positions used to time evaluation.
    #[arg(long, default_value_t = 32)]
    pub eval_positions: usize,
    #[arg(long)]
    pub output: PathBuf,
}

#[derive(Args, Debug)]
pub struct TrainArgs {
    #[arg(long)]
    pub train: PathBuf,
    #[arg(long)]
    pub tune: PathBuf,
    /// The committed P6 contract (`v3-p6-recipe.json`).
    #[arg(long)]
    pub recipe: PathBuf,
    #[arg(long)]
    pub seed: u64,
    #[arg(long, value_enum)]
    pub device: Dev,
    #[arg(long)]
    pub run_dir: PathBuf,
    #[arg(long)]
    pub summary: PathBuf,
}

#[derive(Args, Debug)]
pub struct B0Args {
    #[command(flatten)]
    pub data: crate::v3_p5::DataArgs,
    #[arg(long)]
    pub tune: PathBuf,
    #[arg(long)]
    pub tune_trace: PathBuf,
    /// The committed P5 screen contract (`v3-p5-recipe.json`).
    #[arg(long)]
    pub recipe: PathBuf,
    #[arg(long, value_enum)]
    pub device: Dev,
    /// Directory holding `v3-p5-run-lr3e-4-seed{5101,5102}/final`.
    #[arg(long)]
    pub p5_runs: PathBuf,
    /// Directory holding `v3-p5-run-lr3e-4-seed5103/final`.
    #[arg(long)]
    pub p6_runs: PathBuf,
    /// Committed run summaries (`docs/evidence/v3`), used to verify each reproduction.
    #[arg(long)]
    pub evidence: PathBuf,
    /// Directory receiving `b0-seed{seed}.json`.
    #[arg(long)]
    pub output_dir: PathBuf,
}

#[derive(Args, Debug)]
pub struct GateArgs {
    #[arg(long)]
    pub tune: PathBuf,
    /// Directory holding `b0-seed{seed}.json`.
    #[arg(long)]
    pub b0_dir: PathBuf,
    /// Directory holding `allinfo-seed{seed}.json` (per-position results).
    #[arg(long)]
    pub allinfo_dir: PathBuf,
    #[arg(long)]
    pub recipe: PathBuf,
    #[arg(long)]
    pub output: PathBuf,
}

pub fn parse_layout(s: &str) -> anyhow::Result<(usize, usize)> {
    let (m, a) = s
        .split_once('x')
        .ok_or_else(|| anyhow::anyhow!("layout '{s}': expected MxA"))?;
    let l = (m.parse::<usize>()?, a.parse::<usize>()?);
    anyhow::ensure!(
        LAYOUT_LADDER.contains(&l),
        "layout {s} is not on the frozen ladder {LAYOUT_LADDER:?}"
    );
    Ok(l)
}

pub fn run(cmd: P6Cmd) -> anyhow::Result<()> {
    match cmd {
        P6Cmd::Census(a) => run_census(&a),
        P6Cmd::Recipe(a) => run_recipe(&a),
        P6Cmd::Preflight(a) => {
            let dev = a.device;
            dispatch(dev, move |d| match d {
                Dispatch::Cpu => preflight::<recur64_model::train::CpuTrainBackend>(&a, false),
                #[cfg(feature = "cuda")]
                Dispatch::Cuda => {
                    preflight::<burn::backend::Autodiff<burn::backend::Cuda>>(&a, true)
                }
            })
        }
        P6Cmd::Train(a) => {
            let dev = a.device;
            dispatch(dev, move |d| match d {
                Dispatch::Cpu => train::<recur64_model::train::CpuTrainBackend>(&a, false),
                #[cfg(feature = "cuda")]
                Dispatch::Cuda => train::<burn::backend::Autodiff<burn::backend::Cuda>>(&a, true),
            })
        }
        P6Cmd::B0Reference(a) => {
            let dev = a.device;
            dispatch(dev, move |d| match d {
                Dispatch::Cpu => b0_reference::<recur64_model::train::CpuTrainBackend>(&a),
                #[cfg(feature = "cuda")]
                Dispatch::Cuda => b0_reference::<burn::backend::Autodiff<burn::backend::Cuda>>(&a),
            })
        }
        P6Cmd::Gate(a) => run_gate(&a),
    }
}

fn run_census(a: &CensusArgs) -> anyhow::Result<()> {
    let t = load_targets(&a.train, Which::Train)?;
    let started = Instant::now();
    let c = census(&t)?;
    anyhow::ensure!(
        c.max_child_legal <= 256,
        "a depth-1 state has {} legal moves, above the StateQuery cap of 256 (the tool would refuse it; no truncation is allowed)",
        c.max_child_legal
    );
    write_json(
        &a.output,
        &serde_json::json!({
            "schema": "v3_p6_state_census_v1",
            "label": "TRAIN only (P25_DATA_V1); no TUNE or HOLDOUT_C position was read",
            "input_contract": recur64_model::config::ALL_INFO_INPUT,
            "dataset_digest": t.digest,
            "count_method": "authoritative move generator (GameState); equals the exhaustive tree builder (tested)",
            "census_wall_s": started.elapsed().as_secs_f64(),
            "census": c,
        }),
    )?;
    println!(
        "census: {} positions; future states median {} p95 {} max {}; {} state transitions",
        c.positions,
        c.total_future_states.median,
        c.total_future_states.p95,
        c.total_future_states.max,
        c.total_state_transitions
    );
    Ok(())
}

fn run_recipe(a: &RecipeArgs) -> anyhow::Result<()> {
    let (m, ac) = parse_layout(&a.layout)?;
    let r = P6Recipe::contract(m, ac).for_seed(P6_SEEDS[0]);
    r.validate()?;
    let mut contract = P6Recipe::contract(m, ac);
    contract.seed = None;
    write_json(
        &a.output,
        &serde_json::json!({
            "schema": "v3_p6_recipe_file_v1",
            "contract_digest": contract.contract_digest(),
            "selected_lr": SELECTED_LR,
            "seeds": P6_SEEDS,
            "layout": {"micro": m, "accum": ac},
            "ladder": LAYOUT_LADDER,
            "recipe": contract,
        }),
    )?;
    println!("P6 contract digest {}", contract.contract_digest());
    Ok(())
}

fn load_p6_contract(path: &Path) -> anyhow::Result<P6Recipe> {
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
    anyhow::ensure!(
        v["schema"] == "v3_p6_recipe_file_v1",
        "not a P6 recipe file"
    );
    let r: P6Recipe = serde_json::from_value(v["recipe"].clone())?;
    anyhow::ensure!(r.seed.is_none(), "{} is not a P6 contract", path.display());
    anyhow::ensure!(
        v["contract_digest"] == r.contract_digest(),
        "the recipe file's digest does not match its contents"
    );
    r.clone().for_seed(P6_SEEDS[0]).validate()?;
    Ok(r)
}

/// Total device memory in MiB, if `nvidia-smi` reports it (recorded, never assumed).
fn gpu_total_mb() -> Option<u64> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .next()?
        .trim()
        .parse()
        .ok()
}

fn preflight<TB: AutodiffBackend>(a: &PreflightArgs, gpu: bool) -> anyhow::Result<()> {
    let device: TB::Device = Default::default();
    let inner = Default::default();
    model_io::verify_device::<TB::InnerBackend>(&inner)?;
    let (m, ac) = parse_layout(&a.layout)?;
    let recipe = P6Recipe::contract(m, ac).for_seed(P6_SEEDS[0]);
    recipe.validate()?;
    let train_ds = load_targets(&a.train, Which::Train)?;
    eprintln!("TRAIN verified: {} positions", train_ds.positions.len());

    let (result, gpu_samples) = monitor(gpu, || -> anyhow::Result<serde_json::Value> {
        let mut tr = P6Trainer::<TB>::new(recipe.clone(), &train_ds, &device)?;
        let params = tr.model.num_params();
        let (mut walls, mut builds, mut models, mut vram) = (vec![], vec![], vec![], vec![]);
        let (mut states, mut max_micro) = (vec![], 0usize);
        let mut updates = Vec::new();
        for _ in 0..a.updates {
            let rec = tr.step(&train_ds, &device)?;
            eprintln!(
                "update {} wall {:.1}s (trees {:.1}s, model {:.1}s) loss {:.4} grad {:.3} states {} max-micro {}",
                rec.update,
                rec.wall_s,
                rec.report.tree_build_s,
                rec.report.model_s,
                rec.report.policy_loss,
                rec.report.grad_norm,
                rec.report.states_supplied,
                rec.report.max_micro_states
            );
            walls.push(rec.wall_s);
            builds.push(rec.report.tree_build_s);
            models.push(rec.report.model_s);
            states.push(rec.report.states_supplied);
            max_micro = max_micro.max(rec.report.max_micro_states);
            vram.push(sample_gpu().map(|(mem, _, _)| mem));
            updates.push(serde_json::to_value(&rec)?);
        }
        // Gradient coverage of a real full-geometry update.
        let idx: Vec<usize> = (0..recipe.effective_batch).collect();
        let (grads, _) = compute_update(&tr.model, &train_ds, &idx, &recipe, &device)?;
        let rows = gradient_coverage::<TB, _>(&tr.model, &grads);
        let mut uncovered = Vec::new();
        for r in &rows {
            anyhow::ensure!(r.finite, "non-finite gradient in {}", r.name);
            // Exempt: the WDL head (no WDL loss in P6) and key biases (a constant added
            // to every key cancels in the softmax; true gradient exactly zero).
            let inert = r.name.starts_with("root.wdl.")
                || r.name.ends_with(".k_proj.bias")
                || is_inherited_inert_key_bias(&r.name);
            if inert {
                anyhow::ensure!(
                    r.max_abs <= INERT_NOISE_BOUND,
                    "{} should be inert but has gradient {}",
                    r.name,
                    r.max_abs
                );
            } else if !r.nonzero {
                uncovered.push(r.name.clone());
            }
        }
        anyhow::ensure!(
            uncovered.is_empty(),
            "parameters without a gradient in a real update: {uncovered:?}"
        );

        // Evaluation timing on a strided TRAIN subset (TUNE is not touched).
        let sub = {
            let total = train_ds.positions.len();
            let stride = (total / a.eval_positions.max(1)).max(1);
            let mut t = train_ds.clone();
            t.positions = (0..total)
                .step_by(stride)
                .take(a.eval_positions)
                .map(|i| train_ds.positions[i].clone())
                .collect();
            t
        };
        let model = tr.inference_model();
        let t0 = Instant::now();
        evaluate_all_info(&model, &sub, P6_EVAL_BATCH, &inner)?;
        let eval_per_position = t0.elapsed().as_secs_f64() / sub.positions.len() as f64;

        let steady: Vec<f64> = walls.iter().skip(1).copied().collect();
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
        let steady_mean = if steady.is_empty() {
            mean(&walls)
        } else {
            mean(&steady)
        };
        let train_s = steady_mean * recipe.updates as f64;
        let eval_s = eval_per_position * TUNE_POSITIONS as f64;
        let ckpt_s = (recipe.updates / CHECKPOINT_EVERY) as f64 * 15.0;
        let total = train_s + eval_s + ckpt_s;
        let warm: Vec<u64> = vram.iter().skip(2).flatten().copied().collect();
        let plateau = warm.len() < 2
            || (warm.iter().max().copied().unwrap_or(0) - warm.iter().min().copied().unwrap_or(0))
                as f64
                <= VRAM_PLATEAU_FRACTION * warm.iter().max().copied().unwrap_or(1) as f64;
        Ok(serde_json::json!({
            "parameters": params,
            "updates": updates,
            "steady_update_wall_s_mean": steady_mean,
            "cold_update_wall_s": walls.first(),
            "tree_build_s_mean": mean(&builds),
            "model_s_mean": mean(&models),
            "states_per_update_mean": states.iter().sum::<usize>() as f64 / states.len().max(1) as f64,
            "max_states_in_one_microbatch": max_micro,
            "vram_mb_after_each_update": vram,
            "vram_plateau_stable": plateau,
            "eval_s_per_position": eval_per_position,
            "gradient_coverage": {"parameter_tensors": rows.len(), "uncovered": uncovered},
            "projection": {
                "basis": "steady-state mean update wall x 800 + one TUNE evaluation (per-position cost measured on a TRAIN subset x 4500) + 16 checkpoints assumed 15 s each",
                "train_s": train_s,
                "evaluation_s": eval_s,
                "checkpoint_s_assumed": ckpt_s,
                "single_run_total_s": total,
                "single_run_total_h": total / 3600.0,
                "limit_h": WALL_LIMIT_S / 3600.0,
                "within_limit": total < WALL_LIMIT_S,
            },
        }))
    });
    let total_mb = if gpu { gpu_total_mb() } else { None };
    let mut report = match result {
        Ok(v) => v,
        Err(e) => {
            // A visible failure (for example a genuine out-of-memory): recorded, non-zero exit.
            write_json(
                &a.output,
                &serde_json::json!({"ok": false, "qualifies": false, "layout": a.layout, "error": format!("{e:#}")}),
            )?;
            return Err(e);
        }
    };
    let peak = gpu_samples.peak_vram_mb;
    let fits = match (peak, total_mb) {
        (Some(p), Some(t)) => (p as f64) <= VRAM_DEVICE_FRACTION * t as f64,
        _ => !gpu,
    };
    let within = report["projection"]["within_limit"] == true;
    let plateau = report["vram_plateau_stable"] == true;
    let o = report.as_object_mut().expect("object");
    o.insert("ok".into(), true.into());
    o.insert("qualifies".into(), (fits && within && plateau).into());
    o.insert(
        "qualification_rule".into(),
        serde_json::json!({
            "runs_correctly": true,
            "peak_vram_within_fraction_of_device": VRAM_DEVICE_FRACTION,
            "vram_plateau_fraction": VRAM_PLATEAU_FRACTION,
            "projected_run_limit_h": WALL_LIMIT_S / 3600.0,
            "fits_vram": fits,
            "vram_plateau_stable": plateau,
            "within_time_limit": within,
        }),
    );
    o.insert(
        "tested".into(),
        "real TRAIN updates at full geometry with live exhaustive StateQuery trees on the requested device".into(),
    );
    o.insert(
        "device".into(),
        format!("{:?}", if gpu { Dev::Cuda } else { Dev::Cpu }).into(),
    );
    o.insert(
        "layout".into(),
        serde_json::json!({"micro": m, "accum": ac}),
    );
    o.insert("gpu_total_mb".into(), total_mb.into());
    o.insert("gpu".into(), serde_json::to_value(&gpu_samples)?);
    o.insert("recipe_contract_digest".into(), {
        let mut c = recipe.clone();
        c.seed = None;
        c.digest().into()
    });
    write_json(&a.output, &report)?;
    println!(
        "layout {}: qualifies={} (fits {fits}, plateau {plateau}, projected {:.2} h)",
        a.layout,
        report["qualifies"],
        report["projection"]["single_run_total_h"]
            .as_f64()
            .unwrap_or(f64::NAN)
    );
    Ok(())
}

fn latest_state(dir: &Path, digest: &str) -> anyhow::Result<Option<PathBuf>> {
    let mut best: Option<(u64, PathBuf)> = None;
    for d in state_dirs(dir) {
        let side = d.join("p6-state.json");
        if !side.exists() {
            continue;
        }
        let st: recur64_runtime::p6::train::P6State =
            serde_json::from_slice(&std::fs::read(&side)?)?;
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

fn perpos_json(
    seed: u64,
    digest: &str,
    positions: &[recur64_runtime::proof::targets::ProofPosition],
    results: &[ExampleResult],
    label: &str,
) -> serde_json::Value {
    serde_json::json!({
        "schema": "v3_p6_per_position_v1",
        "label": label,
        "seed": seed,
        "recipe_digest": digest,
        "dataset_digest": recur64_runtime::p5::recipe::TUNE_DIGEST,
        "ids": positions.iter().map(|p| p.id.clone()).collect::<Vec<_>>(),
        "cells": positions.iter().map(cell_key).collect::<Vec<_>>(),
        "top1": results.iter().map(|r| r.top1).collect::<Vec<_>>(),
        "correct_mass": results.iter().map(|r| r.mass).collect::<Vec<_>>(),
        "ce": results.iter().map(|r| r.ce).collect::<Vec<_>>(),
        "entropy": results.iter().map(|r| r.entropy).collect::<Vec<_>>(),
    })
}

fn train<TB: AutodiffBackend>(a: &TrainArgs, gpu: bool) -> anyhow::Result<()> {
    let device: TB::Device = Default::default();
    let inner = Default::default();
    model_io::verify_device::<TB::InnerBackend>(&inner)?;
    anyhow::ensure!(
        P6_SEEDS.contains(&a.seed),
        "P6 trains only the paired seeds {P6_SEEDS:?}"
    );
    let recipe = load_p6_contract(&a.recipe)?.for_seed(a.seed);
    let digest = recipe.digest();
    let train_ds = load_targets(&a.train, Which::Train)?;
    let tune = load_targets(&a.tune, Which::Tune)?;
    eprintln!(
        "TRAIN {} / TUNE {} verified; run digest {digest}",
        train_ds.positions.len(),
        tune.positions.len()
    );
    std::fs::create_dir_all(&a.run_dir)?;
    let resume_from = latest_state(&a.run_dir, &digest)?;
    if resume_from.is_none() {
        require_empty_run_dir(&a.run_dir)?;
    }
    let started = Instant::now();

    let (result, gpu_samples) = monitor(gpu, || -> anyhow::Result<serde_json::Value> {
        let mut tr = match &resume_from {
            Some(d) => {
                eprintln!("resuming from {}", d.display());
                P6Trainer::<TB>::load(d, recipe.clone(), &train_ds, &device)?
            }
            None => P6Trainer::<TB>::new(recipe.clone(), &train_ds, &device)?,
        };
        let start_update = tr.updates_done;
        while tr.updates_done < recipe.updates {
            let rec = tr.step(&train_ds, &device)?;
            if rec.update % 10 == 0 || rec.update < 3 {
                eprintln!(
                    "update {} lr {:.3e} wall {:.1}s (trees {:.1}s) loss {:.4} grad {:.3} states {}",
                    rec.update,
                    rec.lr,
                    rec.wall_s,
                    rec.report.tree_build_s,
                    rec.report.policy_loss,
                    rec.report.grad_norm,
                    rec.report.states_supplied
                );
            }
            if tr.updates_done % CHECKPOINT_EVERY == 0 {
                let dirs = state_dirs(&a.run_dir);
                let dir = &dirs[((tr.updates_done / CHECKPOINT_EVERY) % 2) as usize];
                std::fs::create_dir_all(dir)?;
                tr.save(dir)?;
            }
        }
        let dir = a.run_dir.join("final");
        std::fs::create_dir_all(&dir)?;
        tr.save(&dir)?;
        write_json(
            &a.run_dir.join("history.json"),
            &serde_json::to_value(&tr.history)?,
        )?;
        // The one TUNE evaluation of this model: the final update, no checkpoint selection.
        let model = tr.inference_model();
        let t0 = Instant::now();
        let (results, summary, states) = evaluate_all_info(&model, &tune, P6_EVAL_BATCH, &inner)?;
        let eval_s = t0.elapsed().as_secs_f64();
        write_json(
            &a.run_dir.join(format!("allinfo-seed{}.json", a.seed)),
            &perpos_json(
                a.seed,
                &digest,
                &tune.positions,
                &results,
                "ALL-INFO per-position TUNE results at update 800",
            ),
        )?;
        Ok(serde_json::json!({
            "tune_summary": summary,
            "tune_states_supplied": states,
            "tune_eval_wall_s": eval_s,
            "provenance": {
                "start_kind": if resume_from.is_some() { "resumed" } else { "fresh" },
                "start_update": start_update,
                "resumptions_including_this_invocation": tr.resumptions,
                "prior_resumptions": tr.resumptions.saturating_sub(u32::from(resume_from.is_some())),
            },
        }))
    });

    let mut summary = serde_json::json!({
        "schema": "v3_p6_run_summary_v1",
        "identity": "p6_all_info_run",
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
            let hist: Vec<recur64_runtime::p6::train::UpdateRecord> =
                serde_json::from_slice(&std::fs::read(a.run_dir.join("history.json"))?)?;
            anyhow::ensure!(
                hist.len() as u64 == recipe.updates,
                "history holds {} updates, expected {}",
                hist.len(),
                recipe.updates
            );
            let tail: Vec<f64> = hist
                .iter()
                .rev()
                .take(100)
                .map(|h| h.report.policy_loss)
                .collect();
            o.insert("eligible".into(), true.into());
            o.insert("ineligible_reason".into(), serde_json::Value::Null);
            o.insert("updates_trained".into(), (hist.len() as u64).into());
            o.insert(
                "train_wall_s".into(),
                hist.iter().map(|h| h.wall_s).sum::<f64>().into(),
            );
            o.insert(
                "tree_build_wall_s".into(),
                hist.iter()
                    .map(|h| h.report.tree_build_s)
                    .sum::<f64>()
                    .into(),
            );
            o.insert(
                "states_supplied_per_update_mean".into(),
                (hist.iter().map(|h| h.report.states_supplied).sum::<usize>() as f64
                    / hist.len() as f64)
                    .into(),
            );
            o.insert(
                "max_states_in_one_microbatch".into(),
                hist.iter().map(|h| h.report.max_micro_states).max().into(),
            );
            o.insert(
                "final_train_policy_loss_mean_last_100_updates".into(),
                (tail.iter().sum::<f64>() / tail.len() as f64).into(),
            );
            o.insert(
                "loss_curve".into(),
                serde_json::to_value(
                    hist.iter()
                        .step_by(10)
                        .map(|h| {
                            serde_json::json!({
                                "update": h.update, "lr": h.lr,
                                "policy": h.report.policy_loss, "grad_norm": h.report.grad_norm,
                            })
                        })
                        .collect::<Vec<_>>(),
                )?,
            );
            for (k, v) in extra.as_object().expect("object") {
                o.insert(k.clone(), v.clone());
            }
        }
        Err(e) => {
            let msg = format!("{e:#}");
            let Some(class) = ineligibility_class(&msg) else {
                return Err(e);
            };
            o.insert("eligible".into(), false.into());
            o.insert("ineligible_class".into(), class.into());
            o.insert("ineligible_reason".into(), msg.into());
            write_json(&a.summary, &summary)?;
            return Err(e);
        }
    }
    write_json(&a.summary, &summary)?;
    println!(
        "P6 ALL-INFO seed {}: trained 800 updates and evaluated TUNE once",
        a.seed
    );
    Ok(())
}

fn b0_reference<TB: AutodiffBackend>(a: &B0Args) -> anyhow::Result<()> {
    let device: TB::Device = Default::default();
    let inner = Default::default();
    model_io::verify_device::<TB::InnerBackend>(&inner)?;
    let contract = crate::v3_p5::load_contract(&a.recipe)?;
    let train_ds = load_dataset(&a.data.train, &a.data.train_trace, &Expected::train())?;
    let tune = load_dataset(&a.tune, &a.tune_trace, &Expected::tune())?;
    std::fs::create_dir_all(&a.output_dir)?;
    let mut manifest = Vec::new();
    for seed in P6_SEEDS {
        let recipe: Recipe = contract.clone().for_run(SELECTED_LR, seed);
        let digest = recipe.digest();
        let stem = crate::v3_p5::run_stem(SELECTED_LR, seed);
        let (root, committed_path) = if seed == 5103 {
            (&a.p6_runs, a.evidence.join(format!("{stem}.json")))
        } else {
            (&a.p5_runs, a.evidence.join(format!("{stem}.json")))
        };
        let committed: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&committed_path)?)?;
        anyhow::ensure!(
            committed["recipe_digest"] == digest.as_str() && committed["eligible"] == true,
            "{stem}: the committed summary is not this recipe's eligible run"
        );
        let final_dir = root.join(&stem).join("final");
        // Trainer::load verifies the sidecar digest, the checkpoint metadata and every
        // consistency invariant (architecture, seed, peak LR, update count) first.
        let tr = Trainer::<TB>::load(&final_dir, recipe.clone(), &train_ds, &device)?;
        anyhow::ensure!(
            tr.updates_done == recipe.updates
                && recipe.peak_lr == Some(SELECTED_LR)
                && recipe.seed == Some(seed),
            "{stem}: not the final update-800 checkpoint of the selected recipe"
        );
        let model = tr.inference_model();
        let out = evaluate(&model, &tune, 0, EvalSelection::Active, 64, &inner)?;
        // The re-evaluated B0 policy must reproduce the committed update-800 B0 evaluation.
        let fresh = serde_json::to_value(&out.summary)?;
        let diff = {
            let c = &committed["evaluations"]["800"]["active"][0]["summary"];
            anyhow::ensure!(c["budget"] == 0, "{stem}: committed B0 entry not found");
            let mut worst = 0.0f64;
            let mut groups: Vec<(&serde_json::Value, &serde_json::Value)> =
                vec![(&fresh["pooled"], &c["pooled"])];
            for (k, cell) in c["cells"].as_object().expect("cells") {
                groups.push((&fresh["cells"][k], cell));
            }
            for (f, cc) in groups {
                for key in ["top1", "correct_mass", "ce", "entropy"] {
                    worst = worst.max(
                        (f[key].as_f64().unwrap_or(f64::NAN)
                            - cc[key].as_f64().unwrap_or(f64::NAN))
                        .abs(),
                    );
                }
            }
            worst
        };
        anyhow::ensure!(
            diff <= B0_TOL,
            "{stem}: re-evaluated B0 differs from the committed evaluation by {diff:e} (> {B0_TOL:e}): STOP"
        );
        let doc = perpos_json(
            seed,
            &digest,
            tune.positions(),
            &out.per_position,
            "ACTIVE selected-recipe checkpoint at B0 (zero queries), update 800",
        );
        write_json(&a.output_dir.join(format!("b0-seed{seed}.json")), &doc)?;
        let gate_top1 = gate_cell_mean(&doc);
        eprintln!(
            "{stem}: B0 reproduces the committed evaluation (max |diff| {diff:e}); KQRvK M3 top-1 {gate_top1:.4}"
        );
        manifest.push(serde_json::json!({
            "seed": seed, "recipe_digest": digest, "run": stem, "checkpoint": "final (update 800)",
            "peak_lr": SELECTED_LR, "committed_reproduction_max_abs_diff": diff,
            "kqrvk_m3_top1": gate_top1, "pooled_b0": fresh["pooled"],
        }));
    }
    write_json(
        &a.output_dir.join("b0-manifest.json"),
        &serde_json::json!({
            "schema": "v3_p6_b0_manifest_v1",
            "selected_recipe_digest_without_seed": crate::v3_p5::P5_SELECTED_DIGEST,
            "contract_digest": contract.contract_digest(),
            "tolerance": B0_TOL,
            "runs": manifest,
        }),
    )?;
    println!("B0 references written to {}", a.output_dir.display());
    Ok(())
}

/// Mean top-1 over the gate cell of a per-position document.
fn gate_cell_mean(doc: &serde_json::Value) -> f64 {
    let (cells, top1) = (doc["cells"].as_array(), doc["top1"].as_array());
    let (Some(cells), Some(top1)) = (cells, top1) else {
        return f64::NAN;
    };
    let v: Vec<f64> = cells
        .iter()
        .zip(top1)
        .filter(|(c, _)| *c == GATE_CELL)
        .map(|(_, t)| t.as_f64().unwrap_or(f64::NAN))
        .collect();
    v.iter().sum::<f64>() / v.len().max(1) as f64
}

fn read_perpos(path: &Path, seed: u64, ids: &[String]) -> anyhow::Result<serde_json::Value> {
    let v: serde_json::Value = serde_json::from_slice(
        &std::fs::read(path).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?,
    )?;
    anyhow::ensure!(
        v["schema"] == "v3_p6_per_position_v1" && v["seed"] == seed,
        "{}: not the seed-{seed} per-position document",
        path.display()
    );
    anyhow::ensure!(
        v["dataset_digest"] == recur64_runtime::p5::recipe::TUNE_DIGEST,
        "{}: not V3_TUNE_V1 results",
        path.display()
    );
    let got: Vec<&str> = v["ids"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("no ids"))?
        .iter()
        .map(|x| x.as_str().unwrap_or(""))
        .collect();
    anyhow::ensure!(
        got.len() == ids.len() && got.iter().zip(ids).all(|(a, b)| a == b),
        "{}: positions are not V3_TUNE_V1 in dataset order",
        path.display()
    );
    Ok(v)
}

fn sha256_of(v: &serde_json::Value) -> String {
    use sha2::{Digest, Sha256};
    format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(v).expect("serialises"))
    )
}

fn run_gate(a: &GateArgs) -> anyhow::Result<()> {
    anyhow::ensure!(
        !a.output.exists(),
        "{} already exists: Gate I is applied once",
        a.output.display()
    );
    let contract = load_p6_contract(&a.recipe)?;
    let tune = load_targets(&a.tune, Which::Tune)?;
    anyhow::ensure!(tune.positions.len() == TUNE_POSITIONS, "TUNE size");
    let ids: Vec<String> = tune.positions.iter().map(|p| p.id.clone()).collect();
    let gate_idx: Vec<usize> = tune
        .positions
        .iter()
        .enumerate()
        .filter(|(_, p)| cell_key(p) == GATE_CELL)
        .map(|(i, _)| i)
        .collect();
    anyhow::ensure!(
        gate_idx.len() == 750,
        "{} gate-cell positions, expected 750",
        gate_idx.len()
    );
    let mut ai_top1 = Vec::new();
    let mut b0_top1 = Vec::new();
    let mut seeds_json = Vec::new();
    let mut diag = Vec::new();
    for seed in P6_SEEDS {
        let ai = read_perpos(
            &a.allinfo_dir.join(format!("allinfo-seed{seed}.json")),
            seed,
            &ids,
        )?;
        let b0 = read_perpos(&a.b0_dir.join(format!("b0-seed{seed}.json")), seed, &ids)?;
        anyhow::ensure!(
            ai["recipe_digest"] == contract.clone().for_seed(seed).digest().as_str(),
            "seed {seed}: ALL-INFO results are from another recipe"
        );
        let pick = |v: &serde_json::Value, key: &str| -> Vec<f64> {
            gate_idx
                .iter()
                .map(|&i| v[key][i].as_f64().unwrap_or(f64::NAN))
                .collect()
        };
        let mean_all = |v: &serde_json::Value, key: &str| -> f64 {
            let x = v[key].as_array().expect("array");
            x.iter()
                .map(|y| y.as_f64().unwrap_or(f64::NAN))
                .sum::<f64>()
                / x.len() as f64
        };
        ai_top1.push(pick(&ai, "top1"));
        b0_top1.push(pick(&b0, "top1"));
        let m = |v: &serde_json::Value, key: &str| -> f64 {
            let p = pick(v, key);
            p.iter().sum::<f64>() / p.len() as f64
        };
        seeds_json.push(serde_json::json!({
            "seed": seed,
            "allinfo_recipe_digest": ai["recipe_digest"],
            "b0_recipe_digest": b0["recipe_digest"],
            "allinfo_per_position_sha256": sha256_of(&ai),
            "b0_per_position_sha256": sha256_of(&b0),
            "kqrvk_m3_top1": {"allinfo": m(&ai, "top1"), "b0": m(&b0, "top1")},
        }));
        diag.push(serde_json::json!({
            "seed": seed,
            "kqrvk_m3": {
                "allinfo": {"top1": m(&ai, "top1"), "correct_mass": m(&ai, "correct_mass"), "ce": m(&ai, "ce"), "entropy": m(&ai, "entropy")},
                "b0": {"top1": m(&b0, "top1"), "correct_mass": m(&b0, "correct_mass"), "ce": m(&b0, "ce"), "entropy": m(&b0, "entropy")},
            },
            "pooled": {
                "allinfo": {"top1": mean_all(&ai, "top1"), "correct_mass": mean_all(&ai, "correct_mass"), "ce": mean_all(&ai, "ce"), "entropy": mean_all(&ai, "entropy")},
                "b0": {"top1": mean_all(&b0, "top1"), "correct_mass": mean_all(&b0, "correct_mass"), "ce": mean_all(&b0, "ce"), "entropy": mean_all(&b0, "entropy")},
            },
        }));
    }
    let g = gate1(&ai_top1, &b0_top1)?;
    let ai_abs = ai_top1.iter().flatten().sum::<f64>() / (ai_top1.len() * ai_top1[0].len()) as f64;
    let b0_abs = b0_top1.iter().flatten().sum::<f64>() / (b0_top1.len() * b0_top1[0].len()) as f64;
    let verdict = if g.pass {
        "GATE I PASS - RAW FUTURE-STATE INFORMATION IS SUFFICIENT FOR THIS MODEL FAMILY ON V3_TUNE_V1"
    } else {
        "GATE I FAIL"
    };
    write_json(
        &a.output,
        &serde_json::json!({
            "schema": "v3_p6_gate1_v1",
            "rule": "ALL-INFO - B0 >= +0.20 top-1 on KQRvK M3 (V3_TUNE_V1, n=750) AND the paired 95% position-level bootstrap CI wholly > 0; nothing else enters pass/fail",
            "estimator": "d[i,s] = 1(AllInfo_s correct on i) - 1(B0_s correct on i); d_i = mean_s d[i,s]; Delta = mean_i d_i; bootstrap resamples positions with all seed pairs kept together",
            "bootstrap": {"resamples": RESAMPLES, "seed": format!("0x{BOOTSTRAP_SEED:X}"), "interval": "percentile 2.5% / 97.5% (sorted ranks 499 and 19499 of 20000)"},
            "threshold": THRESHOLD,
            "gate": g,
            "verdict": verdict,
            "allinfo_absolute_top1_kqrvk_m3_diagnostic": {
                "value": ai_abs, "reference_not_gating": 0.75,
                "note": "0.75 is a reference value only and does not affect pass/fail",
            },
            "b0_absolute_top1_kqrvk_m3": b0_abs,
            "seeds": seeds_json,
            "diagnostics_not_gating": diag,
            "tune_dataset_digest": tune.digest,
            "p6_contract_digest": contract.contract_digest(),
            "selected_p5_recipe_digest_without_seed": crate::v3_p5::P5_SELECTED_DIGEST,
            "gate_cell_position_ids": gate_idx.iter().map(|&i| ids[i].clone()).collect::<Vec<_>>(),
            "provenance_top1_by_seed": {
                "allinfo": ai_top1, "b0": b0_top1,
                "note": "top-1 correctness (0/1) of every KQRvK M3 position per seed, in gate_cell_position_ids order; d[i,s] is their difference",
            },
            "holdout_c": "never loaded; unevaluated",
        }),
    )?;
    println!(
        "Gate I: Delta {:+.4}, CI [{:.4}, {:.4}], threshold {THRESHOLD}: {verdict}",
        g.delta, g.ci_lower, g.ci_upper
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouts_on_the_ladder_parse_and_others_are_refused() {
        assert_eq!(parse_layout("16x8").unwrap(), (16, 8));
        assert_eq!(parse_layout("2x64").unwrap(), (2, 64));
        assert!(parse_layout("32x4").is_err());
        assert!(parse_layout("8").is_err());
    }

    #[test]
    fn gate_cell_mean_averages_only_the_gate_cell() {
        let doc = serde_json::json!({
            "cells": ["KQRvK M3", "KQRvK M2", "KQRvK M3", "KRRvK M3"],
            "top1": [1.0, 0.0, 0.0, 1.0],
        });
        assert!((gate_cell_mean(&doc) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn gate_refuses_to_overwrite_an_existing_result() {
        let d = std::env::temp_dir().join(format!("recur64-p6-gate-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("g.json"), b"{}").unwrap();
        let a = GateArgs {
            tune: d.join("none"),
            b0_dir: d.clone(),
            allinfo_dir: d.clone(),
            recipe: d.join("none"),
            output: d.join("g.json"),
        };
        assert!(
            run_gate(&a)
                .unwrap_err()
                .to_string()
                .contains("applied once")
        );
        std::fs::remove_dir_all(&d).unwrap();
    }
}
