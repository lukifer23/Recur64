//! `recur64 v420`: model-info, smoke (engineering timings, no science), train, measure (applies the
//! frozen gates once). TRAIN data only; every sealed set is refused by `V4Data::load`.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use burn::tensor::backend::AutodiffBackend;
use clap::{Args, Subcommand};
use recur64_runtime::gpu_telemetry::sample_gpu;
use recur64_runtime::model_io;
use recur64_v4::data::V4Data;
use recur64_v420::gates::{self, GateInputs};
use recur64_v420::model::{DepthModel, Geometry};
use recur64_v420::train::{
    Arm, Recipe, Trainer, UPDATES, arm as arm_by_name, eval_positions, load_weights,
};

use crate::v3_p5::{Dev, dispatch, write_json};

/// Headline cell (hardest dev cell in V4 Stage A: KQRvK M3, top-1 0.637 on seed 5101).
pub const HEADLINE_FAMILY: &str = "KQRvK";
pub const HEADLINE_DEPTH: u8 = 3;
const SEEDS: [u64; 3] = [5101, 5102, 5103];

#[derive(Subcommand, Debug)]
pub enum V420Cmd {
    /// Parameter counts and geometry of one arm (no device work).
    ModelInfo(InfoArgs),
    /// Engineering timing and VRAM of a few updates per arm (NO science, no result).
    Smoke(SmokeArgs),
    /// Train one arm/seed with the frozen recipe.
    Train(TrainArgs),
    /// Evaluate all runs and apply the frozen gates once.
    Measure(MeasureArgs),
}

#[derive(Args, Debug)]
pub struct InfoArgs {
    #[arg(long)]
    pub arm: String,
}

#[derive(Args, Debug, Clone)]
pub struct SmokeArgs {
    #[arg(long)]
    pub train: PathBuf,
    #[arg(long, value_enum)]
    pub device: Dev,
    #[arg(long, default_value_t = 4)]
    pub updates: u64,
    #[arg(long)]
    pub output: PathBuf,
}

#[derive(Args, Debug, Clone)]
pub struct TrainArgs {
    #[arg(long)]
    pub train: PathBuf,
    #[arg(long)]
    pub arm: String,
    #[arg(long)]
    pub seed: u64,
    #[arg(long, value_enum)]
    pub device: Dev,
    #[arg(long)]
    pub run_dir: PathBuf,
}

#[derive(Args, Debug, Clone)]
pub struct MeasureArgs {
    #[arg(long)]
    pub train: PathBuf,
    #[arg(long, value_enum)]
    pub device: Dev,
    /// Directory holding `<ARM>-seed<SEED>` run directories.
    #[arg(long)]
    pub runs_root: PathBuf,
    #[arg(long, default_value_t = 64)]
    pub micro: usize,
    #[arg(long)]
    pub output: PathBuf,
}

pub fn run(cmd: V420Cmd) -> anyhow::Result<()> {
    match cmd {
        V420Cmd::ModelInfo(a) => model_info(&a),
        V420Cmd::Smoke(a) => {
            let dev = a.device;
            dispatch(dev, move |d| match d {
                crate::v3_p5::Dispatch::Cpu => smoke::<recur64_model::train::CpuTrainBackend>(&a, "cpu"),
                #[cfg(feature = "cuda")]
                crate::v3_p5::Dispatch::Cuda => {
                    smoke::<burn::backend::Autodiff<burn::backend::Cuda>>(&a, "cuda")
                }
            })
        }
        V420Cmd::Train(a) => {
            let dev = a.device;
            dispatch(dev, move |d| match d {
                crate::v3_p5::Dispatch::Cpu => train::<recur64_model::train::CpuTrainBackend>(&a, "cpu"),
                #[cfg(feature = "cuda")]
                crate::v3_p5::Dispatch::Cuda => {
                    train::<burn::backend::Autodiff<burn::backend::Cuda>>(&a, "cuda")
                }
            })
        }
        V420Cmd::Measure(a) => {
            let dev = a.device;
            dispatch(dev, move |d| match d {
                crate::v3_p5::Dispatch::Cpu => measure::<recur64_model::train::CpuTrainBackend>(&a),
                #[cfg(feature = "cuda")]
                crate::v3_p5::Dispatch::Cuda => {
                    measure::<burn::backend::Autodiff<burn::backend::Cuda>>(&a)
                }
            })
        }
    }
}

fn model_info(a: &InfoArgs) -> anyhow::Result<()> {
    let arm = arm_by_name(&a.arm)?;
    let g = Geometry::v4(arm.unique_blocks);
    let m = DepthModel::<recur64_model::train::CpuTrainBackend>::new(&g, &Default::default());
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "identity": recur64_v420::model::IDENTITY, "arm": arm, "geometry": g,
            "parameters": m.num_params_total(), "eval_depths": arm.eval_ks(),
            "executed_board_blocks_at_train_depth": match arm.k {
                recur64_v420::train::KSpec::Fixed(k) => serde_json::json!(k),
                recur64_v420::train::KSpec::Sampled(max) => serde_json::json!(format!("1..={max} (sampled)")),
            },
        }))?
    );
    Ok(())
}

fn smoke<B: AutodiffBackend>(a: &SmokeArgs, label: &str) -> anyhow::Result<()> {
    let device = B::Device::default();
    model_io::verify_device::<B>(&device)?;
    let data = V4Data::load(&a.train)?;
    let mut rows = Vec::new();
    for name in ["R1", "RK2", "RK4", "RV", "WIDE", "BASE8"] {
        let arm = arm_by_name(name)?;
        let mut recipe = Recipe::new(arm.clone(), 1, &data);
        recipe.updates = a.updates;
        let mut t = Trainer::<B>::new(recipe, &device);
        let mut walls = Vec::new();
        let mut peak = 0u64;
        for _ in 0..a.updates {
            let rec = t.step(&data, &device)?;
            walls.push(rec.wall_s);
            if let Some((m, _, _)) = sample_gpu().filter(|_| label == "cuda") {
                peak = peak.max(m);
            }
        }
        rows.push(serde_json::json!({
            "arm": name, "parameters": t.model.num_params_total(), "update_wall_s": walls,
            "peak_vram_mb_sampled": (label == "cuda").then_some(peak),
            "losses": t.history.iter().map(|h| h.loss).collect::<Vec<_>>(),
        }));
        println!("{name}: {} params, walls {:?}", t.model.num_params_total(), walls);
    }
    write_json(
        &a.output,
        &serde_json::json!({
            "schema": "v420_smoke_v1", "device": label, "updates_each": a.updates, "rows": rows,
            "note": "ENGINEERING timing only; no science claim, nothing here feeds any gate",
        }),
    )?;
    Ok(())
}

fn train<B: AutodiffBackend>(a: &TrainArgs, label: &str) -> anyhow::Result<()> {
    anyhow::ensure!(!a.run_dir.join("run-summary.json").exists(), "{}: already trained; refusing to overwrite", a.run_dir.display());
    let device = B::Device::default();
    model_io::verify_device::<B>(&device)?;
    let data = V4Data::load(&a.train)?;
    let arm = arm_by_name(&a.arm)?;
    let recipe = Recipe::new(arm, a.seed, &data);
    println!("recipe digest {}", recipe.digest());
    let mut t = Trainer::<B>::new(recipe, &device);
    let t0 = Instant::now();
    let mut peak = 0u64;
    while t.updates_done < t.recipe.updates {
        match t.step(&data, &device) {
            Ok(rec) => {
                if rec.update % 50 == 0 || rec.update + 1 == UPDATES {
                    if let Some((m, _, _)) = sample_gpu().filter(|_| label == "cuda") {
                        peak = peak.max(m);
                    }
                    println!(
                        "update {:>4} k={} lr {:.2e} loss {:.4} ({:.2}s) elapsed {:.0}s vram {} MiB",
                        rec.update, rec.k, rec.lr, rec.loss, rec.wall_s, t0.elapsed().as_secs_f64(), peak
                    );
                }
            }
            Err(e) => {
                // A failed run stays visible: write the reason and the history so far.
                std::fs::create_dir_all(&a.run_dir)?;
                write_json(
                    &a.run_dir.join("run-aborted.json"),
                    &serde_json::json!({
                        "schema": "v420_run_aborted_v1", "arm": a.arm, "seed": a.seed,
                        "updates_done": t.updates_done, "reason": format!("{e:#}"),
                        "health": t.health,
                        "recipe_digest": t.recipe.digest(),
                    }),
                )?;
                return Err(e);
            }
        }
    }
    t.save(&a.run_dir, label)?;
    println!("done: {} updates in {:.0}s -> {}", t.updates_done, t0.elapsed().as_secs_f64(), a.run_dir.display());
    Ok(())
}

struct Loaded<B: burn::tensor::backend::Backend> {
    model: DepthModel<B>,
    arm: Arm,
}

fn cell_of(data: &V4Data, i: usize) -> bool {
    let p = data.position(i);
    p.family == HEADLINE_FAMILY && p.mate_depth == HEADLINE_DEPTH
}

fn measure<B: AutodiffBackend>(a: &MeasureArgs) -> anyhow::Result<()> {
    let device = B::Device::default();
    model_io::verify_device::<B>(&device)?;
    let idev = Default::default();
    let data = V4Data::load(&a.train)?;
    let dev_idx: Vec<usize> = data.dev.clone();
    let head_idx: Vec<usize> = dev_idx.iter().copied().filter(|&i| cell_of(&data, i)).collect();
    anyhow::ensure!(!head_idx.is_empty(), "headline cell has no dev positions");

    // S0: every required run completed, finite and healthy; failed runs stay visible.
    let mut s0_ok = true;
    let mut s0_notes = Vec::new();
    let mut loaded: BTreeMap<(String, u64), Loaded<B::InnerBackend>> = BTreeMap::new();
    let required: Vec<(&str, Vec<u64>)> = vec![
        ("R1", SEEDS.to_vec()),
        ("RK2", SEEDS.to_vec()),
        ("RK4", SEEDS.to_vec()),
        ("RV", SEEDS.to_vec()),
        ("WIDE", SEEDS.to_vec()),
        ("BASE8", vec![5101]),
    ];
    for (name, seeds) in &required {
        for &s in seeds {
            let dir = a.runs_root.join(format!("{name}-seed{s}"));
            let aborted = dir.join("run-aborted.json").exists();
            let summary: Option<serde_json::Value> = std::fs::read(dir.join("run-summary.json"))
                .ok()
                .and_then(|b| serde_json::from_slice(&b).ok());
            let ok = !aborted
                && summary.as_ref().is_some_and(|v| {
                    v["all_losses_finite"] == true && v["health_ok"] == true && v["updates"] == UPDATES
                });
            if !ok {
                s0_ok = false;
                s0_notes.push(format!("{name} seed {s}: missing, aborted, unhealthy or incomplete"));
                continue;
            }
            let arm = arm_by_name(name)?;
            let g = Geometry::v4(arm.unique_blocks);
            loaded.insert((name.to_string(), s), Loaded { model: load_weights::<B>(&dir, &g, &device)?, arm });
        }
    }
    anyhow::ensure!(
        s0_ok,
        "S0 failed, nothing was evaluated: {}",
        s0_notes.join("; ")
    );

    // Evaluate every loaded model at every depth it supports.
    let mut tables = serde_json::Map::new();
    let mut head: BTreeMap<(String, usize), Vec<Vec<f64>>> = BTreeMap::new();
    let mut base8_all = None;
    for ((name, seed), l) in &loaded {
        for k in l.arm.eval_ks() {
            let all = eval_positions(&l.model, &data, &dev_idx, k, a.micro, &idev)?;
            let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
            // Per-cell breakdown.
            let mut cells: BTreeMap<String, (Vec<f64>, Vec<f64>)> = BTreeMap::new();
            for (pos, &i) in dev_idx.iter().enumerate() {
                let p = data.position(i);
                let e = cells.entry(format!("{} M{}", p.family, p.mate_depth)).or_default();
                e.0.push(all.top1[pos]);
                e.1.push(all.ce[pos]);
            }
            let head_top1: Vec<f64> = dev_idx
                .iter()
                .enumerate()
                .filter(|(_, i)| cell_of(&data, **i))
                .map(|(pos, _)| all.top1[pos])
                .collect();
            tables.insert(
                format!("{name}/seed{seed}/K{k}"),
                serde_json::json!({
                    "top1": mean(&all.top1), "ce": mean(&all.ce), "correct_mass": mean(&all.mass),
                    "headline_top1": mean(&head_top1), "headline_n": head_top1.len(),
                    "by_cell": cells.iter().map(|(c, (t, e))| (c.clone(), serde_json::json!({"top1": mean(t), "ce": mean(e), "n": t.len()}))).collect::<BTreeMap<_, _>>(),
                }),
            );
            head.entry((name.clone(), k)).or_default().push(head_top1);
            if name == "BASE8" {
                base8_all = Some(mean(&all.top1));
            }
        }
    }
    let get = |n: &str, k: usize| -> anyhow::Result<&Vec<Vec<f64>>> {
        head.get(&(n.to_string(), k)).ok_or_else(|| anyhow::anyhow!("no headline data for {n} K={k}"))
    };
    let (report, verdict) = gates::apply(&GateInputs {
        r1: get("R1", 1)?,
        rk4: get("RK4", 4)?,
        rv_k4: get("RV", 4)?,
        rv_k8: get("RV", 8)?,
        rv_k16: get("RV", 16)?,
        wide: get("WIDE", 4)?,
        base8_top1_all_dev: base8_all,
        s0_ok,
    })?;
    write_json(
        &a.output,
        &serde_json::json!({
            "schema": "v420_measure_v1", "gates": report, "verdict": verdict,
            "headline_cell": format!("{HEADLINE_FAMILY} M{HEADLINE_DEPTH}"),
            "headline_positions": head_idx.len(),
            "tables": tables,
            "data": {"train_digest": data.targets.digest, "fit_digest": data.fit_digest, "dev_digest": data.dev_digest,
                     "v4_tune_v1_evaluated": false, "holdout_c_evaluated": false},
        }),
    )?;
    println!("verdict: {verdict}; wrote {}", a.output.display());
    Ok(())
}
