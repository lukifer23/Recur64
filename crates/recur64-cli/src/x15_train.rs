//! `recur64 x15 train-probe | eval-reasoning` - fixed-data reasoning screen.
//!
//! Trains X15 on `ReasoningTargetsV1` (full batch, no self-play, no pilot) and
//! measures, with the SAME weights, whether more thoughts move the policy and
//! value closer to the deepest teacher search.
//!
//! Evaluation uses the diagnostic forward once at the largest T. The k-th
//! diagnostic readout is exactly what a T=k run would output as its final
//! readout (the loop up to step k does not depend on T; tested), so one pass
//! yields T=1..Tmax. Every run writes `experiment.json` with a canonical hash
//! of its scientific settings.

use std::path::{Path, PathBuf};
use std::time::Instant;

use burn::module::AutodiffModule;
use burn::optim::{GradientsAccumulator, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::activation;
use burn::tensor::backend::AutodiffBackend;
use clap::Args;
use serde::Serialize;
use sha2::{Digest, Sha256};

use recur64_model::checkpoint::{CheckpointMeta, save_training_chimera};

/// Hard maximum thought count the model supports.
const MAX_THOUGHTS: usize = 8;
use recur64_model::chimera::ChimeraModel;
use recur64_model::config::ProbeConfig;
use recur64_model::experimental::DeepSupervisionMode;
use recur64_model::loss::{policy_ce, thought_loss_with};
use recur64_model::model::Readout;
use recur64_runtime::gpu_telemetry::monitor;
use recur64_runtime::model_io;
use recur64_runtime::reasoning_targets::{PositionTarget, ReasoningTargetsV1, rebuild_state};
use recur64_runtime::x15_inputs::{X15Batch, build_x15_batch_padded, provider_for_config};

#[derive(Args, Debug)]
pub struct TrainProbeArgs {
    #[arg(long, default_value = "configs/x15_cuda.toml")]
    pub config: PathBuf,
    #[arg(long)]
    pub targets: PathBuf,
    #[arg(long)]
    pub out: PathBuf,
    /// Thoughts used in training.
    #[arg(long, default_value_t = 4)]
    pub thoughts: usize,
    /// Override the config's deep supervision mode.
    #[arg(long)]
    pub supervision: Option<String>,
    #[arg(long)]
    pub intermediate_weight: Option<f32>,
    #[arg(long, default_value_t = 1e-4)]
    pub lr: f64,
    #[arg(long, default_value_t = 5)]
    pub warmup: usize,
    #[arg(long, default_value_t = 40)]
    pub updates: usize,
    /// Weight of the value (p_win - p_loss vs root value) MSE term.
    #[arg(long, default_value_t = 1.0)]
    pub value_weight: f32,
    #[arg(long, default_value_t = 1)]
    pub seed: u64,
    /// Evaluate on val every N updates (0 = only at the end).
    #[arg(long, default_value_t = 10)]
    pub eval_every: usize,
    /// Thoughts evaluated (diagnostic forward at this T).
    #[arg(long, default_value_t = 4)]
    pub eval_thoughts: usize,
    /// Resume weights + optimizer from a checkpoint directory.
    #[arg(long)]
    pub resume: Option<PathBuf>,
    /// Train on this many train positions only (micro-overfit); 0 = all.
    #[arg(long, default_value_t = 0)]
    pub limit: usize,
    /// Positions per forward/backward chunk (gradients are accumulated); bounds VRAM.
    #[arg(long, default_value_t = 32)]
    pub micro_batch: usize,
    /// Positions per optimizer update (a multiple of --micro-batch), taken in a
    /// seeded hash order cycling over the train split. 0 = the whole split every
    /// update (the original full-batch behaviour).
    #[arg(long, default_value_t = 0)]
    pub batch_positions: usize,
    /// Bounded-time guard for a single run.
    #[arg(long, default_value_t = 1500.0)]
    pub max_seconds: f64,
}

#[derive(Args, Debug)]
pub struct EvalArgs {
    #[arg(long, default_value = "configs/x15_cuda.toml")]
    pub config: PathBuf,
    #[arg(long)]
    pub targets: PathBuf,
    /// Checkpoint directory to evaluate (weights + meta).
    #[arg(long)]
    pub checkpoint: Vec<PathBuf>,
    /// Which target split to evaluate (`val` = tuning, `confirm` = fresh confirmation).
    #[arg(long, default_value = "val")]
    pub split: String,
    #[arg(long, default_value_t = 4)]
    pub thoughts: usize,
    /// T_train=1 baseline checkpoint(s): report (checkpoint at the largest T) minus
    /// (baseline at T=1), paired over positions, pooled over checkpoints by position.
    #[arg(long)]
    pub baseline: Vec<PathBuf>,
    /// Also report the train split.
    #[arg(long, default_value_t = false)]
    pub include_train: bool,
    #[arg(long)]
    pub json_out: Option<PathBuf>,
}

/// One split's tensors and host copies of the targets.
struct Data<B: Backend> {
    batch: X15Batch<B>,
    /// Per rung: `[b, width]` teacher policy in candidate order.
    policy: Vec<Tensor<B, 2>>,
    /// Per rung: `[b]` teacher root value.
    value: Vec<Tensor<B, 1>>,
    host_policy: Vec<Vec<Vec<f32>>>, // [rung][pos][legal]
    host_value: Vec<Vec<f32>>,       // [rung][pos]
    host_best: Vec<Vec<usize>>,      // [rung][pos]
    categories: Vec<String>,
}

impl<B: Backend> Data<B> {
    fn len(&self) -> usize {
        self.batch.batch
    }
}

fn build_data<B: Backend>(
    positions: &[&PositionTarget],
    cfg: &ProbeConfig,
    device: &B::Device,
    min_width: usize,
) -> anyhow::Result<Data<B>> {
    anyhow::ensure!(!positions.is_empty(), "empty split");
    let provider = provider_for_config(&cfg.experimental)?;
    let states = positions
        .iter()
        .map(|p| rebuild_state(&p.start_fen, &p.prefix))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let batch = build_x15_batch_padded::<B>(
        &states,
        &cfg.experimental,
        provider.as_ref(),
        device,
        min_width,
    )?;
    let width = batch.cands.width;
    let rungs = positions[0].rungs.len();
    let (mut policy, mut value) = (Vec::new(), Vec::new());
    let (mut hp, mut hv, mut hb) = (Vec::new(), Vec::new(), Vec::new());
    for r in 0..rungs {
        let mut flat = vec![0.0f32; positions.len() * width];
        let mut rows = Vec::new();
        let mut vals = Vec::new();
        let mut best = Vec::new();
        for (b, p) in positions.iter().enumerate() {
            let rung = &p.rungs[r];
            anyhow::ensure!(
                p.legal.len() <= width,
                "{}: legal list wider than batch",
                p.id
            );
            flat[b * width..b * width + rung.policy.len()].copy_from_slice(&rung.policy);
            rows.push(rung.policy.clone());
            vals.push(rung.root_value);
            best.push(rung.best);
        }
        policy.push(Tensor::<B, 2>::from_data(
            TensorData::new(flat, [positions.len(), width]),
            device,
        ));
        value.push(Tensor::<B, 1>::from_data(
            TensorData::new(vals.clone(), [positions.len()]),
            device,
        ));
        hp.push(rows);
        hv.push(vals);
        hb.push(best);
    }
    Ok(Data {
        batch,
        policy,
        value,
        host_policy: hp,
        host_value: hv,
        host_best: hb,
        categories: positions.iter().map(|p| p.category.clone()).collect(),
    })
}

pub(crate) fn split<'a>(t: &'a ReasoningTargetsV1, which: &str) -> Vec<&'a PositionTarget> {
    t.positions.iter().filter(|p| p.split == which).collect()
}

/// Policy CE to the rung target plus the value MSE (`p_win - p_loss` vs root value).
fn readout_terms<B: Backend>(
    r: &Readout<B>,
    policy: &Tensor<B, 2>,
    value: &Tensor<B, 1>,
    value_weight: f32,
) -> Tensor<B, 1> {
    let pce = policy_ce(&r.policy, policy);
    let p = activation::softmax(r.wdl_logits.clone(), 1);
    let v = p.clone().narrow(1, 0, 1).squeeze_dim::<1>(1) - p.narrow(1, 2, 1).squeeze_dim::<1>(1);
    let mse = (v - value.clone()).powf_scalar(2.0).mean();
    pce + mse * value_weight
}

// --- evaluation ------------------------------------------------------------------

#[derive(Serialize, Clone)]
pub(crate) struct PerThought {
    pub(crate) t: usize,
    pub(crate) kl_deep: Vec<f32>,
    pub(crate) ce_deep: Vec<f32>,
    pub(crate) top1: Vec<f32>,
    pub(crate) entropy: Vec<f32>,
    pub(crate) value_abs_err: Vec<f32>,
    pub(crate) value_pred: Vec<f32>,
    pub(crate) p_win: Vec<f32>,
    pub(crate) p_draw: Vec<f32>,
    pub(crate) latent_delta: f32,
    pub(crate) policy_kl_prev: f32,
    pub(crate) wdl_l1_prev: f32,
}

pub(crate) fn mean(v: &[f32]) -> f32 {
    if v.is_empty() {
        f32::NAN
    } else {
        v.iter().sum::<f32>() / v.len() as f32
    }
}

fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Deterministic paired bootstrap of `mean(a - b)`: (mean, lo95, hi95).
pub(crate) fn paired_bootstrap(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
    let d: Vec<f32> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    let n = d.len();
    if n == 0 {
        return (f32::NAN, f32::NAN, f32::NAN);
    }
    let mut means: Vec<f32> = (0..2000u64)
        .map(|r| {
            let mut s = 0.0f32;
            for i in 0..n {
                let k = (mix(r * 1_000_003 + i as u64) % n as u64) as usize;
                s += d[k];
            }
            s / n as f32
        })
        .collect();
    means.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    (mean(&d), means[50], means[1949])
}

fn evaluate<B: Backend>(model: &ChimeraModel<B>, data: &Data<B>, tmax: usize) -> Vec<PerThought> {
    let out = model.forward_thoughts_diagnostic(&data.batch.input, &data.batch.cands, tmax);
    let deep = data.host_policy.len() - 1;
    let width = data.batch.cands.width;
    let n = data.len();
    let to_vec = |t: Tensor<B, 1>| t.into_data().to_vec::<f32>().unwrap_or_default();
    let mut result = Vec::new();
    for (k, (readout, m)) in out.readouts.iter().zip(&out.thoughts).enumerate() {
        let lp = readout
            .policy
            .log_probs
            .clone()
            .into_data()
            .to_vec::<f32>()
            .unwrap_or_default();
        let wdl = activation::softmax(readout.wdl_logits.clone(), 1)
            .into_data()
            .to_vec::<f32>()
            .unwrap_or_default();
        let (mut kl, mut ce, mut top1, mut ent, mut verr, mut vpred) =
            (vec![], vec![], vec![], vec![], vec![], vec![]);
        let (mut pw, mut pd) = (vec![], vec![]);
        for b in 0..n {
            let pi = &data.host_policy[deep][b];
            let row = &lp[b * width..b * width + pi.len()];
            let h: f32 = pi.iter().filter(|p| **p > 0.0).map(|p| -p * p.ln()).sum();
            let c: f32 = pi.iter().zip(row).map(|(p, l)| -p * l).sum();
            ce.push(c);
            kl.push(c - h);
            let argmax = row
                .iter()
                .enumerate()
                .fold(0usize, |bi, (i, v)| if *v > row[bi] { i } else { bi });
            top1.push(f32::from(argmax == data.host_best[deep][b]));
            ent.push(-row.iter().map(|l| l.exp() * l).sum::<f32>());
            let (w, d, l) = (wdl[b * 3], wdl[b * 3 + 1], wdl[b * 3 + 2]);
            let v = w - l;
            vpred.push(v);
            verr.push((v - data.host_value[deep][b]).abs());
            pw.push(w);
            pd.push(d);
        }
        result.push(PerThought {
            t: k + 1,
            kl_deep: kl,
            ce_deep: ce,
            top1,
            entropy: ent,
            value_abs_err: verr,
            value_pred: vpred,
            p_win: pw,
            p_draw: pd,
            latent_delta: mean(&to_vec(m.latent_delta_norm.clone())),
            policy_kl_prev: mean(&to_vec(m.policy_kl_prev.clone())),
            wdl_l1_prev: mean(&to_vec(m.wdl_l1_prev.clone())),
        });
    }
    result
}

fn half_diff(a: &[f32], b: &[f32], parity: usize) -> f32 {
    let d: Vec<f32> = a
        .iter()
        .zip(b)
        .enumerate()
        .filter(|(i, _)| i % 2 == parity)
        .map(|(_, (x, y))| x - y)
        .collect();
    mean(&d)
}

fn print_eval(label: &str, rows: &[PerThought]) {
    println!(
        "  [{label}] n={}  (metrics vs deepest teacher rung)",
        rows[0].kl_deep.len()
    );
    println!(
        "  {:<3} {:>9} {:>9} {:>7} {:>8} {:>9} {:>12} {:>10} {:>18}",
        "T", "KL", "CE", "top1", "entropy", "|v err|", "dKL vs T1", "95% CI", "even/odd half dKL"
    );
    let base = &rows[0].kl_deep;
    for r in rows {
        let (d, lo, hi) = paired_bootstrap(&r.kl_deep, base);
        println!(
            "  {:<3} {:>9.4} {:>9.4} {:>7.3} {:>8.3} {:>9.4} {:>+12.4} [{:+.4},{:+.4}] {:>+9.4}/{:+.4}",
            r.t,
            mean(&r.kl_deep),
            mean(&r.ce_deep),
            mean(&r.top1),
            mean(&r.entropy),
            mean(&r.value_abs_err),
            d,
            lo,
            hi,
            half_diff(&r.kl_deep, base, 0),
            half_diff(&r.kl_deep, base, 1)
        );
    }
}

fn eval_json(rows: &[PerThought], data_ids: &[String]) -> serde_json::Value {
    let base = &rows[0].kl_deep;
    serde_json::json!({
        "positions": data_ids,
        "thoughts": rows.iter().map(|r| {
            let (d, lo, hi) = paired_bootstrap(&r.kl_deep, base);
            serde_json::json!({
                "t": r.t, "kl_deep": mean(&r.kl_deep), "ce_deep": mean(&r.ce_deep),
                "top1": mean(&r.top1), "entropy": mean(&r.entropy),
                "value_abs_err": mean(&r.value_abs_err),
                "dkl_vs_t1": {"mean": d, "lo95": lo, "hi95": hi},
                "latent_delta": r.latent_delta, "policy_kl_prev": r.policy_kl_prev,
                "wdl_l1_prev": r.wdl_l1_prev,
                "per_position": {"kl_deep": r.kl_deep, "top1": r.top1, "value_abs_err": r.value_abs_err},
            })
        }).collect::<Vec<_>>(),
    })
}

// --- experiment record ---------------------------------------------------------------

fn experiment_record(
    cfg: &ProbeConfig,
    targets: &ReasoningTargetsV1,
    args: &TrainProbeArgs,
    n_train: usize,
    init: serde_json::Value,
    start_update: usize,
) -> anyhow::Result<serde_json::Value> {
    let scientific = serde_json::json!({
        "model": cfg.model,
        "experimental_identity": cfg.experimental.identity()?,
        "train_thoughts": args.thoughts,
        "supervision": cfg.experimental.deep_supervision.label(),
        "intermediate_weight": cfg.experimental.intermediate_weight,
        "value_weight": args.value_weight,
        "loss_normalization": "total_active_weight_v1",
        "targets_digest": targets.digest,
        "teacher": targets.teacher,
        "train_positions": n_train,
        "optimizer_contract": recur64_model::train::OPTIMIZER_CONTRACT,
        "batch_positions": args.batch_positions,
        "data_order": if args.batch_positions > 0 { "fnv1a(id) mixed with seed, cycling" } else { "file_order_full_batch" },
        "lr": args.lr,
        "warmup_updates": args.warmup,
        "initialization": init,
        "start_update": start_update,
        "segment_updates": args.updates,
        "final_update": start_update + args.updates,
        "updates": args.updates,
        "seed": args.seed,
    });
    let hash = format!("{:x}", Sha256::digest(serde_json::to_vec(&scientific)?));
    Ok(serde_json::json!({
        "schema": "x15_experiment_v1",
        "experiment_hash": hash,
        "git_rev": recur64_runtime::provenance::git_revision().unwrap_or("unknown"),
        "scientific": scientific,
    }))
}

// --- train ---------------------------------------------------------------------------

fn train<B: AutodiffBackend>(cfg: &mut ProbeConfig, args: &TrainProbeArgs) -> anyhow::Result<()> {
    if let Some(s) = &args.supervision {
        cfg.experimental.deep_supervision = match s.as_str() {
            "final_only_v1" => DeepSupervisionMode::FinalOnlyV1,
            "same_target_v1" => DeepSupervisionMode::SameTargetV1,
            "progressive_search_v1" => DeepSupervisionMode::ProgressiveSearchV1,
            other => anyhow::bail!("unknown supervision mode {other:?}"),
        };
    }
    if let Some(w) = args.intermediate_weight {
        cfg.experimental.intermediate_weight = w;
    }
    cfg.experimental.validate(cfg.model.width)?;
    // Training and evaluation thought counts must be within the model hard
    // maximum (8); a reasoning-disabled model runs exactly one thought. The
    // config's thought_steps is the designed T, not a cap: a T_train=1
    // control on a T=4 config is legitimate and is recorded as train_thoughts.
    let max_t = if cfg.experimental.reasoning.enabled {
        MAX_THOUGHTS
    } else {
        1
    };
    anyhow::ensure!(
        (1..=max_t).contains(&args.thoughts),
        "train thoughts {} outside 1..={max_t}",
        args.thoughts
    );
    // The evaluation depth is a request, clamped to what the model can run (a
    // reasoning-disabled model always evaluates one thought); 0 is a mistake.
    anyhow::ensure!(args.eval_thoughts >= 1, "eval thoughts must be >= 1");
    let eval_thoughts = args.eval_thoughts.min(max_t);
    let mode = cfg.experimental.deep_supervision;
    let weight = cfg.experimental.intermediate_weight;

    let targets = ReasoningTargetsV1::load(&args.targets)?;
    recur64_runtime::reasoning_targets::audit(&targets)?;
    let mut train_pos = split(&targets, "train");
    if args.limit > 0 {
        train_pos.truncate(args.limit);
    }
    let val_pos = split(&targets, "val");
    let device: B::Device = Default::default();
    B::seed(&device, args.seed);

    // Deterministic training order: file order for full-batch runs, otherwise a
    // seeded hash order over position ids (recorded in experiment.json).
    let mut ordered: Vec<&PositionTarget> = train_pos.clone();
    if args.batch_positions > 0 {
        ordered.sort_by_key(|p| (mix(args.seed ^ id_hash(&p.id)), p.id.clone()));
    }
    // One fixed candidate width for the whole run, so the GPU sees one shape.
    let widest = ordered.iter().map(|p| p.legal.len()).max().unwrap_or(0);
    let fixed_width = recur64_model::action::CandidateBatch::WIDTH_BUCKETS
        .iter()
        .copied()
        .find(|&w| w >= widest)
        .unwrap_or(widest);
    let chunks = ordered
        .chunks(args.micro_batch.max(1))
        .map(|c| build_data::<B>(c, cfg, &device, fixed_width))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let chunks_per_update = if args.batch_positions > 0 {
        (args.batch_positions / args.micro_batch.max(1)).max(1)
    } else {
        chunks.len()
    };
    let inner_device: Device<B::InnerBackend> = Default::default();
    let val_data = if val_pos.is_empty() {
        None
    } else {
        Some(build_data::<B::InnerBackend>(
            &val_pos,
            cfg,
            &inner_device,
            0,
        )?)
    };
    let val_ids: Vec<String> = val_pos.iter().map(|p| p.id.clone()).collect();
    let rungs = chunks[0].policy.len();

    let mut optim = recur64_model::train::adamw::<B, ChimeraModel<B>>();
    let (mut model, mut start_update, init) = match &args.resume {
        Some(dir) => {
            let (m, o, meta) = model_io::load_chimera_training::<B, _>(
                dir,
                &cfg.model,
                &cfg.experimental,
                optim,
                &device,
            )?;
            optim = o;
            println!(
                "resumed {} at update {}",
                dir.display(),
                meta.update_counter
            );
            let parent_hash = std::fs::read(dir.join("experiment.json"))
                .ok()
                .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
                .and_then(|v| v["experiment_hash"].as_str().map(str::to_owned));
            let init = serde_json::json!({
                "kind": "resume",
                "checkpoint": dir.display().to_string(),
                "parent_model_id": meta.model_id,
                "parent_experiment_hash": parent_hash,
                "resume_update_counter": meta.update_counter,
            });
            (m, meta.update_counter as usize, init)
        }
        None => (
            model_io::build_chimera::<B>(&cfg.model, &cfg.experimental, &device)?,
            0,
            serde_json::json!({"kind": "fresh", "seed": args.seed}),
        ),
    };
    let record = experiment_record(cfg, &targets, args, train_pos.len(), init, start_update)?;
    std::fs::create_dir_all(&args.out)?;
    std::fs::write(
        args.out.join("experiment.json"),
        serde_json::to_vec_pretty(&record)?,
    )?;
    println!(
        "train-probe: experiment {}...  T_train={} supervision={} w_int={} lr={} updates={} train={} val={} rungs={}",
        &record["experiment_hash"].as_str().unwrap_or("")[..12],
        args.thoughts,
        mode.label(),
        weight,
        args.lr,
        args.updates,
        train_pos.len(),
        val_pos.len(),
        rungs
    );

    let started = Instant::now();
    let (result, gpu) = monitor(true, || -> anyhow::Result<(ChimeraModel<B>, _)> {
        let mut first_loss = f32::NAN;
        for u in 0..args.updates {
            if started.elapsed().as_secs_f64() > args.max_seconds {
                anyhow::bail!("max_seconds {} exceeded at update {u}", args.max_seconds);
            }
            let lr =
                args.lr * (((start_update + u + 1) as f64) / args.warmup.max(1) as f64).min(1.0);
            let mut acc = GradientsAccumulator::new();
            let mut loss_v = 0.0f32;
            let first = ((start_update + u) * chunks_per_update) % chunks.len();
            let selected: Vec<&Data<B>> = (0..chunks_per_update)
                .map(|k| &chunks[(first + k) % chunks.len()])
                .collect();
            let n_update: usize = selected.iter().map(|c| c.len()).sum();
            for chunk in selected {
                let out =
                    model.forward_thoughts(&chunk.batch.input, &chunk.batch.cands, args.thoughts);
                // Chunk means weighted by chunk size: the accumulated
                // gradient equals the full-batch mean-loss gradient.
                let frac = chunk.len() as f32 / n_update as f32;
                let loss = thought_loss_with(
                    &out.readouts,
                    rungs,
                    mode,
                    weight,
                    args.thoughts,
                    |r, i| readout_terms(r, &chunk.policy[i], &chunk.value[i], args.value_weight),
                )? * frac;
                let l: f32 = loss.clone().into_scalar().elem();
                anyhow::ensure!(l.is_finite(), "non-finite loss {l} at update {u}");
                loss_v += l;
                acc.accumulate(&model, GradientsParams::from_grads(loss.backward(), &model));
            }
            if u == 0 {
                first_loss = loss_v;
            }
            let grads = acc.grads();
            if u == 0 || (u + 1) % args.eval_every.max(1) == 0 || u + 1 == args.updates {
                let norms = model.subsystem_grad_norms(&grads);
                anyhow::ensure!(
                    norms.iter().all(|(_, n)| n.is_finite()),
                    "non-finite gradient norm at update {u}: {norms:?}"
                );
                let worst = norms.iter().map(|(_, n)| *n).fold(0.0f32, f32::max);
                println!(
                    "  update {:>4}  loss {:.4}  lr {:.2e}  max_subsystem_grad {:.3e}  ({:.0}s)",
                    start_update + u + 1,
                    loss_v,
                    lr,
                    worst,
                    started.elapsed().as_secs_f64()
                );
            }
            model = optim.step(lr, model, grads);
            if let (Some(val_data), true) = (
                &val_data,
                args.eval_every > 0 && (u + 1) % args.eval_every == 0 && u + 1 < args.updates,
            ) {
                let rows = evaluate(&model.valid(), val_data, eval_thoughts);
                let s: Vec<String> = rows
                    .iter()
                    .map(|r| format!("T{}={:.4}", r.t, mean(&r.kl_deep)))
                    .collect();
                println!("    val KL to deep teacher: {}", s.join(" "));
            }
        }
        println!("  first loss {first_loss:.4}");
        Ok((model, optim))
    });
    let (model, optim) = result?;
    start_update += args.updates;

    let meta = CheckpointMeta::new(
        cfg.model.clone(),
        1,
        false,
        start_update as u64,
        args.lr,
        args.seed,
        0,
        "x15",
        "fp32",
    )
    .with_experimental(cfg.experimental.clone());
    save_training_chimera(&args.out, &model, &optim, &meta)?;
    println!(
        "saved {} (update {start_update}); gpu peak_vram={:?} MiB util_busy_mean={:?}",
        args.out.display(),
        gpu.peak_vram_mb,
        gpu.util_busy_mean
    );

    if let Some(val_data) = &val_data {
        let rows = evaluate(&model.valid(), val_data, eval_thoughts);
        println!("final held-out evaluation (SAME weights, diagnostic forward):");
        print_eval("val", &rows);
        std::fs::write(
            args.out.join("eval-val.json"),
            serde_json::to_vec_pretty(&eval_json(&rows, &val_ids))?,
        )?;
    } else {
        println!(
            "no val split in this targets file: evaluate the checkpoint with eval-reasoning / eval-tactics"
        );
    }
    Ok(())
}

pub fn run_train(args: TrainProbeArgs) -> anyhow::Result<()> {
    let mut cfg = load(&args.config)?;
    match cfg.device {
        recur64_model::config::DeviceKind::Cpu => {
            train::<recur64_model::train::CpuTrainBackend>(&mut cfg, &args)
        }
        recur64_model::config::DeviceKind::Cuda => {
            #[cfg(feature = "cuda")]
            {
                train::<burn::backend::Autodiff<burn::backend::Cuda>>(&mut cfg, &args)
            }
            #[cfg(not(feature = "cuda"))]
            {
                anyhow::bail!("CUDA support is not compiled; rebuild with --features cuda")
            }
        }
    }
}

fn load(path: &Path) -> anyhow::Result<ProbeConfig> {
    let cfg = ProbeConfig::from_toml_str(&std::fs::read_to_string(path)?)?;
    anyhow::ensure!(
        cfg.experimental.is_chimera(),
        "{} is not chimera_v1",
        path.display()
    );
    cfg.experimental.validate(cfg.model.width)?;
    Ok(cfg)
}

// --- eval ----------------------------------------------------------------------------

/// Evaluate one checkpoint on `pos` at T=1..tmax. Inputs are built from the
/// CHECKPOINT'S OWN experimental contract (compute / visual / candidate facts),
/// so variants with different pathways can be evaluated by one command.
pub(crate) fn ckpt_rows<B: Backend>(
    cfg: &ProbeConfig,
    ck: &Path,
    pos: &[&PositionTarget],
    tmax: usize,
    device: &B::Device,
) -> anyhow::Result<(Vec<PerThought>, Vec<String>)> {
    let meta: CheckpointMeta = serde_json::from_slice(&std::fs::read(ck.join("meta.json"))?)?;
    let mut cfg_ck = cfg.clone();
    cfg_ck.experimental = meta.experimental.clone();
    let data = build_data::<B>(pos, &cfg_ck, device, 0)?;
    let model = model_io::load_chimera::<B>(ck, &cfg.model, &meta.experimental, device)?;
    let tmax = if meta.experimental.reasoning.enabled {
        tmax
    } else {
        1
    };
    Ok((evaluate(&model, &data, tmax), data.categories))
}

/// Pooled, position-clustered effect of `T=t` vs `T=1` across checkpoints.
///
/// The experimental unit is the held-out POSITION: every checkpoint evaluates
/// the same positions, so per-position deltas are first averaged over
/// checkpoints (seeds), and only then bootstrapped over positions. Seeds are
/// never concatenated as if they were independent observations.
fn pooled_effect(per_ckpt: &[Vec<PerThought>], t: usize) -> serde_json::Value {
    let n = per_ckpt[0][0].kl_deep.len();
    let avg = |k: usize| -> Vec<f32> {
        (0..n)
            .map(|i| per_ckpt.iter().map(|c| c[k].kl_deep[i]).sum::<f32>() / per_ckpt.len() as f32)
            .collect()
    };
    let (a, b) = (avg(t - 1), avg(0));
    let (d, lo, hi) = paired_bootstrap(&a, &b);
    serde_json::json!({
        "t": t, "positions": n, "checkpoints": per_ckpt.len(),
        "mean_dkl_vs_t1": d, "lo95": lo, "hi95": hi,
        "even_half": half_diff(&a, &b, 0), "odd_half": half_diff(&a, &b, 1),
        "method": "mean over checkpoints per position, then paired bootstrap over positions (2000 resamples, deterministic)",
    })
}

fn eval_ckpt<B: Backend>(cfg: &ProbeConfig, args: &EvalArgs) -> anyhow::Result<()> {
    let device: B::Device = Default::default();
    let targets = ReasoningTargetsV1::load(&args.targets)?;
    println!(
        "eval-reasoning: split={} targets_digest={}... (same weights, T=1..{})",
        args.split,
        &targets.digest[..12],
        args.thoughts
    );
    let mut out = serde_json::Map::new();
    let splits: Vec<&str> = if args.include_train {
        vec![args.split.as_str(), "train"]
    } else {
        vec![args.split.as_str()]
    };
    for name in splits {
        let pos = split(&targets, name);
        anyhow::ensure!(!pos.is_empty(), "no positions in split {name:?}");
        let ids: Vec<String> = pos.iter().map(|p| p.id.clone()).collect();
        let mut per_ckpt: Vec<Vec<PerThought>> = Vec::new();
        let mut per_json = Vec::new();
        let mut categories = Vec::new();
        for ck in &args.checkpoint {
            let (rows, cats) = ckpt_rows::<B>(cfg, ck, &pos, args.thoughts, &device)?;
            categories = cats;
            println!("checkpoint {}", ck.display());
            print_eval(name, &rows);
            let mut j = eval_json(&rows, &ids);
            j["checkpoint"] = serde_json::json!(ck.display().to_string());
            per_json.push(j);
            per_ckpt.push(rows);
        }
        let mut entry = serde_json::json!({"checkpoints": per_json, "categories": categories});
        if per_ckpt.len() > 1 && args.thoughts >= 2 {
            println!(
                "  POOLED over {} checkpoints, clustered by position:",
                per_ckpt.len()
            );
            let mut pooled = Vec::new();
            for t in 2..=args.thoughts {
                let p = pooled_effect(&per_ckpt, t);
                println!(
                    "    T{t} - T1: {:+.4}  95% CI [{:+.4},{:+.4}]  even/odd half {:+.4}/{:+.4}",
                    p["mean_dkl_vs_t1"].as_f64().unwrap_or(f64::NAN),
                    p["lo95"].as_f64().unwrap_or(f64::NAN),
                    p["hi95"].as_f64().unwrap_or(f64::NAN),
                    p["even_half"].as_f64().unwrap_or(f64::NAN),
                    p["odd_half"].as_f64().unwrap_or(f64::NAN)
                );
                pooled.push(p);
            }
            entry["pooled_position_clustered"] = serde_json::json!(pooled);
        }
        if !args.baseline.is_empty() {
            let mut base_rows: Vec<Vec<f32>> = Vec::new();
            for bk in &args.baseline {
                let (rows, _) = ckpt_rows::<B>(cfg, bk, &pos, 1, &device)?;
                base_rows.push(rows[0].kl_deep.clone());
            }
            let nb = base_rows.len() as f32;
            let base: Vec<f32> = (0..pos.len())
                .map(|i| base_rows.iter().map(|r| r[i]).sum::<f32>() / nb)
                .collect();
            let tmax = args.thoughts;
            let cand: Vec<f32> = (0..pos.len())
                .map(|i| {
                    per_ckpt.iter().map(|c| c[tmax - 1].kl_deep[i]).sum::<f32>()
                        / per_ckpt.len() as f32
                })
                .collect();
            let (d, lo, hi) = paired_bootstrap(&cand, &base);
            println!(
                "  VS BASELINE (T_train=1 at T=1, {} ckpt): checkpoint T{tmax} mean KL {:.4} vs baseline {:.4}; diff {:+.4} 95% CI [{:+.4},{:+.4}] (position-clustered)",
                base_rows.len(),
                mean(&cand),
                mean(&base),
                d,
                lo,
                hi
            );
            entry["vs_baseline"] = serde_json::json!({
                "t": tmax, "candidate_mean_kl": mean(&cand), "baseline_mean_kl": mean(&base),
                "diff": d, "lo95": lo, "hi95": hi,
            });
        }
        out.insert(name.into(), entry);
    }
    if let Some(p) = &args.json_out {
        std::fs::write(
            p,
            serde_json::to_vec_pretty(&serde_json::Value::Object(out))?,
        )?;
    }
    Ok(())
}

pub fn run_eval(args: EvalArgs) -> anyhow::Result<()> {
    let cfg = load(&args.config)?;
    match cfg.device {
        recur64_model::config::DeviceKind::Cpu => eval_ckpt::<burn::backend::Flex>(&cfg, &args),
        recur64_model::config::DeviceKind::Cuda => {
            #[cfg(feature = "cuda")]
            {
                let (r, gpu) = monitor(true, || eval_ckpt::<burn::backend::Cuda>(&cfg, &args));
                println!("gpu telemetry   : peak_vram={:?} MiB", gpu.peak_vram_mb);
                r
            }
            #[cfg(not(feature = "cuda"))]
            {
                anyhow::bail!("CUDA support is not compiled; rebuild with --features cuda")
            }
        }
    }
}

/// FNV-1a 64-bit hash of a position id (stable across platforms).
fn id_hash(id: &str) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in id.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}
