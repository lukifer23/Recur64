//! `recur64 x2 ...` - Chimera V2 experiment harness (info / gen / train / eval /
//! analyze / bench). Policy-primary training on the exact tool-necessity dataset.
//!
//! Pre-registered choices (docs/HP_X2_EXPERIMENTS.md):
//!  * loss = policy CE to the uniform-over-correct target + `value_weight` (0.1) x
//!    WDL CE toward "win" (every exact mate-in-2 root is a win);
//!  * variable budgets `uniform_1_4_v1`: each optimizer update contains one
//!    micro-batch per budget T=1..4 (rotated), final-readout-only loss per micro-batch
//!    (`budget_final_v1`); single-pass variants train at fixed T=1;
//!  * one AdamW contract for all budgets; LR chosen on the tune set, never per budget.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use burn::module::AutodiffModule;
use burn::optim::{GradientsAccumulator, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use clap::{Args, Subcommand};
use serde::Serialize;
use sha2::{Digest, Sha256};

use recur64_compute::{NativeWorldModel, WasmWorldModel, WorldModelProvider};
use recur64_coproc::world::{HISTORY_CONTRACT, WorldHorizon, WorldStats};
use recur64_core::GameState;
use recur64_model::checkpoint::{CheckpointMeta, save_training_v2};
use recur64_model::chimera2::{ChimeraV2Model, V2Options};
use recur64_model::config::ProbeConfig;
use recur64_model::experimental::InfoSchedule;
use recur64_model::loss::{Targets, policy_ce, wdl_ce};
use recur64_runtime::gpu_telemetry::monitor;
use recur64_runtime::model_io;
use recur64_runtime::v2_inputs::{build_v2_batch, compute_world_bytes, horizon_for_budget};

use crate::x2_data::{GenArgs, V2Data, run_gen};
use crate::x15_train::{mean, paired_bootstrap};

#[derive(Args, Debug)]
pub struct X2Args {
    #[command(subcommand)]
    pub command: X2Command,
}

#[derive(Subcommand, Debug)]
pub enum X2Command {
    /// Parameters by subsystem, identity, and per-budget compute counts.
    Info(InfoArgs),
    /// Generate the exact tool-necessity mate-in-2 train/tune/confirm splits + audits.
    Gen(GenArgs),
    /// Train one V2 variant (variable-budget or fixed-budget).
    Train(TrainArgs),
    /// Per-position evaluation of checkpoint(s) at T=1..N (same weights).
    Eval(EvalArgs),
    /// Pre-registered Q1/Q2/Q3 analysis and outcome classification.
    Analyze(AnalyzeArgs),
    /// Per-horizon world-model cost (native, WASM) and per-budget GPU forward cost.
    Bench(BenchArgs),
}

pub fn run(args: X2Args) -> anyhow::Result<()> {
    // The generated Burn record types are deep: run everything on a large stack.
    let handle = std::thread::Builder::new()
        .name("x2".into())
        .stack_size(256 * 1024 * 1024)
        .spawn(move || match args.command {
            X2Command::Info(a) => run_info(a),
            X2Command::Gen(a) => run_gen(a),
            X2Command::Train(a) => run_train(a),
            X2Command::Eval(a) => run_eval(a),
            X2Command::Analyze(a) => run_analyze(a),
            X2Command::Bench(a) => run_bench(a),
        })?;
    handle
        .join()
        .unwrap_or_else(|_| Err(anyhow::anyhow!("x2 worker thread panicked")))
}

fn load_cfg(path: &Path) -> anyhow::Result<ProbeConfig> {
    let cfg = ProbeConfig::from_toml_str(&std::fs::read_to_string(path)?)?;
    anyhow::ensure!(
        cfg.experimental.is_chimera_v2(),
        "{} is not a chimera_v2 config",
        path.display()
    );
    cfg.experimental.validate(cfg.model.width)?;
    cfg.experimental.validate_v2_model(&cfg.model)?;
    Ok(cfg)
}

fn file_sha(path: &Path) -> anyhow::Result<String> {
    let mut h = Sha256::new();
    h.update(std::fs::read(path)?);
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn provider(cfg: &ProbeConfig) -> anyhow::Result<Option<Box<dyn WorldModelProvider>>> {
    let v = &cfg.experimental.v2;
    if v.info_schedule.needs_world_model() {
        Ok(Some(recur64_compute::world_model_provider_for(
            v.world_provider,
        )?))
    } else {
        Ok(None)
    }
}

/// Full-horizon world bytes for every state, computed on `threads` workers.
fn cache_world(
    states: &[GameState],
    prov: &dyn WorldModelProvider,
    w_cap: usize,
    r_cap: usize,
    threads: usize,
) -> anyhow::Result<Vec<Vec<u8>>> {
    let per = states.len().div_ceil(threads.max(1)).max(1);
    let parts: Vec<anyhow::Result<Vec<Vec<u8>>>> = std::thread::scope(|s| {
        let hs: Vec<_> = states
            .chunks(per)
            .map(|c| {
                s.spawn(move || compute_world_bytes(c, prov, w_cap, r_cap, WorldHorizon::Replies))
            })
            .collect();
        hs.into_iter()
            .map(|h| {
                h.join()
                    .unwrap_or_else(|_| Err(anyhow::anyhow!("world worker panicked")))
            })
            .collect()
    });
    let mut out = Vec::with_capacity(states.len());
    for p in parts {
        out.extend(p?);
    }
    Ok(out)
}

// --- budgets and targets ---------------------------------------------------------

#[derive(Clone, Copy, Debug)]
enum Budget {
    /// Every update trains T=1..4 (one micro-batch each, rotated).
    Uniform14,
    Fixed(usize),
}

impl Budget {
    fn parse(s: &str) -> anyhow::Result<Self> {
        if s == "uniform_1_4_v1" {
            return Ok(Budget::Uniform14);
        }
        if let Some(t) = s.strip_prefix("fixed_")
            && let Ok(t) = t.parse::<usize>()
            && (1..=8).contains(&t)
        {
            return Ok(Budget::Fixed(t));
        }
        anyhow::bail!("unknown budget policy {s:?} (uniform_1_4_v1 | fixed_<T>)")
    }
    fn label(self) -> String {
        match self {
            Budget::Uniform14 => "uniform_1_4_v1".into(),
            Budget::Fixed(t) => format!("fixed_{t}"),
        }
    }
    fn t_for(self, update: usize, k: usize) -> usize {
        match self {
            Budget::Uniform14 => 1 + (update + k) % 4,
            Budget::Fixed(t) => t,
        }
    }
}

fn targets_for<B: Backend>(
    data: &V2Data,
    idx: &[usize],
    width: usize,
    device: &B::Device,
) -> Targets<B> {
    let mut pol = vec![0f32; idx.len() * width];
    for (row, &i) in idx.iter().enumerate() {
        for (c, v) in data.target(i).into_iter().enumerate() {
            pol[row * width + c] = v;
        }
    }
    Targets {
        policy_target: Tensor::<B, 2>::from_data(
            burn::tensor::TensorData::new(pol, [idx.len(), width]),
            device,
        ),
        wdl_target: Tensor::<B, 1, Int>::zeros([idx.len()], device),
        wdl_mask: None,
    }
}

// --- evaluation ------------------------------------------------------------------

#[derive(Serialize, Clone, Default)]
struct PlannerRow {
    thought: usize,
    mean_abs: f32,
    rms: f32,
    delta: f32,
    gate_mean: f32,
    gate_std: f32,
    share_board: f32,
    share_cand: f32,
    share_succ: f32,
    share_reply: f32,
}

#[derive(Serialize, Clone, Default)]
struct BudgetEval {
    t: usize,
    horizon: u8,
    top1: Vec<f32>,
    mass: Vec<f32>,
    ce: Vec<f32>,
    entropy: Vec<f32>,
    planner: Vec<PlannerRow>,
    board_encoder_runs: usize,
    planner_steps: usize,
    world_stats: WorldStatsJson,
    tool_cpu_us: u64,
    forward_us: u64,
}

#[derive(Serialize, Clone, Default)]
struct WorldStatsJson {
    root_candidates: u64,
    root_moves_applied: u64,
    reply_moves_enumerated: u64,
    reply_moves_applied: u64,
    next_moves_enumerated: u64,
    next_moves_applied: u64,
    continuation_moves_enumerated: u64,
    successor_records: u64,
    reply_records: u64,
}

impl WorldStatsJson {
    fn add(&mut self, s: &WorldStats) {
        self.root_candidates += u64::from(s.root_candidates);
        self.root_moves_applied += u64::from(s.root_moves_applied);
        self.reply_moves_enumerated += u64::from(s.reply_moves_enumerated);
        self.reply_moves_applied += u64::from(s.reply_moves_applied);
        self.next_moves_enumerated += u64::from(s.next_moves_enumerated);
        self.next_moves_applied += u64::from(s.next_moves_applied);
        self.continuation_moves_enumerated += u64::from(s.continuation_moves_enumerated);
        self.successor_records += u64::from(s.successor_records);
        self.reply_records += u64::from(s.reply_records);
    }
}

fn to_vec<B: Backend>(t: Tensor<B, 1>) -> Vec<f32> {
    t.into_data().to_vec::<f32>().unwrap_or_default()
}

/// Evaluate `model` on every position of `data` at budget `t`. Inputs are built at
/// the least world-model horizon the schedule needs at `t` (real per-horizon work).
#[allow(clippy::too_many_arguments)]
fn eval_budget<B: Backend>(
    model: &ChimeraV2Model<B>,
    cfg: &ProbeConfig,
    data: &V2Data,
    states: &[GameState],
    prov: Option<&dyn WorldModelProvider>,
    t: usize,
    batch: usize,
    device: &B::Device,
) -> anyhow::Result<BudgetEval> {
    let sched = cfg.experimental.v2.info_schedule;
    let horizon = horizon_for_budget(sched, t).unwrap_or(WorldHorizon::Root);
    let mut ev = BudgetEval {
        t,
        horizon: horizon.code(),
        ..Default::default()
    };
    let mut planner_acc: Vec<PlannerRow> = Vec::new();
    let mut weight = 0f32;
    for (ci, chunk) in states.chunks(batch).enumerate() {
        let b = build_v2_batch::<B>(chunk, &cfg.experimental, prov, None, horizon, device)?;
        ev.tool_cpu_us += b.phases.world_us;
        ev.world_stats.add(&b.phases.stats);
        let t0 = Instant::now();
        let out = model.forward(&b.input, &b.cands, t, V2Options::default());
        let readout = out.readouts.last().expect("a forward yields a readout");
        let width = b.cands.width;
        let lp = readout
            .policy
            .log_probs
            .clone()
            .into_data()
            .to_vec::<f32>()
            .unwrap_or_default();
        ev.forward_us += t0.elapsed().as_micros() as u64;
        ev.board_encoder_runs += out.counts.board_encoder_runs;
        ev.planner_steps += out.counts.planner_steps;
        for (row, pos) in data.positions[ci * batch..ci * batch + chunk.len()]
            .iter()
            .enumerate()
        {
            let l = &lp[row * width..row * width + pos.legal_n];
            anyhow::ensure!(
                l.iter().all(|v| v.is_finite()),
                "non-finite log-probability at {} (T={t})",
                pos.id
            );
            let p: Vec<f32> = l.iter().map(|v| v.exp()).collect();
            let top = p
                .iter()
                .enumerate()
                .fold(
                    (0usize, f32::MIN),
                    |a, (i, v)| if *v > a.1 { (i, *v) } else { a },
                )
                .0;
            ev.top1.push(f32::from(pos.correct.contains(&top)));
            ev.mass.push(pos.correct.iter().map(|c| p[*c]).sum());
            ev.ce
                .push(-pos.correct.iter().map(|c| l[*c]).sum::<f32>() / pos.correct.len() as f32);
            ev.entropy
                .push(-p.iter().zip(l).map(|(p, l)| p * l).sum::<f32>());
        }
        // Planner diagnostics, averaged over positions (weighted by chunk size).
        let w = chunk.len() as f32;
        if planner_acc.is_empty() {
            planner_acc = vec![PlannerRow::default(); out.diag.len()];
        }
        for (acc, d) in planner_acc.iter_mut().zip(&out.diag) {
            let m = |x: Tensor<B, 1>| mean(&to_vec(x)) * w;
            acc.thought = d.thought;
            acc.mean_abs += m(d.mean_abs.clone());
            acc.rms += m(d.rms.clone());
            acc.delta += m(d.delta.clone());
            acc.gate_mean += m(d.gate_mean.clone());
            acc.gate_std += m(d.gate_std.clone());
            acc.share_board += m(d.share_board.clone());
            acc.share_cand += m(d.share_cand.clone());
            acc.share_succ += m(d.share_succ.clone());
            acc.share_reply += m(d.share_reply.clone());
        }
        weight += w;
    }
    for a in &mut planner_acc {
        for f in [
            &mut a.mean_abs,
            &mut a.rms,
            &mut a.delta,
            &mut a.gate_mean,
            &mut a.gate_std,
            &mut a.share_board,
            &mut a.share_cand,
            &mut a.share_succ,
            &mut a.share_reply,
        ] {
            *f /= weight.max(1.0);
        }
    }
    ev.planner = planner_acc;
    Ok(ev)
}

// --- train -----------------------------------------------------------------------

#[derive(Args, Debug)]
pub struct TrainArgs {
    #[arg(long)]
    pub config: PathBuf,
    #[arg(long)]
    pub train: PathBuf,
    #[arg(long)]
    pub tune: PathBuf,
    #[arg(long)]
    pub out: PathBuf,
    /// `uniform_1_4_v1` (variable budget) or `fixed_<T>`.
    #[arg(long, default_value = "uniform_1_4_v1")]
    pub budget: String,
    #[arg(long, default_value_t = 1e-4)]
    pub lr: f64,
    #[arg(long, default_value_t = 30)]
    pub warmup: usize,
    #[arg(long, default_value_t = 400)]
    pub updates: usize,
    /// Weight of the WDL-toward-win term (policy is the primary objective).
    #[arg(long, default_value_t = 0.1)]
    pub value_weight: f32,
    #[arg(long, default_value_t = 1)]
    pub seed: u64,
    #[arg(long, default_value_t = 32)]
    pub micro_batch: usize,
    /// Positions per optimizer update (a multiple of --micro-batch).
    #[arg(long, default_value_t = 128)]
    pub batch_positions: usize,
    #[arg(long, default_value_t = 50)]
    pub eval_every: usize,
    /// Train on this many positions only (micro-overfit); 0 = all.
    #[arg(long, default_value_t = 0)]
    pub limit: usize,
    #[arg(long)]
    pub resume: Option<PathBuf>,
    #[arg(long, default_value_t = 3000.0)]
    pub max_seconds: f64,
    #[arg(long, default_value_t = 6)]
    pub threads: usize,
}

fn train_generic<B: AutodiffBackend>(cfg: &ProbeConfig, args: &TrainArgs) -> anyhow::Result<()> {
    let v = &cfg.experimental.v2;
    let budget = Budget::parse(&args.budget)?;
    anyhow::ensure!(
        args.batch_positions.is_multiple_of(args.micro_batch)
            && args.batch_positions >= args.micro_batch,
        "batch_positions must be a multiple of micro_batch"
    );
    if let Budget::Fixed(t) = budget {
        anyhow::ensure!(t <= v.max_thoughts, "fixed budget above max_thoughts");
    }
    let train = V2Data::load(&args.train)?;
    let tune = V2Data::load(&args.tune)?;
    // Hard gates: capacity and the history contract, recomputed from the positions.
    let a_train = train.audit_and_enforce(v.w_cap, v.r_cap)?;
    let a_tune = tune.audit_and_enforce(v.w_cap, v.r_cap)?;
    println!(
        "audits ok: train {} (max legal {}, max replies {}), tune {} (max legal {}, max replies {}); caps w={} r={}; {HISTORY_CONTRACT}",
        a_train.positions,
        a_train.max_legal,
        a_train.max_replies,
        a_tune.positions,
        a_tune.max_legal,
        a_tune.max_replies,
        v.w_cap,
        v.r_cap
    );
    let mut order: Vec<usize> = (0..train.positions.len()).collect();
    if args.limit > 0 {
        order.truncate(args.limit);
    }
    order.sort_by_key(|i| (mix(args.seed ^ mix(*i as u64)), *i));
    let states = train.states()?;
    let tune_states = tune.states()?;
    let prov = provider(cfg)?;
    let world = match &prov {
        Some(p) => {
            let t0 = Instant::now();
            let w = cache_world(&states, p.as_ref(), v.w_cap, v.r_cap, args.threads)?;
            println!(
                "world model cached for {} positions in {:.1}s ({} MB)",
                w.len(),
                t0.elapsed().as_secs_f64(),
                w.iter().map(Vec::len).sum::<usize>() / 1_000_000
            );
            Some(w)
        }
        None => None,
    };
    let device: B::Device = Default::default();
    B::seed(&device, args.seed);
    let mut optim = recur64_model::train::adamw::<B, ChimeraV2Model<B>>();
    let (mut model, start_update) = match &args.resume {
        Some(dir) => {
            let (m, o, meta) = model_io::load_chimera_v2_training::<B, _>(
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
            (m, meta.update_counter as usize)
        }
        None => (
            model_io::build_chimera_v2::<B>(&cfg.model, &cfg.experimental, &device)?,
            0,
        ),
    };
    let record = serde_json::json!({
        "schema": "x2_experiment_v1",
        "config": cfg.name,
        "identity": cfg.experimental.identity()?,
        "model": cfg.model,
        "params": model.num_params(),
        "budget": budget.label(),
        "loss": {"policy": "ce_to_uniform_correct", "value_weight": args.value_weight, "value": "wdl_ce_to_win"},
        "optimizer": {"kind": "adamw", "lr": args.lr, "warmup": args.warmup},
        "updates": args.updates, "start_update": start_update,
        "micro_batch": args.micro_batch, "batch_positions": args.batch_positions,
        "seed": args.seed, "train_positions": order.len(),
        "train_sha256": file_sha(&args.train)?, "tune_sha256": file_sha(&args.tune)?,
        "history_contract": HISTORY_CONTRACT,
        "world_model_version": recur64_coproc::world::WORLD_MODEL_VERSION,
        // Training reads a Replies-horizon cache: capability training only. Latency /
        // compute-frontier numbers come from `x2 bench` / `x2 eval`, which run LIVE.
        "tool_compute_mode": if world.is_some() { "cached" } else { "none" },
        "world_cache": world.as_ref().map(|_| serde_json::json!({
            "horizon": "replies", "w_cap": v.w_cap, "r_cap": v.r_cap,
            "positions": order.len(),
        })),
        "git": recur64_runtime::provenance::git_revision().unwrap_or("unknown"),
    });
    let hash = {
        let mut h = Sha256::new();
        h.update(serde_json::to_vec(&record)?);
        h.finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    };
    std::fs::create_dir_all(&args.out)?;
    let mut rec = record;
    rec["experiment_hash"] = serde_json::Value::String(hash.clone());
    std::fs::write(
        args.out.join("experiment.json"),
        serde_json::to_vec_pretty(&rec)?,
    )?;
    println!(
        "x2 train: {} schedule={} budget={} lr={} updates={} params={} experiment {}",
        cfg.name,
        v.info_schedule.label(),
        budget.label(),
        args.lr,
        args.updates,
        model.num_params(),
        &hash[..12]
    );

    let chunks_per_update = args.batch_positions / args.micro_batch;
    let mut cursor = start_update * args.batch_positions;
    let started = Instant::now();
    let tune_max_t = if matches!(v.info_schedule, InfoSchedule::Progressive) {
        4
    } else {
        1
    };
    let (result, gpu) = monitor(true, || -> anyhow::Result<(ChimeraV2Model<B>, _)> {
        for u in 0..args.updates {
            anyhow::ensure!(
                started.elapsed().as_secs_f64() <= args.max_seconds,
                "max_seconds {} exceeded at update {u}",
                args.max_seconds
            );
            let lr =
                args.lr * (((start_update + u + 1) as f64) / args.warmup.max(1) as f64).min(1.0);
            let mut acc = GradientsAccumulator::new();
            let mut by_t: BTreeMap<usize, (f32, usize)> = BTreeMap::new();
            for k in 0..chunks_per_update {
                let idx: Vec<usize> = (0..args.micro_batch)
                    .map(|j| order[(cursor + k * args.micro_batch + j) % order.len()])
                    .collect();
                let chunk: Vec<GameState> = idx.iter().map(|i| states[*i].clone()).collect();
                let wb: Option<Vec<Vec<u8>>> = world
                    .as_ref()
                    .map(|w| idx.iter().map(|i| w[*i].clone()).collect());
                let t = budget.t_for(start_update + u, k);
                let b = build_v2_batch::<B>(
                    &chunk,
                    &cfg.experimental,
                    prov.as_deref(),
                    wb.as_deref(),
                    WorldHorizon::Replies,
                    &device,
                )?;
                let tg = targets_for::<B>(&train, &idx, b.cands.width, &device);
                let out = model.forward(&b.input, &b.cands, t, V2Options::default());
                let r = out.readouts.last().expect("readout");
                let loss = (policy_ce(&r.policy, &tg.policy_target)
                    + wdl_ce(&r.wdl_logits, &tg.wdl_target) * args.value_weight)
                    / chunks_per_update as f32;
                let l: f32 = loss.clone().into_scalar().elem();
                anyhow::ensure!(l.is_finite(), "non-finite loss {l} at update {u} (T={t})");
                let e = by_t.entry(t).or_insert((0.0, 0));
                e.0 += l * chunks_per_update as f32;
                e.1 += 1;
                acc.accumulate(&model, GradientsParams::from_grads(loss.backward(), &model));
            }
            cursor += args.batch_positions;
            model = optim.step(lr, model, acc.grads());
            if u == 0 || (u + 1) % 10 == 0 || u + 1 == args.updates {
                let s: Vec<String> = by_t
                    .iter()
                    .map(|(t, (l, n))| format!("T{t}={:.4}", l / *n as f32))
                    .collect();
                println!(
                    "  update {:>4}  loss {}  lr {:.2e}  ({:.0}s)",
                    start_update + u + 1,
                    s.join(" "),
                    lr,
                    started.elapsed().as_secs_f64()
                );
            }
            if args.eval_every > 0 && (u + 1) % args.eval_every == 0 && u + 1 < args.updates {
                let m = model.valid();
                let inner: Device<B::InnerBackend> = Default::default();
                let mut line = Vec::new();
                for t in 1..=tune_max_t {
                    let e = eval_budget::<B::InnerBackend>(
                        &m,
                        cfg,
                        &tune,
                        &tune_states,
                        prov.as_deref(),
                        t,
                        64,
                        &inner,
                    )?;
                    line.push(format!(
                        "T{t} top1 {:.3} mass {:.3}",
                        mean(&e.top1),
                        mean(&e.mass)
                    ));
                }
                println!("    tune: {}", line.join(" | "));
            }
        }
        Ok((model, optim))
    });
    let (model, optim) = result?;
    let meta = CheckpointMeta::new(
        cfg.model.clone(),
        1,
        false,
        (start_update + args.updates) as u64,
        args.lr,
        args.seed,
        0,
        "x2",
        "fp32",
    )
    .with_experimental(cfg.experimental.clone());
    save_training_v2(&args.out, &model, &optim, &meta)?;
    println!(
        "saved {} (update {}); {:.0}s; gpu peak_vram={:?} MiB util_busy_mean={:?}",
        args.out.display(),
        start_update + args.updates,
        started.elapsed().as_secs_f64(),
        gpu.peak_vram_mb,
        gpu.util_busy_mean
    );
    // Final tune evaluation, all budgets.
    let m = model.valid();
    let inner: Device<B::InnerBackend> = Default::default();
    let mut rows = serde_json::Map::new();
    for t in 1..=tune_max_t {
        let e = eval_budget::<B::InnerBackend>(
            &m,
            cfg,
            &tune,
            &tune_states,
            prov.as_deref(),
            t,
            64,
            &inner,
        )?;
        println!(
            "  final tune T{t}: top1 {:.4}  mass {:.4}  ce {:.4}  (chance {:.4})",
            mean(&e.top1),
            mean(&e.mass),
            mean(&e.ce),
            tune.audit.mean_chance
        );
        rows.insert(
            format!("T{t}"),
            serde_json::json!({"top1": mean(&e.top1), "mass": mean(&e.mass), "ce": mean(&e.ce)}),
        );
    }
    std::fs::write(
        args.out.join("tune-summary.json"),
        serde_json::to_vec_pretty(&serde_json::Value::Object(rows))?,
    )?;
    Ok(())
}

fn run_train(args: TrainArgs) -> anyhow::Result<()> {
    let cfg = load_cfg(&args.config)?;
    match cfg.device {
        recur64_model::config::DeviceKind::Cpu => {
            train_generic::<recur64_model::train::CpuTrainBackend>(&cfg, &args)
        }
        recur64_model::config::DeviceKind::Cuda => {
            #[cfg(feature = "cuda")]
            {
                train_generic::<burn::backend::Autodiff<burn::backend::Cuda>>(&cfg, &args)
            }
            #[cfg(not(feature = "cuda"))]
            {
                anyhow::bail!("CUDA support is not compiled; rebuild with --features cuda")
            }
        }
    }
}

// --- eval ------------------------------------------------------------------------

#[derive(Args, Debug)]
pub struct EvalArgs {
    #[arg(long)]
    pub config: PathBuf,
    #[arg(long)]
    pub data: PathBuf,
    #[arg(long)]
    pub checkpoint: PathBuf,
    /// Largest budget evaluated (default: 4 for progressive, else 1).
    #[arg(long)]
    pub thoughts: Option<usize>,
    #[arg(long)]
    pub json_out: PathBuf,
    #[arg(long, default_value_t = 64)]
    pub batch: usize,
}

fn eval_generic<B: Backend>(cfg: &ProbeConfig, args: &EvalArgs) -> anyhow::Result<()> {
    let v = &cfg.experimental.v2;
    let data = V2Data::load(&args.data)?;
    let audit = data.audit_and_enforce(v.w_cap, v.r_cap)?;
    let states = data.states()?;
    let device: B::Device = Default::default();
    let model =
        model_io::load_chimera_v2::<B>(&args.checkpoint, &cfg.model, &cfg.experimental, &device)?;
    let prov = provider(cfg)?;
    let tmax = args
        .thoughts
        .unwrap_or(if matches!(v.info_schedule, InfoSchedule::Progressive) {
            4
        } else {
            1
        });
    let mut budgets = serde_json::Map::new();
    for t in 1..=tmax {
        let e = eval_budget::<B>(
            &model,
            cfg,
            &data,
            &states,
            prov.as_deref(),
            t,
            args.batch,
            &device,
        )?;
        println!(
            "{} T{t}: top1 {:.4}  mass {:.4}  ce {:.4}  entropy {:.3}  board_runs {}  planner_steps {}  tool_cpu {:.2}s  forward {:.2}s  (chance {:.4})",
            cfg.name,
            mean(&e.top1),
            mean(&e.mass),
            mean(&e.ce),
            mean(&e.entropy),
            e.board_encoder_runs,
            e.planner_steps,
            e.tool_cpu_us as f64 / 1e6,
            e.forward_us as f64 / 1e6,
            audit.mean_chance
        );
        budgets.insert(format!("T{t}"), serde_json::to_value(&e)?);
    }
    let out = serde_json::json!({
        "schema": "x2_eval_v1",
        "config": cfg.name,
        "schedule": v.info_schedule.label(),
        "checkpoint": args.checkpoint.display().to_string(),
        "data_sha256": file_sha(&args.data)?,
        "history_contract": HISTORY_CONTRACT,
        "mean_chance": audit.mean_chance,
        "ids": data.positions.iter().map(|p| p.id.clone()).collect::<Vec<_>>(),
        "kinds": data.positions.iter().map(|p| p.kind.clone()).collect::<Vec<_>>(),
        "budgets": budgets,
    });
    if let Some(d) = args.json_out.parent() {
        std::fs::create_dir_all(d)?;
    }
    std::fs::write(&args.json_out, serde_json::to_vec(&out)?)?;
    Ok(())
}

fn run_eval(args: EvalArgs) -> anyhow::Result<()> {
    let cfg = load_cfg(&args.config)?;
    match cfg.device {
        recur64_model::config::DeviceKind::Cpu => eval_generic::<burn::backend::Flex>(&cfg, &args),
        recur64_model::config::DeviceKind::Cuda => {
            #[cfg(feature = "cuda")]
            {
                eval_generic::<burn::backend::Cuda>(&cfg, &args)
            }
            #[cfg(not(feature = "cuda"))]
            {
                anyhow::bail!("CUDA support is not compiled; rebuild with --features cuda")
            }
        }
    }
}

// --- analyze ---------------------------------------------------------------------

#[derive(Args, Debug)]
pub struct AnalyzeArgs {
    /// Progressive-variant eval JSONs (one per seed).
    #[arg(long, required = true)]
    pub prog: Vec<PathBuf>,
    /// All-info one-pass eval JSONs (one per seed, matched by order).
    #[arg(long)]
    pub allinfo: Vec<PathBuf>,
    /// Root-only eval JSONs (one per seed).
    #[arg(long)]
    pub root: Vec<PathBuf>,
    #[arg(long)]
    pub json_out: Option<PathBuf>,
}

struct EvalFile {
    v: serde_json::Value,
}

impl EvalFile {
    fn load(p: &Path) -> anyhow::Result<Self> {
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(p)?)?;
        anyhow::ensure!(
            v["schema"] == "x2_eval_v1",
            "{}: not an x2 eval file",
            p.display()
        );
        Ok(Self { v })
    }
    fn metric(&self, t: usize, name: &str) -> Vec<f32> {
        self.v["budgets"][format!("T{t}")][name]
            .as_array()
            .map(|a| {
                a.iter()
                    .map(|x| x.as_f64().unwrap_or(f64::NAN) as f32)
                    .collect()
            })
            .unwrap_or_default()
    }
    fn kinds(&self) -> Vec<String> {
        self.v["kinds"]
            .as_array()
            .map(|a| {
                a.iter()
                    .map(|x| x.as_str().unwrap_or("").to_string())
                    .collect()
            })
            .unwrap_or_default()
    }
}

fn load_all(ps: &[PathBuf]) -> anyhow::Result<Vec<EvalFile>> {
    ps.iter().map(|p| EvalFile::load(p)).collect()
}

fn seed_mean(fs: &[EvalFile], t: usize, m: &str) -> Vec<f32> {
    let cols: Vec<Vec<f32>> = fs.iter().map(|f| f.metric(t, m)).collect();
    let n = cols.first().map_or(0, Vec::len);
    (0..n)
        .map(|i| cols.iter().map(|c| c[i]).sum::<f32>() / cols.len() as f32)
        .collect()
}

#[derive(Serialize)]
struct Cmp {
    label: String,
    metric: String,
    mean_diff: f32,
    lo95: f32,
    hi95: f32,
    per_seed_diff: Vec<f32>,
    /// mean_diff > 0, CI wholly above 0, every seed positive.
    signal: bool,
}

fn compare(label: &str, metric: &str, a: (&[EvalFile], usize), b: (&[EvalFile], usize)) -> Cmp {
    let (ma, mb) = (seed_mean(a.0, a.1, metric), seed_mean(b.0, b.1, metric));
    let (m, lo, hi) = paired_bootstrap(&ma, &mb);
    let per_seed: Vec<f32> =
        a.0.iter()
            .zip(b.0)
            .map(|(x, y)| mean(&x.metric(a.1, metric)) - mean(&y.metric(b.1, metric)))
            .collect();
    let signal = m > 0.0 && lo > 0.0 && per_seed.iter().all(|d| *d > 0.0);
    println!(
        "  {label:<36} {metric:<5} diff {m:+.4}  95% CI [{lo:+.4}, {hi:+.4}]  per-seed {:?}  => {}",
        per_seed
            .iter()
            .map(|d| format!("{d:+.4}"))
            .collect::<Vec<_>>(),
        if signal { "SIGNAL" } else { "no signal" }
    );
    Cmp {
        label: label.into(),
        metric: metric.into(),
        mean_diff: m,
        lo95: lo,
        hi95: hi,
        per_seed_diff: per_seed,
        signal,
    }
}

fn run_analyze(args: AnalyzeArgs) -> anyhow::Result<()> {
    let prog = load_all(&args.prog)?;
    let all = load_all(&args.allinfo)?;
    let root = load_all(&args.root)?;
    anyhow::ensure!(
        all.is_empty() || all.len() == prog.len(),
        "allinfo and prog need the same number of seeds"
    );
    for f in prog.iter().chain(&all).chain(&root) {
        anyhow::ensure!(
            f.v["ids"] == prog[0].v["ids"] && f.v["data_sha256"] == prog[0].v["data_sha256"],
            "eval files were not produced on the same confirmation data"
        );
    }
    println!(
        "chance top-1 (mean correct/legal): {:.4}",
        prog[0].v["mean_chance"].as_f64().unwrap_or(f64::NAN)
    );
    for (name, fs) in [
        ("progressive", &prog),
        ("all-info", &all),
        ("root-only", &root),
    ] {
        for t in 1..=4 {
            if fs.first().is_some_and(|f| !f.metric(t, "top1").is_empty()) {
                println!(
                    "  {name:<12} T{t}: top1 {:.4}  mass {:.4}  ce {:.4}",
                    mean(&seed_mean(fs, t, "top1")),
                    mean(&seed_mean(fs, t, "mass")),
                    mean(&seed_mean(fs, t, "ce"))
                );
            }
        }
    }
    let mut cmps = Vec::new();
    println!("Q1  does progressive information help? (same weights, T3/T4 vs T1)");
    let mut q1 = false;
    for t in [3usize, 4] {
        for m in ["top1", "mass"] {
            let c = compare(&format!("progressive T{t} - T1"), m, (&prog, t), (&prog, 1));
            if m == "top1" {
                q1 |= c.signal;
            }
            cmps.push(c);
        }
    }
    println!("Q3  does the extra integration step help after all information? (T4 vs T3)");
    cmps.push(compare(
        "progressive T4 - T3",
        "top1",
        (&prog, 4),
        (&prog, 3),
    ));
    let mut q2 = None;
    if !all.is_empty() {
        println!(
            "Q2  does iterative integration add value beyond availability? (progressive T4 vs all-info T1)"
        );
        let c = compare(
            "progressive T4 - all-info T1",
            "top1",
            (&prog, 4),
            (&all, 1),
        );
        q2 = Some(c.signal);
        cmps.push(c);
        cmps.push(compare(
            "progressive T4 - all-info T1",
            "mass",
            (&prog, 4),
            (&all, 1),
        ));
        // The reverse direction: does the one-pass model match or beat progressive?
        cmps.push(compare(
            "all-info T1 - progressive T4",
            "top1",
            (&all, 1),
            (&prog, 4),
        ));
    }
    let mut vs_root = false;
    if !root.is_empty() {
        println!("Root-only control (progressive vs root-only at T1)");
        for t in [1usize, 3, 4] {
            let c = compare(
                &format!("progressive T{t} - root-only T1"),
                "top1",
                (&prog, t),
                (&root, 1),
            );
            vs_root |= c.signal;
            cmps.push(c);
        }
    }
    // Planner stability: finite and bounded across T1..T4 (RMS max/min < 4).
    let mut stable = true;
    for f in &prog {
        let rms: Vec<f32> = (1..=4)
            .filter_map(|t| {
                f.v["budgets"][format!("T{t}")]["planner"]
                    .as_array()
                    .and_then(|a| a.last())
                    .and_then(|r| r["rms"].as_f64())
                    .map(|x| x as f32)
            })
            .collect();
        let (lo, hi) = rms
            .iter()
            .fold((f32::MAX, f32::MIN), |a, v| (a.0.min(*v), a.1.max(*v)));
        stable &= rms.iter().all(|v| v.is_finite()) && !rms.is_empty() && hi / lo.max(1e-9) < 4.0;
    }
    // Per-family breakdown of the T4-T1 gain (the PARTIAL GO - PLANNER condition).
    let kinds = prog[0].kinds();
    let (a, b) = (seed_mean(&prog, 4, "top1"), seed_mean(&prog, 1, "top1"));
    let mut fam = BTreeMap::new();
    for k in kinds.iter().collect::<std::collections::BTreeSet<_>>() {
        let (x, y): (Vec<f32>, Vec<f32>) = kinds
            .iter()
            .enumerate()
            .filter(|(_, kk)| *kk == k)
            .map(|(i, _)| (a[i], b[i]))
            .unzip();
        let (m, lo, hi) = paired_bootstrap(&x, &y);
        println!("  family {k:<5} progressive T4-T1 top1 {m:+.4} [{lo:+.4}, {hi:+.4}]");
        fam.insert(k.clone(), (m, lo, hi));
    }
    let outcome = if q1 && q2 == Some(true) && stable {
        "FULL GO"
    } else if q1 {
        "PARTIAL GO - TOOL"
    } else if vs_root && stable {
        "PARTIAL GO - PLANNER"
    } else {
        "NO-GO"
    };
    println!(
        "planner stable across T1..T4: {stable}\nQ1 tool-use signal: {q1}   Q2 iterative signal: {q2:?}   progressive beats root-only: {vs_root}\nOUTCOME: {outcome}"
    );
    if let Some(p) = args.json_out {
        std::fs::write(
            p,
            serde_json::to_vec_pretty(&serde_json::json!({
                "outcome": outcome, "q1_tool_use": q1, "q2_iterative": q2,
                "beats_root_only": vs_root, "planner_stable": stable,
                "comparisons": cmps, "families": fam,
            }))?,
        )?;
    }
    Ok(())
}

// --- info ------------------------------------------------------------------------

#[derive(Args, Debug)]
pub struct InfoArgs {
    #[arg(long)]
    pub config: PathBuf,
}

fn run_info(args: InfoArgs) -> anyhow::Result<()> {
    let cfg = load_cfg(&args.config)?;
    let device = Default::default();
    let model = model_io::build_chimera_v2_unverified::<burn::backend::Flex>(
        &cfg.model,
        &cfg.experimental,
        &device,
    )?;
    println!("config   : {}", cfg.name);
    println!(
        "identity : {}",
        serde_json::to_string(&cfg.experimental.identity()?)?
    );
    let mut total = 0usize;
    for (name, n) in model.param_breakdown() {
        println!("  {name:<28} {n:>12}");
        total += n;
    }
    println!(
        "  {:<28} {:>12}  ({:.1} MB fp32)",
        "TOTAL",
        model.num_params(),
        model.num_params() as f64 * 4.0 / 1e6
    );
    anyhow::ensure!(
        total == model.num_params(),
        "subsystem breakdown does not sum to the total"
    );
    // Per-budget compute counts from a real forward on CPU over a few positions.
    let v = &cfg.experimental.v2;
    let states: Vec<GameState> = recur64_runtime::x15_inputs::probe_positions(4, 30)
        .into_iter()
        .filter(|s| !s.is_terminal() && s.legal_actions().len() <= v.w_cap)
        .take(2)
        .collect();
    let prov = provider(&cfg)?;
    for t in 1..=4 {
        let h = horizon_for_budget(v.info_schedule, t).unwrap_or(WorldHorizon::Root);
        let b = build_v2_batch::<burn::backend::Flex>(
            &states,
            &cfg.experimental,
            prov.as_deref(),
            None,
            h,
            &device,
        )?;
        let out = model.forward(&b.input, &b.cands, t, V2Options::default());
        println!(
            "  T={t}: horizon {}  board_encoder_runs {}  planner_steps {}  successor_encodes {}  reply_set_encodes {}  world: {} state expansions, {} successor records, {} reply records",
            h.label(),
            out.counts.board_encoder_runs,
            out.counts.planner_steps,
            out.counts.successor_encodes,
            out.counts.reply_set_encodes,
            b.phases.world_states,
            b.phases.successors,
            b.phases.replies
        );
    }
    Ok(())
}

// --- bench -----------------------------------------------------------------------

#[derive(Args, Debug)]
pub struct BenchArgs {
    #[arg(long)]
    pub config: PathBuf,
    #[arg(long)]
    pub data: PathBuf,
    #[arg(long, default_value_t = 256)]
    pub positions: usize,
    #[arg(long, default_value_t = 32)]
    pub wasm_positions: usize,
    #[arg(long, default_value_t = 32)]
    pub batch: usize,
    #[arg(long, default_value_t = 6)]
    pub warmup: usize,
    #[arg(long, default_value_t = 12)]
    pub iters: usize,
    #[arg(long)]
    pub json_out: Option<PathBuf>,
}

fn bench_generic<B: Backend>(cfg: &ProbeConfig, args: &BenchArgs) -> anyhow::Result<()> {
    let v = &cfg.experimental.v2;
    let data = V2Data::load(&args.data)?;
    data.audit_and_enforce(v.w_cap, v.r_cap)?;
    let states: Vec<GameState> = data.states()?.into_iter().take(args.positions).collect();
    let mut report = serde_json::Map::new();
    // World model per horizon: native wall, WASM wall (subset), counters.
    let native = NativeWorldModel;
    let wasm = WasmWorldModel::new()?;
    let mut horizons = Vec::new();
    for h in WorldHorizon::ALL {
        let t0 = Instant::now();
        let out = compute_world_bytes(&states, &native, v.w_cap, v.r_cap, h)?;
        let native_us = t0.elapsed().as_micros() as f64;
        let sub = &states[..args.wasm_positions.min(states.len())];
        let t1 = Instant::now();
        let out_w = compute_world_bytes(sub, &wasm, v.w_cap, v.r_cap, h)?;
        let wasm_us = t1.elapsed().as_micros() as f64;
        anyhow::ensure!(
            out_w.iter().zip(&out).all(|(a, b)| a == b),
            "native != WASM at {}",
            h.label()
        );
        let mut st = WorldStatsJson::default();
        for o in &out {
            st.add(&WorldStats::from_bytes(o));
        }
        let per = states.len() as f64;
        println!(
            "world {:<9}: native {:.1} us/pos ({} pos), WASM {:.1} us/pos ({} pos, byte-identical); root moves applied {}, reply moves applied {}, next moves enumerated {}, records S {} R {}, payload {} B/pos",
            h.label(),
            native_us / per,
            states.len(),
            wasm_us / sub.len().max(1) as f64,
            sub.len(),
            st.root_moves_applied / states.len() as u64,
            st.reply_moves_applied / states.len() as u64,
            st.next_moves_enumerated / states.len() as u64,
            st.successor_records / states.len() as u64,
            st.reply_records / states.len() as u64,
            recur64_coproc::world::horizon_payload_bytes(v.w_cap, v.r_cap, h)
        );
        horizons.push(serde_json::json!({
            "horizon": h.label(), "native_us_per_position": native_us / per,
            "wasm_us_per_position": wasm_us / sub.len().max(1) as f64,
            "stats_total": st, "positions": states.len(),
            "payload_bytes": recur64_coproc::world::horizon_payload_bytes(v.w_cap, v.r_cap, h),
        }));
    }
    report.insert("world".into(), serde_json::Value::Array(horizons));
    // GPU forward per budget (steady state, cold start separated).
    let device: B::Device = Default::default();
    let model = model_io::build_chimera_v2::<B>(&cfg.model, &cfg.experimental, &device)?;
    let prov = provider(cfg)?;
    let sub: Vec<GameState> = states.iter().take(args.batch).cloned().collect();
    let mut fwd = Vec::new();
    let (res, gpu) = monitor(true, || -> anyhow::Result<()> {
        for t in 1..=4usize {
            let h = horizon_for_budget(v.info_schedule, t).unwrap_or(WorldHorizon::Root);
            let cold = Instant::now();
            let mut first = true;
            let mut walls = Vec::new();
            let mut tool = Vec::new();
            for i in 0..(args.warmup + args.iters) {
                let b = build_v2_batch::<B>(
                    &sub,
                    &cfg.experimental,
                    prov.as_deref(),
                    None,
                    h,
                    &device,
                )?;
                let t0 = Instant::now();
                let out = model.forward(&b.input, &b.cands, t, V2Options::default());
                let _ = out
                    .readouts
                    .last()
                    .expect("readout")
                    .wdl_logits
                    .clone()
                    .into_data();
                let w = t0.elapsed().as_secs_f64() * 1e3;
                if first {
                    println!(
                        "  T={t}: cold first batch {:.0} ms (JIT/autotune included)",
                        cold.elapsed().as_secs_f64() * 1e3
                    );
                    first = false;
                }
                if i >= args.warmup {
                    walls.push(w);
                    tool.push(b.phases.world_us as f64 / 1e3);
                }
            }
            let m = walls.iter().sum::<f64>() / walls.len() as f64;
            let tm = tool.iter().sum::<f64>() / tool.len() as f64;
            println!(
                "  T={t}: GPU forward {m:.2} ms/batch of {} (steady), tool CPU {tm:.2} ms/batch, end-to-end {:.2} ms",
                sub.len(),
                m + tm
            );
            fwd.push(serde_json::json!({"t": t, "forward_ms": m, "tool_cpu_ms": tm, "end_to_end_ms": m + tm, "batch": sub.len()}));
        }
        Ok(())
    });
    res?;
    println!(
        "gpu: peak_vram={:?} MiB util_busy_mean={:?}",
        gpu.peak_vram_mb, gpu.util_busy_mean
    );
    report.insert("forward".into(), serde_json::Value::Array(fwd));
    report.insert("peak_vram_mb".into(), serde_json::json!(gpu.peak_vram_mb));
    if let Some(p) = &args.json_out {
        std::fs::write(
            p,
            serde_json::to_vec_pretty(&serde_json::Value::Object(report))?,
        )?;
    }
    Ok(())
}

fn run_bench(args: BenchArgs) -> anyhow::Result<()> {
    let cfg = load_cfg(&args.config)?;
    match cfg.device {
        recur64_model::config::DeviceKind::Cpu => bench_generic::<burn::backend::Flex>(&cfg, &args),
        recur64_model::config::DeviceKind::Cuda => {
            #[cfg(feature = "cuda")]
            {
                bench_generic::<burn::backend::Cuda>(&cfg, &args)
            }
            #[cfg(not(feature = "cuda"))]
            {
                anyhow::bail!("CUDA support is not compiled; rebuild with --features cuda")
            }
        }
    }
}
