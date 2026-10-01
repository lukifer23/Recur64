//! `recur64 v3-qual`: engineering qualification of `active_search_v3` on the
//! requested device.
//!
//! Measured, never assumed:
//! * the device known-answer guard;
//! * finite forward at every budget with the Gate VI invariants and the
//!   in-function execution counters;
//! * cold first call (JIT, autotune) separated from steady state;
//! * the compute split: CPU exact queries, GPU root, GPU queried-state encoder,
//!   selector and planner, with the device synchronised before every clock read;
//! * finite backward and per-parameter gradient coverage on update one, plus a
//!   real optimizer update;
//! * a build/forward/drop lifecycle for VRAM, and GPU telemetry sampled through
//!   `nvidia-smi`.
//!
//! Every measured section runs the real graph with the LIVE state-query tool. A
//! device that is requested but unavailable is an error (no CPU substitution). A
//! section that fails is recorded as failed and stays in the report. Nothing here
//! is a science result.

use std::panic::AssertUnwindSafe;
use std::path::PathBuf;
use std::time::Instant;

use burn::optim::{GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use clap::Args;

use recur64_core::GameState;
use recur64_model::active::counters;
use recur64_model::active::coverage::{gradient_coverage, is_stop_head};
use recur64_model::active::loss::selector_loss;
use recur64_model::active::{
    ActiveSearchModel, EdgeRef, QueryScript, RunOptions, ScriptStep, Selection, Tree,
};
use recur64_model::config::{Architecture, DeviceKind, Precision, ProbeConfig};
use recur64_model::loss::{policy_ce, wdl_ce};
use recur64_model::net::NeuralModel;
use recur64_model::train::adamw;
use recur64_runtime::gpu_telemetry::{monitor, sample_gpu};
use recur64_runtime::model_io;

use crate::v3_verdict;

/// Stack of the qualification thread (see `run`).
const QUAL_STACK_BYTES: usize = 512 * 1024 * 1024;

#[derive(Args, Debug, Clone)]
pub struct V3QualArgs {
    /// ProbeConfig TOML (active_search_v3 geometry + device).
    #[arg(long)]
    pub config: PathBuf,
    /// Output directory for the JSON report.
    #[arg(long)]
    pub output: PathBuf,
    /// Deterministic heavy-family position pool size.
    #[arg(long, default_value_t = 64)]
    pub positions: usize,
    /// Query budgets to qualify (comma-separated).
    #[arg(long, default_value = "0,2,4,8,16")]
    pub budgets: String,
    /// Inference batch sizes (comma-separated).
    #[arg(long, default_value = "1,8,16")]
    pub batches: String,
    /// Timed repetitions per (batch, budget) after one cold and one warm call.
    #[arg(long, default_value_t = 5)]
    pub reps: usize,
    /// Training budgets for the backward/coverage section (comma-separated).
    #[arg(long, default_value = "2,4,8")]
    pub train_budgets: String,
    /// Training batch size.
    #[arg(long, default_value_t = 8)]
    pub train_batch: usize,
    /// Timed training updates per budget (after one warmup update).
    #[arg(long, default_value_t = 3)]
    pub train_updates: usize,
    /// Build/forward/drop lifecycle repetitions.
    #[arg(long, default_value_t = 6)]
    pub lifecycle_reps: usize,
    /// Allow budgets above the V3.0 scientific maximum (16) up to the engineering
    /// ceiling. The report is then marked engineering only.
    #[arg(long, default_value_t = false)]
    pub engineering_stress: bool,
    /// Consecutive inference calls on one resident model for the VRAM-plateau test.
    #[arg(long, default_value_t = 150)]
    pub resident_calls: usize,
    /// Consecutive training updates on one resident model for the VRAM-plateau test.
    #[arg(long, default_value_t = 40)]
    pub resident_updates: usize,
    /// Seconds of back-to-back B8 inference used to measure sustained GPU
    /// utilization (0 skips it).
    #[arg(long, default_value_t = 8)]
    pub sustained_seconds: u64,
    #[arg(long, default_value_t = 20250930)]
    pub seed: u64,
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Deterministic KQRvK positions (white to move, black not in check): the
/// heavy stress family the model is built for. Only used for timing and
/// engineering checks, never for labels.
fn heavy_pool(n: usize, seed: u64) -> Vec<GameState> {
    let mut rng = Rng(seed);
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let mut sq = [0usize; 4];
        let mut ok = true;
        for i in 0..4 {
            sq[i] = (rng.next() % 64) as usize;
            if sq[..i].contains(&sq[i]) {
                ok = false;
            }
        }
        if !ok {
            continue;
        }
        let mut board = ['1'; 64];
        // index = rank * 8 + file with rank 0 = rank 1.
        for (s, c) in sq.iter().zip(['K', 'Q', 'R', 'k']) {
            board[*s] = c;
        }
        let mut fen = String::new();
        for rank in (0..8).rev() {
            let mut empty = 0;
            for file in 0..8 {
                match board[rank * 8 + file] {
                    '1' => empty += 1,
                    c => {
                        if empty > 0 {
                            fen.push_str(&empty.to_string());
                            empty = 0;
                        }
                        fen.push(c);
                    }
                }
            }
            if empty > 0 {
                fen.push_str(&empty.to_string());
            }
            if rank > 0 {
                fen.push('/');
            }
        }
        fen.push_str(" w - - 0 1");
        // `from_fen` accepts adjacent kings and an attacked opponent king, which
        // are not legal positions; require that no legal move captures the black
        // king.
        if let Ok(g) = GameState::from_fen(&fen)
            && !g.is_terminal()
            && g.legal_actions().len() >= 16
            && g.legal_standard_moves()
                .iter()
                .all(|m| m.to as usize != sq[3])
        {
            out.push(g);
        }
    }
    out
}

fn parse_list(s: &str) -> anyhow::Result<Vec<usize>> {
    if s.trim().is_empty() {
        return Ok(Vec::new());
    }
    s.split(',')
        .map(|x| {
            x.trim()
                .parse::<usize>()
                .map_err(|e| anyhow::anyhow!("bad list entry {x:?}: {e}"))
        })
        .collect()
}

fn stats(v: &[f64]) -> serde_json::Value {
    if v.is_empty() {
        return serde_json::Value::Null;
    }
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let mean = s.iter().sum::<f64>() / s.len() as f64;
    let q = |p: f64| s[((s.len() - 1) as f64 * p).round() as usize];
    serde_json::json!({"n": s.len(), "mean": mean, "p50": q(0.5), "p95": q(0.95), "min": s[0], "max": s[s.len() - 1]})
}

/// Engineering schedule for the backward section only: follow the first frontier
/// edge and supervise the first two. It is a test double for the QueryScript
/// interface, NOT a teacher; ProofTraceV1 arrives in P4.
struct EngineeringSchedule;
impl QueryScript for EngineeringSchedule {
    fn next(
        &mut self,
        _example: usize,
        _step: usize,
        frontier: &[EdgeRef],
        _tree: &Tree,
    ) -> anyhow::Result<ScriptStep> {
        Ok(ScriptStep {
            follow: 0,
            targets: (0..frontier.len().min(2)).collect(),
        })
    }
}

fn inference_section<B: Backend>(
    cfg: &ProbeConfig,
    args: &V3QualArgs,
    pool: &[GameState],
    device: &B::Device,
    gpu: bool,
) -> anyhow::Result<serde_json::Value> {
    let model = model_io::build_as::<B, ActiveSearchModel<B>>(&cfg.model, device)?;
    let budgets = parse_list(&args.budgets)?;
    let batches = parse_list(&args.batches)?;
    let mut rows = Vec::new();
    for &bs in &batches {
        anyhow::ensure!(
            bs <= pool.len(),
            "batch {bs} exceeds the pool {}",
            pool.len()
        );
        let states = &pool[..bs];
        let mut b0_total = None;
        for &budget in &budgets {
            let mut opts = RunOptions::forced(budget);
            opts.timing_sync = true;
            opts.engineering_stress = args.engineering_stress;
            let t = Instant::now();
            let cold = model.run(states, &opts, Selection::Active, device);
            let cold_s = t.elapsed().as_secs_f64();
            let row = match cold {
                Err(e) => {
                    serde_json::json!({"batch": bs, "budget": budget, "ok": false, "error": format!("{e:#}")})
                }
                Ok(_) => {
                    let _ = model.run(states, &opts, Selection::Active, device)?; // warm
                    let mut walls = Vec::new();
                    let (mut cpu, mut root, mut qenc, mut plan) = (vec![], vec![], vec![], vec![]);
                    let mut last = None;
                    let mut finite = true;
                    let mut inv = Ok(());
                    let mut measured_ok = true;
                    let (_, samples) = monitor(gpu, || {
                        for _ in 0..args.reps {
                            let before = counters::snapshot();
                            let out = match model.run(states, &opts, Selection::Active, device) {
                                Ok(o) => o,
                                Err(_) => {
                                    measured_ok = false;
                                    return;
                                }
                            };
                            let ran = counters::snapshot().since(before);
                            let a = &out.accounting;
                            if ran.root_stage != 1
                                || ran.query_encoder != budget
                                || ran.planner_update != budget
                            {
                                inv = Err(format!("measured executions {ran:?} at B{budget}"));
                            }
                            if let Err(e) = a.check_invariants() {
                                inv = Err(e);
                            }
                            let lp: Vec<f32> = out
                                .readout
                                .policy
                                .log_probs
                                .clone()
                                .into_data()
                                .to_vec()
                                .unwrap_or_default();
                            finite &= !lp.is_empty() && lp.iter().all(|v| v.is_finite());
                            walls.push(a.total_s);
                            cpu.push(a.cpu_query_s);
                            root.push(a.root_encoder_s);
                            qenc.push(a.query_encoder_s);
                            plan.push(a.planner_selector_s);
                            last = Some(out);
                        }
                    });
                    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
                    let total_mean = mean(&walls);
                    if budget == 0 {
                        b0_total = Some(total_mean);
                    }
                    // Same runs without the per-step health checks (each adds host
                    // synchronisations), so their cost is visible and separable.
                    let mut lean = opts.clone();
                    lean.health_checks = false;
                    let mut lean_walls = Vec::new();
                    let mut lean_plan = Vec::new();
                    for _ in 0..args.reps {
                        if let Ok(o) = model.run(states, &lean, Selection::Active, device) {
                            lean_walls.push(o.accounting.total_s);
                            lean_plan.push(o.accounting.planner_selector_s);
                        }
                    }
                    // The frozen non-learned comparator on the same checkpoint and budget.
                    let mut fixed_walls = Vec::new();
                    let mut fixed_ok = true;
                    let mut fixed_finite = true;
                    for _ in 0..args.reps {
                        match model.run(states, &lean, Selection::Fixed, device) {
                            Ok(o) => {
                                let lp: Vec<f32> = o
                                    .readout
                                    .policy
                                    .log_probs
                                    .clone()
                                    .into_data()
                                    .to_vec()
                                    .unwrap_or_default();
                                fixed_finite &= !lp.is_empty() && lp.iter().all(|v| v.is_finite());
                                fixed_walls.push(o.accounting.total_s);
                            }
                            Err(_) => fixed_ok = false,
                        }
                    }
                    let acct = last
                        .as_ref()
                        .map(|o| serde_json::to_value(&o.accounting).unwrap());
                    let facts_s = last.as_ref().map_or(0.0, |o| o.accounting.root_facts_s);
                    let depths: Vec<u32> = last
                        .as_ref()
                        .map(|o| {
                            o.accounting
                                .query_depths
                                .iter()
                                .flatten()
                                .copied()
                                .collect()
                        })
                        .unwrap_or_default();
                    serde_json::json!({
                        "batch": bs, "budget": budget,
                        "ok": measured_ok && inv.is_ok() && finite,
                        "finite": finite,
                        "invariants": match &inv { Ok(()) => "pass".to_string(), Err(e) => e.clone() },
                        "cold_first_call_s": cold_s,
                        "steady_total_s": stats(&walls),
                        "steady_mean_split_s": {
                            "cpu_exact_queries": mean(&cpu),
                            "gpu_root_encoder": mean(&root),
                            "gpu_query_state_encoder": mean(&qenc),
                            "planner_selector": mean(&plan),
                        },
                        "share_of_total": {
                            "cpu_exact_queries": mean(&cpu) / total_mean.max(1e-12),
                            "gpu_root_encoder": mean(&root) / total_mean.max(1e-12),
                            "gpu_query_state_encoder": mean(&qenc) / total_mean.max(1e-12),
                            "planner_selector": mean(&plan) / total_mean.max(1e-12),
                        },
                        "fixed_selection": {"ok": fixed_ok, "finite": fixed_ok && fixed_finite, "steady_total_without_health_checks_s": stats(&fixed_walls)},
                        "root_candidate_facts_cpu_s": facts_s,
                        "steady_total_without_health_checks_s": stats(&lean_walls),
                        "planner_selector_without_health_checks_s": mean(&lean_plan),
                        "wall_ratio_vs_b0_measured": b0_total.map(|b| total_mean / b.max(1e-12)),
                        "wall_per_example_s": total_mean / bs as f64,
                        "mean_query_depth": if depths.is_empty() { 0.0 } else { depths.iter().map(|&d| f64::from(d)).sum::<f64>() / depths.len() as f64 },
                        "accounting_last_rep": acct,
                        "gpu": samples,
                    })
                }
            };
            rows.push(row);
        }
    }
    Ok(
        serde_json::json!({"rows": rows, "params": model.num_params(), "param_groups": model.param_groups()}),
    )
}

/// Back-to-back B8 inference on one batch for a fixed wall time, with GPU
/// utilization sampled throughout: answers how busy the device is under a
/// sustained load of this workload.
fn sustained_section<B: Backend>(
    cfg: &ProbeConfig,
    args: &V3QualArgs,
    pool: &[GameState],
    device: &B::Device,
    gpu: bool,
) -> anyhow::Result<serde_json::Value> {
    if args.sustained_seconds == 0 {
        return Ok(serde_json::Value::Null);
    }
    let model = model_io::build_as::<B, ActiveSearchModel<B>>(&cfg.model, device)?;
    let bs = 16.min(pool.len());
    let states = &pool[..bs];
    let mut opts = RunOptions::forced(8);
    opts.health_checks = false;
    let _ = model.run(states, &opts, Selection::Active, device)?; // warm
    let deadline = Instant::now() + std::time::Duration::from_secs(args.sustained_seconds);
    let mut calls = 0usize;
    let mut cpu = 0.0;
    let mut total = 0.0;
    let (_, samples) = monitor(gpu, || {
        while Instant::now() < deadline {
            if let Ok(o) = model.run(states, &opts, Selection::Active, device) {
                cpu += o.accounting.cpu_query_s;
                total += o.accounting.total_s;
                calls += 1;
            }
        }
    });
    Ok(serde_json::json!({
        "budget": 8, "batch": bs, "seconds": args.sustained_seconds,
        "calls": calls,
        "positions_per_second": (calls * bs) as f64 / args.sustained_seconds as f64,
        "cpu_exact_query_share": cpu / total.max(1e-12),
        "gpu": samples,
        "note": "nvidia-smi samples every 500 ms; utilization is the fraction of the sample window in which any kernel ran, so it overstates how full the device is.",
    }))
}

fn training_section<B: AutodiffBackend>(
    cfg: &ProbeConfig,
    args: &V3QualArgs,
    pool: &[GameState],
    device: &B::Device,
    gpu: bool,
) -> anyhow::Result<Vec<serde_json::Value>> {
    let mut out_rows = Vec::new();
    for budget in parse_list(&args.train_budgets)? {
        let res =
            std::panic::catch_unwind(AssertUnwindSafe(|| -> anyhow::Result<serde_json::Value> {
                let states = &pool[..args.train_batch.min(pool.len())];
                let mut model = model_io::build_as::<B, ActiveSearchModel<B>>(&cfg.model, device)?;
                let mut optim = adamw::<B, _>();
                let opts = RunOptions {
                    health_checks: false,
                    timing_sync: true,
                    engineering_stress: args.engineering_stress,
                    ..RunOptions::forced(budget)
                };
                let mut losses = Vec::new();
                let mut coverage_json = serde_json::Value::Null;
                let mut step_times = Vec::new();
                let mut finite = true;
                let (_, samples) = monitor(gpu, || {
                    for u in 0..(1 + args.train_updates) {
                        let t = Instant::now();
                        let mut sched = EngineeringSchedule;
                        let Ok(run) =
                            model.run(states, &opts, Selection::Script(&mut sched), device)
                        else {
                            finite = false;
                            return;
                        };
                        let [b, w] = run.readout.policy.mask.dims();
                        let mut tgt = vec![0.0f32; b * w];
                        for i in 0..b {
                            tgt[i * w] = 0.5;
                            tgt[i * w + 1] = 0.5;
                        }
                        let target = Tensor::<B, 2>::from_data(
                            burn::tensor::TensorData::new(tgt, [b, w]),
                            device,
                        );
                        let sel = selector_loss(&run.selector_steps);
                        // Policy term plus a neutral WDL term (uniform draw class) so the
                        // coverage check also covers the WDL head, which V3.0 keeps
                        // compatible but does not train for.
                        let wdl_target = Tensor::<B, 1, Int>::from_data(
                            burn::tensor::TensorData::new(vec![1i32; b], [b]),
                            device,
                        );
                        let mut loss = policy_ce(&run.readout.policy, &target)
                            + wdl_ce(&run.readout.wdl_logits, &wdl_target);
                        if let Some(s) = sel {
                            loss = loss + s;
                        }
                        let l = loss.clone().into_data().to_vec::<f32>().unwrap_or_default();
                        finite &= l.first().is_some_and(|v| v.is_finite());
                        losses.push(l.first().copied().unwrap_or(f32::NAN));
                        let grads = GradientsParams::from_grads(loss.backward(), &model);
                        if u == 0 {
                            let rows = gradient_coverage::<B, _>(&model, &grads);
                            let bad: Vec<_> = rows
                                .iter()
                                .filter(|r| {
                                    !is_stop_head(&r.name) && !(r.has_grad && r.finite && r.nonzero)
                                })
                                .map(|r| r.name.clone())
                                .collect();
                            let stop_nonzero =
                                rows.iter().any(|r| is_stop_head(&r.name) && r.nonzero);
                            let nonfinite = rows.iter().any(|r| !r.finite);
                            finite &= !nonfinite;
                            coverage_json = serde_json::json!({
                                "parameter_tensors": rows.len(),
                                "without_finite_nonzero_gradient_excluding_stop": bad,
                                "stop_head_gradient_nonzero": stop_nonzero,
                                "any_nonfinite_gradient": nonfinite,
                            });
                        }
                        model = optim.step(1e-4, model, grads);
                        let _ = B::sync(device);
                        if u > 0 {
                            step_times.push(t.elapsed().as_secs_f64());
                        }
                    }
                });
                Ok(serde_json::json!({
                    "budget": budget, "batch": states.len(), "ok": finite, "update_ok": finite,
                    "finite": finite,
                    "losses": losses,
                    "gradient_coverage_update_one": coverage_json,
                    "sec_per_update": stats(&step_times),
                    "gpu": samples,
                }))
            }));
        out_rows.push(match res {
            Ok(Ok(v)) => v,
            Ok(Err(e)) => serde_json::json!({"budget": budget, "ok": false, "error": format!("{e:#}")}),
            Err(_) => serde_json::json!({"budget": budget, "ok": false, "error": "panicked (for example CUDA out of memory)"}),
        });
    }
    Ok(out_rows)
}

/// One real teacher-forced update with the engineering schedule.
fn teacher_step<B: AutodiffBackend, O: Optimizer<ActiveSearchModel<B>, B>>(
    model: ActiveSearchModel<B>,
    optim: &mut O,
    states: &[GameState],
    budget: usize,
    device: &B::Device,
) -> anyhow::Result<(ActiveSearchModel<B>, f32)> {
    let opts = RunOptions {
        health_checks: false,
        ..RunOptions::forced(budget)
    };
    let mut sched = EngineeringSchedule;
    let run = model.run(states, &opts, Selection::Script(&mut sched), device)?;
    let [b, w] = run.readout.policy.mask.dims();
    let mut tgt = vec![0.0f32; b * w];
    for i in 0..b {
        tgt[i * w] = 0.5;
        tgt[i * w + 1] = 0.5;
    }
    let target = Tensor::<B, 2>::from_data(burn::tensor::TensorData::new(tgt, [b, w]), device);
    let mut loss = policy_ce(&run.readout.policy, &target);
    if let Some(sel) = selector_loss(&run.selector_steps) {
        loss = loss + sel;
    }
    let l = loss.clone().into_data().to_vec::<f32>().unwrap_or_default();
    let grads = GradientsParams::from_grads(loss.backward(), &model);
    Ok((
        optim.step(1e-4, model, grads),
        l.first().copied().unwrap_or(f32::NAN),
    ))
}

/// Save a real training checkpoint on the device, load it into a fresh template,
/// and prove the restored model computes the same policy as the saved one.
fn checkpoint_section<B, I>(
    cfg: &ProbeConfig,
    args: &V3QualArgs,
    pool: &[GameState],
    device: &B::Device,
    inner: &I::Device,
) -> anyhow::Result<serde_json::Value>
where
    B: AutodiffBackend<InnerBackend = I>,
    I: Backend,
{
    use burn::module::AutodiffModule;
    use recur64_model::checkpoint::{CheckpointMeta, load_training, save_training};
    let states = &pool[..4.min(pool.len())];
    let template = model_io::build_as::<B, ActiveSearchModel<B>>(&cfg.model, device)?;
    let mut optim = adamw::<B, _>();
    let (model, loss) = teacher_step(template.clone(), &mut optim, states, 4, device)?;
    let dir = std::env::temp_dir().join("recur64_v3_qual_ckpt");
    let _ = std::fs::remove_dir_all(&dir);
    let meta = CheckpointMeta::new(
        cfg.model.clone(),
        1,
        false,
        1,
        1e-4,
        args.seed,
        0,
        "v3-qual",
        "fp32",
    );
    save_training::<B, _, _>(&dir, &model, &optim, &meta)?;
    let (loaded, _o, m) = load_training::<B, _, _>(&dir, template, adamw::<B, _>(), device)?;
    let opts = RunOptions::forced(4);
    let a = model
        .valid()
        .run(states, &opts, Selection::Fixed, inner)?
        .readout
        .policy
        .log_probs
        .into_data()
        .to_vec::<f32>()
        .unwrap_or_default();
    let b = loaded
        .valid()
        .run(states, &opts, Selection::Fixed, inner)?
        .readout
        .policy
        .log_probs
        .into_data()
        .to_vec::<f32>()
        .unwrap_or_default();
    let max_diff = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    let _ = std::fs::remove_dir_all(&dir);
    Ok(serde_json::json!({
        "update_loss": loss,
        "loaded_architecture": m.architecture,
        "loaded_model_id_present": !m.model_id.is_empty(),
        "policy_outputs_compared": a.len(),
        "max_abs_policy_diff_saved_vs_loaded": max_diff,
        "ok": !a.is_empty() && a.len() == b.len() && max_diff <= 1e-6,
        "note": "same device, same kernels; CUDA bit-exactness is reported, not required.",
    }))
}

/// Dynamic queried-state legal widths change tensor shapes round by round. Run the
/// same batch-1 B8 query on many different positions twice: if first-time shapes
/// trigger JIT/autotune stalls, the first pass is much slower than the second.
fn dynamic_width_section<B: Backend>(
    cfg: &ProbeConfig,
    pool: &[GameState],
    device: &B::Device,
) -> anyhow::Result<serde_json::Value> {
    let model = model_io::build_as::<B, ActiveSearchModel<B>>(&cfg.model, device)?;
    let mut opts = RunOptions::forced(8);
    opts.health_checks = false;
    opts.timing_sync = true;
    let n = 48.min(pool.len());
    let mut first = Vec::new();
    let mut second = Vec::new();
    let mut widths = std::collections::BTreeSet::new();
    let mut shapes = std::collections::BTreeSet::new();
    let _ = model.run(&pool[..1], &opts, Selection::Active, device)?; // global warmup
    for pass in 0..2 {
        for g in &pool[..n] {
            let out = model.run(std::slice::from_ref(g), &opts, Selection::Active, device)?;
            if pass == 0 {
                first.push(out.accounting.total_s);
                shapes.insert(out.accounting.query_action_widths.clone());
                widths.extend(out.accounting.query_action_widths.iter().copied());
            } else {
                second.push(out.accounting.total_s);
            }
        }
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    let first_stats = stats(&first);
    let second_stats = stats(&second);
    Ok(serde_json::json!({
        "positions": n,
        "distinct_query_action_widths": widths.len(),
        "query_action_widths_seen": widths,
        "distinct_width_sequences": shapes.len(),
        "first_pass_s": first_stats,
        "second_pass_s": second_stats,
        "first_over_second_mean": mean(&first) / mean(&second).max(1e-12),
        "first_max_over_second_p50": first.iter().cloned().fold(0.0, f64::max)
            / second_stats["p50"].as_f64().unwrap_or(1.0).max(1e-12),
        "interpretation": "a first-pass mean near the second-pass mean and a modest max/p50 ratio mean dynamic widths do not cause pathological JIT or scheduling; large ratios would be reported as a measured problem before any bucketing change",
    }))
}

/// Slope (MiB per sample) from the second sample to the last, and the verdict.
fn plateau(samples: &[Option<u64>]) -> serde_json::Value {
    let known: Vec<u64> = samples.iter().flatten().copied().collect();
    let slope = (known.len() >= 3)
        .then(|| (*known.last().unwrap() as f64 - known[1] as f64) / (known.len() - 2) as f64);
    let growth = (known.len() >= 3).then(|| *known.last().unwrap() as i64 - known[1] as i64);
    serde_json::json!({
        "vram_samples_mb": samples,
        "growth_from_second_sample_mb": growth,
        "slope_mb_per_sample": slope,
        // One allocator page is 32 MiB; growth beyond a couple of pages is a trend.
        "plateau": growth.is_some_and(|g| g <= 64),
    })
}

/// One resident model, many consecutive calls: the VRAM property that matters for a
/// science run (a run builds its model once). Inference and training are separate.
fn resident_section<TB, I>(
    cfg: &ProbeConfig,
    args: &V3QualArgs,
    pool: &[GameState],
    device: &TB::Device,
    inner: &I::Device,
) -> anyhow::Result<serde_json::Value>
where
    TB: AutodiffBackend<InnerBackend = I>,
    I: Backend,
{
    let every = 10usize;
    // Inference: B8 on a batch of 16, health checks on (the science default).
    let model = model_io::build_as::<I, ActiveSearchModel<I>>(&cfg.model, inner)?;
    let states = &pool[..16.min(pool.len())];
    let opts = RunOptions::forced(8);
    let _ = model.run(states, &opts, Selection::Active, inner)?;
    let mut inf = vec![{
        let _ = I::sync(inner);
        sample_gpu().map(|s| s.0)
    }];
    for i in 1..=args.resident_calls {
        let _ = model.run(states, &opts, Selection::Active, inner)?;
        if i % every == 0 {
            let _ = I::sync(inner);
            inf.push(sample_gpu().map(|s| s.0));
        }
    }
    drop(model);
    I::memory_cleanup(inner);
    // Training: B4 on a batch of 8, real optimizer updates.
    let mut tmodel = model_io::build_as::<TB, ActiveSearchModel<TB>>(&cfg.model, device)?;
    let mut optim = adamw::<TB, _>();
    let tstates = &pool[..8.min(pool.len())];
    (tmodel, _) = teacher_step(tmodel, &mut optim, tstates, 4, device)?;
    let mut train = vec![{
        let _ = TB::sync(device);
        sample_gpu().map(|s| s.0)
    }];
    for i in 1..=args.resident_updates {
        (tmodel, _) = teacher_step(tmodel, &mut optim, tstates, 4, device)?;
        if i % every == 0 {
            let _ = TB::sync(device);
            train.push(sample_gpu().map(|s| s.0));
        }
    }
    Ok(serde_json::json!({
        "sample_every_calls": every,
        "inference_b8_batch16": {"calls": args.resident_calls, "vram": plateau(&inf)},
        "training_b4_batch8": {"updates": args.resident_updates, "vram": plateau(&train)},
        "rule": "plateau iff VRAM grows by at most 64 MiB (two allocator pages) from the second sample to the last",
    }))
}

/// VRAM across repeated build / (forward) / drop cycles, split by what each cycle
/// does and by whether the allocator pool is explicitly released afterwards, so
/// growth can be attributed. nvidia-smi reports process memory; CubeCL keeps freed
/// pages in a per-thread pool until `memory_cleanup` (D44), so the slope (MiB per
/// cycle) over many cycles is the evidence, not one reading.
fn lifecycle_section<B: Backend>(
    cfg: &ProbeConfig,
    args: &V3QualArgs,
    pool: &[GameState],
    device: &B::Device,
) -> anyhow::Result<serde_json::Value> {
    let states = &pool[..8.min(pool.len())];
    let baseline = sample_gpu().map(|s| s.0);
    let mut modes = serde_json::Map::new();
    for (name, budget) in [
        ("build_then_drop", None),
        ("build_b0_then_drop", Some(0usize)),
        ("build_b8_then_drop", Some(8usize)),
    ] {
        for cleanup in [false, true] {
            let mut vram = Vec::new();
            for _ in 0..args.lifecycle_reps {
                {
                    let m = model_io::build_as::<B, ActiveSearchModel<B>>(&cfg.model, device)?;
                    if let Some(b) = budget {
                        let _ = m.run(states, &RunOptions::forced(b), Selection::Active, device)?;
                    }
                    let _ = B::sync(device);
                }
                if cleanup {
                    B::memory_cleanup(device);
                }
                std::thread::sleep(std::time::Duration::from_millis(300));
                vram.push(sample_gpu().map(|s| s.0));
            }
            let known: Vec<u64> = vram.iter().flatten().copied().collect();
            // Slope from the second reading on: the first cycle after a mode change
            // may include a one-off pool release or growth.
            let slope = if known.len() >= 3 {
                Some((*known.last().unwrap() as f64 - known[1] as f64) / (known.len() - 2) as f64)
            } else {
                None
            };
            let tail = if known.len() >= 3 {
                let t = &known[known.len() - 3..];
                Some(t.iter().max().unwrap() - t.iter().min().unwrap())
            } else {
                None
            };
            modes.insert(
                format!("{name}{}", if cleanup { "+memory_cleanup" } else { "" }),
                serde_json::json!({
                    "memory_cleanup_after_drop": cleanup,
                    "vram_after_each_cycle_mb": vram,
                    "slope_mb_per_cycle_from_second_reading": slope,
                    "spread_of_last_three_mb": tail,
                    "plateau": slope.is_some_and(|s| s < 1.0),
                }),
            );
        }
    }
    // Control: the historical V2.5 model built and dropped under the same loop and
    // the same cleanup. If it grows identically, the growth is not V3-specific.
    for cleanup in [true] {
        let mut vram = Vec::new();
        for _ in 0..args.lifecycle_reps {
            {
                let _m = model_io::build_as::<B, recur64_model::candidate::CandidateV25Model<B>>(
                    &recur64_model::config::ModelConfig::candidate_v25(true),
                    device,
                )?;
                let _ = B::sync(device);
            }
            if cleanup {
                B::memory_cleanup(device);
            }
            std::thread::sleep(std::time::Duration::from_millis(300));
            vram.push(sample_gpu().map(|s| s.0));
        }
        let known: Vec<u64> = vram.iter().flatten().copied().collect();
        let slope = (known.len() >= 3)
            .then(|| (*known.last().unwrap() as f64 - known[1] as f64) / (known.len() - 2) as f64);
        modes.insert(
            "control_candidate_v25_build_then_drop+memory_cleanup".to_string(),
            serde_json::json!({
                "memory_cleanup_after_drop": cleanup,
                "vram_after_each_cycle_mb": vram,
                "slope_mb_per_cycle_from_second_reading": slope,
                "plateau": slope.is_some_and(|s| s < 1.0),
            }),
        );
    }
    Ok(serde_json::json!({
        "baseline_vram_mb": baseline,
        "cycles_per_mode": args.lifecycle_reps,
        "modes": modes,
        "plateau_rule": "slope < 1 MiB per cycle from the second reading to the last",
    }))
}

/// Run one section; an infrastructure error or panic becomes a recorded
/// `section_error` instead of aborting the report.
fn guarded(name: &str, f: impl FnOnce() -> anyhow::Result<serde_json::Value>) -> serde_json::Value {
    match std::panic::catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => serde_json::json!({"section_error": format!("{name}: {e:#}")}),
        Err(_) => serde_json::json!({"section_error": format!("{name}: panicked")}),
    }
}

fn run_device<TB: AutodiffBackend>(
    cfg: &ProbeConfig,
    args: &V3QualArgs,
    gpu: bool,
) -> anyhow::Result<serde_json::Value> {
    let device: TB::Device = Default::default();
    let inner = Default::default();
    // Known-answer guard first. A failure is recorded in the report (and fails the
    // qualification) instead of aborting before anything is written.
    let guard = model_io::verify_device::<TB::InnerBackend>(&inner);
    let pool = heavy_pool(args.positions, args.seed);
    let head = |guard_text: String| {
        serde_json::json!({
            "report_schema": v3_verdict::REPORT_SCHEMA_V2,
            "config": cfg.name,
            "architecture": cfg.model.architecture.id(),
            "device": format!("{:?}", cfg.device),
            "precision": cfg.precision.label(),
            "pool": {"positions": pool.len(), "family": "KQRvK, white to move, >= 16 legal moves", "seed": args.seed},
            "device_known_answer_guard": guard_text,
            "engineering_only": args.engineering_stress,
        })
    };
    let mut report = match guard {
        Ok(()) => {
            let mut r = head("passed".to_string());
            let inference = guarded("inference", || {
                inference_section::<TB::InnerBackend>(cfg, args, &pool, &inner, gpu)
            });
            let training = guarded("training", || {
                Ok(serde_json::Value::Array(training_section::<TB>(
                    cfg, args, &pool, &device, gpu,
                )?))
            });
            let checkpoint = guarded("checkpoint", || {
                checkpoint_section::<TB, TB::InnerBackend>(cfg, args, &pool, &device, &inner)
            });
            let resident = guarded("resident_model_vram", || {
                resident_section::<TB, TB::InnerBackend>(cfg, args, &pool, &device, &inner)
            });
            // Diagnostics: recorded, never gating.
            let sustained = guarded("sustained", || {
                sustained_section::<TB::InnerBackend>(cfg, args, &pool, &inner, gpu)
            });
            let dynamic_widths = guarded("dynamic_query_widths", || {
                dynamic_width_section::<TB::InnerBackend>(cfg, &pool, &inner)
            });
            let lifecycle = guarded("lifecycle", || {
                lifecycle_section::<TB::InnerBackend>(cfg, args, &pool, &inner)
            });
            let o = r.as_object_mut().expect("object");
            o.insert("inference".into(), inference);
            o.insert("training".into(), training);
            o.insert("checkpoint".into(), checkpoint);
            o.insert("resident_model_vram".into(), resident);
            o.insert("sustained_load".into(), sustained);
            o.insert("dynamic_query_widths".into(), dynamic_widths);
            o.insert("lifecycle".into(), lifecycle);
            r
        }
        Err(e) => head(format!("failed: {e:#}")),
    };
    let verdict = v3_verdict::evaluate(&report);
    let o = report.as_object_mut().expect("object");
    o.insert(
        "synchronisation_policy".into(),
        "timing_sync = true: the device is synchronised (Backend::sync) before every section clock read, so GPU section times are completion times; health checks add host reads (reported separately as the without_health_checks columns); the selector result is read to the host every round because the exact query is a CPU operation".into(),
    );
    o.insert(
        "qualification_gates_ok".into(),
        verdict.qualification_gates_ok.into(),
    );
    o.insert(
        "qualification_gate_details".into(),
        serde_json::to_value(&verdict.qualification_gate_details)?,
    );
    o.insert(
        "diagnostics_complete".into(),
        verdict.diagnostics_complete.into(),
    );
    o.insert(
        "diagnostic_findings".into(),
        serde_json::to_value(&verdict.diagnostic_findings)?,
    );
    o.insert(
        "recorded_limitations".into(),
        serde_json::to_value(&verdict.recorded_limitations)?,
    );
    // Deprecated alias, kept so older consumers still parse the report. It has
    // exactly the meaning of `qualification_gates_ok` (the original P3 field did
    // not include every gate; see docs/V3_EXPERIMENTS.md V3-E8).
    o.insert(
        "all_sections_ok".into(),
        verdict.qualification_gates_ok.into(),
    );
    o.insert(
        "all_sections_ok_note".into(),
        "DEPRECATED alias of qualification_gates_ok".into(),
    );
    o.insert(
        "claims".into(),
        serde_json::json!({
            "tested": "the real graph with the live state-query tool ran on the requested device in every gating section",
            "not_claimed": "no science result; B16 is not described as 16x compute (see wall_ratio_vs_b0_measured)",
        }),
    );
    Ok(report)
}

fn args_for_thread(a: &V3QualArgs) -> V3QualArgs {
    a.clone()
}

pub fn run(args: V3QualArgs) -> anyhow::Result<()> {
    let cfg = ProbeConfig::from_toml_str(&std::fs::read_to_string(&args.config)?)?;
    cfg.model.validate()?;
    // Refuse out-of-range budgets before any model or device work.
    {
        use recur64_model::config::{ACTIVE_ENGINEERING_MAX_BUDGET, ACTIVE_MAX_BUDGET};
        let ceiling = if args.engineering_stress {
            ACTIVE_ENGINEERING_MAX_BUDGET
        } else {
            ACTIVE_MAX_BUDGET
        };
        for b in parse_list(&args.budgets)?
            .into_iter()
            .chain(parse_list(&args.train_budgets)?)
        {
            anyhow::ensure!(
                b <= ceiling,
                "budget {b} exceeds the maximum {ceiling} (V3.0 science is B0..B{ACTIVE_MAX_BUDGET};                  larger budgets need --engineering-stress and are engineering only)"
            );
        }
    }
    anyhow::ensure!(
        cfg.model.architecture == Architecture::ActiveSearchV3,
        "v3-qual qualifies active_search_v3 only (config is {})",
        cfg.model.architecture.id()
    );
    anyhow::ensure!(
        cfg.precision == Precision::Fp32,
        "v3-qual is an FP32 qualification (V3.0 science is FP32)"
    );
    // The autodiff graph of a multi-round training step is deep, and dropping it
    // recurses. The default 1 MiB Windows main-thread stack overflowed at B8, so
    // the whole qualification runs on a thread with an explicit large stack.
    let report = std::thread::Builder::new()
        .name("v3-qual".into())
        .stack_size(QUAL_STACK_BYTES)
        .spawn({
            let (cfg, args) = (cfg.clone(), args_for_thread(&args));
            move || -> anyhow::Result<serde_json::Value> {
                match cfg.device {
                    DeviceKind::Cpu => {
                        run_device::<recur64_model::train::CpuTrainBackend>(&cfg, &args, false)
                    }
                    #[cfg(feature = "cuda")]
                    DeviceKind::Cuda => run_device::<burn::backend::Autodiff<burn::backend::Cuda>>(
                        &cfg, &args, true,
                    ),
                    #[cfg(not(feature = "cuda"))]
                    DeviceKind::Cuda => {
                        anyhow::bail!("CUDA support is not compiled; rebuild with --features cuda")
                    }
                }
            }
        })?
        .join()
        .map_err(|_| anyhow::anyhow!("v3-qual thread panicked"))??;
    std::fs::create_dir_all(&args.output)?;
    let path = args.output.join(format!("v3-qual-{}.json", cfg.name));
    std::fs::write(&path, serde_json::to_vec_pretty(&report)?)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    eprintln!("wrote {}", path.display());
    // The process status carries the verdict: a script never has to parse JSON.
    if report["qualification_gates_ok"] != true {
        let failed: Vec<String> = report["qualification_gate_details"]
            .as_array()
            .map(|g| {
                g.iter()
                    .filter(|x| x["ok"] == false)
                    .map(|x| format!("{} ({})", x["name"], x["detail"]))
                    .collect()
            })
            .unwrap_or_default();
        anyhow::bail!(
            "v3-qual: qualification gates FAILED (report written to {}): {}",
            path.display(),
            failed.join("; ")
        );
    }
    Ok(())
}

#[derive(Args, Debug)]
pub struct V3QualVerdictArgs {
    /// An existing v3-qual JSON report.
    #[arg(long)]
    pub report: PathBuf,
    /// Where to write the derived summary.
    #[arg(long)]
    pub output: PathBuf,
}

/// Apply the hardened verdict to an existing report and write a deterministic
/// summary. The source report is never modified.
pub fn run_verdict(args: V3QualVerdictArgs) -> anyhow::Result<()> {
    use sha2::{Digest, Sha256};
    let bytes = std::fs::read(&args.report)?;
    let report: serde_json::Value = serde_json::from_slice(&bytes)?;
    let v = v3_verdict::evaluate(&report);
    let legacy = report["report_schema"] != v3_verdict::REPORT_SCHEMA_V2;
    let summary = serde_json::json!({
        "summary_schema": "p3_qualification_summary_v2",
        "provenance": if legacy {
            "DERIVED FROM EXISTING MEASURED EVIDENCE: the hardened verdict logic applied to a report produced before that logic existed"
        } else {
            "MEASURED: report produced by the hardened harness"
        },
        "source_report": {
            "file": args.report.file_name().map(|n| n.to_string_lossy().into_owned()),
            "sha256": format!("{:x}", Sha256::digest(&bytes)),
            "report_schema": report["report_schema"],
            "original_all_sections_ok": report["all_sections_ok"],
            "config": report["config"],
            "device": report["device"],
            "precision": report["precision"],
        },
        "qualification_gates_ok": v.qualification_gates_ok,
        "qualification_gate_details": v.qualification_gate_details,
        "diagnostics_complete": v.diagnostics_complete,
        "diagnostic_findings": v.diagnostic_findings,
        "recorded_limitations": v.recorded_limitations,
    });
    if let Some(dir) = args.output.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(&args.output, serde_json::to_vec_pretty(&summary)?)?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    anyhow::ensure!(
        v.qualification_gates_ok,
        "hardened verdict: qualification gates FAILED: {:?}",
        v.failed()
    );
    Ok(())
}
