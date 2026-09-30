//! `recur64 v25-qual` — P0.5 / P0.6 qualification for `candidate_v25` (or any
//! architecture config): CandidateFacts CPU cost, forward latency by batch,
//! learner layouts at a fixed effective batch, and a build/forward/drop lifecycle
//! check, with GPU VRAM/utilization sampled through `nvidia-smi`.
//!
//! Everything measured runs the real graph on the requested device. A device
//! that is requested but unavailable is an error (no CPU substitution). A layout
//! that fails (for example CUDA out of memory) is recorded as failed and stays in
//! the report.

use std::panic::AssertUnwindSafe;
use std::path::PathBuf;
use std::time::Instant;

use burn::tensor::backend::AutodiffBackend;
use clap::Args;

use recur64_core::{
    ActionId, CandidateFactsV1, GameState, ObservationV1, StandardMove, candidate_facts,
    encode_observation_v1,
};
use recur64_model::candidate::CandidateInputs;
use recur64_model::config::ProbeConfig;
use recur64_model::loss::Targets;
use recur64_model::net::NeuralModel;
use recur64_model::train::adamw;
use recur64_runtime::accum::{LossMode, MicroBatch, accumulated_update};
use recur64_runtime::gpu_telemetry::{monitor, sample_gpu};
use recur64_runtime::inference::{BatchEvaluator, BatchedModel};
use recur64_runtime::model_io;

#[derive(Args, Debug)]
pub struct V25QualArgs {
    /// ProbeConfig TOML (model geometry + device).
    #[arg(long)]
    pub config: PathBuf,
    /// Output directory for the JSON report.
    #[arg(long)]
    pub output: PathBuf,
    /// Deterministic position pool size.
    #[arg(long, default_value_t = 1024)]
    pub positions: usize,
    /// Forward batch sizes (comma-separated).
    #[arg(long, default_value = "8,16,32,48,64,96")]
    pub forward_batches: String,
    /// Learner layouts PHYSICALxACCUM (same effective batch).
    #[arg(long, default_value = "32x8,64x4,128x2")]
    pub layouts: String,
    /// Timed updates per layout (after 2 warmup updates).
    #[arg(long, default_value_t = 6)]
    pub updates: usize,
    /// Timed forward repetitions per batch (after 3 warmup calls).
    #[arg(long, default_value_t = 10)]
    pub reps: usize,
    /// Build/forward/drop lifecycle repetitions.
    #[arg(long, default_value_t = 4)]
    pub lifecycle_reps: usize,
    /// Peak VRAM ceiling for a layout to be selectable (MiB).
    #[arg(long, default_value_t = 12288)]
    pub vram_limit_mb: u64,
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

/// Deterministic realistic positions: seeded random walks from the start.
fn position_pool(n: usize, seed: u64) -> Vec<GameState> {
    let mut rng = Rng(seed);
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let mut g = GameState::startpos();
        let depth = 4 + (rng.next() % 80) as usize;
        for _ in 0..depth {
            let a = g.legal_actions();
            if a.is_empty() || g.is_terminal() {
                break;
            }
            let id = a[(rng.next() as usize) % a.len()];
            let (f, t, p) = id.to_physical(g.perspective());
            g.apply(StandardMove::new(f, t, (!p.is_none()).then_some(p)))
                .expect("legal move applies");
        }
        if !g.is_terminal() && !g.legal_actions().is_empty() {
            out.push(g);
        }
    }
    out
}

struct Prepared {
    obs: Vec<ObservationV1>,
    legal: Vec<Vec<ActionId>>,
    facts: Vec<Vec<CandidateFactsV1>>,
}

fn pctl(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    sorted[((sorted.len() - 1) as f64 * q).round() as usize]
}

/// P0.5: CandidateFacts CPU cost on the pool, measured on its own.
fn facts_cost(pool: &[GameState]) -> serde_json::Value {
    let mut per_pos_us = Vec::with_capacity(pool.len());
    let mut moves = 0usize;
    let t0 = Instant::now();
    for s in pool {
        let t = Instant::now();
        let f = candidate_facts(s);
        per_pos_us.push(t.elapsed().as_secs_f64() * 1e6);
        moves += f.len();
    }
    let wall = t0.elapsed().as_secs_f64();
    per_pos_us.sort_by(f64::total_cmp);
    serde_json::json!({
        "positions": pool.len(),
        "positions_per_sec": pool.len() as f64 / wall,
        "candidate_moves_per_sec": moves as f64 / wall,
        "mean_legal_width": moves as f64 / pool.len() as f64,
        "per_position_us_p50": pctl(&per_pos_us, 0.5),
        "per_position_us_p95": pctl(&per_pos_us, 0.95),
        "wall_s": wall,
    })
}

fn run_guarded<T>(f: impl FnOnce() -> anyhow::Result<T>) -> Result<T, String> {
    match std::panic::catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(v)) => Ok(v),
        Ok(Err(e)) => Err(format!("{e:#}")),
        Err(p) => Err(format!(
            "panic: {}",
            p.downcast_ref::<String>()
                .cloned()
                .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "unknown".into())
        )),
    }
}

fn parse_list(s: &str) -> anyhow::Result<Vec<usize>> {
    s.split(',')
        .map(|x| x.trim().parse::<usize>().map_err(Into::into))
        .collect()
}

fn parse_layouts(s: &str) -> anyhow::Result<Vec<(usize, usize)>> {
    s.split(',')
        .map(|l| {
            let (p, a) = l
                .trim()
                .split_once('x')
                .ok_or_else(|| anyhow::anyhow!("layout '{l}' is not PHYSICALxACCUM"))?;
            Ok((p.parse()?, a.parse()?))
        })
        .collect()
}

fn forward_section<B, M>(
    cfg: &ProbeConfig,
    args: &V25QualArgs,
    prep: &Prepared,
    device: &B::Device,
) -> anyhow::Result<Vec<serde_json::Value>>
where
    B: burn::prelude::Backend,
    M: NeuralModel<B>,
{
    let gpu = cfg.device == recur64_model::config::DeviceKind::Cuda;
    let model = model_io::build_as::<B, M>(&cfg.model, device)?;
    let needs = model.needs_candidate_facts();
    let batched = BatchedModel::new(model, 1, device.clone());
    let mut rows = Vec::new();
    for size in parse_list(&args.forward_batches)? {
        let pick = |k: usize| -> Vec<usize> {
            (0..size).map(|i| (k * size + i) % prep.obs.len()).collect()
        };
        let call = |k: usize| -> Result<f64, String> {
            let idx = pick(k);
            let obs: Vec<_> = idx.iter().map(|&i| prep.obs[i].clone()).collect();
            let legal: Vec<_> = idx.iter().map(|&i| prep.legal[i].clone()).collect();
            let facts: Vec<_> = idx
                .iter()
                .map(|&i| needs.then(|| prep.facts[i].clone()))
                .collect();
            let t = Instant::now();
            batched
                .evaluate_batch_with_facts(&obs, &legal, &facts)
                .map_err(|e| e.to_string())?;
            Ok(t.elapsed().as_secs_f64() * 1e3)
        };
        let result = run_guarded(|| {
            for k in 0..3 {
                call(k).map_err(anyhow::Error::msg)?;
            }
            let (ms, samples) = monitor(gpu, || {
                (0..args.reps)
                    .map(|k| call(k + 3))
                    .collect::<Result<Vec<f64>, String>>()
            });
            Ok((ms.map_err(anyhow::Error::msg)?, samples))
        });
        rows.push(match result {
            Ok((mut ms, samples)) => {
                let mean = ms.iter().sum::<f64>() / ms.len() as f64;
                ms.sort_by(f64::total_cmp);
                serde_json::json!({
                    "batch": size, "ok": true, "reps": args.reps,
                    "forward_ms_mean": mean, "forward_ms_p50": pctl(&ms, 0.5),
                    "positions_per_sec": size as f64 / (mean / 1e3),
                    "gpu": samples,
                })
            }
            Err(e) => serde_json::json!({ "batch": size, "ok": false, "error": e }),
        });
    }
    Ok(rows)
}

/// P0.5 end-to-end proxy: CPU assembly (observation + facts) vs GPU forward at
/// one batch size, so the facts share of evaluation wall is visible.
fn assembly_vs_forward<B, M>(
    cfg: &ProbeConfig,
    pool: &[GameState],
    device: &B::Device,
    batch: usize,
    reps: usize,
) -> anyhow::Result<serde_json::Value>
where
    B: burn::prelude::Backend,
    M: NeuralModel<B>,
{
    let model = model_io::build_as::<B, M>(&cfg.model, device)?;
    let needs = model.needs_candidate_facts();
    let batched = BatchedModel::new(model, 1, device.clone());
    let (mut obs_ms, mut facts_ms, mut fwd_ms) = (0.0, 0.0, 0.0);
    for k in 0..reps + 2 {
        let states: Vec<&GameState> = (0..batch)
            .map(|i| &pool[(k * batch + i) % pool.len()])
            .collect();
        let t = Instant::now();
        let obs: Vec<_> = states.iter().map(|s| encode_observation_v1(s)).collect();
        let legal: Vec<_> = states.iter().map(|s| s.legal_actions()).collect();
        let o = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        let facts: Vec<Option<Vec<CandidateFactsV1>>> = states
            .iter()
            .map(|s| needs.then(|| candidate_facts(s)))
            .collect();
        let f = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        batched
            .evaluate_batch_with_facts(&obs, &legal, &facts)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let w = t.elapsed().as_secs_f64() * 1e3;
        if k >= 2 {
            obs_ms += o;
            facts_ms += f;
            fwd_ms += w;
        }
    }
    let n = reps as f64;
    let total = obs_ms + facts_ms + fwd_ms;
    Ok(serde_json::json!({
        "batch": batch, "reps": reps,
        "observation_and_legal_ms_mean": obs_ms / n,
        "facts_ms_mean": facts_ms / n,
        "forward_ms_mean": fwd_ms / n,
        "facts_share_of_batch_eval": facts_ms / total,
        "note": "proxy: excludes search overhead; the real self-play share is re-measured in P4 scheduling",
    }))
}

#[allow(clippy::too_many_arguments)]
fn layout_section<B, M>(
    cfg: &ProbeConfig,
    args: &V25QualArgs,
    prep: &Prepared,
    device: &B::Device,
) -> anyhow::Result<Vec<serde_json::Value>>
where
    B: AutodiffBackend,
    M: NeuralModel<B> + burn::module::AutodiffModule<B>,
{
    let gpu = cfg.device == recur64_model::config::DeviceKind::Cuda;
    let needs_facts = {
        let probe = M::build(&cfg.model, device)?;
        probe.needs_candidate_facts()
    };
    let mut rows = Vec::new();
    for (physical, accum) in parse_layouts(&args.layouts)? {
        let outcome = run_guarded(|| {
            let mut model = model_io::build_as::<B, M>(&cfg.model, device)?;
            let mut optim = adamw::<B, M>();
            let mut cursor = 0usize;
            let mut make_update =
                |model: M,
                 optim: &mut _,
                 lr: f64|
                 -> anyhow::Result<(M, recur64_runtime::accum::UpdateReport)> {
                    let mut micros = Vec::with_capacity(accum);
                    for _ in 0..accum {
                        let idx: Vec<usize> = (0..physical)
                            .map(|i| (cursor + i) % prep.obs.len())
                            .collect();
                        cursor += physical;
                        let obs: Vec<&ObservationV1> = idx.iter().map(|&i| &prep.obs[i]).collect();
                        let legal: Vec<Vec<ActionId>> =
                            idx.iter().map(|&i| prep.legal[i].clone()).collect();
                        let fr: Vec<&[CandidateFactsV1]> =
                            idx.iter().map(|&i| prep.facts[i].as_slice()).collect();
                        let inp = CandidateInputs::<B>::from_parts(&obs, &legal, &fr, device)?;
                        let (b, w) = (physical, inp.cands.width);
                        let mut t = vec![0.0f32; b * w];
                        for i in 0..b {
                            t[i * w] = 1.0;
                        }
                        let targets = Targets {
                            policy_target: burn::prelude::Tensor::<B, 2>::from_data(
                                burn::tensor::TensorData::new(t, [b, w]),
                                device,
                            ),
                            wdl_target:
                                burn::prelude::Tensor::<B, 1, burn::prelude::Int>::from_data(
                                    burn::tensor::TensorData::new(vec![1i32; b], [b]),
                                    device,
                                ),
                        };
                        micros.push(MicroBatch {
                            board: inp.board,
                            cands: inp.cands,
                            facts: needs_facts.then_some(inp.facts),
                            targets,
                            examples: b,
                        });
                    }
                    accumulated_update::<B, M, _, _>(model, optim, micros, lr, LossMode::Full)
                        .map_err(anyhow::Error::msg)
                };
            for _ in 0..2 {
                let (m, _) = make_update(model, &mut optim, 1e-4)?;
                model = m;
            }
            let t0 = Instant::now();
            let (result, samples) = monitor(gpu, || -> anyhow::Result<_> {
                let mut reports = Vec::new();
                for _ in 0..args.updates {
                    let (m, r) = make_update(model, &mut optim, 1e-4)?;
                    model = m;
                    reports.push(r);
                }
                Ok(reports)
            });
            let reports = result?;
            Ok((reports, t0.elapsed().as_secs_f64(), samples))
        });
        rows.push(match outcome {
            Ok((reports, wall, samples)) => {
                let finite = reports
                    .iter()
                    .all(|r| r.total_loss.is_finite() && r.grad_norm.is_finite());
                let per_update = wall / reports.len() as f64;
                serde_json::json!({
                    "layout": format!("{physical}x{accum}"), "physical": physical, "accum": accum,
                    "effective_batch": physical * accum, "ok": true, "finite": finite,
                    "updates": reports.len(), "sec_per_update": per_update,
                    "examples_per_sec": (physical * accum) as f64 / per_update,
                    "last_loss": reports.last().map(|r| r.total_loss),
                    "max_grad_norm": reports.iter().map(|r| r.grad_norm).fold(0.0f32, f32::max),
                    "gpu": samples,
                })
            }
            Err(e) => serde_json::json!({
                "layout": format!("{physical}x{accum}"), "physical": physical, "accum": accum,
                "effective_batch": physical * accum, "ok": false, "error": e,
            }),
        });
    }
    Ok(rows)
}

/// P0 item 17: repeated build / forward / drop must not grow device memory.
fn lifecycle_section<B, M>(
    cfg: &ProbeConfig,
    args: &V25QualArgs,
    prep: &Prepared,
    device: &B::Device,
) -> anyhow::Result<serde_json::Value>
where
    B: burn::prelude::Backend,
    M: NeuralModel<B>,
{
    let mut vram = Vec::new();
    for _ in 0..args.lifecycle_reps {
        {
            let model = model_io::build_as::<B, M>(&cfg.model, device)?;
            let needs = model.needs_candidate_facts();
            let batched = BatchedModel::new(model, 1, device.clone());
            let n = 32.min(prep.obs.len());
            let facts: Vec<_> = (0..n)
                .map(|i| needs.then(|| prep.facts[i].clone()))
                .collect();
            batched
                .evaluate_batch_with_facts(&prep.obs[..n], &prep.legal[..n], &facts)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
        } // model + BatchedModel dropped here (memory_cleanup runs in Drop)
        if let Some((mem, _, _)) = sample_gpu() {
            vram.push(mem);
        }
    }
    let growth = match (vram.get(1), vram.last()) {
        (Some(a), Some(b)) if vram.len() >= 3 => Some(*b as i64 - *a as i64),
        _ => None,
    };
    Ok(serde_json::json!({
        "reps": args.lifecycle_reps,
        "vram_mb_after_each_rep": vram,
        "growth_mb_after_rep2": growth,
        "rule": "pass iff growth after rep 2 <= 256 MiB (pre-registered)",
        "pass": growth.map(|g| g <= 256),
    }))
}

fn run_arch<B, MT, MI>(cfg: &ProbeConfig, args: &V25QualArgs) -> anyhow::Result<serde_json::Value>
where
    B: AutodiffBackend,
    MT: NeuralModel<B> + burn::module::AutodiffModule<B>,
    MI: NeuralModel<B::InnerBackend>,
{
    let device: B::Device = Default::default();
    let inner = Default::default();
    // Device guard first: build_as runs the CUDA known-answer check and errors
    // visibly if the requested device cannot run kernels.
    model_io::verify_device::<B::InnerBackend>(&inner)?;

    let pool = position_pool(args.positions, args.seed);
    let cost = facts_cost(&pool);
    let prep = Prepared {
        obs: pool.iter().map(encode_observation_v1).collect(),
        legal: pool.iter().map(|s| s.legal_actions()).collect(),
        facts: pool.iter().map(candidate_facts).collect(),
    };
    let forward = forward_section::<B::InnerBackend, MI>(cfg, args, &prep, &inner)?;
    let assembly = assembly_vs_forward::<B::InnerBackend, MI>(cfg, &pool, &inner, 32, 10)?;
    let layouts = layout_section::<B, MT>(cfg, args, &prep, &device)?;
    let lifecycle = lifecycle_section::<B::InnerBackend, MI>(cfg, args, &prep, &inner)?;

    // Selection rule (pre-registered): fastest layout with finite loss and
    // gradients whose peak VRAM is within the ceiling.
    let selected = layouts
        .iter()
        .filter(|l| {
            l["ok"] == true
                && l["finite"] == true
                && l["gpu"]["peak_vram_mb"]
                    .as_u64()
                    .is_none_or(|v| v <= args.vram_limit_mb)
        })
        .min_by(|a, b| {
            a["sec_per_update"]
                .as_f64()
                .unwrap_or(f64::MAX)
                .total_cmp(&b["sec_per_update"].as_f64().unwrap_or(f64::MAX))
        })
        .map(|l| l["layout"].clone());

    Ok(serde_json::json!({
        "config": cfg.name,
        "architecture": cfg.model.architecture.id(),
        "device": format!("{:?}", cfg.device),
        "precision": cfg.precision.label(),
        "params": {
            "total": MT::build(&cfg.model, &device)?.param_count(),
        },
        "facts_cost": cost,
        "assembly_vs_forward": assembly,
        "forward": forward,
        "layouts": layouts,
        "lifecycle": lifecycle,
        "selection_rule": format!(
            "fastest layout with finite loss+grads and peak VRAM <= {} MiB",
            args.vram_limit_mb
        ),
        "selected_layout": selected,
        "positions_pool": args.positions,
        "seed": args.seed,
    }))
}

pub fn run(args: V25QualArgs) -> anyhow::Result<()> {
    use recur64_model::config::{Architecture, DeviceKind};
    use recur64_model::model::ProbeModel;

    let cfg = ProbeConfig::from_toml_str(&std::fs::read_to_string(&args.config)?)?;
    cfg.model.validate()?;
    anyhow::ensure!(
        cfg.precision == recur64_model::config::Precision::Fp32,
        "v25-qual is an FP32 qualification (fusion/autotune/TF32 are not part of V2.5)"
    );
    let report = match (cfg.device, cfg.model.architecture) {
        (DeviceKind::Cpu, Architecture::CandidateV25) => run_arch::<
            recur64_model::train::CpuTrainBackend,
            recur64_model::candidate::CandidateV25Model<recur64_model::train::CpuTrainBackend>,
            recur64_model::candidate::CandidateV25Model<burn::backend::Flex>,
        >(&cfg, &args)?,
        (DeviceKind::Cpu, Architecture::ProbeV1) => run_arch::<
            recur64_model::train::CpuTrainBackend,
            ProbeModel<recur64_model::train::CpuTrainBackend>,
            ProbeModel<burn::backend::Flex>,
        >(&cfg, &args)?,
        #[cfg(feature = "cuda")]
        (DeviceKind::Cuda, Architecture::CandidateV25) => run_arch::<
            burn::backend::Autodiff<burn::backend::Cuda>,
            recur64_model::candidate::CandidateV25Model<
                burn::backend::Autodiff<burn::backend::Cuda>,
            >,
            recur64_model::candidate::CandidateV25Model<burn::backend::Cuda>,
        >(&cfg, &args)?,
        #[cfg(feature = "cuda")]
        (DeviceKind::Cuda, Architecture::ProbeV1) => run_arch::<
            burn::backend::Autodiff<burn::backend::Cuda>,
            ProbeModel<burn::backend::Autodiff<burn::backend::Cuda>>,
            ProbeModel<burn::backend::Cuda>,
        >(&cfg, &args)?,
        #[cfg(not(feature = "cuda"))]
        (DeviceKind::Cuda, _) => {
            anyhow::bail!("CUDA support is not compiled; rebuild with --features cuda")
        }
    };
    std::fs::create_dir_all(&args.output)?;
    let path = args.output.join(format!("v25-qual-{}.json", cfg.name));
    std::fs::write(&path, serde_json::to_vec_pretty(&report)?)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    println!("wrote {}", path.display());
    Ok(())
}
