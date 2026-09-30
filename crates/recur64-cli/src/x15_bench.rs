//! `recur64 x15 bench` - steady-state X15 timing and memory, one process.
//!
//! Two modes, always labelled in the output:
//!
//! * `infer`: the NORMAL forward path (`forward_thoughts`, training-semantics
//!   readouts). This is the one to quote for latency and compute matching. The
//!   diagnostic forward is deliberately not benchmarked here: it runs the
//!   output blocks after every thought and would make recurrence look more
//!   expensive than it is.
//! * `train`: forward + backward + gradient accumulation + AdamW step on a
//!   fixed real batch (`final_only_v1`, synthetic uniform targets - this
//!   measures cost and memory, not learning).
//!
//! Warm-up iterations (kernel JIT / autotune) are timed separately from the
//! steady state. Host data preparation is excluded (the batch is built once)
//! and reported on its own. No background processes are spawned; GPU telemetry
//! is the in-process scoped sampler.

use std::time::Instant;

use burn::optim::{GradientsAccumulator, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use clap::{Args, ValueEnum};

use recur64_model::chimera::ChimeraModel;
use recur64_model::config::ProbeConfig;
use recur64_model::loss::{Targets, readout_loss};
use recur64_model::model::CandidateTensors;
use recur64_runtime::gpu_telemetry::monitor;
use recur64_runtime::x15_inputs::{build_x15_batch, probe_positions, provider_for_config};

#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum BenchMode {
    /// Normal (non-diagnostic) forward only.
    Infer,
    /// Repeated build / forward / drop cycles; VRAM must plateau.
    Lifecycle,
    /// Forward + backward + accumulate + AdamW step.
    Train,
}

#[derive(Args, Debug)]
pub struct BenchArgs {
    #[arg(long, default_value = "configs/x15_cuda.toml")]
    pub config: std::path::PathBuf,
    #[arg(long, value_enum)]
    pub mode: BenchMode,
    /// Thought counts to measure.
    #[arg(long, value_delimiter = ',', default_value = "1,2,4")]
    pub thoughts: Vec<usize>,
    /// Positions per forward (infer) or per micro-batch (train).
    #[arg(long, default_value_t = 16)]
    pub batch: usize,
    /// Micro-batches accumulated per optimizer update (train).
    #[arg(long, default_value_t = 1)]
    pub accum: usize,
    /// Untimed warm-up iterations (JIT, autotune).
    #[arg(long, default_value_t = 2)]
    pub warmup: usize,
    /// Timed iterations.
    #[arg(long, default_value_t = 5)]
    pub iters: usize,
    /// Learning rate for the train step (throughput probe only).
    #[arg(long, default_value_t = 1e-4)]
    pub lr: f64,
}

fn sync<B: Backend>(device: &B::Device) {
    let _ = B::sync(device);
}

fn uniform_targets<B: Backend>(
    cands: &CandidateTensors<B>,
    batch: usize,
    device: &B::Device,
) -> Targets<B> {
    let mask = cands.mask.clone().float();
    let denom = mask.clone().sum_dim(1).clamp_min(1.0);
    Targets {
        policy_target: mask / denom,
        wdl_target: Tensor::<B, 1, Int>::zeros([batch], device),
        wdl_mask: None,
    }
}

fn report_gpu(gpu: &recur64_runtime::gpu_telemetry::GpuSamples) -> String {
    format!(
        "peak_vram={:?} MiB util_max={:?}% busy_mean={:.0} temp_max={:?} C",
        gpu.peak_vram_mb,
        gpu.util_max,
        gpu.util_busy_mean.unwrap_or(0.0),
        gpu.temp_max_c
    )
}

pub fn run_infer<B: Backend>(cfg: &ProbeConfig, args: &BenchArgs) -> anyhow::Result<()> {
    let device: B::Device = Default::default();
    let model =
        recur64_runtime::model_io::build_chimera::<B>(&cfg.model, &cfg.experimental, &device)?;
    let provider = provider_for_config(&cfg.experimental)?;
    let states = probe_positions(args.batch.max(1), 40);
    let prep = Instant::now();
    let batch = build_x15_batch::<B>(&states, &cfg.experimental, provider.as_ref(), &device)?;
    sync::<B>(&device);
    println!(
        "bench infer (NORMAL forward, not diagnostic): batch={} params={} host_prep={:.1} ms",
        batch.batch,
        model.num_params(),
        prep.elapsed().as_secs_f64() * 1e3
    );
    for &t in &args.thoughts {
        let warm = Instant::now();
        for _ in 0..args.warmup.max(1) {
            let out = model.forward_thoughts(&batch.input, &batch.cands, t);
            let _ = out
                .readouts
                .last()
                .map(|r| r.wdl_logits.clone().into_data());
        }
        let warm_s = warm.elapsed().as_secs_f64();
        let ((per_iter, finite), gpu) = monitor(true, || {
            let start = Instant::now();
            let mut finite = true;
            for _ in 0..args.iters.max(1) {
                let out = model.forward_thoughts(&batch.input, &batch.cands, t);
                let wdl = out.readouts.last().expect("readout").wdl_logits.clone();
                let v = wdl.into_data().to_vec::<f32>().unwrap_or_default();
                finite &= !v.is_empty() && v.iter().all(|x| x.is_finite());
            }
            sync::<B>(&device);
            (
                start.elapsed().as_secs_f64() / args.iters.max(1) as f64,
                finite,
            )
        });
        println!(
            "  T={t}: {:.1} ms/forward  {:.1} pos/s  warmup={:.1}s finite={finite}  {}",
            per_iter * 1e3,
            batch.batch as f64 / per_iter,
            warm_s,
            report_gpu(&gpu)
        );
    }
    Ok(())
}

pub fn run_train<B: AutodiffBackend>(cfg: &ProbeConfig, args: &BenchArgs) -> anyhow::Result<()> {
    let device: B::Device = Default::default();
    let provider = provider_for_config(&cfg.experimental)?;
    let states = probe_positions(args.batch.max(1), 40);
    let accum = args.accum.max(1);
    println!(
        "bench train (fwd+bwd+accum+AdamW, final_only, synthetic targets): micro-batch={} accum={} effective={}",
        args.batch,
        accum,
        args.batch * accum
    );
    for &t in &args.thoughts {
        // Fresh model per T so peak memory is attributable to that T.
        let mut model =
            recur64_runtime::model_io::build_chimera::<B>(&cfg.model, &cfg.experimental, &device)?;
        let mut optim = recur64_model::train::adamw::<B, _>();
        let batch = build_x15_batch::<B>(&states, &cfg.experimental, provider.as_ref(), &device)?;
        let targets = uniform_targets(&batch.cands, batch.batch, &device);

        let warm = Instant::now();
        let mut first_loss = f32::NAN;
        for i in 0..args.warmup.max(1) {
            let (m, l, _) = update(model, &mut optim, &batch, &targets, t, accum, args.lr);
            model = m;
            if i == 0 {
                first_loss = l;
            }
        }
        sync::<B>(&device);
        let warm_s = warm.elapsed().as_secs_f64();

        let (result, gpu) = monitor(true, || -> anyhow::Result<(f64, f32, bool)> {
            let start = Instant::now();
            let mut last = f32::NAN;
            let mut finite = true;
            for _ in 0..args.iters.max(1) {
                let (m, l, f) = update(model, &mut optim, &batch, &targets, t, accum, args.lr);
                model = m;
                last = l;
                finite &= f;
            }
            sync::<B>(&device);
            Ok((
                start.elapsed().as_secs_f64() / args.iters.max(1) as f64,
                last,
                finite,
            ))
        });
        let (per_step, last_loss, finite) = result?;
        println!(
            "  T={t}: {:.0} ms/update  {:.1} ex/s  loss {first_loss:.4}->{last_loss:.4} finite={finite} warmup={warm_s:.1}s  {}",
            per_step * 1e3,
            (args.batch * accum) as f64 / per_step,
            report_gpu(&gpu)
        );
        anyhow::ensure!(finite, "non-finite loss at T={t}");
    }
    Ok(())
}

/// One optimizer update over `accum` micro-batches of the same fixed batch.
fn update<B, O>(
    model: ChimeraModel<B>,
    optim: &mut O,
    batch: &recur64_runtime::x15_inputs::X15Batch<B>,
    targets: &Targets<B>,
    t: usize,
    accum: usize,
    lr: f64,
) -> (ChimeraModel<B>, f32, bool)
where
    B: AutodiffBackend,
    O: Optimizer<ChimeraModel<B>, B>,
{
    let mut acc = GradientsAccumulator::new();
    let mut loss_sum = 0.0f32;
    let mut finite = true;
    for _ in 0..accum {
        let out = model.forward_thoughts(&batch.input, &batch.cands, t);
        let loss = readout_loss(&out.readouts[0], targets) / accum as f32;
        let l: f32 = loss.clone().into_scalar().elem();
        loss_sum += l;
        finite &= l.is_finite();
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        acc.accumulate(&model, grads);
    }
    let model = optim.step(lr, model, acc.grads());
    (model, loss_sum, finite)
}

/// Build -> forward -> drop -> cleanup, repeated. Post-cleanup VRAM must
/// plateau; a monotonic climb is a per-owner leak (the D44 failure mode).
pub fn run_lifecycle<B: Backend>(cfg: &ProbeConfig, args: &BenchArgs) -> anyhow::Result<()> {
    let device: B::Device = Default::default();
    let provider = provider_for_config(&cfg.experimental)?;
    let states = probe_positions(args.batch.max(1), 40);
    let t = args.thoughts.last().copied().unwrap_or(1);
    println!(
        "bench lifecycle: {} cycles, T={t}, batch={}",
        args.iters, args.batch
    );
    let mut after = Vec::new();
    for cycle in 0..args.iters.max(2) {
        let model = recur64_runtime::model_io::build_chimera_unverified::<B>(
            &cfg.model,
            &cfg.experimental,
            &device,
        )?;
        let batch = build_x15_batch::<B>(&states, &cfg.experimental, provider.as_ref(), &device)?;
        let out = model.forward_thoughts(&batch.input, &batch.cands, t);
        let _ = out
            .readouts
            .last()
            .map(|r| r.wdl_logits.clone().into_data());
        let during = recur64_runtime::gpu_telemetry::sample_gpu().map(|s| s.0);
        drop(out);
        drop(batch);
        drop(model);
        let _ = B::sync(&device);
        B::memory_cleanup(&device);
        let post = recur64_runtime::gpu_telemetry::sample_gpu().map(|s| s.0);
        println!("  cycle {cycle}: during={during:?} MiB after_drop={post:?} MiB");
        after.push(post);
    }
    if let (Some(Some(first)), Some(Some(last))) = (after.get(1), after.last()) {
        let growth = last.saturating_sub(*first);
        println!("post-drop VRAM cycle1 -> last: {first} -> {last} MiB (growth {growth} MiB)");
        anyhow::ensure!(
            growth <= 64,
            "VRAM did not plateau: grew {growth} MiB across cycles"
        );
    }
    Ok(())
}
