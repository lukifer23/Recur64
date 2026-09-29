//! `recur64 x15 ...` — the X15 / Chimera fast probe harness.
//!
//! The owner's experimental philosophy is that long runs do not come first, so
//! every subcommand here is sized for a probe class:
//!
//! | command | class | budget |
//! |---|---|---|
//! | `x15 info` | P0 | seconds (CPU) |
//! | `x15 sanity` | P0 | <= 3 min |
//! | `x15 parity` | P0 | seconds..minutes |
//! | `x15 grads` | P0/P1 | <= 3 min |
//! | `x15 thoughts` | P1 | 30 s .. 3 min |
//!
//! No subcommand trains for hundreds of games, uses an external engine, or
//! claims the architecture works.

use std::path::{Path, PathBuf};

use burn::prelude::*;
use clap::{Args, Subcommand};

use recur64_core::GameState;
use recur64_model::chimera::ChimeraModel;
use recur64_model::config::ProbeConfig;
use recur64_model::experimental::Architecture;
use recur64_model::loss::{Targets, readout_loss};
use recur64_model::model::CandidateTensors;
use recur64_runtime::x15_inputs::{build_x15_batch, probe_positions, provider_for_config};

#[derive(Args, Debug)]
pub struct X15Args {
    #[command(subcommand)]
    pub command: X15Command,
}

#[derive(Subcommand, Debug)]
pub enum X15Command {
    /// Parameter accounting by subsystem, plus the config's identity fields.
    Info(ConfigArg),
    /// Forward sanity: finite, policy normalizes, WDL roughly neutral, T1/T2/T4
    /// execute, parameter count is the same at every T.
    Sanity(SanityArgs),
    /// Native vs WebAssembly `ComputeBankV1` parity, byte for byte (P0 gate).
    Parity(ParityArgs),
    /// Steady-state latency / throughput / VRAM (normal forward or train step).
    Bench(crate::x15_bench::BenchArgs),
    /// Module gradient probe: every gated subsystem must receive a non-zero
    /// gradient on the first training step.
    Grads(BatchArgs),
    /// Per-thought metrics for one batch.
    Thoughts(BatchArgs),
}

#[derive(Args, Debug)]
pub struct ConfigArg {
    #[arg(long)]
    pub config: PathBuf,
}

#[derive(Args, Debug)]
pub struct SanityArgs {
    #[arg(long)]
    pub config: PathBuf,
    /// Thought counts to execute.
    #[arg(long, value_delimiter = ',', default_value = "1,2,4")]
    pub thoughts: Vec<usize>,
    #[arg(long, default_value_t = 8)]
    pub batch: usize,
}

#[derive(Args, Debug)]
pub struct BatchArgs {
    #[arg(long)]
    pub config: PathBuf,
    /// Thought count for the pass.
    #[arg(long, default_value_t = 4)]
    pub thoughts: usize,
    #[arg(long, default_value_t = 8)]
    pub batch: usize,
}

#[derive(Args, Debug)]
pub struct ParityArgs {
    /// Random legal positions to compare (plus a fixed special-move set).
    #[arg(long, default_value_t = 300)]
    pub positions: usize,
}

fn load_config(path: &Path) -> anyhow::Result<ProbeConfig> {
    let text = std::fs::read_to_string(path)?;
    let cfg = ProbeConfig::from_toml_str(&text)?;
    anyhow::ensure!(
        cfg.experimental.is_chimera(),
        "{} is not a chimera_v1 config (architecture = {:?}); the x15 command \
         group only runs the X15 family",
        path.display(),
        cfg.experimental.architecture
    );
    cfg.experimental.validate(cfg.model.width)?;
    Ok(cfg)
}

fn build_chimera<B: Backend>(
    cfg: &ProbeConfig,
    device: &B::Device,
) -> anyhow::Result<ChimeraModel<B>> {
    // Every X15 build goes through the runtime boundary, which runs the
    // behavioural device check (D57) before any weights are created.
    recur64_runtime::model_io::build_chimera::<B>(&cfg.model, &cfg.experimental, device)
}

/// Build a uniform policy target and neutral WDL target over a real batch.
fn synthetic_targets<B: Backend>(
    cands: &CandidateTensors<B>,
    batch: usize,
    device: &B::Device,
) -> Targets<B> {
    let mask = cands.mask.clone().float();
    let denom = mask.clone().sum_dim(1).clamp_min(1.0);
    let policy_target = mask / denom;
    Targets {
        policy_target,
        wdl_target: Tensor::<B, 1, Int>::zeros([batch], device),
        wdl_mask: None,
    }
}

pub fn run(args: X15Args) -> anyhow::Result<()> {
    match args.command {
        X15Command::Info(a) => run_info(a),
        X15Command::Sanity(a) => dispatch_device(&a.config, &a, run_sanity),
        X15Command::Grads(a) => dispatch_device(&a.config, &a, run_grads),
        X15Command::Thoughts(a) => dispatch_device(&a.config, &a, run_thoughts),
        X15Command::Parity(a) => run_parity(a),
        X15Command::Bench(a) => run_bench(a),
    }
}

fn dispatch_device<A, F>(config: &Path, args: &A, f: F) -> anyhow::Result<()>
where
    F: Fn(&ProbeConfig, &A) -> anyhow::Result<()>,
{
    let cfg = load_config(config)?;
    match cfg.device {
        recur64_model::config::DeviceKind::Cpu => f(&cfg, args),
        recur64_model::config::DeviceKind::Cuda => {
            #[cfg(feature = "cuda")]
            {
                // In-process, scoped telemetry: the sampler thread is joined
                // when the work ends, so nothing outlives the command.
                let (result, gpu) = recur64_runtime::gpu_telemetry::monitor(true, || f(&cfg, args));
                println!(
                    "gpu telemetry   : peak_vram={:?} MiB util_max={:?}% util_busy_mean={:?} temp_max={:?} C ({} samples)",
                    gpu.peak_vram_mb, gpu.util_max, gpu.util_busy_mean, gpu.temp_max_c, gpu.samples
                );
                result
            }
            #[cfg(not(feature = "cuda"))]
            {
                anyhow::bail!("CUDA support is not compiled; rebuild with --features cuda")
            }
        }
    }
}

// --- info ------------------------------------------------------------------

fn run_info(args: ConfigArg) -> anyhow::Result<()> {
    let cfg = load_config(&args.config)?;
    let device = Default::default();
    let model = build_chimera::<burn::backend::Flex>(&cfg, &device)?;

    println!("config          : {}", cfg.name);
    println!(
        "architecture    : {}",
        cfg.experimental.architecture.label()
    );
    println!(
        "thought steps   : {} (reasoning latents {})",
        cfg.experimental.thought_steps, cfg.experimental.reasoning_tokens
    );
    println!(
        "aux width/heads : {}/{}",
        cfg.experimental.reasoning.aux_width, cfg.experimental.reasoning.aux_heads
    );
    println!(
        "compute         : provider={} mate_depth={} tokens={} fields={}",
        cfg.experimental.compute.provider.label(),
        cfg.experimental.compute.mate_depth,
        cfg.experimental.compute.tokens,
        cfg.experimental.compute.fields
    );
    println!(
        "visual          : provider={} resolution={} channels={} blocks={}",
        cfg.experimental.visual.provider.label(),
        cfg.experimental.visual.resolution,
        cfg.experimental.visual.channels,
        cfg.experimental.visual.blocks
    );
    println!(
        "deep supervision: {}",
        cfg.experimental.deep_supervision.label()
    );
    println!(
        "retrieval       : {} (memory_tokens {})",
        cfg.experimental.retrieval.provider.label(),
        cfg.experimental.retrieval.memory_tokens
    );
    println!();

    println!("parameter breakdown (unique; shared counted once):");
    let mut total = 0usize;
    for (name, n) in model.param_breakdown() {
        println!("  {name:<42} {n:>12}");
        total += n;
    }
    println!("  {:<42} {total:>12}", "TOTAL");
    assert_eq!(
        total,
        model.num_params(),
        "the breakdown must sum to Module::num_params"
    );
    println!(
        "  {:<42} {:>12} ({:.1} MiB at FP32)",
        "estimated parameter bytes",
        total * 4,
        (total * 4) as f64 / (1024.0 * 1024.0)
    );

    println!();
    println!("executed blocks / parameter count by thought count:");
    println!(
        "  {:<6} {:>16} {:>16} {:>14}",
        "T", "executed blocks", "parameters", "vs T=1"
    );
    let base_blocks = recur64_model::chimera::executed_blocks(&cfg.model, &cfg.experimental, 1);
    for t in 1..=cfg.experimental.thought_steps.max(1) {
        let blocks = recur64_model::chimera::executed_blocks(&cfg.model, &cfg.experimental, t);
        println!(
            "  T={t:<4} {:>16} {:>16} {:>13.2}x",
            blocks,
            model.num_params(),
            blocks as f64 / base_blocks as f64
        );
    }
    Ok(())
}

// --- sanity ----------------------------------------------------------------

fn run_sanity(cfg: &ProbeConfig, args: &SanityArgs) -> anyhow::Result<()> {
    match cfg.device {
        recur64_model::config::DeviceKind::Cpu => run_sanity_impl::<burn::backend::Flex>(cfg, args),
        recur64_model::config::DeviceKind::Cuda => {
            #[cfg(feature = "cuda")]
            {
                run_sanity_impl::<burn::backend::Cuda>(cfg, args)
            }
            #[cfg(not(feature = "cuda"))]
            {
                anyhow::bail!("CUDA support is not compiled; rebuild with --features cuda")
            }
        }
    }
}

fn run_sanity_impl<B: Backend>(cfg: &ProbeConfig, args: &SanityArgs) -> anyhow::Result<()> {
    let device: B::Device = Default::default();
    let model = build_chimera::<B>(cfg, &device)?;
    let provider = provider_for_config(&cfg.experimental)?;
    let states = probe_positions(args.batch.max(1), 40);
    let batch = build_x15_batch::<B>(&states, &cfg.experimental, provider.as_ref(), &device)?;

    let params = model.num_params();
    println!(
        "architecture    : {} ({} parameters)",
        Architecture::ChimeraV1.label(),
        params
    );
    println!(
        "batch           : {} positions; compute={} visual={}",
        batch.batch,
        batch.input.compute.is_some(),
        batch.input.visual.is_some()
    );
    println!(
        "phase times (us): observation {} compute {} render {} upload {}",
        batch.phases.observation_us,
        batch.phases.compute_us,
        batch.phases.visual_render_us,
        batch.phases.upload_us
    );

    for &t in &args.thoughts {
        let out = model.forward_thoughts(&batch.input, &batch.cands, t);
        let readout = out
            .readouts
            .last()
            .ok_or_else(|| anyhow::anyhow!("no readout at T={t}"))?;

        let lp = readout
            .policy
            .log_probs
            .clone()
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        anyhow::ensure!(
            lp.iter().all(|v| v.is_finite()),
            "T={t}: policy is not finite"
        );
        let wdl = readout
            .wdl_logits
            .clone()
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        anyhow::ensure!(
            wdl.iter().all(|v| v.is_finite()),
            "T={t}: WDL is not finite"
        );

        // Legal policy mass per row must be exactly 1. Padded entries carry
        // exactly 0 log-probability by design, so the mask must be applied.
        let width = batch.cands.width;
        // Read the mask back as floats: a device boolean tensor comes back as
        // `Bool(U8)` data on CUDA, which `to_vec::<bool>` refuses.
        let mask: Vec<bool> = batch
            .cands
            .mask
            .clone()
            .float()
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .into_iter()
            .map(|v| v > 0.5)
            .collect();
        for row in 0..batch.batch {
            let sum: f32 = (0..width)
                .filter(|k| mask[row * width + k])
                .map(|k| lp[row * width + k].exp())
                .sum();
            anyhow::ensure!(
                (sum - 1.0).abs() < 1e-4,
                "T={t} row {row}: legal policy mass {sum} is not 1"
            );
        }
        // A fresh network must be near-neutral: head v2 zero-initializes WDL
        // and both WDL terms are zero at step 0.
        let max_abs_value = wdl.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        anyhow::ensure!(
            max_abs_value < 1e-5,
            "T={t}: a fresh X15 must start neutral, saw |logit| {max_abs_value}"
        );
        anyhow::ensure!(
            model.num_params() == params,
            "parameter count changed with T: {} -> {}",
            params,
            model.num_params()
        );

        let entropy = out
            .thoughts
            .last()
            .map(|m| {
                m.policy_entropy
                    .clone()
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap_or_default()
            })
            .unwrap_or_default();
        let mean_entropy = if entropy.is_empty() {
            f32::NAN
        } else {
            entropy.iter().sum::<f32>() / entropy.len() as f32
        };
        println!(
            "T={t:<4} readouts={} thoughts={} executed_blocks={} mean_policy_entropy={mean_entropy:.4} max|WDL logit|={max_abs_value:.2e}",
            out.readouts.len(),
            out.thoughts.len(),
            out.executed_blocks,
        );
    }
    println!("PASS: forward sanity");
    Ok(())
}

// --- grads -----------------------------------------------------------------

fn run_grads(cfg: &ProbeConfig, args: &BatchArgs) -> anyhow::Result<()> {
    match cfg.device {
        recur64_model::config::DeviceKind::Cpu => {
            run_grads_impl::<recur64_model::train::CpuTrainBackend>(cfg, args)
        }
        recur64_model::config::DeviceKind::Cuda => {
            #[cfg(feature = "cuda")]
            {
                type CudaTrain = burn::backend::Autodiff<burn::backend::Cuda>;
                run_grads_impl::<CudaTrain>(cfg, args)
            }
            #[cfg(not(feature = "cuda"))]
            {
                anyhow::bail!("CUDA support is not compiled; rebuild with --features cuda")
            }
        }
    }
}

fn run_grads_impl<B: burn::tensor::backend::AutodiffBackend>(
    cfg: &ProbeConfig,
    args: &BatchArgs,
) -> anyhow::Result<()> {
    let device: B::Device = Default::default();
    let model = build_chimera::<B>(cfg, &device)?;
    let provider = provider_for_config(&cfg.experimental)?;
    let states = probe_positions(args.batch.max(1), 40);
    let batch = build_x15_batch::<B>(&states, &cfg.experimental, provider.as_ref(), &device)?;
    let targets = synthetic_targets(&batch.cands, batch.batch, &device);

    let out = model.forward_thoughts(&batch.input, &batch.cands, args.thoughts.max(1));
    anyhow::ensure!(
        !out.readouts.is_empty(),
        "the gradient probe needs at least one readout"
    );
    // Sum every readout's loss, so a subsystem that only contributes to an
    // intermediate readout (auxiliary deep supervision) still gets gradient.
    let mut loss: Option<Tensor<B, 1>> = None;
    for r in &out.readouts {
        let l = readout_loss(r, &targets);
        loss = Some(match loss {
            None => l,
            Some(acc) => acc + l,
        });
    }
    let loss = loss.unwrap();
    let grads = burn::optim::GradientsParams::from_grads(loss.backward(), &model);

    println!(
        "module gradient probe: T={} batch={} readouts={}",
        args.thoughts.max(1),
        batch.batch,
        out.readouts.len()
    );
    let mut worst: Vec<(&'static str, f32)> = Vec::new();
    for (name, norm) in model.subsystem_grad_norms(&grads) {
        println!("  {name:<20} grad_norm = {norm:.6e}");
        worst.push((name, norm));
    }
    // Every gated subsystem must receive a non-zero gradient on the first step.
    let zero: Vec<&str> = worst
        .iter()
        .filter(|(_, n)| !(*n).is_finite() || *n <= 0.0)
        .map(|(n, _)| *n)
        .collect();
    anyhow::ensure!(
        zero.is_empty(),
        "these subsystems received a zero or non-finite gradient on the first step: {zero:?}"
    );
    println!("PASS: every gated subsystem received gradient");
    Ok(())
}

// --- thoughts --------------------------------------------------------------

fn run_thoughts(cfg: &ProbeConfig, args: &BatchArgs) -> anyhow::Result<()> {
    match cfg.device {
        recur64_model::config::DeviceKind::Cpu => {
            run_thoughts_impl::<burn::backend::Flex>(cfg, args)
        }
        recur64_model::config::DeviceKind::Cuda => {
            #[cfg(feature = "cuda")]
            {
                run_thoughts_impl::<burn::backend::Cuda>(cfg, args)
            }
            #[cfg(not(feature = "cuda"))]
            {
                anyhow::bail!("CUDA support is not compiled; rebuild with --features cuda")
            }
        }
    }
}

fn run_thoughts_impl<B: Backend>(cfg: &ProbeConfig, args: &BatchArgs) -> anyhow::Result<()> {
    let device: B::Device = Default::default();
    let model = build_chimera::<B>(cfg, &device)?;
    let provider = provider_for_config(&cfg.experimental)?;
    let states = probe_positions(args.batch.max(1), 40);
    let batch = build_x15_batch::<B>(&states, &cfg.experimental, provider.as_ref(), &device)?;

    let t = args.thoughts.max(1);
    // Diagnostic forward: a readout after EVERY thought, whatever the
    // configured deep-supervision mode. Same network, same final output.
    let out = model.forward_thoughts_diagnostic(&batch.input, &batch.cands, t);
    println!(
        "thought progression (T={t}, batch={}, deep_supervision={}, diagnostic readouts):",
        batch.batch,
        cfg.experimental.deep_supervision.label()
    );
    println!(
        "  {:<4} {:>9} {:>8} {:>8} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>8}",
        "t",
        "entropy",
        "p(win)",
        "p(draw)",
        "|Z|",
        "d|Z|",
        "square",
        "compute",
        "visual",
        "KL(t|t-1)",
        "dWDL(L1)",
        "g_reason"
    );
    let mean = |v: Vec<f32>| {
        if v.is_empty() {
            f32::NAN
        } else {
            v.iter().sum::<f32>() / v.len() as f32
        }
    };
    let read = |x: &Tensor<B, 1>| x.clone().into_data().to_vec::<f32>().unwrap_or_default();
    for (i, m) in out.thoughts.iter().enumerate() {
        let wdl = m
            .wdl
            .clone()
            .into_data()
            .to_vec::<f32>()
            .unwrap_or_default();
        let batch_rows = wdl.len() / 3;
        let col = |c: usize| (0..batch_rows).map(|r| wdl[r * 3 + c]).collect::<Vec<_>>();
        println!(
            "  {:<4} {:>9.4} {:>8.4} {:>8.4} {:>9.4} {:>9.2e} {:>9.2e} {:>9.2e} {:>9.2e} {:>9.2e} {:>9.2e} {:>8.3}",
            i + 1,
            mean(read(&m.policy_entropy)),
            mean(col(0)),
            mean(col(1)),
            mean(read(&m.latent_norm)),
            mean(read(&m.latent_delta_norm)),
            mean(read(&m.square_pathway)),
            mean(read(&m.compute_pathway)),
            mean(read(&m.visual_pathway)),
            mean(read(&m.policy_kl_prev)),
            mean(read(&m.wdl_l1_prev)),
            mean(read(&m.gate_reason)),
        );
    }
    Ok(())
}

// --- parity ----------------------------------------------------------------

fn run_parity(args: ParityArgs) -> anyhow::Result<()> {
    use recur64_compute::{ComputeProvider, NativeProvider, WasmProvider};
    use recur64_coproc::{OUTPUT_LEN, empty_output};

    let edge: Vec<(&str, u8)> = vec![
        (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            2,
        ),
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", 2),
        (
            "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
            2,
        ),
        ("1n5k/P7/8/8/8/8/8/K7 w - - 0 1", 2),
        ("2r4k/1P6/8/8/8/8/8/K7 w - - 0 1", 2),
        ("4Q2k/8/8/8/8/8/8/K7 b - - 0 1", 2),
        ("8/8/8/8/8/2k5/4R3/4K3 b - - 0 1", 2),
        ("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1", 2),
        ("7k/8/8/8/8/8/6Q1/6K1 w - - 0 1", 2),
        ("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", 2),
        ("8/8/8/4k3/8/8/4B3/4K3 w - - 0 1", 2),
        ("8/8/8/4k3/8/8/3Q4/4K3 w - - 60 40", 2),
    ];

    let mut inputs: Vec<Vec<u8>> = Vec::new();
    for (fen, depth) in edge {
        let state = GameState::from_fen(fen)?;
        let obs = recur64_core::encode_observation_v1(&state);
        inputs.push(recur64_compute::encode_input(
            &obs,
            &state.legal_actions(),
            depth,
        )?);
    }
    for state in probe_positions(args.positions, 90) {
        let obs = recur64_core::encode_observation_v1(&state);
        inputs.push(recur64_compute::encode_input(
            &obs,
            &state.legal_actions(),
            1,
        )?);
    }
    for state in probe_positions(args.positions / 5 + 1, 70) {
        let obs = recur64_core::encode_observation_v1(&state);
        inputs.push(recur64_compute::encode_input(
            &obs,
            &state.legal_actions(),
            2,
        )?);
    }

    let native = NativeProvider;
    let wasm = WasmProvider::new()?;
    let mut a: Vec<Vec<u8>> = inputs.iter().map(|_| empty_output()).collect();
    let mut b: Vec<Vec<u8>> = inputs.iter().map(|_| empty_output()).collect();
    let t0 = std::time::Instant::now();
    native.compute_batch(&inputs, &mut a)?;
    let native_us = t0.elapsed().as_micros() as u64;
    let t1 = std::time::Instant::now();
    wasm.compute_batch(&inputs, &mut b)?;
    let wasm_us = t1.elapsed().as_micros() as u64;

    anyhow::ensure!(
        a.len() == b.len() && a.iter().zip(&b).all(|(x, y)| x == y),
        "NativeV1 and WasmV1 disagree; STOP and do not train on this build"
    );
    for x in &a {
        anyhow::ensure!(x.len() == OUTPUT_LEN, "bank has the wrong length");
    }
    println!(
        "PASS: native == wasm on {} positions ({} bytes each)",
        inputs.len(),
        OUTPUT_LEN
    );
    println!(
        "  native {} ms ({} us/position), wasm {} ms ({} us/position)",
        native_us / 1000,
        native_us / inputs.len().max(1) as u64,
        wasm_us / 1000,
        wasm_us / inputs.len().max(1) as u64
    );
    println!(
        "  wasm overhead vs native: {:.2}x",
        wasm_us as f64 / native_us.max(1) as f64
    );
    println!(
        "  compute_bank_version    : {}",
        recur64_coproc::COMPUTE_BANK_VERSION
    );
    Ok(())
}

/// Kept for tests and future probes: the experimental default for a bare block.
#[cfg(test)]
#[allow(dead_code)]
pub fn default_experimental() -> recur64_model::experimental::ExperimentalConfig {
    recur64_model::experimental::ExperimentalConfig::default()
}

// --- bench -----------------------------------------------------------------

fn run_bench(args: crate::x15_bench::BenchArgs) -> anyhow::Result<()> {
    use crate::x15_bench::{BenchMode, run_infer, run_train};
    let cfg = load_config(&args.config)?;
    match (cfg.device, args.mode) {
        (recur64_model::config::DeviceKind::Cpu, BenchMode::Infer) => {
            run_infer::<burn::backend::Flex>(&cfg, &args)
        }
        (recur64_model::config::DeviceKind::Cpu, BenchMode::Train) => {
            run_train::<recur64_model::train::CpuTrainBackend>(&cfg, &args)
        }
        #[cfg(feature = "cuda")]
        (recur64_model::config::DeviceKind::Cuda, BenchMode::Infer) => {
            run_infer::<burn::backend::Cuda>(&cfg, &args)
        }
        #[cfg(feature = "cuda")]
        (recur64_model::config::DeviceKind::Cuda, BenchMode::Train) => {
            run_train::<burn::backend::Autodiff<burn::backend::Cuda>>(&cfg, &args)
        }
        #[cfg(not(feature = "cuda"))]
        (recur64_model::config::DeviceKind::Cuda, _) => {
            anyhow::bail!("CUDA support is not compiled; rebuild with --features cuda")
        }
    }
}
