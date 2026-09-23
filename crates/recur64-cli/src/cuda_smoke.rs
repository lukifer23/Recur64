//! `recur64 cuda-smoke` — Stage A GPU backend proof.
//!
//! Runs the actual probe graph on the NVIDIA GPU: FP32 forward at R=1/2/4,
//! backward, an AdamW update, and a training-checkpoint round trip. Errors
//! visibly on any non-finite value. This is a systems proof, not a benchmark.

use std::path::PathBuf;

use burn::backend::Cuda;
use burn::backend::cuda::CudaDevice;
use burn::optim::{GradientsParams, Optimizer};
use burn::prelude::*;
use clap::Args;

use recur64_model::Precision;
use recur64_model::checkpoint::{CheckpointMeta, SCHEMA_VERSION, load_training, save_training};
use recur64_model::config::ProbeConfig;
use recur64_model::fixture::SynthFixture;
use recur64_model::loss::model_loss;
use recur64_model::model::ProbeModel;
use recur64_model::train::adamw;

type TrainCuda = burn::backend::Autodiff<Cuda>;

#[derive(Args, Debug)]
pub struct CudaSmokeArgs {
    #[arg(long, default_value = "configs/micro.toml")]
    pub config: PathBuf,
    #[arg(long, default_value_t = 1)]
    pub seed: u64,
    #[arg(long, default_value = "runs/cuda-smoke")]
    pub output: PathBuf,
}

fn scalar<B: Backend>(t: Tensor<B, 1>) -> f32 {
    t.into_data().to_vec::<f32>().unwrap()[0]
}

pub fn run_cuda_smoke(args: CudaSmokeArgs) -> anyhow::Result<()> {
    let text = std::fs::read_to_string(&args.config)?;
    let cfg = ProbeConfig::from_toml_str(&text)?;
    anyhow::ensure!(
        matches!(cfg.precision, Precision::Fp32),
        "cuda-smoke is FP32 only; requested {}",
        cfg.precision.label()
    );

    let device: CudaDevice = Default::default();
    Cuda::<f32, i32>::sync(&device)?;
    println!("recur64 cuda-smoke");
    println!("device          : {device:?}");
    println!("config          : {}", cfg.name);

    // --- FP32 inference at each configured recurrence ---
    let model = ProbeModel::<Cuda>::new(cfg.model.clone(), &device);
    let fx = SynthFixture::new(4, cfg.model.in_features, args.seed);
    let (board, cands, _t) = fx.tensors::<Cuda>(&device);
    for r in &cfg.recurrence {
        let out = model.forward_r(board.clone(), &cands, *r, false);
        Cuda::<f32, i32>::sync(&device)?;
        let lp = out.readouts[0]
            .policy
            .log_probs
            .clone()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        anyhow::ensure!(
            lp.iter().all(|v| v.is_finite()),
            "non-finite GPU policy output at R={r}"
        );
        println!(
            "forward         : R={r} executed_blocks={} finite=yes",
            out.executed_blocks
        );
    }

    // --- backward + AdamW update ---
    let tmodel = ProbeModel::<TrainCuda>::new(cfg.model.clone(), &device);
    let mut optim = adamw::<TrainCuda, ProbeModel<TrainCuda>>();
    let (tboard, tcands, ttargets) = fx.tensors::<TrainCuda>(&device);
    let out = tmodel.forward_r(tboard, &tcands, 2, false);
    let loss = model_loss(&out, &ttargets);
    let l0 = scalar(loss.clone());
    anyhow::ensure!(l0.is_finite(), "non-finite GPU loss");
    let grads = GradientsParams::from_grads(loss.backward(), &tmodel);
    let w0 = tmodel.core_weight_scalar();
    let tmodel = optim.step(3e-4, tmodel, grads);
    Cuda::<f32, i32>::sync(&device)?;
    let w1 = tmodel.core_weight_scalar();
    anyhow::ensure!(w1.is_finite(), "non-finite GPU weight after update");
    anyhow::ensure!(w0 != w1, "optimizer did not move GPU parameters");
    println!("backward+adamw  : loss={l0:.6} core_w {w0:.6} -> {w1:.6}");

    // --- training checkpoint round trip ---
    let dir = args.output.join("ckpt");
    let _ = std::fs::remove_dir_all(&dir);
    let meta = CheckpointMeta {
        schema_version: SCHEMA_VERSION,
        recur64_version: recur64_model::VERSION.to_string(),
        git_revision: None,
        backend: "cuda (Burn 0.21.0)".to_string(),
        precision: "fp32".to_string(),
        model: cfg.model.clone(),
        recurrence: 2,
        deep_supervision: false,
        step: 1,
        lr: 3e-4,
        seed: args.seed,
        rng_state: 0,
    };
    save_training(&dir, &tmodel, &optim, &meta)?;
    let template = ProbeModel::<TrainCuda>::new(cfg.model.clone(), &device);
    let fresh = adamw::<TrainCuda, ProbeModel<TrainCuda>>();
    let (tmodel2, _optim2, meta2) = load_training(&dir, template, fresh, &device)?;
    Cuda::<f32, i32>::sync(&device)?;
    let w2 = tmodel2.core_weight_scalar();
    anyhow::ensure!(
        (w1 - w2).abs() < 1e-4,
        "GPU checkpoint weight mismatch: {w1} vs {w2}"
    );
    println!(
        "checkpoint      : step={} core_w {w1:.6} -> {w2:.6} (delta {:.2e})",
        meta2.step,
        (w1 - w2).abs()
    );

    println!("\nCUDA SMOKE: PASS");
    Ok(())
}
