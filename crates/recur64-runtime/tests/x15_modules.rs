//! X15 / Chimera integration tests: the module-gradient gate, the "pathways
//! off" control, the native/WASM parity gate, and the thought-step contract.
//!
//! These are the assertions the architecture brief calls probes; they live
//! here as tests so the gate runs with the rest of the workspace.

// The fixtures below build configs field by field for readability.
#![allow(clippy::field_reassign_with_default)]

use burn::prelude::*;
use burn::tensor::backend::Backend;

use recur64_coproc::ComputeProviderKind;
use recur64_model::chimera::ChimeraModel;
use recur64_model::config::ModelConfig;
use recur64_model::experimental::{
    Architecture, ComputeConfig, DeepSupervisionMode, ExperimentalConfig, ReasoningConfig,
    VisualConfig, VisualProviderKind,
};
use recur64_model::loss::{Targets, readout_loss};
use recur64_runtime::x15_inputs::{build_x15_batch, probe_positions, provider_for_config};

type TrainB = burn::backend::Autodiff<burn::backend::Flex>;
type EvalB = burn::backend::Flex;

/// The trunk geometry from `configs/x15.toml`, shrunk so a CPU test is fast.
fn tiny_cfg() -> ModelConfig {
    ModelConfig {
        width: 64,
        heads: 4,
        ffn: 96,
        input_blocks: 1,
        core_blocks: 2,
        output_blocks: 1,
        squares: 64,
        in_features: 119,
        policy_dim: 16,
        wdl_classes: 3,
        promo_codes: 5,
        rms_eps: 1e-5,
    }
}

fn exp() -> ExperimentalConfig {
    let mut e = ExperimentalConfig::default();
    e.architecture = Architecture::ChimeraV1;
    e.thought_steps = 4;
    e.reasoning_tokens = 4;
    e.reasoning = ReasoningConfig {
        aux_width: 32,
        aux_heads: 4,
        latent_ffn: 64,
        ..ReasoningConfig::default()
    };
    e.compute = ComputeConfig {
        provider: ComputeProviderKind::NativeV1,
        enabled: true,
        ..ComputeConfig::default()
    };
    e.visual = VisualConfig {
        provider: VisualProviderKind::RenderV1,
        enabled: true,
        resolution: 64,
        channels: 8,
        blocks: 1,
    };
    e
}
fn targets<B: Backend>(
    cands: &recur64_model::model::CandidateTensors<B>,
    b: usize,
    device: &B::Device,
) -> Targets<B> {
    let mask = cands.mask.clone().float();
    let denom = mask.clone().sum_dim(1).clamp_min(1.0);
    Targets {
        policy_target: mask / denom,
        wdl_target: Tensor::<B, 1, Int>::zeros([b], device),
        wdl_mask: None,
    }
}

#[test]
fn every_gated_subsystem_receives_gradient_on_the_first_step() {
    let device = Default::default();
    let cfg = tiny_cfg();
    let e = exp();
    e.validate(cfg.width).unwrap();
    let model = ChimeraModel::<TrainB>::new(cfg.clone(), e.clone(), &device);
    let provider = provider_for_config(&e).unwrap();
    let states = probe_positions(2, 12);
    let batch = build_x15_batch::<TrainB>(&states, &e, provider.as_ref(), &device).unwrap();

    let out = model.forward_thoughts(&batch.input, &batch.cands, 4);
    assert_eq!(out.readouts.len(), 1, "final_only_v1 reads out once");
    assert_eq!(
        out.executed_blocks,
        cfg.executed_blocks_final(4),
        "executed-block accounting at T=4"
    );
    let t = targets(&batch.cands, batch.batch, &device);
    let loss = readout_loss(&out.readouts[0], &t);
    let grads = burn::optim::GradientsParams::from_grads(loss.backward(), &model);

    let norms = model.subsystem_grad_norms(&grads);
    assert_eq!(norms.len(), 8);
    for (name, norm) in &norms {
        assert!(
            norm.is_finite() && *norm > 0.0,
            "subsystem {name} received a zero or non-finite gradient: {norm}"
        );
    }
    // The whole point of the gating work: the visual and compute pathways are
    // not silently starved.
    for required in ["compute", "visual", "reasoning_latents", "gates"] {
        let (_, n) = norms.iter().find(|(k, _)| *k == required).unwrap();
        assert!(*n > 0.0, "{required} gradient is {n}");
    }
}

#[test]
fn parameter_count_is_fixed_across_thought_steps() {
    let device = Default::default();
    let cfg = tiny_cfg();
    let e = exp();
    let model = ChimeraModel::<EvalB>::new(cfg.clone(), e.clone(), &device);
    let provider = provider_for_config(&e).unwrap();
    let states = probe_positions(2, 10);
    let batch = build_x15_batch::<EvalB>(&states, &e, provider.as_ref(), &device).unwrap();

    let params = model.num_params();
    let mut last = None;
    for t in [1usize, 2, 4, 8] {
        let out = model.forward_thoughts(&batch.input, &batch.cands, t);
        assert_eq!(
            model.num_params(),
            params,
            "T={t} changed the parameter count"
        );
        assert_eq!(out.thoughts.len(), 1);
        let blocks = out.executed_blocks;
        assert_eq!(
            blocks,
            cfg.executed_blocks_final(t),
            "T={t} executed-block accounting"
        );
        if let Some(prev) = last {
            assert!(blocks > prev, "T={t} must execute more blocks than T-1");
        }
        last = Some(blocks);
    }
    assert_eq!(params, model.num_params());
}

#[test]
fn the_bank_layout_matches_the_model_contract() {
    // The model projects `[b, tokens, fields]` with one linear map, so the
    // coprocessor's packed layout and the experimental config must agree.
    let e = ExperimentalConfig::default();
    assert_eq!(
        e.compute.tokens,
        recur64_coproc::SQUARES + recur64_coproc::GLOBAL_TOKENS
    );
    assert_eq!(e.compute.fields, recur64_coproc::SQ_FIELDS);
    assert_eq!(
        recur64_coproc::OUTPUT_LEN,
        e.compute.tokens * e.compute.fields,
        "ComputeBankV1 must be exactly a uniform [tokens, fields] grid"
    );
}

#[test]
fn compute_and_visual_off_reduces_to_the_symbolic_control() {
    let device = Default::default();
    let cfg = tiny_cfg();
    // All auxiliary pathways off, one thought: the square-token transformer.
    let mut e = ExperimentalConfig::default();
    e.architecture = Architecture::ChimeraV1;
    e.thought_steps = 1;
    e.reasoning = ReasoningConfig {
        enabled: false,
        aux_width: 32,
        aux_heads: 4,
        latent_ffn: 64,
        ..ReasoningConfig::default()
    };
    e.compute = ComputeConfig {
        provider: ComputeProviderKind::None,
        enabled: false,
        ..ComputeConfig::default()
    };
    e.visual = VisualConfig {
        provider: VisualProviderKind::None,
        enabled: false,
        ..VisualConfig::default()
    };
    e.validate(cfg.width).unwrap();

    let model = ChimeraModel::<EvalB>::new(cfg, e.clone(), &device);
    // With no provider active, the host wiring must pass no compute or visual
    // tensors at all, and must do no rendering or coprocessor work.
    let provider = provider_for_config(&e).unwrap();
    assert_eq!(provider.kind(), ComputeProviderKind::None);
    let states = probe_positions(2, 10);
    let batch = build_x15_batch::<EvalB>(&states, &e, provider.as_ref(), &device).unwrap();
    assert!(
        batch.input.compute.is_none(),
        "no bank when the pathway is off"
    );
    assert!(
        batch.input.visual.is_none(),
        "no image when the pathway is off"
    );
    assert_eq!(batch.phases.compute_us, 0, "compute pathway did no work");
    assert_eq!(
        batch.phases.visual_render_us, 0,
        "visual pathway did no work"
    );

    let out = model.forward_thoughts(&batch.input, &batch.cands, 1);
    assert_eq!(out.readouts.len(), 1);
    let lp = out.readouts[0]
        .policy
        .log_probs
        .clone()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    assert!(lp.iter().all(|v| v.is_finite()), "control path is finite");
}

#[test]
fn deep_supervision_modes_read_out_per_thought() {
    let device = Default::default();
    let cfg = tiny_cfg();
    let provider = provider_for_config(&exp()).unwrap();
    let states = probe_positions(2, 8);

    for (mode, expected) in [
        (DeepSupervisionMode::FinalOnlyV1, 1usize),
        (DeepSupervisionMode::SameTargetV1, 4),
        (DeepSupervisionMode::ProgressiveSearchV1, 4),
    ] {
        let mut e = exp();
        e.deep_supervision = mode;
        let model = ChimeraModel::<EvalB>::new(cfg.clone(), e.clone(), &device);
        let batch = build_x15_batch::<EvalB>(&states, &e, provider.as_ref(), &device).unwrap();
        let out = model.forward_thoughts(&batch.input, &batch.cands, 4);
        assert_eq!(
            out.readouts.len(),
            expected,
            "{} read out the wrong number of thoughts",
            mode.label()
        );
        assert_eq!(out.thoughts.len(), expected);
    }
}

#[test]
fn native_and_wasm_providers_produce_the_same_tokens_end_to_end() {
    // The parity guarantee has to survive the host wiring, not just the raw
    // coprocessor call.
    let device: burn::tensor::Device<EvalB> = Default::default();
    let states = probe_positions(6, 20);
    let mut e = exp();
    let mut a = e.clone();
    a.compute.provider = ComputeProviderKind::NativeV1;
    e.compute.provider = ComputeProviderKind::WasmV1;

    let pa = provider_for_config(&a).unwrap();
    let pw = provider_for_config(&e).unwrap();
    let ba = build_x15_batch::<EvalB>(&states, &a, pa.as_ref(), &device).unwrap();
    let bw = build_x15_batch::<EvalB>(&states, &e, pw.as_ref(), &device).unwrap();
    assert!(ba.input.compute.is_some() && bw.input.compute.is_some());
    let x = ba
        .input
        .compute
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let y = bw
        .input
        .compute
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    assert_eq!(x, y, "native and wasm banks must be identical after wiring");

    // The visual path is provider-independent and deterministic.
    let v1 = ba
        .input
        .visual
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let v2 = bw
        .input
        .visual
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    assert_eq!(v1, v2);
}

#[test]
fn an_x15_checkpoint_is_refused_by_the_probe_loader_and_vice_versa() {
    use recur64_model::checkpoint::CheckpointMeta;

    let mut meta = CheckpointMeta::new(tiny_cfg(), 1, false, 0, 1e-4, 1, 0, "cpu", "fp32");
    // A fresh (probe) checkpoint is refused as X15.
    assert!(meta.check_architecture(Architecture::ProbeV1).is_ok());
    assert!(
        meta.check_architecture(Architecture::ChimeraV1).is_err(),
        "an F15/R15 checkpoint must never load as X15"
    );

    let e = exp();
    meta = meta.with_experimental(e.clone());
    assert_eq!(meta.architecture, Architecture::ChimeraV1);
    assert_eq!(
        meta.head_version,
        recur64_model::experimental::CHIMERA_HEAD_VERSION
    );
    assert!(meta.check_architecture(Architecture::ChimeraV1).is_ok());
    assert!(meta.check_architecture(Architecture::ProbeV1).is_err());
    assert!(meta.check_contracts().is_ok());
    assert!(meta.check_experimental(&e).is_ok());
    let mut other = e.clone();
    other.thought_steps = 2;
    assert!(
        meta.check_experimental(&other).is_err(),
        "a different thought-step count must be refused"
    );
}

#[test]
fn a_chimera_autodiff_backend_survives_a_real_optimizer_step() {
    use burn::optim::Optimizer;

    let device = Default::default();
    let cfg = tiny_cfg();
    let e = exp();
    let model = ChimeraModel::<TrainB>::new(cfg, e.clone(), &device);
    let mut optim = recur64_model::train::adamw::<TrainB, _>();
    let provider = provider_for_config(&e).unwrap();
    let states = probe_positions(2, 12);
    let batch = build_x15_batch::<TrainB>(&states, &e, provider.as_ref(), &device).unwrap();
    let t = targets(&batch.cands, batch.batch, &device);

    let before = model.num_params();
    let out = model.forward_thoughts(&batch.input, &batch.cands, 2);
    let loss = readout_loss(&out.readouts[0], &t);
    let loss_value: f32 = loss.clone().into_scalar().elem();
    let grads = burn::optim::GradientsParams::from_grads(loss.backward(), &model);
    let model = optim.step(1e-3, model, grads);
    assert!(
        loss_value.is_finite(),
        "loss must be finite, got {loss_value}"
    );
    assert_eq!(model.num_params(), before);

    // The step really moved the parameters.
    let out2 = model.forward_thoughts(&batch.input, &batch.cands, 2);
    let loss2: f32 = readout_loss(&out2.readouts[0], &t).into_scalar().elem();
    assert!(loss2.is_finite());
    assert_ne!(
        loss_value.to_bits(),
        loss2.to_bits(),
        "an optimizer step must change the loss on a fixed batch"
    );
}

#[test]
fn unused_imports_are_intentional_here() {
    // Keeps the helper import honest if the tests above change shape.
    fn _assert_device<B: Backend>(_: &B::Device) {}
    let _ = _assert_device::<EvalB>;
}
