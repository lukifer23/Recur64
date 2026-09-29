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

/// Fields that only the latent / compute / visual pathways own. The symbolic
/// control must be a function of none of them.
const AUX_FIELDS: &[&str] = &[
    "latent_emb",
    "latent_init",
    "latent_norm",
    "latent_ffn1",
    "latent_ffn2",
    "xattn_sq",
    "xattn_compute",
    "xattn_visual",
    "feedback",
    "gate_compute_logit",
    "gate_visual_logit",
    "gate_reason_logit",
    "compute_proj",
    "compute_type_emb",
    "visual",
    "wdl_from_latent",
];

/// Replaces every auxiliary parameter with N(0, 1) noise.
struct RandomizeAux {
    stack: Vec<String>,
    touched: usize,
}

impl<B: Backend> burn::module::ModuleMapper<B> for RandomizeAux {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.stack.push(name.to_string());
    }
    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.stack.pop();
    }
    fn map_float<const D: usize>(
        &mut self,
        param: burn::module::Param<Tensor<B, D>>,
    ) -> burn::module::Param<Tensor<B, D>> {
        if !self.stack.iter().any(|n| AUX_FIELDS.contains(&n.as_str())) {
            return param;
        }
        self.touched += 1;
        let (id, t, mapper) = param.consume();
        let noise = t.random_like(burn::tensor::Distribution::Normal(0.0, 1.0));
        burn::module::Param::from_mapped_value(id, noise, mapper)
    }
}

fn symbolic_only_exp() -> ExperimentalConfig {
    let mut e = exp();
    e.thought_steps = 1;
    e.reasoning.enabled = false;
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
    e
}

fn readout_vecs(r: &recur64_model::model::Readout<EvalB>) -> (Vec<f32>, Vec<f32>) {
    (
        r.policy
            .log_probs
            .clone()
            .into_data()
            .to_vec::<f32>()
            .unwrap(),
        r.wdl_logits.clone().into_data().to_vec::<f32>().unwrap(),
    )
}

fn metric_vec(t: Tensor<EvalB, 1>) -> Vec<f32> {
    t.into_data().to_vec::<f32>().unwrap()
}

#[test]
fn symbolic_only_control_is_independent_of_every_auxiliary_parameter() {
    let device = Default::default();
    let cfg = tiny_cfg();
    let e = symbolic_only_exp();
    e.validate(cfg.width).unwrap();

    let model = ChimeraModel::<EvalB>::new(cfg.clone(), e.clone(), &device);
    // No provider is active, so the host wiring passes no compute/visual input
    // and does no coprocessor or rendering work.
    let provider = provider_for_config(&e).unwrap();
    assert_eq!(provider.kind(), ComputeProviderKind::None);
    let states = probe_positions(2, 10);
    let batch = build_x15_batch::<EvalB>(&states, &e, provider.as_ref(), &device).unwrap();
    assert!(batch.input.compute.is_none() && batch.input.visual.is_none());
    assert_eq!(batch.phases.compute_us, 0);
    assert_eq!(batch.phases.visual_render_us, 0);

    let base = model.forward_thoughts(&batch.input, &batch.cands, 1);
    assert_eq!(base.readouts.len(), 1);
    assert_eq!(base.thoughts.len(), 1);
    let (lp0, wdl0) = readout_vecs(&base.readouts[0]);
    assert!(lp0.iter().chain(&wdl0).all(|v| v.is_finite()));
    // One core pass only: no extra blocks beyond the T=1 accounting.
    assert_eq!(base.executed_blocks, cfg.executed_blocks_final(1));

    // Randomize every auxiliary parameter (including the zero-init
    // `wdl_from_latent`); the control's output must be bit-identical.
    let mut mapper = RandomizeAux {
        stack: Vec::new(),
        touched: 0,
    };
    let scrambled = model.clone().map(&mut mapper);
    assert!(
        mapper.touched > 20,
        "the mapper must actually reach the auxiliary parameters (touched {})",
        mapper.touched
    );
    let out = scrambled.forward_thoughts(&batch.input, &batch.cands, 1);
    let (lp1, wdl1) = readout_vecs(&out.readouts[0]);
    assert_eq!(lp0, lp1, "policy depends on an auxiliary parameter");
    assert_eq!(wdl0, wdl1, "WDL depends on an auxiliary parameter");

    // The check has teeth: with the latents ON, the same scrambling changes
    // the output.
    let mut full_exp = exp();
    full_exp.thought_steps = 2;
    let full = ChimeraModel::<EvalB>::new(cfg, full_exp.clone(), &device);
    let full_provider = provider_for_config(&full_exp).unwrap();
    let full_batch =
        build_x15_batch::<EvalB>(&states, &full_exp, full_provider.as_ref(), &device).unwrap();
    let a = full.forward_thoughts(&full_batch.input, &full_batch.cands, 2);
    let mut mapper = RandomizeAux {
        stack: Vec::new(),
        touched: 0,
    };
    let b = full
        .clone()
        .map(&mut mapper)
        .forward_thoughts(&full_batch.input, &full_batch.cands, 2);
    assert_ne!(
        readout_vecs(&a.readouts[0]),
        readout_vecs(&b.readouts[0]),
        "scrambling auxiliary parameters must change a latent-enabled model"
    );
}

#[test]
fn compute_or_visual_without_reasoning_is_refused() {
    let cfg = tiny_cfg();
    let mut e = symbolic_only_exp();
    e.compute = ComputeConfig {
        provider: ComputeProviderKind::NativeV1,
        enabled: true,
        ..ComputeConfig::default()
    };
    assert!(e.validate(cfg.width).is_err());
    let mut e = symbolic_only_exp();
    e.visual = VisualConfig {
        provider: VisualProviderKind::RenderV1,
        enabled: true,
        ..VisualConfig::default()
    };
    assert!(e.validate(cfg.width).is_err());
}

#[test]
fn diagnostic_forward_reads_every_thought_without_changing_the_network() {
    let device = Default::default();
    let cfg = tiny_cfg();
    let e = exp();
    assert_eq!(e.deep_supervision, DeepSupervisionMode::FinalOnlyV1);
    let model = ChimeraModel::<EvalB>::new(cfg.clone(), e.clone(), &device);
    let identity = model.experimental().identity().unwrap();
    let provider = provider_for_config(&e).unwrap();
    let states = probe_positions(3, 12);
    let batch = build_x15_batch::<EvalB>(&states, &e, provider.as_ref(), &device).unwrap();

    let normal = model.forward_thoughts(&batch.input, &batch.cands, 4);
    assert_eq!(normal.readouts.len(), 1, "final_only_v1 supervises once");
    assert_eq!(normal.thoughts.len(), 1);

    let diag = model.forward_thoughts_diagnostic(&batch.input, &batch.cands, 4);
    assert_eq!(diag.readouts.len(), 4);
    assert_eq!(diag.thoughts.len(), 4);
    assert_eq!(
        diag.executed_blocks,
        cfg.executed_blocks_final(4) + cfg.output_blocks * 3,
        "diagnostic accounting includes the extra output blocks"
    );
    // Same weights, same final function.
    assert_eq!(
        readout_vecs(&normal.readouts[0]),
        readout_vecs(&diag.readouts[3]),
        "the diagnostic final readout must equal the normal one"
    );
    // Identity is a property of the config, which the diagnostic path cannot
    // touch.
    assert_eq!(identity, model.experimental().identity().unwrap());

    // Consecutive-thought metrics: none on row 0, finite and non-negative
    // afterwards.
    let m0 = &diag.thoughts[0];
    assert!(
        metric_vec(m0.policy_kl_prev.clone())
            .iter()
            .all(|v| *v == 0.0)
    );
    assert!(metric_vec(m0.wdl_l1_prev.clone()).iter().all(|v| *v == 0.0));
    for m in &diag.thoughts[1..] {
        assert!(
            metric_vec(m.policy_kl_prev.clone())
                .iter()
                .all(|v| v.is_finite() && *v >= -1e-6)
        );
        assert!(
            metric_vec(m.wdl_l1_prev.clone())
                .iter()
                .all(|v| v.is_finite() && *v >= 0.0)
        );
    }

    // T1 diagnostic == T1 normal.
    let n1 = model.forward_thoughts(&batch.input, &batch.cands, 1);
    let d1 = model.forward_thoughts_diagnostic(&batch.input, &batch.cands, 1);
    assert_eq!(d1.readouts.len(), 1);
    assert_eq!(readout_vecs(&n1.readouts[0]), readout_vecs(&d1.readouts[0]));
}

#[test]
fn every_validating_visual_resolution_renders_and_encodes() {
    let device = Default::default();
    let cfg = tiny_cfg();
    let mut accepted = 0;
    for resolution in [32usize, 64, 96, 128, 192] {
        let mut e = exp();
        e.thought_steps = 1;
        e.visual.resolution = resolution;
        if e.validate(cfg.width).is_err() {
            continue;
        }
        accepted += 1;
        // Renderable AND encodable: build the batch (renderer), then run the
        // forward pass (encoder).
        let model = ChimeraModel::<EvalB>::new(cfg.clone(), e.clone(), &device);
        let provider = provider_for_config(&e).unwrap();
        let states = probe_positions(1, 6);
        let batch = build_x15_batch::<EvalB>(&states, &e, provider.as_ref(), &device)
            .unwrap_or_else(|err| {
                panic!("resolution {resolution} validates but does not render: {err}")
            });
        let out = model.forward_thoughts(&batch.input, &batch.cands, 1);
        assert_eq!(out.readouts.len(), 1);
    }
    assert!(accepted >= 1, "at least the X1 resolution must validate");
}

#[test]
fn thought_loss_matches_each_supervision_mode_and_refuses_diagnostic_readouts() {
    use recur64_model::loss::thought_loss;
    let device = Default::default();
    let cfg = tiny_cfg();
    for (mode, ladder) in [
        (DeepSupervisionMode::FinalOnlyV1, 1usize),
        (DeepSupervisionMode::SameTargetV1, 1),
        (DeepSupervisionMode::ProgressiveSearchV1, 3),
    ] {
        let mut e = exp();
        e.thought_steps = 3;
        e.deep_supervision = mode;
        let model = ChimeraModel::<EvalB>::new(cfg.clone(), e.clone(), &device);
        let provider = provider_for_config(&e).unwrap();
        let states = probe_positions(2, 10);
        let batch = build_x15_batch::<EvalB>(&states, &e, provider.as_ref(), &device).unwrap();
        let tg: Vec<_> = (0..ladder)
            .map(|_| targets(&batch.cands, batch.batch, &device))
            .collect();

        let out = model.forward_thoughts(&batch.input, &batch.cands, 3);
        let loss = thought_loss(&out.readouts, &tg, mode, e.intermediate_weight, 3)
            .unwrap_or_else(|err| panic!("{}: {err}", mode.label()));
        assert!(loss.into_scalar().is_finite());

        // A diagnostic forward has 3 readouts even under final_only_v1, which
        // supervises 1: it must be refused, not silently summed.
        if mode == DeepSupervisionMode::FinalOnlyV1 {
            let diag = model.forward_thoughts_diagnostic(&batch.input, &batch.cands, 3);
            assert!(thought_loss(&diag.readouts, &tg, mode, e.intermediate_weight, 3).is_err());
        }
    }
}
