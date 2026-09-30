//! Chimera V2 contract tests: one-shot board encoding, bounded planner, tool
//! schedule, exact world-model alignment, checkpoint refusal, masking.

#![allow(clippy::field_reassign_with_default)]

use burn::prelude::*;

use recur64_compute::NativeWorldModel;
use recur64_coproc::ComputeProviderKind;
use recur64_coproc::world::{REPLY_BYTES, SUCC_BYTES, reply_offset, succ_offset, world_output_len};
use recur64_core::{GameState, StandardMove};
use recur64_model::chimera2::{ChimeraV2Model, V2Options};
use recur64_model::config::ModelConfig;
use recur64_model::experimental::{
    Architecture, CandidateFactsConfig, CandidateFactsProviderKind, ChimeraV2Config,
    ExperimentalConfig, InfoSchedule, VisualConfig, VisualProviderKind,
};
use recur64_runtime::v2_inputs::{build_v2_batch, compute_world_bytes, root_facts_from_bytes};

type EvalB = burn::backend::Flex;
type TrainB = burn::backend::Autodiff<burn::backend::Flex>;

fn tiny_cfg() -> ModelConfig {
    ModelConfig {
        width: 32,
        heads: 4,
        ffn: 48,
        input_blocks: 1,
        core_blocks: 1,
        output_blocks: 0,
        squares: 64,
        in_features: 119,
        policy_dim: 8,
        wdl_classes: 3,
        promo_codes: 5,
        rms_eps: 1e-5,
    }
}

fn exp(schedule: InfoSchedule) -> ExperimentalConfig {
    let mut e = ExperimentalConfig::default();
    e.architecture = Architecture::ChimeraV2;
    e.v2 = ChimeraV2Config {
        cand_dim: 32,
        planner_dim: 32,
        planner_heads: 4,
        planner_ffn: 64,
        workspace_tokens: 4,
        reply_dim: 32,
        succ_hidden: 8,
        max_thoughts: 8,
        info_schedule: schedule,
        world_provider: ComputeProviderKind::NativeV1,
        w_cap: 64,
        r_cap: 12,
        fact_shortcut: false,
    };
    e.candidate_facts = CandidateFactsConfig {
        provider: CandidateFactsProviderKind::NativeV1,
        enabled: true,
        gain: 1.0,
        hidden: 8,
    };
    e.visual = VisualConfig {
        provider: VisualProviderKind::RenderV1,
        enabled: false,
        resolution: 64,
        channels: 4,
        blocks: 1,
    };
    e
}

fn positions() -> Vec<GameState> {
    [
        "6k1/5ppp/8/8/8/8/8/R6K w - - 0 1",
        "k7/8/1K6/8/8/8/8/6R1 w - - 0 1",
        "8/8/8/4k3/8/3K4/3Q4/8 w - - 0 1",
        "4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1",
        "7k/4P3/8/8/8/8/8/K7 w - - 0 1",
        "3k4/8/3K4/8/8/8/8/2Q5 w - - 0 1",
    ]
    .iter()
    .map(|f| GameState::from_fen(f).unwrap())
    .collect()
}

fn logits(out: &recur64_model::chimera2::ChimeraV2Output<EvalB>, i: usize) -> Vec<f32> {
    out.readouts[i]
        .policy
        .log_probs
        .clone()
        .into_data()
        .to_vec::<f32>()
        .unwrap()
}

fn vec1(t: Tensor<EvalB, 1>) -> Vec<f32> {
    t.into_data().to_vec::<f32>().unwrap()
}

#[test]
fn the_board_encoder_runs_exactly_once_at_every_thought_budget() {
    let device = Default::default();
    let e = exp(InfoSchedule::Progressive);
    let model = ChimeraV2Model::<EvalB>::new(tiny_cfg(), e.clone(), &device);
    let states = positions();
    let batch =
        build_v2_batch::<EvalB>(&states, &e, Some(&NativeWorldModel), None, &device).unwrap();
    let mut params = None;
    for t in 1..=4usize {
        let before = model.board_encoder_runs();
        let out = model.forward(&batch.input, &batch.cands, t, V2Options::default());
        assert_eq!(
            model.board_encoder_runs() - before,
            1,
            "T={t}: board encoder runs"
        );
        assert_eq!(out.counts.board_encoder_runs, 1);
        assert_eq!(out.counts.planner_steps, t);
        assert_eq!(out.readouts.len(), 1, "final-only readout");
        assert_eq!(out.diag.len(), t);
        // Parameter count does not depend on T.
        let n = model.num_params();
        assert_eq!(*params.get_or_insert(n), n);
    }
}

#[test]
fn planner_state_stays_bounded_through_eight_thoughts() {
    let device = Default::default();
    let e = exp(InfoSchedule::Progressive);
    let model = ChimeraV2Model::<EvalB>::new(tiny_cfg(), e.clone(), &device);
    let states = positions();
    let batch =
        build_v2_batch::<EvalB>(&states, &e, Some(&NativeWorldModel), None, &device).unwrap();
    let out = model.forward(
        &batch.input,
        &batch.cands,
        8,
        V2Options { all_readouts: true },
    );
    assert_eq!(out.readouts.len(), 8);
    for d in &out.diag {
        let (m, r) = (vec1(d.mean_abs.clone()), vec1(d.rms.clone()));
        for (a, b) in m.iter().zip(&r) {
            assert!(a.is_finite() && b.is_finite());
            assert!(
                *b > 0.3 && *b < 2.5,
                "thought {}: RMS(Z) = {b} (mean |Z| {a})",
                d.thought
            );
        }
        let g = vec1(d.gate_mean.clone());
        assert!(
            g.iter().all(|x| *x > 0.0 && *x < 1.0),
            "gate in (0,1): {g:?}"
        );
    }
    // No monotone growth: the last thought is not larger than the first by a wide margin.
    let first = vec1(out.diag[0].rms.clone());
    let last = vec1(out.diag[7].rms.clone());
    for (a, b) in first.iter().zip(&last) {
        assert!(*b < *a * 2.0, "RMS grew from {a} to {b}");
    }
}

/// Independent expectation of a packed child board from the child GameState.
fn expected_child_board(child: &GameState) -> Vec<u8> {
    let fen = child.to_fen();
    let mut it = fen.split(' ');
    let placement = it.next().unwrap();
    let black_to_move = it.next().unwrap() == "b";
    let mut grid = [0u8; 64];
    for (row, rank_str) in placement.split('/').enumerate() {
        let rank = 7 - row;
        let mut file = 0usize;
        for ch in rank_str.chars() {
            if let Some(d) = ch.to_digit(10) {
                file += d as usize;
            } else {
                let idx = match ch.to_ascii_lowercase() {
                    'p' => 0,
                    'n' => 1,
                    'b' => 2,
                    'r' => 3,
                    'q' => 4,
                    _ => 5,
                };
                let white_piece = ch.is_ascii_uppercase();
                let own = white_piece != black_to_move;
                grid[rank * 8 + file] = if own { 1 + idx } else { 7 + idx };
                file += 1;
            }
        }
    }
    (0..64)
        .map(|c| grid[if black_to_move { c ^ 56 } else { c }])
        .collect()
}

fn child_of(state: &GameState, id: recur64_core::ActionId) -> GameState {
    let (from, to, promo) = id.to_physical(state.perspective());
    let promotion = if promo.is_none() { None } else { Some(promo) };
    let mut s = state.clone();
    s.apply(StandardMove::new(from, to, promotion)).unwrap();
    s
}

#[test]
fn successors_reconstruct_the_children_and_replies_are_enumerated_exactly_once() {
    let (w_cap, r_cap) = (64, 48);
    let states = positions();
    let bytes = compute_world_bytes(&states, &NativeWorldModel, w_cap, r_cap).unwrap();
    for (s, out) in states.iter().zip(&bytes) {
        assert_eq!(out.len(), world_output_len(w_cap, r_cap));
        let legal = s.legal_actions();
        assert_eq!(u16::from_le_bytes([out[0], out[1]]) as usize, legal.len());
        for (ci, id) in legal.iter().enumerate() {
            let child = child_of(s, *id);
            let base = succ_offset(w_cap) + ci * SUCC_BYTES;
            assert_eq!(
                &out[base + 4..base + 68],
                expected_child_board(&child).as_slice(),
                "candidate {ci}: successor placement"
            );
            let child_legal = if child.is_terminal() {
                0
            } else {
                child.legal_actions().len()
            };
            assert_eq!(
                out[base + 2] as usize,
                child_legal,
                "candidate {ci}: reply count"
            );
            let valid: usize = (0..r_cap)
                .map(|ri| out[reply_offset(w_cap) + (ci * r_cap + ri) * REPLY_BYTES] as usize)
                .sum();
            assert_eq!(
                valid, child_legal,
                "candidate {ci}: every reply exactly once"
            );
        }
    }
}

#[test]
fn world_root_facts_equal_the_native_candidate_facts_v1_and_follow_legal_order() {
    let (w_cap, r_cap) = (64, 12);
    let states = positions();
    let bytes = compute_world_bytes(&states, &NativeWorldModel, w_cap, r_cap).unwrap();
    let native = recur64_runtime::candidate_facts::candidate_facts(&states, w_cap).unwrap();
    let fields = recur64_model::experimental::CANDIDATE_FACT_FIELDS;
    for (i, out) in bytes.iter().enumerate() {
        let from_world = root_facts_from_bytes(out, w_cap);
        let want = &native[i * w_cap * fields..(i + 1) * w_cap * fields];
        assert_eq!(from_world.len(), want.len());
        for (k, (a, b)) in from_world.iter().zip(want).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "position {i} value {k}: world {a} vs candidate_facts {b}"
            );
        }
    }
}

fn scrambled_world_input(
    model: &ChimeraV2Model<EvalB>,
    e: &ExperimentalConfig,
    states: &[GameState],
    which: &str,
) -> recur64_model::chimera2::ChimeraV2Input<EvalB> {
    let device = Default::default();
    let _ = model;
    let mut batch =
        build_v2_batch::<EvalB>(states, e, Some(&NativeWorldModel), None, &device).unwrap();
    let w = batch.input.world.as_mut().unwrap();
    match which {
        "succ" => {
            w.succ_board = w.succ_board.clone().flip([3]);
            w.succ_flags = w.succ_flags.clone().add_scalar(0.7);
        }
        "reply" => {
            w.reply_feats = w.reply_feats.clone().add_scalar(0.9);
        }
        _ => unreachable!(),
    }
    batch.input
}

#[test]
fn each_thought_sees_exactly_the_intended_tokens() {
    let device = Default::default();
    let e = exp(InfoSchedule::Progressive);
    let model = ChimeraV2Model::<EvalB>::new(tiny_cfg(), e.clone(), &device);
    let states = positions();
    let base =
        build_v2_batch::<EvalB>(&states, &e, Some(&NativeWorldModel), None, &device).unwrap();
    let alt_s = scrambled_world_input(&model, &e, &states, "succ");
    let alt_r = scrambled_world_input(&model, &e, &states, "reply");
    let run = |input: &recur64_model::chimera2::ChimeraV2Input<EvalB>, t: usize| {
        logits(
            &model.forward(input, &base.cands, t, V2Options::default()),
            0,
        )
    };
    // Thought 1: root only. Changing successor or reply tensors changes nothing.
    assert_eq!(run(&base.input, 1), run(&alt_s, 1));
    assert_eq!(run(&base.input, 1), run(&alt_r, 1));
    // Thought 2: successors are visible, replies are not.
    assert_ne!(
        run(&base.input, 2),
        run(&alt_s, 2),
        "T=2 must see successors"
    );
    assert_eq!(
        run(&base.input, 2),
        run(&alt_r, 2),
        "T=2 must NOT see replies"
    );
    // Thought 3: replies are now visible too.
    assert_ne!(run(&base.input, 3), run(&alt_r, 3), "T=3 must see replies");
    // Thought 4 reveals nothing new (still sees both); counts prove no extra encoding.
    let c3 = model
        .forward(&base.input, &base.cands, 3, V2Options::default())
        .counts;
    let c4 = model
        .forward(&base.input, &base.cands, 4, V2Options::default())
        .counts;
    assert_eq!(c3.successor_encodes, c4.successor_encodes);
    assert_eq!(c3.reply_set_encodes, c4.reply_set_encodes);
    assert_eq!(c4.planner_steps, 4);
}

#[test]
fn the_root_only_control_is_independent_of_every_tool_token() {
    let device = Default::default();
    let e = exp(InfoSchedule::RootOnly);
    let model = ChimeraV2Model::<EvalB>::new(tiny_cfg(), e.clone(), &device);
    let states = positions();
    let base = build_v2_batch::<EvalB>(&states, &e, None, None, &device).unwrap();
    assert!(
        base.input.world.is_none(),
        "no world model is even computed"
    );
    let out = model.forward(&base.input, &base.cands, 4, V2Options::default());
    assert_eq!(out.counts.successor_encodes, 0);
    assert_eq!(out.counts.reply_set_encodes, 0);
    assert!(logits(&out, 0).iter().all(|v| v.is_finite()));
}

#[test]
fn all_info_at_one_step_contains_the_same_content_as_progressive_at_three() {
    let device = Default::default();
    let ep = exp(InfoSchedule::Progressive);
    let ea = exp(InfoSchedule::AllAtOnce);
    let mp = ChimeraV2Model::<EvalB>::new(tiny_cfg(), ep.clone(), &device);
    let ma = ChimeraV2Model::<EvalB>::new(tiny_cfg(), ea.clone(), &device);
    assert_eq!(
        mp.num_params(),
        ma.num_params(),
        "same trainable parameter set"
    );
    let states = positions();
    let bp = build_v2_batch::<EvalB>(&states, &ep, Some(&NativeWorldModel), None, &device).unwrap();
    let ba = build_v2_batch::<EvalB>(&states, &ea, Some(&NativeWorldModel), None, &device).unwrap();
    let cp = mp
        .forward(&bp.input, &bp.cands, 3, V2Options::default())
        .counts;
    let ca = ma
        .forward(&ba.input, &ba.cands, 1, V2Options::default())
        .counts;
    assert_eq!(cp.successor_encodes, ca.successor_encodes);
    assert_eq!(cp.reply_set_encodes, ca.reply_set_encodes);
    assert_eq!(ca.planner_steps, 1);
    // The unpacked tool tensors are byte-identical inputs.
    let (wp, wa) = (
        bp.input.world.as_ref().unwrap(),
        ba.input.world.as_ref().unwrap(),
    );
    assert_eq!(
        wp.reply_feats.clone().into_data().to_vec::<f32>().unwrap(),
        wa.reply_feats.clone().into_data().to_vec::<f32>().unwrap()
    );
}

#[test]
fn reply_padding_and_masks_do_not_leak_into_the_output() {
    let device = Default::default();
    let e = exp(InfoSchedule::AllAtOnce);
    let model = ChimeraV2Model::<EvalB>::new(tiny_cfg(), e.clone(), &device);
    let states = positions();
    let base =
        build_v2_batch::<EvalB>(&states, &e, Some(&NativeWorldModel), None, &device).unwrap();
    let clean = logits(
        &model.forward(&base.input, &base.cands, 1, V2Options::default()),
        0,
    );
    // Put garbage into every padded reply slot (mask == 0); nothing may change.
    let mut dirty =
        build_v2_batch::<EvalB>(&states, &e, Some(&NativeWorldModel), None, &device).unwrap();
    let w = dirty.input.world.as_mut().unwrap();
    let mask = w.reply_mask.clone().unsqueeze_dim::<4>(3);
    let noise = w.reply_feats.clone().ones_like().mul_scalar(3.7);
    w.reply_feats = w.reply_feats.clone() + noise * mask.neg().add_scalar(1.0);
    let noisy = logits(
        &model.forward(&dirty.input, &dirty.cands, 1, V2Options::default()),
        0,
    );
    for (a, b) in clean.iter().zip(&noisy) {
        assert!(
            (a - b).abs() < 1e-5,
            "masked reply padding leaked: {a} vs {b}"
        );
    }
}

#[test]
fn visual_off_does_not_execute_the_cnn_and_facts_params_matter() {
    let device = Default::default();
    let e = exp(InfoSchedule::Progressive);
    let model = ChimeraV2Model::<EvalB>::new(tiny_cfg(), e.clone(), &device);
    let states = positions();
    let base =
        build_v2_batch::<EvalB>(&states, &e, Some(&NativeWorldModel), None, &device).unwrap();
    assert!(
        base.input.visual.is_none(),
        "no image is rendered when visual is off"
    );
    // Scrambling the (unused) visual parameters changes nothing.
    struct Scramble {
        stack: Vec<String>,
        names: Vec<&'static str>,
    }
    impl<B: Backend> burn::module::ModuleMapper<B> for Scramble {
        fn enter_module(&mut self, name: &str, _: &str) {
            self.stack.push(name.to_string());
        }
        fn exit_module(&mut self, _: &str, _: &str) {
            self.stack.pop();
        }
        fn map_float<const D: usize>(
            &mut self,
            param: burn::module::Param<Tensor<B, D>>,
        ) -> burn::module::Param<Tensor<B, D>> {
            if !self.stack.iter().any(|n| self.names.contains(&n.as_str())) {
                return param;
            }
            let (id, t, mapper) = param.consume();
            let noise = t.random_like(burn::tensor::Distribution::Normal(0.0, 1.0));
            burn::module::Param::from_mapped_value(id, noise, mapper)
        }
    }
    let before = logits(
        &model.forward(&base.input, &base.cands, 3, V2Options::default()),
        0,
    );
    let mut vis = Scramble {
        stack: vec![],
        names: vec!["visual", "visual_gate"],
    };
    let m2 = model.clone().map(&mut vis);
    assert_eq!(
        before,
        logits(
            &m2.forward(&base.input, &base.cands, 3, V2Options::default()),
            0
        )
    );
    let mut facts = Scramble {
        stack: vec![],
        names: vec!["facts_enc"],
    };
    let m3 = model.clone().map(&mut facts);
    assert_ne!(
        before,
        logits(
            &m3.forward(&base.input, &base.cands, 3, V2Options::default()),
            0
        )
    );
}

#[test]
fn v2_checkpoints_are_refused_by_v1_and_probe_loaders_and_vice_versa() {
    big_stack(v2_checkpoint_refusal_body);
}

fn v2_checkpoint_refusal_body() {
    use recur64_model::checkpoint::{CheckpointMeta, save_training_v2};
    use recur64_runtime::model_io;
    let device = Default::default();
    let cfg = tiny_cfg();
    let e = exp(InfoSchedule::Progressive);
    let model = ChimeraV2Model::<TrainB>::new(cfg.clone(), e.clone(), &device);
    let optim = recur64_model::train::adamw::<TrainB, _>();
    let dir = std::env::temp_dir().join(format!("recur64-v2-ckpt-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let meta = CheckpointMeta::new(cfg.clone(), 1, false, 0, 1e-4, 1, 0, "test", "fp32")
        .with_experimental(e.clone());
    save_training_v2(&dir, &model, &optim, &meta).unwrap();

    // V2 loads its own.
    assert!(model_io::load_chimera_v2::<EvalB>(&dir, &cfg, &e, &device).is_ok());
    // Probe and V1 loaders refuse it.
    assert!(model_io::load_unverified::<EvalB>(&dir, &cfg, &device).is_err());
    let mut v1 = ExperimentalConfig::default();
    v1.architecture = Architecture::ChimeraV1;
    assert!(model_io::load_chimera::<EvalB>(&dir, &cfg, &v1, &device).is_err());
    // A different V2 contract is refused too.
    let mut other = e.clone();
    other.v2.info_schedule = InfoSchedule::AllAtOnce;
    assert!(model_io::load_chimera_v2::<EvalB>(&dir, &cfg, &other, &device).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_probe_config_and_a_v1_config_are_refused_by_the_v2_builder() {
    let device = Default::default();
    let cfg = tiny_cfg();
    assert!(
        recur64_runtime::model_io::build_chimera_v2::<EvalB>(
            &cfg,
            &ExperimentalConfig::default(),
            &device
        )
        .is_err()
    );
    let mut v1 = ExperimentalConfig::default();
    v1.architecture = Architecture::ChimeraV1;
    assert!(recur64_runtime::model_io::build_chimera_v2::<EvalB>(&cfg, &v1, &device).is_err());
}

#[test]
fn a_tiny_v2_model_takes_an_optimizer_step_and_gradients_reach_every_subsystem() {
    big_stack(optimizer_step_body);
}

fn optimizer_step_body() {
    use burn::optim::{GradientsParams, Optimizer};
    use recur64_model::loss::{Targets, readout_loss};
    let device = Default::default();
    let e = exp(InfoSchedule::Progressive);
    let mut model = ChimeraV2Model::<TrainB>::new(tiny_cfg(), e.clone(), &device);
    let mut optim = recur64_model::train::adamw::<TrainB, _>();
    let states = positions();
    let batch =
        build_v2_batch::<TrainB>(&states, &e, Some(&NativeWorldModel), None, &device).unwrap();
    let mask = batch.cands.mask.clone().float();
    let denom = mask.clone().sum_dim(1).clamp_min(1.0);
    let t = Targets {
        policy_target: mask / denom,
        wdl_target: Tensor::<TrainB, 1, Int>::zeros([states.len()], &device),
        wdl_mask: None,
    };
    let mut first = f32::NAN;
    let mut last = f32::NAN;
    for step in 0..15 {
        let out = model.forward(&batch.input, &batch.cands, 3, V2Options::default());
        let loss = readout_loss(&out.readouts[0], &t);
        let l: f32 = loss.clone().into_scalar().elem();
        assert!(l.is_finite(), "non-finite loss at step {step}");
        if step == 0 {
            first = l;
        }
        last = l;
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optim.step(1e-3, model, grads);
    }
    assert!(last < first, "loss did not fall: {first} -> {last}");
}

/// Run `f` on a thread with a large stack: the generated Burn record types for a
/// model this size are deep, and debug-build test threads only get 2 MB.
fn big_stack<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(f)
        .unwrap()
        .join()
        .unwrap()
}
