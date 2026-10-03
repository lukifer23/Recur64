use burn::prelude::*;
use recur64_core::GameState;
use recur64_v5::config::V5Config;
use recur64_v5::graph::{AcquiredGraph, EpisodeKey, Schedule, acquire};
use recur64_v5::model::{CounterfactualRelationalLoop, Treatment, V5Inputs};

type B = burn::backend::Flex;

static RNG: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn root() -> GameState {
    GameState::from_fen("6k1/8/8/8/8/8/4Q3/3RK3 w - - 0 1").unwrap()
}

fn graph(root: &GameState, q: usize) -> AcquiredGraph {
    acquire(
        root,
        EpisodeKey {
            position_id: "model-invariant-fixture".into(),
            schedule: Schedule::UniformFrontierV1,
            run_seed: 5301,
            occurrence_ordinal: 0,
        },
        q,
        None,
    )
    .unwrap()
}

fn values<const D: usize>(x: Tensor<B, D>) -> Vec<f32> {
    x.into_data().to_vec().unwrap()
}

fn remap_reverse(g: &AcquiredGraph) -> AcquiredGraph {
    let n = g.nodes.len();
    let mut out = g.clone();
    out.nodes.reverse();
    for node in &mut out.nodes {
        let old = node.storage_id as usize;
        node.storage_id = (n - 1 - old) as u32;
        node.parent = node.parent.map(|p| (n - 1 - p as usize) as u32);
    }
    out.digest.clear();
    out.digest = out.compute_digest().unwrap();
    out.verify().unwrap();
    out
}

#[test]
fn frozen_model_size_q0_and_paired_null_contracts_hold() {
    let _guard = RNG.lock().unwrap_or_else(|e| e.into_inner());
    let device = Default::default();
    <B as Backend>::seed(&device, 5301);
    let model = CounterfactualRelationalLoop::<B>::new(V5Config::default(), &device);
    let total = model.num_params();
    println!(
        "V5_MODEL_IDENTITY {}",
        serde_json::json!({
            "config_digest": model.config().scientific_digest().unwrap(),
            "parameters": total, "parameter_breakdown": model.param_breakdown(),
        })
    );
    assert!((6_000_000..=8_000_000).contains(&total), "{total}");

    let root = root();
    let graph = graph(&root, 2);
    let input = V5Inputs::<B>::from_examples(&[(&root, &graph)], &device).unwrap();
    let base = model.base(&input);
    assert_eq!(base.z0.dims(), [1, root.legal_actions().len()]);

    let out = model.paired(&input, 2, Treatment::AllPayloadNull);
    assert!(values(out.raw_delta).into_iter().all(|x| x == 0.0));
    assert!(values(out.centered_delta).into_iter().all(|x| x == 0.0));
    assert_eq!(values(out.logits), values(out.z0));

    // Explicit-mask-dtype parity with the previous FP32 centering operation.
    let normal = model.paired(&input, 2, Treatment::Normal);
    let valid = input.cands.mask.clone().float();
    let count = valid.clone().sum_dim(1).clamp(1.0, f32::MAX);
    let mean = (normal.raw_delta.clone() * valid).sum_dim(1) / count;
    let previous = (normal.raw_delta - mean.expand([input.batch, input.cands.width]))
        .mask_fill(input.cands.mask.clone().bool_not(), 0.0);
    assert_eq!(values(normal.centered_delta), values(previous));
}

#[test]
fn consistently_remapped_storage_order_is_equivariant() {
    let _guard = RNG.lock().unwrap_or_else(|e| e.into_inner());
    let device = Default::default();
    <B as Backend>::seed(&device, 5301);
    let model = CounterfactualRelationalLoop::<B>::new(V5Config::default(), &device);
    let root = root();
    let graph = graph(&root, 4);
    let remapped = remap_reverse(&graph);
    let a = V5Inputs::<B>::from_examples(&[(&root, &graph)], &device).unwrap();
    let b = V5Inputs::<B>::from_examples(&[(&root, &remapped)], &device).unwrap();
    let za = values(model.paired(&a, 2, Treatment::Normal).logits);
    let zb = values(model.paired(&b, 2, Treatment::Normal).logits);
    let max_abs = za
        .iter()
        .zip(&zb)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max);
    assert!(max_abs <= 1.0e-6, "max_abs={max_abs:e}");
}

#[test]
fn composition_mask_zeros_payloads_at_the_encoded_anchor_boundary() {
    let _guard = RNG.lock().unwrap_or_else(|e| e.into_inner());
    let device = Default::default();
    <B as Backend>::seed(&device, 5301);
    let model = CounterfactualRelationalLoop::<B>::new(V5Config::default(), &device);
    let root = root();
    let graph = graph(&root, 4);
    let input = V5Inputs::<B>::from_examples(&[(&root, &graph)], &device).unwrap();
    let mask = input
        .payload_token_mask(&[vec![false; graph.actual_q]], &device)
        .unwrap();
    let base = model.base(&input);
    let out = model.paired_with_base_payload_mask(&input, base, 4, Treatment::Normal, Some(mask));
    assert!(values(out.raw_delta).into_iter().all(|x| x == 0.0));
    assert!(values(out.centered_delta).into_iter().all(|x| x == 0.0));
    assert_eq!(values(out.logits), values(out.z0));
}

#[test]
fn traced_execution_is_the_same_reader_computation_and_exposes_each_loop() {
    let _guard = RNG.lock().unwrap_or_else(|e| e.into_inner());
    let device = Default::default();
    <B as Backend>::seed(&device, 5301);
    let model = CounterfactualRelationalLoop::<B>::new(V5Config::default(), &device);
    let root = root();
    let graph = graph(&root, 4);
    let input = V5Inputs::<B>::from_examples(&[(&root, &graph)], &device).unwrap();
    let normal = model.paired(&input, 4, Treatment::Normal);
    let traced = model.paired_traced_with_base_payload_mask(
        &input,
        model.base(&input),
        4,
        Treatment::Normal,
        None,
    );
    assert_eq!(values(normal.logits), values(traced.output.logits));
    assert_eq!(traced.factual.len(), 4);
    assert_eq!(traced.null.len(), 4);
    assert_eq!(traced.factual[0].evidence_attention.dims()[1], 8);
}

#[test]
fn loop_prefix_is_identical_and_hypothesis_feedback_reaches_next_evidence_update() {
    let _guard = RNG.lock().unwrap_or_else(|e| e.into_inner());
    let device = Default::default();
    <B as Backend>::seed(&device, 5301);
    let model = CounterfactualRelationalLoop::<B>::new(V5Config::default(), &device);
    let root = root();
    let graph = graph(&root, 4);
    let input = V5Inputs::<B>::from_examples(&[(&root, &graph)], &device).unwrap();
    let run = |r, treatment| {
        model.paired_traced_with_base_payload_mask(&input, model.base(&input), r, treatment, None)
    };
    let one = run(1, Treatment::Normal);
    let four = run(4, Treatment::Normal);
    let no_feedback = run(4, Treatment::NoHypothesisFeedback);
    for (short, long) in [(&one.factual, &four.factual), (&one.null, &four.null)] {
        assert_eq!(
            values(short[0].evidence_after.clone()),
            values(long[0].evidence_after.clone())
        );
        assert_eq!(
            values(short[0].hypothesis_after.clone()),
            values(long[0].hypothesis_after.clone())
        );
    }
    for (normal, intervened) in [
        (&four.factual, &no_feedback.factual),
        (&four.null, &no_feedback.null),
    ] {
        // H_0 is supplied in both at the first iteration; only the next
        // evidence read can observe the feedback intervention.
        assert_eq!(
            values(normal[0].evidence_after.clone()),
            values(intervened[0].evidence_after.clone())
        );
        assert_eq!(
            values(normal[0].hypothesis_after.clone()),
            values(intervened[0].hypothesis_after.clone())
        );
        let movement = values(normal[1].evidence_after.clone())
            .iter()
            .zip(values(intervened[1].evidence_after.clone()))
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            movement > 1.0e-6,
            "feedback has no measurable path to E_2: {movement}"
        );
    }
    assert_eq!(model.num_params(), 7_162_896);
}
