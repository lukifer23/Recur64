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
