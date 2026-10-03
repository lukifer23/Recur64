use burn::module::AutodiffModule;
use burn::optim::{GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::TensorData;
use recur64_core::GameState;
use recur64_model::active::coverage::gradient_coverage;
use recur64_model::train::adamw;
use recur64_v5::config::V5Config;
use recur64_v5::graph::{EpisodeKey, Schedule, acquire};
use recur64_v5::model::{CounterfactualRelationalLoop, Treatment, V5Inputs};
use recur64_v5::stage::baseline_fingerprint;

type B = burn::backend::Autodiff<burn::backend::Flex>;
type I = burn::backend::Flex;
static RNG: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn fixture() -> (GameState, recur64_v5::graph::AcquiredGraph) {
    let root = GameState::from_fen("6k1/8/8/8/8/8/4Q3/3RK3 w - - 0 1").unwrap();
    let graph = acquire(
        &root,
        EpisodeKey {
            position_id: "gradient-fixture".into(),
            schedule: Schedule::UniformFrontierV1,
            run_seed: 5301,
            occurrence_ordinal: 0,
        },
        2,
        None,
    )
    .unwrap();
    (root, graph)
}

fn candidate_relative<Bk: Backend>(
    model: &CounterfactualRelationalLoop<Bk>,
    input: &V5Inputs<Bk>,
) -> Tensor<Bk, 1> {
    let output = model.paired(input, 2, Treatment::Normal);
    output.centered_delta.clone().slice([0..1, 0..1]).sum()
        - output.centered_delta.slice([0..1, 1..2]).sum()
}

#[test]
fn full_bptt_reaches_reader_groups_while_the_graph_free_base_stays_immutable() {
    let _guard = RNG.lock().unwrap_or_else(|error| error.into_inner());
    let device = Default::default();
    <B as Backend>::seed(&device, 5301);
    let mut model = CounterfactualRelationalLoop::<B>::new(V5Config::default(), &device);
    let (root, graph) = fixture();
    let examples = [(&root, &graph)];
    let mut input = V5Inputs::<B>::from_examples(&examples, &device).unwrap();
    let tracked = input.states.clone().require_grad();
    input.states = tracked.clone();
    let base = model.base_frozen(&examples, &device).unwrap();
    let output = model.paired_with_base(&input, base, 2, Treatment::Normal);
    let loss = output.centered_delta.clone().slice([0..1, 0..1]).sum()
        - output.centered_delta.slice([0..1, 1..2]).sum();
    let raw_grads = loss.backward();
    let input_grad = tracked.grad(&raw_grads).unwrap();
    let input_norm: f32 = input_grad
        .into_data()
        .to_vec::<f32>()
        .unwrap()
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt();
    assert!(input_norm.is_finite() && input_norm > 0.0);

    let grads = GradientsParams::from_grads(raw_grads, &model);
    let coverage = gradient_coverage::<B, _>(&model, &grads);
    for prefix in ["state.", "evidence.", "hypothesis.", "correction_"] {
        assert!(
            coverage
                .iter()
                .any(|row| row.name.starts_with(prefix) && row.finite && row.nonzero),
            "no nonzero finite gradient in {prefix}: {coverage:?}"
        );
    }
    assert!(
        coverage
            .iter()
            .filter(|row| row.name.starts_with("root."))
            .all(|row| !row.has_grad),
        "graph-free frozen baseline received a gradient"
    );
    let before = baseline_fingerprint(&model, &device).unwrap();
    let mut optimizer = adamw::<B, CounterfactualRelationalLoop<B>>();
    model = optimizer.step(1.0e-3, model, grads);
    assert_eq!(before, baseline_fingerprint(&model, &device).unwrap());
}

#[test]
fn returned_payload_autodiff_matches_several_finite_differences() {
    let _guard = RNG.lock().unwrap_or_else(|error| error.into_inner());
    let device = Default::default();
    <B as Backend>::seed(&device, 5301);
    let model = CounterfactualRelationalLoop::<B>::new(V5Config::default(), &device);
    let inference = model.valid();
    let (root, graph) = fixture();
    let examples = [(&root, &graph)];
    let mut ad_input = V5Inputs::<B>::from_examples(&examples, &device).unwrap();
    let tracked = ad_input.states.clone().require_grad();
    ad_input.states = tracked.clone();
    let grads = candidate_relative(&model, &ad_input).backward();
    let analytic = tracked
        .grad(&grads)
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let shape = ad_input.states.dims();
    let count = shape.iter().product::<usize>();
    let epsilon = 5.0e-2_f32;
    let direction_seeds = [0xA501_u64, 0xA502, 0xA503, 0xA504];
    let mut nonzero = 0;
    for seed in direction_seeds {
        let mut rng = seed;
        let direction: Vec<f32> = (0..count)
            .map(|_| {
                if splitmix64(&mut rng) & 1 == 0 {
                    -1.0
                } else {
                    1.0
                }
            })
            .collect();
        let derivative: f32 = analytic
            .iter()
            .zip(&direction)
            .map(|(gradient, delta)| gradient * delta)
            .sum();
        let base_input = V5Inputs::<I>::from_examples(&examples, &device).unwrap();
        let direction_tensor =
            Tensor::<I, 4>::from_data(TensorData::new(direction, shape), &device);
        let mut plus = V5Inputs::<I>::from_examples(&examples, &device).unwrap();
        plus.states = base_input.states.clone() + direction_tensor.clone().mul_scalar(epsilon);
        let mut minus = V5Inputs::<I>::from_examples(&examples, &device).unwrap();
        minus.states = base_input.states - direction_tensor.mul_scalar(epsilon);
        let p = candidate_relative(&inference, &plus)
            .into_data()
            .to_vec::<f32>()
            .unwrap()[0];
        let m = candidate_relative(&inference, &minus)
            .into_data()
            .to_vec::<f32>()
            .unwrap()[0];
        let numeric = (p - m) / (2.0 * epsilon);
        if derivative.abs() > 1.0e-8 || numeric.abs() > 1.0e-8 {
            nonzero += 1;
            let relative =
                (derivative - numeric).abs() / derivative.abs().max(numeric.abs()).max(1.0e-6);
            assert!(
                relative < 0.12,
                "direction seed {seed:x} analytic={derivative:e} numeric={numeric:e} relative={relative:e}"
            );
        }
    }
    assert!(nonzero >= 3, "too few nonsaturated perturbation directions");
}
