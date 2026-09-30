//! P0 gate for `legacy_facts_v25` (P2.5 LF): the legacy head-v2 policy plus a
//! candidate-local CandidateFacts delta.

use burn::optim::GradientsParams;
use burn::prelude::*;

use recur64_core::{CandidateFactsV1, GameState, StandardMove};
use recur64_model::candidate::{CandidateInputs, facts_tensor};
use recur64_model::config::{LegacyFactsConfig, ModelConfig};
use recur64_model::legacy_facts::LegacyFactsModel;
use recur64_model::loss::{Targets, model_loss};
use recur64_model::model::{PolicyOutput, ProbeModel};
use recur64_model::train::CpuTrainBackend;

type B = burn::backend::Flex;

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

fn tiny() -> ModelConfig {
    let mut c = ModelConfig::legacy_facts_v25();
    c.width = 32;
    c.heads = 4;
    c.ffn = 64;
    c.core_blocks = 2;
    c.policy_dim = 16;
    c.legacy_facts = Some(LegacyFactsConfig { facts_hidden: 8 });
    c
}

fn positions(n: usize, seed: u64) -> Vec<GameState> {
    let mut rng = Rng(seed);
    let mut out = Vec::new();
    while out.len() < n {
        let mut g = GameState::startpos();
        for _ in 0..(4 + (rng.next() % 60) as usize) {
            let a = g.legal_actions();
            if a.is_empty() || g.is_terminal() {
                break;
            }
            let id = a[(rng.next() as usize) % a.len()];
            let (f, t, p) = id.to_physical(g.perspective());
            g.apply(StandardMove::new(f, t, (!p.is_none()).then_some(p)))
                .unwrap();
        }
        if !g.is_terminal() && !g.legal_actions().is_empty() {
            out.push(g);
        }
    }
    out
}

/// Probabilities as consumers read them: padded slots are masked to exactly 0.
fn masked_probs(p: &PolicyOutput<B>) -> Vec<f32> {
    let m: Vec<f32> = p.mask.clone().float().into_data().to_vec().unwrap();
    let lp: Vec<f32> = p.log_probs.clone().into_data().to_vec().unwrap();
    lp.iter().zip(&m).map(|(l, k)| l.exp() * k).collect()
}

#[test]
fn full_geometry_is_l_plus_the_facts_mlp_and_a_distinct_identity() {
    let device = Default::default();
    let lf = LegacyFactsModel::<B>::new(ModelConfig::legacy_facts_v25(), &device);
    let l = ProbeModel::<B>::new(
        {
            let mut c = ModelConfig::legacy_facts_v25();
            c.architecture = Default::default();
            c.legacy_facts = None;
            c
        },
        &device,
    );
    let n = lf.num_params();
    let sum: usize = lf.param_breakdown().iter().map(|(_, c)| c).sum();
    eprintln!("LF params {n}, L params {}", l.num_params());
    assert_eq!(n, sum);
    assert_eq!(
        n,
        l.num_params() + 8 * 64 + 64 + 64 + 1,
        "only the facts MLP is added"
    );
    assert_eq!(lf.config().architecture.id(), "legacy_facts_v25");
}

#[test]
fn forward_is_finite_normalized_padded_and_terminal_safe() {
    let device = Default::default();
    let m = LegacyFactsModel::<B>::new(tiny(), &device);
    let mated = GameState::from_fen("R5k1/5ppp/8/8/8/8/8/7K b - - 0 1").unwrap();
    let mut states = positions(10, 4);
    states.push(mated);
    let inp = CandidateInputs::<B>::from_states(&states, &device).unwrap();
    let w = inp.cands.width;
    let out = m.forward(inp.board, &inp.cands, inp.facts);
    let p = masked_probs(&out.readouts[0].policy);
    let wdl: Vec<f32> = out.readouts[0]
        .wdl_logits
        .clone()
        .into_data()
        .to_vec()
        .unwrap();
    assert!(p.iter().chain(&wdl).all(|v| v.is_finite()));
    assert!(
        wdl.iter().all(|v| *v == 0.0),
        "fresh WDL logits are exactly 0"
    );
    for (i, s) in states.iter().enumerate() {
        let n = s.legal_actions().len();
        let sum: f32 = p[i * w..i * w + n].iter().sum();
        if n > 0 {
            assert!((sum - 1.0).abs() < 1e-5, "row {i} sums to {sum}");
        }
        assert!(
            p[i * w + n..(i + 1) * w].iter().all(|v| *v == 0.0),
            "padding row {i}"
        );
    }
}

#[test]
fn facts_change_the_policy_and_zero_facts_reproduce_the_legacy_policy() {
    let device = Default::default();
    let m = LegacyFactsModel::<B>::new(tiny(), &device);
    let states = positions(8, 11);
    let inp = CandidateInputs::<B>::from_states(&states, &device).unwrap();
    let w = inp.cands.width;

    // Real facts vs altered facts: different policy.
    let altered = {
        let rows: Vec<Vec<CandidateFactsV1>> = states
            .iter()
            .map(|s| {
                recur64_core::candidate_facts(s)
                    .into_iter()
                    .map(|mut r| {
                        r[0] = 1.0 - r[0];
                        r[4] = 1.0 - r[4];
                        r
                    })
                    .collect()
            })
            .collect();
        let refs: Vec<&[CandidateFactsV1]> = rows.iter().map(Vec::as_slice).collect();
        facts_tensor::<B>(&refs, w, &device).unwrap()
    };
    let a = masked_probs(
        &m.forward(inp.board.clone(), &inp.cands, inp.facts.clone())
            .readouts[0]
            .policy,
    );
    let b = masked_probs(&m.forward(inp.board.clone(), &inp.cands, altered).readouts[0].policy);
    let diff = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        diff > 1e-6,
        "facts must change the policy (max diff {diff})"
    );

    // All-zero facts add the SAME delta to every candidate of a row (only biases
    // survive), which the softmax cancels: the wrapped legacy model's policy is
    // reproduced.
    let zeros = inp.facts.clone().zeros_like();
    let lf0 = masked_probs(&m.forward(inp.board.clone(), &inp.cands, zeros).readouts[0].policy);
    let base = masked_probs(
        &m.probe()
            .forward_r(inp.board.clone(), &inp.cands, 1, false)
            .readouts[0]
            .policy,
    );
    let d0 = lf0
        .iter()
        .zip(&base)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        d0 < 1e-5,
        "zero facts must reproduce the legacy policy (max diff {d0})"
    );
}

#[test]
fn fresh_full_model_prior_is_not_pathologically_concentrated() {
    let device = Default::default();
    let m = LegacyFactsModel::<B>::new(ModelConfig::legacy_facts_v25(), &device);
    let states = positions(48, 0xA11CE);
    let inp = CandidateInputs::<B>::from_states(&states, &device).unwrap();
    let w = inp.cands.width;
    let out = m.forward(inp.board, &inp.cands, inp.facts);
    let p = masked_probs(&out.readouts[0].policy);
    let wdl: Vec<f32> = out.readouts[0]
        .wdl_logits
        .clone()
        .into_data()
        .to_vec()
        .unwrap();
    assert!(wdl.iter().all(|v| *v == 0.0));
    let (mut ratio, mut worst) = (0.0f64, 0.0f64);
    for (i, s) in states.iter().enumerate() {
        let n = s.legal_actions().len();
        let row = &p[i * w..i * w + n];
        let h: f64 = row
            .iter()
            .map(|&x| {
                if x > 0.0 {
                    -(x as f64) * (x as f64).ln()
                } else {
                    0.0
                }
            })
            .sum();
        ratio += h / (n as f64).ln();
        worst = worst.max(row.iter().cloned().fold(0.0f32, f32::max) as f64 * n as f64);
    }
    let mean_ratio = ratio / states.len() as f64;
    eprintln!("LF fresh entropy/uniform = {mean_ratio:.5}; worst top1/uniform = {worst:.3}");
    assert!(worst < 3.0, "pathological concentration {worst}");
    assert!(mean_ratio > 0.9, "fresh prior entropy {mean_ratio}");
}

#[test]
fn every_facts_parameter_gets_a_finite_nonzero_gradient_on_update_one() {
    type TB = CpuTrainBackend;
    type Inner = burn::backend::Flex;
    let device = Default::default();
    let m = LegacyFactsModel::<TB>::new(tiny(), &device);
    let states = positions(8, 17);
    let inp = CandidateInputs::<TB>::from_states(&states, &device).unwrap();
    let (b, w) = (states.len(), inp.cands.width);
    let mut t = vec![0.0f32; b * w];
    for i in 0..b {
        t[i * w] = 1.0;
    }
    let targets = Targets {
        policy_target: Tensor::<TB, 2>::from_data(
            burn::tensor::TensorData::new(t, [b, w]),
            &device,
        ),
        wdl_target: Tensor::<TB, 1, Int>::from_data(
            burn::tensor::TensorData::new(vec![1i32; b], [b]),
            &device,
        ),
    };
    let out = m.forward(inp.board.clone(), &inp.cands, inp.facts.clone());
    let loss = model_loss(&out, &targets);
    let grads = GradientsParams::from_grads(loss.backward(), &m);
    for (name, id) in [
        ("facts1", m.facts_weight_id()),
        ("facts2", m.delta_weight_id()),
    ] {
        let g = grads
            .get::<Inner, 2>(id)
            .unwrap_or_else(|| panic!("no gradient for {name}"))
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert!(g.iter().all(|v| v.is_finite()), "{name} non-finite");
        assert!(g.iter().any(|v| v.abs() > 1e-12), "{name} gradient is zero");
    }
}
