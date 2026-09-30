//! P0 architecture-correctness gate for `candidate_v25` (model-level items).
//!
//! Full-geometry checks (parameter count, fresh prior, WDL neutrality) run the
//! real 640-wide model on CPU. Structural checks (equivariance, masking, facts
//! ablation, first-step gradients) use a tiny geometry of the same architecture.

use burn::optim::GradientsParams;
use burn::prelude::*;

use recur64_core::{CandidateFactsV1, GameState, StandardMove};
use recur64_model::candidate::{CandidateInputs, CandidateV25Model, facts_tensor};
use recur64_model::config::{CandidateConfig, ModelConfig};
use recur64_model::loss::{Targets, model_loss};
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

fn tiny(facts_enabled: bool) -> ModelConfig {
    let mut c = ModelConfig::candidate_v25(facts_enabled);
    c.width = 32;
    c.heads = 4;
    c.ffn = 64;
    c.core_blocks = 2;
    c.candidate = Some(CandidateConfig {
        dim: 16,
        heads: 2,
        ffn: 32,
        blocks: 1,
        facts_hidden: 8,
        policy_hidden: 16,
        facts_enabled,
    });
    c
}

/// Deterministic broad position set: seeded random walks from the start.
fn positions(n: usize, seed: u64) -> Vec<GameState> {
    let mut rng = Rng(seed);
    let mut out = Vec::new();
    while out.len() < n {
        let mut g = GameState::startpos();
        let depth = 4 + (rng.next() % 60) as usize;
        for _ in 0..depth {
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

fn probs(model: &CandidateV25Model<B>, inp: &CandidateInputs<B>) -> (Vec<f32>, Vec<f32>, usize) {
    let out = model.forward(inp.board.clone(), &inp.cands, inp.facts.clone());
    let wdl = out.readouts[0]
        .wdl_logits
        .clone()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    (masked_probs(&out.readouts[0].policy), wdl, inp.cands.width)
}

/// Probabilities as consumers read them: padded slots carry log-prob 0 by the
/// shared `PolicyOutput` convention and are removed by the mask, so their
/// probability is exactly 0.
fn masked_probs(p: &recur64_model::model::PolicyOutput<B>) -> Vec<f32> {
    let m: Vec<f32> = p.mask.clone().float().into_data().to_vec().unwrap();
    let lp: Vec<f32> = p.log_probs.clone().into_data().to_vec().unwrap();
    lp.iter().zip(&m).map(|(l, k)| l.exp() * k).collect()
}

#[test]
fn full_geometry_parameter_count_is_in_the_target_range() {
    let device = Default::default();
    let m = CandidateV25Model::<B>::new(ModelConfig::candidate_v25(true), &device);
    let n = m.num_params();
    let sum: usize = m.param_breakdown().iter().map(|(_, c)| c).sum();
    eprintln!("candidate_v25 params: {n}");
    for (name, c) in m.param_breakdown() {
        eprintln!("  {name:<22}{c:>12}");
    }
    assert_eq!(n, sum, "breakdown must account for every parameter");
    assert!(
        (26_000_000..=29_000_000).contains(&n),
        "{n} outside 26M-29M"
    );
    // C0 has the same parameters as CF.
    let c0 = CandidateV25Model::<B>::new(ModelConfig::candidate_v25(false), &device);
    assert_eq!(c0.num_params(), n);
}

#[test]
fn fresh_full_model_has_a_near_uniform_prior_and_neutral_wdl() {
    let device = Default::default();
    let m = CandidateV25Model::<B>::new(ModelConfig::candidate_v25(true), &device);
    let states = positions(48, 0xA11CE);
    let inp = CandidateInputs::<B>::from_states(&states, &device).unwrap();
    let (p, wdl, w) = probs(&m, &inp);
    assert!(p.iter().chain(&wdl).all(|v| v.is_finite()));
    assert!(
        wdl.iter().all(|v| *v == 0.0),
        "fresh WDL logits are exactly 0"
    );
    let (mut ratio_sum, mut worst_top) = (0.0f64, 0.0f64);
    for (i, s) in states.iter().enumerate() {
        let n = s.legal_actions().len();
        let row = &p[i * w..i * w + n];
        let sum: f32 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "row {i} sums to {sum}");
        assert!(p[i * w + n..(i + 1) * w].iter().all(|v| *v == 0.0));
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
        ratio_sum += h / (n as f64).ln();
        let top = row.iter().cloned().fold(0.0f32, f32::max) as f64 * n as f64;
        worst_top = worst_top.max(top);
    }
    let mean_ratio = ratio_sum / states.len() as f64;
    eprintln!("fresh entropy/uniform = {mean_ratio:.5}; worst top1/uniform = {worst_top:.3}");
    assert!(mean_ratio >= 0.98, "fresh prior entropy {mean_ratio}");
    assert!(
        worst_top < 3.0,
        "pathological top-1 concentration {worst_top}"
    );
}

#[test]
fn policy_normalizes_and_padding_is_exactly_zero_and_irrelevant() {
    let device = Default::default();
    let m = CandidateV25Model::<B>::new(tiny(true), &device);
    let states = positions(12, 7);
    let inp = CandidateInputs::<B>::from_states(&states, &device).unwrap();
    let (p, _, w) = probs(&m, &inp);
    for (i, s) in states.iter().enumerate() {
        let n = s.legal_actions().len();
        let sum: f32 = p[i * w..i * w + n].iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(p[i * w + n..(i + 1) * w].iter().all(|v| *v == 0.0));
    }
    // Garbage in padded facts slots must not change any legal probability.
    let facts: Vec<Vec<CandidateFactsV1>> =
        states.iter().map(recur64_core::candidate_facts).collect();
    let mut data = vec![0.0f32; states.len() * w * 8];
    for (i, f) in facts.iter().enumerate() {
        for k in 0..w {
            let row = f.get(k).copied().unwrap_or([0.9; 8]); // padding = garbage
            data[(i * w + k) * 8..(i * w + k + 1) * 8].copy_from_slice(&row);
        }
    }
    let dirty = Tensor::<B, 3>::from_data(
        burn::tensor::TensorData::new(data, [states.len(), w, 8]),
        &device,
    );
    let out = m.forward(inp.board.clone(), &inp.cands, dirty);
    let q = masked_probs(&out.readouts[0].policy);
    for (a, b) in p.iter().zip(&q) {
        assert!((a - b).abs() < 1e-6, "padding content leaked: {a} vs {b}");
    }
}

#[test]
fn terminal_rows_are_finite_and_bypass_the_policy() {
    let device = Default::default();
    let m = CandidateV25Model::<B>::new(tiny(true), &device);
    let mated = GameState::from_fen("R5k1/5ppp/8/8/8/8/8/7K b - - 0 1").unwrap();
    assert!(mated.legal_actions().is_empty());
    let mut states = positions(3, 5);
    states.push(mated);
    let inp = CandidateInputs::<B>::from_states(&states, &device).unwrap();
    let (p, wdl, w) = probs(&m, &inp);
    assert!(p.iter().chain(&wdl).all(|v| v.is_finite()));
    assert!(p[3 * w..4 * w].iter().all(|v| *v == 0.0), "terminal row");
}

#[test]
fn candidate_order_permutation_permutes_the_policy_identically() {
    let device = Default::default();
    let m = CandidateV25Model::<B>::new(tiny(true), &device);
    let states = positions(6, 99);
    for s in &states {
        let one = std::slice::from_ref(s);
        let base = CandidateInputs::<B>::from_states(one, &device).unwrap();
        let (p0, _, w) = probs(&m, &base);
        let n = s.legal_actions().len();
        assert_eq!(w, n);
        // Reverse-and-rotate permutation of legal actions AND facts together.
        let perm: Vec<usize> = (0..n).map(|i| (n - 1 - i + 3) % n).collect();
        let legal = s.legal_actions();
        let facts = recur64_core::candidate_facts(s);
        let legal_p: Vec<_> = perm.iter().map(|&i| legal[i]).collect();
        let facts_p: Vec<CandidateFactsV1> = perm.iter().map(|&i| facts[i]).collect();
        let obs = recur64_core::encode_observation_v1(s);
        let inp =
            CandidateInputs::<B>::from_parts(&[&obs], &[legal_p], &[facts_p.as_slice()], &device)
                .unwrap();
        let (p1, _, _) = probs(&m, &inp);
        for (k, &i) in perm.iter().enumerate() {
            assert!(
                (p1[k] - p0[i]).abs() < 1e-5,
                "candidate {i}->{k}: {} vs {}",
                p0[i],
                p1[k]
            );
        }
    }
}

#[test]
fn c0_ignores_fact_values_and_cf_responds_to_them() {
    let device = Default::default();
    let states = positions(8, 31);
    let inp = CandidateInputs::<B>::from_states(&states, &device).unwrap();
    let w = inp.cands.width;
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
    let max_diff = |m: &CandidateV25Model<B>| {
        let a = m.forward(inp.board.clone(), &inp.cands, inp.facts.clone());
        let b = m.forward(inp.board.clone(), &inp.cands, altered.clone());
        let x = a.readouts[0]
            .policy
            .log_probs
            .clone()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let y = b.readouts[0]
            .policy
            .log_probs
            .clone()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        x.iter()
            .zip(&y)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
    };
    let c0 = CandidateV25Model::<B>::new(tiny(false), &device);
    let cf = CandidateV25Model::<B>::new(tiny(true), &device);
    assert_eq!(
        max_diff(&c0),
        0.0,
        "C0 output must not depend on fact values"
    );
    assert!(
        max_diff(&cf) > 1e-6,
        "CF output must respond to valid fact changes"
    );
}

#[test]
fn facts_encoder_gets_a_finite_nonzero_gradient_on_the_first_step() {
    type TB = CpuTrainBackend;
    type Inner = burn::backend::Flex;
    let device = Default::default();
    let m = CandidateV25Model::<TB>::new(tiny(true), &device);
    let states = positions(8, 17);
    let inp = CandidateInputs::<TB>::from_states(&states, &device).unwrap();
    // Target: all mass on the first legal move of each position.
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
    let g = grads
        .get::<Inner, 2>(m.facts_weight_id())
        .expect("facts encoder gradient")
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    assert!(g.iter().all(|v| v.is_finite()));
    assert!(g.iter().any(|v| v.abs() > 1e-12), "facts gradient is zero");
    let gp = grads
        .get::<Inner, 2>(m.policy_weight_id())
        .expect("policy scorer gradient")
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    assert!(gp.iter().any(|v| v.abs() > 1e-12));
}

#[test]
fn cross_check_candidate_tensor_promotion_codes_are_raw() {
    // Promotion "none" and "knight" must stay distinguishable to the model.
    let s = GameState::from_fen("7k/4P3/8/8/8/8/8/K7 w - - 0 1").unwrap();
    let device = Default::default();
    let inp = CandidateInputs::<B>::from_states(&[s], &device).unwrap();
    let codes: Vec<i32> = inp
        .cands
        .promo_code
        .clone()
        .into_data()
        .convert::<i32>()
        .to_vec::<i32>()
        .unwrap();
    assert!(codes.contains(&0) && codes.iter().any(|&c| c >= 1));
}
