//! Structural regression: LF's wrapped Probe IS the historical legacy model.
//!
//! This lives in its own integration-test file on purpose. Burn's backend RNG is
//! global process state, and Rust runs the tests of one file on parallel threads, so a
//! sibling test consuming random numbers between `seed()` and a model construction
//! makes a same-seed comparison nondeterministic. One test per file means one process
//! with no concurrent RNG consumers.

use recur64_core::{GameState, StandardMove};
use recur64_model::candidate::CandidateInputs;
use recur64_model::config::{LegacyFactsConfig, ModelConfig};
use recur64_model::legacy_facts::LegacyFactsModel;
use recur64_model::model::{ModelOutput, PolicyOutput, ProbeModel};

type B = burn::backend::Flex;

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

fn positions(n: usize) -> Vec<GameState> {
    let mut out = Vec::new();
    let mut g = GameState::startpos();
    out.push(g.clone());
    let mut k = 0u64;
    while out.len() < n {
        let a = g.legal_actions();
        let id = a[(k as usize * 7 + 3) % a.len()];
        let (f, t, p) = id.to_physical(g.perspective());
        g.apply(StandardMove::new(f, t, (!p.is_none()).then_some(p)))
            .unwrap();
        out.push(g.clone());
        k += 1;
    }
    out
}

fn masked_probs(p: &PolicyOutput<B>) -> Vec<f32> {
    let m: Vec<f32> = p.mask.clone().float().into_data().to_vec().unwrap();
    let lp: Vec<f32> = p.log_probs.clone().into_data().to_vec().unwrap();
    lp.iter().zip(&m).map(|(l, k)| l.exp() * k).collect()
}

fn bits(o: &ModelOutput<B>) -> Vec<u32> {
    let r = &o.readouts[0];
    let mut v: Vec<u32> = r
        .policy
        .log_probs
        .clone()
        .into_data()
        .to_vec::<f32>()
        .unwrap()
        .iter()
        .map(|x| x.to_bits())
        .collect();
    v.extend(
        r.wdl_logits
            .clone()
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .map(|x| x.to_bits()),
    );
    v
}

#[test]
fn wrapped_probe_is_the_historical_legacy_model_under_the_same_seed() {
    use burn::tensor::backend::Backend;
    let device = Default::default();
    let cfg = tiny();
    let mut probe_cfg = cfg.clone();
    probe_cfg.architecture = Default::default();
    probe_cfg.legacy_facts = None;
    let inp = CandidateInputs::<B>::from_states(&positions(6), &device).unwrap();

    // An independently constructed Probe and an LF, each built right after seeding the
    // backend identically. LF constructs its wrapped Probe FIRST, so the legacy weights
    // must be bit-identical.
    <B as Backend>::seed(&device, 1234);
    let independent = ProbeModel::<B>::new(probe_cfg, &device);
    <B as Backend>::seed(&device, 1234);
    let lf = LegacyFactsModel::<B>::new(cfg, &device);

    let a = independent.forward_r(inp.board.clone(), &inp.cands, 1, false);
    let b = lf
        .probe()
        .forward_r(inp.board.clone(), &inp.cands, 1, false);
    assert_eq!(
        bits(&a),
        bits(&b),
        "wrapped Probe must equal an independent Probe (bitwise)"
    );

    // And LF with zero facts equals its wrapped Probe (the constant delta cancels).
    let lf0 = lf.forward(
        inp.board.clone(),
        &inp.cands,
        inp.facts.clone().zeros_like(),
    );
    let base = masked_probs(&b.readouts[0].policy);
    let zero = masked_probs(&lf0.readouts[0].policy);
    let d = base
        .iter()
        .zip(&zero)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        d < 1e-6,
        "zero-fact LF must equal its wrapped Probe (max diff {d})"
    );
}
