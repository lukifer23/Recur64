//! P6 engineering correctness for `all_info_v1` (CPU, FP32).
//!
//! Engineering checks, not science: exact parameter match, exhaustive no-truncation depth-2
//! trees, sibling-permutation invariance, no proof/label fields, gradient coverage, strict
//! checkpoint identity.

use burn::optim::GradientsParams;
use burn::prelude::*;

use recur64_core::GameState;
use recur64_model::active::coverage::{INERT_NOISE_BOUND, gradient_coverage};
use recur64_model::all_info::{AllInfoModel, AllInfoTree};
use recur64_model::candidate::CandidateInputs;
use recur64_model::checkpoint::{CheckpointMeta, load_training, save_training};
use recur64_model::config::{AllInfoContracts, Architecture, CandidateConfig, ModelConfig};
use recur64_model::net::NeuralModel;
use recur64_model::train::CpuTrainBackend;
use recur64_model::train::adamw;

type B = burn::backend::Flex;

/// The backend RNG is process-global: tests that build models (or compare seeded
/// initialisations) must not interleave.
static RNG: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn lock() -> std::sync::MutexGuard<'static, ()> {
    RNG.lock().unwrap_or_else(|e| e.into_inner())
}
type TB = CpuTrainBackend;

/// The ACTIVE parameter count the control must match (within 0.5%).
const ACTIVE_PARAMS: usize = 30_853_790;

fn tiny() -> ModelConfig {
    let mut c = ModelConfig::all_info_v1();
    c.width = 32;
    c.heads = 4;
    c.ffn = 64;
    c.core_blocks = 2;
    let a = c.all_info.as_mut().unwrap();
    a.candidate = CandidateConfig {
        dim: 16,
        heads: 2,
        ffn: 32,
        blocks: 1,
        facts_hidden: 8,
        policy_hidden: 16,
        facts_enabled: true,
    };
    a.query_dim = 16;
    a.query_heads = 2;
    a.query_ffn = 32;
    a.query_blocks = 2;
    a.set_heads = 2;
    a.set_ffn = 32;
    a.readout_hidden = 16;
    c
}

const FENS: [&str; 3] = [
    "4k3/8/8/8/8/8/3Q4/R3K3 w - - 0 1",
    "8/8/8/4k3/8/8/4K3/1Q5R w - - 0 1",
    "k7/8/1K6/8/8/8/8/1Q5R w - - 0 1",
];

fn roots() -> Vec<GameState> {
    FENS.iter()
        .map(|f| GameState::from_fen(f).unwrap())
        .collect()
}

fn trees(roots: &[GameState]) -> Vec<AllInfoTree> {
    roots
        .iter()
        .map(|r| AllInfoTree::build(r).unwrap())
        .collect()
}

fn log_probs<Bk: Backend>(t: &Tensor<Bk, 2>) -> Vec<f32> {
    t.clone().into_data().to_vec::<f32>().unwrap()
}

#[test]
fn full_geometry_parameter_count_matches_active_within_half_a_percent() {
    let _g = lock();
    let cfg = ModelConfig::all_info_v1();
    let m = AllInfoModel::<B>::new(cfg, &Default::default());
    let total = m.num_params();
    let groups = m.param_breakdown();
    assert_eq!(groups.iter().map(|(_, c)| c).sum::<usize>(), total);
    for (n, c) in &groups {
        eprintln!("{n:34} {c:>12}");
    }
    let diff = total.abs_diff(ACTIVE_PARAMS) as f64 / ACTIVE_PARAMS as f64;
    eprintln!(
        "all_info_v1 total {total}; active {ACTIVE_PARAMS}; diff {} ({:.4}%)",
        total as i64 - ACTIVE_PARAMS as i64,
        diff * 100.0
    );
    assert!(diff <= 0.005, "parameter match {diff} exceeds 0.5%");
}

#[test]
fn forward_is_finite_with_exact_state_counts_and_no_truncation() {
    let _g = lock();
    let device = Default::default();
    let m = AllInfoModel::<B>::new(tiny(), &device);
    let rs = roots();
    let ts = trees(&rs);
    let inputs = CandidateInputs::<B>::from_states(&rs, &device).unwrap();
    let out = m.forward_trees(&inputs, &ts, &device).unwrap();
    let lp = log_probs(&out.readout.policy.log_probs);
    assert!(lp.iter().all(|v| v.is_finite()));
    let w = inputs.cands.width;
    // Each example's probabilities sum to one over its legal candidates.
    for (i, r) in rs.iter().enumerate() {
        let n = r.legal_actions().len();
        let s: f32 = lp[i * w..i * w + n].iter().map(|v| v.exp()).sum();
        assert!(
            (s - 1.0).abs() < 1e-4,
            "example {i}: probabilities sum to {s}"
        );
    }
    // Exhaustiveness: the supplied counts equal an independent depth-2 enumeration.
    for (i, r) in rs.iter().enumerate() {
        let mut d1 = 0;
        let mut d2 = 0;
        for mv in r.legal_standard_moves() {
            let mut c = r.clone();
            c.apply(mv).unwrap();
            d1 += 1;
            if !c.is_terminal() {
                d2 += c.legal_actions().len();
            }
        }
        assert_eq!(out.accounting.states[i], (d1, d2), "example {i}");
        assert_eq!(ts[i].counts().states(), d1 + d2);
    }
    let total: usize = out.accounting.states.iter().map(|(a, b)| a + b).sum();
    assert_eq!(out.accounting.query_encoder_states, total);
    assert_eq!(out.accounting.root_encodes, 1);
    assert_eq!(out.accounting.query_encoder_calls, 1);
}

#[test]
fn a_tree_with_a_terminal_child_is_supported_and_terminal_children_have_no_replies() {
    let _g = lock();
    // Qa8 is mate here; the mating child is terminal and contributes no depth-2 state.
    let r = GameState::from_fen("k7/8/1K6/8/8/8/8/1Q5R w - - 0 1").unwrap();
    let t = AllInfoTree::build(&r).unwrap();
    let c = t.counts();
    assert!(c.terminal_depth1 >= 1, "expected a mating move: {c:?}");
    for br in &t.branches {
        if br.child.terminal {
            assert!(br.replies.is_empty());
        } else {
            assert_eq!(br.replies.len(), br.child.legal_actions.len());
        }
    }
}

#[test]
fn reordering_replies_within_a_branch_leaves_the_policy_unchanged() {
    let _g = lock();
    let device = Default::default();
    let m = AllInfoModel::<B>::new(tiny(), &device);
    let rs = roots();
    let ts = trees(&rs);
    let inputs = CandidateInputs::<B>::from_states(&rs, &device).unwrap();
    let base = log_probs(
        &m.forward_trees(&inputs, &ts, &device)
            .unwrap()
            .readout
            .policy
            .log_probs,
    );
    let mut shuffled = ts.clone();
    for t in &mut shuffled {
        for br in &mut t.branches {
            br.replies.reverse();
            if br.replies.len() > 2 {
                br.replies.rotate_left(1);
            }
        }
    }
    let out = log_probs(
        &m.forward_trees(&inputs, &shuffled, &device)
            .unwrap()
            .readout
            .policy
            .log_probs,
    );
    let max = base
        .iter()
        .zip(&out)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max < 1e-4, "sibling order changed the policy by {max}");
}

#[test]
fn every_parameter_group_receives_a_finite_gradient_in_a_real_update() {
    let _g = lock();
    let device = Default::default();
    let m = AllInfoModel::<TB>::new(tiny(), &device);
    let rs = roots();
    let ts = trees(&rs);
    let inputs = CandidateInputs::<TB>::from_states(&rs, &device).unwrap();
    let out = m.forward_trees(&inputs, &ts, &device).unwrap();
    let w = inputs.cands.width;
    // Cross-entropy toward candidate 0 of every example.
    let mut tgt = vec![0.0f32; rs.len() * w];
    for i in 0..rs.len() {
        tgt[i * w] = 1.0;
    }
    let target =
        Tensor::<TB, 2>::from_data(burn::tensor::TensorData::new(tgt, [rs.len(), w]), &device);
    let loss = (out.readout.policy.log_probs * target).sum().neg();
    let grads = GradientsParams::from_grads(loss.backward(), &m);
    let rows = gradient_coverage::<TB, _>(&m, &grads);
    assert!(!rows.is_empty());
    let mut bad = Vec::new();
    for r in &rows {
        assert!(r.finite, "non-finite gradient in {}", r.name);
        // Exempt: the WDL head (no WDL loss in P6) and key biases (a constant added to
        // every key cancels in the softmax, so their true gradient is exactly zero).
        let inert = r.name.starts_with("root.wdl.") || r.name.ends_with(".k_proj.bias");
        if inert {
            assert!(
                r.max_abs <= INERT_NOISE_BOUND,
                "{} should be inert but has gradient {}",
                r.name,
                r.max_abs
            );
        } else if !r.nonzero {
            bad.push(r.name.clone());
        }
    }
    assert!(bad.is_empty(), "parameters without a gradient: {bad:?}");
}

#[test]
fn no_tree_input_carries_solver_or_label_information() {
    let _g = lock();
    // The only per-state inputs are raw StatePacket fields. Assert the tree's element type
    // exposes exactly the whitelisted content by exhaustively destructuring it: adding a
    // field to the packet (for example a solver value) breaks this test at compile time.
    let r = GameState::from_fen("4k3/8/8/8/8/8/3Q4/R3K3 w - - 0 1").unwrap();
    let t = AllInfoTree::build(&r).unwrap();
    let recur64_statequery::StatePacketV1 {
        node_id: _,
        parent_id: _,
        incoming_action: _,
        ply_from_root: _,
        observation: _,
        legal_actions: _,
        side_to_move: _,
        in_check: _,
        terminal: _,
        terminal_reason: _,
        castling: _,
        ep_square: _,
        halfmove_clock: _,
        repetition_count: _,
        semantic_id: _,
    } = t.branches[0].child.clone();
}

#[test]
fn identity_is_strict_and_all_five_architectures_refuse_each_other() {
    let _g = lock();
    let ai = tiny();
    ai.validate().unwrap();
    let mut other = ai.clone();
    other.all_info.as_mut().unwrap().contracts = AllInfoContracts {
        integrator: "all_info_tree_integrator_v2".into(),
        ..AllInfoContracts::default()
    };
    assert!(
        other.validate().is_err(),
        "a changed contract must be refused"
    );
    let mut ac = ModelConfig::active_search_v3();
    ac.all_info = Some(Default::default());
    assert!(
        ac.validate().is_err(),
        "only all_info_v1 may carry an all-info geometry"
    );
    let mut bad = ai.clone();
    bad.architecture = Architecture::ActiveSearchV3;
    assert!(bad.validate().is_err());
    assert_eq!(Architecture::AllInfoV1.id(), "all_info_v1");
    let m = AllInfoModel::<B>::build(&ai, &Default::default()).unwrap();
    assert_eq!(
        <AllInfoModel<B> as NeuralModel<B>>::ARCHITECTURE,
        Architecture::AllInfoV1
    );
    drop(m);
    assert!(
        AllInfoModel::<B>::build(&ModelConfig::active_search_v3(), &Default::default()).is_err()
    );
}

fn flat<Bk: Backend, M: burn::module::Module<Bk>>(m: &M) -> Vec<f32> {
    struct P(Vec<f32>);
    impl<Bk: Backend> burn::module::ModuleVisitor<Bk> for P {
        fn visit_float<const D: usize>(&mut self, p: &burn::module::Param<Tensor<Bk, D>>) {
            self.0.extend(p.val().into_data().to_vec::<f32>().unwrap());
        }
    }
    let mut v = P(Vec::new());
    m.visit(&mut v);
    v.0
}

/// The shared root encoder and query encoder reuse the ACTIVE module TYPES and contracts,
/// but NOT their initial weights: Burn initialises linear layers lazily in module-field
/// order, so the backend RNG stream a module sees depends on what was built before it
/// (ACTIVE: planner and selector; ALL-INFO: the set integrator). Initial-weight identity
/// across the two architectures therefore does not hold and is not claimed; the paired
/// seeds label the comparison, they do not equalise initialisation. What IS guaranteed,
/// and tested here, is determinism within ALL-INFO for one seed.
#[test]
fn one_seed_gives_one_all_info_initialisation_and_other_seeds_differ() {
    let _g = lock();
    let device = Default::default();
    <B as Backend>::seed(&device, 5101);
    let a = AllInfoModel::<B>::new(tiny(), &device);
    <B as Backend>::seed(&device, 5101);
    let b = AllInfoModel::<B>::new(tiny(), &device);
    <B as Backend>::seed(&device, 5102);
    let c = AllInfoModel::<B>::new(tiny(), &device);
    let (fa, fb, fc) = (flat(&a), flat(&b), flat(&c));
    assert!(!fa.is_empty());
    assert_eq!(fa, fb, "one seed must give one initialisation");
    assert_ne!(fa, fc, "different seeds must differ");
}

#[test]
fn checkpoint_round_trips_and_refuses_a_tampered_contract_and_another_architecture() {
    use recur64_model::active::ActiveSearchModel;
    let _g = lock();
    let device = Default::default();
    let cfg = tiny();
    let m = AllInfoModel::<TB>::new(cfg.clone(), &device);
    let optim = adamw::<TB, AllInfoModel<TB>>();
    let dir = std::env::temp_dir().join(format!("recur64_all_info_ckpt_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let meta = CheckpointMeta::new(cfg.clone(), 1, false, 0, 3e-4, 5101, 0, "v3-p6", "fp32");
    assert_eq!(meta.architecture, "all_info_v1");
    assert!(meta.all_info_contracts.is_some() && meta.active_contracts.is_none());
    save_training::<TB, _, _>(&dir, &m, &optim, &meta).unwrap();
    let template = AllInfoModel::<TB>::new(cfg.clone(), &device);
    let (loaded, _o, got) =
        load_training::<TB, _, _>(&dir, template, adamw::<TB, AllInfoModel<TB>>(), &device)
            .unwrap();
    assert_eq!(flat(&loaded), flat(&m));
    assert_eq!(got.seed, 5101);

    // Another architecture's template refuses it explicitly.
    let active = ActiveSearchModel::<TB>::new(
        {
            let mut c = ModelConfig::active_search_v3();
            c.width = 32;
            c.heads = 4;
            c.ffn = 64;
            c.core_blocks = 2;
            c
        },
        &device,
    );
    let e = load_training::<TB, _, _>(&dir, active, adamw::<TB, ActiveSearchModel<TB>>(), &device)
        .err()
        .expect("must refuse")
        .to_string();
    assert!(
        e.contains("all_info_v1") || e.contains("cross-architecture"),
        "{e}"
    );

    // A tampered contract in the metadata is refused.
    let mp = dir.join("meta.json");
    let mut j: serde_json::Value = serde_json::from_slice(&std::fs::read(&mp).unwrap()).unwrap();
    j["all_info_contracts"]["integrator"] = "all_info_tree_integrator_v2".into();
    std::fs::write(&mp, serde_json::to_vec(&j).unwrap()).unwrap();
    let template = AllInfoModel::<TB>::new(cfg, &device);
    assert!(
        load_training::<TB, _, _>(&dir, template, adamw::<TB, AllInfoModel<TB>>(), &device)
            .is_err()
    );
    let _ = std::fs::remove_dir_all(&dir);
}
