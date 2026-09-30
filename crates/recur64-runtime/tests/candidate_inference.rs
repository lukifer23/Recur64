//! `candidate_v25` through the inference owner and PUCT: facts cross the thread
//! channel aligned with the legal actions, the batched and single-position
//! evaluators agree, missing facts are an error (never a zero fallback), and
//! legacy Probe evaluators never pay for facts.

use recur64_core::{GameState, candidate_facts, encode_observation_v1};
use recur64_model::candidate::CandidateV25Model;
use recur64_model::config::{CandidateConfig, ModelConfig};
use recur64_model::model::ProbeModel;
use recur64_runtime::evaluator::SyncEvaluator;
use recur64_runtime::inference::{BatchedModel, InferenceConfig, InferenceOwner};
use recur64_search::puct::{PuctConfig, search};
use recur64_search::{ChessGame, EvalError, EvalRequest, Evaluator};

type B = burn::backend::Flex;

fn cand_cfg() -> ModelConfig {
    let mut c = ModelConfig::candidate_v25(true);
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
        facts_enabled: true,
    });
    c
}

fn states() -> Vec<GameState> {
    let mut g = GameState::startpos();
    let mut out = vec![g.clone()];
    for m in ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"] {
        g.apply_uci(m).unwrap();
        out.push(g.clone());
    }
    out.push(GameState::from_fen("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1").unwrap());
    out
}

#[test]
fn batched_and_sync_evaluators_agree_and_facts_are_required() {
    let device = Default::default();
    let model = CandidateV25Model::<B>::new(cand_cfg(), &device);
    let sync = SyncEvaluator::new(model.clone(), 1, device);
    let owner = InferenceOwner::spawn(
        BatchedModel::new(model.clone(), 1, device),
        InferenceConfig::default(),
    );
    let batched = owner.evaluator();
    assert!(batched.needs_candidate_facts() && sync.needs_candidate_facts());

    for s in states() {
        let obs = encode_observation_v1(&s);
        let legal = s.legal_actions();
        let facts = candidate_facts(&s);
        let req = |f| EvalRequest {
            observation: &obs,
            legal: &legal,
            side_to_move: s.side_to_move(),
            facts: f,
        };
        let a = sync.evaluate(req(Some(&facts))).unwrap();
        let b = batched.evaluate(req(Some(&facts))).unwrap();
        assert_eq!(a.policy.len(), legal.len());
        for (x, y) in a.policy.iter().zip(&b.policy) {
            assert!((x - y).abs() < 1e-5, "sync {x} vs batched {y}");
        }
        assert!((a.value - b.value).abs() < 1e-6);
        // Missing or misaligned facts are errors on both paths.
        assert!(matches!(
            batched.evaluate(req(None)),
            Err(EvalError::Invalid(_))
        ));
        assert!(matches!(
            sync.evaluate(req(None)),
            Err(EvalError::Invalid(_))
        ));
        assert!(matches!(
            batched.evaluate(req(Some(&facts[..facts.len() - 1]))),
            Err(EvalError::Invalid(_))
        ));
    }
    owner.shutdown();
}

#[test]
fn multi_leaf_puct_runs_end_to_end_with_facts_and_shares_batches() {
    let device = Default::default();
    let model = CandidateV25Model::<B>::new(cand_cfg(), &device);
    let owner = InferenceOwner::spawn(
        BatchedModel::new(model, 1, device),
        InferenceConfig::default(),
    );
    let ev = owner.evaluator();
    let cfg = PuctConfig {
        c_puct: 1.5,
        simulations: 24,
        leaves_in_flight: 4,
        solver: false,
    };
    let root = ChessGame::new(GameState::startpos(), &ev);
    let r = search(root, &cfg).expect("search over the candidate model");
    assert!(r.total_visits > 0);
    assert!(r.root_value.is_finite());
    let m = owner.metrics().snapshot();
    assert_eq!(m.errors, 0);
    assert!(m.completed >= 24, "{m:?}");
    owner.shutdown();
}

#[test]
fn probe_evaluators_never_request_facts() {
    let device = Default::default();
    let mut cfg = cand_cfg();
    cfg.architecture = Default::default();
    cfg.candidate = None;
    cfg.core_blocks = 2;
    let model = ProbeModel::<B>::new(cfg, &device);
    let owner = InferenceOwner::spawn(
        BatchedModel::new(model, 1, device),
        InferenceConfig::default(),
    );
    let ev = owner.evaluator();
    assert!(!ev.needs_candidate_facts());
    let s = GameState::startpos();
    let root = ChessGame::new(s, &ev);
    let cfg = PuctConfig {
        c_puct: 1.5,
        simulations: 8,
        leaves_in_flight: 2,
        solver: false,
    };
    search(root, &cfg).unwrap();
    owner.shutdown();
}
