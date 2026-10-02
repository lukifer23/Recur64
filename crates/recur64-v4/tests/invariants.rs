//! V4 architectural invariants (CPU, FP32). These are hard correctness gates that must pass
//! before any training. They test the mechanism, not chess strength. Numbering follows
//! `docs/V4_RESEARCH_PLAN.md` section 8.

use burn::optim::{GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::TensorData;

use recur64_core::GameState;
use recur64_model::active::coverage::gradient_coverage;
use recur64_model::active::{EdgeRef, QueryScript, ScriptStep, Tree};
use recur64_model::candidate::CandidateInputs;
use recur64_model::checkpoint::{CheckpointMeta, load_training, save_training};
use recur64_model::config::{Architecture, CandidateConfig, ModelConfig};
use recur64_model::net::NeuralModel;
use recur64_model::train::{CpuTrainBackend, adamw};

use recur64_v4::accounting::Accounting;
use recur64_v4::content::{ACTION_CONTENT, ContentBatch, FLAG_CONTENT, action_content};
use recur64_v4::ledger::EvidenceLedger;
use recur64_v4::model::EvidenceBeliefModel;
use recur64_v4::session::{Freeze, RunOptions, Selection, Session, V4Output};
use recur64_v4::utility::FrontierEdgeView;

type B = burn::backend::Flex;
type TB = CpuTrainBackend;

/// The backend RNG is process-global: tests that build models must not interleave.
static RNG: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn lock() -> std::sync::MutexGuard<'static, ()> {
    RNG.lock().unwrap_or_else(|e| e.into_inner())
}

fn tiny() -> ModelConfig {
    let mut c = ModelConfig::evidence_belief_v4();
    c.width = 32;
    c.heads = 4;
    c.ffn = 64;
    c.core_blocks = 2;
    let e = c.evidence.as_mut().unwrap();
    e.candidate = CandidateConfig {
        dim: 16,
        heads: 2,
        ffn: 32,
        blocks: 1,
        facts_hidden: 8,
        policy_hidden: 16,
        facts_enabled: true,
    };
    e.base_hidden = 16;
    e.query_heads = 2;
    e.query_ffn = 32;
    e.query_blocks = 1;
    e.content_dim = 16;
    e.content_heads = 2;
    e.content_ffn = 32;
    e.content_blocks = 1;
    e.message_dim = 16;
    e.pair_dim = 16;
    e.key_dim = 8;
    e.trust_hidden = 8;
    e.utility_hidden = 16;
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

fn v<Bk: Backend, const D: usize>(t: &Tensor<Bk, D>) -> Vec<f32> {
    t.clone().into_data().to_vec::<f32>().unwrap()
}

fn model() -> EvidenceBeliefModel<B> {
    EvidenceBeliefModel::<B>::new(tiny(), &Default::default())
}

/// Queries the first `n` root actions (ascending ActionId) of each example, in forward or
/// reverse order: the same evidence SET acquired in two different orders.
struct RootSetScript {
    n: usize,
    reverse: bool,
}

impl QueryScript for RootSetScript {
    fn next(
        &mut self,
        _example: usize,
        step: usize,
        frontier: &[EdgeRef],
        tree: &Tree,
    ) -> anyhow::Result<ScriptStep> {
        let set = &tree.node(0).legal[..self.n];
        let want = if self.reverse {
            set[self.n - 1 - step]
        } else {
            set[step]
        };
        let follow = frontier
            .iter()
            .position(|e| e.node_slot == 0 && e.action == want)
            .ok_or_else(|| anyhow::anyhow!("action {want} is not on the frontier at step {step}"))?;
        Ok(ScriptStep {
            follow,
            targets: Vec::new(),
        })
    }
}

/// Always extends the deepest known node (a chain of replies), so evaluation-only ablations can
/// replay paths that go below depth 1.
struct DeepScript;

impl QueryScript for DeepScript {
    fn next(
        &mut self,
        _example: usize,
        _step: usize,
        frontier: &[EdgeRef],
        _tree: &Tree,
    ) -> anyhow::Result<ScriptStep> {
        let follow = frontier
            .iter()
            .enumerate()
            .max_by_key(|(_, e)| e.parent_depth)
            .map(|(i, _)| i)
            .unwrap();
        Ok(ScriptStep {
            follow,
            targets: Vec::new(),
        })
    }
}

fn run_fixed(m: &EvidenceBeliefModel<B>, budget: usize) -> V4Output<B> {
    m.run(
        &roots(),
        &RunOptions::new(budget),
        Selection::Fixed,
        0,
        &Default::default(),
    )
    .unwrap()
}

// ---- 1, 2, 15: B0 isolation ------------------------------------------------------------------

#[test]
fn b0_is_independent_of_the_query_budget() {
    let _g = lock();
    let m = model();
    let z: Vec<Vec<f32>> = [0usize, 2, 4, 8]
        .iter()
        .map(|&b| v(&run_fixed(&m, b).z0))
        .collect();
    for w in z.windows(2) {
        assert_eq!(w[0], w[1], "B0 logits changed with the query budget");
    }
}

#[test]
fn b0_logits_are_exactly_the_base_logits() {
    let _g = lock();
    let m = model();
    let device = Default::default();
    let rs = roots();
    let inputs = CandidateInputs::<B>::from_states(&rs, &device).unwrap();
    let stage = m.base_stage(inputs.board.clone(), &inputs.cands, inputs.facts.clone());
    let out0 = run_fixed(&m, 0);
    assert_eq!(v(&out0.logits), v(&stage.z0), "budget 0 is not literally the base tower");
    assert_eq!(v(&out0.delta), vec![0.0; out0.delta.dims().iter().product()]);
    // The trait path (used by generic tooling) is B0 too.
    let via_trait = m.forward_inputs(inputs.board, &inputs.cands, Some(inputs.facts), 1, false);
    assert_eq!(
        v(&via_trait.readouts[0].policy.log_probs),
        v(&out0.base_log_probs)
    );
}

#[test]
fn deleting_all_evidence_reproduces_b0() {
    let _g = lock();
    let m = model();
    let device = Default::default();
    let mut s = Session::new(&m, &roots(), RunOptions::new(4), 0, &device).unwrap();
    s.run_all(&mut Selection::Fixed).unwrap();
    assert!(s.ledger().slots() == 4, "evidence was not acquired");
    let with = s.belief();
    let empty = EvidenceLedger::<B>::new(s.batch(), m.evidence().message_dim);
    let without = s.belief_for(&empty);
    assert_eq!(v(&without.z), v(s.z0()), "an empty ledger did not give back z0");
    assert_ne!(v(&with.z), v(s.z0()), "real evidence left the belief at z0");
}

// ---- 3, 4, 5: content causality ------------------------------------------------------------

fn content_batch(m: &EvidenceBeliefModel<B>, n: usize) -> ContentBatch<B> {
    let _ = m;
    let device = Default::default();
    let rs = roots();
    let obs: Vec<Vec<f32>> = rs
        .iter()
        .map(|r| recur64_core::encode_observation_v1(r).as_slice().to_vec())
        .collect();
    let acts: Vec<u16> = (0..n).map(|i| rs[i % rs.len()].legal_actions()[0].index() as u16).collect();
    let pick = |i: usize| obs[i % obs.len()].as_slice();
    let root: Vec<&[f32]> = (0..n).map(pick).collect();
    let parent: Vec<&[f32]> = (0..n).map(pick).collect();
    let child: Vec<&[f32]> = (0..n).map(|i| pick(i + 1)).collect();
    ContentBatch::<B>::from_host(&root, &parent, &child, &acts, &vec![[0.0, 1.0]; n], &device)
        .unwrap()
}

#[test]
fn zeroed_query_content_produces_a_bitwise_zero_message() {
    let _g = lock();
    let m = model();
    let c = content_batch(&m, 3);
    let real = v(&m.encode_content(&c));
    assert!(real.iter().any(|x| x.abs() > 1e-6), "real content gave a zero message");
    let zero = v(&m.encode_content(&c.zeroed()));
    assert!(zero.iter().all(|&x| x == 0.0), "zero content gave a non-zero message: {zero:?}");
}

#[test]
fn a_zero_message_gives_an_identity_belief_update_for_any_routing() {
    let _g = lock();
    let m = model();
    let device = Default::default();
    let rs = roots();
    let inputs = CandidateInputs::<B>::from_states(&rs, &device).unwrap();
    let stage = m.base_stage(inputs.board.clone(), &inputs.cands, inputs.facts.clone());
    let md = m.evidence().message_dim;
    let b = rs.len();
    // Zero messages with arbitrary, mutually different routing metadata.
    for routing in [
        (vec![0usize, 1, 2], vec![1u32, 2, 3]),
        (vec![5, 7, 0], vec![4, 1, 8]),
        (vec![3, 3, 3], vec![2, 2, 2]),
    ] {
        let mut ledger = EvidenceLedger::<B>::new(b, md);
        ledger
            .append(
                &device,
                &[0, 1, 2],
                Tensor::<B, 2>::zeros([3, md], &device),
                &routing.0,
                &routing.1,
            )
            .unwrap();
        let out = m.belief_update(stage.tokens.clone(), inputs.cands.mask.clone(), &ledger);
        let delta = v(&out.delta);
        assert!(delta.iter().all(|&d| d == 0.0), "zero messages moved the belief: {delta:?}");
        let z = v(&(stage.z0.clone() + out.delta));
        assert_eq!(z, v(&stage.z0), "z0 + zero delta is not bitwise z0");
    }
}

#[test]
fn zero_content_replay_returns_exactly_to_b0_whatever_the_path() {
    let _g = lock();
    let m = model();
    let device = Default::default();
    let mut results = Vec::new();
    for reverse in [false, true] {
        let out = m
            .run(
                &roots(),
                &RunOptions::zero_content_eval(3),
                Selection::Script(&mut RootSetScript { n: 3, reverse }),
                0,
                &device,
            )
            .unwrap();
        assert_eq!(v(&out.logits), v(&out.z0), "zero-content replay left B0");
        results.push(v(&out.logits));
    }
    assert_eq!(results[0], results[1], "routing alone changed the policy");
    // Zero/shuffled content is refused outside an external replay script.
    for sel in [Selection::Fixed, Selection::Utility, Selection::Random(1)] {
        let e = m
            .run(&roots(), &RunOptions::zero_content_eval(2), sel, 0, &device)
            .err()
            .expect("must refuse");
        assert!(e.to_string().contains("evaluation-only"), "{e}");
    }
}

// ---- 6: permutation invariance ---------------------------------------------------------------

#[test]
fn the_same_evidence_acquired_in_another_order_gives_the_same_belief() {
    let _g = lock();
    let m = model();
    let device = Default::default();
    let a = m
        .run(
            &roots(),
            &RunOptions::new(3),
            Selection::Script(&mut RootSetScript { n: 3, reverse: false }),
            0,
            &device,
        )
        .unwrap();
    let b = m
        .run(
            &roots(),
            &RunOptions::new(3),
            Selection::Script(&mut RootSetScript { n: 3, reverse: true }),
            0,
            &device,
        )
        .unwrap();
    let (za, zb) = (v(&a.logits), v(&b.logits));
    let worst = za
        .iter()
        .zip(&zb)
        .filter(|(x, y)| x.abs() < 1e8 && y.abs() < 1e8)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(worst < 1e-5, "acquisition order changed the belief by {worst}");
    assert!(
        a.delta.clone().abs().max().into_scalar() > 1e-7,
        "no evidence effect to be invariant about"
    );
}

#[test]
fn permuting_ledger_slots_does_not_change_the_belief() {
    let _g = lock();
    let m = model();
    let device = Default::default();
    let mut s = Session::new(&m, &roots(), RunOptions::new(4), 0, &device).unwrap();
    s.run_all(&mut Selection::Fixed).unwrap();
    let base = v(&s.belief().z);
    for perm in [vec![3usize, 2, 1, 0], vec![1, 3, 0, 2], vec![2, 0, 3, 1]] {
        let p = s.ledger().permuted(&perm, &device).unwrap();
        let got = v(&s.belief_for(&p).z);
        let worst = base
            .iter()
            .zip(&got)
            .filter(|(x, _)| x.abs() < 1e8)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-5, "slot permutation {perm:?} changed the belief by {worst}");
    }
}

// ---- 7, 8: content matters -------------------------------------------------------------------

#[test]
fn real_content_changes_evidence_and_unrelated_content_gives_a_different_result() {
    let _g = lock();
    let m = model();
    let device = Default::default();
    let mut diag = RunOptions::new(3);
    diag.diagnostics = true;
    let normal = m
        .run(
            &roots(),
            &diag,
            Selection::Script(&mut RootSetScript { n: 3, reverse: false }),
            0,
            &device,
        )
        .unwrap();
    assert!(normal.messages.iter().all(|x| x.norm.unwrap() > 0.0), "a real query gave a null message");
    assert!(normal.messages.iter().all(|x| x.trust.unwrap() > 0.0));
    let zero = m
        .run(
            &roots(),
            &RunOptions::zero_content_eval(3),
            Selection::Script(&mut RootSetScript { n: 3, reverse: false }),
            0,
            &device,
        )
        .unwrap();
    let shuffled = m
        .run(
            &roots(),
            &RunOptions::shuffled_content_eval(3),
            Selection::Script(&mut RootSetScript { n: 3, reverse: false }),
            0,
            &device,
        )
        .unwrap();
    let max_abs_diff = |a: &Tensor<B, 2>, b: &Tensor<B, 2>| {
        v(a).iter()
            .zip(v(b))
            .filter(|(x, _)| x.abs() < 1e8)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };
    assert!(max_abs_diff(&normal.logits, &zero.logits) > 1e-7, "content did not change the belief");
    assert!(
        max_abs_diff(&normal.logits, &shuffled.logits) > 1e-7,
        "unrelated content gave the same result as the real content"
    );
}

// ---- 9: boundedness --------------------------------------------------------------------------

#[test]
fn evidence_deltas_are_finite_and_bounded_even_for_extreme_content() {
    let _g = lock();
    let m = model();
    let device = Default::default();
    let bound = m.evidence().delta_bound as f32;
    let out = run_fixed(&m, 8);
    assert!(v(&out.delta).iter().all(|d| d.is_finite() && d.abs() <= bound));
    // Adversarial content: observations scaled to +-1e4 and a random-sign pattern.
    let rs = roots();
    let inputs = CandidateInputs::<B>::from_states(&rs, &device).unwrap();
    let stage = m.base_stage(inputs.board.clone(), &inputs.cands, inputs.facts.clone());
    let c = content_batch(&m, 3);
    let scale = |t: Tensor<B, 3>, s: f32| t.mul_scalar(s);
    for s in [1e2f32, 1e4, -1e4] {
        let big = ContentBatch::<B> {
            root: scale(c.root.clone(), s),
            parent: scale(c.parent.clone(), s),
            child: scale(c.child.clone(), -s),
            action: c.action.clone(),
            flags: c.flags.clone(),
        };
        let msg = m.encode_content(&big);
        assert!(v(&msg).iter().all(|x| x.is_finite()), "non-finite message at scale {s}");
        let mut ledger = EvidenceLedger::<B>::new(3, m.evidence().message_dim);
        ledger.append(&device, &[0, 1, 2], msg, &[0, 1, 2], &[1, 1, 1]).unwrap();
        let d = m
            .belief_update(stage.tokens.clone(), inputs.cands.mask.clone(), &ledger)
            .delta;
        assert!(v(&d).iter().all(|x| x.is_finite() && x.abs() <= bound), "delta escaped its bound at scale {s}");
    }
}

// ---- 10: parameter count ---------------------------------------------------------------------

#[test]
fn parameter_count_is_independent_of_the_budget_and_groups_sum_exactly() {
    let _g = lock();
    let m = model();
    let before = m.num_params();
    for b in [0usize, 2, 4, 8] {
        let _ = run_fixed(&m, b);
        assert_eq!(m.num_params(), before, "parameter count changed after a B{b} run");
    }
    let groups = m.param_breakdown();
    assert_eq!(groups.iter().map(|(_, c)| c).sum::<usize>(), before);
    let (base, evidence, utility) = m.group_counts();
    assert_eq!(base + evidence + utility, before, "parameter groups do not cover the model");
    assert!(evidence > 0 && utility > 0 && base > evidence);
    // The full production geometry is within the 20-40M target and never above it.
    let full = EvidenceBeliefModel::<B>::new(ModelConfig::evidence_belief_v4(), &Default::default());
    let n = full.num_params();
    eprintln!("full evidence_belief_v4 parameters: {n}");
    for (name, c) in full.param_breakdown() {
        eprintln!("{name:34} {c:>12}");
    }
    assert!((20_000_000..=40_000_000).contains(&n), "full model has {n} parameters");
}

// ---- 11, 12: accounting and StateQuery-only --------------------------------------------------

#[test]
fn statequery_accounting_is_exact() {
    let _g = lock();
    let m = model();
    let out = run_fixed(&m, 4);
    let a = &out.accounting;
    assert_eq!(a.successful_queries, vec![4, 4, 4]);
    assert_eq!(a.total_successful_queries, 12);
    assert_eq!(a.state_transitions, 12);
    assert_eq!(a.root_encoder_runs, 1);
    assert_eq!(a.evidence_encoder_examples, 12);
    assert_eq!(a.ledger_messages, vec![4, 4, 4]);
    assert_eq!(a.probe_queries, 0);
    a.check_invariants().unwrap();
    // The invariant checker really detects each kind of leak.
    let mut bad = a.clone();
    bad.state_transitions += 1;
    assert!(bad.check_invariants().is_err());
    let mut bad: Accounting = a.clone();
    bad.ledger_messages[0] -= 1;
    assert!(bad.check_invariants().is_err());
    let mut bad = a.clone();
    bad.evidence_encoder_examples -= 1;
    assert!(bad.check_invariants().is_err());
}

#[test]
fn every_queried_state_comes_only_through_statequery() {
    let _g = lock();
    let m = model();
    let device = Default::default();
    let mut s = Session::new(&m, &roots(), RunOptions::new(3), 0, &device).unwrap();
    s.run_all(&mut Selection::Fixed).unwrap();
    for e in 0..s.batch() {
        let q = s.managers()[e].successful_queries() as usize;
        assert_eq!(q, 3);
        assert_eq!(s.trees()[e].len(), 1 + q, "tree holds a node the tool did not return");
        assert_eq!(s.ledger().count(e), q, "ledger holds a message the tool did not return");
    }
}

// ---- 13: the utility head cannot see the unseen child ---------------------------------------

#[test]
fn utility_scoring_executes_no_query_and_takes_only_parent_known_inputs() {
    let _g = lock();
    let m = model();
    let device = Default::default();
    let mut s = Session::new(&m, &roots(), RunOptions::new(4), 0, &device).unwrap();
    s.advance(&mut Selection::Utility).unwrap();
    let fronts = s.frontiers();
    let before: Vec<u32> = s.managers().iter().map(|m| m.successful_queries()).collect();
    let (scores_a, view) = s.utilities(&fronts).unwrap();
    let (scores_b, _) = s.utilities(&fronts).unwrap();
    let after: Vec<u32> = s.managers().iter().map(|m| m.successful_queries()).collect();
    assert_eq!(before, after, "scoring a frontier executed a query");
    assert_eq!(v(&scores_a), v(&scores_b), "scoring is not a pure function of the current state");
    // Exhaustive destructuring: adding any field (for example a child tensor) breaks this test
    // and forces a review of invariant 13.
    let FrontierEdgeView {
        edge_emb: _,
        parent_pool: _,
        branch_token: _,
        belief: _,
        ledger_summary: _,
        edge_feats: _,
        mask: _,
    } = view;
    assert_eq!(s.accounting().probe_queries, 0);
}

// ---- 14: probes cannot contaminate the real trajectory ---------------------------------------

#[test]
fn counterfactual_forks_do_not_contaminate_the_real_trajectory() {
    let _g = lock();
    let m = model();
    let device = Default::default();
    let drive = |probe: bool| -> (Vec<f32>, Vec<u32>, Accounting) {
        let mut s = Session::new(&m, &roots(), RunOptions::new(4), 0, &device).unwrap();
        s.advance(&mut Selection::Fixed).unwrap();
        s.advance(&mut Selection::Fixed).unwrap();
        if probe {
            let fronts = s.frontiers();
            let tree_len: Vec<usize> = s.trees().iter().map(Tree::len).collect();
            let queries: Vec<u32> = s.managers().iter().map(|m| m.successful_queries()).collect();
            let legal: Vec<u64> = s.managers().iter().map(|m| m.legal_moves_generated()).collect();
            let ledger = s.ledger().host_messages().unwrap();
            let belief = v(&s.belief().z);
            let picks: Vec<usize> = (0..fronts[0].len().min(5)).collect();
            let got = s.probe_edges(0, &fronts[0], &picks).unwrap();
            assert_eq!(got.len(), picks.len());
            assert_eq!(s.accounting().probe_queries, picks.len() as u64);
            assert_eq!(tree_len, s.trees().iter().map(Tree::len).collect::<Vec<_>>());
            assert_eq!(queries, s.managers().iter().map(|m| m.successful_queries()).collect::<Vec<_>>());
            assert_eq!(legal, s.managers().iter().map(|m| m.legal_moves_generated()).collect::<Vec<_>>());
            assert_eq!(ledger, s.ledger().host_messages().unwrap());
            assert_eq!(belief, v(&s.belief().z));
            // Probes of different edges return different exact children.
            let digests: std::collections::HashSet<_> =
                got.iter().map(|p| p.child_digest.clone()).collect();
            assert_eq!(digests.len(), got.len());
        }
        s.run_all(&mut Selection::Fixed).unwrap();
        let out = s.finish().unwrap();
        (v(&out.logits), out.accounting.successful_queries.iter().map(|&q| q as u32).collect(), out.accounting)
    };
    let (z_plain, q_plain, a_plain) = drive(false);
    let (z_probe, q_probe, a_probe) = drive(true);
    assert_eq!(z_plain, z_probe, "probing changed the real trajectory's final belief");
    assert_eq!(q_plain, q_probe);
    assert_eq!(a_plain.total_successful_queries, a_probe.total_successful_queries);
    assert_eq!(a_plain.probe_queries, 0);
    assert_eq!(a_probe.probe_queries, 5, "probe forks must be accounted separately");
    a_probe.check_invariants().unwrap();
}

// ---- 16: identities --------------------------------------------------------------------------

#[test]
fn v4_refuses_every_historical_identity_and_every_historical_identity_refuses_v4() {
    let _g = lock();
    let device = Default::default();
    // Building a V4 model from another architecture's config is refused.
    for other in [
        ModelConfig::active_search_v3(),
        ModelConfig::all_info_v1(),
        ModelConfig::candidate_v25(true),
        ModelConfig::legacy_facts_v25(),
    ] {
        let id = other.architecture.id();
        let e = <EvidenceBeliefModel<B> as NeuralModel<B>>::build(&other, &device)
            .err()
            .unwrap_or_else(|| panic!("V4 built from {id}"));
        assert!(!e.to_string().is_empty(), "{id}");
    }
    // The V4 config is refused by the historical refusal hook, visibly and before any work.
    let cfg = tiny();
    let e = cfg.refuse_evidence_v4("`bench`").err().expect("refused").to_string();
    assert!(e.contains("evidence_belief_v4") && e.contains("`bench`"), "{e}");
    // A V3/V3.5 identity is refused as a V4 checkpoint and vice versa (metadata level).
    for other in [ModelConfig::active_search_v3(), ModelConfig::all_info_v1()] {
        let meta = CheckpointMeta::new(other.clone(), 1, false, 0, 1e-3, 1, 0, "t", "fp32");
        assert!(meta.check_model(&cfg).is_err(), "{} accepted as V4", other.architecture.id());
        let v4 = CheckpointMeta::new(cfg.clone(), 1, false, 0, 1e-3, 1, 0, "t", "fp32");
        assert!(v4.check_model(&other).is_err(), "V4 accepted as {}", other.architecture.id());
    }
    assert_eq!(cfg.architecture, Architecture::EvidenceBeliefV4);
}

#[test]
fn a_v4_checkpoint_round_trips_and_a_tampered_contract_is_refused() {
    let _g = lock();
    let device = Default::default();
    let cfg = tiny();
    let m = EvidenceBeliefModel::<TB>::new(cfg.clone(), &device);
    let optim = adamw::<TB, EvidenceBeliefModel<TB>>();
    let dir = std::env::temp_dir().join(format!("recur64-v4-ckpt-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let meta = CheckpointMeta::new(cfg.clone(), 1, false, 0, 1e-3, 5101, 0, "v4-test", "fp32");
    assert_eq!(meta.architecture, "evidence_belief_v4");
    assert!(meta.evidence_contracts.is_some() && meta.active_contracts.is_none());
    save_training::<TB, _, _>(&dir, &m, &optim, &meta).unwrap();
    let template = EvidenceBeliefModel::<TB>::new(cfg.clone(), &device);
    let (loaded, _o, got) = load_training::<TB, _, _>(
        &dir,
        template,
        adamw::<TB, EvidenceBeliefModel<TB>>(),
        &device,
    )
    .unwrap();
    assert_eq!(got.seed, 5101);
    let rs = roots();
    let inputs = CandidateInputs::<TB>::from_states(&rs, &device).unwrap();
    let z = |m: &EvidenceBeliefModel<TB>| {
        v(&m.base_stage(inputs.board.clone(), &inputs.cands, inputs.facts.clone()).z0)
    };
    assert_eq!(z(&loaded), z(&m), "the loaded V4 model computes a different base belief");

    // A V3 template refuses the V4 directory explicitly.
    let active = recur64_model::active::ActiveSearchModel::<TB>::new(
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
    let e = load_training::<TB, _, _>(
        &dir,
        active,
        adamw::<TB, recur64_model::active::ActiveSearchModel<TB>>(),
        &device,
    )
    .err()
    .expect("must refuse")
    .to_string();
    assert!(e.contains("evidence_belief_v4") || e.contains("cross-architecture"), "{e}");

    // A tampered contract in the metadata is refused.
    let mp = dir.join("meta.json");
    let mut j: serde_json::Value = serde_json::from_slice(&std::fs::read(&mp).unwrap()).unwrap();
    j["evidence_contracts"]["belief_update"] = "belief_update_gated_sum_v2".into();
    std::fs::write(&mp, serde_json::to_vec(&j).unwrap()).unwrap();
    let template = EvidenceBeliefModel::<TB>::new(cfg, &device);
    assert!(
        load_training::<TB, _, _>(&dir, template, adamw::<TB, EvidenceBeliefModel<TB>>(), &device)
            .is_err()
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// ---- Law A under optimisation ----------------------------------------------------------------

#[test]
fn a_frozen_base_gets_no_gradient_evidence_gets_gradient_and_b0_is_bit_identical_after_updates() {
    let _g = lock();
    let device = Default::default();
    let cfg = tiny();
    let mut m = EvidenceBeliefModel::<TB>::new(cfg, &device);
    let mut optim = adamw::<TB, EvidenceBeliefModel<TB>>();
    let rs = roots();
    let b0 = |m: &EvidenceBeliefModel<TB>| {
        let o = m
            .run(&rs, &RunOptions::new(0), Selection::Fixed, 0, &device)
            .unwrap();
        v(&o.z0)
    };
    let before = b0(&m);
    for update in 0..3 {
        let out = m
            .run(
                &rs,
                &RunOptions::new(4).with_freeze(Freeze::BASE),
                Selection::Fixed,
                0,
                &device,
            )
            .unwrap();
        let [b, w] = out.log_probs.dims();
        let mut t = vec![0.0f32; b * w];
        for e in 0..b {
            t[e * w] = 1.0;
        }
        let target = Tensor::<TB, 2>::from_data(TensorData::new(t, [b, w]), &device);
        let loss = -(out.log_probs.clone() * target).sum() / b as f32;
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &m);
        if update == 0 {
            let cov = gradient_coverage::<TB, _>(&m, &grads);
            for row in &cov {
                if row.name.starts_with("base.") {
                    assert!(!row.has_grad, "frozen base parameter {} received a gradient", row.name);
                }
                if row.name.starts_with("encoder.") || row.name.starts_with("update.") {
                    assert!(
                        row.has_grad && row.finite && row.nonzero,
                        "evidence parameter {} got no usable gradient ({row:?})",
                        row.name
                    );
                }
            }
            assert!(cov.iter().any(|r| r.name.starts_with("encoder.") && r.nonzero));
        }
        m = optim.step(1e-2, m, grads);
    }
    assert_eq!(before, b0(&m), "evidence-only updates changed B0");
    // Sanity: the evidence actually learned something (the belief at B4 moved).
    let after = m
        .run(&rs, &RunOptions::new(4), Selection::Fixed, 0, &device)
        .unwrap();
    assert!(v(&after.delta).iter().any(|d| d.abs() > 1e-6));
    let _ = (ACTION_CONTENT, FLAG_CONTENT, action_content(0));
}

#[test]
fn zero_content_along_the_deep_breadth_first_path_also_returns_exactly_to_b0() {
    let _g = lock();
    let m = model();
    // A chain of replies: depth 1, 2, 3 ... are all queried.
    let out = m
        .run(
            &roots(),
            &RunOptions::zero_content_eval(6),
            Selection::Script(&mut DeepScript),
            0,
            &Default::default(),
        )
        .unwrap();
    assert!(out.traces[0].iter().any(|t| t.depth >= 3), "the path never went below depth 2");
    assert_eq!(v(&out.logits), v(&out.z0));
    // Real content on the same path does move the belief.
    let real = m
        .run(
            &roots(),
            &RunOptions::new(6),
            Selection::Script(&mut DeepScript),
            0,
            &Default::default(),
        )
        .unwrap();
    assert_ne!(v(&real.logits), v(&real.z0));
}
