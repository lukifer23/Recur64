//! P2 engineering correctness for `active_search_v3` (CPU, FP32).
//!
//! These are engineering checks, not science: finite forward at every budget,
//! exact accounting, root-once, strict identity, gradient coverage on the first
//! update, deterministic traces, and the frozen FIXED comparator.

use burn::module::{AutodiffModule, Module, ModuleVisitor, Param};
use burn::optim::{GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

use recur64_core::{ActionId, GameState};
use recur64_model::active::loss::selector_loss;
use recur64_model::active::{
    ActiveSearchModel, EdgeRef, QueryScript, RunOptions, ScriptStep, Selection, Tree,
};
use recur64_model::checkpoint::{CheckpointMeta, load_training, save_training};
use recur64_model::config::{ActiveContracts, Architecture, CandidateConfig, ModelConfig};
use recur64_model::loss::{policy_ce, wdl_ce};
use recur64_model::net::NeuralModel;
use recur64_model::train::{CpuTrainBackend, adamw};

type B = burn::backend::Flex;
type TB = CpuTrainBackend;

fn tiny() -> ModelConfig {
    let mut c = ModelConfig::active_search_v3();
    c.width = 32;
    c.heads = 4;
    c.ffn = 64;
    c.core_blocks = 2;
    let a = c.active.as_mut().unwrap();
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
    a.workspace_tokens = 4;
    a.planner_heads = 2;
    a.planner_ffn = 32;
    a.selector_hidden = 16;
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

fn opts(budget: usize) -> RunOptions {
    RunOptions::forced(budget)
}

fn log_probs<Bk: Backend>(out: &recur64_model::active::ActiveOutput<Bk>) -> Vec<f32> {
    out.readout
        .policy
        .log_probs
        .clone()
        .into_data()
        .to_vec::<f32>()
        .unwrap()
}

/// Test double of the `QueryScript` interface: follow the first frontier edge and
/// supervise the first two. Used only to exercise the interface and the losses.
struct FirstTwo;
impl QueryScript for FirstTwo {
    fn next(
        &mut self,
        _example: usize,
        _step: usize,
        frontier: &[EdgeRef],
        _tree: &Tree,
    ) -> anyhow::Result<ScriptStep> {
        Ok(ScriptStep {
            follow: 0,
            targets: (0..frontier.len().min(2)).collect(),
        })
    }
}

#[test]
fn full_geometry_parameter_count_and_breakdown() {
    let device = Default::default();
    let m = ActiveSearchModel::<B>::new(ModelConfig::active_search_v3(), &device);
    let total = m.num_params();
    let parts: usize = m.param_breakdown().iter().map(|(_, n)| n).sum();
    assert_eq!(total, parts, "breakdown must sum to the parameter count");
    for (name, n) in m.param_breakdown() {
        eprintln!("  {name:32} {n:>10}");
    }
    eprintln!("active_search_v3 total unique parameters: {total}");
    assert!(
        (27_000_000..=35_000_000).contains(&total),
        "total {total} outside the V3.0 sanity range"
    );
    eprintln!(
        "stop-head parameters (zero gradient while masked): {}",
        m.stop_head_params()
    );
}

#[test]
fn budget_zero_is_the_trait_forward_and_does_no_query_work() {
    let device = Default::default();
    let m = ActiveSearchModel::<B>::new(tiny(), &device);
    let states = roots();
    let out = m
        .run(&states, &opts(0), Selection::Active, &device)
        .unwrap();
    let a = &out.accounting;
    assert_eq!(a.total_successful_queries, 0);
    assert_eq!((a.query_encoder_calls, a.planner_update_calls), (0, 0));
    assert_eq!(a.root_encoder_runs, 1);
    assert_eq!(a.state_transitions, 0);
    a.check_invariants().unwrap();
    // The NeuralModel trait path is explicitly budget 0 and must agree exactly.
    let inp =
        recur64_model::candidate::CandidateInputs::<B>::from_states(&states, &device).unwrap();
    let t = NeuralModel::<B>::forward_inputs(&m, inp.board, &inp.cands, Some(inp.facts), 1, false);
    let via_trait = t.readouts[0]
        .policy
        .log_probs
        .clone()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    assert_eq!(log_probs(&out), via_trait);
}

#[test]
fn finite_outputs_and_exact_accounting_at_every_budget() {
    let device = Default::default();
    let m = ActiveSearchModel::<B>::new(tiny(), &device);
    let states = roots();
    let b = states.len();
    let params_before = m.num_params();
    for budget in [0usize, 2, 4, 8, 16] {
        let out = m
            .run(&states, &opts(budget), Selection::Active, &device)
            .unwrap();
        let a = &out.accounting;
        a.check_invariants()
            .unwrap_or_else(|e| panic!("B{budget}: {e}"));
        assert!(
            log_probs(&out).iter().all(|v| v.is_finite()),
            "B{budget} non-finite"
        );
        assert_eq!(a.requested_budget, budget);
        assert_eq!(
            a.successful_queries,
            vec![budget; b],
            "B{budget} forced budget"
        );
        assert_eq!(a.total_successful_queries, budget * b);
        assert_eq!(a.query_encoder_examples, budget * b);
        assert_eq!(a.planner_update_examples, budget * b);
        assert_eq!(a.query_encoder_calls, budget);
        assert_eq!(a.planner_update_calls, budget);
        assert_eq!(a.state_transitions, budget * b);
        assert_eq!(a.unique_nodes, b * (1 + budget));
        assert_eq!(
            a.root_encoder_runs, 1,
            "root encoder exactly once at B{budget}"
        );
        assert_eq!(a.stop_calls, 0);
        assert_eq!(a.exhausted_examples, 0);
        assert!(a.legal_moves_generated > 0 || budget == 0);
        assert_eq!(
            out.traces.iter().map(Vec::len).collect::<Vec<_>>(),
            vec![budget; b]
        );
        // Root CandidateFacts are baseline work, independent of the budget.
        assert_eq!(a.root_facts_positions, b);
    }
    assert_eq!(
        m.num_params(),
        params_before,
        "parameter count is budget independent"
    );
}

#[test]
fn measured_executions_match_the_accounting_at_every_budget() {
    use recur64_model::active::counters::snapshot;
    let device = Default::default();
    let m = ActiveSearchModel::<B>::new(tiny(), &device);
    let states = roots();
    for budget in [0usize, 2, 4, 8, 16] {
        let before = snapshot();
        let out = m
            .run(&states, &opts(budget), Selection::Active, &device)
            .unwrap();
        let ran = snapshot().since(before);
        // The heavy root encoder genuinely ran once, whatever the budget.
        assert_eq!(ran.root_stage, 1, "B{budget}: root stage executions");
        // One batched query-encoder call and one planner update per round.
        assert_eq!(ran.query_encoder, budget, "B{budget}: query encoder calls");
        assert_eq!(ran.planner_update, budget, "B{budget}: planner updates");
        let a = &out.accounting;
        assert_eq!(ran.root_stage, a.root_encoder_runs);
        assert_eq!(ran.query_encoder, a.query_encoder_calls);
        assert_eq!(ran.planner_update, a.planner_update_calls);
    }
}

#[test]
fn queries_change_the_policy_and_each_query_is_a_distinct_edge() {
    let device = Default::default();
    let m = ActiveSearchModel::<B>::new(tiny(), &device);
    let states = roots();
    let b0 = m
        .run(&states, &opts(0), Selection::Active, &device)
        .unwrap();
    let b4 = m
        .run(&states, &opts(4), Selection::Active, &device)
        .unwrap();
    assert_ne!(
        log_probs(&b0),
        log_probs(&b4),
        "acquired evidence must reach the readout"
    );
    for (e, trace) in b4.traces.iter().enumerate() {
        let mut seen = std::collections::HashSet::new();
        for q in trace {
            assert!(
                seen.insert((q.parent_slot, q.action)),
                "example {e}: edge queried twice"
            );
        }
    }
}

#[test]
fn traces_are_deterministic_for_every_selection_mode() {
    let device = Default::default();
    let m = ActiveSearchModel::<B>::new(tiny(), &device);
    let states = roots();
    let trace = |sel: Selection<'_>| -> Vec<Vec<(usize, u16)>> {
        m.run(&states, &opts(6), sel, &device)
            .unwrap()
            .traces
            .iter()
            .map(|t| t.iter().map(|q| (q.parent_slot, q.action)).collect())
            .collect()
    };
    assert_eq!(trace(Selection::Active), trace(Selection::Active));
    assert_eq!(trace(Selection::Fixed), trace(Selection::Fixed));
    assert_eq!(trace(Selection::Random(7)), trace(Selection::Random(7)));
    assert_ne!(trace(Selection::Random(7)), trace(Selection::Random(8)));
}

#[test]
fn fixed_schedule_is_canonical_bfs_and_independent_of_the_weights() {
    let device = Default::default();
    let states = roots();
    let run_with = |seed_model: u64| {
        // Two different weight draws: Burn seeds its global RNG per backend.
        <B as Backend>::seed(&device, seed_model);
        let m = ActiveSearchModel::<B>::new(tiny(), &device);
        m.run(&states, &opts(8), Selection::Fixed, &device).unwrap()
    };
    let a = run_with(1);
    let b = run_with(2);
    for (e, state) in states.iter().enumerate() {
        let ta: Vec<_> = a.traces[e]
            .iter()
            .map(|q| (q.parent_slot, q.action))
            .collect();
        let tb: Vec<_> = b.traces[e]
            .iter()
            .map(|q| (q.parent_slot, q.action))
            .collect();
        assert_eq!(ta, tb, "FIXED must not depend on the network");
        // With at least B legal root moves the schedule is the first B root moves
        // in ActionId order, all at depth 1 (the documented consequence).
        let legal: Vec<u16> = state
            .legal_actions()
            .iter()
            .map(|x| x.index() as u16)
            .collect();
        assert!(legal.len() >= 8);
        let want: Vec<(usize, u16)> = legal[..8].iter().map(|&x| (0usize, x)).collect();
        assert_eq!(ta, want, "example {e}");
        assert!(a.traces[e].iter().all(|q| q.depth == 1));
        // Branches are the root candidate indices 0..8.
        let branches: Vec<usize> = a.traces[e].iter().map(|q| q.branch).collect();
        assert_eq!(branches, (0..8).collect::<Vec<_>>());
    }
}

#[test]
fn a_script_that_reaches_a_terminal_node_loses_that_nodes_frontier() {
    // Qg8# mates; querying it yields a terminal child.
    let device = Default::default();
    let m = ActiveSearchModel::<B>::new(tiny(), &device);
    let fen = "k7/8/1K6/8/8/8/8/6Q1 w - - 0 1";
    let states = vec![GameState::from_fen(fen).unwrap()];
    struct MateFirst(u16);
    impl QueryScript for MateFirst {
        fn next(
            &mut self,
            _e: usize,
            step: usize,
            frontier: &[EdgeRef],
            _t: &Tree,
        ) -> anyhow::Result<ScriptStep> {
            let follow = if step == 0 {
                frontier
                    .iter()
                    .position(|x| x.action == self.0)
                    .expect("mating move is on the root frontier")
            } else {
                0
            };
            Ok(ScriptStep {
                follow,
                targets: vec![follow],
            })
        }
    }
    let gs = &states[0];
    let mating = gs
        .legal_actions()
        .into_iter()
        .find(|a| {
            let (f, t, p) = a.to_physical(gs.perspective());
            let mut g = gs.clone();
            g.apply(recur64_core::StandardMove::new(
                f,
                t,
                (!p.is_none()).then_some(p),
            ))
            .unwrap();
            g.is_terminal()
        })
        .expect("a mating move exists");
    let mut script = MateFirst(mating.index() as u16);
    let out = m
        .run(&states, &opts(3), Selection::Script(&mut script), &device)
        .unwrap();
    assert!(out.traces[0][0].terminal);
    assert_eq!(out.accounting.terminal_nodes, 1);
    // No later query may descend from the terminal child (slot 1).
    assert!(out.traces[0][1..].iter().all(|q| q.parent_slot != 1));
    out.accounting.check_invariants().unwrap();
}

#[test]
fn an_exhausted_frontier_is_reported_not_hidden() {
    // With a ply cap of 1 every child is terminal, so the frontier is exactly the
    // root's legal moves and a larger budget cannot be spent.
    let device = Default::default();
    let m = ActiveSearchModel::<B>::new(tiny(), &device);
    let states = vec![GameState::startpos().with_max_plies(1)];
    let n_root = states[0].legal_actions().len();
    assert_eq!(n_root, 20);
    let out = m
        .run(&states, &opts(24), Selection::Fixed, &device)
        .unwrap();
    let a = &out.accounting;
    assert_eq!(a.successful_queries, vec![n_root]);
    assert_eq!(a.exhausted_examples, 1);
    assert_eq!(a.steps_executed, n_root);
    assert_eq!(a.terminal_nodes, n_root);
    assert!(log_probs(&out).iter().all(|v| v.is_finite()));
}

#[test]
fn unsupported_requests_refuse_visibly() {
    let device = Default::default();
    let m = ActiveSearchModel::<B>::new(tiny(), &device);
    let states = roots();
    let mut o = opts(2);
    o.stop_masked = false;
    let e = m
        .run(&states, &o, Selection::Active, &device)
        .err()
        .unwrap();
    assert!(e.to_string().contains("STOP"), "{e}");
    let e = m
        .run(&states, &opts(10_000), Selection::Active, &device)
        .err()
        .unwrap();
    assert!(e.to_string().contains("budget"), "{e}");
    let terminal = vec![GameState::from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1").unwrap()];
    assert!(
        m.run(&terminal, &opts(1), Selection::Active, &device)
            .is_err()
    );
    assert!(m.run(&[], &opts(1), Selection::Active, &device).is_err());
}

// ---------------------------------------------------------------------------
// Gradient coverage
// ---------------------------------------------------------------------------

struct Coverage<'a, Bk: AutodiffBackend> {
    grads: &'a GradientsParams,
    path: Vec<String>,
    rows: Vec<(String, bool, bool, bool)>, // name, has grad, finite, nonzero
    _p: std::marker::PhantomData<Bk>,
}

impl<Bk: AutodiffBackend> Coverage<'_, Bk> {
    fn record<const D: usize>(&mut self, id: burn::module::ParamId) {
        let name = self.path.join(".");
        match self.grads.get::<Bk::InnerBackend, D>(id) {
            None => self.rows.push((name, false, true, false)),
            Some(g) => {
                let v = g.into_data().to_vec::<f32>().unwrap();
                self.rows.push((
                    name,
                    true,
                    v.iter().all(|x| x.is_finite()),
                    v.iter().any(|x| x.abs() > 1e-12),
                ));
            }
        }
    }
}

impl<Bk: AutodiffBackend> ModuleVisitor<Bk> for Coverage<'_, Bk> {
    fn enter_module(&mut self, name: &str, _c: &str) {
        self.path.push(name.to_string());
    }
    fn exit_module(&mut self, _name: &str, _c: &str) {
        self.path.pop();
    }
    fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<Bk, D>>) {
        self.record::<D>(p.id);
    }
}

fn targets_first_two<Bk: Backend>(
    out: &recur64_model::active::ActiveOutput<Bk>,
    device: &Bk::Device,
) -> (Tensor<Bk, 2>, Tensor<Bk, 1, Int>) {
    let [b, w] = out.readout.policy.mask.dims();
    let mut t = vec![0.0f32; b * w];
    for i in 0..b {
        t[i * w] = 0.5;
        t[i * w + 1] = 0.5;
    }
    (
        Tensor::from_data(burn::tensor::TensorData::new(t, [b, w]), device),
        Tensor::<Bk, 1, Int>::from_data(burn::tensor::TensorData::new(vec![1i32; b], [b]), device),
    )
}

#[test]
fn every_subsystem_gets_a_finite_nonzero_gradient_on_update_one() {
    let device = Default::default();
    let m = ActiveSearchModel::<TB>::new(tiny(), &device);
    let states = roots();
    let mut script = FirstTwo;
    let out = m
        .run(&states, &opts(3), Selection::Script(&mut script), &device)
        .unwrap();
    assert_eq!(out.selector_steps.len(), 3);
    let (pt, wt) = targets_first_two(&out, &device);
    let sel = selector_loss(&out.selector_steps).expect("selector loss");
    let loss = policy_ce(&out.readout.policy, &pt) + wdl_ce(&out.readout.wdl_logits, &wt) + sel;
    let grads = GradientsParams::from_grads(loss.backward(), &m);
    let mut cov = Coverage::<TB> {
        grads: &grads,
        path: Vec::new(),
        rows: Vec::new(),
        _p: std::marker::PhantomData,
    };
    m.visit(&mut cov);
    assert!(
        cov.rows.len() > 50,
        "visitor saw only {} parameters",
        cov.rows.len()
    );
    let mut bad = Vec::new();
    let mut stop_rows = 0;
    for (name, has, finite, nonzero) in &cov.rows {
        let is_stop = name.contains("stop_hidden") || name.contains("stop_out");
        if is_stop {
            stop_rows += 1;
            assert!(finite, "{name}: non-finite");
            assert!(
                !nonzero,
                "{name}: STOP is masked, its gradient must be exactly zero"
            );
        } else if !(*has && *finite && *nonzero) {
            bad.push(format!(
                "{name} has={has} finite={finite} nonzero={nonzero}"
            ));
        }
    }
    assert_eq!(stop_rows, 4, "stop head = two linears (weight+bias)");
    assert!(
        bad.is_empty(),
        "parameters without a finite non-zero gradient:\n{}",
        bad.join("\n")
    );
}

#[test]
fn a_real_update_changes_the_weights_and_the_loss_is_finite() {
    let device = Default::default();
    let states = roots();
    let mut model = ActiveSearchModel::<TB>::new(tiny(), &device);
    let mut optim = adamw::<TB, _>();
    let mut losses = Vec::new();
    for _ in 0..3 {
        let mut script = FirstTwo;
        let out = model
            .run(&states, &opts(2), Selection::Script(&mut script), &device)
            .unwrap();
        let (pt, wt) = targets_first_two(&out, &device);
        let loss = policy_ce(&out.readout.policy, &pt)
            + wdl_ce(&out.readout.wdl_logits, &wt)
            + selector_loss(&out.selector_steps).unwrap();
        losses.push(loss.clone().into_data().to_vec::<f32>().unwrap()[0]);
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optim.step(3e-3, model, grads);
    }
    assert!(losses.iter().all(|l| l.is_finite()), "{losses:?}");
    assert!(
        losses[2] < losses[0],
        "three real updates should reduce the loss: {losses:?}"
    );
}

// ---------------------------------------------------------------------------
// Identity and checkpoints
// ---------------------------------------------------------------------------

fn meta(cfg: ModelConfig) -> CheckpointMeta {
    CheckpointMeta::new(cfg, 1, false, 0, 0.0, 1, 0, "flex", "fp32")
}

fn configs() -> Vec<(&'static str, ModelConfig)> {
    let probe: ModelConfig = serde_json::from_str(
        r#"{"width":384,"heads":12,"ffn":768,"input_blocks":0,"core_blocks":8,"output_blocks":0}"#,
    )
    .unwrap();
    vec![
        ("probe_v1", probe),
        ("candidate_v25", ModelConfig::candidate_v25(true)),
        ("legacy_facts_v25", ModelConfig::legacy_facts_v25()),
        ("active_search_v3", ModelConfig::active_search_v3()),
    ]
}

#[test]
fn every_ordered_pair_of_four_architectures_is_refused_explicitly() {
    let cfgs = configs();
    for (na, ca) in &cfgs {
        let m = meta(ca.clone());
        m.check_contracts().unwrap();
        m.check_model(ca).unwrap();
        for (nb, cb) in &cfgs {
            if na == nb {
                continue;
            }
            let e = m.check_model(cb).unwrap_err().to_string();
            assert!(
                e.contains("cross-architecture"),
                "{na} -> {nb} must be refused explicitly: {e}"
            );
        }
    }
}

#[test]
fn a_changed_v3_contract_is_refused_by_config_model_identity_and_checkpoint() {
    type Edit = fn(&mut ActiveContracts);
    let edits: [(&str, Edit); 11] = [
        ("root_encoder", |c| {
            c.root_encoder = "v25_root_encoder_v2".into()
        }),
        ("root_candidate_tokens", |c| {
            c.root_candidate_tokens = "x".into()
        }),
        ("state_query", |c| c.state_query = "state_query_v2".into()),
        ("query_state_encoder", |c| {
            c.query_state_encoder = "x".into()
        }),
        ("frontier", |c| c.frontier = "x".into()),
        ("search_memory", |c| c.search_memory = "x".into()),
        ("selector", |c| c.selector = "x".into()),
        ("planner", |c| c.planner = "x".into()),
        ("proof_trace", |c| c.proof_trace = "x".into()),
        ("budget_training", |c| {
            c.budget_training = "budget_0_2_4_8_16_v1".into()
        }),
        ("root_policy", |c| c.root_policy = "x".into()),
    ];
    let good = ModelConfig::active_search_v3();
    let good_meta = meta(good.clone());
    for (name, edit) in edits {
        // The configuration refuses to validate.
        let mut bad = good.clone();
        edit(bad.active.as_mut().map(|a| &mut a.contracts).unwrap());
        assert!(bad.validate().is_err(), "{name}: config must refuse");
        // A checkpoint recorded under the good contracts refuses the changed config.
        assert!(
            good_meta.check_model(&bad).is_err(),
            "{name}: model identity"
        );
        // A checkpoint recorded under a changed contract refuses to load.
        let mut m = good_meta.clone();
        edit(m.active_contracts.as_mut().unwrap());
        assert!(m.check_contracts().is_err(), "{name}: checkpoint contracts");
    }
    // A missing contract block is refused too.
    let mut m = good_meta.clone();
    m.active_contracts = None;
    assert!(m.check_contracts().is_err());
    // The two identities serialize differently (hash-relevant config fields).
    let mut other = good.clone();
    other.active.as_mut().unwrap().workspace_tokens = 9;
    assert_ne!(
        serde_json::to_value(&good).unwrap(),
        serde_json::to_value(&other).unwrap()
    );
    assert!(good_meta.check_model(&other).is_err());
}

#[test]
fn historical_identities_do_not_grow_an_active_key() {
    for (name, cfg) in configs().into_iter().take(3) {
        let v = serde_json::to_value(&cfg).unwrap();
        assert!(!v.as_object().unwrap().contains_key("active"), "{name}");
        assert!(meta(cfg).active_contracts.is_none(), "{name}");
    }
    assert!(
        meta(ModelConfig::active_search_v3())
            .active_contracts
            .is_some()
    );
    assert_eq!(
        ModelConfig::active_search_v3().architecture,
        Architecture::ActiveSearchV3
    );
}

#[test]
fn save_load_round_trip_and_resume_match_an_uninterrupted_run() {
    let device = Default::default();
    let states = roots();
    let cfg = tiny();
    let step = |model: ActiveSearchModel<TB>, optim: &mut _| {
        let mut script = FirstTwo;
        let out = model
            .run(&states, &opts(2), Selection::Script(&mut script), &device)
            .unwrap();
        let (pt, wt) = targets_first_two(&out, &device);
        let loss = policy_ce(&out.readout.policy, &pt)
            + wdl_ce(&out.readout.wdl_logits, &wt)
            + selector_loss(&out.selector_steps).unwrap();
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        Optimizer::step(optim, 3e-3, model, grads)
    };
    let probe = |m: &ActiveSearchModel<TB>| -> Vec<f32> {
        let out = m
            .valid()
            .run(&states, &opts(2), Selection::Active, &device)
            .unwrap();
        out.readout
            .policy
            .log_probs
            .into_data()
            .to_vec::<f32>()
            .unwrap()
    };
    let model0 = ActiveSearchModel::<TB>::new(cfg.clone(), &device);
    let optim0 = adamw::<TB, _>();

    // Uninterrupted: two steps.
    let (mut ma, mut oa) = (model0.clone(), optim0.clone());
    ma = step(ma, &mut oa);
    ma = step(ma, &mut oa);

    // Interrupted: one step, save, load, one step.
    let (mut mb, mut ob) = (model0.clone(), optim0.clone());
    mb = step(mb, &mut ob);
    let dir = std::env::temp_dir().join("recur64_v3_active_ckpt");
    let _ = std::fs::remove_dir_all(&dir);
    save_training::<TB, _, _>(&dir, &mb, &ob, &meta(cfg.clone())).expect("save");
    let (mut mb2, mut ob2, loaded) =
        load_training::<TB, _, _>(&dir, model0.clone(), optim0.clone(), &device).expect("load");
    assert_eq!(loaded.architecture, "active_search_v3");
    assert_eq!(probe(&mb), probe(&mb2), "weights must be restored exactly");
    mb2 = step(mb2, &mut ob2);

    let (pa, pb) = (probe(&ma), probe(&mb2));
    let max_diff = pa
        .iter()
        .zip(&pb)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-4,
        "resumed run diverged from uninterrupted by {max_diff}"
    );

    // A checkpoint refuses to load into a different architecture's template.
    let cand = recur64_model::candidate::CandidateV25Model::<TB>::new(
        {
            let mut c = ModelConfig::candidate_v25(true);
            c.width = 32;
            c.heads = 4;
            c.ffn = 64;
            c.core_blocks = 1;
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
        },
        &device,
    );
    let err = load_training::<TB, _, _>(&dir, cand, adamw::<TB, _>(), &device).err();
    assert!(err.is_some(), "cross-architecture template must be refused");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn action_embeddings_use_only_the_queried_node() {
    // The raw-action embeddings are built from a node's own square features and
    // action geometry. Two different root moves leading to the same child
    // position are not distinguishable before being queried: here we check that
    // the model exposes the action geometry only (no CandidateFacts on queried
    // nodes): its root path is the only consumer of the facts encoder.
    let device = Default::default();
    let m = ActiveSearchModel::<B>::new(tiny(), &device);
    let names: Vec<&str> = m.param_breakdown().into_iter().map(|(n, _)| n).collect();
    assert_eq!(
        names
            .iter()
            .filter(|n| n.contains("facts"))
            .collect::<Vec<_>>(),
        vec![&"root.facts_encoder"],
        "CandidateFacts belong to the root path only"
    );
    let _ = ActionId::from_index(0).unwrap();
}
