//! V3.5 tests: the choice-vs-supervision separation, label-only targets on learner
//! prefixes, detached-rollout / autodiff-replay parity, the whole-update normalisation,
//! recipe and init identity, and exact CPU resume.
//!
//! Tiny geometry of the same architecture on the CPU backend; positions are real exact
//! mates found by the deterministic generator. Nothing is canned.

use std::collections::HashSet;

use burn::prelude::*;
use recur64_core::GameState;
use recur64_model::active::{
    ActiveOutput, ActiveSearchModel, EdgeRef, QueryTargetProvider, RunOptions, Selection, Tree,
};
use recur64_model::train::adamw;
use recur64_statequery::QueryManager;

use crate::p5::data::BudgetSamplers;
use crate::p5::recipe::{BUDGETS, Layout};
use crate::p5::tests::{TB, dataset, flat, params, plan_of, rng_lock, tiny_model};
use crate::proof::trace::Path;
use crate::proof::trace_teacher::{edge_path, node_paths};

use super::recipe::{Recipe35, SELECTED_LR, TRAINING_ID};
use super::target::ProofTargetProvider;
use super::train::{Trainer35, compute_update, load_init_weights, rollout};

fn recipe35(micro: usize, accum: usize, updates: u64, seed: u64) -> Recipe35 {
    let mut r = Recipe35::contract(Layout::DEFAULT, false)
        .for_seed(seed)
        .unwrap();
    r.base.model = tiny_model();
    r.base.micro = micro;
    r.base.accum = accum;
    r.base.effective_batch = micro * accum;
    r.base.budget_sequence = (0..accum).map(|j| BUDGETS[j % 4]).collect();
    r.base.updates = updates;
    r.base.warmup = 1;
    r.base.eval_updates = vec![0, updates];
    r
}

fn roots(ds: &crate::p5::data::Dataset, idx: &[usize]) -> Vec<GameState> {
    idx.iter()
        .map(|&i| GameState::from_fen(&ds.positions()[i].fen).unwrap())
        .collect()
}

fn log_probs(out: &ActiveOutput<TB>) -> Vec<f32> {
    out.readout
        .policy
        .log_probs
        .clone()
        .into_data()
        .to_vec::<f32>()
        .unwrap()
}

fn selector_loss(out: &ActiveOutput<TB>) -> Option<f32> {
    recur64_model::active::loss::selector_nll_sum(&out.selector_steps)
        .map(|(s, _)| s.into_data().to_vec::<f32>().unwrap()[0])
}

/// A provider that returns arbitrary (but valid) labels: it must not change anything
/// the model computes forward.
struct Garbage {
    pick: usize,
}

impl QueryTargetProvider for Garbage {
    fn targets(
        &mut self,
        _e: usize,
        step: usize,
        frontier: &[EdgeRef],
        _t: &Tree,
    ) -> anyhow::Result<Vec<usize>> {
        Ok(vec![(self.pick + step) % frontier.len()])
    }
}

fn model_and_ds(seed: u64, n: usize) -> (ActiveSearchModel<TB>, crate::p5::data::Dataset) {
    let (ds, _d) = dataset(n, seed);
    let device = Default::default();
    <TB as Backend>::seed(&device, 5101);
    let model = ActiveSearchModel::<TB>::new(tiny_model(), &device);
    (model, ds)
}

#[test]
fn a_target_provider_cannot_change_the_trajectory_or_any_forward_value() {
    let _g = rng_lock();
    let (model, ds) = model_and_ds(201, 12);
    let device = Default::default();
    let r = roots(&ds, &[0, 1, 2, 3, 4, 5]);
    for budget in [2usize, 4, 8] {
        let opts = RunOptions::forced(budget);
        let plain = model.run(&r, &opts, Selection::Active, &device).unwrap();
        let traces: Vec<&_> = (0..6).map(|i| &ds.traces[i]).collect();
        let mut real = ProofTargetProvider::new(traces);
        let labelled = model
            .run(&r, &opts, Selection::ActiveLabelled(&mut real), &device)
            .unwrap();
        let mut junk = Garbage { pick: 3 };
        let garbage = model
            .run(&r, &opts, Selection::ActiveLabelled(&mut junk), &device)
            .unwrap();
        for other in [&labelled, &garbage] {
            assert_eq!(
                plain.chosen, other.chosen,
                "B{budget}: the chosen edges differ"
            );
            assert_eq!(
                format!(
                    "{:?}",
                    plain
                        .traces
                        .iter()
                        .map(|t| t
                            .iter()
                            .map(|q| (q.parent_slot, q.action, q.branch, q.depth))
                            .collect::<Vec<_>>())
                        .collect::<Vec<_>>()
                ),
                format!(
                    "{:?}",
                    other
                        .traces
                        .iter()
                        .map(|t| t
                            .iter()
                            .map(|q| (q.parent_slot, q.action, q.branch, q.depth))
                            .collect::<Vec<_>>())
                        .collect::<Vec<_>>()
                ),
            );
            assert_eq!(
                log_probs(&plain),
                log_probs(other),
                "B{budget}: the final policy differs bit-for-bit"
            );
        }
        // Only the selector loss reacts to the labels.
        assert!(selector_loss(&plain).is_none());
        let (a, b) = (selector_loss(&labelled), selector_loss(&garbage));
        assert!(a.is_some() && b.is_some());
        assert_ne!(a, b, "different labels must give a different selector loss");
    }
}

#[test]
fn changing_labels_changes_only_the_selector_gradient() {
    let _g = rng_lock();
    let (model, ds) = model_and_ds(202, 12);
    let device = Default::default();
    let r = roots(&ds, &[0, 1, 2, 3]);
    let opts = RunOptions::forced(4);
    let mut p1 = Garbage { pick: 0 };
    let mut p2 = Garbage { pick: 5 };
    let o1 = model
        .run(&r, &opts, Selection::ActiveLabelled(&mut p1), &device)
        .unwrap();
    let o2 = model
        .run(&r, &opts, Selection::ActiveLabelled(&mut p2), &device)
        .unwrap();
    assert_eq!(log_probs(&o1), log_probs(&o2));
    let grads = |o: &ActiveOutput<TB>| {
        let (s, _) = recur64_model::active::loss::selector_nll_sum(&o.selector_steps).unwrap();
        burn::optim::GradientsParams::from_grads(s.backward(), &model)
    };
    let (g1, g2) = (flat(&grads(&o1), &model), flat(&grads(&o2), &model));
    assert_ne!(g1, g2, "the selector gradient must depend on the labels");
}

fn apply(mgr: &mut QueryManager, tree: &mut Tree, e: &EdgeRef) {
    let pkt = mgr.query(tree.node(e.node_slot).id, e.action).unwrap();
    tree.add_child(e, &pkt).unwrap();
}

#[test]
fn an_off_target_learner_branch_gets_legal_nonempty_refute_targets() {
    let _g = rng_lock();
    let (ds, _d) = dataset(24, 203);
    let mut found = 0usize;
    for (i, trace) in ds.traces.iter().enumerate() {
        let root = GameState::from_fen(&ds.positions()[i].fen).unwrap();
        let n_root = root.legal_actions().len();
        for k in 0..n_root {
            let mut mgr = QueryManager::new(root.clone()).unwrap();
            let mut tree = Tree::new(&mgr.packet(0).unwrap()).unwrap();
            let edge = tree.frontier()[k].clone();
            apply(&mut mgr, &mut tree, &edge);
            let paths = node_paths(&tree);
            let s: HashSet<Path> = paths.iter().filter(|p| !p.is_empty()).cloned().collect();
            let adm = trace.admissible(&s);
            if adm.refute.is_empty() {
                continue;
            }
            found += 1;
            let frontier = tree.frontier();
            let mut prov = ProofTargetProvider::new(vec![trace]);
            let t = prov.targets(0, 0, &frontier, &tree).unwrap();
            let all = adm.all();
            let got: HashSet<Path> = t.iter().map(|&j| edge_path(&paths, &frontier[j])).collect();
            assert_eq!(got, all.iter().cloned().collect::<HashSet<_>>());
            assert!(
                adm.refute.iter().all(|p| got.contains(p)),
                "A_refute edges are targets"
            );
            assert!(t.iter().all(|&j| j < frontier.len()));
            assert!(prov.stats[0].refute_steps == 1 && prov.stats[0].supervised == 1);
        }
    }
    assert!(
        found > 0,
        "no off-target learner branch with refutations in the fixture"
    );
}

#[test]
fn arbitrary_learner_prefixes_get_legal_targets_and_completion_empties_them() {
    let _g = rng_lock();
    let (ds, _d) = dataset(24, 204);
    let mut incomplete = 0usize;
    let mut complete = 0usize;
    for (i, trace) in ds.traces.iter().enumerate() {
        let root = GameState::from_fen(&ds.positions()[i].fen).unwrap();
        let mut mgr = QueryManager::new(root.clone()).unwrap();
        let mut tree = Tree::new(&mgr.packet(0).unwrap()).unwrap();
        let mut prov = ProofTargetProvider::new(vec![trace]);
        for step in 0..8 {
            let frontier = tree.frontier();
            if frontier.is_empty() {
                break;
            }
            let paths = node_paths(&tree);
            let s: HashSet<Path> = paths.iter().filter(|p| !p.is_empty()).cloned().collect();
            let t = prov.targets(0, step, &frontier, &tree).unwrap();
            if trace.is_complete(&s) {
                complete += 1;
                assert!(t.is_empty(), "no selector target after proof completion");
            } else {
                incomplete += 1;
                assert!(!t.is_empty() && t.iter().all(|&j| j < frontier.len()));
                let got: HashSet<Path> =
                    t.iter().map(|&j| edge_path(&paths, &frontier[j])).collect();
                assert_eq!(got, trace.admissible(&s).all().into_iter().collect());
            }
            // Test learners (not part of production): even positions follow their first
            // target (so some reach proof completion and keep querying afterwards); odd
            // positions take a fixed label-independent mixing walk.
            let e = if i % 2 == 0 && !t.is_empty() {
                frontier[t[0]].clone()
            } else {
                frontier[(i * 7 + step * 3 + 1) % frontier.len()].clone()
            };
            apply(&mut mgr, &mut tree, &e);
        }
    }
    assert!(incomplete > 0 && complete > 0, "{incomplete} / {complete}");
}

#[test]
fn a_proof_complete_learner_path_gets_no_further_loss_while_active_keeps_spending_budget() {
    let _g = rng_lock();
    let (model, ds) = model_and_ds(205, 15);
    let device = Default::default();
    let idx: Vec<usize> = (0..12).collect();
    let r = roots(&ds, &idx);
    let items: Vec<_> = idx
        .iter()
        .map(|&i| crate::p5::train::Item {
            index: i,
            ordinal: 0,
        })
        .collect();
    let ro = rollout(&model, &ds, &r, &items, 8, false, &device).unwrap();
    // Active spends the whole forced budget whether or not the proof completed.
    let plain = model
        .run(&r, &RunOptions::forced(8), Selection::Active, &device)
        .unwrap();
    assert_eq!(ro.follow, plain.chosen);
    assert_eq!(ro.successful, plain.accounting.successful_queries);
    let mut saw_completed = false;
    for (e, t) in ro.targets.iter().enumerate() {
        let mut after = false;
        for step_targets in t {
            if after {
                assert!(step_targets.is_empty());
            }
            if step_targets.is_empty() {
                after = true; // empty target <=> proof complete (never empty while incomplete)
            }
        }
        if after {
            saw_completed = true;
            assert_eq!(
                ro.follow[e].len(),
                ro.successful[e],
                "queries continue after completion"
            );
        }
    }
    assert!(saw_completed || ro.proofs_completed == 0);
    let total: usize = ro
        .targets
        .iter()
        .map(|t| t.iter().filter(|x| !x.is_empty()).count())
        .sum();
    assert_eq!(total, ro.supervised);
}

#[test]
fn the_replay_reproduces_the_detached_rollout_and_the_update_is_microbatch_independent() {
    let _g = rng_lock();
    let (ds, _d) = dataset(15, 206);
    let device = Default::default();
    let recipe = recipe35(2, 8, 2, 5101);
    <TB as Backend>::seed(&device, 5101);
    let model = ActiveSearchModel::<TB>::new(recipe.base.model.clone(), &device);
    let items: Vec<(usize, usize, u64)> = (0..24)
        .map(|k| (BUDGETS[k % 4], (k * 7) % ds.positions().len(), k as u64))
        .collect();
    let (g_a, r_a) = compute_update(&model, &ds, &plan_of(&items, 6), &recipe, &device).unwrap();
    let (g_b, r_b) = compute_update(&model, &ds, &plan_of(&items, 2), &recipe, &device).unwrap();
    let (g_c, r_c) = compute_update(&model, &ds, &plan_of(&items, 3), &recipe, &device).unwrap();
    for r in [&r_a, &r_b, &r_c] {
        assert!(r.max_replay_policy_diff <= recipe.replay_policy_tolerance);
        assert_eq!(r.examples, 24);
        assert_eq!(
            r.supervised_decisions,
            r.per_budget
                .iter()
                .map(|b| b.supervised_decisions)
                .sum::<usize>()
        );
    }
    for r in [&r_b, &r_c] {
        assert!((r.policy_loss - r_a.policy_loss).abs() < 1e-4);
        assert!((r.selector_loss - r_a.selector_loss).abs() < 1e-4);
        assert_eq!(r.supervised_decisions, r_a.supervised_decisions);
    }
    assert!((r_a.total_loss - (r_a.policy_loss + r_a.selector_loss)).abs() < 1e-9);
    let rel = |x: &[f32], y: &[f32]| {
        let num: f32 = x
            .iter()
            .zip(y)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        let den: f32 = x.iter().map(|a| a.abs()).fold(1e-12, f32::max);
        num / den
    };
    let (fa, fb, fc) = (flat(&g_a, &model), flat(&g_b, &model), flat(&g_c, &model));
    assert!(rel(&fa, &fb) < 2e-3 && rel(&fa, &fc) < 2e-3);
    assert!(r_a.supervised_decisions > 0);
}

#[test]
fn the_selector_weight_does_not_depend_on_how_many_b0_examples_exist() {
    let _g = rng_lock();
    let (ds, _d) = dataset(15, 207);
    let device = Default::default();
    let recipe = recipe35(2, 8, 2, 5101);
    <TB as Backend>::seed(&device, 5101);
    let model = ActiveSearchModel::<TB>::new(recipe.base.model.clone(), &device);
    let base: Vec<(usize, usize, u64)> = (0..12)
        .map(|k| (BUDGETS[1 + k % 3], (k * 5) % 15, k as u64))
        .collect();
    let (_, r1) = compute_update(&model, &ds, &plan_of(&base, 3), &recipe, &device).unwrap();
    let mut with_b0 = base.clone();
    for k in 0..12 {
        with_b0.push((0, (k * 3) % 15, 100 + k as u64));
    }
    let (_, r2) = compute_update(&model, &ds, &plan_of(&with_b0, 3), &recipe, &device).unwrap();
    assert!((r1.selector_loss - r2.selector_loss).abs() < 1e-5);
    assert_eq!(r1.supervised_decisions, r2.supervised_decisions);
    assert_eq!(r2.per_budget[0].supervised_decisions, 0);
}

#[test]
fn the_recipe_identity_is_distinct_and_refuses_every_unauthorised_change() {
    let base = recipe35(2, 4, 4, 5101);
    base.validate_for_training().unwrap();
    assert_eq!(base.training, TRAINING_ID);
    // Not a P5 recipe: a different schema and teacher string.
    assert_ne!(base.base.teacher, crate::p5::recipe::TEACHER_ID);
    let d = base.digest();
    for mutate in [
        |r: &mut Recipe35| r.base.peak_lr = Some(1.5e-4),
        |r: &mut Recipe35| r.base.seed = Some(5104),
        |r: &mut Recipe35| r.init_p5_recipe_digest = "0".repeat(64),
        |r: &mut Recipe35| r.base.budget_sequence = vec![0, 2, 4, 16],
        |r: &mut Recipe35| r.base.precision = "bf16".into(),
        |r: &mut Recipe35| r.base.selector_weight = 0.5,
        |r: &mut Recipe35| r.replay_policy_tolerance = 1e-2,
    ] {
        let mut r = base.clone();
        mutate(&mut r);
        assert_ne!(r.digest(), d);
        assert!(
            r.validate_for_training().is_err(),
            "an unauthorised change was accepted"
        );
    }
    assert!(
        Recipe35::contract(Layout::DEFAULT, false)
            .for_seed(1)
            .is_err()
    );
    // Seeds differ in identity, share the contract.
    let b = recipe35(2, 4, 4, 5102);
    assert_ne!(b.digest(), d);
    assert_eq!(b.contract_digest(), base.contract_digest());
    assert_eq!(base.base.peak_lr, Some(SELECTED_LR));
}

#[test]
fn init_refuses_anything_but_the_seeds_selected_p5_final_checkpoint() {
    let _g = rng_lock();
    let (ds, _d) = dataset(15, 208);
    let device = Default::default();
    // A real (tiny) P5 checkpoint: valid, but not the frozen selected recipe.
    let p5 = {
        let mut r = crate::p5::recipe::Recipe::screen_contract(Layout::DEFAULT, false);
        r.model = tiny_model();
        r.micro = 2;
        r.accum = 4;
        r.effective_batch = 8;
        r.budget_sequence = vec![0, 2, 4, 8];
        r.updates = 1;
        r.warmup = 1;
        r.eval_updates = vec![0, 1];
        r.for_run(SELECTED_LR, 5101)
    };
    let mut t = crate::p5::train::Trainer::<TB>::new(p5, &ds, &device).unwrap();
    t.step(&ds, &device).unwrap();
    let dir = crate::p5::tests::tmp("p35_init_refusal");
    t.save(&dir).unwrap();
    let recipe = recipe35(2, 4, 4, 5101);
    let e = load_init_weights::<TB>(&recipe, &dir, &device)
        .err()
        .unwrap();
    assert!(e.to_string().contains("init checkpoint carries"), "{e}");
    assert!(load_init_weights::<TB>(&recipe, &dir.join("missing"), &device).is_err());
}

#[test]
fn cpu_resume_is_bit_exact_and_refuses_another_recipe() {
    let _g = rng_lock();
    let (ds, _d) = dataset(15, 209);
    let device = Default::default();
    let recipe = recipe35(2, 4, 4, 5101);
    let fresh = |recipe: &Recipe35| {
        <TB as Backend>::seed(&device, 5101);
        Trainer35::<TB> {
            model: ActiveSearchModel::<TB>::new(recipe.base.model.clone(), &device),
            optim: adamw::<TB, ActiveSearchModel<TB>>(),
            samplers: BudgetSamplers::new(&ds.cells(), 5101),
            recipe: recipe.clone(),
            updates_done: 0,
            history: Vec::new(),
            resumptions: 0,
        }
    };
    let mut full = fresh(&recipe);
    for _ in 0..4 {
        full.step(&ds, &device).unwrap();
    }
    let mut part = fresh(&recipe);
    for _ in 0..2 {
        part.step(&ds, &device).unwrap();
    }
    let dir = crate::p5::tests::tmp("p35_resume");
    part.save(&dir).unwrap();
    let mut resumed = Trainer35::<TB>::load(&dir, recipe.clone(), &ds, &device).unwrap();
    assert_eq!(resumed.resumptions, 1);
    for _ in 0..2 {
        resumed.step(&ds, &device).unwrap();
    }
    assert_eq!(
        params(&full.model),
        params(&resumed.model),
        "resume is not bit-exact"
    );
    let other = recipe35(2, 4, 4, 5102);
    assert!(Trainer35::<TB>::load(&dir, other, &ds, &device).is_err());
    let mut changed = recipe.clone();
    changed.base.warmup = 2;
    assert!(Trainer35::<TB>::load(&dir, changed, &ds, &device).is_err());
}

#[test]
fn every_update_has_equal_budget_exposure_and_never_plans_b16() {
    let _g = rng_lock();
    let (ds, _d) = dataset(15, 210);
    let recipe = recipe35(2, 8, 2, 5101);
    let mut t = Trainer35::<TB> {
        model: ActiveSearchModel::<TB>::new(recipe.base.model.clone(), &Default::default()),
        optim: adamw::<TB, ActiveSearchModel<TB>>(),
        samplers: BudgetSamplers::new(&ds.cells(), 5101),
        recipe,
        updates_done: 0,
        history: Vec::new(),
        resumptions: 0,
    };
    for _ in 0..3 {
        let plan = t.plan_next();
        for b in BUDGETS {
            let n: usize = plan
                .micros
                .iter()
                .filter(|m| m.budget == b)
                .map(|m| m.items.len())
                .sum();
            assert_eq!(n, 2 * 8 / 4, "budget {b}");
        }
        assert!(plan.micros.iter().all(|m| m.budget != 16));
    }
}
