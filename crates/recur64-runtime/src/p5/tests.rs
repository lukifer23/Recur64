//! P5 tests: the training teacher and its latch, determinism, input refusal,
//! sampler balance, the loss normalisation against a reference, recipe-mismatch
//! refusal, exact CPU resume and the offline selector diagnostics.
//!
//! The model is a tiny geometry of the same architecture on the CPU backend and the
//! data are real exact positions found by a deterministic seeded search; nothing is
//! canned. Expected-artifact digests are parameters of the loaders, so the refusal
//! logic is exercised against fixtures while production code only ever passes the
//! frozen constants.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::PathBuf;

use burn::optim::GradientsParams;
use burn::prelude::*;
use recur64_core::GameState;
use recur64_model::active::{ActiveSearchModel, EdgeRef, QueryScript, RunOptions, Selection, Tree};
use recur64_model::config::{CandidateConfig, ModelConfig};
use recur64_model::train::CpuTrainBackend;
use recur64_statequery::QueryManager;

use crate::proof::generator::{Rng, canonical_key, legal_cozy_moves, sample_position};
use crate::proof::mate::MateSolver;
use crate::proof::targets::{ProofPosition, ProofTargets, Split};
use crate::proof::trace::{Path, PositionTrace, build_trace};
use crate::proof::trace_store::{audit_traces, generate_traces};
use crate::proof::trace_teacher::{edge_path, node_paths};

use super::data::{BudgetSamplers, Dataset, Expected, load_dataset};
use super::eval::{EvalSelection, classify_queries, evaluate, screen_score};
use super::recipe::{BUDGETS, Layout, Recipe, TUNE_DIGEST, teacher_key_base};
use super::teacher::{SeededProofTeacher, follow_key, simulate_episode};
use super::train::{Item, Micro, Trainer, UpdatePlan, compute_update};

pub(crate) type TB = CpuTrainBackend;

/// The backend RNG is process-global: tests that seed it or build models must not interleave
/// (P5 and P6 tests share this lock).
static RNG: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub(crate) fn rng_lock() -> std::sync::MutexGuard<'static, ()> {
    RNG.lock().unwrap_or_else(|e| e.into_inner())
}

const KQ: &[char] = &['K', 'Q'];
const KR: &[char] = &['K', 'R'];
const KQR: &[char] = &['K', 'Q', 'R'];

pub(crate) fn tiny_model() -> ModelConfig {
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

/// A small, valid recipe for tests (layout checks are about the production recipe;
/// the tiny trainer only needs its fields).
fn tiny_recipe(micro: usize, accum: usize, updates: u64, seed: u64) -> Recipe {
    let mut r = Recipe::screen_contract(Layout::DEFAULT, false);
    r.model = tiny_model();
    r.micro = micro;
    r.accum = accum;
    r.effective_batch = micro * accum;
    r.budget_sequence = (0..accum).map(|j| BUDGETS[j % 4]).collect();
    r.updates = updates;
    r.warmup = 1;
    r.eval_updates = vec![0, updates];
    r.for_run(3.0e-3, seed)
}

fn position(id: &str, fen: &str, family: &str, solver: &mut MateSolver) -> Option<ProofPosition> {
    let state = GameState::from_fen(fen).ok()?;
    let d = solver.mate_depth(state.board(), 3)?;
    let moves = legal_cozy_moves(&state);
    let correct = solver.correct_moves(state.board(), d, &moves);
    let legal: Vec<u16> = state
        .legal_actions()
        .iter()
        .map(|a| a.index() as u16)
        .collect();
    Some(ProofPosition {
        id: id.to_string(),
        fen: fen.to_string(),
        split: Split::Train,
        family: family.to_string(),
        mate_depth: d,
        canon: canonical_key(fen),
        chance_top1: correct.len() as f32 / legal.len() as f32,
        legal,
        correct: correct.iter().map(|&i| i as u32).collect(),
        generator_seed: 0,
    })
}

fn find(
    white: &[char],
    family: &str,
    depth: u8,
    seed: u64,
    pred: impl Fn(&PositionTrace) -> bool,
) -> (ProofPosition, PositionTrace) {
    let mut rng = Rng(seed);
    let mut solver = MateSolver::new();
    for i in 0..60_000 {
        let Some((fen, _)) = sample_position(&mut rng, white) else {
            continue;
        };
        if solver.table_size() > 2_000_000 {
            solver = MateSolver::new();
        }
        let Some(p) = position(
            &format!("p5-{family}-{depth}-{seed}-{i}"),
            &fen,
            family,
            &mut solver,
        ) else {
            continue;
        };
        if p.mate_depth != depth {
            continue;
        }
        let t = build_trace(&mut solver, &p).unwrap();
        if pred(&t) {
            return (p, t);
        }
    }
    panic!("no {family} M{depth} fixture");
}

pub(crate) fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("recur64_p5_{name}"));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// A small mixed dataset (several cells, all depths) with verified traces on disk.
pub(crate) fn mini_dataset(
    split: Split,
    dir: &std::path::Path,
    count: usize,
    seed: u64,
) -> (PathBuf, PathBuf) {
    let mut solver = MateSolver::new();
    let mut rng = Rng(seed);
    let mut positions = Vec::new();
    let mut seen = HashSet::new();
    let mut i = 0;
    while positions.len() < count {
        let (white, fam) = [(KQ, "KQvK"), (KR, "KRvK"), (KQR, "KQRvK")][positions.len() % 3];
        let Some((fen, _)) = sample_position(&mut rng, white) else {
            continue;
        };
        if !seen.insert(canonical_key(&fen)) {
            continue;
        }
        if let Some(mut p) = position(&format!("mini-{seed}-{i}"), &fen, fam, &mut solver)
            && p.mate_depth <= 3
        {
            p.split = split;
            positions.push(p);
        }
        i += 1;
    }
    let t = ProofTargets::new(
        split,
        seed,
        serde_json::json!({"fixture": "p5-mini"}),
        positions,
    );
    let data = dir.join(format!("proof-{}.json", split.label()));
    t.save(&data).unwrap();
    let tdir = dir.join(format!("traces-{}", split.label()));
    generate_traces(&t, &tdir, 5, 2, &|_, _, _| {}).unwrap();
    audit_traces(&t, &tdir, 2, &|_, _, _| {}).unwrap();
    (data, tdir)
}

fn expected_for(path: &std::path::Path, tdir: &std::path::Path, split: Split) -> Expected {
    let t = ProofTargets::load(path).unwrap();
    let m = crate::proof::trace_store::TraceManifest::load(tdir).unwrap();
    Expected {
        digest: t.digest.clone(),
        positions: t.positions.len(),
        trace_manifest: m.manifest_digest,
        split,
    }
}

pub(crate) fn dataset(count: usize, seed: u64) -> (Dataset, PathBuf) {
    let d = tmp(&format!("ds_{seed}_{count}"));
    let (p, t) = mini_dataset(Split::Train, &d, count, seed);
    let ds = load_dataset(&p, &t, &expected_for(&p, &t, Split::Train)).unwrap();
    (ds, d)
}

// ---------------------------------------------------------------------------
// Teacher and latch
// ---------------------------------------------------------------------------

fn run_teacher_step(
    t: &mut SeededProofTeacher<'_>,
    step: usize,
    tree: &Tree,
) -> (recur64_model::active::ScriptStep, Vec<EdgeRef>) {
    let frontier = tree.frontier();
    (t.next(0, step, &frontier, tree).unwrap(), frontier)
}

fn apply_edge(mgr: &mut QueryManager, tree: &mut Tree, e: &EdgeRef) {
    let pkt = mgr.query(tree.node(e.node_slot).id, e.action).unwrap();
    tree.add_child(e, &pkt).unwrap();
}

#[test]
fn the_completion_latch_stops_all_process_targets_even_when_filler_opens_an_incorrect_branch() {
    let _g = rng_lock();
    // M1 with several correct root moves AND incorrect moves that have refutations.
    let (p, t) = find(KQ, "KQvK", 1, 31, |t| {
        t.nodes[t.root as usize].alts.len() >= 2 && !t.refutations.is_empty()
    });
    let root = GameState::from_fen(&p.fen).unwrap();
    let mut mgr = QueryManager::new(root).unwrap();
    let mut tree = Tree::new(&mgr.packet(0).unwrap()).unwrap();
    let mut teacher = SeededProofTeacher::new(vec![&t], vec![123]);

    // Step 0: the proof is incomplete; the target is the complete tied set of mating moves.
    let (s0, f0) = run_teacher_step(&mut teacher, 0, &tree);
    assert_eq!(s0.targets.len(), t.nodes[t.root as usize].alts.len());
    assert!(s0.targets.contains(&s0.follow));
    apply_edge(&mut mgr, &mut tree, &f0[s0.follow].clone());
    assert!(!teacher.is_latched(0));

    // Step 1: the M1 proof completed on query 1. The latch engages: no target.
    let (s1, f1) = run_teacher_step(&mut teacher, 1, &tree);
    assert!(s1.targets.is_empty());
    assert!(teacher.is_latched(0));
    assert_eq!(teacher.records[0].completed_at, Some(1));
    assert_eq!(teacher.records[0].supervised, 1);

    // Force a FILLER query that opens an incorrect root branch (as a different,
    // non-BFS choice would): step 1 followed BFS, so open the incorrect branch
    // explicitly, as a stand-in for any later filler.
    let paths = node_paths(&tree);
    let bad = t.refutations[0].root_action;
    let e = f1
        .iter()
        .find(|e| e.node_slot == 0 && e.action == bad)
        .expect("the incorrect root edge is on the frontier")
        .clone();
    assert!(!paths.is_empty());
    apply_edge(&mut mgr, &mut tree, &e);

    // The generic A(S) now contains refutation edges...
    let paths = node_paths(&tree);
    let s: HashSet<Path> = paths.iter().filter(|p| !p.is_empty()).cloned().collect();
    assert!(
        !t.admissible(&s).refute.is_empty(),
        "generic A(S) re-activates refutation edges"
    );
    // ...but the P5 teacher stays latched complete and emits NO further target.
    for step in 2..6 {
        let frontier = tree.frontier();
        let st = teacher.next(0, step, &frontier, &tree).unwrap();
        assert!(
            st.targets.is_empty(),
            "step {step}: a latched example never gets a target again"
        );
        assert!(st.follow < frontier.len());
    }
    assert_eq!(
        teacher.records[0].supervised, 1,
        "supervision never reactivates"
    );
}

#[test]
fn after_completion_the_remaining_budget_is_spent_in_fixed_bfs_order() {
    let _g = rng_lock();
    let (p, t) = find(KQ, "KQvK", 1, 41, |_| true);
    let root = GameState::from_fen(&p.fen).unwrap();
    let mut mgr = QueryManager::new(root).unwrap();
    let mut tree = Tree::new(&mgr.packet(0).unwrap()).unwrap();
    let mut teacher = SeededProofTeacher::new(vec![&t], vec![7]);
    let (s0, f0) = run_teacher_step(&mut teacher, 0, &tree);
    apply_edge(&mut mgr, &mut tree, &f0[s0.follow].clone());
    for step in 1..5 {
        let (st, f) = run_teacher_step(&mut teacher, step, &tree);
        let bfs = f
            .iter()
            .enumerate()
            .min_by_key(|(_, e)| e.bfs_key())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(st.follow, bfs, "filler must follow fixed_bfs_actionid_v1");
        assert!(st.targets.is_empty());
        apply_edge(&mut mgr, &mut tree, &f[st.follow].clone());
    }
}

#[test]
fn the_follow_choice_is_deterministic_seeded_and_model_independent() {
    let _g = rng_lock();
    let (p, t) = find(KQR, "KQRvK", 2, 51, |t| {
        t.nodes[t.root as usize].alts.len() >= 2
    });
    let root = GameState::from_fen(&p.fen).unwrap();
    let first_follow = |key: u64| {
        let mgr = QueryManager::new(root.clone()).unwrap();
        let tree = Tree::new(&mgr.packet(0).unwrap()).unwrap();
        let mut teacher = SeededProofTeacher::new(vec![&t], vec![key]);
        let (s, f) = run_teacher_step(&mut teacher, 0, &tree);
        (f[s.follow].action, s.targets.len())
    };
    // Same key: identical, always.
    for key in [1u64, 99, 123456789] {
        assert_eq!(first_follow(key), first_follow(key));
    }
    // Different occurrences traverse different tied orderings.
    let base = teacher_key_base(5101);
    let picks: BTreeSet<u16> = (0..64u64)
        .map(|ordinal| first_follow(follow_key(base, &p.id, ordinal, 8)).0)
        .collect();
    let tied = t.admissible(&HashSet::new()).proof.len();
    assert!(tied >= 2);
    assert!(
        picks.len() >= 2,
        "64 occurrences of one position with {tied} tied first moves must not all choose the same edge"
    );
    // It is not the lexicographically first edge for every key.
    let lexi = *t
        .admissible(&HashSet::new())
        .proof
        .iter()
        .next()
        .unwrap()
        .first()
        .unwrap();
    assert!(picks.iter().any(|&a| a != lexi) || picks.len() > 1);
    // A different run seed gives a different key base.
    assert_ne!(
        follow_key(teacher_key_base(5101), &p.id, 3, 4),
        follow_key(teacher_key_base(5102), &p.id, 3, 4)
    );
}

#[test]
fn the_key_does_not_use_a_process_randomised_hasher() {
    let _g = rng_lock();
    // Pinned values: a std DefaultHasher would change between processes.
    assert_eq!(super::teacher::fnv1a(b"abc"), 0xE71F_A219_0541_574B);
    assert_eq!(follow_key(0, "x", 0, 0), follow_key(0, "x", 0, 0));
    let a = follow_key(teacher_key_base(5101), "kqr-m3-00017", 12, 8);
    assert_eq!(a, follow_key(teacher_key_base(5101), "kqr-m3-00017", 12, 8));
}

#[test]
fn while_incomplete_the_teacher_never_enters_an_incorrect_branch_and_completes_in_q_star() {
    let _g = rng_lock();
    for (white, fam, depth, seed) in [
        (KQR, "KQRvK", 2u8, 61u64),
        (KQR, "KQRvK", 3, 62),
        (KR, "KRvK", 3, 63),
    ] {
        let (p, t) = find(white, fam, depth, seed, |t| t.q_star <= 40);
        let root = GameState::from_fen(&p.fen).unwrap();
        for key in [3u64, 4, 5] {
            let rec = simulate_episode(&t, &root, 16, key).unwrap();
            // On-proof from the empty set: exactly Q* supervised decisions, then the latch.
            assert_eq!(
                rec.supervised as u64,
                t.q_star.min(16).min(t.q_star),
                "{}",
                t.id
            );
            if t.q_star <= 16 {
                assert_eq!(rec.completed_at, Some(t.q_star as usize));
            } else {
                assert_eq!(rec.completed_at, None);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Inputs and samplers
// ---------------------------------------------------------------------------

#[test]
fn dataset_loading_refuses_every_identity_mismatch() {
    let _g = rng_lock();
    let d = tmp("refuse");
    let (p, tdir) = mini_dataset(Split::Train, &d, 6, 71);
    let good = expected_for(&p, &tdir, Split::Train);
    load_dataset(&p, &tdir, &good).unwrap();

    let mut e = good.clone();
    e.digest = "0".repeat(64);
    assert!(
        load_dataset(&p, &tdir, &e)
            .unwrap_err()
            .to_string()
            .contains("frozen")
    );
    let mut e = good.clone();
    e.positions += 1;
    assert!(load_dataset(&p, &tdir, &e).is_err());
    let mut e = good.clone();
    e.trace_manifest = "1".repeat(64);
    assert!(load_dataset(&p, &tdir, &e).is_err());
    let mut e = good.clone();
    e.split = Split::Tune;
    assert!(load_dataset(&p, &tdir, &e).is_err(), "wrong split");

    // A missing trace shard is refused.
    let shard = tdir.join("trace-shard-00000.json");
    let bytes = std::fs::read(&shard).unwrap();
    std::fs::remove_file(&shard).unwrap();
    assert!(load_dataset(&p, &tdir, &good).is_err());
    std::fs::write(&shard, &bytes).unwrap();
    load_dataset(&p, &tdir, &good).unwrap();

    // A trace whose id or fen differs from its source is refused (edit + re-digest).
    let mut s: crate::proof::trace_store::TraceShard = serde_json::from_slice(&bytes).unwrap();
    s.positions[0].fen = s.positions[1].fen.clone();
    s.digest = s.compute_digest();
    std::fs::write(&shard, serde_json::to_vec(&s).unwrap()).unwrap();
    assert!(load_dataset(&p, &tdir, &good).is_err());
    std::fs::write(&shard, &bytes).unwrap();

    // An audit with failures, or none, is refused.
    let audit = tdir.join("audit-manifest.json");
    let abytes = std::fs::read(&audit).unwrap();
    std::fs::remove_file(&audit).unwrap();
    assert!(load_dataset(&p, &tdir, &good).is_err());
    std::fs::write(&audit, abytes).unwrap();

    // HOLDOUT and CONFIRM splits are refused by the loader, and the frozen digests are
    // the only ones accepted in production.
    let (hp, ht) = {
        let (path, tdir2) = mini_dataset(Split::Train, &tmp("refuse_h"), 6, 72);
        let mut t = ProofTargets::load(&path).unwrap();
        t.split = Split::HoldoutC;
        for x in &mut t.positions {
            x.split = Split::HoldoutC;
        }
        t.digest = t.compute_digest();
        let hp = path.with_file_name("proof-holdout_c.json");
        t.save(&hp).unwrap();
        (hp, tdir2)
    };
    for split in [Split::HoldoutC, Split::Confirm, Split::HoldoutA] {
        let e = Expected {
            digest: "x".into(),
            positions: 6,
            trace_manifest: "x".into(),
            split,
        };
        assert!(load_dataset(&hp, &ht, &e).is_err(), "{split:?}");
    }
    assert_eq!(TUNE_DIGEST.len(), 64);
}

#[test]
fn per_budget_samplers_are_independent_balanced_and_identical_across_learning_rates() {
    let _g = rng_lock();
    // 15 cells like P25 TRAIN, of very different sizes.
    let mut cells = Vec::new();
    for (fam, sizes) in [
        ("KQvK", [246usize, 462, 862]),
        ("KRvK", [153, 426, 352]),
        ("KQQvK", [500, 500, 183]),
        ("KQRvK", [500, 500, 500]),
        ("KRRvK", [500, 500, 500]),
    ] {
        for (d, n) in sizes.iter().enumerate() {
            for _ in 0..*n {
                cells.push((fam.to_string(), d as u8 + 1));
            }
        }
    }
    let mut a = BudgetSamplers::new(&cells, 5101);
    let mut b = BudgetSamplers::new(&cells, 5101);
    let mut c = BudgetSamplers::new(&cells, 5102);
    let mut seqs: HashMap<usize, Vec<usize>> = HashMap::new();
    for budget in BUDGETS {
        for _ in 0..900 {
            let (i, o1) = a.draw(budget);
            let (j, o2) = b.draw(budget);
            assert_eq!((i, o1), (j, o2), "same seed => identical sequence");
            seqs.entry(budget).or_default().push(i);
            c.draw(budget);
        }
    }
    // Budgets draw different sequences (independent streams).
    for (x, y) in [(0, 2), (2, 4), (4, 8)] {
        assert_ne!(seqs[&x], seqs[&y], "B{x} and B{y} must not share a stream");
    }
    // A different seed differs.
    let mut c2 = BudgetSamplers::new(&cells, 5102);
    let mut a2 = BudgetSamplers::new(&cells, 5101);
    let differ = (0..200).any(|_| c2.draw(8).0 != a2.draw(8).0);
    assert!(differ);
    // Each budget is balanced across all 15 cells within one example.
    for (budget, st) in a.stats() {
        assert_eq!(st.examples_drawn, 900, "B{budget}");
        assert_eq!(st.cells.len(), 15);
        let counts: Vec<u64> = st.cells.iter().map(|c| c.examples_consumed).collect();
        assert!(
            counts.iter().max().unwrap() - counts.iter().min().unwrap() <= 1,
            "B{budget} {counts:?}"
        );
    }
    // Resume: fast-forwarding a fresh sampler reproduces the continuation exactly.
    let draws = a.draws();
    let mut r = BudgetSamplers::new(&cells, 5101);
    r.fast_forward(&draws).unwrap();
    for budget in BUDGETS {
        for _ in 0..50 {
            assert_eq!(a.draw(budget), r.draw(budget));
        }
    }
    assert!(
        r.fast_forward(&draws).is_err(),
        "only a fresh sampler can be fast-forwarded"
    );
}

#[test]
fn every_optimizer_update_has_equal_budget_exposure() {
    let _g = rng_lock();
    let (ds, _d) = dataset(12, 81);
    let recipe = tiny_recipe(2, 8, 2, 5101);
    let mut tr = Trainer::<TB>::new(recipe, &ds, &Default::default()).unwrap();
    for _ in 0..3 {
        let plan = tr.plan_next();
        assert_eq!(plan.examples(), 16);
        for b in BUDGETS {
            let n: usize = plan
                .micros
                .iter()
                .filter(|m| m.budget == b)
                .map(|m| m.items.len())
                .sum();
            assert_eq!(n, 4, "B{b}");
        }
    }
    assert!(tr.samplers.draws().iter().all(|&d| d == 12));
}

// ---------------------------------------------------------------------------
// Loss normalisation
// ---------------------------------------------------------------------------

pub(crate) fn flat(grads: &GradientsParams, model: &ActiveSearchModel<TB>) -> Vec<f32> {
    struct V<'a> {
        g: &'a GradientsParams,
        out: Vec<f32>,
    }
    impl burn::module::ModuleVisitor<TB> for V<'_> {
        fn visit_float<const D: usize>(&mut self, p: &burn::module::Param<Tensor<TB, D>>) {
            match self
                .g
                .get::<<TB as burn::tensor::backend::AutodiffBackend>::InnerBackend, D>(p.id)
            {
                Some(t) => self.out.extend(t.into_data().to_vec::<f32>().unwrap()),
                None => self
                    .out
                    .extend(std::iter::repeat_n(0.0f32, p.val().shape().num_elements())),
            }
        }
    }
    let mut v = V {
        g: grads,
        out: Vec::new(),
    };
    burn::module::Module::visit(model, &mut v);
    v.out
}

pub(crate) fn plan_of(items: &[(usize, usize, u64)], micro: usize) -> UpdatePlan {
    // items: (budget, index, ordinal) grouped by budget into microbatches of `micro`.
    let mut micros = Vec::new();
    for b in BUDGETS {
        let its: Vec<Item> = items
            .iter()
            .filter(|(bb, _, _)| *bb == b)
            .map(|&(_, index, ordinal)| Item { index, ordinal })
            .collect();
        for chunk in its.chunks(micro) {
            micros.push(Micro {
                budget: b,
                items: chunk.to_vec(),
            });
        }
    }
    UpdatePlan { micros }
}

#[test]
fn the_accumulated_update_equals_the_monolithic_objective_and_ignores_microbatching() {
    let _g = rng_lock();
    let (ds, _d) = dataset(15, 91);
    let device = Default::default();
    let recipe = tiny_recipe(2, 8, 2, 5101);
    <TB as Backend>::seed(&device, 5101);
    let model = ActiveSearchModel::<TB>::new(recipe.model.clone(), &device);
    // The same 24 occurrences, three budget mixes.
    let items: Vec<(usize, usize, u64)> = (0..24)
        .map(|k| (BUDGETS[k % 4], (k * 7) % ds.positions().len(), k as u64))
        .collect();
    let (g_a, r_a) = compute_update(&model, &ds, &plan_of(&items, 6), &recipe, &device).unwrap();
    let (g_b, r_b) = compute_update(&model, &ds, &plan_of(&items, 2), &recipe, &device).unwrap();
    let (g_c, r_c) = compute_update(&model, &ds, &plan_of(&items, 3), &recipe, &device).unwrap();

    // 1. The reported objective is mean policy CE + 1.0 * mean supervised selector CE,
    //    independent of how the update is split into microbatches.
    for r in [&r_b, &r_c] {
        assert!(
            (r.policy_loss - r_a.policy_loss).abs() < 1e-4,
            "{} vs {}",
            r.policy_loss,
            r_a.policy_loss
        );
        assert!((r.selector_loss - r_a.selector_loss).abs() < 1e-4);
        assert_eq!(r.supervised_decisions, r_a.supervised_decisions);
        assert_eq!(r.examples, 24);
    }
    assert!((r_a.total_loss - (r_a.policy_loss + 1.0 * r_a.selector_loss)).abs() < 1e-9);

    // 2. The gradient is independent of the microbatch split.
    let (fa, fb, fc) = (flat(&g_a, &model), flat(&g_b, &model), flat(&g_c, &model));
    assert_eq!(fa.len(), fb.len());
    let rel = |x: &[f32], y: &[f32]| {
        let num: f32 = x
            .iter()
            .zip(y)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        let den: f32 = x.iter().map(|a| a.abs()).fold(1e-12, f32::max);
        num / den
    };
    assert!(rel(&fa, &fb) < 2e-3, "microbatch 6 vs 2: {}", rel(&fa, &fb));
    assert!(rel(&fa, &fc) < 2e-3, "microbatch 6 vs 3: {}", rel(&fa, &fc));
}

#[test]
fn the_selector_weight_does_not_depend_on_how_many_b0_examples_exist() {
    let _g = rng_lock();
    // The selector term is a mean over supervised decisions only: adding B0 examples
    // (which have none) must change the policy mean but not the selector mean.
    let (ds, _d) = dataset(15, 92);
    let device = Default::default();
    let recipe = tiny_recipe(2, 8, 2, 5101);
    <TB as Backend>::seed(&device, 5101);
    let model = ActiveSearchModel::<TB>::new(recipe.model.clone(), &device);
    let base: Vec<(usize, usize, u64)> = (0..12)
        .map(|k| (BUDGETS[1 + k % 3], (k * 5) % 15, k as u64))
        .collect();
    let (_, r1) = compute_update(&model, &ds, &plan_of(&base, 3), &recipe, &device).unwrap();
    let mut with_b0 = base.clone();
    for k in 0..12 {
        with_b0.push((0, (k * 3) % 15, 100 + k as u64));
    }
    let (_, r2) = compute_update(&model, &ds, &plan_of(&with_b0, 3), &recipe, &device).unwrap();
    assert!(
        (r1.selector_loss - r2.selector_loss).abs() < 1e-5,
        "{} vs {}",
        r1.selector_loss,
        r2.selector_loss
    );
    assert_eq!(r1.supervised_decisions, r2.supervised_decisions);
    assert_eq!(
        r2.per_budget[0].supervised_decisions, 0,
        "B0 contributes no selector loss"
    );
    assert_ne!(r1.policy_loss, r2.policy_loss);
}

// ---------------------------------------------------------------------------
// Checkpoint identity and exact CPU resume
// ---------------------------------------------------------------------------

pub(crate) fn params(m: &ActiveSearchModel<TB>) -> Vec<f32> {
    struct P(Vec<f32>);
    impl burn::module::ModuleVisitor<TB> for P {
        fn visit_float<const D: usize>(&mut self, p: &burn::module::Param<Tensor<TB, D>>) {
            self.0.extend(p.val().into_data().to_vec::<f32>().unwrap());
        }
    }
    let mut v = P(Vec::new());
    burn::module::Module::visit(m, &mut v);
    v.0
}

#[test]
fn cpu_resume_is_bit_exact_under_the_p5_recipe() {
    let _g = rng_lock();
    let (ds, _d) = dataset(15, 101);
    let device = Default::default();
    let recipe = tiny_recipe(2, 4, 4, 5101);

    let mut full = Trainer::<TB>::new(recipe.clone(), &ds, &device).unwrap();
    for _ in 0..4 {
        full.step(&ds, &device).unwrap();
    }

    let mut a = Trainer::<TB>::new(recipe.clone(), &ds, &device).unwrap();
    for _ in 0..2 {
        a.step(&ds, &device).unwrap();
    }
    let dir = tmp("resume");
    a.save(&dir).unwrap();
    let mut b = Trainer::<TB>::load(&dir, recipe.clone(), &ds, &device).unwrap();
    assert_eq!(b.updates_done, 2);
    assert_eq!(b.samplers.draws(), a.samplers.draws());
    for _ in 0..2 {
        b.step(&ds, &device).unwrap();
    }
    let (pf, pb) = (params(&full.model), params(&b.model));
    assert_eq!(pf.len(), pb.len());
    let max = pf
        .iter()
        .zip(&pb)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert_eq!(
        max, 0.0,
        "resumed run diverged by {max} over the model parameters"
    );
    // The sampled identities and the loss history agree too.
    assert_eq!(full.samplers.draws(), b.samplers.draws());
    for (x, y) in full.history.iter().zip(&b.history) {
        assert_eq!(x.update, y.update);
        assert_eq!(x.report.policy_loss, y.report.policy_loss);
        assert_eq!(x.report.selector_loss, y.report.selector_loss);
        assert_eq!(x.lr, y.lr);
    }
}

#[test]
fn a_checkpoint_refuses_any_other_recipe() {
    let _g = rng_lock();
    let (ds, _d) = dataset(15, 102);
    let device = Default::default();
    let recipe = tiny_recipe(2, 4, 4, 5101);
    let mut a = Trainer::<TB>::new(recipe.clone(), &ds, &device).unwrap();
    a.step(&ds, &device).unwrap();
    let dir = tmp("mismatch");
    a.save(&dir).unwrap();
    Trainer::<TB>::load(&dir, recipe.clone(), &ds, &device).unwrap();
    for mutate in [
        (|r: &mut Recipe| r.peak_lr = Some(1.5e-4)) as fn(&mut Recipe),
        |r| r.seed = Some(5102),
        |r| r.selector_weight = 0.5,
        |r| r.teacher = "other".into(),
        |r| r.micro += 1,
        |r| r.health_checks = !r.health_checks,
        |r| r.model.active.as_mut().unwrap().workspace_tokens = 6,
    ] {
        let mut other = recipe.clone();
        mutate(&mut other);
        let e = Trainer::<TB>::load(&dir, other, &ds, &device).err();
        assert!(e.is_some(), "a checkpoint must refuse a different recipe");
    }
    // A corrupted sidecar digest is refused as well.
    let mut st: super::train::P5State =
        serde_json::from_slice(&std::fs::read(dir.join("p5-state.json")).unwrap()).unwrap();
    st.recipe_digest = "0".repeat(64);
    std::fs::write(dir.join("p5-state.json"), serde_json::to_vec(&st).unwrap()).unwrap();
    assert!(Trainer::<TB>::load(&dir, recipe, &ds, &device).is_err());
}

type Case<T> = (&'static str, fn(&mut T));

#[test]
fn resume_refuses_every_inconsistent_sidecar_and_checkpoint() {
    let _g = rng_lock();
    use super::train::P5State;
    let (ds, _d) = dataset(15, 103);
    let device = Default::default();
    let recipe = tiny_recipe(2, 4, 4, 5101);
    let mut a = Trainer::<TB>::new(recipe.clone(), &ds, &device).unwrap();
    for _ in 0..2 {
        a.step(&ds, &device).unwrap();
    }
    let dir = tmp("consistency");
    a.save(&dir).unwrap();
    let b = Trainer::<TB>::load(&dir, recipe.clone(), &ds, &device).unwrap();
    assert_eq!((b.updates_done, b.resumptions), (2, 1));

    let side = dir.join("p5-state.json");
    let good = std::fs::read(&side).unwrap();
    let state: P5State = serde_json::from_slice(&good).unwrap();
    let cases: Vec<Case<P5State>> = vec![
        ("updates beyond the recipe", |s| s.updates_done = 99),
        ("short history", |s| {
            s.history.pop();
        }),
        ("long history", |s| {
            let l = s.history[0].clone();
            s.history.push(l);
        }),
        ("mislabelled history", |s| s.history[1].update = 7),
        ("non-finite loss", |s| {
            s.history[0].report.total_loss = f64::NAN
        }),
        ("non-finite grad", |s| {
            s.history[0].report.grad_norm = f32::INFINITY
        }),
        ("wrong history lr", |s| s.history[1].lr *= 2.0),
        ("sampler vector too short", |s| {
            s.sampler_draws.pop();
        }),
        ("sampler draws too high", |s| s.sampler_draws[2] += 1),
        ("sampler draws too low", |s| s.sampler_draws[0] -= 1),
    ];
    for (name, mutate) in cases {
        let mut st = state.clone();
        mutate(&mut st);
        std::fs::write(&side, serde_json::to_vec(&st).unwrap()).unwrap();
        assert!(
            Trainer::<TB>::load(&dir, recipe.clone(), &ds, &device).is_err(),
            "sidecar corruption not refused: {name}"
        );
    }
    std::fs::write(&side, &good).unwrap();

    // Checkpoint metadata must agree with the sidecar and recipe.
    let meta_path = dir.join("checkpoint").join("meta.json");
    let meta_good = std::fs::read(&meta_path).unwrap();
    let meta_cases: Vec<Case<serde_json::Value>> = vec![
        ("step", |m| m["step"] = 1.into()),
        ("update_counter", |m| m["update_counter"] = 1.into()),
        ("lr_schedule_step", |m| m["lr_schedule_step"] = 1.into()),
        ("seed", |m| m["seed"] = 5102.into()),
        ("peak lr", |m| m["lr"] = 1.5e-4.into()),
        ("precision", |m| m["precision"] = "bf16".into()),
        ("backend", |m| m["backend"] = "other".into()),
        ("recurrence", |m| m["recurrence"] = 4.into()),
        ("deep supervision", |m| m["deep_supervision"] = true.into()),
        ("architecture", |m| m["architecture"] = "probe_v1".into()),
    ];
    for (name, mutate) in meta_cases {
        let mut m: serde_json::Value = serde_json::from_slice(&meta_good).unwrap();
        mutate(&mut m);
        std::fs::write(&meta_path, serde_json::to_vec(&m).unwrap()).unwrap();
        assert!(
            Trainer::<TB>::load(&dir, recipe.clone(), &ds, &device).is_err(),
            "checkpoint metadata corruption not refused: {name}"
        );
    }
    std::fs::write(&meta_path, &meta_good).unwrap();
    Trainer::<TB>::load(&dir, recipe.clone(), &ds, &device).unwrap();

    // A checkpoint carried into another LR/seed's sidecar refuses even though every
    // tensor shape matches: the sidecar is rewritten to the other recipe, the
    // checkpoint metadata still names the original.
    for mutate in [
        (|r: &mut Recipe| r.peak_lr = Some(1.5e-4)) as fn(&mut Recipe),
        |r| r.seed = Some(5102),
    ] {
        let mut other = recipe.clone();
        mutate(&mut other);
        let mut st = state.clone();
        st.recipe_digest = other.digest();
        st.recipe = other.clone();
        // The history lr values belong to the original schedule; rebuild them so
        // only the checkpoint metadata can reveal the swap.
        if let Some(lr) = other.peak_lr {
            for h in &mut st.history {
                h.lr = crate::learner::lr_at(h.update, lr, other.warmup, other.updates);
            }
        }
        std::fs::write(&side, serde_json::to_vec(&st).unwrap()).unwrap();
        assert!(
            Trainer::<TB>::load(&dir, other, &ds, &device).is_err(),
            "a checkpoint copied from another lr/seed must refuse"
        );
    }
}

#[test]
fn the_query_content_ablation_replays_the_path_and_changes_only_the_content() {
    use super::ablation::evaluate_query_content_ablation;
    let _g = rng_lock();
    let (ds, _d) = dataset(15, 321);
    let device = Default::default();
    let mut t = Trainer::<TB>::new(tiny_recipe(2, 4, 6, 5101), &ds, &device).unwrap();
    // A few updates so the planner actually reads the queried content.
    for _ in 0..4 {
        t.step(&ds, &device).unwrap();
    }
    let model = t.inference_model();
    for source in [EvalSelection::Teacher, EvalSelection::Active] {
        for budget in [2usize, 4] {
            let r = evaluate_query_content_ablation(
                &model,
                &ds,
                budget,
                source,
                4,
                &Default::default(),
            )
            .unwrap();
            // Replaying the recorded path with normal content reproduces the source run.
            assert!(
                r.replay_vs_source_max_abs_ce_diff < 1e-5,
                "{source:?} B{budget}: replay differs from the source by {}",
                r.replay_vs_source_max_abs_ce_diff
            );
            assert_eq!(r.source.pooled.n, 15);
            eprintln!(
                "{source:?} B{budget}: normal ce {:.9} ablated ce {:.9} changed {}",
                r.normal_state_content.pooled.ce,
                r.ablated_query_state_content.pooled.ce,
                r.positions_top1_changed_by_ablation
            );
            // Removing the state content changes what the planner sees, hence the policy.
            assert!(
                r.ablated_query_state_content.pooled.ce != r.normal_state_content.pooled.ce,
                "{source:?} B{budget}: ablation had no effect on a model that reads the content"
            );
            assert!(r.ablated_query_state_content.pooled.ce.is_finite());
        }
    }
}

#[test]
fn the_ablation_is_evaluation_only_and_refused_outside_an_external_replay() {
    let _g = rng_lock();
    let device = Default::default();
    let (ds, _d) = dataset(6, 322);
    let t = Trainer::<TB>::new(tiny_recipe(2, 4, 2, 5101), &ds, &device).unwrap();
    let model = t.inference_model();
    // No scientific constructor requests it.
    assert!(RunOptions::forced(4).ablation.is_none());
    let roots: Vec<GameState> = ds
        .positions()
        .iter()
        .take(2)
        .map(|p| GameState::from_fen(&p.fen).unwrap())
        .collect();
    let opts = RunOptions::query_content_ablation_v1(2);
    for sel in [Selection::Active, Selection::Fixed, Selection::Random(1)] {
        let e = model.run(&roots, &opts, sel, &Default::default()).err();
        assert!(
            e.is_some_and(|e| e.to_string().contains("evaluation-only")),
            "the ablation must be refused for learned, fixed and random selection"
        );
    }
    // A script that supervises (a training script) is refused too.
    struct Supervising;
    impl QueryScript for Supervising {
        fn next(
            &mut self,
            _e: usize,
            _s: usize,
            _f: &[EdgeRef],
            _t: &Tree,
        ) -> anyhow::Result<recur64_model::active::ScriptStep> {
            Ok(recur64_model::active::ScriptStep {
                follow: 0,
                targets: vec![0],
            })
        }
    }
    let e = model
        .run(
            &roots,
            &opts,
            Selection::Script(&mut Supervising),
            &Default::default(),
        )
        .err()
        .expect("must refuse");
    assert!(e.to_string().contains("never supervises"), "{e}");
}

// ---------------------------------------------------------------------------
// Evaluation and selector diagnostics
// ---------------------------------------------------------------------------

#[test]
fn the_diagnostic_classifies_queries_against_the_proof_structure() {
    let _g = rng_lock();
    let (p, t) = find(KQR, "KQRvK", 2, 111, |t| {
        t.refutations.iter().any(|r| r.replies.len() >= 2)
    });
    let root = GameState::from_fen(&p.fen).unwrap();
    let mut mgr = QueryManager::new(root).unwrap();
    let mut tree = Tree::new(&mgr.packet(0).unwrap()).unwrap();
    let mut teacher = SeededProofTeacher::new(vec![&t], vec![5]);
    // An all-proof trajectory: every query is proof_admissible, residual falls by 1 each.
    let mut records = Vec::new();
    for step in 0..t.q_star as usize {
        let frontier = tree.frontier();
        let st = teacher.next(0, step, &frontier, &tree).unwrap();
        let e = frontier[st.follow].clone();
        let pkt = mgr.query(tree.node(e.node_slot).id, e.action).unwrap();
        let slot = tree.add_child(&e, &pkt).unwrap();
        records.push(recur64_model::active::QueryRecord {
            step,
            parent_slot: e.node_slot,
            action: e.action,
            branch: e.branch,
            depth: tree.node(slot).depth,
            terminal: pkt.terminal,
            frontier_size: frontier.len(),
            selector_entropy: None,
            selector_margin: None,
        });
    }
    let d = classify_queries(&t, &records, t.q_star as usize);
    assert_eq!(d.queries as u64, t.q_star);
    assert_eq!(d.proof_admissible as u64, t.q_star);
    assert_eq!(
        (d.refute_admissible, d.off_target, d.post_completion_queries),
        (0, 0, 0)
    );
    assert_eq!(d.residual_decrease_sum as u64, t.q_star);
    assert_eq!(d.queries_reducing_residual as u64, t.q_star);
    assert_eq!(d.final_residual_sum, 0);
    assert_eq!(d.complete_after_budget, 1);
    assert_eq!(d.first_query_correct_root, 1);
    assert_eq!(d.ideal_ceiling, 1);

    // A wasteful trajectory: open an incorrect root branch first (off-proof), then its
    // refutation: the first is off_target for the empty set, the second refute_admissible.
    let r = t.refutations.iter().find(|r| r.replies.len() >= 2).unwrap();
    let mut mgr = QueryManager::new(GameState::from_fen(&p.fen).unwrap()).unwrap();
    let mut tree = Tree::new(&mgr.packet(0).unwrap()).unwrap();
    let mut recs = Vec::new();
    for (step, want) in [
        (0usize, vec![r.root_action]),
        (1, vec![r.root_action, r.replies[0]]),
    ] {
        let frontier = tree.frontier();
        let paths = node_paths(&tree);
        let e = frontier
            .iter()
            .find(|e| edge_path(&paths, e) == want)
            .expect("edge on the frontier")
            .clone();
        let pkt = mgr.query(tree.node(e.node_slot).id, e.action).unwrap();
        let slot = tree.add_child(&e, &pkt).unwrap();
        recs.push(recur64_model::active::QueryRecord {
            step,
            parent_slot: e.node_slot,
            action: e.action,
            branch: e.branch,
            depth: tree.node(slot).depth,
            terminal: pkt.terminal,
            frontier_size: frontier.len(),
            selector_entropy: None,
            selector_margin: None,
        });
    }
    let d = classify_queries(&t, &recs, 2);
    assert_eq!(
        (d.proof_admissible, d.refute_admissible, d.off_target),
        (0, 1, 1)
    );
    assert_eq!(
        d.residual_decrease_sum, 0,
        "off-proof queries do not reduce the residual"
    );
    assert_eq!(d.first_query_correct_root, 0);
    assert_eq!(d.complete_after_budget, 0);
}

#[test]
fn the_refined_diagnostic_separates_pre_and_post_completion_queries() {
    let _g = rng_lock();
    use super::eval::classify_queries_refined;
    let (p, t) = find(KQR, "KQRvK", 2, 111, |t| {
        t.refutations.iter().any(|r| r.replies.len() >= 2)
    });
    let build = |script: &dyn Fn(usize, &Tree) -> Option<Vec<u16>>, steps: usize| {
        let mut mgr = QueryManager::new(GameState::from_fen(&p.fen).unwrap()).unwrap();
        let mut tree = Tree::new(&mgr.packet(0).unwrap()).unwrap();
        let mut teacher = SeededProofTeacher::new(vec![&t], vec![5]);
        let mut recs = Vec::new();
        for step in 0..steps {
            let frontier = tree.frontier();
            let e = match script(step, &tree) {
                Some(want) => {
                    let paths = node_paths(&tree);
                    frontier
                        .iter()
                        .find(|e| edge_path(&paths, e) == want)
                        .expect("edge on the frontier")
                        .clone()
                }
                None => {
                    let st = teacher.next(0, step, &frontier, &tree).unwrap();
                    frontier[st.follow].clone()
                }
            };
            let pkt = mgr.query(tree.node(e.node_slot).id, e.action).unwrap();
            let slot = tree.add_child(&e, &pkt).unwrap();
            recs.push(recur64_model::active::QueryRecord {
                step,
                parent_slot: e.node_slot,
                action: e.action,
                branch: e.branch,
                depth: tree.node(slot).depth,
                terminal: pkt.terminal,
                frontier_size: frontier.len(),
                selector_entropy: Some(1.0),
                selector_margin: Some(0.5),
            });
        }
        recs
    };
    // Q* proof queries from the teacher, then two filler queries after completion.
    let q = t.q_star as usize;
    let recs = build(&|_, _| None, q);
    let d = classify_queries_refined(&t, &recs, q).unwrap();
    assert_eq!(d.pre_completion_queries, q);
    assert_eq!(d.pre_completion_proof_admissible, q);
    assert_eq!(d.post_completion_queries, 0);
    assert_eq!(d.first_completion_step.get(&(q as u32)), Some(&1));
    assert_eq!(d.never_complete, 0);
    assert_eq!(d.pre_completion_residual_decrease_sum as usize, q);
    assert_eq!(d.pre_completion_selector_stat_count, q);

    // The teacher latches after completion and spends the rest in fixed BFS order.
    let recs = build(&|_, _| None, q + 2);
    let d = classify_queries_refined(&t, &recs, q + 2).unwrap();
    assert_eq!(
        (d.pre_completion_queries, d.post_completion_queries),
        (q, 2)
    );
    assert_eq!(d.post_completion_proof_admissible, 0);
    assert_eq!(
        d.post_completion_refute_admissible + d.post_completion_off_target,
        2
    );
    assert_eq!(d.positions_with_post_completion_queries, 1);
    assert_eq!(d.queries, q + 2);
    for k in [1u32, 2, 4, 8] {
        let want = u64::from(q as u32 <= k && k as usize <= q + 2);
        assert_eq!(d.complete_after_query.get(&k).copied().unwrap_or(0), want);
    }

    // An incomplete trajectory never completes and has no post-completion queries.
    let r = t.refutations.iter().find(|r| r.replies.len() >= 2).unwrap();
    let want = [vec![r.root_action], vec![r.root_action, r.replies[0]]];
    let recs = build(&|step, _| Some(want[step].clone()), 2);
    let d = classify_queries_refined(&t, &recs, 2).unwrap();
    assert_eq!(d.never_complete, 1);
    assert!(d.first_completion_step.is_empty());
    assert_eq!(d.post_completion_queries, 0);
    assert_eq!(d.pre_completion_queries, 2);
    assert_eq!(
        (
            d.pre_completion_proof_admissible,
            d.pre_completion_refute_admissible,
            d.pre_completion_off_target
        ),
        (0, 1, 1)
    );
}

#[test]
fn evaluation_is_complete_finite_and_the_screen_score_follows_the_frozen_formula() {
    let _g = rng_lock();
    let (ds, _d) = dataset(18, 121);
    let device = Default::default();
    <CpuTrainBackend as Backend>::seed(&device, 7);
    let model = ActiveSearchModel::<burn::backend::Flex>::new(tiny_model(), &device);
    let mut act = Vec::new();
    for b in BUDGETS {
        let out = evaluate(&model, &ds, b, EvalSelection::Active, 7, &device).unwrap();
        assert_eq!(out.summary.pooled.n, 18);
        assert!(out.summary.pooled.ce.is_finite() && out.summary.pooled.entropy.is_finite());
        assert!((0.0..=1.0).contains(&out.summary.pooled.top1));
        assert!((0.0..=1.0).contains(&out.summary.pooled.correct_mass));
        assert!(out.summary.cells.len() >= 6, "mixed-cell fixture");
        assert_eq!(out.selector_diag.is_some(), b > 0);
        if let Some((all, cells)) = &out.selector_diag {
            assert_eq!(all.examples, 18);
            assert_eq!(all.queries, 18 * b.min(8));
            assert_eq!(cells.values().map(|c| c.examples).sum::<usize>(), 18);
            assert_eq!(
                all.proof_admissible + all.refute_admissible + all.off_target,
                all.queries
            );
        }
        act.push(out.summary);
    }
    // The score needs exactly six cells per budget: the fixture has nine, so it refuses.
    assert!(act.iter().all(|s| s.cells.len() != 6) || screen_score(&act).is_ok());
    if act[0].cells.len() != 6 {
        assert!(screen_score(&act).is_err());
    }
    // A synthetic six-cell set follows the formula exactly.
    let mut six = act.clone();
    for (k, s) in six.iter_mut().enumerate() {
        let keep: Vec<_> = s.cells.keys().take(6).cloned().collect();
        s.cells.retain(|c, _| keep.contains(c));
        for (j, m) in s.cells.values_mut().enumerate() {
            m.ce = (k * 10 + j) as f64;
        }
    }
    let expect = (0..4)
        .map(|k| (0..6).map(|j| (k * 10 + j) as f64).sum::<f64>() / 6.0)
        .sum::<f64>()
        / 4.0;
    assert!((screen_score(&six).unwrap() - expect).abs() < 1e-12);
    // B16 or a missing budget is refused.
    assert!(screen_score(&six[..3]).is_err());

    // Teacher and FIXED diagnostics run and are finite; teacher mode reports the NLL.
    let t = evaluate(&model, &ds, 4, EvalSelection::Teacher, 6, &device).unwrap();
    assert!(t.selector_nll.is_some_and(|(s, c)| s.is_finite() && c > 0));
    let f = evaluate(&model, &ds, 4, EvalSelection::Fixed, 6, &device).unwrap();
    assert!(f.summary.pooled.ce.is_finite());
    // B16 is not an evaluation budget in P5.
    assert!(evaluate(&model, &ds, 16, EvalSelection::Active, 6, &device).is_err());
}

#[test]
fn p5_never_trains_b16_or_uses_holdout_c() {
    let _g = rng_lock();
    let mut r = tiny_recipe(2, 8, 2, 1);
    assert!(!r.budgets.contains(&16) && !r.budget_sequence.contains(&16));
    r.budgets.push(16);
    assert!(Recipe::validate(&r).is_err());
    // The data loader cannot be pointed at HOLDOUT_C: it is a distinct digest and split.
    assert_ne!(crate::proof::custody::HOLDOUT_C_DIGEST, TUNE_DIGEST);
    assert_ne!(
        crate::proof::custody::HOLDOUT_C_DIGEST,
        super::recipe::TRAIN_DIGEST
    );
}

#[test]
fn a_training_update_runs_end_to_end_and_reduces_nothing_it_should_not() {
    let _g = rng_lock();
    // One real update: finite losses, per-budget exposure, B0 has no supervised steps,
    // completed proofs are counted, and the optimizer moves the weights.
    let (ds, _d) = dataset(15, 131);
    let device = Default::default();
    let recipe = tiny_recipe(2, 4, 3, 5101);
    let mut tr = Trainer::<TB>::new(recipe, &ds, &device).unwrap();
    let before = params(&tr.model);
    let rec = tr.step(&ds, &device).unwrap();
    assert!(rec.report.total_loss.is_finite() && rec.report.grad_norm.is_finite());
    assert_eq!(rec.report.examples, 8);
    assert_eq!(rec.report.per_budget[0].budget, 0);
    assert_eq!(rec.report.per_budget[0].supervised_decisions, 0);
    assert!(rec.report.per_budget[3].supervised_decisions > 0);
    assert_ne!(
        before,
        params(&tr.model),
        "the update must change the weights"
    );
    assert!(rec.lr > 0.0);
    // The same occurrence of the same position always gets the same teacher trajectory.
    let p = &ds.positions()[0];
    let root = GameState::from_fen(&p.fen).unwrap();
    let k = follow_key(teacher_key_base(5101), &p.id, 3, 8);
    let a = simulate_episode(&ds.traces[0], &root, 8, k).unwrap();
    let b = simulate_episode(&ds.traces[0], &root, 8, k).unwrap();
    assert_eq!(
        (a.supervised, a.completed_at),
        (b.supervised, b.completed_at)
    );
}

#[test]
fn model_free_simulation_matches_the_real_run_tree_semantics() {
    let _g = rng_lock();
    // simulate_episode must reproduce exactly what ActiveSearchModel::run does with
    // the same teacher: the supervised count of every example equals the model's.
    let (ds, _d) = dataset(9, 141);
    let device = Default::default();
    let model = ActiveSearchModel::<burn::backend::Flex>::new(tiny_model(), &device);
    for budget in [2usize, 4, 8] {
        let idx: Vec<usize> = (0..9).collect();
        let roots: Vec<GameState> = idx
            .iter()
            .map(|&i| GameState::from_fen(&ds.positions()[i].fen).unwrap())
            .collect();
        let keys: Vec<u64> = idx
            .iter()
            .map(|&i| follow_key(teacher_key_base(1), &ds.positions()[i].id, i as u64, budget))
            .collect();
        let traces: Vec<&PositionTrace> = idx.iter().map(|&i| &ds.traces[i]).collect();
        let mut teacher = SeededProofTeacher::new(traces, keys.clone());
        let mut opts = RunOptions::forced(budget);
        opts.health_checks = false;
        let out = model
            .run(&roots, &opts, Selection::Script(&mut teacher), &device)
            .unwrap();
        let counted: usize = out
            .selector_steps
            .iter()
            .map(|s| s.has_target.iter().filter(|&&t| t).count())
            .sum();
        let mut total = 0;
        for (n, &i) in idx.iter().enumerate() {
            let rec = simulate_episode(&ds.traces[i], &roots[n], budget, keys[n]).unwrap();
            assert_eq!(
                rec.supervised, teacher.records[n].supervised,
                "example {n} B{budget}"
            );
            assert_eq!(rec.completed_at, teacher.records[n].completed_at);
            total += rec.supervised;
        }
        assert_eq!(total, counted);
    }
}
