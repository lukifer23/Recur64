//! P6 trainer, loaders and evaluation, on a tiny `all_info_v1` geometry (CPU, FP32) with
//! real exact positions. Production code only ever passes the frozen constants.

use burn::prelude::*;

use recur64_model::all_info::AllInfoModel;
use recur64_model::config::{CandidateConfig, ModelConfig};
use recur64_model::train::CpuTrainBackend;

use crate::p5::tests::{mini_dataset, tmp};
use crate::proof::targets::{ProofTargets, Split};

use super::data::{Which, load_targets};
use super::eval::evaluate_all_info;
use super::recipe::P6Recipe;
use super::train::{P6State, P6Trainer};

type TB = CpuTrainBackend;

fn tiny_model() -> ModelConfig {
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

fn tiny_recipe(micro: usize, accum: usize, updates: u64, seed: u64) -> P6Recipe {
    let mut r = P6Recipe::contract(micro, accum);
    r.model = tiny_model();
    r.updates = updates;
    r.warmup = 1;
    r.peak_lr = Some(3.0e-3);
    r.for_seed(seed)
}

fn train_targets(count: usize, seed: u64) -> ProofTargets {
    let d = tmp(&format!("p6_ds_{seed}_{count}"));
    let (p, _t) = mini_dataset(Split::Train, &d, count, seed);
    ProofTargets::load(&p).unwrap()
}

fn params<B: Backend>(m: &AllInfoModel<B>) -> Vec<f32> {
    struct P(Vec<f32>);
    impl<Bk: Backend> burn::module::ModuleVisitor<Bk> for P {
        fn visit_float<const D: usize>(&mut self, p: &burn::module::Param<Tensor<Bk, D>>) {
            self.0.extend(p.val().into_data().to_vec::<f32>().unwrap());
        }
    }
    let mut v = P(Vec::new());
    burn::module::Module::visit(m, &mut v);
    v.0
}

#[test]
fn cpu_resume_is_bit_exact_under_the_p6_recipe() {
    let _g = crate::p5::tests::rng_lock();
    let ds = train_targets(15, 301);
    let device = Default::default();
    let recipe = tiny_recipe(2, 4, 4, 5101);
    // The tiny effective batch is 8; positions are drawn from the 15 cells' sampler.
    let mut full = P6Trainer::<TB>::new(recipe.clone(), &ds, &device).unwrap();
    for _ in 0..4 {
        full.step(&ds, &device).unwrap();
    }
    let mut a = P6Trainer::<TB>::new(recipe.clone(), &ds, &device).unwrap();
    for _ in 0..2 {
        a.step(&ds, &device).unwrap();
    }
    let dir = tmp("p6_resume");
    a.save(&dir).unwrap();
    let mut b = P6Trainer::<TB>::load(&dir, recipe, &ds, &device).unwrap();
    assert_eq!((b.updates_done, b.resumptions), (2, 1));
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
    assert_eq!(max, 0.0, "resumed run diverged by {max}");
    for (x, y) in full.history.iter().zip(&b.history) {
        assert_eq!(x.report.policy_loss, y.report.policy_loss);
        assert_eq!(x.lr, y.lr);
    }
    assert_eq!(full.sampler.examples_drawn(), b.sampler.examples_drawn());
}

#[test]
fn resume_refuses_every_inconsistent_sidecar_and_checkpoint() {
    let _g = crate::p5::tests::rng_lock();
    let ds = train_targets(15, 302);
    let device = Default::default();
    let recipe = tiny_recipe(2, 4, 4, 5101);
    let mut a = P6Trainer::<TB>::new(recipe.clone(), &ds, &device).unwrap();
    for _ in 0..2 {
        a.step(&ds, &device).unwrap();
    }
    let dir = tmp("p6_consistency");
    a.save(&dir).unwrap();
    P6Trainer::<TB>::load(&dir, recipe.clone(), &ds, &device).unwrap();

    // Another recipe (seed, LR, loss weight) never loads the checkpoint.
    for mutate in [
        (|r: &mut P6Recipe| r.seed = Some(5102)) as fn(&mut P6Recipe),
        |r| r.peak_lr = Some(1.5e-4),
        |r| r.selector_weight = 1.0,
        |r| r.micro += 1,
    ] {
        let mut other = recipe.clone();
        mutate(&mut other);
        assert!(P6Trainer::<TB>::load(&dir, other, &ds, &device).is_err());
    }

    let side = dir.join("p6-state.json");
    let good = std::fs::read(&side).unwrap();
    let state: P6State = serde_json::from_slice(&good).unwrap();
    type Case = (&'static str, fn(&mut P6State));
    let cases: Vec<Case> = vec![
        ("updates beyond the recipe", |s| s.updates_done = 99),
        ("short history", |s| {
            s.history.pop();
        }),
        ("mislabelled history", |s| s.history[1].update = 7),
        ("non-finite loss", |s| {
            s.history[0].report.policy_loss = f64::NAN
        }),
        ("wrong history lr", |s| s.history[1].lr *= 2.0),
        ("sampler draws too high", |s| s.sampler_draws += 1),
        ("sampler draws too low", |s| s.sampler_draws -= 1),
    ];
    for (name, mutate) in cases {
        let mut st = state.clone();
        mutate(&mut st);
        std::fs::write(&side, serde_json::to_vec(&st).unwrap()).unwrap();
        assert!(
            P6Trainer::<TB>::load(&dir, recipe.clone(), &ds, &device).is_err(),
            "sidecar corruption not refused: {name}"
        );
    }
    std::fs::write(&side, &good).unwrap();

    let meta_path = dir.join("checkpoint").join("meta.json");
    let meta_good = std::fs::read(&meta_path).unwrap();
    type MCase = (&'static str, fn(&mut serde_json::Value));
    let meta_cases: Vec<MCase> = vec![
        ("step", |m| m["step"] = 1.into()),
        ("seed", |m| m["seed"] = 5102.into()),
        ("peak lr", |m| m["lr"] = 1.5e-4.into()),
        ("precision", |m| m["precision"] = "bf16".into()),
        ("backend", |m| m["backend"] = "v3-p5".into()),
        ("architecture", |m| {
            m["architecture"] = "active_search_v3".into()
        }),
    ];
    for (name, mutate) in meta_cases {
        let mut m: serde_json::Value = serde_json::from_slice(&meta_good).unwrap();
        mutate(&mut m);
        std::fs::write(&meta_path, serde_json::to_vec(&m).unwrap()).unwrap();
        assert!(
            P6Trainer::<TB>::load(&dir, recipe.clone(), &ds, &device).is_err(),
            "checkpoint metadata corruption not refused: {name}"
        );
    }
    std::fs::write(&meta_path, &meta_good).unwrap();
    P6Trainer::<TB>::load(&dir, recipe, &ds, &device).unwrap();
}

#[test]
fn an_update_is_finite_counts_every_supplied_state_and_ignores_microbatching() {
    let _g = crate::p5::tests::rng_lock();
    let ds = train_targets(15, 303);
    let device = Default::default();
    // The same eight examples as 2x4 and as 4x2 give the same loss (up to float order).
    let idx: Vec<usize> = (0..8).collect();
    let mut losses = Vec::new();
    for (m, ac) in [(2usize, 4usize), (4, 2), (8, 1)] {
        let recipe = tiny_recipe(m, ac, 2, 5101);
        let t = P6Trainer::<TB>::new(recipe.clone(), &ds, &device).unwrap();
        let (_g, rep) =
            super::train::compute_update(&t.model, &ds, &idx, &recipe, &device).unwrap();
        assert!(rep.policy_loss.is_finite() && rep.grad_norm.is_finite());
        assert_eq!(rep.examples, 8);
        assert!(rep.states_supplied >= 8 * 17, "exhaustive trees are large");
        losses.push(rep.policy_loss);
    }
    for l in &losses[1..] {
        assert!((l - losses[0]).abs() < 1e-5, "{losses:?}");
    }
}

#[test]
fn evaluation_is_complete_and_scored_by_the_shared_metric() {
    let _g = crate::p5::tests::rng_lock();
    let ds = train_targets(15, 304);
    let device = Default::default();
    let recipe = tiny_recipe(2, 4, 2, 5101);
    let t = P6Trainer::<TB>::new(recipe, &ds, &device).unwrap();
    let model = t.inference_model();
    let (res, summary, states) = evaluate_all_info(&model, &ds, 4, &Default::default()).unwrap();
    assert_eq!(res.len(), 15);
    assert!(states > 15 * 17);
    assert_eq!(summary.pooled.n, 15);
    assert!(
        res.iter()
            .all(|r| r.ce.is_finite() && (0.0..=1.0).contains(&r.mass))
    );
    // Batch size must not change per-position results.
    let (res2, _, _) = evaluate_all_info(&model, &ds, 15, &Default::default()).unwrap();
    for (a, b) in res.iter().zip(&res2) {
        assert!((a.ce - b.ce).abs() < 1e-4 && a.top1 == b.top1);
    }
}

#[test]
fn the_loaders_refuse_everything_but_the_frozen_train_and_tune_identities() {
    let _g = crate::p5::tests::rng_lock();
    let d = tmp("p6_loaders");
    // A fixture is not the frozen split: wrong digest and count.
    let (p, _t) = mini_dataset(Split::Train, &d, 6, 305);
    assert!(load_targets(&p, Which::Train).is_err());
    assert!(load_targets(&p, Which::Tune).is_err());
    // A holdout can never be loaded through a working path.
    let src = ProofTargets::load(&p).unwrap();
    let mut positions = src.positions.clone();
    for q in &mut positions {
        q.split = Split::HoldoutC;
    }
    let h = d.join("holdout.json");
    ProofTargets::new(
        Split::HoldoutC,
        306,
        serde_json::json!({"fixture": "p6"}),
        positions,
    )
    .save(&h)
    .unwrap();
    let e = load_targets(&h, Which::Tune).unwrap_err().to_string();
    assert!(e.contains("not allowed") || e.contains("sealed"), "{e}");
    assert!(load_targets(&h, Which::Train).is_err());
}
