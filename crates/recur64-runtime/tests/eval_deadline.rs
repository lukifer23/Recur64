//! D38: the pilot's evaluation honours the run deadline at game boundaries and
//! reports an incomplete evaluation as a typed error the pilot turns into a
//! held cycle, never as a partial result that could inform promotion.

use burn::backend::{Autodiff, Flex};

use recur64_model::checkpoint::{CheckpointMeta, save_training};
use recur64_model::model::ProbeModel;
use recur64_model::train::adamw;
use recur64_runtime::{EvalModels, RunConfig, evaluate_candidate};
use recur64_search::EvalError;

type TB = Autodiff<Flex>;

#[test]
fn evaluation_past_the_deadline_is_a_typed_incomplete_error() {
    let cfg = RunConfig::from_toml_str(
        "run_id = 'deadline'\narena_games = 2\nsimulations_per_move = 2\nply_cap = 8\n[model]\nwidth = 32\nheads = 4\nffn = 64\ninput_blocks = 0\ncore_blocks = 1\noutput_blocks = 0\n",
    )
    .unwrap();
    let dir = std::env::temp_dir().join(format!("recur64-eval-deadline-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let device = Default::default();
    let model = ProbeModel::<TB>::new(cfg.model.clone(), &device);
    let meta = CheckpointMeta::new(cfg.model.clone(), 1, false, 0, 3e-4, 1, 0, "cpu", "fp32");
    save_training(&dir, &model, &adamw::<TB, _>(), &meta).unwrap();
    let models = EvalModels {
        parent_dir: &dir,
        parent_model_id: "m",
        candidate_dir: &dir,
        candidate_model_id: "m",
        reference_dir: &dir,
        reference_model_id: "m",
    };
    let past = std::time::Instant::now() - std::time::Duration::from_secs(1);
    let err = evaluate_candidate::<Flex>(&cfg, &models, 0, &[], 1, &device, Some(past))
        .expect_err("evaluation after the deadline must not complete");
    assert!(
        matches!(
            err.downcast_ref::<EvalError>(),
            Some(EvalError::DeadlineExceeded(_))
        ),
        "{err}"
    );
    // Without a deadline the same evaluation completes.
    evaluate_candidate::<Flex>(&cfg, &models, 0, &[], 1, &device, None).expect("completes");
    std::fs::remove_dir_all(&dir).ok();
}

/// D46: the two-resident-owner evaluation schedule produces results identical
/// to the original three-owner sequence (arena parent, arena reference, raw vs
/// random, raw vs parent), both before and after a promotion.
#[test]
fn two_owner_evaluation_matches_the_three_owner_sequence() {
    use recur64_runtime::eval_policy::{raw_policy_vs_parent, raw_policy_vs_random};
    use recur64_runtime::spawn_owner;
    let cfg = RunConfig::from_toml_str(
        "run_id = 'd46'\narena_games = 4\nsimulations_per_move = 4\nply_cap = 24\n[model]\nwidth = 32\nheads = 4\nffn = 64\ninput_blocks = 0\ncore_blocks = 1\noutput_blocks = 0\n",
    )
    .unwrap();
    let device = Default::default();
    let base = std::env::temp_dir().join(format!("recur64-d46-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    let save = |name: &str, seed: u64| {
        use burn::prelude::Backend;
        <TB as Backend>::seed(&device, seed);
        let dir = base.join(name);
        let model = ProbeModel::<TB>::new(cfg.model.clone(), &device);
        let meta =
            CheckpointMeta::new(cfg.model.clone(), 1, false, 0, 3e-4, seed, 0, "cpu", "fp32");
        save_training(&dir, &model, &adamw::<TB, _>(), &meta).unwrap();
        dir
    };
    let (ref_dir, parent_dir, cand_dir) = (save("ref", 1), save("parent", 2), save("cand", 3));
    let json = |v: &dyn erased::Ser| v.to_json();
    for (parent, parent_id) in [(&ref_dir, "ref"), (&parent_dir, "parent")] {
        let models = EvalModels {
            parent_dir: parent,
            parent_model_id: parent_id,
            candidate_dir: &cand_dir,
            candidate_model_id: "cand",
            reference_dir: &ref_dir,
            reference_model_id: "ref",
        };
        let out = evaluate_candidate::<Flex>(&cfg, &models, 1, &[], 2, &device, None).unwrap();
        assert!(out.max_resident_owners <= 2);
        assert_eq!(out.owners_spawned, if parent_id == "ref" { 2 } else { 3 });

        // The original sequence with all three owners resident.
        let (p, c, r) = (
            spawn_owner::<Flex>(parent, &cfg, &device).unwrap(),
            spawn_owner::<Flex>(&cand_dir, &cfg, &device).unwrap(),
            spawn_owner::<Flex>(&ref_dir, &cfg, &device).unwrap(),
        );
        let (pe, ce, re) = (p.evaluator(), c.evaluator(), r.evaluator());
        let acfg = cfg.arena_config(1, Vec::new(), 2);
        let arena = recur64_eval::run_arena(&pe, &ce, parent_id, "cand", &acfg).unwrap();
        let reference = if parent_id == "ref" {
            arena.clone()
        } else {
            recur64_eval::run_arena(&re, &ce, "ref", "cand", &acfg).unwrap()
        };
        let raw = raw_policy_vs_random(
            &ce,
            4,
            cfg.temperature,
            cfg.ply_cap,
            cfg.seed + 1,
            &[],
            2,
            None,
        )
        .unwrap();
        let raw_parent =
            raw_policy_vs_parent(&ce, &pe, 4, cfg.ply_cap, cfg.seed + 1, &[], 2, None).unwrap();
        assert_eq!(json(&out.arena), json(&arena), "parent arena ({parent_id})");
        assert_eq!(
            json(&out.reference_arena),
            json(&reference),
            "reference arena ({parent_id})"
        );
        assert_eq!(json(&out.raw), json(&raw), "raw vs random ({parent_id})");
        assert_eq!(
            json(&out.raw_parent),
            json(&raw_parent),
            "raw vs parent ({parent_id})"
        );
        drop((pe, ce, re));
        p.shutdown();
        c.shutdown();
        r.shutdown();
    }
    std::fs::remove_dir_all(&base).ok();
}

mod erased {
    pub trait Ser {
        fn to_json(&self) -> String;
    }
    impl<T: serde::Serialize> Ser for T {
        fn to_json(&self) -> String {
            serde_json::to_string(self).unwrap()
        }
    }
}
