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
