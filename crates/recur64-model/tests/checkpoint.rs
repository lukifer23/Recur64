//! Phase 0 checkpoint/resume proof.
//!
//! On deterministic CPU FP32, a resumed run must match an uninterrupted run
//! closely. GPU equality is tolerance-bounded and is not claimed here.

use std::path::PathBuf;

use burn::optim::Optimizer;
use burn::prelude::*;

use recur64_model::checkpoint::{CheckpointMeta, SCHEMA_VERSION, load_training, save_training};
use recur64_model::config::ModelConfig;
use recur64_model::fixture::SynthFixture;
use recur64_model::train::{CpuTrainBackend, adamw, train_step};

type B = CpuTrainBackend;

fn cfg() -> ModelConfig {
    ModelConfig {
        width: 32,
        heads: 4,
        ffn: 64,
        input_blocks: 1,
        core_blocks: 2,
        output_blocks: 1,
        squares: 64,
        in_features: 119,
        policy_dim: 16,
        wdl_classes: 3,
        promo_codes: 5,
        rms_eps: 1e-5,
    }
}

fn scalar(t: Tensor<B, 1>) -> f32 {
    t.into_data().to_vec::<f32>().unwrap()[0]
}

fn tmp_dir() -> PathBuf {
    std::env::temp_dir().join("recur64_phase0_ckpt")
}

#[test]
fn resumed_matches_uninterrupted() {
    let device = Default::default();
    let model0 = recur64_model::model::ProbeModel::<B>::new(cfg(), &device);
    let optim0 = adamw::<B, _>();
    let fx = SynthFixture::new(8, 119, 77);
    let (board, cands, targets) = fx.tensors::<B>(&device);
    let lr = 3e-3;
    let n = 6usize;
    let k = 3usize;

    // Run A: uninterrupted.
    let mut model_a = model0.clone();
    let mut optim_a = optim0.clone();
    let mut loss_a = 0.0;
    for _ in 0..n {
        let (m, l) = train_step(
            model_a,
            &mut optim_a,
            board.clone(),
            &cands,
            &targets,
            1,
            false,
            lr,
        );
        model_a = m;
        loss_a = scalar(l);
    }

    // Run B: train K, checkpoint, reload, train the rest.
    let mut model_b = model0.clone();
    let mut optim_b = optim0.clone();
    for _ in 0..k {
        let (m, _) = train_step(
            model_b,
            &mut optim_b,
            board.clone(),
            &cands,
            &targets,
            1,
            false,
            lr,
        );
        model_b = m;
    }

    let dir = tmp_dir();
    let _ = std::fs::remove_dir_all(&dir);
    let meta = CheckpointMeta {
        schema_version: SCHEMA_VERSION,
        recur64_version: recur64_model::VERSION.to_string(),
        git_revision: None,
        backend: "cpu (Burn Flex) autodiff".to_string(),
        precision: "fp32".to_string(),
        model: cfg(),
        recurrence: 1,
        deep_supervision: false,
        step: k as u64,
        lr,
        seed: 77,
        rng_state: 0,
    };
    save_training(&dir, &model_b, &optim_b, &meta).expect("save checkpoint");

    // Reload into a template that shares the original parameter identities.
    let template = model0.clone();
    let (mut model_b2, mut optim_b2, loaded_meta) =
        load_training(&dir, template, optim0.clone(), &device).expect("load checkpoint");
    assert_eq!(loaded_meta.step, k as u64);
    assert_eq!(loaded_meta.schema_version, SCHEMA_VERSION);
    // Parameter identities and values must survive the round trip.
    assert_eq!(model_b.core_weight_id(), model_b2.core_weight_id());
    assert_eq!(
        model_b.core_weight_scalar(),
        model_b2.core_weight_scalar(),
        "model weights must be restored exactly"
    );
    // Optimizer moments must be present after restore.
    assert_eq!(optim_b.to_record().len(), optim_b2.to_record().len());
    assert!(!optim_b2.to_record().is_empty());

    let mut loss_b = 0.0;
    for _ in k..n {
        let (m, l) = train_step(
            model_b2,
            &mut optim_b2,
            board.clone(),
            &cands,
            &targets,
            1,
            false,
            lr,
        );
        model_b2 = m;
        loss_b = scalar(l);
    }

    // Resumed final loss must match the uninterrupted final loss closely.
    let dl = (loss_a - loss_b).abs();
    let dw = (model_a.core_weight_scalar() - model_b2.core_weight_scalar()).abs();
    println!("uninterrupted loss={loss_a} resumed loss={loss_b} |dl|={dl} |dw|={dw}");
    assert!(dl < 1e-4, "resumed loss {loss_b} vs uninterrupted {loss_a}");
    assert!(dw < 1e-5, "core weight drift {dw}");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn cpu_training_is_deterministic() {
    let device = Default::default();
    let model0 = recur64_model::model::ProbeModel::<B>::new(cfg(), &device);
    let optim0 = adamw::<B, _>();
    let fx = SynthFixture::new(8, 119, 77);
    let (board, cands, targets) = fx.tensors::<B>(&device);
    // Value-preserving clones are required for a meaningful resume comparison.
    let m1 = model0.clone();
    assert_eq!(m1.core_weight_scalar(), model0.core_weight_scalar());

    let run = |mut m: recur64_model::model::ProbeModel<B>, mut o: _| {
        let mut last = 0.0;
        for _ in 0..4 {
            let (nm, l) = train_step(m, &mut o, board.clone(), &cands, &targets, 1, false, 3e-3);
            m = nm;
            last = scalar(l);
        }
        (last, m.core_weight_scalar())
    };
    let (l1, w1) = run(model0.clone(), optim0.clone());
    let (l2, w2) = run(model0, optim0);
    println!("determinism: l1={l1} l2={l2} w1={w1} w2={w2}");
    assert!(
        (l1 - l2).abs() < 1e-6,
        "loss not deterministic: {l1} vs {l2}"
    );
    assert!(
        (w1 - w2).abs() < 1e-6,
        "weights not deterministic: {w1} vs {w2}"
    );
}

#[test]
fn schema_mismatch_is_refused() {
    let device = Default::default();
    let model = recur64_model::model::ProbeModel::<B>::new(cfg(), &device);
    let optim = adamw::<B, _>();
    let dir = tmp_dir().join("bad_schema");
    let _ = std::fs::remove_dir_all(&dir);
    let mut meta = CheckpointMeta {
        schema_version: SCHEMA_VERSION,
        recur64_version: recur64_model::VERSION.to_string(),
        git_revision: None,
        backend: "cpu".to_string(),
        precision: "fp32".to_string(),
        model: cfg(),
        recurrence: 1,
        deep_supervision: false,
        step: 0,
        lr: 3e-4,
        seed: 0,
        rng_state: 0,
    };
    save_training(&dir, &model, &optim, &meta).expect("save");
    meta.schema_version = SCHEMA_VERSION + 99;
    std::fs::write(dir.join("meta.json"), serde_json::to_vec(&meta).unwrap()).unwrap();

    let res = load_training(&dir, model, optim, &device);
    assert!(res.is_err(), "loading a mismatched schema must fail");
    let err = res.err().unwrap();
    assert!(
        err.to_string().contains("schema mismatch"),
        "expected visible schema error, got: {err}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
