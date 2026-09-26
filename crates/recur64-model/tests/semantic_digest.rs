//! Semantic weight digest vs artifact identity (H3.5B / D50 amendment).
//!
//! `model_id` hashes the `.mpk` bytes, which include Burn's generated
//! `ParamId`s. The semantic digest must depend on values, shapes, names and
//! config only.

use std::path::PathBuf;

use burn::module::list_param_ids;
use burn::prelude::*;

use recur64_model::checkpoint::{CheckpointMeta, hash_file, load_training, save_training};
use recur64_model::config::ModelConfig;
use recur64_model::digest::{compare_weights, digest_named_params, semantic_weights_digest};
use recur64_model::model::{NamedParam, ProbeModel};
use recur64_model::train::{CpuTrainBackend, adamw};

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

/// Flex keeps one process-global seeded RNG, and the test harness runs tests
/// on parallel threads. Seed + construct must be atomic or tests interleave
/// draws (measured: 45,828 differing elements without this lock).
static SEED_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn seeded(seed: u64) -> ProbeModel<B> {
    let _guard = SEED_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let device = Default::default();
    <B as Backend>::seed(&device, seed);
    ProbeModel::<B>::new(cfg(), &device)
}

fn tmp_dir(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("recur64_semantic_digest_{name}"))
}

fn save(dir: &PathBuf, model: &ProbeModel<B>) -> String {
    let _ = std::fs::remove_dir_all(dir);
    let meta = CheckpointMeta::new(cfg(), 1, false, 0, 3e-4, 0, 0, "cpu", "fp32");
    save_training(dir, model, &adamw::<B, _>(), &meta).expect("save");
    hash_file(&dir.join("model.mpk")).unwrap()
}

#[test]
fn digest_covers_every_parameter_exactly_once() {
    let model = seeded(1);
    let d = semantic_weights_digest(&model).unwrap();
    assert_eq!(d.element_count, model.num_params());
    assert_eq!(d.tensor_count, list_param_ids(&model).len());
    let names: Vec<String> = model
        .named_float_params()
        .unwrap()
        .into_iter()
        .map(|p| p.name)
        .collect();
    let mut unique = names.clone();
    unique.sort();
    unique.dedup();
    assert_eq!(unique.len(), names.len(), "parameter names must be unique");
}

#[test]
fn digest_ignores_param_ids_but_artifact_hash_does_not() {
    let (a, b) = (seeded(7), seeded(7));
    assert_ne!(
        a.core_weight_id(),
        b.core_weight_id(),
        "fresh construction draws new ParamIds"
    );
    let cmp = compare_weights(&a, &b).unwrap();
    assert_eq!(
        cmp.elements_differing, 0,
        "same seed, same process: same values"
    );
    assert_eq!(
        semantic_weights_digest(&a).unwrap(),
        semantic_weights_digest(&b).unwrap()
    );
    let (ha, hb) = (save(&tmp_dir("ids_a"), &a), save(&tmp_dir("ids_b"), &b));
    assert_ne!(ha, hb, "the .mpk records the differing ParamIds");
    let _ = std::fs::remove_dir_all(tmp_dir("ids_a"));
    let _ = std::fs::remove_dir_all(tmp_dir("ids_b"));
}

#[test]
fn digest_is_stable_across_save_and_load() {
    let model = seeded(3);
    let dir = tmp_dir("roundtrip");
    save(&dir, &model);
    let device = Default::default();
    let (loaded, _, _) = load_training(
        &dir,
        ProbeModel::<B>::new(cfg(), &device),
        adamw::<B, _>(),
        &device,
    )
    .unwrap();
    assert_eq!(
        semantic_weights_digest(&model).unwrap(),
        semantic_weights_digest(&loaded).unwrap()
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn digest_changes_with_one_value_and_different_seeds() {
    let model = seeded(5);
    let base = semantic_weights_digest(&model).unwrap();
    let nudged = model.with_core_weight_scalar(model.core_weight_scalar() + 1e-6);
    assert_ne!(base, semantic_weights_digest(&nudged).unwrap());
    let cmp = compare_weights(&model, &nudged).unwrap();
    assert_eq!(cmp.elements_differing, 1);
    assert_eq!(cmp.tensors_differing, 1);
    assert_ne!(base, semantic_weights_digest(&seeded(6)).unwrap());
}

#[test]
fn shape_and_presence_enter_the_digest() {
    let flat = |dims: Vec<usize>| NamedParam {
        name: "t".into(),
        dims,
        values: Some(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    };
    let a = digest_named_params("{}", &[flat(vec![2, 3])]);
    let b = digest_named_params("{}", &[flat(vec![3, 2])]);
    let c = digest_named_params("{}", &[flat(vec![6])]);
    assert_ne!(a.digest, b.digest);
    assert_ne!(a.digest, c.digest);
    let absent = NamedParam {
        name: "t".into(),
        dims: vec![],
        values: None,
    };
    let empty = NamedParam {
        name: "t".into(),
        dims: vec![0],
        values: Some(vec![]),
    };
    assert_ne!(
        digest_named_params("{}", &[absent]).digest,
        digest_named_params("{}", &[empty]).digest
    );
    // The model config is part of the identity.
    assert_ne!(
        digest_named_params("{}", &[flat(vec![6])]).digest,
        digest_named_params("{\"w\":1}", &[flat(vec![6])]).digest
    );
}
