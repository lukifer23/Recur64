//! Every normal model load refuses a checkpoint whose model configuration
//! differs from the requested one, even when the tensors are shape-compatible.

use burn::backend::{Autodiff, Flex};

use recur64_model::checkpoint::{CheckpointMeta, load_training, save_training};
use recur64_model::config::ModelConfig;
use recur64_model::model::ProbeModel;
use recur64_model::train::adamw;
use recur64_runtime::model_io;

type TB = Autodiff<Flex>;

fn micro() -> ModelConfig {
    ModelConfig {
        width: 32,
        heads: 4,
        ffn: 64,
        input_blocks: 0,
        core_blocks: 1,
        output_blocks: 0,
        squares: 64,
        in_features: 119,
        policy_dim: 16,
        wdl_classes: 3,
        promo_codes: 5,
        rms_eps: 1e-5,
    }
}

#[test]
fn same_shape_model_config_mismatch_is_refused_on_every_load_path() {
    let dir = std::env::temp_dir().join(format!("recur64-model-io-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let device = Default::default();
    let cfg = micro();
    let model = ProbeModel::<TB>::new(cfg.clone(), &device);
    let meta = CheckpointMeta::new(cfg.clone(), 1, false, 0, 3e-4, 1, 0, "cpu", "fp32");
    save_training(&dir, &model, &adamw::<TB, _>(), &meta).expect("save");

    // Matching config loads; recurrence is a runtime choice and is not checked.
    model_io::load::<Flex>(&dir, &cfg, &device).expect("matching config loads");

    // Same tensor shapes, different function: rms_eps.
    let other = ModelConfig {
        rms_eps: 1e-6,
        ..cfg.clone()
    };
    let err = model_io::load::<Flex>(&dir, &other, &device)
        .map(|_| ())
        .expect_err("model_io::load must refuse")
        .to_string();
    assert!(err.contains("model config"), "{err}");
    assert!(
        load_training(
            &dir,
            ProbeModel::<TB>::new(other, &device),
            adamw::<TB, _>(),
            &device
        )
        .is_err(),
        "load_training must refuse"
    );
    std::fs::remove_dir_all(&dir).ok();
}
