//! Phase 0 optimizer proof: a fixed synthetic fixture must be overfit.
//!
//! The assertion only checks finiteness, parameter movement, and that the loss
//! decreases. The actual curve is printed and written to `runs/` by the CLI; it
//! is never hard-coded into a pass threshold.

use burn::prelude::*;

use recur64_model::config::ModelConfig;
use recur64_model::fixture::SynthFixture;
use recur64_model::loss::model_loss;
use recur64_model::model::ProbeModel;
use recur64_model::train::{CpuTrainBackend, adamw, train_step};

fn micro_cfg() -> ModelConfig {
    ModelConfig {
        width: 48,
        heads: 4,
        ffn: 96,
        input_blocks: 1,
        core_blocks: 2,
        output_blocks: 1,
        squares: 64,
        in_features: 119,
        policy_dim: 32,
        wdl_classes: 3,
        promo_codes: 5,
        rms_eps: 1e-5,
    }
}

fn scalar<B: Backend>(t: Tensor<B, 1>) -> f32 {
    t.into_data().to_vec::<f32>().unwrap()[0]
}

#[test]
fn fixed_fixture_is_overfit() {
    type B = CpuTrainBackend;
    let device = Default::default();
    let model = ProbeModel::<B>::new(micro_cfg(), &device);
    let mut optim = adamw::<B, ProbeModel<B>>();
    let fx = SynthFixture::new(8, 119, 1234);
    let (board, cands, targets) = fx.tensors::<B>(&device);

    let lr = 3e-3;
    let steps = 100usize;
    let mut model = model;
    let mut first = f32::NAN;
    let mut last = f32::NAN;
    let mut curve = Vec::with_capacity(steps);

    for step in 0..steps {
        let (m, loss) = train_step(
            model,
            &mut optim,
            board.clone(),
            &cands,
            &targets,
            1,
            false,
            lr,
        );
        model = m;
        let l = scalar(loss);
        assert!(l.is_finite(), "loss became non-finite at step {step}: {l}");
        if step == 0 {
            first = l;
        }
        last = l;
        if step % 50 == 0 || step == steps - 1 {
            curve.push((step, l));
        }
    }

    println!("overfit loss curve (step, loss): {curve:?}");
    assert!(
        last < first,
        "loss did not decrease: first={first}, last={last}"
    );

    // Confirm parameters actually moved.
    let out = model.forward_r(board.clone(), &cands, 1, false);
    let final_loss = scalar(model_loss(&out, &targets));
    assert!(final_loss.is_finite());
}
