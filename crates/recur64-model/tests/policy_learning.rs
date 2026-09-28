//! Diagnostic (2026-09-28): can the policy head learn at all? Across H3.6, P1
//! and V1 the policy training loss stayed flat (e.g. 3.105 -> 3.105 over 400
//! updates) while the WDL loss fell. The overfit test only checks the total
//! loss, which the WDL head alone can satisfy.
//!
//! This trains a tiny model on a fixed fixture with one-hot policy targets and
//! checks the policy component of the loss separately.

use burn::prelude::*;

use recur64_model::config::ModelConfig;
use recur64_model::fixture::SynthFixture;
use recur64_model::model::ProbeModel;
use recur64_model::train::{CpuTrainBackend, adamw, train_step_reporting};

type B = CpuTrainBackend;

fn cfg() -> ModelConfig {
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
fn policy_loss_itself_decreases_on_a_fixed_fixture() {
    let device = Default::default();
    <B as Backend>::seed(&device, 4);
    let mut model = ProbeModel::<B>::new(cfg(), &device);
    let mut optim = adamw::<B, _>();
    let fx = SynthFixture::new(8, 119, 11);
    let (board, cands, targets) = fx.tensors::<B>(&device);
    let mut curve = Vec::new();
    for _ in 0..150 {
        let (m, report) = train_step_reporting(
            model,
            &mut optim,
            board.clone(),
            &cands,
            &targets,
            1,
            false,
            3e-3,
        );
        model = m;
        curve.push((report.policy_loss, report.wdl_loss, report.grad_norm));
    }
    let (p0, w0, _) = curve[0];
    let (p1, w1, _) = *curve.last().unwrap();
    println!("policy {p0:.4} -> {p1:.4} | wdl {w0:.4} -> {w1:.4}");
    println!("first 5: {:?}", &curve[..5]);
    println!("last 5: {:?}", &curve[curve.len() - 5..]);
    assert!(
        p1 < 0.5 * p0,
        "policy loss must fall on a fixed one-hot fixture: {p0} -> {p1}"
    );
}
