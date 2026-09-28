//! Truncated-game policy training: the WDL mask must leave unmasked batches
//! exactly as before and remove masked rows from the WDL term entirely.

use burn::prelude::*;
use burn::tensor::TensorData;

use recur64_model::loss::{wdl_ce, wdl_ce_masked};
use recur64_model::train::CpuTrainBackend;

type B = CpuTrainBackend;

fn logits<Bk: Backend>(dev: &Bk::Device) -> Tensor<Bk, 2> {
    Tensor::from_data(
        TensorData::new(
            vec![
                2.0f32, 0.5, -1.0, -0.3, 0.1, 1.7, 0.0, 0.0, 0.0, 1.2, -2.0, 0.4,
            ],
            [4, 3],
        ),
        dev,
    )
}

fn targets<Bk: Backend>(dev: &Bk::Device) -> Tensor<Bk, 1, Int> {
    Tensor::from_data(TensorData::new(vec![0i64, 2, 1, 1], [4]), dev)
}

fn mask<Bk: Backend>(v: Vec<f32>, dev: &Bk::Device) -> Tensor<Bk, 1> {
    Tensor::from_data(TensorData::new(v, [4]), dev)
}

fn scalar(t: Tensor<B, 1>) -> f32 {
    t.into_data().to_vec::<f32>().unwrap()[0]
}

#[test]
fn no_mask_is_bit_identical_to_the_unmasked_loss() {
    let dev = Default::default();
    let a = scalar(wdl_ce(&logits::<B>(&dev), &targets::<B>(&dev)));
    let b = scalar(wdl_ce_masked(&logits::<B>(&dev), &targets::<B>(&dev), None));
    assert_eq!(a.to_bits(), b.to_bits());
}

#[test]
fn all_ones_mask_equals_the_mean() {
    let dev = Default::default();
    let a = scalar(wdl_ce(&logits::<B>(&dev), &targets::<B>(&dev)));
    let m = mask::<B>(vec![1.0; 4], &dev);
    let b = scalar(wdl_ce_masked(
        &logits::<B>(&dev),
        &targets::<B>(&dev),
        Some(&m),
    ));
    assert!((a - b).abs() < 1e-6, "{a} vs {b}");
}

#[test]
fn masked_rows_do_not_contribute_value_or_gradient() {
    let dev = Default::default();
    // Rows 0 and 2 supervised; rows 1 and 3 masked.
    let m = mask::<B>(vec![1.0, 0.0, 1.0, 0.0], &dev);
    let full = logits::<B>(&dev).require_grad();
    let loss = wdl_ce_masked(&full, &targets::<B>(&dev), Some(&m));
    let value = scalar(loss.clone());

    // Reference: the plain mean over the supervised rows alone.
    let rows = Tensor::<B, 1, Int>::from_data(TensorData::new(vec![0i64, 2], [2]), &dev);
    let sub = logits::<B>(&dev).select(0, rows.clone());
    let sub_t = targets::<B>(&dev).select(0, rows);
    let expect = scalar(wdl_ce(&sub, &sub_t));
    assert!((value - expect).abs() < 1e-6, "{value} vs {expect}");

    let grads = loss.backward();
    let g = full
        .grad(&grads)
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    for (row, supervised) in [(0, true), (1, false), (2, true), (3, false)] {
        let norm: f32 = g[row * 3..row * 3 + 3].iter().map(|x| x.abs()).sum();
        if supervised {
            assert!(norm > 0.0, "supervised row {row} has no gradient");
        } else {
            assert_eq!(norm, 0.0, "masked row {row} has gradient {norm}");
        }
    }
}

#[test]
fn an_all_masked_batch_has_zero_wdl_loss() {
    let dev = Default::default();
    let m = mask::<B>(vec![0.0; 4], &dev);
    let v = scalar(wdl_ce_masked(
        &logits::<B>(&dev),
        &targets::<B>(&dev),
        Some(&m),
    ));
    assert_eq!(v, 0.0);
}
