use burn::prelude::*;
use burn::tensor::{Bool, TensorData};
use recur64_v5::loss::{correct_set_loss, reference_set_loss};

type B = burn::backend::Autodiff<burn::backend::Flex>;

#[test]
fn correct_set_loss_and_gradients_match_a_f64_reference() {
    let device = Default::default();
    let values = vec![0.7_f32, -0.3, 1.2, -2.0];
    let legal = vec![true, true, true, false];
    let correct = vec![true, false, true, false];
    let logits =
        Tensor::<B, 2>::from_data(TensorData::new(values.clone(), [1, 4]), &device).require_grad();
    let legal_t = Tensor::<B, 2, Bool>::from_data(TensorData::new(legal.clone(), [1, 4]), &device);
    let correct_t =
        Tensor::<B, 2, Bool>::from_data(TensorData::new(correct.clone(), [1, 4]), &device);
    let loss = correct_set_loss(logits.clone(), legal_t, correct_t);
    let got = loss.clone().into_data().to_vec::<f32>().unwrap()[0] as f64;
    let reference = reference_set_loss(&values, &legal, &correct);
    assert!((got - reference).abs() < 1.0e-6, "{got} != {reference}");

    let grads = loss.backward();
    let analytic = logits
        .grad(&grads)
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let epsilon = 1.0e-3_f32;
    for index in 0..values.len() {
        let mut plus = values.clone();
        let mut minus = values.clone();
        plus[index] += epsilon;
        minus[index] -= epsilon;
        let numeric = (reference_set_loss(&plus, &legal, &correct)
            - reference_set_loss(&minus, &legal, &correct))
            / (2.0 * epsilon as f64);
        assert!(
            (analytic[index] as f64 - numeric).abs() < 2.0e-4,
            "gradient {index}: {} != {numeric}",
            analytic[index]
        );
    }
}
