//! Policy over all actions; auxiliary only over observed branches and classes.
use crate::model::{Inputs, Output};
use burn::prelude::*;
use burn::tensor::Bool;
fn softplus<B: Backend>(x: Tensor<B, 2>) -> Tensor<B, 2> {
    let m = x.clone().detach().clamp(0., f32::MAX);
    m.clone() + ((x - m.clone()).exp() + m.neg().exp()).log()
}
pub fn eligible_bce<B: Backend>(
    raw: Tensor<B, 2>,
    eligible: Tensor<B, 2, Bool>,
    correct: Tensor<B, 2, Bool>,
) -> Tensor<B, 1> {
    let pos = eligible.clone().bool_and(correct.clone()).float();
    let neg = eligible.bool_and(correct.bool_not()).float();
    let positive = softplus(raw.clone().neg()) * pos.clone();
    let negative = softplus(raw) * neg.clone();
    let np = pos.sum_dim(1);
    let nn = neg.sum_dim(1);
    let ap = np.clone().greater_elem(0.).float();
    let an = nn.clone().greater_elem(0.).float();
    ((positive.sum_dim(1) / np.clamp(1., f32::MAX) + negative.sum_dim(1) / nn.clamp(1., f32::MAX))
        / (ap + an).clamp(1., f32::MAX))
    .mean()
}
pub fn components<B: Backend>(
    out: &Output<B>,
    i: &Inputs<B>,
    correct: Tensor<B, 2, Bool>,
) -> (Tensor<B, 1>, Tensor<B, 1>) {
    let policy =
        recur64_v5::loss::correct_set_loss(out.logits.clone(), i.legal.clone(), correct.clone());
    let aux = out
        .iteration_delta
        .iter()
        .map(|x| eligible_bce(x.clone(), i.eligible.clone(), correct.clone()))
        .reduce(|a, b| a + b)
        .unwrap()
        .div_scalar(out.iteration_delta.len() as f32);
    (policy, aux)
}
#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;
    #[test]
    fn eligible_class_reference_and_empty_support() {
        type B = burn::backend::Flex;
        let d = Default::default();
        let t = |v: Vec<bool>| Tensor::<B, 2, Bool>::from_data(TensorData::new(v, [1, 3]), &d);
        let raw = Tensor::<B, 2>::from_data([[0., 2., -3.]], &d);
        let v = eligible_bce(
            raw.clone(),
            t(vec![true, false, false]),
            t(vec![true, true, false]),
        )
        .into_scalar();
        assert!((v - 2f32.ln()).abs() < 1e-6);
        assert_eq!(
            eligible_bce(
                raw,
                t(vec![false, false, false]),
                t(vec![true, true, false])
            )
            .into_scalar(),
            0.
        );
    }
}

#[cfg(test)]
mod gradient_tests {
    use super::*;
    use burn::tensor::TensorData;
    #[test]
    fn auxiliary_gradient_reference_at_zero_and_class_balance() {
        type B = burn::backend::Autodiff<burn::backend::Flex>;
        let d = Default::default();
        let x = Tensor::<B, 2>::from_data([[0., 0., 4.]], &d).require_grad();
        let eligible = Tensor::from_data([[true, true, false]], &d);
        let correct = Tensor::from_data([[true, false, true]], &d);
        let loss = eligible_bce(x.clone(), eligible, correct);
        let g = x
            .grad(&loss.backward())
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert!((g[0] + 0.25).abs() < 1e-6);
        assert!((g[1] - 0.25).abs() < 1e-6);
        assert_eq!(g[2], 0.);
        let e = Tensor::<B, 2, Bool>::from_data(TensorData::new(vec![true; 3], [1, 3]), &d);
        let c = Tensor::<B, 2, Bool>::from_data([[true, true, false]], &d);
        let v = eligible_bce(Tensor::from_data([[1., -2., 3.]], &d), e, c).into_scalar();
        let expected = (((1f64 + (-1f64).exp()).ln() + (1f64 + 2f64.exp()).ln()) / 2.
            + (1f64 + 3f64.exp()).ln())
            / 2.;
        assert!((v as f64 - expected).abs() < 1e-5);
    }
}
