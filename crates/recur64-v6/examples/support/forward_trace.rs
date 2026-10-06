//! Read-only mirror of frozen stream; runtime production parity is mandatory.
use burn::nn::Linear;
use burn::prelude::*;
use burn::tensor::{TensorData, activation};
use recur64_v6::model::{Inputs, Mlp, Reader};
const D: usize = 256;
pub fn rows<B: Backend>(m: &Linear<B>, x: Tensor<B, 3>) -> Tensor<B, 3> {
    let [b, n, d] = x.dims();
    let y = m.forward(x.reshape([b * n, d]));
    let o = y.dims()[1];
    y.reshape([b, n, o])
}
fn mlp<B: Backend>(m: &Mlp<B>, x: Tensor<B, 3>) -> Tensor<B, 3> {
    let [b, n, d] = x.dims();
    m.second
        .forward(activation::gelu(m.first.forward(x.reshape([b * n, d]))))
        .reshape([b, n, D])
}
fn read<B: Backend>(m: &Reader<B>, h: Tensor<B, 3>) -> Tensor<B, 2> {
    rows(&m.correction_out, activation::gelu(rows(&m.head_hidden, h))).squeeze_dim::<2>(2)
}
fn softmax<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    let m = x.clone().detach().max_dim(2);
    let e = (x - m).exp();
    e.clone() / e.sum_dim(2)
}
pub fn stream<B: Backend>(
    m: &Reader<B>,
    i: &Inputs<B>,
    encoded: Tensor<B, 3>,
    r: usize,
    live: bool,
    turn_blind: bool,
    phase: &mut dyn FnMut(&str),
) -> Vec<Tensor<B, 2>> {
    let [b, q, _] = encoded.dims();
    let w = i.width;
    let n = q + w;
    let device = encoded.device();
    let owner = rows(&m.owner, i.baseline_owner.clone());
    let owned = owner
        .clone()
        .gather(1, i.owners.clone().reshape([b, q, 1]).expand([b, q, D]));
    let s = rows(&m.structure, i.node_structure.clone());
    let sr = rows(&m.structure, i.root_structure.clone());
    let x = if live {
        encoded
    } else {
        Tensor::zeros([b, q, D], &device)
    };
    let nodes = x + s.clone() + owned;
    let unknown = m.unknown.val().reshape([1, 2, D]).expand([b, 2, D]);
    let memory = Tensor::cat(vec![nodes.clone(), unknown.clone()], 1);
    let scores = rows(&m.score, memory.clone())
        .swap_dims(1, 2)
        .expand([b, w, q + 2]);
    let a = softmax(scores.clone() + i.pool_attacker.clone()).matmul(memory.clone());
    let defender_scores = if turn_blind { scores } else { scores.neg() };
    let d = softmax(defender_scores + i.pool_defender.clone()).matmul(memory);
    let init = owner.clone() + mlp(&m.initialize, Tensor::cat(vec![a, d, owner, sr.clone()], 2));
    phase("full_branch_initialize");
    if m.backup.is_empty() {
        return vec![read(m, init)];
    }
    let ni = init
        .clone()
        .gather(1, i.owners.clone().reshape([b, q, 1]).expand([b, q, D]));
    let mut h = Tensor::cat(vec![nodes + ni.clone(), init.clone()], 1);
    let anchor = Tensor::cat(vec![ni, init], 1);
    let structure = Tensor::cat(vec![s, sr], 1);
    let stop = if live {
        i.terminals.clone()
    } else {
        Tensor::from_data(TensorData::new(vec![false; b * n], [b, n]), &device)
    };
    let mut result = Vec::new();
    for _ in 0..r {
        let memory = Tensor::cat(vec![h.clone(), unknown.clone()], 1);
        let scores = rows(&m.score, memory.clone())
            .swap_dims(1, 2)
            .expand([b, n, n + 2]);
        let scores = if turn_blind {
            scores
        } else {
            scores * i.local_sign.clone()
        };
        let allow = if turn_blind {
            i.local_allow_turn_blind.clone()
        } else {
            i.local_allow.clone()
        };
        let pool = softmax(scores + allow).matmul(memory);
        let u = mlp(
            &m.backup[0],
            Tensor::cat(vec![h.clone(), pool, anchor.clone(), structure.clone()], 2),
        )
        .mul_scalar(0.1);
        h = h + u.mask_fill(stop.clone().reshape([b, n, 1]).expand([b, n, D]), 0.);
        result.push(read(m, h.clone().slice([0..b, q..n, 0..D])));
        phase("shared_local_refinement");
    }
    result
}
