//! Full observed-branch initialization followed by optional shared refinement.
use crate::encoder::StateEncoder;
use crate::packet::Packet;
use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::{Bool, Distribution, Int, TensorData, activation};
use serde::{Deserialize, Serialize};
const D: usize = 256;
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Arm {
    SharedBackup,
    OnePass,
}
#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    pub first: Linear<B>,
    pub second: Linear<B>,
}
impl<B: Backend> Mlp<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            first: LinearConfig::new(1024, 768).init(device),
            second: LinearConfig::new(768, D).init(device),
        }
    }
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, n, d] = x.dims();
        self.second
            .forward(activation::gelu(self.first.forward(x.reshape([b * n, d]))))
            .reshape([b, n, D])
    }
}
#[derive(Module, Debug)]
pub struct Reader<B: Backend> {
    pub encoder: StateEncoder<B>,
    pub slots: Linear<B>,
    pub structure: Linear<B>,
    pub owner: Linear<B>,
    pub score: Linear<B>,
    pub unknown: Param<Tensor<B, 2>>,
    pub initialize: Mlp<B>,
    pub head_hidden: Linear<B>,
    pub correction_out: Linear<B>,
    pub backup: Vec<Mlp<B>>,
}
fn rows<B: Backend>(m: &Linear<B>, x: Tensor<B, 3>) -> Tensor<B, 3> {
    let [b, n, d] = x.dims();
    let y = m.forward(x.reshape([b * n, d]));
    let o = y.dims()[1];
    y.reshape([b, n, o])
}
fn softmax<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    let m = x.clone().detach().max_dim(2);
    let e = (x - m).exp();
    e.clone() / e.sum_dim(2)
}
pub struct Inputs<B: Backend> {
    pub states: Tensor<B, 4>,
    pub flags: Tensor<B, 3>,
    pub node_mask: Tensor<B, 2, Bool>,
    pub node_structure: Tensor<B, 3>,
    pub root_structure: Tensor<B, 3>,
    pub owners: Tensor<B, 2, Int>,
    pub baseline_owner: Tensor<B, 3>,
    pub z0: Tensor<B, 2>,
    pub legal: Tensor<B, 2, Bool>,
    pub eligible: Tensor<B, 2, Bool>,
    pub pool_attacker: Tensor<B, 3>,
    pub pool_defender: Tensor<B, 3>,
    pub local_allow: Tensor<B, 3>,
    pub local_sign: Tensor<B, 3>,
    pub local_allow_turn_blind: Tensor<B, 3>,
    pub terminals: Tensor<B, 2, Bool>,
    pub q: usize,
    pub width: usize,
}
pub struct Output<B: Backend> {
    pub logits: Tensor<B, 2>,
    pub raw_delta: Tensor<B, 2>,
    pub centered: Tensor<B, 2>,
    pub iteration_delta: Vec<Tensor<B, 2>>,
}
impl<B: Backend> Reader<B> {
    pub fn new(arm: Arm, device: &B::Device) -> Self {
        let cfg = recur64_v5::config::V5Config::default();
        Self {
            encoder: StateEncoder::new(&cfg, device),
            slots: LinearConfig::new(4 * D, D).init(device),
            structure: LinearConfig::new(48, D).init(device),
            owner: LinearConfig::new(D, D).init(device),
            score: LinearConfig::new(D, 1).with_bias(false).init(device),
            unknown: Param::from_tensor(Tensor::random(
                [2, D],
                Distribution::Normal(0., 0.02),
                device,
            )),
            initialize: Mlp::new(device),
            head_hidden: LinearConfig::new(D, D).init(device),
            correction_out: LinearConfig::new(D, 1).with_bias(false).init(device),
            backup: if arm == Arm::SharedBackup {
                vec![Mlp::new(device)]
            } else {
                Vec::new()
            },
        }
    }
    pub fn arm(&self) -> Arm {
        if self.backup.is_empty() {
            Arm::OnePass
        } else {
            Arm::SharedBackup
        }
    }
    fn read(&self, h: Tensor<B, 3>) -> Tensor<B, 2> {
        rows(
            &self.correction_out,
            activation::gelu(rows(&self.head_hidden, h)),
        )
        .squeeze_dim::<2>(2)
    }
    fn stream(
        &self,
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
        let owner = rows(&self.owner, i.baseline_owner.clone());
        let owned = owner
            .clone()
            .gather(1, i.owners.clone().reshape([b, q, 1]).expand([b, q, D]));
        let s = rows(&self.structure, i.node_structure.clone());
        let sr = rows(&self.structure, i.root_structure.clone());
        let x = if live {
            encoded
        } else {
            Tensor::zeros([b, q, D], &device)
        };
        let nodes = x + s.clone() + owned;
        let unknown = self.unknown.val().reshape([1, 2, D]).expand([b, 2, D]);
        let memory = Tensor::cat(vec![nodes.clone(), unknown.clone()], 1);
        let scores = rows(&self.score, memory.clone())
            .swap_dims(1, 2)
            .expand([b, w, q + 2]);
        let a = softmax(scores.clone() + i.pool_attacker.clone()).matmul(memory.clone());
        let defender_scores = if turn_blind { scores } else { scores.neg() };
        let d = softmax(defender_scores + i.pool_defender.clone()).matmul(memory);
        let init = owner.clone()
            + self
                .initialize
                .forward(Tensor::cat(vec![a, d, owner, sr.clone()], 2));
        phase("full_branch_initialize");
        if self.backup.is_empty() {
            return vec![self.read(init)];
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
            let scores = rows(&self.score, memory.clone())
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
            let u = self.backup[0]
                .forward(Tensor::cat(
                    vec![h.clone(), pool, anchor.clone(), structure.clone()],
                    2,
                ))
                .mul_scalar(0.1);
            h = h + u.mask_fill(stop.clone().reshape([b, n, 1]).expand([b, n, D]), 0.);
            result.push(self.read(h.clone().slice([0..b, q..n, 0..D])));
            phase("shared_local_refinement");
        }
        result
    }
    pub fn forward(
        &self,
        i: &Inputs<B>,
        r: usize,
        all_null: bool,
        turn_blind: bool,
        phase: &mut dyn FnMut(&str),
    ) -> Output<B> {
        assert!(r == 1 || r == 4);
        let [b, q, _, _] = i.states.dims();
        let encoded = self
            .encoder
            .forward(i.states.clone(), i.flags.clone(), i.node_mask.clone())
            .reshape([b, q, 4 * D]);
        let encoded = rows(&self.slots, encoded);
        phase("returned_encoder_and_slots");
        let f = self.stream(i, encoded.clone(), r, !all_null, turn_blind, phase);
        let null = self.stream(i, encoded, r, false, turn_blind, phase);
        let delta: Vec<_> = f
            .into_iter()
            .zip(null)
            .map(|(f, n)| (f - n).mask_fill(i.legal.clone().bool_not(), 0.))
            .collect();
        let raw = delta.last().unwrap().clone();
        let mean = (raw.clone().sum_dim(1) / i.legal.clone().float().sum_dim(1)).expand(raw.dims());
        let centered = (raw.clone() - mean).mask_fill(i.legal.clone().bool_not(), 0.);
        let logits = i.z0.clone() + centered.clone();
        phase("paired_centered_readout");
        Output {
            logits,
            raw_delta: raw,
            centered,
            iteration_delta: delta,
        }
    }
}
fn structural(node: &recur64_v5::graph::AcquiredNode, legal: usize, observed: usize) -> [f32; 48] {
    let mut s = [0.; 48];
    s[node.depth as usize - 1] = 1.;
    s[16 + usize::from(!node.root_to_move)] = 1.;
    s[18..29].copy_from_slice(&node.action_geometry);
    for (i, a) in node.path.iter().enumerate() {
        s[29 + i] = *a as f32 / 65535.;
    }
    s[45] = legal as f32 / 256.;
    s[46] = observed as f32 / 256.;
    s[47] = (legal - observed) as f32 / 256.;
    s
}
pub fn inputs<B: Backend>(
    packets: &[Packet],
    base_owner: Tensor<B, 3>,
    z0: Tensor<B, 2>,
    device: &B::Device,
) -> anyhow::Result<Inputs<B>> {
    anyhow::ensure!(!packets.is_empty(), "empty reader batch");
    let b = packets.len();
    let q = packets.iter().map(|p| p.nodes.len()).max().unwrap().max(1);
    let w = packets.iter().map(|p| p.legal.len()).max().unwrap();
    let n = q + w;
    let mut states = vec![0.; b * q * 64 * 119];
    let mut flags = vec![0.; b * q * 9];
    let mut nm = vec![false; b * q];
    let mut ns = vec![0.; b * q * 48];
    let mut rs = vec![0.; b * w * 48];
    let mut owner = vec![0i32; b * q];
    let mut legal = vec![false; b * w];
    let mut eligible = vec![false; b * w];
    let mut pa = vec![-1e9; b * w * (q + 2)];
    let mut pd = pa.clone();
    let mut local = vec![-1e9; b * n * (n + 2)];
    let mut signs = vec![1.; b * n * (n + 2)];
    let mut terminal = vec![false; b * n];
    for (bi, p) in packets.iter().enumerate() {
        p.verify(&p.source_sha)?;
        for (j, node) in p.nodes.iter().enumerate() {
            states[(bi * q + j) * 64 * 119..(bi * q + j + 1) * 64 * 119]
                .copy_from_slice(&node.payload.observation);
            flags[(bi * q + j) * 9..(bi * q + j + 1) * 9].copy_from_slice(&node.payload.flags);
            nm[bi * q + j] = true;
            owner[bi * q + j] = node.root_candidate as i32;
            ns[(bi * q + j) * 48..(bi * q + j + 1) * 48].copy_from_slice(&structural(
                node,
                p.legal_counts[j],
                p.children(j).len(),
            ));
            terminal[bi * n + j] = node.payload.flags[1] > 0.;
        }
        for a in 0..w {
            legal[bi * w + a] = a < p.legal.len();
            let owned: Vec<_> = p
                .nodes
                .iter()
                .enumerate()
                .filter(|(_, v)| v.root_candidate == a)
                .collect();
            eligible[bi * w + a] = !owned.is_empty();
            rs[(bi * w + a) * 48 + 16] = 1.;
            rs[(bi * w + a) * 48 + 45] = 1. / 256.;
            rs[(bi * w + a) * 48 + 46] = f32::from(!owned.is_empty()) / 256.;
            rs[(bi * w + a) * 48 + 47] = f32::from(owned.is_empty()) / 256.;
            for turn in [true, false] {
                let pool = if turn { &mut pa } else { &mut pd };
                let at = (bi * w + a) * (q + 2);
                let mut unknowns = 0usize;
                let mut count = 0;
                for (j, node) in &owned {
                    if node.root_to_move == turn {
                        pool[at + *j] = 0.;
                        unknowns += p.unknown(*j);
                        count += 1;
                    }
                }
                if unknowns > 0 || count == 0 {
                    pool[at + q + usize::from(!turn)] = (unknowns.max(1) as f32).ln();
                }
            }
        }
        for j in 0..n {
            let (children, unknown, turn) = if j < q && j < p.nodes.len() {
                (p.children(j), p.unknown(j), p.nodes[j].root_to_move)
            } else if j >= q {
                let a = j - q;
                let c: Vec<_> = p
                    .nodes
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| v.root_candidate == a && v.parent.is_none())
                    .map(|(k, _)| k)
                    .collect();
                let u = usize::from(c.is_empty());
                (c, u, true)
            } else {
                (vec![], 1, true)
            };
            let at = (bi * n + j) * (n + 2);
            for c in &children {
                local[at + *c] = 0.;
            }
            if unknown > 0 || children.is_empty() {
                local[at + n + usize::from(!turn)] = (unknown.max(1) as f32).ln();
            }
            if !turn {
                signs[at..at + n + 2].fill(-1.);
            }
        }
    }
    Ok(Inputs {
        states: Tensor::from_data(TensorData::new(states, [b, q, 64, 119]), device),
        flags: Tensor::from_data(TensorData::new(flags, [b, q, 9]), device),
        node_mask: Tensor::from_data(TensorData::new(nm, [b, q]), device),
        node_structure: Tensor::from_data(TensorData::new(ns, [b, q, 48]), device),
        root_structure: Tensor::from_data(TensorData::new(rs, [b, w, 48]), device),
        owners: Tensor::from_data(TensorData::new(owner, [b, q]), device),
        baseline_owner: base_owner,
        z0,
        legal: Tensor::from_data(TensorData::new(legal, [b, w]), device),
        eligible: Tensor::from_data(TensorData::new(eligible, [b, w]), device),
        pool_attacker: Tensor::from_data(TensorData::new(pa, [b, w, q + 2]), device),
        pool_defender: Tensor::from_data(TensorData::new(pd, [b, w, q + 2]), device),
        local_allow_turn_blind: Tensor::from_data(
            TensorData::new(local.clone(), [b, n, n + 2]),
            device,
        ),
        local_allow: Tensor::from_data(TensorData::new(local, [b, n, n + 2]), device),
        local_sign: Tensor::from_data(TensorData::new(signs, [b, n, n + 2]), device),
        terminals: Tensor::from_data(TensorData::new(terminal, [b, n]), device),
        q,
        width: w,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::module::AutodiffModule;
    #[test]
    fn real_geometry_deep_dependency_locality_null_and_shared_initialization() {
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(|| {
                let _seed_guard = crate::SEEDED_TEST_LOCK.lock().unwrap();
                type B = burn::backend::Autodiff<burn::backend::Flex>;
                let device = Default::default();
                let source = "a".repeat(40);
                let p = crate::packet::diagnostic_chain(
                    recur64_core::GameState::from_fen("7k/8/8/8/8/8/3Q4/K1R5 w - - 0 1").unwrap(),
                    &source,
                )
                .unwrap();
                let w = p.legal.len();
                B::seed(&device, 6300);
                let a = Reader::<B>::new(Arm::SharedBackup, &device);
                let ai = crate::qualify::inventory(&a).unwrap();
                B::seed(&device, 6300);
                let b = Reader::<B>::new(Arm::OnePass, &device);
                let bi = crate::qualify::inventory(&b).unwrap();
                assert_eq!(
                    ai.into_iter()
                        .filter(|v| !v.name.starts_with("backup."))
                        .collect::<Vec<_>>(),
                    bi
                );
                let build = |p: &Packet| {
                    inputs(
                        std::slice::from_ref(p),
                        Tensor::<B, 3>::zeros([1, w, 256], &device),
                        Tensor::<B, 2>::zeros([1, w], &device),
                        &device,
                    )
                    .unwrap()
                };
                let mut i = build(&p);
                let original = a.forward(&i, 1, false, false, &mut |_| {});
                let z = original
                    .raw_delta
                    .clone()
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap();
                assert!(
                    z.iter()
                        .enumerate()
                        .all(|(j, v)| p.eligibility()[j] || *v == 0.)
                );
                assert!(
                    a.forward(&i, 4, true, false, &mut |_| {})
                        .raw_delta
                        .into_data()
                        .to_vec::<f32>()
                        .unwrap()
                        .iter()
                        .all(|v| *v == 0.)
                );
                let mut changed = p.clone();
                changed.nodes[4].payload.observation[0] += 0.5;
                changed.digest = changed.content_digest().unwrap();
                let altered = a
                    .forward(&build(&changed), 1, false, false, &mut |_| {})
                    .raw_delta
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap();
                assert_ne!(
                    z[changed.nodes[4].root_candidate].to_bits(),
                    altered[changed.nodes[4].root_candidate].to_bits(),
                    "depth5 payload must reach R1 decision"
                );
                let valid = a.valid();
                let _: usize = valid.num_params(); // Ensure graph-free endpoint module exists.
                i.states = i.states.clone().require_grad();
                let tracked = i.states.clone();
                let out = a.forward(&i, 4, false, false, &mut |_| {});
                let loss = out.raw_delta.clone().square().sum();
                let raw = loss.backward();
                let state_grad = tracked
                    .grad(&raw)
                    .unwrap()
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap();
                assert!(
                    state_grad[4 * 64 * 119..5 * 64 * 119]
                        .iter()
                        .any(|v| *v != 0.)
                );
                let g = burn::optim::GradientsParams::from_grads(raw, &a);
                let all = crate::qualify::gradients(&a, &g).unwrap();
                assert!(all.iter().all(|v| v.finite));

                let root =
                    recur64_core::GameState::from_fen("7k/8/8/8/8/8/3Q4/K1R5 w - - 0 1").unwrap();
                let two = crate::packet::acquire(
                    &root,
                    "isolation",
                    &vec![0.; root.legal_actions().len()],
                    crate::packet::Policy::ExploitTwo,
                    &source,
                )
                .unwrap();
                let pre = a
                    .forward(&build(&two), 4, false, false, &mut |_| {})
                    .raw_delta
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap();
                let mut changed = two.clone();
                let other = changed.nodes[1].root_candidate;
                let keep = changed.nodes[0].root_candidate;
                assert_ne!(other, keep);
                for n in &mut changed.nodes {
                    if n.root_candidate == other {
                        n.payload.observation[0] += 0.5;
                    }
                }
                changed.digest = changed.content_digest().unwrap();
                let post = a
                    .forward(&build(&changed), 4, false, false, &mut |_| {})
                    .raw_delta
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap();
                assert_eq!(
                    pre[keep].to_bits(),
                    post[keep].to_bits(),
                    "hard branch isolation before centering"
                );
                // Policy targets do not occur in packets or model inputs; mutation affects loss only.
                let before = p.content_digest().unwrap();
                let c1 = Tensor::<B, 2, Bool>::from_data(
                    TensorData::new((0..w).map(|j| j == 0).collect::<Vec<_>>(), [1, w]),
                    &device,
                );
                let c2 = c1.clone().bool_not();
                let _ = crate::loss::components(&out, &i, c1);
                let _ = crate::loss::components(&out, &i, c2);
                assert_eq!(before, p.content_digest().unwrap());
                assert_eq!(
                    z,
                    a.forward(&build(&p), 1, false, false, &mut |_| {})
                        .raw_delta
                        .into_data()
                        .to_vec::<f32>()
                        .unwrap()
                );
            })
            .unwrap()
            .join()
            .unwrap();
    }
}
