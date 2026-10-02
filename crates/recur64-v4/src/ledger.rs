//! The explicit evidence ledger (`evidence_ledger_set_v1`).
//!
//! A ledger is a SET of [`EvidenceMessage`]s per example. There is no absolute query-number
//! embedding and no positional encoding: a slot index exists only as storage. Structural
//! metadata (which root branch a message belongs to, its depth and parity) is kept as routing
//! data next to the message and is used only to weight how a message reaches each hypothesis.

use burn::prelude::*;
use burn::tensor::{Int, TensorData};

/// Number of routing classes: `same_branch (2) x depth bucket (8) x parity (2)`.
pub const ROUTE_CLASSES: usize = 32;
/// Depths above this share a bucket.
pub const MAX_DEPTH_BUCKET: u32 = 8;

/// An inspectable record of one acquired message (host side).
#[derive(Debug, Clone, PartialEq)]
pub struct EvidenceMessage {
    pub example: usize,
    /// Query step that produced it (diagnostic only; never an input to the model).
    pub step: usize,
    /// Root branch (index into the root legal list) the queried edge descends from.
    pub branch: usize,
    /// Ply depth of the returned child.
    pub depth: u32,
    /// L2 norm of the message vector (filled when diagnostics are requested).
    pub norm: Option<f32>,
    /// Content-derived trust in `[0, 1)` (filled when diagnostics are requested).
    pub trust: Option<f32>,
}

/// Route class of a message for hypothesis `hyp`.
pub fn route_class(branch: usize, depth: u32, hyp: usize) -> usize {
    let same = usize::from(branch == hyp);
    let bucket = (depth.clamp(1, MAX_DEPTH_BUCKET) - 1) as usize;
    same * 16 + bucket * 2 + (depth % 2) as usize
}

pub struct EvidenceLedger<B: Backend> {
    batch: usize,
    dim: usize,
    /// `[b, J, M]`, `None` while empty.
    pub(crate) messages: Option<Tensor<B, 3>>,
    /// `[b, J]`, 1.0 where a slot holds a message.
    pub(crate) present: Option<Tensor<B, 2>>,
    /// Per example, per slot: root branch (`usize::MAX` where absent).
    branch: Vec<Vec<usize>>,
    depth: Vec<Vec<u32>>,
}

impl<B: Backend> Clone for EvidenceLedger<B> {
    fn clone(&self) -> Self {
        Self {
            batch: self.batch,
            dim: self.dim,
            messages: self.messages.clone(),
            present: self.present.clone(),
            branch: self.branch.clone(),
            depth: self.depth.clone(),
        }
    }
}

impl<B: Backend> EvidenceLedger<B> {
    pub fn new(batch: usize, dim: usize) -> Self {
        Self {
            batch,
            dim,
            messages: None,
            present: None,
            branch: vec![Vec::new(); batch],
            depth: vec![Vec::new(); batch],
        }
    }

    pub fn batch(&self) -> usize {
        self.batch
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of storage slots (queries so far; storage only, not an order signal).
    pub fn slots(&self) -> usize {
        self.branch.first().map_or(0, Vec::len)
    }

    pub fn is_empty(&self) -> bool {
        self.messages.is_none()
    }

    /// Messages held by example `e`.
    pub fn count(&self, e: usize) -> usize {
        self.branch[e].iter().filter(|&&b| b != usize::MAX).count()
    }

    pub fn branch_of(&self, e: usize, slot: usize) -> Option<usize> {
        let b = self.branch[e][slot];
        (b != usize::MAX).then_some(b)
    }

    pub fn depth_of(&self, e: usize, slot: usize) -> u32 {
        self.depth[e][slot]
    }

    /// Append one new slot. `rows[i]` is the example that received `msgs[i]`; every other
    /// example gets an empty slot. `branch` / `depth` are routing metadata per row.
    pub fn append(
        &mut self,
        device: &B::Device,
        rows: &[usize],
        msgs: Tensor<B, 2>,
        branch: &[usize],
        depth: &[u32],
    ) -> anyhow::Result<()> {
        let n = rows.len();
        anyhow::ensure!(
            n > 0 && msgs.dims() == [n, self.dim] && branch.len() == n && depth.len() == n,
            "ledger append: {} rows, message shape {:?}, expected [{n}, {}]",
            n,
            msgs.dims(),
            self.dim
        );
        let b = self.batch;
        let mut map = vec![n as i32; b];
        let mut present = vec![0.0f32; b];
        let mut br = vec![usize::MAX; b];
        let mut dp = vec![0u32; b];
        for (i, &e) in rows.iter().enumerate() {
            anyhow::ensure!(e < b && map[e] == n as i32, "ledger append: bad or repeated row {e}");
            map[e] = i as i32;
            present[e] = 1.0;
            br[e] = branch[i];
            dp[e] = depth[i];
        }
        let map_t = Tensor::<B, 1, Int>::from_data(TensorData::new(map, [b]), device);
        let col = Tensor::cat(vec![msgs, Tensor::<B, 2>::zeros([1, self.dim], device)], 0)
            .select(0, map_t)
            .unsqueeze_dim::<3>(1);
        let pcol = Tensor::<B, 2>::from_data(TensorData::new(present, [b, 1]), device);
        self.messages = Some(match self.messages.take() {
            Some(m) => Tensor::cat(vec![m, col], 1),
            None => col,
        });
        self.present = Some(match self.present.take() {
            Some(p) => Tensor::cat(vec![p, pcol], 1),
            None => pcol,
        });
        for e in 0..b {
            self.branch[e].push(br[e]);
            self.depth[e].push(dp[e]);
        }
        Ok(())
    }

    /// `[b * w * J]` route class per (example, hypothesis, slot), row-major.
    pub fn route_classes(&self, w: usize, device: &B::Device) -> Tensor<B, 1, Int> {
        let j = self.slots();
        let mut v = Vec::with_capacity(self.batch * w * j);
        for e in 0..self.batch {
            for hyp in 0..w {
                for s in 0..j {
                    let c = match self.branch_of(e, s) {
                        Some(br) => route_class(br, self.depth[e][s], hyp),
                        None => 0,
                    };
                    v.push(c as i32);
                }
            }
        }
        let n = v.len();
        Tensor::from_data(TensorData::new(v, [n]), device)
    }

    /// Mean message of each example (zeros while empty): `[b, M]`.
    pub fn mean_message(&self, device: &B::Device) -> Tensor<B, 2> {
        match (&self.messages, &self.present) {
            (Some(m), Some(p)) => {
                let [b, j, d] = m.dims();
                let w = p.clone().unsqueeze_dim::<3>(2).expand([b, j, d]);
                let sum = (m.clone() * w).sum_dim(1).squeeze_dim::<2>(1);
                let cnt = p.clone().sum_dim(1).clamp_min(1.0);
                sum / cnt.expand([b, d])
            }
            _ => Tensor::zeros([self.batch, self.dim], device),
        }
    }

    /// The same ledger with its slots reordered by `perm` (a permutation of `0..J`). Used to
    /// prove permutation invariance: slot order is storage only.
    pub fn permuted(&self, perm: &[usize], device: &B::Device) -> anyhow::Result<Self> {
        let j = self.slots();
        let mut seen = vec![false; j];
        anyhow::ensure!(perm.len() == j, "permutation has the wrong length");
        for &p in perm {
            anyhow::ensure!(p < j && !seen[p], "not a permutation");
            seen[p] = true;
        }
        let mut out = self.clone();
        if let (Some(m), Some(p)) = (&self.messages, &self.present) {
            let idx = Tensor::<B, 1, Int>::from_data(
                TensorData::new(perm.iter().map(|&x| x as i32).collect::<Vec<_>>(), [j]),
                device,
            );
            out.messages = Some(m.clone().select(1, idx.clone()));
            out.present = Some(p.clone().select(1, idx));
            for e in 0..self.batch {
                out.branch[e] = perm.iter().map(|&x| self.branch[e][x]).collect();
                out.depth[e] = perm.iter().map(|&x| self.depth[e][x]).collect();
            }
        }
        Ok(out)
    }

    /// A `k`-example ledger made from `k` copies of example `e` plus one extra slot holding the
    /// `k` hypothetical messages `msgs: [k, M]`. The receiver is untouched (used by
    /// counterfactual probes, which never mutate the real ledger).
    pub fn fork_with(
        &self,
        device: &B::Device,
        e: usize,
        msgs: Tensor<B, 2>,
        branch: &[usize],
        depth: &[u32],
    ) -> anyhow::Result<Self> {
        let k = msgs.dims()[0];
        anyhow::ensure!(
            e < self.batch && msgs.dims()[1] == self.dim && branch.len() == k && depth.len() == k,
            "fork_with: bad shapes"
        );
        let mut out = Self::new(k, self.dim);
        let j = self.slots();
        if let (Some(m), Some(p)) = (&self.messages, &self.present) {
            let idx = Tensor::<B, 1, Int>::from_data(TensorData::new(vec![e as i32; k], [k]), device);
            out.messages = Some(m.clone().select(0, idx.clone()));
            out.present = Some(p.clone().select(0, idx));
            for i in 0..k {
                out.branch[i] = self.branch[e].clone();
                out.depth[i] = self.depth[e].clone();
            }
        }
        debug_assert_eq!(out.slots(), j);
        let rows: Vec<usize> = (0..k).collect();
        out.append(device, &rows, msgs, branch, depth)?;
        Ok(out)
    }

    /// Host copy of the stored message tensor `[b][J][M]` (tests and diagnostics).
    pub fn host_messages(&self) -> anyhow::Result<Option<Vec<f32>>> {
        match &self.messages {
            None => Ok(None),
            Some(m) => Ok(Some(
                m.clone()
                    .into_data()
                    .to_vec::<f32>()
                    .map_err(|e| anyhow::anyhow!("reading ledger to host: {e:?}"))?,
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn route_classes_stay_inside_the_table() {
        for branch in 0..3 {
            for depth in 1..20u32 {
                for hyp in 0..3 {
                    assert!(route_class(branch, depth, hyp) < ROUTE_CLASSES);
                }
            }
        }
    }

    #[test]
    fn route_class_depends_on_branch_membership_not_on_message_order() {
        assert_ne!(route_class(2, 3, 2), route_class(2, 3, 1));
        assert_eq!(route_class(2, 3, 1), route_class(5, 3, 1));
    }
}
