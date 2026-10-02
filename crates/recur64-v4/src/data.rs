//! V4 TRAIN data: P25_DATA_V1 TRAIN split into `V4_TRAIN_FIT` and `V4_TRAIN_DEV` by the frozen
//! rule in `docs/V4_RESEARCH_PLAN.md` section 6. `V4_TRAIN_DEV` is a held-out PART OF TRAIN used
//! for mechanism studies; it is not `v4_tune_v1` and carries no gating authority.
//!
//! Only the frozen P25 TRAIN artifact is accepted. Every other split (V3_TUNE_V1,
//! `v4_tune_v1`, any holdout) is refused by `load_working_split` plus the digest check, so no
//! V4 training or mechanism path can touch a sealed or historical set by accident.

use std::path::Path;

use recur64_core::GameState;
use recur64_runtime::p5::recipe::{TRAIN_DIGEST, TRAIN_POSITIONS};
use recur64_runtime::proof::custody::load_working_split;
use recur64_runtime::proof::sampler::CellSampler;
use recur64_runtime::proof::targets::{ProofPosition, ProofTargets, Split};
use sha2::{Digest, Sha256};

/// Identity of the frozen partition rule.
pub const TRAIN_DEV_RULE: &str = "v4_train_dev_v1";

/// `SHA-256("v4_train_dev_v1|" + canon)`, first 8 bytes big-endian, `mod 10 == 0` -> DEV.
pub fn is_dev(canon: &str) -> bool {
    let mut h = Sha256::new();
    h.update(TRAIN_DEV_RULE.as_bytes());
    h.update(b"|");
    h.update(canon.as_bytes());
    let d = h.finalize();
    let mut b = [0u8; 8];
    b.copy_from_slice(&d[..8]);
    u64::from_be_bytes(b) % 10 == 0
}

fn id_digest(positions: &[ProofPosition], idx: &[usize]) -> String {
    let mut ids: Vec<&str> = idx.iter().map(|&i| positions[i].id.as_str()).collect();
    ids.sort_unstable();
    let mut h = Sha256::new();
    for id in ids {
        h.update(id.as_bytes());
        h.update(b"\n");
    }
    format!("{:x}", h.finalize())
}

pub struct V4Data {
    pub targets: ProofTargets,
    /// Indices into `targets.positions`.
    pub fit: Vec<usize>,
    pub dev: Vec<usize>,
    pub fit_digest: String,
    pub dev_digest: String,
}

impl V4Data {
    /// Load and verify P25 TRAIN, then partition it. Refuses anything but the frozen artifact.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let targets = load_working_split(path, &[Split::Train])?;
        anyhow::ensure!(
            targets.digest == TRAIN_DIGEST && targets.positions.len() == TRAIN_POSITIONS,
            "{}: not the frozen P25 TRAIN (digest {}, {} positions); V4 trains on P25 TRAIN only",
            path.display(),
            targets.digest,
            targets.positions.len()
        );
        Self::from_targets(targets)
    }

    /// Partition already-verified targets (tests build small synthetic sets this way).
    pub fn from_targets(targets: ProofTargets) -> anyhow::Result<Self> {
        let (mut fit, mut dev) = (Vec::new(), Vec::new());
        for (i, p) in targets.positions.iter().enumerate() {
            if is_dev(&p.canon) {
                dev.push(i);
            } else {
                fit.push(i);
            }
        }
        // Split by canonical class: no class may straddle both parts.
        let dev_canons: std::collections::HashSet<&str> = dev
            .iter()
            .map(|&i| targets.positions[i].canon.as_str())
            .collect();
        anyhow::ensure!(
            fit.iter()
                .all(|&i| !dev_canons.contains(targets.positions[i].canon.as_str())),
            "a canonical class straddles V4_TRAIN_FIT and V4_TRAIN_DEV"
        );
        Ok(Self {
            fit_digest: id_digest(&targets.positions, &fit),
            dev_digest: id_digest(&targets.positions, &dev),
            targets,
            fit,
            dev,
        })
    }

    pub fn position(&self, i: usize) -> &ProofPosition {
        &self.targets.positions[i]
    }

    pub fn roots(&self, idx: &[usize]) -> anyhow::Result<Vec<GameState>> {
        idx.iter()
            .map(|&i| {
                GameState::from_fen(&self.position(i).fen)
                    .map_err(|e| anyhow::anyhow!("{}: {e:?}", self.position(i).id))
            })
            .collect()
    }

    /// Dense `[n][w]` uniform targets over the correct set, zero-padded to width `w`.
    pub fn target_rows(&self, idx: &[usize], w: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; idx.len() * w];
        for (r, &i) in idx.iter().enumerate() {
            for (c, v) in self.position(i).target().iter().enumerate() {
                out[r * w + c] = *v;
            }
        }
        out
    }

    pub fn correct(&self, i: usize) -> Vec<usize> {
        self.position(i).correct.iter().map(|&c| c as usize).collect()
    }

    /// `(family, depth)` cell of each index in `part`, for the cell-balanced sampler.
    pub fn cells(&self, part: &[usize]) -> Vec<(String, u8)> {
        part.iter()
            .map(|&i| {
                let p = self.position(i);
                (p.family.clone(), p.mate_depth)
            })
            .collect()
    }
}

/// Deterministic cell-balanced sampler over a part (`cell_balanced_v1`, as in P5).
pub struct PartSampler {
    part: Vec<usize>,
    inner: CellSampler,
}

impl PartSampler {
    pub fn new(data: &V4Data, part: &[usize], seed: u64) -> Self {
        Self {
            part: part.to_vec(),
            inner: CellSampler::new(&data.cells(part), seed),
        }
    }

    pub fn next_batch(&mut self, n: usize) -> Vec<usize> {
        (0..n).map(|_| self.part[self.inner.next_index()]).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_partition_rule_is_deterministic_and_about_ten_percent() {
        let dev = (0..20_000)
            .filter(|i| is_dev(&format!("canon-{i}")))
            .count();
        assert!((1_700..=2_300).contains(&dev), "dev share {dev}/20000");
        assert_eq!(is_dev("abc"), is_dev("abc"));
    }

    #[test]
    fn the_partition_rule_is_the_frozen_one() {
        // Frozen: sha256("v4_train_dev_v1|" + canon), first 8 bytes big-endian, mod 10 == 0.
        let mut h = Sha256::new();
        h.update(b"v4_train_dev_v1|x");
        let d = h.finalize();
        let v = u64::from_be_bytes(d[..8].try_into().unwrap());
        assert_eq!(is_dev("x"), v % 10 == 0);
    }
}
