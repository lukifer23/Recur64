//! Retired P25 loader retained for historical contracts only.
//! Production entry points are locked pending complete measured native DATA-B.

use std::path::Path;

use recur64_runtime::proof::custody::load_working_split;
use recur64_runtime::proof::targets::{ProofPosition, ProofTargets, Split};
use sha2::{Digest, Sha256};

pub const TRAIN_DIGEST: &str = "3b25dc8549dd2fc9d47c30e294c273b3306aecb3eba91b964715326ddf74f2e6";
pub const TRAIN_POSITIONS: usize = 44_332;
pub const FIT_POSITIONS: usize = 39_929;
pub const DEV_POSITIONS: usize = 4_403;
pub const FIT_DIGEST: &str = "a01932d7db863fbd0d160bc04bd3589137449d2c1e33d408f8d5d3f3cb45f8f5";
pub const DEV_DIGEST: &str = "f877219bc87d916ad8478a745f1572821c1a051337c3a071f2b37b9a5f83b899";
pub const PARTITION: &str = "v4_train_dev_v1";

pub fn is_dev(canon: &str) -> bool {
    let mut h = Sha256::new();
    h.update(PARTITION.as_bytes());
    h.update(b"|");
    h.update(canon.as_bytes());
    let d = h.finalize();
    u64::from_be_bytes(d[..8].try_into().expect("eight bytes")) % 10 == 0
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

pub struct V5Data {
    pub targets: ProofTargets,
    pub fit: Vec<usize>,
    pub dev: Vec<usize>,
}

impl V5Data {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        crate::native_data::require_measured_binding()?;
        let targets = load_working_split(path, &[Split::Train])?;
        anyhow::ensure!(
            targets.digest == TRAIN_DIGEST,
            "{}: digest {} is not P25_DATA_V1 TRAIN",
            path.display(),
            targets.digest
        );
        anyhow::ensure!(
            targets.positions.len() == TRAIN_POSITIONS,
            "{}: expected {TRAIN_POSITIONS} positions, found {}",
            path.display(),
            targets.positions.len()
        );
        let (mut fit, mut dev) = (Vec::new(), Vec::new());
        for (i, p) in targets.positions.iter().enumerate() {
            if is_dev(&p.canon) {
                dev.push(i)
            } else {
                fit.push(i)
            }
        }
        anyhow::ensure!(
            fit.len() == FIT_POSITIONS && dev.len() == DEV_POSITIONS,
            "partition counts differ: FIT {} DEV {}",
            fit.len(),
            dev.len()
        );
        anyhow::ensure!(
            id_digest(&targets.positions, &fit) == FIT_DIGEST,
            "FIT sorted-ID digest mismatch"
        );
        anyhow::ensure!(
            id_digest(&targets.positions, &dev) == DEV_DIGEST,
            "DEV sorted-ID digest mismatch"
        );
        let dev_classes: std::collections::HashSet<&str> = dev
            .iter()
            .map(|&i| targets.positions[i].canon.as_str())
            .collect();
        anyhow::ensure!(
            fit.iter()
                .all(|&i| !dev_classes.contains(targets.positions[i].canon.as_str())),
            "canonical class straddles FIT and DEV"
        );
        Ok(Self { targets, fit, dev })
    }

    pub fn verify_custody(&self) -> anyhow::Result<()> {
        crate::native_data::require_measured_binding()?;
        anyhow::ensure!(
            self.targets.digest == TRAIN_DIGEST
                && self.targets.positions.len() == TRAIN_POSITIONS
                && self.fit.len() == FIT_POSITIONS
                && self.dev.len() == DEV_POSITIONS
                && id_digest(&self.targets.positions, &self.fit) == FIT_DIGEST
                && id_digest(&self.targets.positions, &self.dev) == DEV_DIGEST,
            "V5Data is not the exact P25 TRAIN inherited partition"
        );
        Ok(())
    }

    pub fn position(&self, index: usize) -> &ProofPosition {
        &self.targets.positions[index]
    }

    pub fn roots(&self, indices: &[usize]) -> anyhow::Result<Vec<recur64_core::GameState>> {
        indices
            .iter()
            .map(|&index| {
                recur64_core::GameState::from_fen(&self.position(index).fen)
                    .map_err(|e| anyhow::anyhow!("{}: {e:?}", self.position(index).id))
            })
            .collect()
    }

    pub fn cells(&self, part: &[usize]) -> Vec<(String, u8)> {
        part.iter()
            .map(|&index| {
                let position = self.position(index);
                (position.family.clone(), position.mate_depth)
            })
            .collect()
    }

    pub fn validate_root_alignment(
        &self,
        index: usize,
        root: &recur64_core::GameState,
    ) -> anyhow::Result<()> {
        let observed: Vec<u16> = root
            .legal_actions()
            .iter()
            .map(|action| action.index() as u16)
            .collect();
        anyhow::ensure!(
            observed == self.position(index).legal,
            "{}: legal action alignment differs from the dataset",
            self.position(index).id
        );
        anyhow::ensure!(
            !observed.is_empty() && !self.position(index).correct.is_empty(),
            "{}: legal and correct sets must be nonempty",
            self.position(index).id
        );
        anyhow::ensure!(
            self.position(index)
                .correct
                .iter()
                .all(|&correct| (correct as usize) < observed.len()),
            "{}: correct index is outside the legal set",
            self.position(index).id
        );
        Ok(())
    }
}
