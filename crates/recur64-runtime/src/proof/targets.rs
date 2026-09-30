//! `ProofTargetsV1`: a versioned exact dataset of near-mate positions with exact
//! policy targets.
//!
//! Every label comes from exhaustive adversarial rules search
//! ([`super::mate::MateSolver`]); there is no network, no PUCT target, no external
//! engine and no human data. The file carries a content digest over the contract
//! and every position, so a changed position or label is detectable.

use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Schema identifier written to every file.
pub const SCHEMA: &str = "proof_targets_v1";
/// History contract: positions are FEN-only with zero half-move clock and no
/// earlier history (see `mate.rs`).
pub const HISTORY_CONTRACT: &str = "fresh_no_history_v1";
/// Definition of the labels, recorded verbatim in the file.
pub const MATE_DEFINITION: &str = "mate in N: the attacker (side to move) can force checkmate \
within N of its own moves against every legal defender reply; correct root moves are those \
beginning a forced mate of the position's minimal depth";

/// Material families: white pieces besides the king (black has only a king).
pub const FAMILIES: [(&str, &[char]); 5] = [
    ("KQvK", &['K', 'Q']),
    ("KRvK", &['K', 'R']),
    ("KQQvK", &['K', 'Q', 'Q']),
    ("KQRvK", &['K', 'Q', 'R']),
    ("KRRvK", &['K', 'R', 'R']),
];

/// Dataset splits. Hard-disjoint by exact FEN and by symmetry-canonical key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Split {
    Train,
    Tune,
    Confirm,
    /// P2.5 heavy-family evaluation holdouts (A: factorial, B: data scale, C: horizon).
    #[serde(rename = "holdout_a")]
    HoldoutA,
    #[serde(rename = "holdout_b")]
    HoldoutB,
    #[serde(rename = "holdout_c")]
    HoldoutC,
}

impl Split {
    pub fn label(self) -> &'static str {
        match self {
            Split::Train => "train",
            Split::Tune => "tune",
            Split::Confirm => "confirm",
            Split::HoldoutA => "holdout_a",
            Split::HoldoutB => "holdout_b",
            Split::HoldoutC => "holdout_c",
        }
    }
}

/// One position with its exact labels.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProofPosition {
    pub id: String,
    pub fen: String,
    pub split: Split,
    pub family: String,
    /// Minimal mate depth (attacker moves): 1..=5.
    pub mate_depth: u8,
    /// Symmetry-canonical key (minimum over the 8 board symmetries).
    pub canon: String,
    /// Canonical `ActionId` index of every legal move, in `legal_actions()` order.
    pub legal: Vec<u16>,
    /// Indices INTO `legal` of the correct root moves.
    pub correct: Vec<u32>,
    /// `correct / legal`: top-1 accuracy of a uniformly random legal move.
    pub chance_top1: f32,
    /// Seed of the split's generator stream that produced this position.
    pub generator_seed: u64,
}

impl ProofPosition {
    /// Uniform exact target over the correct set, aligned to `legal`.
    pub fn target(&self) -> Vec<f32> {
        let mut t = vec![0.0f32; self.legal.len()];
        let w = 1.0 / self.correct.len() as f32;
        for &c in &self.correct {
            t[c as usize] = w;
        }
        t
    }
}

/// The file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProofTargets {
    pub schema: String,
    pub history_contract: String,
    pub mate_definition: String,
    pub facts_version: u32,
    pub action_version: u32,
    pub split: Split,
    pub seed: u64,
    pub filters: serde_json::Value,
    pub positions: Vec<ProofPosition>,
    /// SHA-256 over the contract fields and every position (see [`Self::compute_digest`]).
    pub digest: String,
}

impl ProofTargets {
    pub fn new(
        split: Split,
        seed: u64,
        filters: serde_json::Value,
        positions: Vec<ProofPosition>,
    ) -> Self {
        let mut t = Self {
            schema: SCHEMA.into(),
            history_contract: HISTORY_CONTRACT.into(),
            mate_definition: MATE_DEFINITION.into(),
            facts_version: recur64_core::CANDIDATE_FACTS_VERSION,
            action_version: recur64_core::ContractVersions::V1.action,
            split,
            seed,
            filters,
            positions,
            digest: String::new(),
        };
        t.digest = t.compute_digest();
        t
    }

    /// Digest over everything except the digest field itself, computed from a
    /// canonical JSON serialization (struct field order is fixed by the types).
    pub fn compute_digest(&self) -> String {
        let mut h = Sha256::new();
        h.update(self.schema.as_bytes());
        h.update(self.history_contract.as_bytes());
        h.update(self.mate_definition.as_bytes());
        h.update(self.facts_version.to_le_bytes());
        h.update(self.action_version.to_le_bytes());
        h.update(self.split.label().as_bytes());
        h.update(self.seed.to_le_bytes());
        h.update(serde_json::to_vec(&self.filters).unwrap_or_default());
        for p in &self.positions {
            h.update(serde_json::to_vec(p).unwrap_or_default());
        }
        format!("{:x}", h.finalize())
    }

    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(path, serde_json::to_vec(self)?)?;
        Ok(())
    }

    /// Load and validate: schema, contracts, digest, and structural label sanity.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let t: ProofTargets = serde_json::from_slice(&std::fs::read(path)?)
            .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
        t.validate()
            .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
        Ok(t)
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(self.schema == SCHEMA, "unknown schema '{}'", self.schema);
        anyhow::ensure!(
            self.history_contract == HISTORY_CONTRACT,
            "history contract '{}' is not {HISTORY_CONTRACT}",
            self.history_contract
        );
        anyhow::ensure!(
            self.facts_version == recur64_core::CANDIDATE_FACTS_VERSION
                && self.action_version == recur64_core::ContractVersions::V1.action,
            "facts/action contract version mismatch"
        );
        anyhow::ensure!(
            self.digest == self.compute_digest(),
            "content digest mismatch: the dataset was modified after generation"
        );
        for p in &self.positions {
            anyhow::ensure!(
                p.split == self.split,
                "{}: split differs from the file",
                p.id
            );
            anyhow::ensure!(
                !p.correct.is_empty() && p.correct.iter().all(|&c| (c as usize) < p.legal.len()),
                "{}: invalid correct set",
                p.id
            );
            anyhow::ensure!(
                (1..=super::mate::MAX_DEPTH).contains(&p.mate_depth),
                "{}: mate depth {} out of range",
                p.id,
                p.mate_depth
            );
            let chance = p.correct.len() as f32 / p.legal.len() as f32;
            anyhow::ensure!(
                (chance - p.chance_top1).abs() < 1e-6,
                "{}: stored chance {} != correct/legal {}",
                p.id,
                p.chance_top1,
                chance
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pos(id: &str) -> ProofPosition {
        ProofPosition {
            id: id.into(),
            fen: "7k/8/6K1/8/8/8/8/1Q6 w - - 0 1".into(),
            split: Split::Train,
            family: "KQvK".into(),
            mate_depth: 2,
            canon: "x".into(),
            legal: vec![1, 2, 3, 4],
            correct: vec![1, 3],
            chance_top1: 0.5,
            generator_seed: 7,
        }
    }

    #[test]
    fn digest_detects_any_change_and_validates() {
        let t = ProofTargets::new(Split::Train, 1, serde_json::json!({"a": 1}), vec![pos("a")]);
        t.validate().unwrap();
        assert_eq!(t.positions[0].target(), vec![0.0, 0.5, 0.0, 0.5]);
        let mut u = t.clone();
        u.positions[0].correct = vec![0];
        u.positions[0].chance_top1 = 0.25;
        assert!(
            u.validate().is_err(),
            "modified labels must fail the digest"
        );
        let mut v = t.clone();
        v.seed = 2;
        assert!(v.validate().is_err());
    }

    #[test]
    fn inconsistent_labels_are_refused() {
        let mut p = pos("a");
        p.correct = vec![9];
        let t = ProofTargets::new(Split::Train, 1, serde_json::json!({}), vec![p]);
        assert!(t.validate().is_err());
        let mut p = pos("b");
        p.chance_top1 = 0.9;
        let t = ProofTargets::new(Split::Train, 1, serde_json::json!({}), vec![p]);
        assert!(t.validate().is_err());
    }
}
