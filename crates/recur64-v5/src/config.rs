//! Frozen V5 geometry and semantic-contract identity.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const ARCHITECTURE: &str = "counterfactual_relational_loop_v1";
pub const ROOT_FRAME: &str = "v5_root_frame_v1";
pub const ROOT_HYPOTHESES: &str = "v5_root_hypotheses_v1";
pub const RETURNED_PAYLOAD: &str = "v5_returned_payload_v1";
pub const STATE_TOKENS: &str = "v5_state_tokens4_v1";
pub const ACQUIRED_GRAPH: &str = "v5_acquired_graph_v1";
pub const RELATIONAL_LOOP: &str = "v5_relational_loop_v1";
pub const INPUT_RECALL: &str = "v5_input_recall_v1";
pub const PAIRED_NULL_READOUT: &str = "v5_paired_null_readout_v1";
pub const CORRECT_SET_LOSS: &str = "v5_correct_set_loss_v1";
pub const PILOT: &str = "v5_fixed_graph_reader_pilot_v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Contracts {
    pub root_frame: String,
    pub root_hypotheses: String,
    pub returned_payload: String,
    pub state_tokens: String,
    pub acquired_graph: String,
    pub relational_loop: String,
    pub input_recall: String,
    pub paired_null_readout: String,
    pub correct_set_loss: String,
    pub pilot: String,
}

impl Default for Contracts {
    fn default() -> Self {
        Self {
            root_frame: ROOT_FRAME.into(),
            root_hypotheses: ROOT_HYPOTHESES.into(),
            returned_payload: RETURNED_PAYLOAD.into(),
            state_tokens: STATE_TOKENS.into(),
            acquired_graph: ACQUIRED_GRAPH.into(),
            relational_loop: RELATIONAL_LOOP.into(),
            input_recall: INPUT_RECALL.into(),
            paired_null_readout: PAIRED_NULL_READOUT.into(),
            correct_set_loss: CORRECT_SET_LOSS.into(),
            pilot: PILOT.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct V5Config {
    pub architecture: String,
    pub width: usize,
    pub heads: usize,
    pub ffn: usize,
    pub root_blocks: usize,
    pub candidate_blocks: usize,
    pub state_blocks: usize,
    pub state_slots: usize,
    pub evidence_blocks: usize,
    pub hypothesis_blocks: usize,
    pub readout_hidden: usize,
    pub dropout: f64,
    pub rms_eps: f64,
    pub residual_alpha: f64,
    pub max_query: usize,
    pub max_depth: usize,
    pub contracts: Contracts,
}

impl Default for V5Config {
    fn default() -> Self {
        Self {
            architecture: ARCHITECTURE.into(),
            width: 256,
            heads: 8,
            ffn: 768,
            root_blocks: 4,
            candidate_blocks: 1,
            state_blocks: 2,
            state_slots: 4,
            evidence_blocks: 1,
            hypothesis_blocks: 1,
            readout_hidden: 256,
            dropout: 0.0,
            rms_eps: 1e-5,
            residual_alpha: 0.5,
            max_query: 16,
            max_depth: 5,
            contracts: Contracts::default(),
        }
    }
}

impl V5Config {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.architecture == ARCHITECTURE,
            "V5 architecture mismatch"
        );
        anyhow::ensure!(
            self.width == 256 && self.heads == 8 && self.ffn == 768,
            "V5 geometry is frozen"
        );
        anyhow::ensure!(
            self.root_blocks == 4 && self.candidate_blocks == 1 && self.state_blocks == 2,
            "V5 encoder block counts are frozen"
        );
        anyhow::ensure!(
            self.state_slots == 4 && self.evidence_blocks == 1 && self.hypothesis_blocks == 1,
            "V5 relational geometry is frozen"
        );
        anyhow::ensure!(self.readout_hidden == 256, "V5 readout width is frozen");
        anyhow::ensure!(self.dropout == 0.0, "V5 dropout must be zero");
        anyhow::ensure!(
            self.rms_eps == 1e-5 && self.residual_alpha == 0.5,
            "V5 normalization/residual contract mismatch"
        );
        anyhow::ensure!(
            self.max_query == 16 && self.max_depth == 5,
            "V5 graph bounds mismatch"
        );
        anyhow::ensure!(
            self.width.is_multiple_of(self.heads),
            "width must divide by heads"
        );
        anyhow::ensure!(
            self.contracts == Contracts::default(),
            "V5 subcontracts mismatch"
        );
        Ok(())
    }

    pub fn scientific_digest(&self) -> anyhow::Result<String> {
        self.validate()?;
        let bytes = serde_json::to_vec(self)?;
        let mut h = Sha256::new();
        h.update(b"recur64.v5.scientific_config.v1\0");
        h.update(bytes);
        Ok(format!("{:x}", h.finalize()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_config_round_trips_and_hashes() {
        let cfg = V5Config::default();
        cfg.validate().unwrap();
        let json = serde_json::to_string(&cfg).unwrap();
        let decoded: V5Config = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, cfg);
        assert_eq!(
            decoded.scientific_digest().unwrap(),
            cfg.scientific_digest().unwrap()
        );
    }

    #[test]
    fn semantic_changes_are_refused() {
        let cfg = V5Config {
            residual_alpha: 0.0,
            ..V5Config::default()
        };
        assert!(cfg.validate().is_err());
        let mut cfg = V5Config::default();
        cfg.contracts.pilot = "other".into();
        assert!(cfg.validate().is_err());
    }
}
