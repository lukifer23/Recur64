//! Phase 0 configuration contracts.
//!
//! These are the *probe* contracts, not the eventual chess observation/action
//! schema. They exist so `recur64 model-info` and `recur64 bench` can describe
//! the exact graph they instantiate.

use serde::{Deserialize, Serialize};

fn d_squares() -> usize {
    64
}
fn d_in_features() -> usize {
    119
}
fn d_policy_dim() -> usize {
    128
}
fn d_wdl_classes() -> usize {
    3
}
fn d_promo_codes() -> usize {
    5
}
fn d_epsilon() -> f64 {
    1e-5
}

/// Precision requested for a run. The backend is responsible for erroring
/// visibly if the requested precision is not supported by the full graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Precision {
    #[default]
    Fp32,
    /// FP32 storage and accumulation; matmul inputs may be rounded to TF32
    /// (10-bit mantissa) on tensor cores (T5, owner-approved). Requires a
    /// binary built with the `tf32` feature.
    Tf32,
    Bf16,
    Fp16,
}

impl Precision {
    pub fn label(&self) -> &'static str {
        match self {
            Precision::Fp32 => "fp32",
            Precision::Tf32 => "tf32",
            Precision::Bf16 => "bf16",
            Precision::Fp16 => "fp16",
        }
    }
}

/// Requested accelerator. Never silently substituted; the CLI errors if the
/// requested device cannot be initialised.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum DeviceKind {
    #[default]
    Cpu,
    Cuda,
}

/// Transformer probe geometry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub width: usize,
    pub heads: usize,
    pub ffn: usize,
    /// Feed-forward control uses input=0, core=N, output=0.
    /// Recurrent R10 uses input=2, core=4, output=2 (all counted once).
    pub input_blocks: usize,
    pub core_blocks: usize,
    pub output_blocks: usize,
    #[serde(default = "d_squares")]
    pub squares: usize,
    #[serde(default = "d_in_features")]
    pub in_features: usize,
    #[serde(default = "d_policy_dim")]
    pub policy_dim: usize,
    #[serde(default = "d_wdl_classes")]
    pub wdl_classes: usize,
    #[serde(default = "d_promo_codes")]
    pub promo_codes: usize,
    #[serde(default = "d_epsilon")]
    pub rms_eps: f64,
    /// Model architecture. Historical configs deserialize as `probe_v1` and, so
    /// that their scientific hashes do not change, the default is never
    /// serialized.
    #[serde(default, skip_serializing_if = "Architecture::is_default")]
    pub architecture: Architecture,
    /// Candidate-token geometry; present iff `architecture == candidate_v25`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate: Option<CandidateConfig>,
    /// Facts-delta geometry; present iff `architecture == legacy_facts_v25`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub legacy_facts: Option<LegacyFactsConfig>,
}

/// Which network a `ModelConfig` instantiates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Architecture {
    /// The historical `ProbeModel` (source/destination bilinear head v2).
    #[default]
    ProbeV1,
    /// Workstation V2.5 candidate transformer: one pass, legal-move tokens.
    CandidateV25,
    /// P2.5: the legacy head-v2 policy plus a candidate-local CandidateFacts
    /// policy delta (the missing cell of the 2x2 factorial).
    LegacyFactsV25,
}

impl Architecture {
    pub fn is_default(&self) -> bool {
        *self == Architecture::ProbeV1
    }

    pub fn id(&self) -> &'static str {
        match self {
            Architecture::ProbeV1 => "probe_v1",
            Architecture::CandidateV25 => "candidate_v25",
            Architecture::LegacyFactsV25 => "legacy_facts_v25",
        }
    }
}

fn d_cand_dim() -> usize {
    256
}
fn d_cand_heads() -> usize {
    4
}
fn d_cand_ffn() -> usize {
    512
}
fn d_cand_blocks() -> usize {
    1
}
fn d_facts_hidden() -> usize {
    64
}
fn d_cand_policy_hidden() -> usize {
    128
}
fn d_true() -> bool {
    true
}

/// Version of the candidate-token contract (move-token construction).
pub const CANDIDATE_TOKEN_CONTRACT: u32 = 1;
/// Version of the candidate-block contract (masked self-attention block).
pub const CANDIDATE_BLOCK_CONTRACT: u32 = 1;
/// Version of the `CandidateFactsV1` field layout consumed by the model.
pub const CANDIDATE_FACTS_VERSION: u32 = recur64_core::CANDIDATE_FACTS_VERSION;
/// Fields per candidate in `CandidateFactsV1`.
pub const CANDIDATE_FACT_FIELDS: usize = recur64_core::CANDIDATE_FACT_FIELDS;
/// Version of the V2.5 readout function (policy scorer + WDL head).
pub const CANDIDATE_HEAD_VERSION: u32 = 1;

/// Version of the LF fact-delta contract (8 -> hidden -> 1 logit delta added to the
/// legacy policy logit before the masked softmax).
pub const FACT_DELTA_CONTRACT: u32 = 1;

fn d_facts_delta_hidden() -> usize {
    64
}

/// Legacy-plus-facts geometry (P2.5 LF).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LegacyFactsConfig {
    #[serde(default = "d_facts_delta_hidden")]
    pub facts_hidden: usize,
}

impl Default for LegacyFactsConfig {
    fn default() -> Self {
        Self {
            facts_hidden: d_facts_delta_hidden(),
        }
    }
}

/// Candidate-transformer geometry (V2.5).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateConfig {
    #[serde(default = "d_cand_dim")]
    pub dim: usize,
    #[serde(default = "d_cand_heads")]
    pub heads: usize,
    #[serde(default = "d_cand_ffn")]
    pub ffn: usize,
    #[serde(default = "d_cand_blocks")]
    pub blocks: usize,
    #[serde(default = "d_facts_hidden")]
    pub facts_hidden: usize,
    #[serde(default = "d_cand_policy_hidden")]
    pub policy_hidden: usize,
    /// C0 (ablation) sets this false: the facts modules exist and are counted,
    /// but the facts input is zeroed. It changes the function, so it is part of
    /// the model identity.
    #[serde(default = "d_true")]
    pub facts_enabled: bool,
}

impl Default for CandidateConfig {
    fn default() -> Self {
        Self {
            dim: d_cand_dim(),
            heads: d_cand_heads(),
            ffn: d_cand_ffn(),
            blocks: d_cand_blocks(),
            facts_hidden: d_facts_hidden(),
            policy_hidden: d_cand_policy_hidden(),
            facts_enabled: true,
        }
    }
}

impl ModelConfig {
    /// The V2.5 primary geometry: width 640, 10 heads, FFN 1280, 8 unique
    /// blocks, no input/output blocks.
    pub fn candidate_v25(facts_enabled: bool) -> Self {
        Self {
            width: 640,
            heads: 10,
            ffn: 1280,
            input_blocks: 0,
            core_blocks: 8,
            output_blocks: 0,
            squares: d_squares(),
            in_features: d_in_features(),
            policy_dim: d_policy_dim(),
            wdl_classes: d_wdl_classes(),
            promo_codes: d_promo_codes(),
            rms_eps: d_epsilon(),
            architecture: Architecture::CandidateV25,
            candidate: Some(CandidateConfig {
                facts_enabled,
                ..CandidateConfig::default()
            }),
            legacy_facts: None,
        }
    }

    /// P2.5 LF: the L board geometry (width 640, 10 heads, FFN 1280, 8 blocks) with
    /// the legacy head-v2 policy plus the CandidateFacts policy delta.
    pub fn legacy_facts_v25() -> Self {
        Self {
            width: 640,
            heads: 10,
            ffn: 1280,
            input_blocks: 0,
            core_blocks: 8,
            output_blocks: 0,
            squares: d_squares(),
            in_features: d_in_features(),
            policy_dim: d_policy_dim(),
            wdl_classes: d_wdl_classes(),
            promo_codes: d_promo_codes(),
            rms_eps: d_epsilon(),
            architecture: Architecture::LegacyFactsV25,
            candidate: None,
            legacy_facts: Some(LegacyFactsConfig::default()),
        }
    }

    /// Refuse an architecture/geometry combination that is not a real model.
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.legacy_facts.is_none() || self.architecture == Architecture::LegacyFactsV25,
            "only legacy_facts_v25 may carry a facts-delta geometry"
        );
        if self.architecture == Architecture::LegacyFactsV25 {
            anyhow::ensure!(
                self.candidate.is_none(),
                "legacy_facts_v25 must not carry a candidate-token geometry"
            );
            anyhow::ensure!(
                self.legacy_facts.is_some(),
                "legacy_facts_v25 requires a facts-delta geometry"
            );
            anyhow::ensure!(
                self.input_blocks == 0 && self.output_blocks == 0,
                "legacy_facts_v25 is one pass: no input or output blocks"
            );
            return Ok(());
        }
        match (self.architecture, &self.candidate) {
            (Architecture::ProbeV1, None) => Ok(()),
            (Architecture::ProbeV1, Some(_)) => {
                anyhow::bail!("probe_v1 must not carry a candidate geometry")
            }
            (Architecture::LegacyFactsV25, _) => unreachable!("handled above"),
            (Architecture::CandidateV25, None) => {
                anyhow::bail!("candidate_v25 requires a candidate geometry")
            }
            (Architecture::CandidateV25, Some(c)) => {
                anyhow::ensure!(
                    self.input_blocks == 0 && self.output_blocks == 0,
                    "candidate_v25 has no input or output blocks (one pass, unique core blocks only)"
                );
                anyhow::ensure!(
                    c.dim.is_multiple_of(c.heads),
                    "candidate dim {} not divisible by heads {}",
                    c.dim,
                    c.heads
                );
                Ok(())
            }
        }
    }

    /// Refuse a recurrence the architecture cannot execute. candidate_v25 is
    /// strictly one pass; the field is never silently ignored.
    pub fn check_recurrence(&self, recurrence: usize) -> anyhow::Result<()> {
        if matches!(
            self.architecture,
            Architecture::CandidateV25 | Architecture::LegacyFactsV25
        ) {
            anyhow::ensure!(
                recurrence == 1,
                "{} is a one-pass architecture; recurrence {recurrence} is refused",
                self.architecture.id()
            );
        }
        Ok(())
    }
}

impl ModelConfig {
    pub fn unique_blocks(&self) -> usize {
        self.input_blocks + self.core_blocks + self.output_blocks
    }

    /// Executed transformer blocks for final-output inference at recurrence `r`.
    pub fn executed_blocks_final(&self, r: usize) -> usize {
        self.input_blocks + self.core_blocks * r + self.output_blocks
    }

    /// Executed transformer blocks for deep-supervision training at recurrence `r`
    /// (output blocks evaluated at every readout).
    pub fn executed_blocks_deep_supervision(&self, r: usize) -> usize {
        self.input_blocks + (self.core_blocks + self.output_blocks) * r
    }

    pub fn head_dim(&self) -> usize {
        assert!(
            self.width.is_multiple_of(self.heads),
            "width {} not divisible by heads {}",
            self.width,
            self.heads
        );
        self.width / self.heads
    }
}

/// A complete Phase 0 probe configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeConfig {
    pub name: String,
    pub model: ModelConfig,
    #[serde(default)]
    pub precision: Precision,
    #[serde(default)]
    pub device: DeviceKind,
    /// Recurrence values this probe is allowed to execute (e.g. [1,2,4]).
    #[serde(default = "default_recurrence")]
    pub recurrence: Vec<usize>,
    /// If true, output blocks run at every recurrent readout during training.
    #[serde(default)]
    pub deep_supervision: bool,
    #[serde(default = "default_batch")]
    pub batch_size: usize,
    #[serde(default)]
    pub seed: u64,
}

fn default_recurrence() -> Vec<usize> {
    vec![1]
}
fn default_batch() -> usize {
    16
}

impl ProbeConfig {
    pub fn from_toml_str(s: &str) -> anyhow::Result<Self> {
        Ok(toml::from_str(s)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f10_executed_blocks_match_spec() {
        let m = ModelConfig {
            width: 384,
            heads: 12,
            ffn: 768,
            input_blocks: 0,
            core_blocks: 8,
            output_blocks: 0,
            squares: 64,
            in_features: 119,
            policy_dim: 128,
            wdl_classes: 3,
            promo_codes: 5,
            rms_eps: 1e-5,
            architecture: Default::default(),
            candidate: None,
            legacy_facts: None,
        };
        assert_eq!(m.unique_blocks(), 8);
        assert_eq!(m.executed_blocks_final(1), 8);
    }

    #[test]
    fn r10_executed_blocks_match_spec() {
        let m = ModelConfig {
            width: 384,
            heads: 12,
            ffn: 768,
            input_blocks: 2,
            core_blocks: 4,
            output_blocks: 2,
            squares: 64,
            in_features: 119,
            policy_dim: 128,
            wdl_classes: 3,
            promo_codes: 5,
            rms_eps: 1e-5,
            architecture: Default::default(),
            candidate: None,
            legacy_facts: None,
        };
        assert_eq!(m.unique_blocks(), 8);
        // 2 + 4R + 2
        assert_eq!(m.executed_blocks_final(1), 8);
        assert_eq!(m.executed_blocks_final(2), 12);
        assert_eq!(m.executed_blocks_final(4), 20);
        // 2 + 6R
        assert_eq!(m.executed_blocks_deep_supervision(1), 8);
        assert_eq!(m.executed_blocks_deep_supervision(2), 14);
        assert_eq!(m.executed_blocks_deep_supervision(4), 26);
    }

    #[test]
    fn probe_identity_is_unchanged_by_the_architecture_field() {
        // A historical config (no architecture key) deserializes as probe_v1 and
        // re-serializes WITHOUT the new keys, so scientific hashes are stable.
        let legacy = r#"{"width":384,"heads":12,"ffn":768,"input_blocks":0,"core_blocks":8,"output_blocks":0}"#;
        let m: ModelConfig = serde_json::from_str(legacy).unwrap();
        assert_eq!(m.architecture, Architecture::ProbeV1);
        assert!(m.candidate.is_none());
        let v = serde_json::to_value(&m).unwrap();
        let obj = v.as_object().unwrap();
        assert!(!obj.contains_key("architecture") && !obj.contains_key("candidate"));
        assert_eq!(obj.len(), 12, "exactly the twelve historical keys");
        m.validate().unwrap();
    }

    #[test]
    fn candidate_v25_round_trips_and_validates() {
        let m = ModelConfig::candidate_v25(true);
        let back: ModelConfig = serde_json::from_str(&serde_json::to_string(&m).unwrap()).unwrap();
        assert_eq!(back.architecture, Architecture::CandidateV25);
        assert_eq!(back.candidate, m.candidate);
        m.validate().unwrap();
        assert_ne!(
            serde_json::to_value(ModelConfig::candidate_v25(true)).unwrap(),
            serde_json::to_value(ModelConfig::candidate_v25(false)).unwrap(),
            "C0 and CF are distinct identities"
        );
        assert_eq!((m.width, m.heads, m.ffn, m.core_blocks), (640, 10, 1280, 8));
        assert_eq!(m.head_dim(), 64);
    }

    #[test]
    fn inconsistent_architecture_and_recurrence_are_refused() {
        let mut m = ModelConfig::candidate_v25(true);
        assert!(m.check_recurrence(1).is_ok());
        assert!(
            m.check_recurrence(2).is_err(),
            "no recurrence on candidate_v25"
        );
        m.input_blocks = 1;
        assert!(m.validate().is_err());
        let mut m = ModelConfig::candidate_v25(true);
        m.candidate = None;
        assert!(m.validate().is_err());
        let mut p = ModelConfig::candidate_v25(true);
        p.architecture = Architecture::ProbeV1;
        assert!(p.validate().is_err());
    }
}
