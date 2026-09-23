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
    Bf16,
    Fp16,
}

impl Precision {
    pub fn label(&self) -> &'static str {
        match self {
            Precision::Fp32 => "fp32",
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
}
