//! Phase 2 run configuration. Every run records its fully resolved config.

use serde::{Deserialize, Serialize};

use recur64_model::config::{DeviceKind, ModelConfig, Precision};

fn default_recurrence() -> usize {
    1
}
fn default_simulations() -> u32 {
    16
}
fn default_c_puct() -> f32 {
    1.0
}
fn default_temperature() -> f32 {
    1.0
}
fn default_active_games() -> u32 {
    32
}
fn default_cpu_workers() -> usize {
    8
}
fn default_ply_cap() -> u32 {
    256
}
fn default_max_batch() -> usize {
    32
}
fn default_batch_timeout_us() -> u64 {
    500
}
fn default_shard_max_games() -> usize {
    256
}
fn default_train_batch() -> usize {
    32
}
fn default_max_updates() -> usize {
    10
}
fn default_lr() -> f64 {
    3e-4
}
fn default_arena_games() -> u32 {
    4
}
fn default_budget_minutes() -> u64 {
    10
}
fn default_precision() -> String {
    "fp32".to_string()
}
fn default_device() -> String {
    "cpu".to_string()
}
fn default_seed() -> u64 {
    1
}

/// A complete, resolved Phase 2 run configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunConfig {
    pub run_id: String,
    pub model: ModelConfig,
    #[serde(default = "default_recurrence")]
    pub recurrence: usize,
    #[serde(default = "default_precision")]
    pub precision: String,
    #[serde(default = "default_device")]
    pub device: String,

    // Self-play.
    #[serde(default = "default_simulations")]
    pub simulations_per_move: u32,
    #[serde(default = "default_c_puct")]
    pub c_puct: f32,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_active_games")]
    pub active_games: u32,
    #[serde(default = "default_cpu_workers")]
    pub cpu_workers: usize,
    #[serde(default = "default_ply_cap")]
    pub ply_cap: u32,
    #[serde(default = "default_seed")]
    pub seed: u64,

    // Inference batching.
    #[serde(default = "default_max_batch")]
    pub max_inference_batch: usize,
    #[serde(default = "default_batch_timeout_us")]
    pub batch_timeout_us: u64,

    // Replay.
    #[serde(default = "default_shard_max_games")]
    pub shard_max_games: usize,

    // Learner.
    #[serde(default = "default_train_batch")]
    pub train_batch: usize,
    #[serde(default = "default_max_updates")]
    pub max_updates: usize,
    #[serde(default = "default_lr")]
    pub lr: f64,

    // Arena.
    #[serde(default = "default_arena_games")]
    pub arena_games: u32,

    #[serde(default = "default_budget_minutes")]
    pub run_budget_minutes: u64,

    /// Optional start position for self-play. Defaults to the standard start.
    /// A simple endgame start lets a systems smoke produce completed games.
    #[serde(default)]
    pub start_fen: Option<String>,

    /// Label for the hardware scheduling profile this run resolved to
    /// (e.g. "hp-home"). Scheduling parameters may differ per machine; the
    /// scientific parameters above must not.
    #[serde(default)]
    pub hardware_profile: Option<String>,

    /// Label for the model profile this run used (e.g. "f15", "r15").
    #[serde(default)]
    pub model_profile: Option<String>,
}

impl RunConfig {
    pub fn from_toml_str(s: &str) -> anyhow::Result<Self> {
        Ok(toml::from_str(s)?)
    }

    /// Parse the device string into a typed device kind.
    pub fn device_kind(&self) -> anyhow::Result<DeviceKind> {
        match self.device.as_str() {
            "cpu" => Ok(DeviceKind::Cpu),
            "cuda" => Ok(DeviceKind::Cuda),
            other => anyhow::bail!("unknown device '{other}' (expected cpu or cuda)"),
        }
    }

    /// Parse the precision string into a typed precision.
    pub fn precision_kind(&self) -> anyhow::Result<Precision> {
        match self.precision.as_str() {
            "fp32" => Ok(Precision::Fp32),
            "bf16" => Ok(Precision::Bf16),
            "fp16" => Ok(Precision::Fp16),
            other => anyhow::bail!("unknown precision '{other}' (expected fp32, bf16 or fp16)"),
        }
    }

    /// Refuse a device/precision combination that has not been tested end-to-end.
    /// This is called on every run path (not only `bench`) so a BF16/FP16 request
    /// fails visibly instead of silently running something else.
    pub fn ensure_supported(&self) -> anyhow::Result<()> {
        recur64_model::precision::ensure_supported(self.precision_kind()?, self.device_kind()?)
    }
}
