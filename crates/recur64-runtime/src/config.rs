//! Phase 2/3 run configuration. Every run records its fully resolved config.

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
fn default_cycles() -> u32 {
    1
}
fn default_replay_max_positions() -> u64 {
    100_000
}
fn default_replay_reuse_target() -> f64 {
    2.0
}
fn default_accumulation_steps() -> usize {
    4
}

/// How the self-play snapshot is chosen between cycles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum SnapshotPolicy {
    /// The candidate becomes the next self-play snapshot only if all health
    /// gates pass and the arena score is not below `promotion_score_floor`.
    #[default]
    Conservative,
    /// Always keep the initial reference as the self-play snapshot.
    FrozenReference,
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
    /// Phase 3 baselines use the standard start.
    #[serde(default)]
    pub start_fen: Option<String>,

    // --- Phase 3 ---
    /// Number of bounded collect/train/evaluate cycles in a `pilot` run.
    #[serde(default = "default_cycles")]
    pub cycles: u32,
    /// Optional total-position budget across the pilot (stop when reached).
    #[serde(default)]
    pub position_budget: Option<u64>,
    /// Bounded replay capacity. Oldest shards are archived when exceeded.
    #[serde(default = "default_replay_max_positions")]
    pub replay_max_positions: u64,
    /// Target `examples_consumed / new_positions_inserted`.
    #[serde(default = "default_replay_reuse_target")]
    pub replay_reuse_target: f64,
    /// LR warmup updates. `None` scales to the budget (`min(1000, ~10%)`).
    #[serde(default)]
    pub warmup_updates: Option<u64>,
    /// Planned total updates for the cosine schedule. `None` derives it.
    #[serde(default)]
    pub planned_updates: Option<u64>,
    /// Gradient-accumulation steps (effective batch = train_batch * this).
    #[serde(default = "default_accumulation_steps")]
    pub accumulation_steps: usize,
    /// If set, use argmax (temperature 0) after this ply. `None` = sample all
    /// plies at `temperature` (the Phase 3 baseline convention).
    #[serde(default)]
    pub argmax_after_ply: Option<u32>,
    /// Path to a frozen opening suite used only for evaluation.
    #[serde(default)]
    pub opening_suite: Option<String>,
    /// Snapshot selection policy between cycles.
    #[serde(default)]
    pub snapshot_policy: SnapshotPolicy,
    /// Arena score floor (candidate) for conservative promotion.
    #[serde(default = "default_score_floor")]
    pub promotion_score_floor: f64,

    // --- HP experiment ---
    /// Label for the hardware scheduling profile this run resolved to
    /// (e.g. "hp-home"). Scheduling parameters may differ per machine; the
    /// scientific parameters above must not.
    #[serde(default)]
    pub hardware_profile: Option<String>,

    /// Label for the model profile this run used (e.g. "f15", "r15").
    #[serde(default)]
    pub model_profile: Option<String>,
}

fn default_score_floor() -> f64 {
    0.35
}

impl RunConfig {
    pub fn from_toml_str(s: &str) -> anyhow::Result<Self> {
        Ok(toml::from_str(s)?)
    }

    /// Effective training batch = physical batch * accumulation steps.
    pub fn effective_batch(&self) -> usize {
        self.train_batch.max(1) * self.accumulation_steps.max(1)
    }

    /// Resolved warmup updates for the schedule.
    pub fn resolved_warmup(&self) -> u64 {
        self.warmup_updates.unwrap_or_else(|| {
            let planned = self.planned_updates.unwrap_or(self.max_updates as u64);
            (planned / 10).clamp(10, 1000)
        })
    }

    /// Resolved planned updates for the cosine schedule.
    pub fn resolved_planned_updates(&self) -> u64 {
        self.planned_updates
            .unwrap_or(self.max_updates as u64)
            .max(1)
    }

    /// Stable hash of the resolved config (for provenance).
    pub fn config_hash(&self) -> String {
        use sha2::{Digest, Sha256};
        let bytes = serde_json::to_vec(self).unwrap_or_default();
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        format!("{:x}", hasher.finalize())
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

#[cfg(test)]
mod tests {
    use super::*;

    fn base_toml() -> &'static str {
        r#"
run_id = "t"
device = "cpu"
precision = "fp32"
[model]
width = 32
heads = 4
ffn = 64
input_blocks = 0
core_blocks = 1
output_blocks = 0
"#
    }

    fn f10_toml() -> &'static str {
        r#"
run_id = "t"
device = "cuda"
recurrence = 1
simulations_per_move = 64
temperature = 1.0
cycles = 4
replay_max_positions = 100000
replay_reuse_target = 2.0
train_batch = 64
accumulation_steps = 4
max_updates = 200
arena_games = 50
snapshot_policy = "conservative"

[model]
width = 384
heads = 12
ffn = 768
input_blocks = 0
core_blocks = 8
output_blocks = 0
"#
    }

    #[test]
    fn parses_phase3_fields_and_defaults() {
        let cfg = RunConfig::from_toml_str(f10_toml()).unwrap();
        assert_eq!(cfg.cycles, 4);
        assert_eq!(cfg.replay_max_positions, 100_000);
        assert_eq!(cfg.effective_batch(), 256);
        assert_eq!(cfg.snapshot_policy, SnapshotPolicy::Conservative);
        // Omitted -> defaults.
        assert_eq!(cfg.precision, "fp32");
        assert!(cfg.start_fen.is_none());
        assert!(cfg.argmax_after_ply.is_none());
        // Warmup scales to ~10% of planned updates (max_updates=200 -> 20).
        assert_eq!(cfg.resolved_warmup(), 20);
        assert_eq!(cfg.resolved_planned_updates(), 200);
        assert!(!cfg.config_hash().is_empty());
    }

    #[test]
    fn warmup_scaling_is_bounded() {
        let mut cfg = RunConfig::from_toml_str(f10_toml()).unwrap();
        cfg.max_updates = 100_000;
        assert_eq!(cfg.resolved_warmup(), 1000); // capped
        cfg.max_updates = 10;
        assert_eq!(cfg.resolved_warmup(), 10); // floored
    }

    #[test]
    fn device_and_precision_parse() {
        let mut cfg = RunConfig::from_toml_str(base_toml()).unwrap();
        assert_eq!(cfg.device_kind().unwrap(), DeviceKind::Cpu);
        assert_eq!(cfg.precision_kind().unwrap(), Precision::Fp32);
        cfg.device = "cuda".into();
        assert_eq!(cfg.device_kind().unwrap(), DeviceKind::Cuda);
        cfg.device = "bogus".into();
        assert!(cfg.device_kind().is_err());
    }

    #[test]
    fn gate_rejects_untested_precision_visibly() {
        let mut cfg = RunConfig::from_toml_str(base_toml()).unwrap();
        assert!(cfg.ensure_supported().is_ok(), "fp32 must be accepted");
        cfg.precision = "bf16".into();
        assert!(cfg.ensure_supported().is_err(), "bf16 must be refused");
        cfg.precision = "fp16".into();
        assert!(cfg.ensure_supported().is_err(), "fp16 must be refused");
    }

    #[test]
    fn schedule_override_fields_round_trip() {
        let cfg = RunConfig::from_toml_str(base_toml()).unwrap();
        assert_eq!(cfg.active_games, 32);
        assert_eq!(cfg.cpu_workers, 8);
        assert_eq!(cfg.max_inference_batch, 32);
        assert_eq!(cfg.batch_timeout_us, 500);
    }
}
