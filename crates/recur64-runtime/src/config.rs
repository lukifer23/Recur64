//! Phase 2/3 run configuration. Every run records its fully resolved config.

use serde::{Deserialize, Serialize};

/// Stable contract for deriving a self-play game's RNG seed from its global
/// game id. This belongs in scientific identity because changing it changes
/// every generated trajectory after the first cycle.
pub const SELFPLAY_SEED_POLICY: &str = "base_seed_plus_global_game_id_v1";

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
fn default_min_decisive_games() -> u32 {
    4
}

/// Version of the conservative promotion rule implemented in `pilot.rs`. Part
/// of the scientific identity; bump it whenever the rule changes.
pub const PROMOTION_RULE_VERSION: &str = "conservative-v2:audit_ok,inference_errors=0,updates>0,finite_metrics,achieved_reuse>=0.8*target,decisive>=min_decisive,parent_score>0.5,parent_score>=floor";

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

/// One cycle's learner workload derived from the reuse target.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct UpdatePlan {
    pub requested_examples: f64,
    /// Updates the reuse target asks for (uncapped).
    pub requested_updates: usize,
    /// Updates actually scheduled (`min(requested, max_updates)`).
    pub scheduled_updates: usize,
    pub max_updates: usize,
    /// True when the `max_updates` safety cap reduced the requested work.
    pub cap_bound: bool,
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
    /// Legacy total-game count and concurrency request. When the new pair is
    /// absent, cpu_workers caps actual concurrency; migrate configs explicitly.
    #[serde(default = "default_active_games")]
    pub active_games: u32,
    /// Total games to collect in one cycle. Must be paired with concurrent_games.
    #[serde(default)]
    pub games_per_cycle: Option<u32>,
    /// Maximum simultaneous games. Must be paired with games_per_cycle.
    #[serde(default)]
    pub concurrent_games: Option<u32>,
    /// Maximum self-play worker threads. Applies to legacy and new configs.
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
    /// Arena score floor (candidate) for conservative promotion. It can only
    /// tighten the rule: a score must also be strictly above 0.5.
    #[serde(default = "default_score_floor")]
    pub promotion_score_floor: f64,
    /// Decisive (won or lost) parent-arena games required before a score can
    /// promote. A diagnostic floor, not an Elo sample size.
    #[serde(default = "default_min_decisive_games")]
    pub promotion_min_decisive_games: u32,
    /// Frozen reference checkpoint directory to start the pilot from (an
    /// operational path; identity is `reference_model_id`).
    #[serde(default)]
    pub reference_checkpoint: Option<String>,
    /// Expected content-hash model id of `reference_checkpoint`. Scientific.
    #[serde(default)]
    pub reference_model_id: Option<String>,

    // --- HP experiment ---
    /// Label for the hardware scheduling profile this run resolved to
    /// (e.g. "workstation-main"). Scheduling parameters may differ per machine; the
    /// scientific parameters above must not.
    #[serde(default)]
    pub hardware_profile: Option<String>,

    /// Label for the model profile this run used (e.g. "f10", "r10").
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

    /// Resolved `(games_total, concurrency)` for one collection.
    ///
    /// New configs supply `games_per_cycle` (total games) and `concurrent_games`
    /// (desired simultaneous games); `cpu_workers` caps the worker threads and
    /// the game count caps the concurrency.
    ///
    /// Legacy configs (both fields absent) keep the Phase 3 D24 semantics
    /// exactly: `active_games` is both the total and the concurrency, and
    /// `cpu_workers` is *not* applied. This preserves the behavior of the
    /// historical `configs/f10-*.toml` runs. Migrate a config explicitly by
    /// setting both new fields.
    pub fn collection_shape(&self) -> anyhow::Result<(u32, usize)> {
        let (games, concurrency) = match (self.games_per_cycle, self.concurrent_games) {
            (Some(g), Some(c)) => (g, (c as usize).min(self.cpu_workers).min(g as usize)),
            (None, None) => (self.active_games, self.active_games as usize),
            _ => anyhow::bail!("games_per_cycle and concurrent_games must be set together"),
        };
        anyhow::ensure!(
            games > 0 && concurrency > 0 && self.cpu_workers > 0,
            "collection counts and cpu_workers must be positive"
        );
        Ok((games, concurrency))
    }

    /// Reuse target schedules work from newly collected, completed-game plies.
    /// max_updates is a safety cap, never the default requested workload.
    pub fn reuse_updates(&self, new_trainable_positions: u64) -> anyhow::Result<(usize, f64)> {
        anyhow::ensure!(
            self.replay_reuse_target.is_finite() && self.replay_reuse_target > 0.0,
            "replay_reuse_target must be finite and positive"
        );
        let requested_examples = new_trainable_positions as f64 * self.replay_reuse_target;
        let requested = (requested_examples / self.effective_batch() as f64).ceil() as usize;
        Ok((requested.min(self.max_updates), requested_examples))
    }

    /// The full per-cycle update plan, including whether the `max_updates`
    /// safety cap binds. A binding cap means the cycle cannot reach the
    /// intended reuse target, so it must be reported, never hidden.
    pub fn update_plan(&self, new_trainable_positions: u64) -> anyhow::Result<UpdatePlan> {
        let (scheduled_updates, requested_examples) =
            self.reuse_updates(new_trainable_positions)?;
        let requested_updates =
            (requested_examples / self.effective_batch() as f64).ceil() as usize;
        Ok(UpdatePlan {
            requested_examples,
            requested_updates,
            scheduled_updates,
            max_updates: self.max_updates,
            cap_bound: requested_updates > scheduled_updates,
        })
    }

    /// Resolved warmup updates for the schedule.
    pub fn resolved_warmup(&self) -> u64 {
        self.lr_schedule().0
    }

    /// Resolved planned updates for the cosine schedule.
    pub fn resolved_planned_updates(&self) -> u64 {
        self.lr_schedule().1
    }

    /// `(warmup_updates, planned_updates)` used by every learner path and by
    /// the scientific hash. An explicit `planned_updates` is used as-is. When it
    /// is unset the legacy derivation spans the pilot (`max_updates * cycles`),
    /// which makes `cycles` part of the schedule for those configs.
    pub fn lr_schedule(&self) -> (u64, u64) {
        let planned = self
            .planned_updates
            .unwrap_or(self.max_updates as u64 * self.cycles.max(1) as u64)
            .max(1);
        let warmup = self
            .warmup_updates
            .unwrap_or((planned / 10).clamp(10, 1000));
        (warmup, planned)
    }

    /// Canonical digest of the configured opening suite, or `None` for the
    /// standard start. Fails if a requested suite cannot be loaded.
    pub fn opening_suite_digest(&self) -> anyhow::Result<Option<String>> {
        self.opening_suite
            .as_ref()
            .map(|p| Ok(recur64_eval::OpeningSuite::load(std::path::Path::new(p))?.digest()))
            .transpose()
    }

    /// Stable hash of the resolved config (for provenance).
    pub fn config_hash(&self) -> String {
        use sha2::{Digest, Sha256};
        let bytes = serde_json::to_vec(self).unwrap_or_default();
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        format!("{:x}", hasher.finalize())
    }

    pub fn resolved_config_hash(&self) -> String {
        self.config_hash()
    }

    /// The fields that define the experiment. Excluded on purpose: device,
    /// concurrency, cpu_workers, inference batch/timeout, hardware/model
    /// labels, run_id, cycles, wall/position budgets and the max_updates
    /// safety cap (a binding cap is reported as a reuse shortfall).
    pub fn scientific_identity(&self) -> anyhow::Result<serde_json::Value> {
        let (warmup, planned) = self.lr_schedule();
        Ok(serde_json::json!({
            "identity_version": 3,
            "model": self.model,
            "recurrence": self.recurrence,
            "precision": self.precision,
            "reference_model_id": self.reference_model_id,
            "seed": self.seed,
            "search": {
                "simulations_per_move": self.simulations_per_move,
                "c_puct": self.c_puct,
                "temperature": self.temperature,
                "argmax_after_ply": self.argmax_after_ply,
                "ply_cap": self.ply_cap,
                "start_fen": self.start_fen,
            },
            "collection": {
                "games_per_cycle": self.collection_shape()?.0,
                "seed_policy": SELFPLAY_SEED_POLICY,
            },
            "optimizer": recur64_model::train::OPTIMIZER_CONTRACT,
            "training": {
                "lr": self.lr,
                "effective_batch": self.effective_batch(),
                "warmup_updates": warmup,
                "planned_updates": planned,
                "replay_reuse_target": self.replay_reuse_target,
            },
            "replay": {
                "max_positions": self.replay_max_positions,
                "shard_max_games": self.shard_max_games,
                "sampler": "shard_recency_weight_index_plus_1_v1",
            },
            "evaluation": {
                "opening_suite_digest": self.opening_suite_digest()?,
                "arena_games": self.arena_games,
            },
            "promotion": {
                "snapshot_policy": self.snapshot_policy,
                "rule": PROMOTION_RULE_VERSION,
                "score_floor": self.promotion_score_floor,
                "min_decisive_games": self.promotion_min_decisive_games,
            },
        }))
    }

    /// Experiment identity hash (see [`Self::scientific_identity`]).
    pub fn scientific_config_hash(&self) -> anyhow::Result<String> {
        use sha2::{Digest, Sha256};
        Ok(format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&self.scientific_identity()?)?)
        ))
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
        recur64_model::precision::ensure_supported(self.precision_kind()?, self.device_kind()?)?;
        if self.device_kind()? == DeviceKind::Cuda {
            ensure_nvrtc_on_path()?;
        }
        Ok(())
    }
}

/// cudarc loads NVRTC lazily on a worker thread. When it is missing, the
/// panic is confined to that thread and JIT kernels silently do nothing, so a
/// "CUDA" run can finish and write garbage. Refuse to start instead.
/// True when a file name is any NVRTC shared library, across the versioned
/// names cudarc may try (`nvrtc.dll`, `nvrtc64_12.dll`, `nvrtc64_120_0.dll`,
/// `libnvrtc.so.12`, ...). Matching any NVRTC library avoids refusing a valid
/// user-space runtime over an exact-name mismatch.
fn is_nvrtc_library_name(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    name.contains("nvrtc") && (name.ends_with(".dll") || name.contains(".so"))
}

fn ensure_nvrtc_on_path() -> anyhow::Result<()> {
    let mut dirs: Vec<std::path::PathBuf> = std::env::var_os("PATH")
        .map(|p| std::env::split_paths(&p).collect())
        .unwrap_or_default();
    if let Some(cuda) = std::env::var_os("CUDA_PATH") {
        dirs.push(std::path::Path::new(&cuda).join("bin"));
    }
    let found = dirs.iter().any(|d| {
        std::fs::read_dir(d).is_ok_and(|entries| {
            entries
                .filter_map(Result::ok)
                .any(|e| is_nvrtc_library_name(&e.file_name().to_string_lossy()))
        })
    });
    anyhow::ensure!(
        found,
        "CUDA requested but NVRTC (nvrtc*.dll / libnvrtc.so*) is not on PATH or \
         CUDA_PATH\\bin. Set CUDA_PATH and PATH for this process (see \
         docs/HARDWARE.md and docs/DECISIONS.md D3, user-space CUDA 12.9.1). \
         Refusing to run: without NVRTC the JIT kernels silently no-op."
    );
    Ok(())
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
        // Legacy schedule spans the pilot: max_updates=200 x cycles=4 = 800
        // planned updates (what pilot.rs trained with), warmup ~10% = 80.
        assert_eq!(cfg.resolved_planned_updates(), 800);
        assert_eq!(cfg.resolved_warmup(), 80);
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

    #[test]
    fn collection_counts_and_scientific_hash_are_independent_of_scheduling() {
        let mut cfg = RunConfig::from_toml_str(base_toml()).unwrap();
        cfg.games_per_cycle = Some(64);
        cfg.concurrent_games = Some(12);
        cfg.cpu_workers = 12;
        assert_eq!(cfg.collection_shape().unwrap(), (64, 12));
        let scientific = cfg.scientific_config_hash().unwrap();
        let resolved = cfg.resolved_config_hash();
        cfg.concurrent_games = Some(8);
        cfg.cpu_workers = 8;
        cfg.max_inference_batch = 64;
        cfg.batch_timeout_us = 2000;
        cfg.device = "cuda".into();
        cfg.hardware_profile = Some("other-machine".into());
        cfg.run_budget_minutes = 90;
        cfg.position_budget = Some(1234);
        cfg.run_id = "renamed".into();
        assert_eq!(cfg.scientific_config_hash().unwrap(), scientific);
        assert_ne!(cfg.resolved_config_hash(), resolved);
        cfg.concurrent_games = None;
        assert!(cfg.collection_shape().is_err());
    }

    #[test]
    fn scientific_hash_tracks_search_training_promotion_and_suite_content() {
        let mut base = RunConfig::from_toml_str(base_toml()).unwrap();
        base.games_per_cycle = Some(16);
        base.concurrent_games = Some(12);
        base.planned_updates = Some(200);
        base.warmup_updates = Some(20);
        let h = base.scientific_config_hash().unwrap();

        // Explicit schedule: cycles is run length only.
        let mut c = base.clone();
        c.cycles = 4;
        assert_eq!(c.scientific_config_hash().unwrap(), h);

        type Mutation = Box<dyn Fn(&mut RunConfig)>;
        let mutations: Vec<(&str, Mutation)> = vec![
            ("sims", Box::new(|c| c.simulations_per_move = 32)),
            ("c_puct", Box::new(|c| c.c_puct = 1.5)),
            ("ply_cap", Box::new(|c| c.ply_cap = 300)),
            ("lr", Box::new(|c| c.lr = 1e-4)),
            ("effective_batch", Box::new(|c| c.accumulation_steps = 8)),
            ("planned", Box::new(|c| c.planned_updates = Some(300))),
            ("reuse", Box::new(|c| c.replay_reuse_target = 3.0)),
            (
                "games_per_cycle",
                Box::new(|c| c.games_per_cycle = Some(24)),
            ),
            ("shard_max_games", Box::new(|c| c.shard_max_games = 64)),
            (
                "min_decisive",
                Box::new(|c| c.promotion_min_decisive_games = 2),
            ),
            (
                "snapshot",
                Box::new(|c| c.snapshot_policy = SnapshotPolicy::FrozenReference),
            ),
            (
                "reference",
                Box::new(|c| c.reference_model_id = Some("abc".into())),
            ),
        ];
        for (name, mutate) in mutations {
            let mut c = base.clone();
            mutate(&mut c);
            assert_ne!(
                c.scientific_config_hash().unwrap(),
                h,
                "{name} must change identity"
            );
        }
        // Equal effective batch from a different physical split is hardware.
        let mut c = base.clone();
        c.train_batch = 16;
        c.accumulation_steps = 8;
        assert_eq!(c.scientific_config_hash().unwrap(), h);

        // Same suite path, different contents -> different identity.
        let path =
            std::env::temp_dir().join(format!("recur64-suite-id-{}.toml", std::process::id()));
        let start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        let e4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1";
        std::fs::write(
            &path,
            format!(
                "version = 1
provenance = 'a'
openings = ['{start}']
"
            ),
        )
        .unwrap();
        let mut c = base.clone();
        c.opening_suite = Some(path.to_string_lossy().into_owned());
        let h1 = c.scientific_config_hash().unwrap();
        std::fs::write(
            &path,
            format!(
                "version = 1
provenance = 'b'
openings = ['{start}']
"
            ),
        )
        .unwrap();
        assert_eq!(
            c.scientific_config_hash().unwrap(),
            h1,
            "provenance text is not content"
        );
        std::fs::write(
            &path,
            format!(
                "version = 1
provenance = 'a'
openings = ['{e4}']
"
            ),
        )
        .unwrap();
        assert_ne!(
            c.scientific_config_hash().unwrap(),
            h1,
            "suite content must change identity"
        );
        std::fs::remove_file(&path).unwrap();
        assert!(
            c.scientific_config_hash().is_err(),
            "missing suite must fail identity"
        );
    }

    #[test]
    fn reuse_target_schedules_updates_and_cap_is_explicit() {
        let mut cfg = RunConfig::from_toml_str(base_toml()).unwrap();
        cfg.train_batch = 8;
        cfg.accumulation_steps = 4;
        cfg.replay_reuse_target = 2.0;
        cfg.max_updates = 100;
        assert_eq!(cfg.reuse_updates(65).unwrap(), (5, 130.0));
        cfg.max_updates = 3;
        assert_eq!(cfg.reuse_updates(65).unwrap().0, 3);
    }

    #[test]
    fn update_plan_reports_a_binding_cap() {
        let mut cfg = RunConfig::from_toml_str(base_toml()).unwrap();
        cfg.train_batch = 8;
        cfg.accumulation_steps = 4;
        cfg.replay_reuse_target = 2.0;
        cfg.max_updates = 5;
        let fits = cfg.update_plan(65).unwrap();
        assert_eq!((fits.requested_updates, fits.scheduled_updates), (5, 5));
        assert!(!fits.cap_bound, "cap equal to the request does not bind");
        cfg.max_updates = 3;
        let capped = cfg.update_plan(65).unwrap();
        assert_eq!((capped.requested_updates, capped.scheduled_updates), (5, 3));
        assert!(capped.cap_bound);
        assert_eq!(capped.requested_examples, 130.0);
    }

    /// Legacy configs (no explicit pair) keep the Phase 3 D24 behavior:
    /// `active_games` is both the total and the concurrency, and `cpu_workers`
    /// is not applied. New explicit pairs are capped by `cpu_workers`.
    #[test]
    fn legacy_collection_shape_preserves_d24_and_new_pair_is_capped() {
        let mut cfg = RunConfig::from_toml_str(base_toml()).unwrap();
        cfg.active_games = 64;
        cfg.cpu_workers = 8;
        assert_eq!(
            cfg.collection_shape().unwrap(),
            (64, 64),
            "legacy configs must not be silently shrunk to cpu_workers"
        );
        cfg.games_per_cycle = Some(64);
        cfg.concurrent_games = Some(64);
        assert_eq!(
            cfg.collection_shape().unwrap(),
            (64, 8),
            "explicit concurrency is capped by cpu_workers"
        );
    }

    /// The frozen mainline reference config must parse and freeze its science.
    #[test]
    fn f10_reference_config_is_frozen_and_valid() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../configs/phase4/f10-reference.toml");
        let mut cfg = RunConfig::from_toml_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        cfg.opening_suite = Some(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../configs/openings-v1.toml")
                .to_string_lossy()
                .into_owned(),
        );
        assert_eq!(cfg.model.unique_blocks(), 8);
        assert_eq!(cfg.recurrence, 1);
        assert_eq!(
            cfg.collection_shape().unwrap().0,
            cfg.games_per_cycle.unwrap()
        );
        assert!(!cfg.scientific_config_hash().unwrap().is_empty());
        assert!(!cfg.resolved_config_hash().is_empty());
    }

    /// The CUDA fail-fast must accept every versioned NVRTC shared library name
    /// cudarc may try, not just one exact name.
    #[test]
    fn nvrtc_library_names_are_recognized_across_versions() {
        assert!(is_nvrtc_library_name("nvrtc.dll"));
        assert!(is_nvrtc_library_name("nvrtc64.dll"));
        assert!(is_nvrtc_library_name("nvrtc64_12.dll"));
        assert!(is_nvrtc_library_name("nvrtc64_120_0.dll"));
        assert!(is_nvrtc_library_name("libnvrtc.so.12"));
        assert!(is_nvrtc_library_name("libnvrtc.so"));
        assert!(!is_nvrtc_library_name("cudart64_12.dll"));
        assert!(!is_nvrtc_library_name("cublas64_12.dll"));
        assert!(!is_nvrtc_library_name("nvJitLink_120_0.dll"));
    }
}
