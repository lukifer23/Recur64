//! Phase 2/3 run configuration. Every run records its fully resolved config.

use serde::{Deserialize, Serialize};

/// Stable contract for deriving a self-play game's RNG seed from its global
/// game id. This belongs in scientific identity because changing it changes
/// every generated trajectory after the first cycle.
pub const SELFPLAY_SEED_POLICY: &str = "base_seed_plus_global_game_id_v1";

use recur64_eval::{ArenaEarlyAdjudication, ArenaRngPolicy, ArenaTreePolicy};
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
fn default_root_dirichlet_alpha() -> f32 {
    0.3
}
fn default_leaves_in_flight() -> u32 {
    1
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

/// Which arena score conservative promotion reads (R15-P0.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum PromotionScore {
    /// conservative-v2: `candidate_score` over non-truncated games.
    #[default]
    PerGameV1,
    /// promotion-v3: the `material_v1` adjudicated score over every game, so
    /// failing to convert can neither help nor hide.
    AdjudicatedMaterialV1,
}

impl PromotionScore {
    pub fn is_default(&self) -> bool {
        *self == Self::PerGameV1
    }
}

/// promotion-v3 rule text (recorded when `promotion_score` is adjudicated).
pub const PROMOTION_RULE_V3: &str = "conservative-v3:audit_ok,inference_errors=0,updates>0,finite_metrics,achieved_reuse>=0.8*target,adjudicated_decisive>=min_decisive,adjudicated_score>0.5,adjudicated_score>=floor;adjudication=material_v1";

/// What happens to the learner state when a candidate is held (D48).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TrainerPolicy {
    /// D31: a held candidate is discarded; the next cycle trains again from
    /// the last promoted snapshot's weights and optimizer state.
    #[default]
    DiscardHeld,
    /// AlphaGo Zero style: the learner keeps its weights and optimizer state
    /// across cycles whether or not the candidate is promoted. Promotion only
    /// decides which network generates self-play and is the arena parent.
    Continuous,
}

/// Self-play health thresholds that stop a pilot at a cycle boundary (D49).
/// `None` disables a check. These bound execution; they never change what a
/// cycle does, so they are excluded from the scientific identity.
///
/// Unknown keys are refused: a misspelled or misplaced threshold must not
/// silently disable a stop (H3.5B found a smoke config whose checks were off).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct HealthStops {
    /// Stop when the self-play draw share reaches this in two consecutive
    /// cycles.
    #[serde(default)]
    pub draw_share_two_cycles: Option<f64>,
    /// Stop when (threefold + fifty-move) / games reaches this in any cycle.
    #[serde(default)]
    pub threefold_fifty: Option<f64>,
    /// Stop when truncated / games reaches this in any cycle.
    #[serde(default)]
    pub truncation: Option<f64>,
    /// H3.5B: stop at the cycle boundary once the trainer's global step
    /// reaches `planned_updates` (the cosine LR is 0 from there on), so no
    /// cycle trains at LR = 0 as if it were normal learning. Omitted from the
    /// serialized config when unset, so earlier resolved hashes are unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lr_schedule_end: Option<bool>,
}

impl HealthStops {
    /// The reason to stop because the LR schedule is exhausted, if enabled.
    pub fn check_schedule(&self, step_end: u64, planned_updates: u64) -> Option<String> {
        (self.lr_schedule_end == Some(true) && step_end >= planned_updates).then(|| {
            format!("lr_schedule_exhausted: trainer step {step_end} >= planned {planned_updates}")
        })
    }

    /// The reason to stop after a cycle, given this cycle's shares and
    /// whether the previous cycle's draw share already crossed the threshold.
    pub fn check(
        &self,
        draw_share: f64,
        threefold_fifty: f64,
        truncation: f64,
        previous_draw_share_high: bool,
    ) -> Option<String> {
        if let Some(t) = self.threefold_fifty.filter(|t| threefold_fifty >= *t) {
            return Some(format!("threefold_fifty {threefold_fifty:.3} >= {t}"));
        }
        if let Some(t) = self.truncation.filter(|t| truncation >= *t) {
            return Some(format!("truncation {truncation:.3} >= {t}"));
        }
        if let Some(t) = self
            .draw_share_two_cycles
            .filter(|t| draw_share >= *t && previous_draw_share_high)
        {
            return Some(format!("draw_share {draw_share:.3} >= {t} for two cycles"));
        }
        None
    }

    /// Whether this cycle's draw share crosses the two-cycle threshold.
    pub fn draw_share_high(&self, draw_share: f64) -> bool {
        self.draw_share_two_cycles.is_some_and(|t| draw_share >= t)
    }
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
    /// Evaluation scheduling (execution only, excluded from the scientific
    /// identity): games played at once in pilot evaluation matches, and the
    /// evaluation owners' batch cap. `None` keeps the historical schedule
    /// (self-play concurrency and `max_inference_batch`). Omitted when unset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eval_concurrency: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eval_max_inference_batch: Option<usize>,
    /// Round inference candidate widths up to fixed buckets (D55 perf pass;
    /// execution only: outputs equal to float noise, excluded from identity).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub inference_candidate_buckets: bool,
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
    /// Root Dirichlet noise concentration for self-play (AlphaZero chess:
    /// 0.3). Only used when `root_dirichlet_epsilon > 0`. Searched arenas
    /// reuse this alpha when `arena_root_dirichlet_epsilon > 0` (D45).
    #[serde(default = "default_root_dirichlet_alpha")]
    pub root_dirichlet_alpha: f32,
    /// Root noise mixing weight for self-play; `0.0` (default) disables it.
    #[serde(default)]
    pub root_dirichlet_epsilon: f32,
    /// Searched-arena opening phase (D45): sample from visit counts at
    /// temperature 1 for this many plies after the opening, then argmax.
    /// `None` (default) = argmax from ply 0, the original arena contract.
    #[serde(default)]
    pub arena_sample_plies: Option<u32>,
    /// Searched-arena root noise weight (alpha = `root_dirichlet_alpha`).
    /// `0.0` (default) = noise-free, the original arena contract.
    #[serde(default)]
    pub arena_root_dirichlet_epsilon: f32,
    /// Arena per-game seed derivation (H3.5B). The default `per_game_v1` is
    /// the historical contract; it is omitted from the serialized config so
    /// the resolved hash of every earlier config is unchanged.
    #[serde(default, skip_serializing_if = "ArenaRngPolicy::is_default")]
    pub arena_rng_policy: ArenaRngPolicy,
    /// Arena search-tree routing (D54). The historical `per_node_side_v1`
    /// mixes both networks inside every search; `root_player_v1` gives each
    /// player its own tree. Omitted from the serialized config when default.
    #[serde(default, skip_serializing_if = "ArenaTreePolicy::is_default")]
    pub arena_tree_policy: ArenaTreePolicy,
    /// D56 early material adjudication in arenas. `shadow` observes only and
    /// is identity-neutral; `enforce` is a new scientific identity.
    #[serde(default, skip_serializing_if = "ArenaEarlyAdjudication::is_default")]
    pub arena_early_adjudication: ArenaEarlyAdjudication,
    /// Leaves selected with virtual loss and evaluated together per search
    /// round (D47), for self-play and arenas alike. `1` (default) is the
    /// original one-leaf search; values above 1 are a new identity.
    #[serde(default = "default_leaves_in_flight")]
    pub search_leaves_in_flight: u32,
    /// Learner continuity across held cycles (D48). The default keeps D31.
    #[serde(default)]
    pub trainer_policy: TrainerPolicy,
    /// Pilot health stops (execution bounds, not scientific identity): stop
    /// at a cycle boundary when self-play drifts into an attractor.
    #[serde(default)]
    pub health_stops: HealthStops,
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
    /// Arena score read by conservative promotion (R15-P0.1). The default is
    /// conservative-v2 and is omitted from the serialized config.
    #[serde(default, skip_serializing_if = "PromotionScore::is_default")]
    pub promotion_score: PromotionScore,
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

    /// Evaluation identity. The arena exploration fields (D45) appear only
    /// when they differ from the original deterministic, noise-free arena, so
    /// every configuration recorded before D45 keeps a reproducible hash.
    fn evaluation_identity(&self) -> anyhow::Result<serde_json::Value> {
        let mut v = serde_json::json!({
            "opening_suite_digest": self.opening_suite_digest()?,
            "arena_games": self.arena_games,
        });
        if self.arena_sample_plies.is_some() || self.arena_root_dirichlet_epsilon != 0.0 {
            v["arena_exploration"] = serde_json::json!({
                "sample_plies": self.arena_sample_plies,
                "root_dirichlet_alpha": self.root_dirichlet_alpha,
                "root_dirichlet_epsilon": self.arena_root_dirichlet_epsilon,
            });
        }
        // H3.5B: recorded only when it differs from the historical per-game
        // seeds, so earlier identities are unchanged.
        if !self.arena_rng_policy.is_default() {
            v["arena_rng_policy"] = serde_json::to_value(self.arena_rng_policy)?;
        }
        // D54: recorded only when it differs from the historical routing.
        if !self.arena_tree_policy.is_default() {
            v["arena_tree_policy"] = serde_json::to_value(self.arena_tree_policy)?;
        }
        // D56: only enforcement changes results; shadow mode only observes.
        if self.arena_early_adjudication == ArenaEarlyAdjudication::Enforce {
            v["arena_early_adjudication"] = serde_json::json!("early_material_v1:+5x40");
        }
        Ok(v)
    }

    /// The searched-arena configuration for one evaluation (seed offset
    /// `offset`, e.g. the cycle), from this run's search and arena contract.
    pub fn arena_config(
        &self,
        offset: u64,
        openings: Vec<String>,
        concurrency: usize,
    ) -> recur64_eval::ArenaConfig {
        recur64_eval::ArenaConfig {
            games: self.arena_games,
            simulations: self.simulations_per_move,
            c_puct: self.c_puct,
            recurrence: self.recurrence,
            ply_cap: self.ply_cap,
            seed: self.seed.wrapping_add(offset),
            openings,
            concurrency,
            sample_plies: self.arena_sample_plies,
            root_dirichlet_alpha: self.root_dirichlet_alpha,
            root_dirichlet_epsilon: self.arena_root_dirichlet_epsilon,
            deadline: None,
            leaves_in_flight: self.search_leaves_in_flight,
            rng_policy: self.arena_rng_policy,
            tree_policy: self.arena_tree_policy,
            early_adjudication: self.arena_early_adjudication,
        }
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
        let mut identity = serde_json::json!({
            "identity_version": 4,
            "model": self.model,
            "model_head_version": recur64_model::model::HEAD_VERSION,
            "recurrence": self.recurrence,
            "precision": self.precision,
            "reference_model_id": self.reference_model_id,
            "seed": self.seed,
            "search": {
                "simulations_per_move": self.simulations_per_move,
                "c_puct": self.c_puct,
                "temperature": self.temperature,
                "argmax_after_ply": self.argmax_after_ply,
                "root_dirichlet_alpha": self.root_dirichlet_alpha,
                "root_dirichlet_epsilon": self.root_dirichlet_epsilon,
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
            "evaluation": self.evaluation_identity()?,
            "promotion": {
                "snapshot_policy": self.snapshot_policy,
                "rule": self.promotion_rule(),
                "score_floor": self.promotion_score_floor,
                "min_decisive_games": self.promotion_min_decisive_games,
            },
        });
        // Multi-leaf search (D47) enters the identity only when enabled, so
        // every pre-D47 identity stays reproducible.
        if self.search_leaves_in_flight > 1 {
            identity["search_execution"] = serde_json::json!({
                "leaves_in_flight": self.search_leaves_in_flight,
                "virtual_loss": 1.0,
            });
        }
        // Continuous training (D48) is recorded only when enabled, so every
        // pre-D48 identity stays reproducible.
        if self.trainer_policy != TrainerPolicy::DiscardHeld {
            identity["trainer_policy"] = serde_json::to_value(self.trainer_policy)?;
        }
        Ok(identity)
    }

    /// This config with the evaluation owners' batch cap applied (scheduling
    /// only; see `eval_max_inference_batch`).
    pub fn for_evaluation(&self) -> RunConfig {
        let mut c = self.clone();
        if let Some(b) = self.eval_max_inference_batch {
            c.max_inference_batch = b;
        }
        c
    }

    /// The promotion rule text this config runs (v2 by default, v3 when the
    /// adjudicated score is selected).
    pub fn promotion_rule(&self) -> &'static str {
        match self.promotion_score {
            PromotionScore::PerGameV1 => PROMOTION_RULE_VERSION,
            PromotionScore::AdjudicatedMaterialV1 => PROMOTION_RULE_V3,
        }
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
        anyhow::ensure!(
            self.search_leaves_in_flight >= 1,
            "search_leaves_in_flight must be >= 1"
        );
        anyhow::ensure!(
            (0.0..=1.0).contains(&self.root_dirichlet_epsilon)
                && (0.0..=1.0).contains(&self.arena_root_dirichlet_epsilon),
            "root_dirichlet_epsilon and arena_root_dirichlet_epsilon must be in [0, 1]"
        );
        anyhow::ensure!(
            (self.root_dirichlet_epsilon == 0.0 && self.arena_root_dirichlet_epsilon == 0.0)
                || (self.root_dirichlet_alpha > 0.0 && self.root_dirichlet_alpha.is_finite()),
            "root_dirichlet_alpha must be finite and > 0 when root noise is enabled"
        );
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
    fn health_stops_trigger_on_the_preregistered_conditions() {
        let h = HealthStops {
            draw_share_two_cycles: Some(0.85),
            threefold_fifty: Some(0.60),
            truncation: Some(0.25),
            lr_schedule_end: None,
        };
        assert_eq!(
            h.check(0.73, 0.33, 0.01, false),
            None,
            "smoke v2 cycle 3 is healthy"
        );
        assert_eq!(
            h.check(0.90, 0.10, 0.0, false),
            None,
            "one high-draw cycle is not enough"
        );
        assert!(
            h.check(0.90, 0.10, 0.0, true)
                .unwrap()
                .contains("two cycles")
        );
        assert!(
            h.check(0.5, 0.61, 0.0, false)
                .unwrap()
                .contains("threefold_fifty")
        );
        assert!(
            h.check(0.5, 0.1, 0.30, false)
                .unwrap()
                .contains("truncation")
        );
        assert!(h.draw_share_high(0.85) && !h.draw_share_high(0.84));
        assert_eq!(
            HealthStops::default().check(1.0, 1.0, 1.0, true),
            None,
            "off by default"
        );
        // Health stops are execution bounds: they never enter the identity.
        let base = RunConfig::from_toml_str(base_toml()).unwrap();
        let mut with = base.clone();
        with.health_stops = h;
        assert_eq!(
            base.scientific_config_hash().unwrap(),
            with.scientific_config_hash().unwrap()
        );
        // H3.5B schedule guard: off unless enabled, stops at the endpoint.
        assert_eq!(h.check_schedule(10_000, 370), None, "off unless enabled");
        let g = HealthStops {
            lr_schedule_end: Some(true),
            ..h
        };
        assert_eq!(g.check_schedule(369, 370), None);
        assert!(
            g.check_schedule(370, 370)
                .unwrap()
                .contains("lr_schedule_exhausted")
        );
        with.health_stops = g;
        assert_eq!(
            base.scientific_config_hash().unwrap(),
            with.scientific_config_hash().unwrap()
        );
        // Unset, it stays out of the serialized (resolved) config.
        assert!(
            !serde_json::to_string(&base)
                .unwrap()
                .contains("lr_schedule_end")
        );
        // A run-level key misplaced under [model] is refused (H3.5 ran K = 1).
        let misplaced = base_toml().replace(
            "[model]",
            "[model]
search_leaves_in_flight = 2",
        );
        assert!(RunConfig::from_toml_str(&misplaced).is_err());
        // A misspelled key is refused rather than silently disabling a stop.
        let bad = format!(
            "{}
[health_stops]
truncaton = 0.25
",
            base_toml()
        );
        assert!(RunConfig::from_toml_str(&bad).is_err());
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

    /// The corrected Phase 4 smoke realizes the measured 32-way schedule
    /// (cpu_workers must not silently cap it) and its max_updates safety cap
    /// does not bind at the pre-registered expected workload.
    #[test]
    fn f10_smoke_config_realizes_schedule_and_cap_does_not_bind() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../configs/phase4/f10-smoke.toml");
        let cfg = RunConfig::from_toml_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(cfg.collection_shape().unwrap(), (32, 32));
        assert_eq!(cfg.effective_batch(), 256);
        assert_eq!(cfg.simulations_per_move, 64);
        assert_eq!(cfg.argmax_after_ply, Some(30));
        assert_eq!(cfg.root_dirichlet_epsilon, 0.25);
        assert_eq!(cfg.lr_schedule(), (10, 90));
        assert!(cfg.reference_model_id.is_some() && cfg.reference_checkpoint.is_some());
        // Pre-registered expectation: ~5,685 new trainable positions/cycle.
        let plan = cfg.update_plan(5685).unwrap();
        assert_eq!(plan.requested_updates, 45);
        assert!(!plan.cap_bound);
        // Even a 3x larger cycle would still fit under the cap.
        assert!(!cfg.update_plan(3 * 5685).unwrap().cap_bound);
    }

    /// Smoke v2 realizes the adopted post-smoke contract (D45 arena, D47
    /// K = 2, D48 continuous trainer) and its safety cap does not bind.
    #[test]
    fn f10_smoke_v2_config_is_the_adopted_contract() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../configs/phase4/f10-smoke-v2.toml");
        let cfg = RunConfig::from_toml_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(cfg.collection_shape().unwrap(), (64, 32));
        assert_eq!(cfg.search_leaves_in_flight, 2);
        assert_eq!(cfg.max_inference_batch, 64);
        assert_eq!(cfg.trainer_policy, TrainerPolicy::Continuous);
        assert_eq!(cfg.arena_sample_plies, Some(30));
        assert_eq!(cfg.arena_root_dirichlet_epsilon, 0.25);
        assert_eq!(cfg.lr_schedule(), (27, 267));
        let plan = cfg.update_plan(11_370).unwrap();
        assert_eq!(plan.requested_updates, 89);
        assert!(!plan.cap_bound && !cfg.update_plan(3 * 11_370).unwrap().cap_bound);
        let arena = cfg.arena_config(0, Vec::new(), 32);
        assert_eq!(arena.leaves_in_flight, 2);
    }

    /// P4.6 qualification: the smoke v2 contract over 10 cycles with the
    /// pre-registered health stops parsed and a non-binding cap.
    #[test]
    fn f10_qual_config_is_smoke_v2_contract_with_health_stops() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/phase4");
        let qual =
            RunConfig::from_toml_str(&std::fs::read_to_string(dir.join("f10-qual.toml")).unwrap())
                .unwrap();
        let smoke = RunConfig::from_toml_str(
            &std::fs::read_to_string(dir.join("f10-smoke-v2.toml")).unwrap(),
        )
        .unwrap();
        assert_eq!(qual.cycles, 10);
        assert_eq!(qual.health_stops.draw_share_two_cycles, Some(0.85));
        assert_eq!(qual.health_stops.threefold_fifty, Some(0.60));
        assert_eq!(qual.health_stops.truncation, Some(0.25));
        assert_eq!(qual.lr_schedule(), (110, 1100));
        assert!(!qual.update_plan(3 * 16_215).unwrap().cap_bound);
        // Same search, arena, learner and trainer contract as smoke v2.
        assert_eq!(qual.search_leaves_in_flight, smoke.search_leaves_in_flight);
        assert_eq!(qual.trainer_policy, smoke.trainer_policy);
        assert_eq!(qual.arena_sample_plies, smoke.arena_sample_plies);
        assert_eq!(qual.simulations_per_move, smoke.simulations_per_move);
        assert_eq!(
            qual.collection_shape().unwrap(),
            smoke.collection_shape().unwrap()
        );
        assert_eq!(qual.reference_model_id, smoke.reference_model_id);
    }

    /// H3.5B P5: the corrected F15 smoke is pinned field by field, and its
    /// per-cycle safety cap cannot bind for the expected, stress or
    /// theoretical-maximum workload (32 games x 400-ply cap).
    #[test]
    fn f15_smoke_v2_config_is_the_h3_contract() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../configs/hp/f15-smoke-v2.toml");
        let cfg = RunConfig::from_toml_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        // Model: F15 512/8/768, 0 + 8 + 0, head v2, 15,154,632 params.
        let m = &cfg.model;
        assert_eq!((m.width, m.heads, m.ffn), (512, 8, 768));
        assert_eq!((m.input_blocks, m.core_blocks, m.output_blocks), (0, 8, 0));
        assert_eq!(recur64_model::model::HEAD_VERSION, 2);
        assert_eq!(cfg.recurrence, 1);
        assert_eq!(
            (cfg.device.as_str(), cfg.precision.as_str()),
            ("cuda", "fp32")
        );
        assert_eq!(
            cfg.reference_model_id.as_deref(),
            Some("d89b408fcc7a3cd9874a0ffbbcafd121c6c78cbb9818d4adaac0fe8d51ea234b")
        );
        // Search: 32 sims, D41 exploration, K = 2.
        assert_eq!(cfg.simulations_per_move, 32);
        assert_eq!(cfg.temperature, 1.0);
        assert_eq!(cfg.argmax_after_ply, Some(30));
        assert_eq!(cfg.root_dirichlet_alpha, 0.3);
        assert_eq!(cfg.root_dirichlet_epsilon, 0.25);
        assert_eq!(cfg.ply_cap, 400);
        assert_eq!(cfg.search_leaves_in_flight, 2);
        // Collection and hardware schedule (H3.3).
        assert_eq!(cfg.collection_shape().unwrap(), (32, 8));
        assert_eq!(cfg.cpu_workers, 8);
        assert_eq!(cfg.max_inference_batch, 16);
        assert_eq!(cfg.batch_timeout_us, 1000);
        assert_eq!((cfg.train_batch, cfg.accumulation_steps), (32, 4));
        assert_eq!(cfg.effective_batch(), 128);
        assert_eq!(cfg.replay_reuse_target, 2.0);
        // Evaluation: D45 V2, promotion conservative-v2, continuous trainer.
        assert_eq!(cfg.arena_games, 32);
        assert_eq!(cfg.arena_sample_plies, Some(30));
        assert_eq!(cfg.arena_root_dirichlet_epsilon, 0.25);
        assert_eq!(cfg.arena_rng_policy, ArenaRngPolicy::PairedCommonV1);
        assert_eq!(
            cfg.arena_config(0, Vec::new(), 8).rng_policy,
            ArenaRngPolicy::PairedCommonV1
        );
        assert_eq!(cfg.trainer_policy, TrainerPolicy::Continuous);
        assert_eq!(cfg.snapshot_policy, SnapshotPolicy::Conservative);
        assert_eq!(cfg.promotion_score_floor, 0.5);
        assert_eq!(cfg.promotion_min_decisive_games, 4);
        // D49 + schedule guard, 3 cycles, amended LR schedule.
        let h = cfg.health_stops;
        assert_eq!(h.draw_share_two_cycles, Some(0.85));
        assert_eq!(h.threefold_fifty, Some(0.60));
        assert_eq!(h.truncation, Some(0.25));
        assert_eq!(h.lr_schedule_end, Some(true));
        assert_eq!(cfg.cycles, 3);
        assert_eq!(cfg.lr_schedule(), (37, 370));
        assert_eq!(cfg.max_updates, 256);
        // Workloads: expected T0, 1.75x stress, theoretical maximum.
        for (positions, updates) in [(5_247, 82), (9_183, 144), (32 * 400, 200)] {
            let plan = cfg.update_plan(positions).unwrap();
            assert_eq!(plan.requested_updates, updates, "{positions} positions");
            assert!(!plan.cap_bound, "cap binds at {positions} positions");
        }
        // Planned = expected cycle 0 + two stress cycles; LR > 0 until then.
        assert_eq!(82 + 144 + 144, cfg.lr_schedule().1);
        let (w, p) = cfg.lr_schedule();
        assert!(crate::learner::lr_at(p - 1, cfg.lr, w, p) > 0.0);
        assert_eq!(crate::learner::lr_at(p, cfg.lr, w, p), 0.0);
    }

    /// R15-P1: the three per-arm smoke configs are the pre-registered contract
    /// and differ only in `recurrence`, the recurrence-matched reference
    /// artifact of identical weights, and labels / wall budget.
    #[test]
    fn r15_smoke_arms_differ_only_in_recurrence() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/hp");
        let load = |r: u32| {
            let mut c = RunConfig::from_toml_str(
                &std::fs::read_to_string(dir.join(format!("r15-smoke-r{r}.toml"))).unwrap(),
            )
            .unwrap();
            // The suite path is repo-relative; tests run from the crate dir.
            assert_eq!(c.opening_suite.as_deref(), Some("configs/openings-v1.toml"));
            c.opening_suite = Some(
                dir.join("../openings-v1.toml")
                    .to_string_lossy()
                    .into_owned(),
            );
            c
        };
        let arms = [load(1), load(2), load(4)];
        // One set of reference weights (semantic digest 17b03869…), one
        // recurrence-matched checkpoint artifact per arm (R15-P1 amendment).
        let refs = [
            "385f4f27b5e9d0540d0ed8db7e5d89178432adfd9c562cb25d268c66e77c7965",
            "0cd0036cae8bfb541cab792f1844eda0f47d69b08f7e6d955ea188c1b723a47b",
            "6c41aec0c95d861e8e01bfd20616bf9b71d361e01120d272ea568d7f38b0d580",
        ];
        for ((cfg, r), reference) in arms.iter().zip([1usize, 2, 4]).zip(refs) {
            assert_eq!(cfg.recurrence, r);
            assert_eq!(cfg.reference_model_id.as_deref(), Some(reference));
            let m = &cfg.model;
            assert_eq!((m.width, m.heads, m.ffn), (512, 8, 768));
            assert_eq!((m.input_blocks, m.core_blocks, m.output_blocks), (2, 4, 2));
            assert_eq!(cfg.search_leaves_in_flight, 2);
            assert_eq!(cfg.simulations_per_move, 32);
            assert_eq!(cfg.collection_shape().unwrap().0, 32);
            assert_eq!(cfg.effective_batch(), 128);
            assert_eq!(cfg.lr_schedule(), (37, 370));
            assert_eq!(cfg.max_updates, 256);
            assert_eq!(cfg.trainer_policy, TrainerPolicy::Continuous);
            assert_eq!(cfg.arena_rng_policy, ArenaRngPolicy::PairedCommonV1);
            assert_eq!(cfg.arena_tree_policy, ArenaTreePolicy::RootPlayerV1);
            assert_eq!(cfg.promotion_score, PromotionScore::AdjudicatedMaterialV1);
            assert_eq!(cfg.arena_early_adjudication, ArenaEarlyAdjudication::Off);
            assert!(cfg.inference_candidate_buckets);
            assert_eq!(cfg.health_stops.draw_share_two_cycles, Some(0.85));
            assert_eq!(cfg.health_stops.threefold_fifty, Some(0.60));
            assert_eq!(cfg.health_stops.truncation, Some(0.25));
            assert_eq!(cfg.health_stops.lr_schedule_end, Some(true));
            assert_eq!(cfg.cycles, 3);
            for (positions, updates) in [(5_247, 82), (12_800, 200)] {
                let plan = cfg.update_plan(positions).unwrap();
                assert_eq!(plan.requested_updates, updates);
                assert!(!plan.cap_bound);
            }
        }
        // Everything scientific except recurrence is identical across arms.
        let strip = |c: &RunConfig| {
            let mut v = c.scientific_identity().unwrap();
            v["recurrence"] = serde_json::Value::Null;
            v["reference_model_id"] = serde_json::Value::Null;
            v
        };
        assert_eq!(strip(&arms[0]), strip(&arms[1]));
        assert_eq!(strip(&arms[0]), strip(&arms[2]));
        assert_ne!(
            arms[0].scientific_config_hash().unwrap(),
            arms[2].scientific_config_hash().unwrap()
        );
    }

    /// D45 must not change any identity recorded before it: the original
    /// deterministic, noise-free arena reproduces the smoke's recorded
    /// scientific hash exactly, while an arena exploration variant is a new
    /// identity.
    #[test]
    fn arena_exploration_is_a_new_identity_and_default_is_unchanged() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mut cfg = RunConfig::from_toml_str(
            &std::fs::read_to_string(root.join("configs/phase4/f10-smoke.toml")).unwrap(),
        )
        .unwrap();
        cfg.opening_suite = Some(
            root.join("configs/openings-v1.toml")
                .to_string_lossy()
                .into_owned(),
        );
        assert_eq!(
            cfg.scientific_config_hash().unwrap(),
            "5548bfaf796a4a3da88da03407ef6ee0e7198b572d30de022d6dae018fa723a3",
            "recorded P4.5 smoke identity must be reproducible"
        );
        let mut sampled = cfg.clone();
        sampled.arena_sample_plies = Some(8);
        assert_ne!(
            sampled.scientific_config_hash().unwrap(),
            cfg.scientific_config_hash().unwrap()
        );
        let arena = sampled.arena_config(3, Vec::new(), 32);
        assert_eq!(arena.sample_plies, Some(8));
        assert_eq!(arena.seed, cfg.seed + 3);
        assert_eq!(arena.root_dirichlet_epsilon, 0.0);
    }

    /// Evaluation scheduling keys are execution-only: they change neither the
    /// scientific identity nor (when unset) the serialized config.
    #[test]
    fn eval_scheduling_is_excluded_from_identity() {
        let base = RunConfig::from_toml_str(base_toml()).unwrap();
        assert!(
            !serde_json::to_string(&base)
                .unwrap()
                .contains("eval_concurrency")
        );
        assert!(
            !serde_json::to_string(&base)
                .unwrap()
                .contains("inference_candidate_buckets")
        );
        let mut e = base.clone();
        e.eval_concurrency = Some(32);
        e.eval_max_inference_batch = Some(32);
        e.inference_candidate_buckets = true;
        assert_eq!(
            base.scientific_config_hash().unwrap(),
            e.scientific_config_hash().unwrap()
        );
        assert_eq!(e.for_evaluation().max_inference_batch, 32);
        assert_eq!(
            base.for_evaluation().max_inference_batch,
            base.max_inference_batch
        );
    }

    /// D54: root-player arena trees are a new identity; the default is not.
    #[test]
    fn arena_tree_policy_is_a_new_identity_and_default_is_unchanged() {
        let base = RunConfig::from_toml_str(base_toml()).unwrap();
        assert!(
            !serde_json::to_string(&base)
                .unwrap()
                .contains("arena_tree_policy")
        );
        let mut t = base.clone();
        t.arena_tree_policy = ArenaTreePolicy::RootPlayerV1;
        assert_ne!(
            base.scientific_config_hash().unwrap(),
            t.scientific_config_hash().unwrap()
        );
        assert_eq!(
            t.arena_config(0, Vec::new(), 8).tree_policy,
            ArenaTreePolicy::RootPlayerV1
        );
    }

    /// H3.5B: the paired-common arena RNG policy is a new scientific identity;
    /// the default leaves both the scientific hash and the serialized
    /// (resolved) config of every earlier config unchanged.
    #[test]
    fn paired_arena_rng_is_a_new_identity_and_default_is_unchanged() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let text = std::fs::read_to_string(root.join("configs/phase4/f10-smoke.toml")).unwrap();
        let mut cfg = RunConfig::from_toml_str(&text).unwrap();
        cfg.opening_suite = Some(
            root.join("configs/openings-v1.toml")
                .to_string_lossy()
                .into_owned(),
        );
        assert_eq!(cfg.arena_rng_policy, ArenaRngPolicy::PerGameV1);
        assert_eq!(
            cfg.scientific_config_hash().unwrap(),
            "5548bfaf796a4a3da88da03407ef6ee0e7198b572d30de022d6dae018fa723a3"
        );
        let json = serde_json::to_string(&cfg).unwrap();
        assert!(
            !json.contains("arena_rng_policy"),
            "the default must not enter the resolved config"
        );
        let mut paired = cfg.clone();
        paired.arena_rng_policy = ArenaRngPolicy::PairedCommonV1;
        assert_ne!(
            paired.scientific_config_hash().unwrap(),
            cfg.scientific_config_hash().unwrap()
        );
        assert_eq!(
            paired.scientific_identity().unwrap()["evaluation"]["arena_rng_policy"],
            "paired_common_v1"
        );
        let arena = paired.arena_config(3, Vec::new(), 8);
        assert_eq!(arena.rng_policy, ArenaRngPolicy::PairedCommonV1);
        // Round trip through TOML keeps the policy.
        let back = RunConfig::from_toml_str(&toml::to_string(&paired).unwrap()).unwrap();
        assert_eq!(back.arena_rng_policy, ArenaRngPolicy::PairedCommonV1);
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
