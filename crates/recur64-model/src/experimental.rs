//! The X15 / Chimera experimental contract.
//!
//! Everything that makes the X15 laboratory different from the historical
//! probe model lives here, is versioned, and is refused rather than guessed:
//!
//! * [`Architecture`] — which model family a config (and a checkpoint) is.
//! * [`ExperimentalConfig`] — the nested `[experimental]` run-config block.
//! * the provider labels for every gated auxiliary pathway.
//!
//! An absent `[experimental]` block is exactly the historical probe
//! architecture: `ExperimentalConfig::default()` has
//! `architecture = ProbeV1` and every option at its documented default, and it
//! is skipped when serializing, so the resolved and scientific identity hashes
//! of every existing config are unchanged.

use serde::{Deserialize, Serialize};

pub use recur64_coproc::ComputeProviderKind;

/// Which model family a config describes, and therefore which readout-head
/// function its checkpoint holds.
///
/// A checkpoint records this and every load path refuses a mismatch in either
/// direction: a historical F15/R15 checkpoint can never silently run as X15,
/// and an X15 checkpoint can never silently run as the probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Architecture {
    /// Phase 0..R15 probe model ([`crate::model::ProbeModel`]).
    #[default]
    ProbeV1,
    /// X15 "Chimera": symbolic trunk + latent reasoning + compute coprocessor
    /// + visual pathway ([`crate::chimera::ChimeraModel`]).
    ChimeraV1,
}

impl Architecture {
    pub fn label(self) -> &'static str {
        match self {
            Architecture::ProbeV1 => "probe_v1",
            Architecture::ChimeraV1 => "chimera_v1",
        }
    }

    /// The readout-head function version this architecture's weights were
    /// trained for. `ProbeV1` is head v2 (the H3/P4 head); `ChimeraV1` starts
    /// its own head lineage at v1 because its heads read a reasoning-
    /// conditioned stream and are not interchangeable with the probe's.
    pub fn head_version(self) -> u32 {
        match self {
            Architecture::ProbeV1 => crate::model::HEAD_VERSION,
            Architecture::ChimeraV1 => CHIMERA_HEAD_VERSION,
        }
    }
}

/// Readout-head function version for the Chimera family.
/// v2 adds the per-candidate fact bias to the policy readout. X15 checkpoints
/// written under v1 are refused (their parameter set differs).
pub const CHIMERA_HEAD_VERSION: u32 = 2;

/// Version of the reasoning-loop contract (the *order* of operations inside a
/// thought). Bump when the documented order changes.
pub const REASONING_CONTRACT_VERSION: &str = "chimera-thought-loop-v1";

/// The single visual resolution X1 accepts (renderable AND encodable).
pub const VISUAL_RESOLUTION_X1: usize = 64;

/// How the reasoning latents are maintained across thoughts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningMode {
    /// Persistent `[b, K, Daux]` latents updated once per thought.
    #[default]
    LatentV1,
}

/// What the training loss does with the intermediate thought readouts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum DeepSupervisionMode {
    /// Only the final thought contributes loss (default).
    #[default]
    FinalOnlyV1,
    /// Every thought contributes the same target, with a small auxiliary
    /// weight on the intermediate ones.
    SameTargetV1,
    /// Thought `t` supervises against the teacher target searched to
    /// `ladder[t]` simulations (`ReasoningTargetsV1`).
    ProgressiveSearchV1,
}

impl DeepSupervisionMode {
    pub fn label(self) -> &'static str {
        match self {
            DeepSupervisionMode::FinalOnlyV1 => "final_only_v1",
            DeepSupervisionMode::SameTargetV1 => "same_target_v1",
            DeepSupervisionMode::ProgressiveSearchV1 => "progressive_search_v1",
        }
    }

    /// Whether intermediate thoughts are read out at all.
    pub fn reads_intermediate(self) -> bool {
        self != DeepSupervisionMode::FinalOnlyV1
    }
}

/// Where the visual board image comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum VisualProviderKind {
    /// The visual pathway is off.
    None,
    /// The deterministic procedural renderer
    /// ([`recur64_coproc::visual`], `visual_board_v1`).
    #[default]
    RenderV1,
}

impl VisualProviderKind {
    pub fn label(self) -> &'static str {
        match self {
            VisualProviderKind::None => "none",
            VisualProviderKind::RenderV1 => "render_v1",
        }
    }
}

/// The retrieval / RAG extension point. **No retrieval is active in X1.**
///
/// A later X2 may add self-generated solved-position retrieval behind exactly
/// this interface plus a `memory_tokens` input on the reasoning bus; no vector
/// store, no opening database and no external chess corpus exists today.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum RetrievalProviderKind {
    /// Retrieval is off (the only implemented value).
    #[default]
    None,
}

impl RetrievalProviderKind {
    pub fn label(self) -> &'static str {
        match self {
            RetrievalProviderKind::None => "none",
        }
    }
}

fn d_thought_steps() -> usize {
    4
}
fn d_reasoning_tokens() -> usize {
    4
}
fn d_aux_width() -> usize {
    128
}
fn d_aux_heads() -> usize {
    4
}
fn d_latent_ffn() -> usize {
    256
}
fn d_true() -> bool {
    true
}
fn d_mate_depth() -> u8 {
    1
}
fn d_compute_tokens() -> usize {
    recur64_coproc::SQUARES + recur64_coproc::GLOBAL_TOKENS
}
fn d_compute_fields() -> usize {
    recur64_coproc::SQ_FIELDS
}
fn d_intermediate_weight() -> f32 {
    0.25
}

fn d_visual_resolution() -> usize {
    64
}
fn d_visual_channels() -> usize {
    48
}
fn d_visual_blocks() -> usize {
    3
}

/// The latent-reasoning subsystem geometry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReasoningConfig {
    #[serde(default)]
    pub mode: ReasoningMode,
    /// Whether the reasoning latents (and therefore the thought loop) run.
    #[serde(default = "d_true")]
    pub enabled: bool,
    /// Attention width of the reasoning bus. The cross-attention projections
    /// work at this width, not at the symbolic trunk width, so three
    /// auxiliary attention paths stay inside the parameter budget.
    #[serde(default = "d_aux_width")]
    pub aux_width: usize,
    #[serde(default = "d_aux_heads")]
    pub aux_heads: usize,
    #[serde(default = "d_latent_ffn")]
    pub latent_ffn: usize,
}

impl Default for ReasoningConfig {
    fn default() -> Self {
        Self {
            mode: ReasoningMode::LatentV1,
            enabled: true,
            aux_width: d_aux_width(),
            aux_heads: d_aux_heads(),
            latent_ffn: d_latent_ffn(),
        }
    }
}

/// The deterministic coprocessor subsystem.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComputeConfig {
    #[serde(default)]
    pub provider: ComputeProviderKind,
    /// Whether the compute tokens reach the reasoning bus.
    #[serde(default = "d_true")]
    pub enabled: bool,
    /// Bounded exact-tactics depth: `0` off, `1` mate-in-1, `2` mate-in-2.
    /// Depth 2 is an exhaustive exact search and is for diagnostics only; the
    /// training path runs at depth 1.
    #[serde(default = "d_mate_depth")]
    pub mate_depth: u8,
    /// Tokens in the bank (must equal the coprocessor's layout).
    #[serde(default = "d_compute_tokens")]
    pub tokens: usize,
    /// Bytes per token (must equal the coprocessor's layout).
    #[serde(default = "d_compute_fields")]
    pub fields: usize,
}

impl Default for ComputeConfig {
    fn default() -> Self {
        Self {
            provider: ComputeProviderKind::None,
            // Off unless a config says otherwise: `enabled` without a provider
            // is refused, so the intent is always explicit.
            enabled: false,
            mate_depth: d_mate_depth(),
            tokens: d_compute_tokens(),
            fields: d_compute_fields(),
        }
    }
}

/// The visual-board subsystem.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VisualConfig {
    #[serde(default)]
    pub provider: VisualProviderKind,
    /// Whether the visual tokens reach the reasoning bus.
    #[serde(default = "d_true")]
    pub enabled: bool,
    /// Image side in pixels (64 or 96).
    #[serde(default = "d_visual_resolution")]
    pub resolution: usize,
    /// Residual CNN stem width.
    #[serde(default = "d_visual_channels")]
    pub channels: usize,
    /// Number of residual blocks.
    #[serde(default = "d_visual_blocks")]
    pub blocks: usize,
}

impl Default for VisualConfig {
    fn default() -> Self {
        Self {
            provider: VisualProviderKind::RenderV1,
            // Off unless a config says otherwise (see `ComputeConfig`).
            enabled: false,
            resolution: d_visual_resolution(),
            channels: d_visual_channels(),
            blocks: d_visual_blocks(),
        }
    }
}

/// Version of the candidate-fact contract (`CandidateFactsV1`).
pub const CANDIDATE_FACTS_VERSION: &str = "candidate_facts_v1";

/// Fields per candidate in `CandidateFactsV1`.
pub const CANDIDATE_FACT_FIELDS: usize = 8;

fn d_facts_gain() -> f32 {
    1.0
}

fn d_facts_hidden() -> usize {
    16
}

/// Where per-candidate exact facts come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CandidateFactsProviderKind {
    /// The pathway is off.
    #[default]
    None,
    /// Native, deterministic one-ply facts about every legal move
    /// (`recur64_runtime::candidate_facts`).
    NativeV1,
}

impl CandidateFactsProviderKind {
    pub fn label(self) -> &'static str {
        match self {
            CandidateFactsProviderKind::None => "none",
            CandidateFactsProviderKind::NativeV1 => "native_v1",
        }
    }

    pub fn is_active(self) -> bool {
        self != CandidateFactsProviderKind::None
    }
}

/// The per-candidate exact-fact pathway. Its output is a per-move bias on the
/// policy logits (a small MLP whose last layer starts at zero), so a fresh
/// network is neutral. It is independent of the latent reasoning state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateFactsConfig {
    #[serde(default)]
    pub provider: CandidateFactsProviderKind,
    #[serde(default)]
    pub enabled: bool,
    /// Multiplier on the fact bias: bias = gain * MLP(facts). AdamW moves each
    /// weight by about the learning rate per step, so a zero-initialised MLP alone
    /// learns a bias of only ~0.01 logits in a short run (measured); the gain sets
    /// the scale at which the facts can influence the policy. Default 1.0.
    #[serde(default = "d_facts_gain")]
    pub gain: f32,
    /// Hidden width of the fact MLP.
    #[serde(default = "d_facts_hidden")]
    pub hidden: usize,
}

impl Default for CandidateFactsConfig {
    fn default() -> Self {
        Self {
            provider: CandidateFactsProviderKind::None,
            enabled: false,
            gain: d_facts_gain(),
            hidden: d_facts_hidden(),
        }
    }
}

/// The retrieval extension point (inactive in X1).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct RetrievalConfig {
    #[serde(default)]
    pub provider: RetrievalProviderKind,
    /// Number of `memory_tokens` the reasoning bus would accept. Must be 0
    /// while every provider is `none`.
    #[serde(default)]
    pub memory_tokens: usize,
}

/// The complete `[experimental]` block.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExperimentalConfig {
    #[serde(default)]
    pub architecture: Architecture,
    /// Thoughts the loop runs. `1` is the square-token transformer control.
    #[serde(default = "d_thought_steps")]
    pub thought_steps: usize,
    #[serde(default = "d_reasoning_tokens")]
    pub reasoning_tokens: usize,
    #[serde(default)]
    pub deep_supervision: DeepSupervisionMode,
    /// Loss weight of each intermediate thought (the final thought is always
    /// weight 1). Used by `same_target_v1` and `progressive_search_v1`; part of
    /// the scientific identity.
    #[serde(default = "d_intermediate_weight")]
    pub intermediate_weight: f32,
    #[serde(default)]
    pub reasoning: ReasoningConfig,
    #[serde(default)]
    pub compute: ComputeConfig,
    #[serde(default)]
    pub visual: VisualConfig,
    #[serde(default)]
    pub retrieval: RetrievalConfig,
    /// Per-candidate exact facts (off by default).
    #[serde(default)]
    pub candidate_facts: CandidateFactsConfig,
}

impl Default for ExperimentalConfig {
    fn default() -> Self {
        Self {
            architecture: Architecture::ProbeV1,
            thought_steps: d_thought_steps(),
            reasoning_tokens: d_reasoning_tokens(),
            deep_supervision: DeepSupervisionMode::FinalOnlyV1,
            intermediate_weight: d_intermediate_weight(),
            reasoning: ReasoningConfig::default(),
            compute: ComputeConfig::default(),
            visual: VisualConfig::default(),
            retrieval: RetrievalConfig::default(),
            candidate_facts: CandidateFactsConfig::default(),
        }
    }
}

impl ExperimentalConfig {
    /// True when this is exactly the historical probe architecture, so the
    /// block can be omitted from serialization and identity.
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }

    /// Whether this config describes the X15 family.
    pub fn is_chimera(&self) -> bool {
        self.architecture == Architecture::ChimeraV1
    }

    /// Validate the block, refusing combinations that would silently do
    /// something other than what the config says.
    pub fn validate(&self, model_width: usize) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.thought_steps >= 1,
            "experimental.thought_steps must be >= 1"
        );
        anyhow::ensure!(
            self.reasoning_tokens >= 1,
            "experimental.reasoning_tokens must be >= 1"
        );
        if !self.is_chimera() {
            anyhow::ensure!(
                self.is_default(),
                "the [experimental] block sets options but architecture is not \
                 \"chimera_v1\"; set architecture = \"chimera_v1\" or remove the block"
            );
            return Ok(());
        }
        anyhow::ensure!(
            self.thought_steps <= 8,
            "experimental.thought_steps {} exceeds the supported maximum of 8",
            self.thought_steps
        );
        anyhow::ensure!(
            self.reasoning.enabled || self.thought_steps == 1,
            "thought_steps {} requires the reasoning latents: with \
             reasoning.enabled = false the thought loop has no state to carry",
            self.thought_steps
        );
        anyhow::ensure!(
            self.reasoning
                .aux_width
                .is_multiple_of(self.reasoning.aux_heads),
            "experimental.reasoning.aux_width {} is not divisible by aux_heads {}",
            self.reasoning.aux_width,
            self.reasoning.aux_heads
        );
        anyhow::ensure!(
            self.reasoning.aux_heads >= 1,
            "experimental.reasoning.aux_heads must be >= 1"
        );
        anyhow::ensure!(
            self.compute.mate_depth <= 2,
            "experimental.compute.mate_depth must be 0, 1 or 2"
        );
        anyhow::ensure!(
            self.compute.tokens == d_compute_tokens(),
            "experimental.compute.tokens {} does not match the coprocessor layout {}",
            self.compute.tokens,
            d_compute_tokens()
        );
        anyhow::ensure!(
            self.compute.fields == d_compute_fields(),
            "experimental.compute.fields {} does not match the coprocessor layout {}",
            self.compute.fields,
            d_compute_fields()
        );
        // X1 supports exactly one visual resolution: the one that is both
        // renderable (VisualBoardV1 supports 64 and 96) and encodable (the
        // stride-2 encoder needs 8 * 2^k with at least two stages: 64, 128,
        // ...). The intersection is 64. A config that validates must render
        // AND encode; `tests/x15_modules.rs` checks this end to end.
        anyhow::ensure!(
            self.visual.resolution == VISUAL_RESOLUTION_X1,
            "experimental.visual.resolution {} is not supported in X1 (only {VISUAL_RESOLUTION_X1}: \
             the renderer supports 64/96, the encoder 64/128/...)",
            self.visual.resolution
        );
        anyhow::ensure!(
            self.visual.channels >= 1 && self.visual.blocks >= 1,
            "experimental.visual.channels and blocks must be >= 1"
        );
        anyhow::ensure!(
            self.retrieval.provider == RetrievalProviderKind::None
                && self.retrieval.memory_tokens == 0,
            "no retrieval provider is implemented in X1; retrieval must stay \
             \"none\" with memory_tokens = 0"
        );
        anyhow::ensure!(
            self.intermediate_weight.is_finite() && (0.0..=1.0).contains(&self.intermediate_weight),
            "experimental.intermediate_weight {} must be in [0, 1]",
            self.intermediate_weight
        );
        anyhow::ensure!(
            !self.candidate_facts.enabled || self.candidate_facts.provider.is_active(),
            "experimental.candidate_facts.enabled is true but provider is \"none\""
        );
        anyhow::ensure!(
            self.candidate_facts.hidden >= 1
                && self.candidate_facts.gain.is_finite()
                && self.candidate_facts.gain > 0.0,
            "experimental.candidate_facts.hidden must be >= 1 and gain finite and > 0"
        );
        // Compute and visual tokens are consumed only by the latent thought
        // loop; with the latents off they would be dead weight, and a
        // "symbolic-only control" must be independent of them.
        anyhow::ensure!(
            self.reasoning.enabled || (!self.compute.enabled && !self.visual.enabled),
            "experimental.compute/visual.enabled require reasoning.enabled = true: \
             their tokens only feed the latent reasoning state"
        );
        anyhow::ensure!(
            !self.compute.enabled || self.compute.provider.is_active(),
            "experimental.compute.enabled is true but provider is \"none\""
        );
        anyhow::ensure!(
            !self.visual.enabled || self.visual.provider != VisualProviderKind::None,
            "experimental.visual.enabled is true but provider is \"none\""
        );
        anyhow::ensure!(
            !self.reasoning.enabled || self.reasoning.aux_width <= model_width * 2,
            "experimental.reasoning.aux_width {} is implausibly large for a trunk \
             width of {model_width}",
            self.reasoning.aux_width
        );
        Ok(())
    }

    /// The identity fields a checkpoint and a scientific hash must record.
    pub fn identity(&self) -> anyhow::Result<serde_json::Value> {
        Ok(serde_json::json!({
            "architecture": self.architecture,
            "architecture_head_version": self.architecture.head_version(),
            "reasoning_contract_version": REASONING_CONTRACT_VERSION,
            "thought_steps": self.thought_steps,
            "reasoning_tokens": self.reasoning_tokens,
            "deep_supervision": self.deep_supervision,
            "intermediate_weight": self.intermediate_weight,
            "reasoning": self.reasoning,
            "compute": self.compute,
            "visual": self.visual,
            "retrieval": self.retrieval,
            "candidate_facts": self.candidate_facts,
            "candidate_facts_version": CANDIDATE_FACTS_VERSION,
            "compute_bank_version": recur64_coproc::COMPUTE_BANK_VERSION,
            "visual_render_version": recur64_coproc::VISUAL_RENDER_VERSION,
        }))
    }
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;

    #[test]
    fn default_is_the_probe_architecture() {
        let d = ExperimentalConfig::default();
        assert!(d.is_default());
        assert!(!d.is_chimera());
        assert_eq!(d.architecture, Architecture::ProbeV1);
        assert_eq!(d.architecture.head_version(), crate::model::HEAD_VERSION);
    }

    #[test]
    fn default_serializes_to_the_config_it_documents() {
        let d = ExperimentalConfig::default();
        let back: ExperimentalConfig =
            toml::from_str(&toml::to_string(&d).unwrap()).expect("round trip");
        assert_eq!(back, d);
    }

    #[test]
    fn unknown_keys_are_refused() {
        let e = toml::from_str::<ExperimentalConfig>("architecture = \"chimera_v1\"\nwibble = 1");
        assert!(e.is_err());
        let e = toml::from_str::<ExperimentalConfig>(
            "architecture = \"chimera_v1\"\n[reasoning]\nnope = 1",
        );
        assert!(e.is_err());
    }

    #[test]
    fn chimera_validation() {
        let mut c = ExperimentalConfig::default();
        c.architecture = Architecture::ChimeraV1;
        assert!(c.validate(512).is_ok(), "a bare chimera config is valid");
        assert_eq!(c.architecture.head_version(), CHIMERA_HEAD_VERSION);
        // The auxiliary pathways are off unless asked for.
        assert!(!c.compute.enabled && !c.visual.enabled);

        // Turning a pathway on without a provider is refused.
        let mut on = c.clone();
        on.compute.enabled = true;
        assert!(on.validate(512).is_err());
        let mut on = c.clone();
        on.visual.enabled = true;
        on.visual.provider = VisualProviderKind::None;
        assert!(on.validate(512).is_err());
        // With a provider it is accepted.
        let mut on = c.clone();
        on.compute.enabled = true;
        on.compute.provider = ComputeProviderKind::NativeV1;
        assert!(on.validate(512).is_ok());

        // thought steps > 1 without the latents is refused.
        let mut bad = c.clone();
        bad.reasoning.enabled = false;
        assert!(bad.validate(512).is_err());

        // Aux width must divide the heads.
        let mut bad = c.clone();
        bad.reasoning.aux_width = 130;
        assert!(bad.validate(512).is_err());

        // The bank layout is pinned.
        let mut bad = c.clone();
        bad.compute.tokens = 65;
        assert!(bad.validate(512).is_err());

        // Retrieval stays off in X1.
        let mut bad = c.clone();
        bad.retrieval.memory_tokens = 8;
        assert!(bad.validate(512).is_err());

        // compute.enabled without a provider is refused.
        let mut bad = c.clone();
        bad.compute.provider = ComputeProviderKind::None;
        bad.compute.enabled = true;
        assert!(bad.validate(512).is_err());
    }

    #[test]
    fn non_chimera_with_options_is_refused() {
        let mut c = ExperimentalConfig::default();
        c.thought_steps = 2;
        assert!(c.validate(512).is_err());
    }

    #[test]
    fn identity_records_every_science_affecting_field() {
        let mut c = ExperimentalConfig::default();
        c.architecture = Architecture::ChimeraV1;
        let id = c.identity().unwrap();
        for key in [
            "architecture",
            "architecture_head_version",
            "reasoning_contract_version",
            "thought_steps",
            "reasoning_tokens",
            "deep_supervision",
            "reasoning",
            "compute",
            "visual",
            "retrieval",
            "compute_bank_version",
            "visual_render_version",
        ] {
            assert!(id.get(key).is_some(), "identity is missing {key}");
        }
        assert_eq!(id["compute_bank_version"], "compute_bank_v1");
        assert_eq!(id["visual_render_version"], "visual_board_v1");
    }
}
