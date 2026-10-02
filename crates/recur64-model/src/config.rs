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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
    /// Active-search geometry and contract identities; present iff
    /// `architecture == active_search_v3`. Skipped when absent so every
    /// historical scientific hash is unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active: Option<ActiveConfig>,
    /// P6 ALL-INFO geometry and contract identities; present iff
    /// `architecture == all_info_v1`. Skipped when absent, so every historical and
    /// V3 scientific hash is unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub all_info: Option<AllInfoConfig>,
    /// V4 evidence-belief geometry and contract identities; present iff
    /// `architecture == evidence_belief_v4`. Skipped when absent, so every historical,
    /// V3 and ALL-INFO scientific hash is unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evidence: Option<EvidenceConfig>,
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
    /// Recur64 V3: one root encoding plus a budgeted number of exact
    /// single-edge state queries (`active_search_v3`).
    ActiveSearchV3,
    /// Recur64 V3 P6: the separately trained information-sufficiency control that
    /// receives an exhaustive raw depth-2 tree at once (`all_info_v1`).
    AllInfoV1,
    /// Recur64 V4: an immutable base belief over root-action hypotheses plus an explicit,
    /// content-causal evidence ledger and a learned query-utility head
    /// (`evidence_belief_v4`).
    EvidenceBeliefV4,
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
            Architecture::ActiveSearchV3 => "active_search_v3",
            Architecture::AllInfoV1 => "all_info_v1",
            Architecture::EvidenceBeliefV4 => "evidence_belief_v4",
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
///
/// * v1 - engineering only, never science: the final `Linear(hidden -> 1)` had a bias,
///   which adds the same constant to every candidate of a row and is cancelled by the
///   softmax (an inert parameter with identically zero policy gradient).
/// * v2 - the final layer has NO bias.
pub const FACT_DELTA_CONTRACT: u32 = 2;

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

// ---------------------------------------------------------------------------
// V3 active search (`active_search_v3`). Every contract below enters the model
// identity through `ActiveConfig::contracts`, so a checkpoint built under one
// contract refuses another (see `CheckpointMeta::check_model`).
// ---------------------------------------------------------------------------

pub const ACTIVE_ROOT_ENCODER: &str = "v25_root_encoder_v1";
pub const ACTIVE_ROOT_CANDIDATE_TOKENS: &str = "candidate_token_v3_root_v1";
pub const ACTIVE_STATE_QUERY: &str = "state_query_v1";
pub const ACTIVE_QUERY_STATE_ENCODER: &str = "query_state_encoder_v1";
pub const ACTIVE_FRONTIER: &str = "frontier_v1";
pub const ACTIVE_SEARCH_MEMORY: &str = "branch_workspace_v1";
pub const ACTIVE_SELECTOR: &str = "active_selector_v1";
pub const ACTIVE_PLANNER: &str = "active_planner_v1";
pub const ACTIVE_PROOF_TRACE: &str = "proof_trace_v1";
pub const ACTIVE_BUDGET_TRAINING: &str = "budget_0_2_4_8_v1";
pub const ACTIVE_ROOT_POLICY: &str = "root_policy_v3_v1";

/// Version of the V3 readout function (root policy readout + neutral WDL head).
pub const ACTIVE_HEAD_VERSION: u32 = 1;

/// Largest ply depth the planner depth features accept. Deeper nodes are
/// refused, never clipped.
pub const ACTIVE_MAX_DEPTH: usize = 64;
/// Largest query budget of the V3.0 scientific experiment: B0/2/4/8/16, trained
/// through B8, with B16 the extrapolation point. A larger budget is a different
/// experiment and needs its own preregistration.
pub const ACTIVE_MAX_BUDGET: usize = 16;
/// Implementation ceiling for explicit, non-scientific stress runs
/// (`RunOptions::engineering_stress`). Such runs are recorded as engineering only.
pub const ACTIVE_ENGINEERING_MAX_BUDGET: usize = 64;

/// Versioned identities of every V3 scientific contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActiveContracts {
    pub root_encoder: String,
    pub root_candidate_tokens: String,
    pub state_query: String,
    pub query_state_encoder: String,
    pub frontier: String,
    pub search_memory: String,
    pub selector: String,
    pub planner: String,
    pub proof_trace: String,
    pub budget_training: String,
    pub root_policy: String,
}

impl Default for ActiveContracts {
    fn default() -> Self {
        Self {
            root_encoder: ACTIVE_ROOT_ENCODER.into(),
            root_candidate_tokens: ACTIVE_ROOT_CANDIDATE_TOKENS.into(),
            state_query: ACTIVE_STATE_QUERY.into(),
            query_state_encoder: ACTIVE_QUERY_STATE_ENCODER.into(),
            frontier: ACTIVE_FRONTIER.into(),
            search_memory: ACTIVE_SEARCH_MEMORY.into(),
            selector: ACTIVE_SELECTOR.into(),
            planner: ACTIVE_PLANNER.into(),
            proof_trace: ACTIVE_PROOF_TRACE.into(),
            budget_training: ACTIVE_BUDGET_TRAINING.into(),
            root_policy: ACTIVE_ROOT_POLICY.into(),
        }
    }
}

fn d_q_dim() -> usize {
    256
}
fn d_q_heads() -> usize {
    4
}
fn d_q_ffn() -> usize {
    512
}
fn d_q_blocks() -> usize {
    2
}
fn d_workspace_tokens() -> usize {
    8
}
fn d_planner_heads() -> usize {
    4
}
fn d_planner_ffn() -> usize {
    512
}
fn d_selector_hidden() -> usize {
    256
}
fn d_readout_hidden() -> usize {
    128
}
fn d_rms_ceiling() -> f64 {
    16.0
}

/// Active-search geometry. The workspace, branch memory, edge and node
/// embeddings all have width `query_dim`, which must equal the root candidate
/// dimension.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActiveConfig {
    /// Root candidate geometry (V2.5 CF: dim 256, 4 heads, FFN 512, 1 block, facts on).
    #[serde(default)]
    pub candidate: CandidateConfig,
    #[serde(default = "d_q_dim")]
    pub query_dim: usize,
    #[serde(default = "d_q_heads")]
    pub query_heads: usize,
    #[serde(default = "d_q_ffn")]
    pub query_ffn: usize,
    #[serde(default = "d_q_blocks")]
    pub query_blocks: usize,
    #[serde(default = "d_workspace_tokens")]
    pub workspace_tokens: usize,
    #[serde(default = "d_planner_heads")]
    pub planner_heads: usize,
    #[serde(default = "d_planner_ffn")]
    pub planner_ffn: usize,
    #[serde(default = "d_selector_hidden")]
    pub selector_hidden: usize,
    #[serde(default = "d_readout_hidden")]
    pub readout_hidden: usize,
    /// Health guard: a workspace or branch-memory RMS above this errors.
    #[serde(default = "d_rms_ceiling")]
    pub rms_ceiling: f64,
    #[serde(default)]
    pub contracts: ActiveContracts,
}

impl Default for ActiveConfig {
    fn default() -> Self {
        Self {
            candidate: CandidateConfig::default(),
            query_dim: d_q_dim(),
            query_heads: d_q_heads(),
            query_ffn: d_q_ffn(),
            query_blocks: d_q_blocks(),
            workspace_tokens: d_workspace_tokens(),
            planner_heads: d_planner_heads(),
            planner_ffn: d_planner_ffn(),
            selector_hidden: d_selector_hidden(),
            readout_hidden: d_readout_hidden(),
            rms_ceiling: d_rms_ceiling(),
            contracts: ActiveContracts::default(),
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

// ---------------------------------------------------------------------------
// P6 ALL-INFO (`all_info_v1`): the information-sufficiency control. Separately
// trained; shares the V2.5 root encoder, the root candidate tokens and the
// `query_state_encoder_v1` architecture with `active_search_v3`, and replaces the
// selector/planner by a set integrator over the exhaustive raw depth-2 tree.
// ---------------------------------------------------------------------------

pub const ALL_INFO_INPUT: &str = "all_info_depth2_v1";
pub const ALL_INFO_INTEGRATOR: &str = "all_info_tree_integrator_v1";

/// Version of the ALL-INFO readout function.
pub const ALL_INFO_HEAD_VERSION: u32 = 1;

/// Versioned identities of every ALL-INFO scientific contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AllInfoContracts {
    pub root_encoder: String,
    pub root_candidate_tokens: String,
    pub state_query: String,
    pub query_state_encoder: String,
    pub input: String,
    pub integrator: String,
    pub root_policy: String,
}

impl Default for AllInfoContracts {
    fn default() -> Self {
        Self {
            root_encoder: ACTIVE_ROOT_ENCODER.into(),
            root_candidate_tokens: ACTIVE_ROOT_CANDIDATE_TOKENS.into(),
            state_query: ACTIVE_STATE_QUERY.into(),
            query_state_encoder: ACTIVE_QUERY_STATE_ENCODER.into(),
            input: ALL_INFO_INPUT.into(),
            integrator: ALL_INFO_INTEGRATOR.into(),
            root_policy: ACTIVE_ROOT_POLICY.into(),
        }
    }
}

fn d_set_heads() -> usize {
    4
}
fn d_set_ffn() -> usize {
    768
}

/// ALL-INFO geometry. Token width equals the root candidate dimension and the
/// query-encoder width (shared token space).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AllInfoConfig {
    /// Root candidate geometry (V2.5 CF: dim 256, 4 heads, FFN 512, 1 block, facts on).
    #[serde(default)]
    pub candidate: CandidateConfig,
    #[serde(default = "d_q_dim")]
    pub query_dim: usize,
    #[serde(default = "d_q_heads")]
    pub query_heads: usize,
    #[serde(default = "d_q_ffn")]
    pub query_ffn: usize,
    #[serde(default = "d_q_blocks")]
    pub query_blocks: usize,
    /// Heads of the per-branch and cross-branch set blocks and of the branch pooling.
    #[serde(default = "d_set_heads")]
    pub set_heads: usize,
    /// FFN width of the per-branch and cross-branch set blocks.
    #[serde(default = "d_set_ffn")]
    pub set_ffn: usize,
    #[serde(default = "d_readout_hidden")]
    pub readout_hidden: usize,
    #[serde(default)]
    pub contracts: AllInfoContracts,
}

impl Default for AllInfoConfig {
    fn default() -> Self {
        Self {
            candidate: CandidateConfig::default(),
            query_dim: d_q_dim(),
            query_heads: d_q_heads(),
            query_ffn: d_q_ffn(),
            query_blocks: d_q_blocks(),
            set_heads: d_set_heads(),
            set_ffn: d_set_ffn(),
            readout_hidden: d_readout_hidden(),
            contracts: AllInfoContracts::default(),
        }
    }
}

impl AllInfoConfig {
    /// The geometry the shared V3 modules (root path, query encoder, readout) are built
    /// from. Fields that only the active-search planner/selector use keep their defaults
    /// and never enter an ALL-INFO model.
    pub fn shared_geometry(&self) -> ActiveConfig {
        ActiveConfig {
            candidate: self.candidate.clone(),
            query_dim: self.query_dim,
            query_heads: self.query_heads,
            query_ffn: self.query_ffn,
            query_blocks: self.query_blocks,
            readout_hidden: self.readout_hidden,
            ..ActiveConfig::default()
        }
    }
}

// ---------------------------------------------------------------------------
// V4 `evidence_belief_v4`: root hypothesis bank, immutable base belief z0, explicit
// content-causal evidence messages, set-like ledger, additive belief update, and a
// decision-aligned query-utility head. See docs/V4_RESEARCH_PLAN.md.
// ---------------------------------------------------------------------------

pub const EVIDENCE_BASE_READOUT: &str = "base_readout_v4_v1";
pub const EVIDENCE_ENCODER: &str = "evidence_encoder_v4_v1";
pub const EVIDENCE_LEDGER: &str = "evidence_ledger_set_v1";
pub const EVIDENCE_BELIEF_UPDATE: &str = "belief_update_gated_sum_v1";
pub const EVIDENCE_UTILITY_HEAD: &str = "query_utility_head_v1";

/// Version of the V4 readout function (base logits + additive evidence delta).
pub const EVIDENCE_HEAD_VERSION: u32 = 1;

/// Versioned identities of every V4 scientific contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceContracts {
    pub root_encoder: String,
    pub root_candidate_tokens: String,
    pub state_query: String,
    pub base_readout: String,
    pub evidence_encoder: String,
    pub ledger: String,
    pub belief_update: String,
    pub utility_head: String,
}

impl Default for EvidenceContracts {
    fn default() -> Self {
        Self {
            root_encoder: ACTIVE_ROOT_ENCODER.into(),
            root_candidate_tokens: ACTIVE_ROOT_CANDIDATE_TOKENS.into(),
            state_query: ACTIVE_STATE_QUERY.into(),
            base_readout: EVIDENCE_BASE_READOUT.into(),
            evidence_encoder: EVIDENCE_ENCODER.into(),
            ledger: EVIDENCE_LEDGER.into(),
            belief_update: EVIDENCE_BELIEF_UPDATE.into(),
            utility_head: EVIDENCE_UTILITY_HEAD.into(),
        }
    }
}

fn d_ev_content_dim() -> usize {
    192
}
fn d_ev_content_heads() -> usize {
    4
}
fn d_ev_content_ffn() -> usize {
    384
}
fn d_ev_content_blocks() -> usize {
    2
}
fn d_ev_message_dim() -> usize {
    128
}
fn d_ev_pair_dim() -> usize {
    128
}
fn d_ev_key_dim() -> usize {
    64
}
fn d_ev_trust_hidden() -> usize {
    64
}
fn d_ev_base_hidden() -> usize {
    256
}
fn d_ev_utility_hidden() -> usize {
    256
}
fn d_ev_delta_bound() -> f64 {
    8.0
}

/// V4 geometry. The root candidate dimension is also the width of the (non-evidence)
/// query-state encoder that supplies parent/action representations to the utility head.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidenceConfig {
    /// Root candidate (hypothesis token) geometry: dim 256, 4 heads, FFN 512, 1 block, facts on.
    #[serde(default)]
    pub candidate: CandidateConfig,
    /// Hidden width of the base readout `z0_i = f(h_i, root_context)`.
    #[serde(default = "d_ev_base_hidden")]
    pub base_hidden: usize,
    /// Parent-state encoder used by the utility head only (never by the evidence path).
    #[serde(default = "d_q_heads")]
    pub query_heads: usize,
    #[serde(default = "d_q_ffn")]
    pub query_ffn: usize,
    #[serde(default = "d_q_blocks")]
    pub query_blocks: usize,
    /// Bias-free content encoder token width / heads / FFN / blocks.
    #[serde(default = "d_ev_content_dim")]
    pub content_dim: usize,
    #[serde(default = "d_ev_content_heads")]
    pub content_heads: usize,
    #[serde(default = "d_ev_content_ffn")]
    pub content_ffn: usize,
    #[serde(default = "d_ev_content_blocks")]
    pub content_blocks: usize,
    /// `EvidenceMessage` vector width.
    #[serde(default = "d_ev_message_dim")]
    pub message_dim: usize,
    /// Width of the hypothesis x message interaction.
    #[serde(default = "d_ev_pair_dim")]
    pub pair_dim: usize,
    /// Width of the content-derived attention key used by the routing gate.
    #[serde(default = "d_ev_key_dim")]
    pub key_dim: usize,
    #[serde(default = "d_ev_trust_hidden")]
    pub trust_hidden: usize,
    #[serde(default = "d_ev_utility_hidden")]
    pub utility_hidden: usize,
    /// `delta_z` is passed through `B * tanh(x / B)`; B is part of the model function.
    #[serde(default = "d_ev_delta_bound")]
    pub delta_bound: f64,
    #[serde(default)]
    pub contracts: EvidenceContracts,
}

impl Default for EvidenceConfig {
    fn default() -> Self {
        Self {
            candidate: CandidateConfig::default(),
            base_hidden: d_ev_base_hidden(),
            query_heads: d_q_heads(),
            query_ffn: d_q_ffn(),
            query_blocks: d_q_blocks(),
            content_dim: d_ev_content_dim(),
            content_heads: d_ev_content_heads(),
            content_ffn: d_ev_content_ffn(),
            content_blocks: d_ev_content_blocks(),
            message_dim: d_ev_message_dim(),
            pair_dim: d_ev_pair_dim(),
            key_dim: d_ev_key_dim(),
            trust_hidden: d_ev_trust_hidden(),
            utility_hidden: d_ev_utility_hidden(),
            delta_bound: d_ev_delta_bound(),
            contracts: EvidenceContracts::default(),
        }
    }
}

impl EvidenceConfig {
    /// Width of the root candidate tokens (the hypothesis tokens).
    pub fn token_dim(&self) -> usize {
        self.candidate.dim
    }

    /// The geometry the reused V3 `RootPath` and `QueryEncoder` modules are built from. Only
    /// the root candidate geometry and the query-encoder fields enter; the planner/selector
    /// fields keep their defaults and never enter a V4 model.
    pub fn shared_geometry(&self) -> ActiveConfig {
        ActiveConfig {
            candidate: self.candidate.clone(),
            query_dim: self.candidate.dim,
            query_heads: self.query_heads,
            query_ffn: self.query_ffn,
            query_blocks: self.query_blocks,
            readout_hidden: self.base_hidden,
            ..ActiveConfig::default()
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
            active: None,
            all_info: None,
            evidence: None,
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
            active: None,
            all_info: None,
            evidence: None,
        }
    }

    /// V3 `active_search_v3`: the V2.5 CF root geometry plus the active-search
    /// components. Root CandidateFacts are enabled.
    pub fn active_search_v3() -> Self {
        let mut m = Self::candidate_v25(true);
        m.architecture = Architecture::ActiveSearchV3;
        m.candidate = None;
        m.active = Some(ActiveConfig::default());
        m
    }

    /// P6 `all_info_v1`: the V2.5 CF root geometry plus the exhaustive-tree integrator.
    pub fn all_info_v1() -> Self {
        let mut m = Self::candidate_v25(true);
        m.architecture = Architecture::AllInfoV1;
        m.candidate = None;
        m.all_info = Some(AllInfoConfig::default());
        m
    }

    /// V4 `evidence_belief_v4`: the V2.5 CF root geometry for the base tower plus the
    /// evidence components. Root CandidateFacts are enabled.
    pub fn evidence_belief_v4() -> Self {
        let mut m = Self::candidate_v25(true);
        m.architecture = Architecture::EvidenceBeliefV4;
        m.candidate = None;
        m.evidence = Some(EvidenceConfig::default());
        m
    }

    /// Visible refusal for every historical command that has no `evidence_belief_v4` path.
    /// Call it before any model construction or device work.
    pub fn refuse_evidence_v4(&self, command: &str) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.architecture != Architecture::EvidenceBeliefV4,
            "evidence_belief_v4 is not supported by {command}: that tool builds a different \
             graph and would measure the wrong model. Use `recur64 v4` to describe and run \
             evidence_belief_v4"
        );
        Ok(())
    }

    /// Visible refusal for every historical command that has no `all_info_v1` path.
    pub fn refuse_all_info(&self, command: &str) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.architecture != Architecture::AllInfoV1,
            "all_info_v1 is not supported by {command}: that tool builds a different graph and \
             would measure the wrong model. Use `recur64 model-info` to describe all_info_v1 and \
             `recur64 v3-p6` to run it"
        );
        Ok(())
    }

    /// Visible refusal for every historical command that has no `active_search_v3`
    /// path. Call it before any model construction or device work: a historical
    /// tool must never interpret an active-search config as another architecture.
    pub fn refuse_active_v3(&self, command: &str) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.architecture != Architecture::ActiveSearchV3,
            "active_search_v3 is not supported by {command}: that tool builds a different \
             graph and would measure the wrong model. Use `recur64 model-info` to describe \
             active_search_v3 and `recur64 v3-qual` to qualify it (every budget above 0 \
             needs the live state-query tool)"
        );
        Ok(())
    }

    /// Refuse an architecture/geometry combination that is not a real model.
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.legacy_facts.is_none() || self.architecture == Architecture::LegacyFactsV25,
            "only legacy_facts_v25 may carry a facts-delta geometry"
        );
        anyhow::ensure!(
            self.active.is_none() || self.architecture == Architecture::ActiveSearchV3,
            "only active_search_v3 may carry an active-search geometry"
        );
        anyhow::ensure!(
            self.all_info.is_none() || self.architecture == Architecture::AllInfoV1,
            "only all_info_v1 may carry an ALL-INFO geometry"
        );
        anyhow::ensure!(
            self.evidence.is_none() || self.architecture == Architecture::EvidenceBeliefV4,
            "only evidence_belief_v4 may carry an evidence geometry"
        );
        if self.architecture == Architecture::EvidenceBeliefV4 {
            let e = self.evidence.as_ref().ok_or_else(|| {
                anyhow::anyhow!("evidence_belief_v4 requires an evidence geometry")
            })?;
            anyhow::ensure!(
                self.candidate.is_none()
                    && self.legacy_facts.is_none()
                    && self.active.is_none()
                    && self.all_info.is_none(),
                "evidence_belief_v4 carries its candidate geometry inside `evidence`"
            );
            anyhow::ensure!(
                self.input_blocks == 0 && self.output_blocks == 0,
                "evidence_belief_v4 root encoder has no input or output blocks"
            );
            anyhow::ensure!(
                e.candidate.facts_enabled,
                "evidence_belief_v4 root CandidateFacts must be enabled"
            );
            anyhow::ensure!(
                e.candidate.dim.is_multiple_of(e.candidate.heads)
                    && e.candidate.dim.is_multiple_of(e.query_heads)
                    && e.content_dim.is_multiple_of(e.content_heads),
                "evidence_belief_v4 dims must divide by their head counts"
            );
            anyhow::ensure!(
                e.delta_bound.is_finite() && e.delta_bound > 0.0,
                "evidence delta bound must be finite and positive"
            );
            anyhow::ensure!(
                e.message_dim > 0 && e.pair_dim > 0 && e.key_dim > 0 && e.trust_hidden > 0,
                "evidence widths must be positive"
            );
            anyhow::ensure!(
                e.contracts == EvidenceContracts::default(),
                "evidence_belief_v4 contracts {:?} differ from the current contracts",
                e.contracts
            );
            return Ok(());
        }
        if self.architecture == Architecture::AllInfoV1 {
            let a = self
                .all_info
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("all_info_v1 requires an ALL-INFO geometry"))?;
            anyhow::ensure!(
                self.candidate.is_none() && self.legacy_facts.is_none() && self.active.is_none(),
                "all_info_v1 carries its candidate geometry inside `all_info`"
            );
            anyhow::ensure!(
                self.input_blocks == 0 && self.output_blocks == 0,
                "all_info_v1 root encoder has no input or output blocks"
            );
            anyhow::ensure!(
                a.candidate.dim == a.query_dim,
                "root candidate dim {} must equal query_dim {} (shared token space)",
                a.candidate.dim,
                a.query_dim
            );
            anyhow::ensure!(
                a.candidate.facts_enabled,
                "all_info_v1 root CandidateFacts must be enabled"
            );
            anyhow::ensure!(
                a.query_dim.is_multiple_of(a.query_heads)
                    && a.query_dim.is_multiple_of(a.set_heads)
                    && a.candidate.dim.is_multiple_of(a.candidate.heads),
                "all_info_v1 dims must divide by their head counts"
            );
            anyhow::ensure!(
                a.contracts == AllInfoContracts::default(),
                "all_info_v1 contracts {:?} differ from the current contracts",
                a.contracts
            );
            return Ok(());
        }
        if self.architecture == Architecture::ActiveSearchV3 {
            let a = self
                .active
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("active_search_v3 requires an active geometry"))?;
            anyhow::ensure!(
                self.candidate.is_none() && self.legacy_facts.is_none(),
                "active_search_v3 carries its candidate geometry inside `active`"
            );
            anyhow::ensure!(
                self.input_blocks == 0 && self.output_blocks == 0,
                "active_search_v3 root encoder has no input or output blocks"
            );
            anyhow::ensure!(
                a.candidate.dim == a.query_dim,
                "root candidate dim {} must equal query_dim {} (shared token space)",
                a.candidate.dim,
                a.query_dim
            );
            anyhow::ensure!(
                a.candidate.facts_enabled,
                "active_search_v3 root CandidateFacts must be enabled"
            );
            anyhow::ensure!(
                a.query_dim.is_multiple_of(a.query_heads)
                    && a.query_dim.is_multiple_of(a.planner_heads)
                    && a.candidate.dim.is_multiple_of(a.candidate.heads),
                "active_search_v3 dims must divide by their head counts"
            );
            anyhow::ensure!(a.workspace_tokens > 0, "workspace needs at least one token");
            anyhow::ensure!(
                a.contracts == ActiveContracts::default(),
                "active_search_v3 contracts {:?} differ from the current contracts",
                a.contracts
            );
            return Ok(());
        }
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
            (Architecture::LegacyFactsV25, _)
            | (Architecture::ActiveSearchV3, _)
            | (Architecture::AllInfoV1, _)
            | (Architecture::EvidenceBeliefV4, _) => {
                unreachable!("handled above")
            }
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
            Architecture::CandidateV25
                | Architecture::LegacyFactsV25
                | Architecture::ActiveSearchV3
                | Architecture::AllInfoV1
                | Architecture::EvidenceBeliefV4
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
            active: None,
            all_info: None,
            evidence: None,
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
            active: None,
            all_info: None,
            evidence: None,
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
