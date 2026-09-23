//! Durable schema/contract versions for the chess layer.
//!
//! These constants must be persisted with any chess training artifact (replay
//! shards, checkpoints, run metadata) so that a future change cannot silently
//! reinterpret old data. Checkpoint integration is deferred to Phase 2 (see
//! `docs/DECISIONS.md` D-Phase-1-3); the constants exist now so they can be
//! referenced consistently from the start.

/// Observation encoding version. V1 is `[64, 119]` (see `observation.rs`).
pub const OBSERVATION_VERSION_V1: u32 = 1;

/// Action encoding version. V1 is `((from*64 + to)*5 + promo)` over 20,480 IDs.
pub const ACTION_VERSION_V1: u32 = 1;

/// Rules profile version. V1 is standard chess with auto-claim draws.
pub const RULES_PROFILE_VERSION_V1: u32 = 1;

/// The observation/action/rules contract version bundle a run was produced with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ContractVersions {
    pub observation: u32,
    pub action: u32,
    pub rules_profile: u32,
}

impl ContractVersions {
    /// The current version bundle.
    pub const V1: ContractVersions = ContractVersions {
        observation: OBSERVATION_VERSION_V1,
        action: ACTION_VERSION_V1,
        rules_profile: RULES_PROFILE_VERSION_V1,
    };
}

impl Default for ContractVersions {
    fn default() -> Self {
        Self::V1
    }
}
