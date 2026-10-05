//! Recur64 V5 `counterfactual_relational_loop_v1`.
//!
//! V5 keeps exact state acquisition (Q) separate from repeated applications of
//! one shared hypothesis/evidence loop (R). The model is read through matched
//! factual and all-payload-null streams so only candidate-relative payload
//! corrections reach the frozen baseline.

pub mod config;
pub mod data;
pub mod drill;
pub mod evaluation;
pub mod graph;
pub mod loss;
pub mod model;
pub mod native_data;
pub mod native_data_v2;
pub mod profile;
pub mod qualification;
pub mod stage;
pub mod study;

pub const SQUARES: usize = 64;
pub const IN_FEATURES: usize = 119;
pub const FACT_FIELDS: usize = 8;
pub const ACTION_GEOMETRY: usize = 11;
pub const PAYLOAD_FLAGS: usize = 9;
pub const DEPTH_FEATURES: usize = 16;
pub const TURN_FEATURE_OFFSET: usize = DEPTH_FEATURES;
pub const SLOT_FEATURE_OFFSET: usize = TURN_FEATURE_OFFSET + 2;
pub const ACTION_FEATURE_OFFSET: usize = SLOT_FEATURE_OFFSET + 4;
pub const STRUCTURAL_FEATURES: usize = ACTION_FEATURE_OFFSET + ACTION_GEOMETRY;
pub const MASKED_LOGIT: f32 = -1.0e9;

// Flex seeds its process-global RNG. Serialize the model-building unit tests
// so parallel test scheduling cannot change their deterministic fixtures.
#[cfg(test)]
pub(crate) static CPU_TEST_RNG: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub(crate) fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
