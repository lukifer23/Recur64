//! Recur64 chess contracts (Phase 1).
//!
//! This crate defines Recur64's explicit, versioned chess world: squares and
//! canonical perspective, action identity, legal move integration, game state
//! and history, the rules/draw profile, the observation encoding, UCI/FEN
//! boundaries, and perft. It is CPU-only and does **not** depend on Burn or on
//! `recur64-model`.
//!
//! Phase 1 contains no search, self-play, replay, or learning.

pub mod action;
pub mod error;
pub mod fixtures;
pub mod game;
pub mod observation;
pub mod perft;
pub mod rules;
pub mod schema;
pub mod square;
pub mod uci;

pub use action::{
    ACTION_SPACE, ActionId, ActionList, MAX_LEGAL_MOVES, PROMO_B, PROMO_CODES, PROMO_N, PROMO_NONE,
    PROMO_Q, PROMO_R, PromotionCode,
};
pub use cozy_chess::Board;
pub use error::CoreError;
pub use game::GameState;
pub use observation::{OBS_LEN, ObservationV1, encode_observation_v1};
pub use rules::{Outcome, Termination, is_insufficient_material};
pub use schema::{
    ACTION_VERSION_V1, ContractVersions, OBSERVATION_VERSION_V1, RULES_PROFILE_VERSION_V1,
};
pub use square::{
    Color, File, NUM_SQUARES, Perspective, Piece, Rank, Square, canonical_color, canonical_square,
    flip_rank, invert, square_from_index,
};
pub use uci::StandardMove;
