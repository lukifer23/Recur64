//! Recur64 deterministic chess coprocessor and visual board renderer (X15).
//!
//! This crate is intentionally free of Burn, of `recur64-core` (so it stays
//! buildable for `wasm32-unknown-unknown`) and of every nondeterministic input.
//! It turns a canonical [`ObservationV1`] plus the canonical legal candidate
//! list into:
//!
//! * [`compute::compute_bank`] — `ComputeBankV1`, a fixed-size packed buffer of
//!   *exact* chess facts (attacks, legal-move counts, check relations, bounded
//!   exact tactics). No engine evaluation, no tablebase, no material scalar.
//! * [`visual::render_board`] — `VisualBoardV1`, a canonical top-down RGB board
//!   image built procedurally, with no fonts, textures or external assets.
//!
//! The same `compute_bank` source is compiled natively and into the WASM guest,
//! which is what makes `NativeV1` and `WasmV1` byte-identical by construction.
//!
//! # Input buffer
//!
//! ```text
//! offset 0    u8   version (currently 1)
//! offset 1    u8   mate_search_depth (0 = off, 1 = mate-in-1, 2 = mate-in-2)
//! offset 2    u16  reserved (0)
//! offset 4    u32  n_legal (little-endian)
//! offset 8    256 x (from u8, to u8, promo u8, 0)   canonical squares, promo 0=none
//! offset 1032 7616 x f32  ObservationV1, row-major [square][feature]
//! ```
//!
//! Total `INPUT_LEN` bytes. From/to squares and the promotion code are exactly
//! `recur64_core::ActionId`'s physical-bits-with-canonical-squares form; this
//! crate only reads them, it never canonicalizes again.
//!
//! # Output buffer
//!
//! `64 * SQ_FIELDS` per-square bytes followed by `GLOBAL_TOKENS * GLOBAL_FIELDS`
//! global bytes, all little-endian scalars packed as `u8`.

pub mod board;
pub mod compute;
pub mod input;
pub mod squares;
pub mod visual;
pub mod world;

#[cfg(test)]
mod test_support;

/// Semantic version of the compute-bank feature layout. Part of the X15
/// scientific identity whenever the compute pathway is enabled.
pub const COMPUTE_BANK_VERSION: &str = "compute_bank_v1";

/// Semantic version of the visual renderer. Part of the X15 scientific
/// identity whenever the visual pathway is enabled.
pub const VISUAL_RENDER_VERSION: &str = "visual_board_v1";

/// Observation V1 float count.
pub const OBS_LEN: usize = 7616;
/// Observation V1 byte count.
pub const OBS_BYTES: usize = OBS_LEN * 4;
/// Products-per-square feature count.
pub const FEATURES_PER_SQUARE: usize = 119;
/// Board squares.
pub const SQUARES: usize = 64;
/// Maximum stored legal candidates (chess maximum is 218; 256 is a safe cap).
pub const MAX_LEGAL: usize = 256;
/// Per-square bytes in `ComputeBankV1`.
pub const SQ_FIELDS: usize = 24;
/// Global compute tokens in `ComputeBankV1`.
pub const GLOBAL_TOKENS: usize = 8;
/// Bytes per global compute token.
///
/// Deliberately the same as [`SQ_FIELDS`] so the whole bank is one uniform
/// `[SQUARES + GLOBAL_TOKENS, fields]` token grid the model can project with a
/// single linear map. Only the first 16 bytes of each global token are used;
/// the remainder is reserved and zero.
pub const GLOBAL_FIELDS: usize = SQ_FIELDS;
/// Bytes of each global token that carry meaning.
pub const GLOBAL_PAYLOAD: usize = 16;

/// Header bytes before the move list.
pub const HEADER_BYTES: usize = 8;
/// Byte offset of the observation inside an input buffer.
pub const OBS_OFFSET: usize = HEADER_BYTES + MAX_LEGAL * 4;
/// Fixed input buffer length.
pub const INPUT_LEN: usize = OBS_OFFSET + OBS_BYTES;
/// Fixed `ComputeBankV1` output length.
pub const OUTPUT_LEN: usize = SQUARES * SQ_FIELDS + GLOBAL_TOKENS * GLOBAL_FIELDS;

/// Offset of the global tokens inside a `ComputeBankV1` buffer.
pub const GLOBAL_OFFSET: usize = SQUARES * SQ_FIELDS;

/// Supported input-buffer version.
pub const INPUT_VERSION: u8 = 1;

/// A coprocessor failure. Every failure is visible; there is no fallback path
/// that silently substitutes zeros.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoprocError {
    /// The input buffer length is not `INPUT_LEN`.
    BadInputLength(usize),
    /// The input version byte is not [`INPUT_VERSION`].
    BadVersion(u8),
    /// `n_legal` exceeds [`MAX_LEGAL`].
    TooManyLegal(usize),
    /// A move entry names a square outside `0..64`, or a promotion code
    /// outside `0..=4`.
    BadMove {
        index: usize,
        from: u8,
        to: u8,
        promo: u8,
    },
    /// The observation does not encode exactly one piece-or-empty per square in
    /// frame 0, or the reconstructed position is not a legal chess position.
    InvalidObservation(&'static str),
    /// The reconstructed board refused its castling rights, en-passant square
    /// or halfmove clock.
    InvalidBoard(&'static str),
    /// A world-model capacity (candidates or replies) was exceeded. Never truncated.
    Capacity(&'static str),
    /// The output buffer length is not `OUTPUT_LEN`.
    BadOutputLength(usize),
    /// The renderer was asked for an unsupported image size.
    BadImageSize(usize),
}

impl std::fmt::Display for CoprocError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CoprocError::BadInputLength(n) => {
                write!(f, "input buffer is {n} bytes, expected {INPUT_LEN}")
            }
            CoprocError::BadVersion(v) => {
                write!(f, "input version {v} is not {INPUT_VERSION}")
            }
            CoprocError::TooManyLegal(n) => write!(f, "{n} legal moves exceeds {MAX_LEGAL}"),
            CoprocError::BadMove {
                index,
                from,
                to,
                promo,
            } => write!(
                f,
                "legal move {index} is out of range: from {from} to {to} promo {promo}"
            ),
            CoprocError::InvalidObservation(m) => write!(f, "invalid observation: {m}"),
            CoprocError::InvalidBoard(m) => write!(f, "invalid reconstructed board: {m}"),
            CoprocError::BadOutputLength(n) => {
                write!(f, "output buffer is {n} bytes, expected {OUTPUT_LEN}")
            }
            CoprocError::BadImageSize(n) => write!(f, "unsupported image size {n}"),
            CoprocError::Capacity(m) => write!(f, "world-model capacity: {m}"),
        }
    }
}

impl std::error::Error for CoprocError {}

/// Convenience: allocate a zeroed fixed-size input buffer.
pub fn empty_input() -> Vec<u8> {
    vec![0u8; INPUT_LEN]
}

/// Convenience: allocate a zeroed fixed-size compute-bank output buffer.
pub fn empty_output() -> Vec<u8> {
    vec![0u8; OUTPUT_LEN]
}

/// Which deterministic-compute implementation a run uses.
///
/// The label lives here (not in the host provider crate) because the *model
/// config* needs it while staying free of the host runtime's dependencies.
/// `NativeV1` and `WasmV1` must be semantically identical: they run the same
/// `compute::compute_bank` source, so only the execution backend differs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ComputeProviderKind {
    /// The compute pathway is switched off; no work is done at all.
    #[default]
    None,
    /// Native host implementation of `ComputeBankV1`.
    NativeV1,
    /// WebAssembly implementation of the same `ComputeBankV1`.
    WasmV1,
}

impl ComputeProviderKind {
    /// The config/identity label.
    pub fn label(self) -> &'static str {
        match self {
            ComputeProviderKind::None => "none",
            ComputeProviderKind::NativeV1 => "native_v1",
            ComputeProviderKind::WasmV1 => "wasm_v1",
        }
    }

    /// Parse a config label, refusing anything unknown.
    pub fn parse(label: &str) -> Result<Self, &'static str> {
        match label {
            "none" => Ok(ComputeProviderKind::None),
            "native_v1" => Ok(ComputeProviderKind::NativeV1),
            "wasm_v1" => Ok(ComputeProviderKind::WasmV1),
            _ => Err("unknown compute provider (expected none, native_v1 or wasm_v1)"),
        }
    }

    /// Whether this provider computes anything.
    pub fn is_active(self) -> bool {
        self != ComputeProviderKind::None
    }
}
