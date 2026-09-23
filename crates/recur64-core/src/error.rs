//! Visible errors for the chess contracts. No silent repair or fallback.

use std::fmt;

/// Errors from the Recur64 chess contracts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoreError {
    /// FEN could not be parsed.
    Fen(String),
    /// A move was not legal in the given position.
    IllegalMove(String),
    /// A UCI move string was malformed or illegal.
    InvalidUci(String),
    /// An action index was outside the 20,480-ID space or otherwise invalid.
    InvalidAction(u32),
    /// A promotion code was invalid, or a promotion was required/forbidden.
    InvalidPromotion(String),
    /// The requested side/color/coordinate was invalid.
    InvalidCoordinate(String),
    /// A candidate batch was malformed (duplicate/missing/misdecoded action).
    InvalidCandidates(String),
}

impl fmt::Display for CoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CoreError::Fen(m) => write!(f, "fen error: {m}"),
            CoreError::IllegalMove(m) => write!(f, "illegal move: {m}"),
            CoreError::InvalidUci(m) => write!(f, "invalid uci: {m}"),
            CoreError::InvalidAction(id) => write!(f, "invalid action id: {id}"),
            CoreError::InvalidPromotion(m) => write!(f, "invalid promotion: {m}"),
            CoreError::InvalidCoordinate(m) => write!(f, "invalid coordinate: {m}"),
            CoreError::InvalidCandidates(m) => write!(f, "invalid candidates: {m}"),
        }
    }
}

impl std::error::Error for CoreError {}
