//! Test-only bridge to `recur64-core` and `cozy-chess`. Kept behind
//! `#[cfg(test)]` so the WASM guest never links the game crate.

use cozy_chess::Square;
use recur64_core::GameState;

use crate::input::CoprocMove;

/// Parse a FEN into a game state.
pub fn state_from_fen(fen: &str) -> GameState {
    GameState::from_fen(fen).expect("valid test FEN")
}

/// Observation V1 as a flat float vector.
pub fn encode_observation(state: &GameState) -> Vec<f32> {
    recur64_core::encode_observation_v1(state)
        .as_slice()
        .to_vec()
}

/// The canonical legal candidate list as coprocessor moves.
pub fn legal_moves(state: &GameState) -> Vec<CoprocMove> {
    state
        .legal_actions()
        .iter()
        .map(|a| {
            let (from, to, promo) = a.decode();
            CoprocMove {
                from: from as usize as u8,
                to: to as usize as u8,
                promo: promo.code(),
            }
        })
        .collect()
}

/// The canonical square indices of the legal candidates.
#[allow(dead_code)]
pub fn legal_squares(state: &GameState) -> Vec<(u8, u8)> {
    legal_moves(state)
        .into_iter()
        .map(|m| (m.from, m.to))
        .collect()
}

/// `cozy-chess` square from an index.
#[allow(dead_code)]
pub fn square(index: usize) -> Square {
    Square::index(index)
}
