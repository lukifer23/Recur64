//! Observation V1.
//!
//! The network input before projection is `[batch, 64, 119]`. For each square:
//!
//! - 8 history frames, each `13` piece one-hot channels **including empty** plus
//!   `1` validity bit: `8 * 14 = 112`.
//! - 4 current castling-right indicators (own kingside/queenside, opponent
//!   kingside/queenside), broadcast to every square.
//! - 1 current en-passant-target indicator at the relevant square (FEN-style).
//! - 1 halfmove-clock feature `min(clock, 150) / 150`, broadcast.
//! - 1 repetition-count feature `min(count, 5) / 5`, broadcast.
//!
//! Total `112 + 4 + 1 + 1 + 1 = 119`.
//!
//! Frame 0 is the current position, frame 1 the preceding ply, and so on. The
//! **current** side to move determines canonicalization for the entire
//! observation, applied identically to every frame.

use cozy_chess::{Rank, Square};

use crate::game::GameState;
use crate::square::{Color, Piece};

/// Number of squares.
pub const NUM_SQUARES: usize = 64;
/// Number of history frames.
pub const NUM_FRAMES: usize = 8;
/// Floats per history frame (13 piece channels + validity).
pub const FRAME_LEN: usize = 14;
/// Piece one-hot channels, including empty.
pub const PIECE_CHANNELS: usize = 13;
/// Floats per square.
pub const FEATURES_PER_SQUARE: usize = 119;
/// Total observation length.
pub const OBS_LEN: usize = NUM_SQUARES * FEATURES_PER_SQUARE;

/// Offset of the castling indicators within a square's feature vector.
pub const CASTLE_OFFSET: usize = NUM_FRAMES * FRAME_LEN; // 112
/// Offset of the en-passant indicator.
pub const EP_OFFSET: usize = CASTLE_OFFSET + 4; // 116
/// Offset of the halfmove-clock feature.
pub const HALFMOVE_OFFSET: usize = EP_OFFSET + 1; // 117
/// Offset of the repetition-count feature.
pub const REPETITION_OFFSET: usize = HALFMOVE_OFFSET + 1; // 118

/// Piece channel index for a piece of canonical color.
///
/// Canonical color `White` means "the side to move" (own pieces, 0..5);
/// canonical `Black` means the opponent (6..11); 12 is empty.
fn piece_channel(piece: Piece, canonical_color: Color) -> usize {
    let base = if canonical_color == Color::White {
        0
    } else {
        6
    };
    base + match piece {
        Piece::Pawn => 0,
        Piece::Knight => 1,
        Piece::Bishop => 2,
        Piece::Rook => 3,
        Piece::Queen => 4,
        Piece::King => 5,
    }
}

/// A fixed-size Observation V1 tensor, laid out `[square][feature]`.
#[derive(Clone)]
pub struct ObservationV1(pub [f32; OBS_LEN]);

impl Default for ObservationV1 {
    fn default() -> Self {
        Self([0.0; OBS_LEN])
    }
}

impl ObservationV1 {
    pub fn zeroed() -> Self {
        Self::default()
    }

    /// Read a feature for a canonical square.
    pub fn get(&self, square: usize, feature: usize) -> f32 {
        self.0[square * FEATURES_PER_SQUARE + feature]
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.0
    }

    /// Write into a caller-provided buffer (for batch assembly).
    pub fn encode_into(&self, out: &mut [f32]) {
        out[..OBS_LEN].copy_from_slice(&self.0);
    }
}

/// Encode the current position of `state` as Observation V1.
pub fn encode_observation_v1(state: &GameState) -> ObservationV1 {
    let mut obs = ObservationV1::zeroed();
    let p = state.perspective();
    let board = state.board();
    let side = state.side_to_move();
    let hist = state.history();
    let nframes = hist.len().min(NUM_FRAMES);

    // Castling: own = side to move, opponent = the other side.
    let own = board.castle_rights(side);
    let opp = board.castle_rights(!side);
    let castle = [
        own.short.is_some(),
        own.long.is_some(),
        opp.short.is_some(),
        opp.long.is_some(),
    ];

    // En-passant target square (FEN-style: present after any double pawn push).
    let ep_square: Option<Square> = board.en_passant().map(|file| {
        let rank = if side == Color::White {
            Rank::Sixth
        } else {
            Rank::Third
        };
        Square::new(file, rank)
    });

    let halfmove = (board.halfmove_clock().min(150) as f32) / 150.0;
    let repetition = (state.repetition_count().min(5) as f32) / 5.0;

    for cs in 0..NUM_SQUARES {
        let physical = p.square(Square::index(cs));
        let base = cs * FEATURES_PER_SQUARE;

        for k in 0..NUM_FRAMES {
            let fbase = base + k * FRAME_LEN;
            if k < nframes {
                let frame_board = &hist[hist.len() - 1 - k];
                obs.0[fbase + 13] = 1.0; // validity
                match frame_board.piece_on(physical) {
                    Some(piece) => {
                        let color = frame_board
                            .color_on(physical)
                            .expect("piece_on implies color_on");
                        let canonical = p.color(color);
                        obs.0[fbase + piece_channel(piece, canonical)] = 1.0;
                    }
                    None => obs.0[fbase + 12] = 1.0, // empty
                }
            }
            // Unavailable frames remain all-zero (channels 0..12 and validity).
        }

        for (i, present) in castle.iter().enumerate() {
            obs.0[base + CASTLE_OFFSET + i] = if *present { 1.0 } else { 0.0 };
        }

        if ep_square.is_some_and(|sq| p.square(sq) == Square::index(cs)) {
            obs.0[base + EP_OFFSET] = 1.0;
        }

        obs.0[base + HALFMOVE_OFFSET] = halfmove;
        obs.0[base + REPETITION_OFFSET] = repetition;
    }

    obs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_arithmetic() {
        assert_eq!(NUM_FRAMES * FRAME_LEN, 112);
        assert_eq!(112 + 4 + 1 + 1 + 1, FEATURES_PER_SQUARE);
        assert_eq!(FEATURES_PER_SQUARE, 119);
        assert_eq!(OBS_LEN, 7616);
        assert_eq!(CASTLE_OFFSET, 112);
        assert_eq!(EP_OFFSET, 116);
        assert_eq!(HALFMOVE_OFFSET, 117);
        assert_eq!(REPETITION_OFFSET, 118);
    }

    #[test]
    fn startpos_white_identity() {
        let g = GameState::startpos();
        let obs = encode_observation_v1(&g);

        // White king on e1 (canonical square 4) -> own king channel 5.
        assert_eq!(obs.get(4, 5), 1.0);
        // White pawn on e2 (canonical square 12) -> own pawn channel 0.
        assert_eq!(obs.get(12, 0), 1.0);
        // Black pawn on e7 (canonical square 52) -> opponent pawn channel 6.
        assert_eq!(obs.get(52, 6), 1.0);
        // Empty square a3 (canonical 16) -> empty channel 12.
        assert_eq!(obs.get(16, 12), 1.0);
        // All castling rights present.
        assert_eq!(obs.get(0, CASTLE_OFFSET), 1.0);
        assert_eq!(obs.get(0, CASTLE_OFFSET + 1), 1.0);
        assert_eq!(obs.get(0, CASTLE_OFFSET + 2), 1.0);
        assert_eq!(obs.get(0, CASTLE_OFFSET + 3), 1.0);
        // No en passant.
        assert_eq!(obs.get(0, EP_OFFSET), 0.0);
        // Halfmove 0; repetition count 1 -> 0.2.
        assert_eq!(obs.get(0, HALFMOVE_OFFSET), 0.0);
        assert!((obs.get(0, REPETITION_OFFSET) - 0.2).abs() < 1e-6);
    }

    #[test]
    fn after_e4_black_canonicalization() {
        let mut g = GameState::startpos();
        g.apply_uci("e2e4").unwrap();
        let obs = encode_observation_v1(&g);

        // White pawn now on e4 (physical 28) -> canonical 28^56 = 36, and from
        // Black's perspective it is an *opponent* pawn -> channel 6.
        assert_eq!(obs.get(36, 6), 1.0);
        // Black king e8 (physical 60) -> canonical 4, own king channel 5.
        assert_eq!(obs.get(4, 5), 1.0);
        // En passant target e3 (physical 20) -> canonical 44, FEN-style.
        assert_eq!(obs.get(44, EP_OFFSET), 1.0);
        // Own (black) castling rights still present.
        assert_eq!(obs.get(0, CASTLE_OFFSET), 1.0);
        assert_eq!(obs.get(0, CASTLE_OFFSET + 1), 1.0);
    }

    #[test]
    fn unavailable_frames_are_zero_not_empty_boards() {
        let g = GameState::startpos();
        let obs = encode_observation_v1(&g);
        // Only frame 0 exists; frames 1..7 must be all-zero, validity 0.
        for k in 1..NUM_FRAMES {
            for cs in 0..NUM_SQUARES {
                let fbase = cs * FEATURES_PER_SQUARE + k * FRAME_LEN;
                for ch in 0..FRAME_LEN {
                    assert_eq!(obs.0[fbase + ch], 0.0, "frame {k} square {cs} ch {ch}");
                }
            }
        }
        // Specifically, the empty channel of frame 1 is NOT set.
        assert_eq!(obs.get(16, FRAME_LEN + 12), 0.0);
    }

    #[test]
    fn history_frames_use_current_perspective() {
        // After 1.e4, frame 1 is the start position, canonicalized from Black's
        // perspective (current side to move). The white king on e1 (physical 4)
        // becomes canonical 60 and is an *opponent* king (channel 11).
        let mut g = GameState::startpos();
        g.apply_uci("e2e4").unwrap();
        let obs = encode_observation_v1(&g);
        assert_eq!(obs.get(60, FRAME_LEN + 11), 1.0);
        // And the black king e8 in frame 1 is own (channel 5) at canonical 4.
        assert_eq!(obs.get(4, FRAME_LEN + 5), 1.0);
    }

    #[test]
    fn all_features_finite_and_bounded() {
        let mut g = GameState::startpos();
        g.apply_uci_seq(&["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"])
            .unwrap();
        let obs = encode_observation_v1(&g);
        for v in obs.as_slice() {
            assert!(v.is_finite());
            assert!(*v >= 0.0 && *v <= 1.0);
        }
    }
}
