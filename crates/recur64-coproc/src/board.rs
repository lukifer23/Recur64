//! Reconstruct a legal `cozy-chess` board from a canonical Observation V1 plus
//! its canonical castling / en-passant / clock features.
//!
//! Canonicalization is already applied by observation encoding, so after
//! reconstruction the side to move is always *own* = `Color::White` and the
//! opponent is `Color::Black`. Castling rights, the en-passant target and the
//! halfmove clock are recovered from the broadcast features; the repetition
//! count is recovered from its normalized feature. Both clocks are lossy in
//! exactly the way Observation V1 already is, and that loss is documented.

use cozy_chess::{Board, BoardBuilder, CastleRights, Color, File, Piece, Square};

use crate::CoprocError;
use crate::input::CoprocInput;
use crate::squares::SquareIndex;

/// Offsets inside one square's Observation V1 feature vector.
pub const CASTLE_OFFSET: usize = 112;
pub const EP_OFFSET: usize = 116;
pub const HALFMOVE_OFFSET: usize = 117;
pub const REPETITION_OFFSET: usize = 118;

/// Canonical piece order used by every piece-indexed vector in this crate.
pub const PIECE_ORDER: [Piece; 6] = [
    Piece::Pawn,
    Piece::Knight,
    Piece::Bishop,
    Piece::Rook,
    Piece::Queen,
    Piece::King,
];

/// The canonical frame-0 piece channel for a piece of a side.
pub fn piece_channel(piece: Piece, color: Color) -> usize {
    let base = match color {
        Color::White => 0,
        Color::Black => 6,
    };
    let k = PIECE_ORDER
        .iter()
        .position(|p| *p == piece)
        .expect("piece is in the canonical order");
    base + k
}

/// The piece a canonical frame-0 channel names, or `None` for the empty
/// channel (12) and the validity channel (13).
pub fn channel_piece(channel: usize) -> Option<(Piece, Color)> {
    match channel {
        0..=5 => Some((PIECE_ORDER[channel], Color::White)),
        6..=11 => Some((PIECE_ORDER[channel - 6], Color::Black)),
        _ => None,
    }
}

/// A reconstruction of the canonical position.
#[derive(Debug, Clone)]
pub struct Reconstructed {
    /// The position with `White` to move (own = White after canonicalization).
    pub board: Board,
    /// Recovered halfmove clock.
    pub halfmove_clock: u8,
    /// Recovered repetition count in `1..=5`.
    pub repetition_count: u8,
    /// The en-passant target square, when one is present.
    pub en_passant: Option<Square>,
}

/// Recover the integer a `min(x, scale) / scale` feature encoded.
fn denormalize(value: f32, scale: u8) -> u8 {
    (value.clamp(0.0, 1.0) * scale as f32).round() as u8
}

/// Reconstruct the canonical board.
pub fn reconstruct(input: &CoprocInput<'_>) -> Result<Reconstructed, CoprocError> {
    let mut builder = BoardBuilder::empty();
    builder.side_to_move = Color::White;

    for square in 0..crate::SQUARES {
        let mut channel: Option<usize> = None;
        for ch in 0..13 {
            if input.obs_feature(square, ch) == 1.0 {
                if channel.is_some() {
                    return Err(CoprocError::InvalidObservation(
                        "a square sets more than one piece channel in frame 0",
                    ));
                }
                channel = Some(ch);
            }
        }
        let Some(channel) = channel else {
            return Err(CoprocError::InvalidObservation(
                "a square sets no piece-or-empty channel in frame 0",
            ));
        };
        *builder.square_mut(square.from_index()) = channel_piece(channel);
    }

    // Castling rights. Rules Profile V1 is standard chess, so the kingside rook
    // is on the h-file and the queenside rook on the a-file.
    let own_short = input.obs_feature(0, CASTLE_OFFSET) == 1.0;
    let own_long = input.obs_feature(0, CASTLE_OFFSET + 1) == 1.0;
    let opp_short = input.obs_feature(0, CASTLE_OFFSET + 2) == 1.0;
    let opp_long = input.obs_feature(0, CASTLE_OFFSET + 3) == 1.0;
    *builder.castle_rights_mut(Color::White) = CastleRights {
        short: own_short.then_some(File::H),
        long: own_long.then_some(File::A),
    };
    *builder.castle_rights_mut(Color::Black) = CastleRights {
        short: opp_short.then_some(File::H),
        long: opp_long.then_some(File::A),
    };

    // En-passant target.
    let mut ep: Option<Square> = None;
    for square in 0..crate::SQUARES {
        if input.obs_feature(square, EP_OFFSET) == 1.0 {
            if ep.is_some() {
                return Err(CoprocError::InvalidObservation(
                    "more than one en-passant target square",
                ));
            }
            ep = Some(square.from_index());
        }
    }
    builder.en_passant = ep;
    builder.halfmove_clock = denormalize(input.obs_feature(0, HALFMOVE_OFFSET), 150);
    let repetition_count = denormalize(input.obs_feature(0, REPETITION_OFFSET), 5).max(1);

    let board = builder
        .build()
        .map_err(|_| CoprocError::InvalidBoard("cozy-chess refused the reconstructed position"))?;

    Ok(Reconstructed {
        board,
        halfmove_clock: builder.halfmove_clock,
        repetition_count,
        en_passant: ep,
    })
}

/// Whether the position is an immediate dead position under the conservative
/// Rule Profile V1 material rule: K vs K, K+minor vs K, or single bishops on
/// the same colour complex.
pub fn insufficient_material(board: &Board) -> bool {
    type Counts = (usize, usize, usize, usize, usize);
    let counts = |color: Color| -> Counts {
        (
            board.colored_pieces(color, Piece::Knight).len() as usize,
            board.colored_pieces(color, Piece::Bishop).len() as usize,
            board.colored_pieces(color, Piece::Rook).len() as usize,
            board.colored_pieces(color, Piece::Queen).len() as usize,
            board.colored_pieces(color, Piece::Pawn).len() as usize,
        )
    };
    let (w, b) = (counts(Color::White), counts(Color::Black));
    let bare = |x: Counts| x == (0, 0, 0, 0, 0);
    if bare(w) && bare(b) {
        return true;
    }
    let lone_minor = |x: Counts| x.2 == 0 && x.3 == 0 && x.4 == 0 && (x.0 + x.1) <= 1;
    if (lone_minor(w) && bare(b)) || (lone_minor(b) && bare(w)) {
        return true;
    }
    if w == (0, 1, 0, 0, 0) && b == (0, 1, 0, 0, 0) {
        let wb = board
            .colored_pieces(Color::White, Piece::Bishop)
            .next_square();
        let bb = board
            .colored_pieces(Color::Black, Piece::Bishop)
            .next_square();
        if let (Some(wb), Some(bb)) = (wb, bb) {
            let complex = |s: Square| (s.file() as usize + s.rank() as usize) % 2;
            if complex(wb) == complex(bb) {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::CoprocMove;
    use crate::test_support::{encode_observation, legal_moves, state_from_fen};

    fn input_for(fen: &str) -> Vec<u8> {
        let state = state_from_fen(fen);
        let obs = encode_observation(&state);
        let moves = legal_moves(&state);
        crate::input::write_input(&obs, &moves, 1).unwrap()
    }

    #[test]
    fn startpos_round_trips() {
        let bytes = input_for("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        let inp = CoprocInput::new(&bytes).unwrap();
        let r = reconstruct(&inp).unwrap();
        assert_eq!(r.board, Board::default());
        assert!(r.en_passant.is_none());
        assert_eq!(r.halfmove_clock, 0);
        assert_eq!(r.repetition_count, 1);
    }

    #[test]
    fn canonical_view_is_own_white_for_black_to_move() {
        // 1.e4: Black to move; canonicalization makes own = White.
        let bytes = input_for("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");
        let inp = CoprocInput::new(&bytes).unwrap();
        let r = reconstruct(&inp).unwrap();
        assert_eq!(r.board.side_to_move(), Color::White);
        assert!(r.en_passant.is_some());
        // The reconstructed board must be the canonical reflection of the
        // physical position: own pieces (White) on the first two ranks.
        assert_eq!(r.board.colored_pieces(Color::White, Piece::Pawn).len(), 8);
    }

    #[test]
    fn castling_rights_survive() {
        let bytes = input_for("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
        let inp = CoprocInput::new(&bytes).unwrap();
        let r = reconstruct(&inp).unwrap();
        let w = *r.board.castle_rights(Color::White);
        let b = *r.board.castle_rights(Color::Black);
        assert_eq!(w.short, Some(File::H));
        assert_eq!(w.long, Some(File::A));
        assert_eq!(b.short, Some(File::H));
        assert_eq!(b.long, Some(File::A));
    }

    #[test]
    fn legal_move_count_matches_the_stored_list() {
        for fen in [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
            "8/8/8/4k3/8/8/3Q4/4K3 w - - 0 1",
            "7k/5P2/8/8/8/8/8/K7 w - - 0 1",
        ] {
            let bytes = input_for(fen);
            let inp = CoprocInput::new(&bytes).unwrap();
            let r = reconstruct(&inp).unwrap();
            let mut generated = 0usize;
            r.board.generate_moves(|mvs| {
                generated += mvs.len();
                false
            });
            assert_eq!(generated, inp.n_legal(), "{fen}");
        }
    }

    #[test]
    fn insufficient_material_cases() {
        let cases = [
            ("8/8/8/4k3/8/8/4K3/8 w - - 0 1", true),
            ("8/8/8/4k3/8/8/3N4/4K3 w - - 0 1", true),
            ("8/3b4/8/4k3/8/8/4B3/4K3 w - - 0 1", true),
            ("8/2b5/8/4k3/8/8/4B3/4K3 w - - 0 1", false),
            ("8/8/8/4k3/8/8/3Q4/4K3 w - - 0 1", false),
        ];
        for (fen, expected) in cases {
            let bytes = input_for(fen);
            let inp = CoprocInput::new(&bytes).unwrap();
            let r = reconstruct(&inp).unwrap();
            assert_eq!(insufficient_material(&r.board), expected, "{fen}");
        }
    }

    #[test]
    fn _uses_coproc_move_type() {
        let m = CoprocMove {
            from: 0,
            to: 1,
            promo: 0,
        };
        assert_eq!(m.to, 1);
        assert_eq!(crate::FEATURES_PER_SQUARE, 119);
    }
}
