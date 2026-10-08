//! Canonical identities for pawnless, castling-free positions.
//!
//! Symmetry group (order 16): 8 dihedral board symmetries x attacker-colour
//! relabelling. Positions are encoded attacker-relative (piece owner is
//! "attacker" or "defender", not White/Black) together with whether the
//! attacker is to move, so colour-swapped positions coincide. Valid because the
//! domain has no pawns, castling rights, or en-passant, and halfmove clock 0.

use cozy_chess::{Board, Color, Piece, Square};

pub type CanonKey = [u8; 65];

fn transform(idx: usize, t: usize) -> usize {
    let (f, r) = (idx % 8, idx / 8);
    let (f, r) = match t {
        0 => (f, r),
        1 => (7 - f, r),
        2 => (f, 7 - r),
        3 => (7 - f, 7 - r),
        4 => (r, f),
        5 => (7 - r, f),
        6 => (r, 7 - f),
        _ => (7 - r, 7 - f),
    };
    r * 8 + f
}

fn piece_code(p: Piece) -> u8 {
    match p {
        Piece::Pawn => 1,
        Piece::Knight => 2,
        Piece::Bishop => 3,
        Piece::Rook => 4,
        Piece::Queen => 5,
        Piece::King => 6,
    }
}

/// Canonical key of `board` given the attacker colour.
pub fn canonical_key(board: &Board, attacker: Color) -> CanonKey {
    let mut cells = [0u8; 64];
    for sq in Square::ALL {
        if let Some(p) = board.piece_on(sq) {
            let owner = board.color_on(sq).unwrap();
            let code = piece_code(p) + if owner == attacker { 0 } else { 8 };
            cells[sq as usize] = code;
        }
    }
    let attacker_to_move = (board.side_to_move() == attacker) as u8;
    let mut best: Option<CanonKey> = None;
    for t in 0..8 {
        let mut k = [0u8; 65];
        for i in 0..64 {
            k[transform(i, t)] = cells[i];
        }
        k[64] = attacker_to_move;
        if best.as_ref().is_none_or(|b| k < *b) {
            best = Some(k);
        }
    }
    best.unwrap()
}

pub fn key_hex(k: &CanonKey) -> String {
    crate::streams::hex(k)
}

/// Collision-resistant short id from the FULL canonical key (the leading bytes
/// alone are mostly empty squares and must never be used as an id).
pub fn key_id(key_hex: &str) -> String {
    use sha2::{Digest, Sha256};
    crate::streams::hex(&Sha256::digest(key_hex.as_bytes()))[..24].to_string()
}
