//! Frozen, symmetry-aware train/validation/test split of tablebase positions (the 8 dihedral board symmetries
//! are identified, so no symmetric copy of a test position can appear in training).

use cozy_chess::{Board, Color, Piece};

const PIECES: [Piece; 6] = [
    Piece::Pawn,
    Piece::Knight,
    Piece::Bishop,
    Piece::Rook,
    Piece::Queen,
    Piece::King,
];

fn transform(i: usize, f: i32, r: i32) -> (i32, i32) {
    match i {
        0 => (f, r),
        1 => (7 - f, r),
        2 => (f, 7 - r),
        3 => (7 - f, 7 - r),
        4 => (r, f),
        5 => (7 - r, f),
        6 => (r, 7 - f),
        _ => (7 - r, 7 - f),
    }
}

/// Minimum over the 8 symmetries of an FNV-1a hash of the ordered piece placement and side to move.
pub fn canon_key(b: &Board) -> u64 {
    (0..8)
        .map(|t| {
            let mut sq: Vec<(u8, u8, u8)> = Vec::new();
            for (ci, c) in [Color::White, Color::Black].into_iter().enumerate() {
                for (pi, p) in PIECES.iter().enumerate() {
                    for s in b.colored_pieces(c, *p) {
                        let (f, r) = transform(t, s.file() as i32, s.rank() as i32);
                        sq.push((ci as u8, pi as u8, (r * 8 + f) as u8));
                    }
                }
            }
            sq.sort_unstable();
            let mut h: u64 = 0xcbf2_9ce4_8422_2325;
            for (c, p, s) in sq {
                for byte in [c, p, s] {
                    h = (h ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
                }
            }
            h ^ u64::from(b.side_to_move() == Color::Black)
        })
        .min()
        .expect("eight symmetries")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Part {
    Train,
    Val,
    Test,
}

/// `mod 10`: 0 = test, 1 = validation, 2..=9 = train (frozen).
pub fn part_of(b: &Board) -> Part {
    let h = canon_key(b);
    let m = (h ^ (h >> 29)).wrapping_mul(0xbf58_476d_1ce4_e5b9) % 10;
    match m {
        0 => Part::Test,
        1 => Part::Val,
        _ => Part::Train,
    }
}
