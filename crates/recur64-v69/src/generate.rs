//! Fresh root generation: KQQvK, KQRvK, KRRvK, attacker to move.
//!
//! Domain: pawnless, no castling/en-passant, halfmove clock 0, fullmove 1, no
//! prior history. Squares are drawn independently and uniformly per piece;
//! overlaps, adjacent kings and a defender king already in check (illegal with
//! the attacker to move) are rejected and counted. Nothing here reads data.

use crate::streams::{MasterSeed, STREAM_GENERATION};
use cozy_chess::{Board, Color, Piece, Square, get_king_moves};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Family {
    Kqq,
    Kqr,
    Krr,
}

impl Family {
    pub const ALL: [Family; 3] = [Family::Kqq, Family::Kqr, Family::Krr];
    pub fn name(self) -> &'static str {
        match self {
            Family::Kqq => "KQQvK",
            Family::Kqr => "KQRvK",
            Family::Krr => "KRRvK",
        }
    }
    pub fn index(self) -> u64 {
        self as u64
    }
    fn pieces(self) -> [Piece; 2] {
        match self {
            Family::Kqq => [Piece::Queen, Piece::Queen],
            Family::Kqr => [Piece::Queen, Piece::Rook],
            Family::Krr => [Piece::Rook, Piece::Rook],
        }
    }
}

#[derive(Debug)]
pub enum Reject {
    Overlap,
    AdjacentKings,
    DefenderInCheck,
    BoardInvalid,
}

pub struct RootSample {
    pub board: Board,
    pub attacker: Color,
}

/// Attempt number `index` of `family` under the master seed. Pure function of
/// (seed, family, index): thread count and scheduling cannot change it.
pub fn sample_root(seed: &MasterSeed, family: Family, index: u64) -> Result<RootSample, Reject> {
    let label = format!("{STREAM_GENERATION}/{}", family.name());
    let mut rng = seed.stream(&label, index);
    let attacker = if rng.below(2) == 0 { Color::White } else { Color::Black };
    let defender = !attacker;
    let [p1, p2] = family.pieces();
    let ak = Square::index(rng.below(64) as usize);
    let a1 = Square::index(rng.below(64) as usize);
    let a2 = Square::index(rng.below(64) as usize);
    let dk = Square::index(rng.below(64) as usize);
    let sqs = [ak, a1, a2, dk];
    for i in 0..4 {
        for j in i + 1..4 {
            if sqs[i] == sqs[j] {
                return Err(Reject::Overlap);
            }
        }
    }
    if get_king_moves(ak).has(dk) {
        return Err(Reject::AdjacentKings);
    }
    // Build through the FEN path so cozy-chess validation (rule authority)
    // adjudicates legality; the defender-in-check case is classified first for
    // accurate counters.
    let mut grid = [[None::<(Piece, Color)>; 8]; 8];
    let put = |g: &mut [[Option<(Piece, Color)>; 8]; 8], s: Square, p: Piece, c: Color| {
        g[s.rank() as usize][s.file() as usize] = Some((p, c));
    };
    put(&mut grid, ak, Piece::King, attacker);
    put(&mut grid, a1, p1, attacker);
    put(&mut grid, a2, p2, attacker);
    put(&mut grid, dk, Piece::King, defender);
    let mut fen = String::new();
    for r in (0..8).rev() {
        let mut empty = 0;
        for f in 0..8 {
            match grid[r][f] {
                None => empty += 1,
                Some((p, c)) => {
                    if empty > 0 {
                        fen.push_str(&empty.to_string());
                        empty = 0;
                    }
                    let ch = match p {
                        Piece::King => 'k',
                        Piece::Queen => 'q',
                        Piece::Rook => 'r',
                        _ => unreachable!(),
                    };
                    fen.push(if c == Color::White { ch.to_ascii_uppercase() } else { ch });
                }
            }
        }
        if empty > 0 {
            fen.push_str(&empty.to_string());
        }
        if r > 0 {
            fen.push('/');
        }
    }
    fen.push_str(if attacker == Color::White { " w - - 0 1" } else { " b - - 0 1" });
    match fen.parse::<Board>() {
        Ok(b) => Ok(RootSample { board: b, attacker }),
        Err(_) => {
            // Distinguish "defender already in check" from other invalidity by
            // checking attack on the defender king independently of the parser.
            let flipped = fen.replace(" w ", " X ").replace(" b ", " w ").replace(" X ", " b ");
            if flipped.parse::<Board>().is_ok() {
                Err(Reject::DefenderInCheck)
            } else {
                Err(Reject::BoardInvalid)
            }
        }
    }
}
