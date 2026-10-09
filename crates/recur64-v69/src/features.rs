//! Model-facing featurization for the V69 forced-mate task.
//!
//! Input contract (frozen in docs/v69/MODEL_SPEC.md): every example is an
//! immediate nonterminal child with the DEFENDER to move; the attacker is
//! therefore derived as the side NOT to move (no root metadata, ids, row order or
//! labels are consulted). Piece ownership is attacker/defender-relative, and the
//! board is rank-flipped when the attacker is Black so the attacker always
//! "plays up the board" (true symmetry in this pawnless domain). Anything outside
//! the domain fails visibly.

use anyhow::{Result, bail, ensure};
use cozy_chess::{Board, Color, Piece, Square};
use serde::Deserialize;

pub const N_PIECE_CODES: usize = 13;
pub const N_SCALARS: usize = 8;

/// Model-facing row. Unknown JSON fields are rejected.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelRow {
    pub id: String,
    pub fen: String,
    pub budget: u8,
    pub label: bool,
}

/// Inputs only. Deliberately has no label, id or row-order field.
#[derive(Debug, Clone, PartialEq)]
pub struct Features {
    /// 0 = empty; 1..=6 attacker P,N,B,R,Q,K; 7..=12 defender P,N,B,R,Q,K.
    pub piece: [u8; 64],
    pub scalars: [f32; N_SCALARS],
    /// Remaining attacker-move budget (1 or 2).
    pub budget: u8,
    /// 1 if the attacker is to move, else 0 (always 0 in this task).
    pub attacker_to_move: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum FamilyKind {
    Kqq,
    Kqr,
    Krr,
}

impl FamilyKind {
    pub fn name(self) -> &'static str {
        match self {
            FamilyKind::Kqq => "KQQvK",
            FamilyKind::Kqr => "KQRvK",
            FamilyKind::Krr => "KRRvK",
        }
    }
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

/// Parse and validate a child FEN; returns features and its family.
pub fn featurize(fen: &str, budget: u8) -> Result<(Features, FamilyKind)> {
    ensure!(fen.split_whitespace().count() == 6, "FEN must have exactly 6 fields: {fen:?}");
    ensure!(matches!(budget, 1 | 2), "unsupported budget {budget} (only 1 or 2)");
    let b: Board = fen.parse().map_err(|e| anyhow::anyhow!("invalid FEN {fen:?}: {e}"))?;
    let defender = b.side_to_move();
    let attacker = !defender;
    // Domain validation.
    let att = b.colors(attacker);
    let def = b.colors(defender);
    ensure!(def.len() == 1 && !(b.pieces(Piece::King) & def).is_empty(), "defender must have a lone king: {fen}");
    ensure!(!(b.pieces(Piece::King) & att).is_empty(), "attacker king missing");
    ensure!(b.pieces(Piece::Pawn).is_empty() && b.pieces(Piece::Knight).is_empty() && b.pieces(Piece::Bishop).is_empty(), "unsupported piece type in {fen}");
    let q = (b.pieces(Piece::Queen) & att).len();
    let r = (b.pieces(Piece::Rook) & att).len();
    let family = match (q, r) {
        (2, 0) => FamilyKind::Kqq,
        (1, 1) => FamilyKind::Kqr,
        (0, 2) => FamilyKind::Krr,
        _ => bail!("unsupported attacker material Q={q} R={r} in {fen}"),
    };
    ensure!(b.halfmove_clock() <= 1, "halfmove clock {} outside domain", b.halfmove_clock());
    ensure!(b.en_passant().is_none(), "en-passant outside domain");
    for c in [Color::White, Color::Black] {
        let rights = b.castle_rights(c);
        ensure!(rights.short.is_none() && rights.long.is_none(), "castling rights outside domain");
    }
    ensure!(recur64_core::rules::classify(&b, 1, 0, None).is_none(), "terminal child in input: {fen}");

    let flip = attacker == Color::Black;
    let mut piece = [0u8; 64];
    for sq in Square::ALL {
        if let Some(p) = b.piece_on(sq) {
            let owner_def = b.color_on(sq).unwrap() == defender;
            let code = piece_code(p) + if owner_def { 6 } else { 0 };
            let (f, rk) = (sq.file() as usize, sq.rank() as usize);
            let rk = if flip { 7 - rk } else { rk };
            piece[rk * 8 + f] = code;
        }
    }
    let attacker_to_move = (b.side_to_move() == attacker) as u8;
    let scalars = [
        attacker_to_move as f32,
        (1 - attacker_to_move) as f32,
        b.halfmove_clock() as f32 / 100.0,
        0.0, // attacker king-side castling (domain: none)
        0.0, // attacker queen-side
        0.0, // defender king-side
        0.0, // defender queen-side
        0.0, // en-passant present
    ];
    Ok((Features { piece, scalars, budget, attacker_to_move }, family))
}

impl Features {
    /// Board-erasure diagnostic input: all squares empty; retained legitimate
    /// task metadata = remaining budget, attacker-to-move indicator, and rule
    /// scalars (all constants of the task contract).
    pub fn erased(&self) -> Features {
        Features { piece: [0u8; 64], ..self.clone() }
    }
}

pub fn read_rows(text: &str) -> Result<Vec<ModelRow>> {
    let mut rows = Vec::new();
    let mut ids = std::collections::HashSet::new();
    for (i, line) in text.lines().enumerate() {
        let r: ModelRow = serde_json::from_str(line).map_err(|e| anyhow::anyhow!("row {i}: {e}"))?;
        ensure!(ids.insert(r.id.clone()), "duplicate id {}", r.id);
        rows.push(r);
    }
    Ok(rows)
}

/// Square index after applying element `t` (0..8) of the dihedral group D8 to (file, rank).
pub fn d8_square(idx: usize, t: usize) -> usize {
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

impl Features {
    /// Label-preserving augmentation: apply D8 element `t` to the attacker-relative piece grid.
    /// (The pawnless, castling-free domain is invariant under all 8 board symmetries; the colour
    /// relabelling half of the 16-fold group is already quotiented out by `featurize`.)
    pub fn d8(&self, t: usize) -> Features {
        let mut piece = [0u8; 64];
        for sq in 0..64 {
            piece[d8_square(sq, t)] = self.piece[sq];
        }
        Features { piece, ..self.clone() }
    }
}

/// FEN of the position with the board geometrically transformed by `t` (same pieces/colours, same side to move).
pub fn transform_fen(fen: &str, t: usize) -> Result<String> {
    let b: Board = fen.parse().map_err(|e| anyhow::anyhow!("fen: {e}"))?;
    let mut grid = [[' '; 8]; 8];
    for sq in Square::ALL {
        if let Some(p) = b.piece_on(sq) {
            let c = match p {
                Piece::King => 'k',
                Piece::Queen => 'q',
                Piece::Rook => 'r',
                Piece::Bishop => 'b',
                Piece::Knight => 'n',
                Piece::Pawn => 'p',
            };
            let ch = if b.color_on(sq).unwrap() == Color::White { c.to_ascii_uppercase() } else { c };
            let ni = d8_square(sq.rank() as usize * 8 + sq.file() as usize, t);
            grid[ni / 8][ni % 8] = ch;
        }
    }
    let mut out = String::new();
    for r in (0..8).rev() {
        let mut e = 0;
        for f in 0..8 {
            if grid[r][f] == ' ' {
                e += 1;
            } else {
                if e > 0 {
                    out.push_str(&e.to_string());
                    e = 0;
                }
                out.push(grid[r][f]);
            }
        }
        if e > 0 {
            out.push_str(&e.to_string());
        }
        if r > 0 {
            out.push('/');
        }
    }
    let stm = if b.side_to_move() == Color::White { 'w' } else { 'b' };
    Ok(format!("{out} {stm} - - {} 1", b.halfmove_clock()))
}
