//! The learned ordering function: an NNUE-style value network over piece-square tokens that predicts
//! `log2(1 + plies to mate)` of a position (a drawn/dead outcome is `DRAW_VALUE`). It is trained with burn
//! (`examples/train_net.rs`) and deployed as this plain Rust forward pass so the prover can afford one evaluation
//! per candidate move. A parity test checks the Rust forward against the burn forward.
//!
//! Lower value = closer to mate for the attacker. The attacker orders moves by ascending value of the resulting
//! position; the defender orders replies by DESCENDING value (the most resilient reply first, which is also the
//! most likely refutation of a wrong attacker move).

use cozy_chess::{Board, Color, GameStatus, Move, Piece};
use serde::{Deserialize, Serialize};

use crate::search::Orderer;
use crate::tb::moves_of;

/// Planes: white K, Q, R, B, N, then black K.
pub const PLANES: usize = 6;
/// Piece-square tokens.
pub const SQ_TOKENS: usize = PLANES * 64;
/// Relative-displacement tokens: for each of (wK-bK, piece-bK, piece-wK) a 15x15 grid of (dfile, drank).
pub const REL_GRID: usize = 15 * 15;
pub const REL_KINDS: usize = 3;
pub const TOKENS: usize = SQ_TOKENS + REL_KINDS * REL_GRID;
pub const EXTRA: usize = 3;
/// At most 4 piece-square tokens plus 5 relative tokens (two extra pieces) fit in 10 slots.
pub const MAXP: usize = 10;
/// Value of a position with no forced mate (a draw or a dead position): `log2(1 + 64)`.
pub const DRAW_PLIES: f32 = 64.0;

pub fn target(plies: Option<u8>) -> f32 {
    (1.0 + plies.map_or(DRAW_PLIES, f32::from)).log2()
}

fn plane(color: Color, piece: Piece) -> Option<usize> {
    match (color, piece) {
        (Color::White, Piece::King) => Some(0),
        (Color::White, Piece::Queen) => Some(1),
        (Color::White, Piece::Rook) => Some(2),
        (Color::White, Piece::Bishop) => Some(3),
        (Color::White, Piece::Knight) => Some(4),
        (Color::Black, Piece::King) => Some(5),
        _ => None,
    }
}

fn rel(a: (i32, i32), b: (i32, i32), kind: usize) -> u16 {
    let (df, dr) = (a.0 - b.0 + 7, a.1 - b.1 + 7);
    (SQ_TOKENS + kind * REL_GRID) as u16 + (df * 15 + dr) as u16
}

/// Piece-square tokens followed by king-relative displacement tokens (translation-invariant relations the
/// network can learn from: where the pieces are relative to each king). At most [`MAXP`]; returns the count.
pub fn tokens(b: &Board) -> ([u16; MAXP], usize) {
    let mut t = [0u16; MAXP];
    let mut n = 0;
    let mut push = |v: u16, n: &mut usize| {
        assert!(*n < MAXP, "more than {MAXP} tokens: the V500 network covers at most K+2 pieces vs K");
        t[*n] = v;
        *n += 1;
    };
    let mut others: Vec<(i32, i32)> = Vec::new();
    for color in [Color::White, Color::Black] {
        for piece in [Piece::King, Piece::Queen, Piece::Rook, Piece::Bishop, Piece::Knight] {
            if let Some(p) = plane(color, piece) {
                for sq in b.colored_pieces(color, piece) {
                    push((p * 64 + sq as usize) as u16, &mut n);
                    if color == Color::White && piece != Piece::King {
                        others.push((sq.file() as i32, sq.rank() as i32));
                    }
                }
            }
        }
    }
    let k = |s: cozy_chess::Square| (s.file() as i32, s.rank() as i32);
    let (wk, bk) = (k(b.king(Color::White)), k(b.king(Color::Black)));
    push(rel(wk, bk, 0), &mut n);
    for &o in &others {
        push(rel(o, bk, 1), &mut n);
        push(rel(o, wk, 2), &mut n);
    }
    (t, n)
}

/// Side to move is black, in check, and the number of legal moves / 32.
pub fn extras(b: &Board) -> [f32; EXTRA] {
    [
        f32::from(b.side_to_move() == Color::Black),
        f32::from(!b.checkers().is_empty()),
        moves_of(b).len() as f32 / 32.0,
    ]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetFile {
    pub schema: String,
    pub h1: usize,
    pub h2: usize,
    /// `[TOKENS][h1]` row-major.
    pub emb: Vec<f32>,
    /// `[EXTRA][h1]` row-major.
    pub we: Vec<f32>,
    pub b1: Vec<f32>,
    /// `[h1][h2]` row-major.
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
    pub w3: Vec<f32>,
    pub b3: f32,
}

pub const NET_SCHEMA: &str = "v500_net_v2";

#[derive(Debug, Clone)]
pub struct Mlp {
    f: NetFile,
}

impl Mlp {
    pub fn new(f: NetFile) -> anyhow::Result<Self> {
        anyhow::ensure!(f.schema == NET_SCHEMA, "net schema {}", f.schema);
        anyhow::ensure!(
            f.emb.len() == TOKENS * f.h1
                && f.we.len() == EXTRA * f.h1
                && f.b1.len() == f.h1
                && f.w2.len() == f.h1 * f.h2
                && f.b2.len() == f.h2
                && f.w3.len() == f.h2,
            "net tensors do not match h1={} h2={}",
            f.h1,
            f.h2
        );
        Ok(Self { f })
    }

    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        Self::new(serde_json::from_slice(&std::fs::read(path)?)?)
    }

    pub fn file(&self) -> &NetFile {
        &self.f
    }

    /// Value of an arbitrary position from its raw features.
    pub fn eval_features(&self, toks: &[u16], extra: &[f32; EXTRA]) -> f32 {
        let (h1, h2) = (self.f.h1, self.f.h2);
        let mut h = self.f.b1.clone();
        for &t in toks {
            let row = &self.f.emb[t as usize * h1..(t as usize + 1) * h1];
            for (a, w) in h.iter_mut().zip(row) {
                *a += w;
            }
        }
        for (k, x) in extra.iter().enumerate() {
            let row = &self.f.we[k * h1..(k + 1) * h1];
            for (a, w) in h.iter_mut().zip(row) {
                *a += x * w;
            }
        }
        for a in h.iter_mut() {
            *a = a.max(0.0);
        }
        let mut y = self.f.b3;
        for j in 0..h2 {
            let mut z = self.f.b2[j];
            for (i, hv) in h.iter().enumerate() {
                z += hv * self.f.w2[i * h2 + j];
            }
            y += z.max(0.0) * self.f.w3[j];
        }
        y
    }

    pub fn eval(&self, b: &Board) -> f32 {
        let (t, n) = tokens(b);
        self.eval_features(&t[..n], &extras(b))
    }
}

/// The learned orderer.
pub struct Learned {
    pub net: Mlp,
}

const BEST: f32 = -1.0e9;
const WORST: f32 = 1.0e9;

impl Learned {
    /// Value of the position after `m`, from the attacker's point of view (lower is better).
    pub fn attacker_key(&self, b: &Board, m: Move) -> f32 {
        let mut after = b.clone();
        after.play_unchecked(m);
        match after.status() {
            GameStatus::Won => BEST,
            GameStatus::Drawn => WORST,
            GameStatus::Ongoing => {
                if crate::search::dead(&after) {
                    WORST
                } else {
                    self.net.eval(&after)
                }
            }
        }
    }
}

impl Orderer for Learned {
    fn attacker(&mut self, b: &Board, moves: &mut Vec<Move>) {
        let mut keyed: Vec<(f32, Move)> = moves.iter().map(|&m| (self.attacker_key(b, m), m)).collect();
        keyed.sort_by(|a, c| a.0.total_cmp(&c.0));
        *moves = keyed.into_iter().map(|(_, m)| m).collect();
    }

    fn defender(&mut self, b: &Board, moves: &mut Vec<Move>) {
        let mut keyed: Vec<(f32, Move)> = moves
            .iter()
            .map(|&m| {
                let mut next = b.clone();
                next.play_unchecked(m);
                // A capture that kills the position is the best defence (an immediate refutation).
                let v = if next.status() != GameStatus::Ongoing || crate::search::dead(&next) {
                    WORST
                } else {
                    self.net.eval(&next)
                };
                (v, m)
            })
            .collect();
        keyed.sort_by(|a, c| c.0.total_cmp(&a.0));
        *moves = keyed.into_iter().map(|(_, m)| m).collect();
    }
}
