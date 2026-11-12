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
use crate::tb::{count_moves, moves_of};

/// Planes: white K, Q, R, B, N, then black K.
pub const PLANES: usize = 6;
/// Piece-square tokens.
pub const SQ_TOKENS: usize = PLANES * 64;
/// Relative-displacement tokens: for each of (wK-bK, piece-bK, piece-wK) a 15x15 grid of (dfile, drank).
pub const REL_GRID: usize = 15 * 15;
pub const REL_KINDS: usize = 3;
pub const TOKENS: usize = SQ_TOKENS + REL_KINDS * REL_GRID;
/// Colour-conditioned bishop tokens (token set `v3`): (bishop square colour) x (square of the black king / white king /
/// knight), 3 x 2 x 64. They let the first layer see the colour-complex interactions that decide KBNvK.
pub const BC_TOKENS: usize = 3 * 2 * 64;
pub const TOKENS_V3: usize = TOKENS + BC_TOKENS;
pub const TOKEN_SET_V2: &str = "v2";
pub const TOKEN_SET_V3: &str = "v3";
/// Piece-square tokens only (no king-relative displacement tokens): the ablation of the relational tokens.
pub const TOKEN_SET_SQ: &str = "sq";

/// Vocabulary size of a token set.
pub fn vocab(token_set: &str) -> usize {
    match token_set {
        TOKEN_SET_V3 => TOKENS_V3,
        TOKEN_SET_SQ => SQ_TOKENS,
        _ => TOKENS,
    }
}
pub const EXTRA: usize = 3;
/// At most 4 piece-square tokens, 5 relative tokens (two extra pieces) and 3 colour tokens fit in 12 slots.
pub const MAXP: usize = 12;
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
    let mut others = [(0i32, 0i32); MAXP];
    let mut n_others = 0usize;
    for color in [Color::White, Color::Black] {
        for piece in [Piece::King, Piece::Queen, Piece::Rook, Piece::Bishop, Piece::Knight] {
            if let Some(p) = plane(color, piece) {
                for sq in b.colored_pieces(color, piece) {
                    push((p * 64 + sq as usize) as u16, &mut n);
                    if color == Color::White && piece != Piece::King {
                        others[n_others] = (sq.file() as i32, sq.rank() as i32);
                        n_others += 1;
                    }
                }
            }
        }
    }
    let k = |s: cozy_chess::Square| (s.file() as i32, s.rank() as i32);
    let (wk, bk) = (k(b.king(Color::White)), k(b.king(Color::Black)));
    push(rel(wk, bk, 0), &mut n);
    for &o in &others[..n_others] {
        push(rel(o, bk, 1), &mut n);
        push(rel(o, wk, 2), &mut n);
    }
    (t, n)
}

/// Tokens of the `v3` set: the `v2` tokens plus colour-conditioned bishop tokens (a bishop contributes the colour of
/// its square combined with where the black king, the white king and the first knight stand).
pub fn tokens_v3(b: &Board) -> ([u16; MAXP], usize) {
    let (mut t, mut n) = tokens(b);
    let bk = b.king(Color::Black) as usize;
    let wk = b.king(Color::White) as usize;
    let knight = b.colored_pieces(Color::White, Piece::Knight).next_square().map(|s| s as usize);
    for sq in b.colored_pieces(Color::White, Piece::Bishop) {
        let colour = (sq.file() as usize + sq.rank() as usize) % 2;
        let mut push = |v: usize| {
            assert!(n < MAXP, "token overflow");
            t[n] = (TOKENS + v) as u16;
            n += 1;
        };
        push(colour * 64 + bk);
        push(128 + colour * 64 + wk);
        if let Some(k) = knight {
            push(256 + colour * 64 + k);
        }
    }
    (t, n)
}

/// Tokens of the named set.
pub fn tokens_for(token_set: &str, b: &Board) -> ([u16; MAXP], usize) {
    match token_set {
        TOKEN_SET_V3 => tokens_v3(b),
        TOKEN_SET_SQ => {
            let (full, _) = tokens(b);
            let n = b.occupied().len() as usize; // one piece-square token per piece, listed first
            // Unused slots must hold a valid id (0): leftover relative-token ids would index past this smaller vocabulary.
            let mut t = [0u16; MAXP];
            t[..n].copy_from_slice(&full[..n]);
            (t, n)
        }
        _ => tokens(b),
    }
}

/// Side to move is black, in check, and the number of legal moves / 32.
pub fn extras(b: &Board) -> [f32; EXTRA] {
    extras_with(b, count_moves(b))
}

/// [`extras`] when the legal-move count is already known (one move generation per candidate).
pub fn extras_with(b: &Board, n_moves: usize) -> [f32; EXTRA] {
    [
        f32::from(b.side_to_move() == Color::Black),
        f32::from(!b.checkers().is_empty()),
        n_moves as f32 / 32.0,
    ]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetFile {
    pub schema: String,
    /// Token set: `v2` (default for files written before v3) or `v3`.
    #[serde(default = "default_token_set")]
    pub token_set: String,
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
fn default_token_set() -> String {
    TOKEN_SET_V2.to_string()
}
/// Stack-buffer limits of the deployed forward pass.
pub const MAX_H1: usize = 1024;
pub const MAX_H2: usize = 256;

#[derive(Debug, Clone)]
pub struct Mlp {
    f: NetFile,
}

impl Mlp {
    pub fn new(f: NetFile) -> anyhow::Result<Self> {
        anyhow::ensure!(f.schema == NET_SCHEMA, "net schema {}", f.schema);
        anyhow::ensure!(f.h1 <= MAX_H1 && f.h2 <= MAX_H2, "net wider than the deployed forward supports ({}x{})", f.h1, f.h2);
        anyhow::ensure!(
            f.emb.len() == vocab(&f.token_set) * f.h1
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

    /// Value of an arbitrary position from its raw features. Allocation-free (stack buffers of at most
    /// [`MAX_H1`] / [`MAX_H2`] units) and row-major over `w2`, with ReLU-sparsity skipping; numerically equivalent to
    /// the straightforward dense forward up to float summation order.
    pub fn eval_features(&self, toks: &[u16], extra: &[f32; EXTRA]) -> f32 {
        let (h1, h2) = (self.f.h1, self.f.h2);
        let mut h = [0f32; MAX_H1];
        h[..h1].copy_from_slice(&self.f.b1);
        for &t in toks {
            let row = &self.f.emb[t as usize * h1..(t as usize + 1) * h1];
            for (a, w) in h[..h1].iter_mut().zip(row) {
                *a += w;
            }
        }
        for (k, x) in extra.iter().enumerate() {
            let row = &self.f.we[k * h1..(k + 1) * h1];
            for (a, w) in h[..h1].iter_mut().zip(row) {
                *a += x * w;
            }
        }
        let mut z = [0f32; MAX_H2];
        z[..h2].copy_from_slice(&self.f.b2);
        for (i, &hv) in h[..h1].iter().enumerate() {
            if hv > 0.0 {
                let row = &self.f.w2[i * h2..(i + 1) * h2];
                for (a, w) in z[..h2].iter_mut().zip(row) {
                    *a += hv * w;
                }
            }
        }
        let mut y = self.f.b3;
        for (zv, w) in z[..h2].iter().zip(&self.f.w3) {
            y += zv.max(0.0) * w;
        }
        y
    }

    pub fn eval(&self, b: &Board) -> f32 {
        let (t, n) = tokens_for(&self.f.token_set, b);
        self.eval_features(&t[..n], &extras(b))
    }

    /// Value of a child position given its legal-move count (avoids a second move generation).
    pub fn eval_with_moves(&self, b: &Board, n_moves: usize) -> f32 {
        let (t, n) = tokens_for(&self.f.token_set, b);
        self.eval_features(&t[..n], &extras_with(b, n_moves))
    }
}

/// The learned orderer. With `top_p` set, attacker moves are additionally truncated to the smallest prefix whose
/// softmax probability mass (under the network ranking scores, `softmax(-value)`) reaches `top_p`: width is spent
/// where the network is uncertain and saved where it is confident (a mate in one is always kept first).
pub struct Learned {
    pub net: Mlp,
    pub top_p: Option<f32>,
}

const BEST: f32 = -1.0e9;
const WORST: f32 = 1.0e9;

impl Learned {
    /// Ordering key of a child position (lower sorts first). One move generation per child: a position with no legal
    /// move is a checkmate (`BEST` for the attacker) or a stalemate (`WORST`); dead positions are `WORST`.
    fn key(&self, child: &Board) -> f32 {
        let nm = count_moves(child);
        if nm == 0 {
            return if child.checkers().is_empty() { WORST } else { BEST };
        }
        if crate::search::dead(child) {
            return WORST;
        }
        self.net.eval_with_moves(child, nm)
    }

    /// Resilience of the position after a defender reply (higher = the defender lasts longer = more likely a refutation).
    pub fn defender_key(&self, b: &Board, m: Move) -> f32 {
        let mut next = b.clone();
        next.play_unchecked(m);
        let nm = count_moves(&next);
        if nm == 0 || crate::search::dead(&next) {
            WORST
        } else {
            self.net.eval_with_moves(&next, nm)
        }
    }

    /// Value of the position after `m`, from the attacker's point of view (lower is better).
    pub fn attacker_key(&self, b: &Board, m: Move) -> f32 {
        let mut after = b.clone();
        after.play_unchecked(m);
        self.key(&after)
    }
}

impl Orderer for Learned {
    fn attacker(&mut self, b: &Board, moves: &mut Vec<Move>) {
        let mut keyed: Vec<(f32, Move)> = moves.iter().map(|&m| (self.attacker_key(b, m), m)).collect();
        keyed.sort_by(|a, c| a.0.total_cmp(&c.0));
        if let Some(p) = self.top_p {
            if keyed[0].0 > BEST {
                let best = keyed[0].0;
                let w: Vec<f32> = keyed.iter().map(|(k, _)| if *k >= WORST { 0.0 } else { (-(k - best)).exp() }).collect();
                let total: f32 = w.iter().sum();
                let (mut acc, mut keep) = (0.0f32, 0usize);
                for x in &w {
                    keep += 1;
                    acc += x / total;
                    if acc >= p {
                        break;
                    }
                }
                keyed.truncate(keep.max(1));
            } else {
                keyed.truncate(1); // a mate in one: nothing else is needed
            }
        }
        *moves = keyed.into_iter().map(|(_, m)| m).collect();
    }

    fn defender(&mut self, b: &Board, moves: &mut Vec<Move>) {
        let mut keyed: Vec<(f32, Move)> = moves
            .iter()
            .map(|&m| {
                let mut next = b.clone();
                next.play_unchecked(m);
                // A capture that kills the position is the best defence (an immediate refutation).
                let nm = count_moves(&next);
                let v = if nm == 0 || crate::search::dead(&next) {
                    WORST
                } else {
                    self.net.eval_with_moves(&next, nm)
                };
                (v, m)
            })
            .collect();
        keyed.sort_by(|a, c| c.0.total_cmp(&a.0));
        *moves = keyed.into_iter().map(|(_, m)| m).collect();
    }
}

/// Exact ordering accuracy of the DEPLOYED forward pass on TEST white-to-move positions, by optimal depth: is the
/// network's best-scored move an exactly optimal move (and is an optimal move among its top three)?
pub fn ordering_accuracy(tb: &crate::tb::Tablebase, net: &Mlp, max_depth: u8) -> Vec<serde_json::Value> {
    use crate::split::{Part, part_of};
    let mut by: Vec<(u32, u32, u32)> = vec![(0, 0, 0); max_depth as usize + 1];
    for b in tb
        .positions
        .iter()
        .filter(|b| b.side_to_move() == Color::White && b.status() == GameStatus::Ongoing)
    {
        if part_of(b) != Part::Test {
            continue;
        }
        let Some(p) = tb.plies_to_mate(b) else { continue };
        let d = p.div_ceil(2) as usize;
        let mut scored: Vec<(f32, bool)> = moves_of(b)
            .into_iter()
            .map(|m| {
                let mut a = b.clone();
                a.play_unchecked(m);
                let optimal = a.status() == GameStatus::Won || tb.plies_to_mate(&a) == Some(p - 1);
                let v = match a.status() {
                    GameStatus::Won => -1e9,
                    GameStatus::Drawn => 1e9,
                    GameStatus::Ongoing => net.eval(&a),
                };
                (v, optimal)
            })
            .collect();
        scored.sort_by(|x, y| x.0.total_cmp(&y.0));
        let e = &mut by[d];
        e.0 += 1;
        e.1 += u32::from(scored[0].1);
        e.2 += u32::from(scored.iter().take(3).any(|s| s.1));
    }
    by.iter()
        .enumerate()
        .skip(1)
        .filter(|(_, e)| e.0 > 0)
        .map(|(d, e)| {
            serde_json::json!({"depth": d, "n": e.0, "top1_optimal": f64::from(e.1) / f64::from(e.0),
                               "top3_contains_optimal": f64::from(e.2) / f64::from(e.0)})
        })
        .collect()
}
