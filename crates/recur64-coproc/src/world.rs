//! `WorldModelV2`: an exact chess world model for the Chimera V2 planner.
//!
//! It provides CONSEQUENCES AND LEGAL STRUCTURE, not an evaluation: for every
//! legal root move it reports exact one-ply facts, the exact successor placement,
//! and the exact set of legal opponent replies with compact fact summaries of what
//! the next player could then do. There is no search, no visit count, no rollout,
//! no scalar score and no "forced mate" bit; a forced mate in two is something a
//! learner has to *infer* from "for every reply there is a mating move".
//!
//! Like `ComputeBankV1` this is a single source compiled natively and to
//! WebAssembly, so the two providers are byte-identical by construction.
//!
//! # Horizon (what is COMPUTED, independently of what the network may SEE)
//!
//! [`WorldHorizon`] is part of the execution contract. The packed layout is fixed
//! size, but sections beyond the requested horizon are exactly zero AND the work
//! to fill them is not performed:
//!
//! | horizon | fills | deterministic work actually executed |
//! |---|---|---|
//! | `Root` | root facts | per candidate: apply the move and enumerate the opponent's legal replies. This is unavoidable, because `captured`, `mate`, `stalemate` and `attacked-after` need it. |
//! | `Successor` | + successor records | the above plus record packing. Successor terminal code, in-check and reply count were already known from Root; this horizon adds mostly INFORMATION EXPOSURE, not new search. |
//! | `Replies` | + reply records and next-player summaries | the above plus, per reply: apply it, enumerate the next player's moves, and for each of those: apply it, enumerate the continuation and compute its facts. This is the expensive part. |
//!
//! The header carries the work counters, so the executed work is deterministic,
//! parity-checked between native and WASM, and recorded per horizon.
//!
//! # History contract (`fresh_no_history_v1`)
//!
//! The reference has no game history, so repetition is NOT modelled. The world
//! model therefore REFUSES any position whose observation reports a repetition
//! count above 1, and the primary V2 data must satisfy the fresh/no-history
//! convention. A search depth of three plies (root move, reply, next move) cannot
//! reach a threefold repetition from a fresh start (the start position can recur at
//! most once), so nothing here is an approximation under that convention. Using this
//! model on history-bearing positions (self-play) needs a new, history-aware contract.
//!
//! # Other contract limits (`world_model_v2`)
//!
//! * The successor position is the exact canonical placement, castling rights,
//!   en-passant file and halfmove clock, not the history planes of Observation V1.
//! * Terminal codes: 0 ongoing, 1 checkmate, 2 stalemate, 3 insufficient material
//!   (Rules Profile V1's conservative rule), 4 fifty-move (halfmove clock >= 100).
//!   A terminal position exposes NO replies, no reply records and no continuation.
//! * A position with more candidates than `w_cap`, or a candidate with more replies
//!   than `r_cap`, is a visible [`CoprocError::Capacity`] error.
//!
//! # Output layout (all `u8`)
//!
//! ```text
//! [0..48)     header: n_cand u16 LE, w_cap u8, r_cap u8, horizon u8, 3 reserved,
//!             then ten u32 LE work counters (see WorldStats)
//! root        w_cap x ROOT_FIELDS(8)  : mate, check, capture, captured value (0..9),
//!                                       attacked-after, promotion, promotion gain (0..8),
//!                                       stalemate
//! successor   w_cap x SUCC_BYTES(72)  : terminal, in_check, n_replies, 0,
//!                                       64 piece codes (child's canonical frame),
//!                                       castle nibble, ep file+1, halfmove clock, 0
//! reply       w_cap x r_cap x REPLY_BYTES(20)
//!                                       : valid, 8 reply facts, terminal, check, n_next,
//!                                         7-byte next-player summary, 1 reserved
//! ```

use cozy_chess::{Board, Color, File, Move, Piece, Square};

use crate::board::{insufficient_material, reconstruct};
use crate::input::CoprocInput;
use crate::squares::{SquareIndex, index_of};
use crate::{CoprocError, INPUT_LEN};

/// Semantic version of the world-model layout and rules.
pub const WORLD_MODEL_VERSION: &str = "world_model_v2";

/// The history contract the model is exact under.
pub const HISTORY_CONTRACT: &str = "fresh_no_history_v1";

/// Bytes of root facts per candidate.
pub const ROOT_FIELDS: usize = 8;
/// Bytes of successor record per candidate.
pub const SUCC_BYTES: usize = 72;
/// Bytes per reply record.
pub const REPLY_BYTES: usize = 20;
/// Header bytes: 8 fixed + ten u32 counters.
pub const WORLD_HEADER: usize = 48;
/// Number of u32 work counters in the header.
pub const STAT_COUNTERS: usize = 10;
/// Largest supported candidate capacity (chess maximum is 218).
pub const MAX_W_CAP: usize = 224;
/// Largest supported reply capacity.
pub const MAX_R_CAP: usize = 96;

/// How much of the world is computed (and therefore how much may be revealed).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum WorldHorizon {
    /// Root candidate facts only.
    Root = 1,
    /// + successor records.
    Successor = 2,
    /// + complete opponent reply records and next-player continuation summaries.
    Replies = 3,
}

impl WorldHorizon {
    pub const ALL: [WorldHorizon; 3] = [
        WorldHorizon::Root,
        WorldHorizon::Successor,
        WorldHorizon::Replies,
    ];

    pub fn code(self) -> u8 {
        self as u8
    }

    pub fn from_code(code: u8) -> Option<Self> {
        match code {
            1 => Some(WorldHorizon::Root),
            2 => Some(WorldHorizon::Successor),
            3 => Some(WorldHorizon::Replies),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            WorldHorizon::Root => "root",
            WorldHorizon::Successor => "successor",
            WorldHorizon::Replies => "replies",
        }
    }
}

/// Deterministic work executed for one position (stored in the header).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WorldStats {
    /// Root candidates processed.
    pub root_candidates: u32,
    /// Root moves applied.
    pub root_moves_applied: u32,
    /// Opponent reply moves enumerated (move generation after each root move).
    pub reply_moves_enumerated: u32,
    /// Opponent replies applied (Replies horizon only).
    pub reply_moves_applied: u32,
    /// Next-player moves enumerated after each reply (Replies horizon only).
    pub next_moves_enumerated: u32,
    /// Next-player moves applied to compute continuation facts (Replies horizon only).
    pub next_moves_applied: u32,
    /// Moves enumerated after those next-player moves (mate / attacked detection).
    pub continuation_moves_enumerated: u32,
    /// Successor records written.
    pub successor_records: u32,
    /// Reply records written.
    pub reply_records: u32,
    pub reserved: u32,
}

impl WorldStats {
    fn as_array(&self) -> [u32; STAT_COUNTERS] {
        [
            self.root_candidates,
            self.root_moves_applied,
            self.reply_moves_enumerated,
            self.reply_moves_applied,
            self.next_moves_enumerated,
            self.next_moves_applied,
            self.continuation_moves_enumerated,
            self.successor_records,
            self.reply_records,
            self.reserved,
        ]
    }

    /// Decode the counters from a packed world buffer.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let mut v = [0u32; STAT_COUNTERS];
        for (i, slot) in v.iter_mut().enumerate() {
            let o = 8 + i * 4;
            *slot = u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
        }
        Self {
            root_candidates: v[0],
            root_moves_applied: v[1],
            reply_moves_enumerated: v[2],
            reply_moves_applied: v[3],
            next_moves_enumerated: v[4],
            next_moves_applied: v[5],
            continuation_moves_enumerated: v[6],
            successor_records: v[7],
            reply_records: v[8],
            reserved: v[9],
        }
    }

    /// Total moves applied plus enumerated: a simple, comparable measure of work.
    pub fn total_move_operations(&self) -> u64 {
        u64::from(self.root_moves_applied)
            + u64::from(self.reply_moves_enumerated)
            + u64::from(self.reply_moves_applied)
            + u64::from(self.next_moves_enumerated)
            + u64::from(self.next_moves_applied)
            + u64::from(self.continuation_moves_enumerated)
    }

    /// Element-wise sum.
    pub fn add(&mut self, other: &WorldStats) {
        self.root_candidates += other.root_candidates;
        self.root_moves_applied += other.root_moves_applied;
        self.reply_moves_enumerated += other.reply_moves_enumerated;
        self.reply_moves_applied += other.reply_moves_applied;
        self.next_moves_enumerated += other.next_moves_enumerated;
        self.next_moves_applied += other.next_moves_applied;
        self.continuation_moves_enumerated += other.continuation_moves_enumerated;
        self.successor_records += other.successor_records;
        self.reply_records += other.reply_records;
    }
}

/// Output length for a given capacity pair.
pub fn world_output_len(w_cap: usize, r_cap: usize) -> usize {
    WORLD_HEADER + w_cap * ROOT_FIELDS + w_cap * SUCC_BYTES + w_cap * r_cap * REPLY_BYTES
}

/// Bytes of a buffer that carry information at `horizon` (the rest is exactly zero).
pub fn horizon_payload_bytes(w_cap: usize, r_cap: usize, horizon: WorldHorizon) -> usize {
    let root = WORLD_HEADER + w_cap * ROOT_FIELDS;
    match horizon {
        WorldHorizon::Root => root,
        WorldHorizon::Successor => root + w_cap * SUCC_BYTES,
        WorldHorizon::Replies => world_output_len(w_cap, r_cap),
    }
}

/// Byte offset of the root-facts section.
pub fn root_offset() -> usize {
    WORLD_HEADER
}

/// Byte offset of the successor section.
pub fn succ_offset(w_cap: usize) -> usize {
    WORLD_HEADER + w_cap * ROOT_FIELDS
}

/// Byte offset of the reply section.
pub fn reply_offset(w_cap: usize) -> usize {
    succ_offset(w_cap) + w_cap * SUCC_BYTES
}

fn value(p: Piece) -> u8 {
    match p {
        Piece::Pawn => 1,
        Piece::Knight | Piece::Bishop => 3,
        Piece::Rook => 5,
        Piece::Queen => 9,
        Piece::King => 0,
    }
}

fn piece_index(p: Piece) -> u8 {
    match p {
        Piece::Pawn => 0,
        Piece::Knight => 1,
        Piece::Bishop => 2,
        Piece::Rook => 3,
        Piece::Queen => 4,
        Piece::King => 5,
    }
}

fn promo_from_code(code: u8) -> Option<Piece> {
    match code {
        1 => Some(Piece::Knight),
        2 => Some(Piece::Bishop),
        3 => Some(Piece::Rook),
        4 => Some(Piece::Queen),
        _ => None,
    }
}

fn moves_of(board: &Board) -> Vec<Move> {
    let mut v = Vec::with_capacity(48);
    board.generate_moves(|mvs| {
        v.extend(mvs);
        false
    });
    v
}

/// The destination square in the UCI-style form the canonical action space uses:
/// `cozy-chess` encodes castling as "king takes own rook", the action space as
/// the king's two-square step.
fn uci_to(board: &Board, mv: Move) -> Square {
    let mover = board.side_to_move();
    if board.piece_on(mv.from) == Some(Piece::King) && board.color_on(mv.to) == Some(mover) {
        let file = if (mv.to.file() as usize) > (mv.from.file() as usize) {
            File::G
        } else {
            File::C
        };
        Square::new(file, mv.from.rank())
    } else {
        mv.to
    }
}

/// 0 ongoing, 1 checkmate, 2 stalemate, 3 insufficient material, 4 fifty-move.
fn terminal_code(board: &Board, has_moves: bool) -> u8 {
    if !has_moves {
        if board.checkers().is_empty() { 2 } else { 1 }
    } else if insufficient_material(board) {
        3
    } else if board.halfmove_clock() >= 100 {
        4
    } else {
        0
    }
}

/// Which counters a `play_facts` call feeds.
#[derive(Clone, Copy)]
enum Level {
    Root,
    Reply,
    Next,
}

/// The eight `candidate_facts_v1` fields for playing `mv` from `board`, plus the
/// position after the move and its legal replies.
struct Played {
    facts: [u8; ROOT_FIELDS],
    after: Board,
    replies: Vec<Move>,
    terminal: u8,
}

fn play_facts(board: &Board, mv: Move, level: Level, stats: &mut WorldStats) -> Played {
    let mover = board.side_to_move();
    let opp = !mover;
    let captured = match board.piece_on(mv.to) {
        Some(p) if board.color_on(mv.to) == Some(opp) => value(p),
        None if board.piece_on(mv.from) == Some(Piece::Pawn) && mv.from.file() != mv.to.file() => 1,
        _ => 0,
    };
    let gain = mv.promotion.map_or(0, |p| value(p) - 1);
    let mut after = board.clone();
    after.play(mv);
    let generated = moves_of(&after);
    match level {
        Level::Root => {
            stats.root_moves_applied += 1;
            stats.reply_moves_enumerated += generated.len() as u32;
        }
        Level::Reply => {
            stats.reply_moves_applied += 1;
            stats.next_moves_enumerated += generated.len() as u32;
        }
        Level::Next => {
            stats.next_moves_applied += 1;
            stats.continuation_moves_enumerated += generated.len() as u32;
        }
    }
    let terminal = terminal_code(&after, !generated.is_empty());
    // A terminal position (mate, stalemate, dead position, fifty-move) has no
    // continuation, exactly as in the game rules: it exposes no replies.
    let replies = if terminal == 0 { generated } else { Vec::new() };
    let dest = uci_to(board, mv);
    let attacked = terminal == 0 && replies.iter().any(|r| uci_to(&after, *r) == dest);
    let facts = [
        u8::from(terminal == 1),
        u8::from(!after.checkers().is_empty()),
        u8::from(captured > 0),
        captured,
        u8::from(attacked),
        u8::from(mv.promotion.is_some()),
        gain,
        u8::from(terminal == 2),
    ];
    Played {
        facts,
        after,
        replies,
        terminal,
    }
}

/// Pack a board into 64 piece codes in the canonical frame of its side to move:
/// 0 empty, 1-6 the mover's P N B R Q K, 7-12 the opponent's.
fn pack_board(board: &Board, out: &mut [u8]) {
    let mover = board.side_to_move();
    let flip = mover == Color::Black;
    for (c, slot) in out.iter_mut().enumerate().take(64) {
        let src = if flip { c ^ 56 } else { c };
        let sq = src.from_index();
        *slot = match (board.piece_on(sq), board.color_on(sq)) {
            (Some(p), Some(col)) => {
                if col == mover {
                    1 + piece_index(p)
                } else {
                    7 + piece_index(p)
                }
            }
            _ => 0,
        };
    }
}

fn cap255(n: usize) -> u8 {
    n.min(255) as u8
}

/// Compute the world model for one position into `out`
/// (`world_output_len(w_cap, r_cap)` bytes, fully overwritten) up to `horizon`.
pub fn world_model(
    input: &[u8],
    w_cap: usize,
    r_cap: usize,
    horizon: WorldHorizon,
    out: &mut [u8],
) -> Result<(), CoprocError> {
    if input.len() != INPUT_LEN {
        return Err(CoprocError::BadInputLength(input.len()));
    }
    if w_cap == 0 || w_cap > MAX_W_CAP || r_cap == 0 || r_cap > MAX_R_CAP {
        return Err(CoprocError::Capacity(
            "w_cap / r_cap outside the supported range",
        ));
    }
    if out.len() != world_output_len(w_cap, r_cap) {
        return Err(CoprocError::BadOutputLength(out.len()));
    }
    out.fill(0);
    let inp = CoprocInput::new(input)?;
    let rec = reconstruct(&inp)?;
    if rec.repetition_count > 1 {
        return Err(CoprocError::InvalidObservation(
            "world_model_v2 is exact only under fresh_no_history_v1: repetition count > 1",
        ));
    }
    let board = &rec.board;
    let n = inp.n_legal();
    if n > w_cap {
        return Err(CoprocError::Capacity("more legal candidates than w_cap"));
    }

    // Map the stored candidate list onto the board's own legal moves, and refuse
    // any mismatch: the world model is only meaningful for the canonical legal set.
    let legal = moves_of(board);
    if legal.len() != n {
        return Err(CoprocError::InvalidObservation(
            "candidate list length differs from the board's legal moves",
        ));
    }
    let mut used = vec![false; legal.len()];
    let mut chosen: Vec<Move> = Vec::with_capacity(n);
    for cand in inp.moves() {
        let want_promo = promo_from_code(cand.promo);
        let found = legal.iter().enumerate().position(|(i, m)| {
            !used[i]
                && index_of(m.from) == cand.from as usize
                && index_of(uci_to(board, *m)) == cand.to as usize
                && m.promotion == want_promo
        });
        let Some(i) = found else {
            return Err(CoprocError::InvalidObservation(
                "a stored candidate is not a legal move of the reconstructed board",
            ));
        };
        used[i] = true;
        chosen.push(legal[i]);
    }

    out[0..2].copy_from_slice(&(n as u16).to_le_bytes());
    out[2] = w_cap as u8;
    out[3] = r_cap as u8;
    out[4] = horizon.code();

    let mut stats = WorldStats {
        root_candidates: n as u32,
        ..WorldStats::default()
    };
    let so = succ_offset(w_cap);
    let ro = reply_offset(w_cap);
    for (ci, mv) in chosen.iter().enumerate() {
        let played = play_facts(board, *mv, Level::Root, &mut stats);
        out[root_offset() + ci * ROOT_FIELDS..root_offset() + (ci + 1) * ROOT_FIELDS]
            .copy_from_slice(&played.facts);
        if horizon < WorldHorizon::Successor {
            continue;
        }

        // Successor record.
        let s = so + ci * SUCC_BYTES;
        out[s] = played.terminal;
        out[s + 1] = u8::from(!played.after.checkers().is_empty());
        out[s + 2] = cap255(played.replies.len());
        pack_board(&played.after, &mut out[s + 4..s + 68]);
        let child = played.after.side_to_move();
        let rights = |color: Color| {
            let r = played.after.castle_rights(color);
            (u8::from(r.short.is_some()), u8::from(r.long.is_some()))
        };
        let (os, ol) = rights(child);
        let (ps, pl) = rights(!child);
        out[s + 68] = os | (ol << 1) | (ps << 2) | (pl << 3);
        out[s + 69] = played.after.en_passant().map_or(0, |f| f as u8 + 1);
        out[s + 70] = played.after.halfmove_clock();
        stats.successor_records += 1;
        if horizon < WorldHorizon::Replies {
            continue;
        }

        // Reply set (only at the Replies horizon; a terminal child has none).
        if played.replies.len() > r_cap {
            return Err(CoprocError::Capacity(
                "a candidate has more replies than r_cap",
            ));
        }
        for (ri, reply) in played.replies.iter().enumerate() {
            let rp = play_facts(&played.after, *reply, Level::Reply, &mut stats);
            let base = ro + (ci * r_cap + ri) * REPLY_BYTES;
            out[base] = 1;
            out[base + 1..base + 9].copy_from_slice(&rp.facts);
            out[base + 9] = rp.terminal;
            out[base + 10] = u8::from(!rp.after.checkers().is_empty());
            out[base + 11] = cap255(rp.replies.len());
            // What the next player (the root mover) could then do.
            let (mut mates, mut checks, mut caps, mut promos) = (0usize, 0usize, 0usize, 0usize);
            let (mut max_cap, mut max_gain) = (0u8, 0u8);
            for nm in &rp.replies {
                let f = play_facts(&rp.after, *nm, Level::Next, &mut stats).facts;
                mates += usize::from(f[0]);
                checks += usize::from(f[1]);
                caps += usize::from(f[2]);
                promos += usize::from(f[5]);
                max_cap = max_cap.max(f[3]);
                max_gain = max_gain.max(f[6]);
            }
            out[base + 12] = cap255(rp.replies.len());
            out[base + 13] = cap255(mates);
            out[base + 14] = cap255(checks);
            out[base + 15] = cap255(caps);
            out[base + 16] = cap255(promos);
            out[base + 17] = max_cap;
            out[base + 18] = max_gain;
            stats.reply_records += 1;
        }
    }
    for (i, v) in stats.as_array().iter().enumerate() {
        out[8 + i * 4..12 + i * 4].copy_from_slice(&v.to_le_bytes());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{encode_observation, legal_moves, state_from_fen};

    fn run(fen: &str, w_cap: usize, r_cap: usize) -> Result<Vec<u8>, CoprocError> {
        let state = state_from_fen(fen);
        let obs = encode_observation(&state);
        let moves = legal_moves(&state);
        let input = crate::input::write_input(&obs, &moves, 0).unwrap();
        let mut out = vec![0u8; world_output_len(w_cap, r_cap)];
        world_model(&input, w_cap, r_cap, WorldHorizon::Replies, &mut out)?;
        Ok(out)
    }

    fn root(out: &[u8], i: usize) -> &[u8] {
        &out[root_offset() + i * ROOT_FIELDS..root_offset() + (i + 1) * ROOT_FIELDS]
    }

    #[test]
    fn output_length_is_a_pure_function_of_the_capacities() {
        assert_eq!(
            world_output_len(64, 16),
            WORLD_HEADER + 64 * 8 + 64 * 72 + 64 * 16 * 20
        );
        let out = run("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1", 48, 8).unwrap();
        assert_eq!(out.len(), world_output_len(48, 8));
        assert_eq!(u16::from_le_bytes([out[0], out[1]]) as usize, {
            let s = state_from_fen("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1");
            legal_moves(&s).len()
        });
    }

    #[test]
    fn the_mating_move_is_flagged_and_its_successor_is_terminal() {
        let fen = "6k1/5ppp/8/8/8/8/8/R6K w - - 0 1";
        let state = state_from_fen(fen);
        let moves = legal_moves(&state);
        let out = run(fen, 48, 8).unwrap();
        let mating: Vec<usize> = (0..moves.len())
            .filter(|i| root(&out, *i)[0] == 1)
            .collect();
        assert_eq!(mating.len(), 1, "exactly one mating move (Ra8#)");
        let s = succ_offset(48) + mating[0] * SUCC_BYTES;
        assert_eq!(out[s], 1, "successor terminal code = checkmate");
        assert_eq!(out[s + 2], 0, "a mated position has no replies");
    }

    #[test]
    fn reply_sets_enumerate_every_reply_once_and_summarise_next_moves() {
        // Ra8+ is not mate here (king escapes): black has replies.
        let fen = "6k1/8/8/8/8/8/8/R6K w - - 0 1";
        let state = state_from_fen(fen);
        let moves = legal_moves(&state);
        let out = run(fen, 64, 16).unwrap();
        for i in 0..moves.len() {
            let s = succ_offset(64) + i * SUCC_BYTES;
            let n_replies = out[s + 2] as usize;
            let mut valid = 0;
            for r in 0..16 {
                let base = reply_offset(64) + (i * 16 + r) * REPLY_BYTES;
                valid += out[base] as usize;
                if out[base] == 0 {
                    assert!(
                        out[base..base + REPLY_BYTES].iter().all(|b| *b == 0),
                        "padding is zero"
                    );
                }
            }
            assert_eq!(
                valid, n_replies,
                "candidate {i}: replies enumerated exactly once"
            );
        }
    }

    /// Independent brute force with plain `cozy-chess`: does `board` (mover to move)
    /// have a mate in one?
    fn mate_in_one(board: &Board) -> bool {
        moves_of(board).into_iter().any(|m| {
            let mut n = board.clone();
            n.play(m);
            n.status() == cozy_chess::GameStatus::Won
        })
    }

    /// Number of first moves after which EVERY reply allows a mate in one.
    fn brute_forcing_first_moves(board: &Board) -> usize {
        if mate_in_one(board) {
            return 0;
        }
        moves_of(board)
            .into_iter()
            .filter(|m| {
                let mut a = board.clone();
                a.play(*m);
                let replies = moves_of(&a);
                a.status() == cozy_chess::GameStatus::Ongoing
                    && !replies.is_empty()
                    && replies.iter().all(|r| {
                        let mut b = a.clone();
                        b.play(*r);
                        b.status() == cozy_chess::GameStatus::Ongoing && mate_in_one(&b)
                    })
            })
            .count()
    }

    /// The same set, derived ONLY from the world model's facts: a first move whose
    /// successor is ongoing, has replies, and whose every reply record reports at
    /// least one mating continuation. There is no forced-mate bit in the output.
    fn forcing_first_moves_from_facts(out: &[u8], n: usize, w_cap: usize, r_cap: usize) -> usize {
        (0..n)
            .filter(|ci| {
                let s = succ_offset(w_cap) + ci * SUCC_BYTES;
                let (terminal, n_replies) = (out[s], out[s + 2] as usize);
                if terminal != 0 || n_replies == 0 {
                    return false;
                }
                (0..r_cap).all(|ri| {
                    let base = reply_offset(w_cap) + (ci * r_cap + ri) * REPLY_BYTES;
                    out[base] == 0 || (out[base + 9] == 0 && out[base + 13] > 0)
                })
            })
            .count()
    }

    #[test]
    fn a_forced_mate_in_two_is_recoverable_from_facts_alone() {
        // White Kc3 and Rook vs Black King: scan placements; the facts-derived forcing
        // set must equal the brute-force set for every position, and the scan must
        // contain real mates in two (so the test cannot pass vacuously).
        let mut positions = 0;
        let mut with_forcing = 0;
        for bk in 0..64usize {
            for rk in 0..64usize {
                let wk = 18usize; // c3
                let adjacent = (bk % 8).abs_diff(wk % 8) <= 1 && (bk / 8).abs_diff(wk / 8) <= 1;
                if bk == wk || rk == wk || rk == bk || adjacent {
                    continue;
                }
                let sq = |i: usize| format!("{}{}", (b'a' + (i % 8) as u8) as char, 1 + i / 8);
                let mut grid = [' '; 64];
                grid[wk] = 'K';
                grid[bk] = 'k';
                grid[rk] = 'R';
                let mut fen = String::new();
                for rank in (0..8).rev() {
                    let mut empty = 0;
                    for file in 0..8 {
                        let c = grid[rank * 8 + file];
                        if c == ' ' {
                            empty += 1;
                        } else {
                            if empty > 0 {
                                fen.push_str(&empty.to_string());
                                empty = 0;
                            }
                            fen.push(c);
                        }
                    }
                    if empty > 0 {
                        fen.push_str(&empty.to_string());
                    }
                    if rank > 0 {
                        fen.push('/');
                    }
                }
                fen.push_str(" w - - 0 1");
                let Ok(board) = Board::from_fen(&fen, false) else {
                    continue;
                };
                // Skip illegal placements (black in check with white to move, etc.).
                if board
                    .null_move()
                    .is_some_and(|nb| !nb.checkers().is_empty())
                {
                    continue;
                }
                let _ = sq(0);
                let state = std::panic::catch_unwind(|| state_from_fen(&fen));
                let Ok(state) = state else { continue };
                if state.legal_actions().is_empty() {
                    continue;
                }
                let n = legal_moves(&state).len();
                let out = run(&fen, 64, 16).unwrap();
                let from_facts = forcing_first_moves_from_facts(&out, n, 64, 16);
                let brute = brute_forcing_first_moves(&board);
                assert_eq!(
                    from_facts, brute,
                    "{fen}: facts-derived {from_facts} vs brute {brute}"
                );
                positions += 1;
                with_forcing += usize::from(brute > 0);
            }
        }
        assert!(positions > 1000, "scanned only {positions} positions");
        assert!(
            with_forcing > 50,
            "only {with_forcing} positions had a forced mate in two"
        );
    }

    #[test]
    fn castling_candidates_map_onto_uci_style_destinations() {
        let fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1";
        let state = state_from_fen(fen);
        let moves = legal_moves(&state);
        // Every stored candidate (including both castling moves) must be accepted.
        let out = run(fen, 64, 32).unwrap();
        assert_eq!(u16::from_le_bytes([out[0], out[1]]) as usize, moves.len());
    }

    #[test]
    fn capacity_overflow_is_a_visible_error_not_a_truncation() {
        let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        assert!(matches!(run(fen, 8, 32), Err(CoprocError::Capacity(_))));
        assert!(matches!(run(fen, 32, 1), Err(CoprocError::Capacity(_))));
    }

    #[test]
    fn deterministic() {
        let fen = "r3k2r/pp3ppp/8/3pP3/8/8/PP3PPP/R3K2R w KQkq d6 0 1";
        assert_eq!(run(fen, 96, 48).unwrap(), run(fen, 96, 48).unwrap());
    }

    #[test]
    fn successor_board_is_in_the_childs_canonical_frame() {
        // After any white move the child mover is black: its own king must be code 6.
        let fen = "4k3/8/8/8/8/8/8/4K3 w - - 0 1";
        let out = run(fen, 16, 8).unwrap();
        let s = succ_offset(16);
        let board = &out[s + 4..s + 68];
        assert_eq!(
            board.iter().filter(|c| **c == 6).count(),
            1,
            "own king (black, the child mover)"
        );
        assert_eq!(
            board.iter().filter(|c| **c == 12).count(),
            1,
            "opponent king (white)"
        );
    }
}

#[cfg(test)]
mod horizon_tests {
    use super::*;
    use crate::test_support::{encode_observation, legal_moves, state_from_fen};

    fn run_h(fen: &str, w_cap: usize, r_cap: usize, h: WorldHorizon) -> Vec<u8> {
        let state = state_from_fen(fen);
        let obs = encode_observation(&state);
        let moves = legal_moves(&state);
        let input = crate::input::write_input(&obs, &moves, 0).unwrap();
        let mut out = vec![0u8; world_output_len(w_cap, r_cap)];
        world_model(&input, w_cap, r_cap, h, &mut out).unwrap();
        out
    }

    const FEN: &str = "6k1/8/8/8/8/8/5PPP/R3K2R w KQ - 0 1";

    #[test]
    fn sections_beyond_the_horizon_are_exactly_zero() {
        let (w, r) = (64, 16);
        let root = run_h(FEN, w, r, WorldHorizon::Root);
        let succ = run_h(FEN, w, r, WorldHorizon::Successor);
        let full = run_h(FEN, w, r, WorldHorizon::Replies);
        for (h, buf) in [
            (WorldHorizon::Root, &root),
            (WorldHorizon::Successor, &succ),
            (WorldHorizon::Replies, &full),
        ] {
            assert_eq!(buf[4], h.code(), "header records the horizon");
            let payload = horizon_payload_bytes(w, r, h);
            assert!(
                buf[payload..].iter().all(|b| *b == 0),
                "{}: bytes beyond the horizon must be exactly zero",
                h.label()
            );
        }
        // The lower horizons are exact prefixes of the fuller ones (apart from the
        // header, whose horizon byte and work counters legitimately differ).
        let root_end = horizon_payload_bytes(w, r, WorldHorizon::Root);
        assert_eq!(root[WORLD_HEADER..root_end], full[WORLD_HEADER..root_end]);
        let succ_end = horizon_payload_bytes(w, r, WorldHorizon::Successor);
        assert_eq!(succ[WORLD_HEADER..succ_end], full[WORLD_HEADER..succ_end]);
        // And the fuller horizon really does contain more.
        assert!(full[succ_end..].iter().any(|b| *b != 0));
        assert!(succ[root_end..succ_end].iter().any(|b| *b != 0));
    }

    #[test]
    fn work_is_only_done_at_the_horizon_that_needs_it() {
        let (w, r) = (64, 16);
        let root = WorldStats::from_bytes(&run_h(FEN, w, r, WorldHorizon::Root));
        let succ = WorldStats::from_bytes(&run_h(FEN, w, r, WorldHorizon::Successor));
        let full = WorldStats::from_bytes(&run_h(FEN, w, r, WorldHorizon::Replies));
        let n = legal_moves(&state_from_fen(FEN)).len() as u32;
        // Root: one application and one move generation per candidate, nothing deeper.
        assert_eq!(root.root_candidates, n);
        assert_eq!(root.root_moves_applied, n);
        assert!(
            root.reply_moves_enumerated > 0,
            "attacked-after needs the replies"
        );
        assert_eq!(root.reply_moves_applied, 0);
        assert_eq!(root.next_moves_enumerated, 0);
        assert_eq!(root.next_moves_applied, 0);
        assert_eq!(root.continuation_moves_enumerated, 0);
        assert_eq!(root.successor_records + root.reply_records, 0);
        // Successor: identical search work, only records are added.
        assert_eq!(succ.successor_records, n);
        assert_eq!(succ.reply_records, 0);
        assert_eq!(succ.total_move_operations(), root.total_move_operations());
        // Replies: the expensive part appears only here.
        assert!(full.total_move_operations() > succ.total_move_operations() * 5);
        assert!(full.reply_moves_applied > 0 && full.next_moves_applied > 0);
        assert_eq!(full.reply_records, full.reply_moves_applied);
        assert_eq!(full.successor_records, n);
    }

    fn child_terminal(fen: &str, which: impl Fn(&[u8]) -> bool) -> (u8, u8) {
        // Find the candidate whose successor satisfies `which` and return
        // (terminal code, n_replies); also assert its reply section is all zero.
        let (w, r) = (64, 16);
        let out = run_h(fen, w, r, WorldHorizon::Replies);
        let n = u16::from_le_bytes([out[0], out[1]]) as usize;
        for ci in 0..n {
            let s = succ_offset(w) + ci * SUCC_BYTES;
            if which(&out[s..s + SUCC_BYTES]) {
                let ro = reply_offset(w) + ci * r * REPLY_BYTES;
                assert!(
                    out[ro..ro + r * REPLY_BYTES].iter().all(|b| *b == 0),
                    "a terminal child must expose no reply records"
                );
                return (out[s], out[s + 2]);
            }
        }
        panic!("no candidate satisfied the predicate in {fen}");
    }

    #[test]
    fn checkmate_stalemate_dead_position_and_fifty_move_expose_no_future() {
        // Checkmate: Ra8#.
        let (t, n) = child_terminal("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1", |s| s[0] == 1);
        assert_eq!((t, n), (1, 0));
        // Stalemate: Qf7 stalemates the king on h8.
        let (t, n) = child_terminal("7k/8/6Q1/8/8/8/8/K7 w - - 0 1", |s| s[0] == 2);
        assert_eq!((t, n), (2, 0));
        // Insufficient material: Kxe2 leaves bare kings. Replies would exist
        // mechanically (king moves), and must NOT be exposed.
        let (t, n) = child_terminal("4k3/8/8/8/8/8/4p3/4K3 w - - 0 1", |s| s[0] == 3);
        assert_eq!((t, n), (3, 0));
        // Fifty-move: a quiet rook move at clock 99 reaches 100.
        let (t, n) = child_terminal("k7/8/8/8/8/8/4R3/4K3 w - - 99 100", |s| s[0] == 4);
        assert_eq!((t, n), (4, 0));
    }

    #[test]
    fn a_position_with_repetition_history_is_refused_by_the_fresh_no_history_contract() {
        // Build an input whose observation reports repetition count 2 by patching the
        // broadcast feature; the model must refuse rather than approximate.
        let fen = "k7/8/8/8/8/8/4R3/4K3 w - - 0 1";
        let state = state_from_fen(fen);
        let mut obs = encode_observation(&state).as_slice().to_vec();
        obs[crate::board::REPETITION_OFFSET] = 0.4; // 2 / 5
        let moves = legal_moves(&state);
        let input = crate::input::write_input(&obs, &moves, 0).unwrap();
        let mut out = vec![0u8; world_output_len(32, 8)];
        assert!(matches!(
            world_model(&input, 32, 8, WorldHorizon::Root, &mut out),
            Err(CoprocError::InvalidObservation(_))
        ));
    }

    #[test]
    fn horizon_codes_round_trip() {
        for h in WorldHorizon::ALL {
            assert_eq!(WorldHorizon::from_code(h.code()), Some(h));
        }
        assert_eq!(WorldHorizon::from_code(0), None);
        assert_eq!(WorldHorizon::from_code(4), None);
    }
}
