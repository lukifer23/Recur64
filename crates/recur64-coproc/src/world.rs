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
//! # Contract limits (`world_model_v2`)
//!
//! * The reference has no game history, so **repetition is not modelled**. The
//!   successor position is the exact canonical placement, castling rights,
//!   en-passant file and halfmove clock, not the history planes of Observation V1.
//! * Terminal codes: 0 ongoing, 1 checkmate, 2 stalemate, 3 insufficient material
//!   (Rules Profile V1's conservative rule), 4 fifty-move (halfmove clock >= 100).
//! * A position with more candidates than `w_cap`, or a candidate with more replies
//!   than `r_cap`, is a visible [`CoprocError::Capacity`] error.
//!
//! # Output layout (all `u8`)
//!
//! ```text
//! [0..8)      header: n_cand u16 LE, w_cap u8, r_cap u8, 4 reserved
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

/// Bytes of root facts per candidate.
pub const ROOT_FIELDS: usize = 8;
/// Bytes of successor record per candidate.
pub const SUCC_BYTES: usize = 72;
/// Bytes per reply record.
pub const REPLY_BYTES: usize = 20;
/// Header bytes.
pub const WORLD_HEADER: usize = 8;
/// Largest supported candidate capacity (chess maximum is 218).
pub const MAX_W_CAP: usize = 224;
/// Largest supported reply capacity.
pub const MAX_R_CAP: usize = 96;

/// Output length for a given capacity pair.
pub fn world_output_len(w_cap: usize, r_cap: usize) -> usize {
    WORLD_HEADER + w_cap * ROOT_FIELDS + w_cap * SUCC_BYTES + w_cap * r_cap * REPLY_BYTES
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

/// The eight `candidate_facts_v1` fields for playing `mv` from `board`, plus the
/// position after the move and its legal replies.
struct Played {
    facts: [u8; ROOT_FIELDS],
    after: Board,
    replies: Vec<Move>,
    terminal: u8,
}

fn play_facts(board: &Board, mv: Move) -> Played {
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
    let replies = moves_of(&after);
    let terminal = terminal_code(&after, !replies.is_empty());
    // A terminal position (mate, stalemate, dead position, fifty-move) has no
    // continuation, exactly as in the game rules: it exposes no replies.
    let replies = if terminal == 0 { replies } else { Vec::new() };
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
/// (`world_output_len(w_cap, r_cap)` bytes, fully overwritten).
pub fn world_model(
    input: &[u8],
    w_cap: usize,
    r_cap: usize,
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

    let so = succ_offset(w_cap);
    let ro = reply_offset(w_cap);
    for (ci, mv) in chosen.iter().enumerate() {
        let played = play_facts(board, *mv);
        out[root_offset() + ci * ROOT_FIELDS..root_offset() + (ci + 1) * ROOT_FIELDS]
            .copy_from_slice(&played.facts);

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

        // Reply set.
        if played.replies.len() > r_cap {
            return Err(CoprocError::Capacity(
                "a candidate has more replies than r_cap",
            ));
        }
        for (ri, reply) in played.replies.iter().enumerate() {
            let rp = play_facts(&played.after, *reply);
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
                let f = play_facts(&rp.after, *nm).facts;
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
        }
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
        world_model(&input, w_cap, r_cap, &mut out)?;
        Ok(out)
    }

    fn root(out: &[u8], i: usize) -> &[u8] {
        &out[root_offset() + i * ROOT_FIELDS..root_offset() + (i + 1) * ROOT_FIELDS]
    }

    #[test]
    fn output_length_is_a_pure_function_of_the_capacities() {
        assert_eq!(
            world_output_len(64, 16),
            8 + 64 * 8 + 64 * 72 + 64 * 16 * 20
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
