//! `ComputeBankV1`: exact, deterministic chess facts for one position.
//!
//! Nothing here is an evaluation. There is no material scalar, no engine
//! output, no tablebase and no book value: the bank gives the model piece
//! counts, attack and defence relations, legal-move geometry, exact check
//! relations and bounded exact tactics, and lets the network learn whatever
//! strategic value those have.

use cozy_chess::{
    BitBoard, Board, Color, Move, Piece, Square, get_bishop_moves, get_king_moves,
    get_knight_moves, get_pawn_attacks, get_rook_moves,
};

use crate::board::{PIECE_ORDER, channel_piece, insufficient_material, piece_channel};
use crate::input::CoprocInput;
use crate::squares::{SquareIndex, index_of};
use crate::{CoprocError, GLOBAL_FIELDS, GLOBAL_OFFSET, OUTPUT_LEN, SQ_FIELDS, SQUARES};

/// Per-square field indices (`SQ_FIELDS` bytes per square).
pub mod sq {
    /// 0 empty; 1..=6 own P,N,B,R,Q,K; 7..=12 opponent P,N,B,R,Q,K.
    pub const PIECE: usize = 0;
    pub const ATTACKED_BY_OWN: usize = 1;
    pub const ATTACKED_BY_OPP: usize = 2;
    /// Own attacker counts by piece type, `3..=8`.
    pub const OWN_ATTACKERS: usize = 3;
    /// Opponent attacker counts by piece type, `9..=14`.
    pub const OPP_ATTACKERS: usize = 9;
    /// Number of own attackers when the square holds an own piece, else 0.
    pub const OWN_DEFENDERS: usize = 15;
    pub const OPP_DEFENDERS: usize = 16;
    pub const LEGAL_FROM: usize = 17;
    pub const LEGAL_TO: usize = 18;
    pub const LEGAL_CAPTURE_TO: usize = 19;
    pub const LEGAL_PROMOTION_TO: usize = 20;
    pub const OWN_ATTACKER_TOTAL: usize = 21;
    pub const OPP_ATTACKER_TOTAL: usize = 22;
    /// Bit 0 own diagonal slider, 1 own orthogonal slider, 2 opp diagonal
    /// slider, 3 opp orthogonal slider, 4 en-passant target, 5 own king in
    /// check, 6 opponent king attacked, 7 occupied.
    pub const RAY_FLAGS: usize = 23;
}

/// Global token indices (`GLOBAL_FIELDS` bytes each).
pub mod global {
    pub const PIECE_COUNTS: usize = 0;
    pub const STATE: usize = 1;
    pub const LEGAL_TOTALS: usize = 2;
    pub const TACTICAL: usize = 3;
    pub const MATE_SEARCH: usize = 4;
    pub const MOBILITY: usize = 5;
    pub const MOVE_CLASSES: usize = 6;
    pub const VERSION: usize = 7;
}

fn put_u16(out: &mut [u8], base: usize, v: u16) {
    let b = v.to_le_bytes();
    out[base] = b[0];
    out[base + 1] = b[1];
}

/// Attack set of one piece placed on `square`, in the presence of `occupied`.
fn attacks(piece: Piece, square: Square, occupied: BitBoard, color: Color) -> BitBoard {
    match piece {
        Piece::Pawn => get_pawn_attacks(square, color),
        Piece::Knight => get_knight_moves(square),
        Piece::Bishop => get_bishop_moves(square, occupied),
        Piece::Rook => get_rook_moves(square, occupied),
        Piece::Queen => get_bishop_moves(square, occupied) | get_rook_moves(square, occupied),
        Piece::King => get_king_moves(square),
    }
}

/// A piece is "hanging" when the opponent attacks its square and none of its
/// own side's pieces (of any type) attack it: every per-piece-type defender
/// channel is zero. Pins and x-rays are ignored, exactly as in the attack
/// tables, and kings count as attackers and defenders.
fn is_hanging(defenders: &[u8; 6], attackers: &[u8; 6]) -> bool {
    defenders.iter().all(|&v| v == 0) && attackers.iter().any(|&v| v > 0)
}

/// The exact facts a single pass over the legal moves yields.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct MoveAnalysis {
    pub legal: u32,
    pub captures: u32,
    pub promotions: u32,
    pub en_passant: u32,
    pub castling: u32,
    pub checking: u32,
    pub quiet: u32,
    pub mate_in_1_available: bool,
    pub mate_in_1_moves: u32,
    pub stalemate_moves: u32,
    pub mate_in_2_available: bool,
    pub mate_in_2_moves: u32,
    pub capture_checks: u32,
    pub quiet_checks: u32,
    pub promotion_checks: u32,
    pub capture_promotions: u32,
    pub en_passant_captures: u32,
    pub double_check_moves: u32,
}

/// Own moves that immediately deliver checkmate or stalemate.
/// Returns `(gives_check, mates, stalemates, checkers_after)`.
fn own_move_is_mate(board: &Board, mv: Move) -> (bool, bool, bool, u32) {
    let mut after = board.clone();
    after.play(mv);
    let gives_check = !after.checkers().is_empty();
    let replies: usize = {
        let mut n = 0usize;
        after.generate_moves(|mvs| {
            n += mvs.len();
            false
        });
        n
    };
    let mates = gives_check && replies == 0;
    let stalemates = !gives_check && replies == 0;
    let after_checkers = after.checkers().len();
    (gives_check, mates, stalemates, after_checkers)
}

/// Whether the side to move has a mate in one after the opponent played
/// `reply` in `after` (i.e. `after` is the opponent's move, side to move is us).
fn has_mate_in_one(board: &Board) -> bool {
    let mut found = false;
    board.generate_moves(|mvs| {
        for mv in mvs {
            let (_, mate, _, _) = own_move_is_mate(board, mv);
            if mate {
                found = true;
                return true;
            }
        }
        false
    });
    found
}

/// Classify every legal move of the side to move, and (optionally) run the
/// bounded exact mate-in-2 search.
pub fn analyze_moves(board: &Board, mate_search_depth: u8) -> MoveAnalysis {
    let mut a = MoveAnalysis::default();
    let mut moves: Vec<Move> = Vec::with_capacity(64);
    board.generate_moves(|mvs| {
        for mv in mvs {
            moves.push(mv);
        }
        false
    });
    a.legal = moves.len() as u32;

    for mv in &moves {
        let piece = board.piece_on(mv.from).expect("legal move has a piece");
        let is_promotion = mv.promotion.is_some();
        let is_en_passant = piece == Piece::Pawn
            && mv.from.file() != mv.to.file()
            && board.piece_on(mv.to).is_none();
        let is_castling = piece == Piece::King && board.colors(board.side_to_move()).has(mv.to);
        let is_capture = board
            .color_on(mv.to)
            .is_some_and(|c| c != board.side_to_move())
            || is_en_passant;

        let (gives_check, mates, stalemates, checkers_after) = own_move_is_mate(board, *mv);

        if is_capture {
            a.captures += 1;
        } else {
            a.quiet += 1;
        }
        if is_promotion {
            a.promotions += 1;
        }
        if is_en_passant {
            a.en_passant += 1;
            a.en_passant_captures += 1;
        }
        if is_castling {
            a.castling += 1;
        }
        if gives_check {
            a.checking += 1;
            if is_capture {
                a.capture_checks += 1;
            } else {
                a.quiet_checks += 1;
            }
            if is_promotion {
                a.promotion_checks += 1;
            }
            if checkers_after >= 2 {
                a.double_check_moves += 1;
            }
        }
        if is_capture && is_promotion {
            a.capture_promotions += 1;
        }
        if mates {
            a.mate_in_1_available = true;
            a.mate_in_1_moves += 1;
        }
        if stalemates {
            a.stalemate_moves += 1;
        }
    }

    if mate_search_depth >= 2 && !a.mate_in_1_available {
        'candidate: for mv in &moves {
            let mut after = board.clone();
            after.play(*mv);
            // The move must not already end the game.
            let mut replies: Vec<Move> = Vec::with_capacity(64);
            after.generate_moves(|mvs| {
                for r in mvs {
                    replies.push(r);
                }
                false
            });
            if replies.is_empty() {
                continue;
            }
            for reply in replies {
                let mut after_reply = after.clone();
                after_reply.play(reply);
                if !has_mate_in_one(&after_reply) {
                    continue 'candidate;
                }
            }
            a.mate_in_2_available = true;
            a.mate_in_2_moves += 1;
        }
    }

    a
}

/// Compute `ComputeBankV1` for one position into `out`.
pub fn compute_bank(input: &[u8], out: &mut [u8]) -> Result<(), CoprocError> {
    if out.len() != OUTPUT_LEN {
        return Err(CoprocError::BadOutputLength(out.len()));
    }
    out.fill(0);
    let inp = CoprocInput::new(input)?;
    let rec = crate::board::reconstruct(&inp)?;
    let board = &rec.board;
    let own = board.side_to_move();
    let opp = !own;
    let occupied = board.occupied();

    // Per-square attacker counts by piece type, for both sides.
    let mut own_att = [[0u8; 6]; SQUARES];
    let mut opp_att = [[0u8; 6]; SQUARES];
    for (side, table) in [(own, &mut own_att), (opp, &mut opp_att)] {
        for (t, piece) in PIECE_ORDER.iter().enumerate() {
            for from in board.colored_pieces(side, *piece) {
                for to in attacks(*piece, from, occupied, side) {
                    let slot = &mut table[index_of(to)][t];
                    *slot = slot.saturating_add(1);
                }
            }
        }
    }

    // Per-square legal geometry over the candidate list the network sees.
    let mut legal_from = [0u8; SQUARES];
    let mut legal_to = [0u8; SQUARES];
    let mut legal_capture_to = [0u8; SQUARES];
    let mut legal_promo_to = [0u8; SQUARES];
    for mv in inp.moves() {
        let from = mv.from as usize;
        let to = mv.to as usize;
        legal_from[from] = legal_from[from].saturating_add(1);
        legal_to[to] = legal_to[to].saturating_add(1);
        let target_occupied = board.piece_on(to.from_index()).is_some();
        let en_passant = board.piece_on(from.from_index()) == Some(Piece::Pawn)
            && from != to
            && !target_occupied;
        if target_occupied || en_passant {
            legal_capture_to[to] = legal_capture_to[to].saturating_add(1);
        }
        if mv.promo > 0 {
            legal_promo_to[to] = legal_promo_to[to].saturating_add(1);
        }
    }

    let own_king = board.king(own);
    let opp_king = board.king(opp);
    let in_check = !board.checkers().is_empty();

    for s in 0..SQUARES {
        let sq = s.from_index();
        let base = s * SQ_FIELDS;
        let piece = board.piece_on(sq);
        if let Some(p) = piece {
            let color = board.color_on(sq).expect("piece implies color");
            out[base + sq::PIECE] = (piece_channel(p, color) + 1) as u8;
        }
        let own_total: u8 = own_att[s].iter().copied().fold(0u8, u8::saturating_add);
        let opp_total: u8 = opp_att[s].iter().copied().fold(0u8, u8::saturating_add);
        out[base + sq::ATTACKED_BY_OWN] = u8::from(own_total > 0);
        out[base + sq::ATTACKED_BY_OPP] = u8::from(opp_total > 0);
        for t in 0..6 {
            out[base + sq::OWN_ATTACKERS + t] = own_att[s][t];
            out[base + sq::OPP_ATTACKERS + t] = opp_att[s][t];
        }
        out[base + sq::OWN_DEFENDERS] = if board.color_on(sq) == Some(own) {
            own_total
        } else {
            0
        };
        out[base + sq::OPP_DEFENDERS] = if board.color_on(sq) == Some(opp) {
            opp_total
        } else {
            0
        };
        out[base + sq::LEGAL_FROM] = legal_from[s];
        out[base + sq::LEGAL_TO] = legal_to[s];
        out[base + sq::LEGAL_CAPTURE_TO] = legal_capture_to[s];
        out[base + sq::LEGAL_PROMOTION_TO] = legal_promo_to[s];
        out[base + sq::OWN_ATTACKER_TOTAL] = own_total;
        out[base + sq::OPP_ATTACKER_TOTAL] = opp_total;

        // Diagonal / orthogonal slider relations.
        let own_diag = own_att[s][2] > 0 || own_att[s][4] > 0;
        let own_orth = own_att[s][3] > 0 || own_att[s][4] > 0;
        let opp_diag = opp_att[s][2] > 0 || opp_att[s][4] > 0;
        let opp_orth = opp_att[s][3] > 0 || opp_att[s][4] > 0;
        let mut flags = 0u8;
        flags |= u8::from(own_diag);
        flags |= u8::from(own_orth) << 1;
        flags |= u8::from(opp_diag) << 2;
        flags |= u8::from(opp_orth) << 3;
        flags |= u8::from(rec.en_passant == Some(sq)) << 4;
        flags |= u8::from(sq == own_king && opp_total > 0) << 5;
        flags |= u8::from(sq == opp_king && own_total > 0) << 6;
        flags |= u8::from(piece.is_some()) << 7;
        out[base + sq::RAY_FLAGS] = flags;
    }

    // --- Global tokens -------------------------------------------------------
    let g = GLOBAL_OFFSET;

    // T0 piece counts.
    {
        let b = g + global::PIECE_COUNTS * GLOBAL_FIELDS;
        let mut own_total = 0u8;
        let mut opp_total = 0u8;
        for (t, piece) in PIECE_ORDER.iter().enumerate() {
            let w = board.colored_pieces(own, *piece).len() as u8;
            let bl = board.colored_pieces(opp, *piece).len() as u8;
            out[b + t] = w;
            out[b + 6 + t] = bl;
            own_total = own_total.saturating_add(w);
            opp_total = opp_total.saturating_add(bl);
        }
        out[b + 12] = own_total;
        out[b + 13] = opp_total;
        out[b + 14] = u8::from(board.colored_pieces(own, Piece::King).len() == 1);
        out[b + 15] = u8::from(board.colored_pieces(opp, Piece::King).len() == 1);
    }

    // T1 position state.
    {
        let b = g + global::STATE * GLOBAL_FIELDS;
        let own_rights = *board.castle_rights(own);
        let opp_rights = *board.castle_rights(opp);
        out[b] = u8::from(own_rights.short.is_some());
        out[b + 1] = u8::from(own_rights.long.is_some());
        out[b + 2] = u8::from(opp_rights.short.is_some());
        out[b + 3] = u8::from(opp_rights.long.is_some());
        out[b + 4] = u8::from(rec.en_passant.is_some());
        out[b + 5] = rec.en_passant.map(|s| index_of(s) as u8).unwrap_or(u8::MAX);
        out[b + 6] = 0; // canonical side to move is always own (White)
        out[b + 7] = u8::from(insufficient_material(board));
        out[b + 8] = rec.halfmove_clock;
        out[b + 9] = rec.repetition_count;
        out[b + 10] = u8::from(in_check);
        out[b + 11] = index_of(own_king) as u8;
        out[b + 12] = index_of(opp_king) as u8;
        out[b + 13] = board.checkers().len() as u8;
        out[b + 14] = own_att[index_of(own_king)][..]
            .iter()
            .copied()
            .fold(0u8, u8::saturating_add);
        out[b + 15] = 0;
    }

    let analysis = analyze_moves(board, inp.mate_search_depth());

    // T2 legal move totals.
    {
        let b = g + global::LEGAL_TOTALS * GLOBAL_FIELDS;
        put_u16(out, b, analysis.legal.min(u16::MAX as u32) as u16);
        put_u16(out, b + 2, analysis.captures.min(u16::MAX as u32) as u16);
        put_u16(out, b + 4, analysis.promotions.min(u16::MAX as u32) as u16);
        put_u16(out, b + 6, analysis.en_passant.min(u16::MAX as u32) as u16);
        put_u16(out, b + 8, analysis.castling.min(u16::MAX as u32) as u16);
        put_u16(out, b + 10, analysis.checking.min(u16::MAX as u32) as u16);
        put_u16(out, b + 12, analysis.quiet.min(u16::MAX as u32) as u16);
    }

    // T3 tactical state.
    {
        let b = g + global::TACTICAL * GLOBAL_FIELDS;
        out[b] = u8::from(in_check);
        out[b + 1] = board.checkers().len() as u8;
        out[b + 2] = u8::from(board.checkers().len() >= 2);
        out[b + 3] = u8::from(analysis.legal == 0 && in_check);
        out[b + 4] = u8::from(analysis.legal == 0 && !in_check);
        out[b + 5] = u8::from(analysis.checking > 0);
        out[b + 6] = u8::from(analysis.mate_in_1_available);
        out[b + 7] = u8::from(analysis.stalemate_moves > 0);
        out[b + 8] = rec.repetition_count;
        out[b + 9] = u8::from(rec.halfmove_clock >= 100);
        // Recapturable material relation, as counts only (no scalar value):
        // the number of own pieces attacked by the opponent and vice versa.
        let mut own_hanging = 0u16;
        let mut opp_hanging = 0u16;
        for s in 0..SQUARES {
            // A piece's defenders are its own side's attackers of its square;
            // its attackers are the other side's.
            match board.color_on(s.from_index()) {
                Some(c) if c == own => {
                    if is_hanging(&own_att[s], &opp_att[s]) {
                        own_hanging += 1;
                    }
                }
                Some(_) if is_hanging(&opp_att[s], &own_att[s]) => {
                    opp_hanging += 1;
                }
                _ => {}
            }
        }
        put_u16(out, b + 10, own_hanging);
        put_u16(out, b + 12, opp_hanging);
        out[b + 14] = 0;
        out[b + 15] = 0;
    }

    // T4 bounded exact tactics.
    {
        let b = g + global::MATE_SEARCH * GLOBAL_FIELDS;
        out[b] = u8::from(analysis.mate_in_1_available);
        put_u16(out, b + 1, analysis.mate_in_1_moves as u16);
        put_u16(out, b + 3, analysis.stalemate_moves as u16);
        out[b + 5] = u8::from(analysis.mate_in_2_available);
        put_u16(out, b + 6, analysis.mate_in_2_moves as u16);
        out[b + 8] = inp.mate_search_depth();
        out[b + 9] = u8::from(in_check && analysis.legal > 0);
    }

    // T5 mobility summary.
    {
        let b = g + global::MOBILITY * GLOBAL_FIELDS;
        let mut pieces_with_moves = 0u16;
        let mut own_pieces = 0u16;
        for (s, &from_count) in legal_from.iter().enumerate() {
            if board.color_on(s.from_index()) == Some(own) {
                own_pieces += 1;
                if from_count > 0 {
                    pieces_with_moves += 1;
                }
            }
        }
        put_u16(out, b, analysis.legal.min(u16::MAX as u32) as u16);
        put_u16(out, b + 2, own_pieces);
        put_u16(out, b + 4, pieces_with_moves);
        put_u16(
            out,
            b + 6,
            if own_pieces > 0 {
                (analysis.legal as f32 / own_pieces as f32).round() as u16
            } else {
                0
            },
        );
        put_u16(out, b + 8, inp.n_legal().min(u16::MAX as usize) as u16);
    }

    // T6 move classes.
    {
        let b = g + global::MOVE_CLASSES * GLOBAL_FIELDS;
        put_u16(out, b, analysis.capture_checks.min(u16::MAX as u32) as u16);
        put_u16(
            out,
            b + 2,
            analysis.quiet_checks.min(u16::MAX as u32) as u16,
        );
        put_u16(
            out,
            b + 4,
            analysis.promotion_checks.min(u16::MAX as u32) as u16,
        );
        put_u16(
            out,
            b + 6,
            analysis.capture_promotions.min(u16::MAX as u32) as u16,
        );
        put_u16(
            out,
            b + 8,
            analysis.en_passant_captures.min(u16::MAX as u32) as u16,
        );
        put_u16(
            out,
            b + 10,
            analysis.double_check_moves.min(u16::MAX as u32) as u16,
        );
    }

    // T7 version stamp: the fixed literal, so a mismatched bank version is
    // visible in the bytes themselves.
    {
        let b = g + global::VERSION * GLOBAL_FIELDS;
        let stamp = crate::COMPUTE_BANK_VERSION.as_bytes();
        let n = stamp.len().min(GLOBAL_FIELDS);
        out[b..b + n].copy_from_slice(&stamp[..n]);
    }

    Ok(())
}

/// A read-only decoded view of one square's compute features (tests, probes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SquareFeatures {
    pub piece: u8,
    pub attacked_by_own: bool,
    pub attacked_by_opp: bool,
    pub own_attackers: [u8; 6],
    pub opp_attackers: [u8; 6],
    pub legal_from: u8,
    pub legal_to: u8,
    pub ray_flags: u8,
}

/// Decode one square from a `ComputeBankV1` buffer.
pub fn square_features(bank: &[u8], square: usize) -> SquareFeatures {
    let b = square * SQ_FIELDS;
    let mut own = [0u8; 6];
    let mut opp = [0u8; 6];
    own.copy_from_slice(&bank[b + sq::OWN_ATTACKERS..b + sq::OWN_ATTACKERS + 6]);
    opp.copy_from_slice(&bank[b + sq::OPP_ATTACKERS..b + sq::OPP_ATTACKERS + 6]);
    SquareFeatures {
        piece: bank[b + sq::PIECE],
        attacked_by_own: bank[b + sq::ATTACKED_BY_OWN] != 0,
        attacked_by_opp: bank[b + sq::ATTACKED_BY_OPP] != 0,
        own_attackers: own,
        opp_attackers: opp,
        legal_from: bank[b + sq::LEGAL_FROM],
        legal_to: bank[b + sq::LEGAL_TO],
        ray_flags: bank[b + sq::RAY_FLAGS],
    }
}

/// The global tokens of a `ComputeBankV1` buffer.
pub fn global_token(bank: &[u8], token: usize) -> &[u8] {
    let b = GLOBAL_OFFSET + token * GLOBAL_FIELDS;
    &bank[b..b + GLOBAL_FIELDS]
}

/// Canonical frame-0 channel of a square, or `None` when the square is empty.
pub fn piece_code(bank: &[u8], square: usize) -> Option<(Piece, Color)> {
    let v = bank[square * SQ_FIELDS + sq::PIECE];
    if v == 0 {
        None
    } else {
        channel_piece(v as usize - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{encode_observation, legal_moves, state_from_fen};
    use recur64_core::GameState;

    fn bank_for(fen: &str, depth: u8) -> Vec<u8> {
        let state = state_from_fen(fen);
        let obs = encode_observation(&state);
        let moves = legal_moves(&state);
        let input = crate::input::write_input(&obs, &moves, depth).unwrap();
        let mut out = crate::empty_output();
        compute_bank(&input, &mut out).unwrap();
        out
    }

    fn hanging_counts(fen: &str) -> (u16, u16) {
        let bank = bank_for(fen, 1);
        let t = global_token(&bank, global::TACTICAL);
        (
            u16::from_le_bytes([t[10], t[11]]),
            u16::from_le_bytes([t[12], t[13]]),
        )
    }

    #[test]
    fn is_hanging_requires_every_defender_channel_to_be_zero() {
        let none = [0u8; 6];
        let one_pawn = [1, 0, 0, 0, 0, 0];
        // attacked + zero defenders => hanging
        assert!(is_hanging(&none, &one_pawn));
        // attacked + one defender (in any channel) => not hanging
        assert!(!is_hanging(&one_pawn, &one_pawn));
        assert!(!is_hanging(&[0, 0, 0, 0, 0, 1], &one_pawn));
        // unattacked + zero defenders => not hanging
        assert!(!is_hanging(&none, &none));
    }

    #[test]
    fn hanging_counts_follow_defenders_and_attackers_on_real_boards() {
        // White knight d4 attacked by Bg7, no white defender.
        assert_eq!(hanging_counts("7k/6b1/8/8/3N4/8/8/K7 w - - 0 1"), (1, 0));
        // A pawn on c3 defends it: attacked but defended => not hanging.
        assert_eq!(hanging_counts("7k/6b1/8/8/3N4/2P5/8/K7 w - - 0 1"), (0, 0));
        // Undefended but unattacked => not hanging.
        assert_eq!(hanging_counts("7k/8/8/8/3N4/8/8/K7 w - - 0 1"), (0, 0));
        // Same position, black to move: the knight is now the opponent's.
        assert_eq!(hanging_counts("7k/6b1/8/8/3N4/8/8/K7 b - - 0 1"), (0, 1));
        // Colour/rank mirror with black to move is the same as the first case.
        assert_eq!(hanging_counts("k7/8/8/3n4/8/8/6B1/7K b - - 0 1"), (1, 0));
    }

    #[test]
    fn startpos_is_20_legal_and_no_tactics() {
        let bank = bank_for(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            1,
        );
        let totals = global_token(&bank, global::LEGAL_TOTALS);
        assert_eq!(u16::from_le_bytes([totals[0], totals[1]]), 20);
        assert_eq!(u16::from_le_bytes([totals[10], totals[11]]), 0);
        let mate = global_token(&bank, global::MATE_SEARCH);
        assert_eq!(mate[0], 0);
        let state = global_token(&bank, global::STATE);
        assert_eq!(state[0], 1); // own kingside
        assert_eq!(state[1], 1); // own queenside
        assert_eq!(state[2], 1); // opponent kingside
        assert_eq!(state[3], 1); // opponent queenside
        assert_eq!(state[4], 0); // no en passant
        let counts = global_token(&bank, global::PIECE_COUNTS);
        assert_eq!(counts[0], 8); // own pawns
        assert_eq!(counts[6], 8); // opponent pawns
        assert_eq!(counts[14], 1); // own king present
    }

    #[test]
    fn mate_in_one_is_detected() {
        // Back-rank mate: Ra8#.
        let bank = bank_for("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1", 1);
        let mate = global_token(&bank, global::MATE_SEARCH);
        assert_eq!(mate[0], 1, "mate-in-1 available");
        let totals = global_token(&bank, global::LEGAL_TOTALS);
        assert!(u16::from_le_bytes([totals[10], totals[11]]) >= 1);
    }

    /// Independent brute-force exact mate-in-2: some own move after which every
    /// opponent reply allows an immediate mate.
    fn brute_mate_in_2(board: &cozy_chess::Board) -> bool {
        use cozy_chess::Move;
        let mut firsts: Vec<Move> = Vec::new();
        board.generate_moves(|mvs| {
            for m in mvs {
                firsts.push(m);
            }
            false
        });
        for m1 in firsts {
            let mut after = board.clone();
            after.play(m1);
            let mut replies: Vec<Move> = Vec::new();
            after.generate_moves(|mvs| {
                for m in mvs {
                    replies.push(m);
                }
                false
            });
            // No replies is mate-in-1, which is a different claim.
            if replies.is_empty() {
                continue;
            }
            let all_forced = replies.iter().all(|r| {
                let mut after2 = after.clone();
                after2.play(*r);
                let mut mate = false;
                after2.generate_moves(|mvs| {
                    for m in mvs {
                        let mut after3 = after2.clone();
                        after3.play(m);
                        if !after3.checkers().is_empty() {
                            let mut n = 0usize;
                            after3.generate_moves(|x| {
                                n += x.len();
                                false
                            });
                            if n == 0 {
                                mate = true;
                                return true;
                            }
                        }
                    }
                    false
                });
                mate
            });
            if all_forced {
                return true;
            }
        }
        false
    }

    /// True when the side that is *not* to move is in check, i.e. the position
    /// is illegal for the side to move (a strict FEN parse is not guaranteed,
    /// and from such a position a legal-move generator would happily capture
    /// the king).
    fn opponent_in_check(board: &cozy_chess::Board) -> bool {
        use cozy_chess::{
            Piece, get_bishop_moves, get_king_moves, get_knight_moves, get_pawn_attacks,
            get_rook_moves,
        };
        let mover = board.side_to_move();
        let opp = !mover;
        let king = board.king(opp);
        let occ = board.occupied();
        let mut attacked = false;
        for piece in Piece::ALL {
            for from in board.colored_pieces(mover, piece) {
                let set = match piece {
                    Piece::Pawn => get_pawn_attacks(from, mover),
                    Piece::Knight => get_knight_moves(from),
                    Piece::Bishop => get_bishop_moves(from, occ),
                    Piece::Rook => get_rook_moves(from, occ),
                    Piece::Queen => get_bishop_moves(from, occ) | get_rook_moves(from, occ),
                    Piece::King => get_king_moves(from),
                };
                if set.has(king) {
                    attacked = true;
                }
            }
        }
        attacked
    }

    /// A FEN for a KQ-vs-K position, so `GameState::from_fen` performs the
    /// legality validation instead of a hand-built board.
    fn fen_of(wk: usize, bk: usize, wq: usize) -> String {
        let mut grid = ['.'; 64];
        grid[wk] = 'K';
        grid[bk] = 'k';
        grid[wq] = 'Q';
        let mut out = String::new();
        for rank in (0..8).rev() {
            let mut empty = 0;
            for file in 0..8 {
                let c = grid[rank * 8 + file];
                if c == '.' {
                    empty += 1;
                } else {
                    if empty > 0 {
                        out.push_str(&empty.to_string());
                        empty = 0;
                    }
                    out.push(c);
                }
            }
            if empty > 0 {
                out.push_str(&empty.to_string());
            }
            if rank > 0 {
                out.push('/');
            }
        }
        out.push_str(" w - - 0 1");
        out
    }

    /// Independent brute-force mate in one.
    fn brute_mate_in_one(board: &cozy_chess::Board) -> bool {
        use cozy_chess::Move;
        let mut found = false;
        board.generate_moves(|mvs| {
            for m in mvs {
                let mut after = board.clone();
                after.play(m);
                if !after.checkers().is_empty() {
                    let mut n = 0usize;
                    after.generate_moves(|x| {
                        n += x.len();
                        false
                    });
                    if n == 0 {
                        found = true;
                        return true;
                    }
                }
            }
            false
        });
        let _ = std::marker::PhantomData::<Move>;
        found
    }

    #[test]
    fn mate_in_two_matches_brute_force_and_is_off_by_default() {
        use cozy_chess::Board;

        // Deterministically find a legal KQ-vs-K position with a forced mate in
        // 2, with the black king on a corner so the search terminates fast.
        // `ComputeBankV1` reports mate-in-2 only when no mate in one exists, so
        // the scan must skip those positions.
        let mut found: Option<(String, Board)> = None;
        'search: for bk in [0usize, 7, 56, 63] {
            for wk in 0..64usize {
                for wq in 0..64usize {
                    let fen = fen_of(wk, bk, wq);
                    let Ok(state) = GameState::from_fen(&fen) else {
                        continue;
                    };
                    let board = state.board().clone();
                    if opponent_in_check(&board) || brute_mate_in_one(&board) {
                        continue;
                    }
                    if brute_mate_in_2(&board) {
                        found = Some((fen, board));
                        break 'search;
                    }
                }
            }
        }
        let (fen, board) = found.expect("the search must find a forced mate in 2 in KQ vs K");

        // Depth 1 never searches mate-in-2, whatever the position.
        let shallow = bank_for(&fen, 1);
        assert_eq!(global_token(&shallow, global::MATE_SEARCH)[5], 0);
        // Depth 2 agrees with the independent brute force.
        let deep = bank_for(&fen, 2);
        assert_eq!(
            global_token(&deep, global::MATE_SEARCH)[5],
            1,
            "{fen} {board}"
        );
        // The reported mate-in-2 move count is at least one.
        let moves = u16::from_le_bytes([
            global_token(&deep, global::MATE_SEARCH)[6],
            global_token(&deep, global::MATE_SEARCH)[7],
        ]);
        assert!(moves >= 1, "{fen}: mate_in_2_moves is {moves}");

        // And a position without a forced mate in 2 is reported as such.
        let bank = bank_for(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            2,
        );
        assert_eq!(global_token(&bank, global::MATE_SEARCH)[5], 0);
        assert!(!brute_mate_in_2(&Board::default()));
    }

    #[test]
    fn check_state_and_checkers_are_exact() {
        // Black king in check from a queen on e8.
        let bank = bank_for("4Q2k/8/8/8/8/8/8/K7 b - - 0 1", 1);
        let tactical = global_token(&bank, global::TACTICAL);
        assert_eq!(tactical[0], 1, "in check");
        assert_eq!(tactical[1], 1, "one checker");
        let state = global_token(&bank, global::STATE);
        assert_eq!(state[10], 1);
    }

    #[test]
    fn en_passant_and_promotion_pair_are_exact() {
        // En passant available: White pawn e5 can take d6 ep.
        let bank = bank_for(
            "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
            1,
        );
        let totals = global_token(&bank, global::LEGAL_TOTALS);
        assert!(
            u16::from_le_bytes([totals[6], totals[7]]) >= 1,
            "ep move counted"
        );
        // Promotion: four promotion moves from a7a8 plus a capture promotion.
        let bank = bank_for("1n5k/P7/8/8/8/8/8/K7 w - - 0 1", 1);
        let totals = global_token(&bank, global::LEGAL_TOTALS);
        assert_eq!(u16::from_le_bytes([totals[4], totals[5]]), 8);
        let classes = global_token(&bank, global::MOVE_CLASSES);
        assert_eq!(
            u16::from_le_bytes([classes[6], classes[7]]),
            4,
            "4 capture promotions"
        );
    }

    #[test]
    fn version_stamp_is_present() {
        let bank = bank_for(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            0,
        );
        let v = global_token(&bank, global::VERSION);
        assert_eq!(&v[..16], b"compute_bank_v1\0");
    }

    #[test]
    fn square_piece_identity_round_trips() {
        let bank = bank_for(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            0,
        );
        // Canonical square 4 is the own king.
        let (piece, color) = piece_code(&bank, 4).unwrap();
        assert_eq!(piece, Piece::King);
        assert_eq!(color, Color::White);
        assert!(piece_code(&bank, 16).is_none());
    }
}
