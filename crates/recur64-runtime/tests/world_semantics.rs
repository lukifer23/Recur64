//! Semantic differential suite: `WorldModelV2` against Recur64 `GameState`.
//!
//! Native == WASM byte equality proves the two providers agree, not that they are
//! right (they run one algorithm). Here every field the world model reports is
//! recomputed independently from `GameState` (the game rules' own terminal
//! classification, legal-move lists and the `GameState`-based `CandidateFactsV1`),
//! and compared as semantic objects. Replies are compared as multisets, so a
//! missing, duplicated or illegal reply is caught.

use recur64_compute::{NativeWorldModel, WorldHorizon};
use recur64_coproc::world::{
    REPLY_BYTES, ROOT_FIELDS, SUCC_BYTES, WorldStats, reply_offset, root_offset, succ_offset,
    world_output_len,
};
use recur64_core::{ActionId, GameState, StandardMove, Termination};
use recur64_runtime::candidate_facts::facts_for;
use recur64_runtime::v2_inputs::compute_world_bytes;
use recur64_runtime::x15_inputs::probe_positions;

fn apply(state: &GameState, id: ActionId) -> GameState {
    let (from, to, promo) = id.to_physical(state.perspective());
    let promotion = if promo.is_none() { None } else { Some(promo) };
    let mut s = state.clone();
    s.apply(StandardMove::new(from, to, promotion)).unwrap();
    s
}

/// 0 ongoing, 1 mate, 2 stalemate, 3 insufficient, 4 fifty-move (repetition is not
/// modelled and must not occur under the fresh/no-history convention).
fn terminal_code(s: &GameState) -> u8 {
    match s.termination() {
        None => 0,
        Some(Termination::Checkmate) => 1,
        Some(Termination::Stalemate) => 2,
        Some(Termination::InsufficientMaterial) => 3,
        Some(Termination::FiftyMoveRule) => 4,
        Some(other) => panic!("unexpected termination {other:?} in a fresh position"),
    }
}

fn raw_facts(row: &[f32]) -> [u8; ROOT_FIELDS] {
    [
        row[0] as u8,
        row[1] as u8,
        row[2] as u8,
        (row[3] * 9.0).round() as u8,
        row[4] as u8,
        row[5] as u8,
        (row[6] * 8.0).round() as u8,
        row[7] as u8,
    ]
}

/// Expected packed successor placement, from the child's FEN.
fn expected_board(child: &GameState) -> Vec<u8> {
    let fen = child.to_fen();
    let mut it = fen.split(' ');
    let placement = it.next().unwrap();
    let black = it.next().unwrap() == "b";
    let mut grid = [0u8; 64];
    for (row, rank_str) in placement.split('/').enumerate() {
        let rank = 7 - row;
        let mut file = 0usize;
        for ch in rank_str.chars() {
            if let Some(d) = ch.to_digit(10) {
                file += d as usize;
            } else {
                let idx = match ch.to_ascii_lowercase() {
                    'p' => 0,
                    'n' => 1,
                    'b' => 2,
                    'r' => 3,
                    'q' => 4,
                    _ => 5,
                };
                let white_piece = ch.is_ascii_uppercase();
                grid[rank * 8 + file] = if white_piece != black {
                    1 + idx
                } else {
                    7 + idx
                };
                file += 1;
            }
        }
    }
    (0..64)
        .map(|c| grid[if black { c ^ 56 } else { c }])
        .collect()
}

/// Castling nibble (own short, own long, opp short, opp long) and en-passant file+1
/// and halfmove clock, from the child's FEN.
fn expected_extras(child: &GameState) -> (u8, u8, u8) {
    let fen = child.to_fen();
    let f: Vec<&str> = fen.split(' ').collect();
    let black = f[1] == "b";
    let castling = f[2];
    let has = |c: char| u8::from(castling.contains(c));
    let (os, ol, ps, pl) = if black {
        (has('k'), has('q'), has('K'), has('Q'))
    } else {
        (has('K'), has('Q'), has('k'), has('q'))
    };
    let ep = if f[3] == "-" {
        0
    } else {
        f[3].as_bytes()[0] - b'a' + 1
    };
    (
        os | (ol << 1) | (ps << 2) | (pl << 3),
        ep,
        f[4].parse::<u8>().unwrap(),
    )
}

/// One expected reply record (without the reserved byte), for multiset comparison.
type ReplyRec = [u8; 19];

fn expected_reply_records(child: &GameState) -> Vec<ReplyRec> {
    if child.is_terminal() {
        return Vec::new();
    }
    let legal = child.legal_actions();
    let facts = facts_for(child);
    legal
        .iter()
        .enumerate()
        .map(|(i, id)| {
            let g = apply(child, *id);
            let mut rec = [0u8; 19];
            rec[0] = 1;
            rec[1..9].copy_from_slice(&raw_facts(&facts[i * 8..(i + 1) * 8]));
            rec[9] = terminal_code(&g);
            rec[10] = u8::from(!g.board().checkers().is_empty());
            let next = if g.is_terminal() {
                Vec::new()
            } else {
                g.legal_actions()
            };
            rec[11] = next.len().min(255) as u8;
            let nf = if next.is_empty() {
                Vec::new()
            } else {
                facts_for(&g)
            };
            let rows: Vec<[u8; 8]> = nf.chunks(8).map(raw_facts).collect();
            rec[12] = rows.len().min(255) as u8;
            rec[13] = rows.iter().filter(|r| r[0] == 1).count().min(255) as u8;
            rec[14] = rows.iter().filter(|r| r[1] == 1).count().min(255) as u8;
            rec[15] = rows.iter().filter(|r| r[2] == 1).count().min(255) as u8;
            rec[16] = rows.iter().filter(|r| r[5] == 1).count().min(255) as u8;
            rec[17] = rows.iter().map(|r| r[3]).max().unwrap_or(0);
            rec[18] = rows.iter().map(|r| r[6]).max().unwrap_or(0);
            rec
        })
        .collect()
}

/// Check one position against the world model; returns the number of candidates.
fn check_position(state: &GameState, w_cap: usize, r_cap: usize) -> usize {
    let fen = state.to_fen();
    let bytes = compute_world_bytes(
        std::slice::from_ref(state),
        &NativeWorldModel,
        w_cap,
        r_cap,
        WorldHorizon::Replies,
    )
    .unwrap_or_else(|e| panic!("{fen}: {e}"));
    let out = &bytes[0];
    assert_eq!(out.len(), world_output_len(w_cap, r_cap));
    let legal = state.legal_actions();
    assert_eq!(
        u16::from_le_bytes([out[0], out[1]]) as usize,
        legal.len(),
        "{fen}: candidate count"
    );
    let root_expected = facts_for(state);
    let (mut total_replies, mut total_next, mut rule_terminal_replies) = (0usize, 0usize, 0usize);
    for (ci, id) in legal.iter().enumerate() {
        // Root facts, in candidate order.
        let r = root_offset() + ci * ROOT_FIELDS;
        assert_eq!(
            out[r..r + ROOT_FIELDS],
            raw_facts(&root_expected[ci * 8..(ci + 1) * 8]),
            "{fen}: root facts of candidate {ci}"
        );
        // Successor.
        let child = apply(state, *id);
        let s = succ_offset(w_cap) + ci * SUCC_BYTES;
        assert_eq!(
            out[s],
            terminal_code(&child),
            "{fen}: terminal code of candidate {ci}"
        );
        assert_eq!(
            out[s + 1],
            u8::from(!child.board().checkers().is_empty()),
            "{fen}: in-check of candidate {ci}"
        );
        assert_eq!(
            &out[s + 4..s + 68],
            expected_board(&child).as_slice(),
            "{fen}: placement of candidate {ci}"
        );
        let (nibble, ep, clock) = expected_extras(&child);
        assert_eq!(
            out[s + 68],
            nibble,
            "{fen}: castling nibble of candidate {ci}"
        );
        assert_eq!(out[s + 69], ep, "{fen}: en-passant file of candidate {ci}");
        assert_eq!(
            out[s + 70],
            clock,
            "{fen}: halfmove clock of candidate {ci}"
        );
        // Replies as a multiset.
        let want = {
            let mut v = expected_reply_records(&child);
            v.sort();
            v
        };
        assert_eq!(
            out[s + 2] as usize,
            want.len(),
            "{fen}: reply count of candidate {ci}"
        );
        let mut got: Vec<ReplyRec> = (0..r_cap)
            .filter_map(|ri| {
                let b = reply_offset(w_cap) + (ci * r_cap + ri) * REPLY_BYTES;
                (out[b] == 1).then(|| {
                    let mut rec = [0u8; 19];
                    rec.copy_from_slice(&out[b..b + 19]);
                    rec
                })
            })
            .collect();
        got.sort();
        assert_eq!(
            got, want,
            "{fen}: reply records of candidate {ci} differ (multiset)"
        );
        // Terminal children advertise no playable future at any depth.
        if child.is_terminal() {
            let base = reply_offset(w_cap) + ci * r_cap * REPLY_BYTES;
            assert!(
                out[base..base + r_cap * REPLY_BYTES]
                    .iter()
                    .all(|b| *b == 0)
            );
            assert_eq!(out[s + 2], 0);
        }
        total_replies += want.len();
        total_next += want.iter().map(|r| r[11] as usize).sum::<usize>();
        // Replies that end the game by material or clock: their moves are generated
        // (to decide terminality) but, being terminal, are not exposed.
        rule_terminal_replies += want.iter().filter(|r| r[9] == 3 || r[9] == 4).count();
    }
    // Counters describe the work that was actually done.
    let stats = WorldStats::from_bytes(out);
    assert_eq!(stats.root_candidates as usize, legal.len());
    assert_eq!(stats.root_moves_applied as usize, legal.len());
    assert_eq!(
        stats.reply_moves_applied as usize, total_replies,
        "{fen}: replies applied"
    );
    assert_eq!(stats.reply_records as usize, total_replies);
    // Enumerated work >= exposed moves; the surplus exists only if some reply is
    // terminal by material/clock (its moves were generated, then hidden).
    assert!(
        stats.next_moves_enumerated as usize >= total_next,
        "{fen}: under-counted work"
    );
    if rule_terminal_replies == 0 {
        assert_eq!(
            stats.next_moves_enumerated as usize, total_next,
            "{fen}: next moves enumerated"
        );
    }
    legal.len()
}

fn edge_fens() -> Vec<(&'static str, &'static str)> {
    vec![
        (
            "castling both sides (white)",
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        ),
        (
            "castling both sides (black to move)",
            "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
        ),
        (
            "castling through attack",
            "r3k2r/8/8/8/8/5r2/8/R3K2R w KQkq - 0 1",
        ),
        (
            "en passant (white)",
            "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
        ),
        (
            "en passant (black)",
            "rnbqkbnr/pppp1ppp/8/8/3Pp3/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 3",
        ),
        (
            "all promotions and capture promotions (white)",
            "1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1",
        ),
        (
            "all promotions and capture promotions (black)",
            "4k3/8/8/8/8/8/p7/1N2K3 b - - 0 1",
        ),
        ("check", "4k3/8/8/8/4r3/8/8/4K3 w - - 0 1"),
        ("double check", "4k3/8/8/8/1b6/3n4/8/4K3 w - - 0 1"),
        ("mate in one available", "6k1/5ppp/8/8/8/8/8/R6K w - - 0 1"),
        ("stalemate available", "7k/8/6Q1/8/8/8/8/K7 w - - 0 1"),
        (
            "insufficient material available",
            "4k3/8/8/8/8/8/4p3/4K3 w - - 0 1",
        ),
        (
            "fifty-move edge (clock 99)",
            "k7/8/8/8/8/8/4R3/4K3 w - - 99 100",
        ),
        (
            "black to move canonicalisation",
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        ),
        ("pinned piece", "4k3/4r3/8/8/8/8/4B3/4K3 w - - 0 1"),
        (
            "start position",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        ),
        (
            "open middlegame",
            "r2q1rk1/pp2bppp/2n1pn2/3p4/3P1B2/2NBPN2/PP3PPP/R2Q1RK1 w - - 0 1",
        ),
    ]
}

#[test]
fn edge_cases_agree_with_gamestate() {
    for (name, fen) in edge_fens() {
        let s = GameState::from_fen(fen).unwrap_or_else(|e| panic!("{name}: {e}"));
        let n = check_position(&s, 128, 64);
        assert!(n > 0, "{name}: no candidates");
    }
}

#[test]
fn the_maximum_legal_move_position_fits_and_agrees() {
    // The well-known 218-legal-move position.
    let fen = "R6R/3Q4/1Q4Q1/4Q3/2Q4Q/Q4Q2/pp1Q4/kBNN1KB1 w - - 0 1";
    let s = GameState::from_fen(fen).unwrap();
    assert_eq!(s.legal_actions().len(), 218);
    assert_eq!(check_position(&s, 224, 32), 218);
}

#[test]
fn several_hundred_random_fresh_positions_agree_with_gamestate() {
    let mut checked = 0usize;
    let mut candidates = 0usize;
    let mut seen = std::collections::HashSet::new();
    for s in probe_positions(700, 70) {
        // A fresh state: the fen only (no move history), as the contract requires.
        let fresh = GameState::from_fen(&s.to_fen()).unwrap();
        if fresh.is_terminal() || !seen.insert(fresh.to_fen()) {
            continue;
        }
        if fresh.legal_actions().len() > 128 {
            continue;
        }
        candidates += check_position(&fresh, 128, 64);
        checked += 1;
        if checked >= 320 {
            break;
        }
    }
    assert!(
        checked >= 250,
        "only {checked} distinct positions were checked"
    );
    assert!(
        candidates > 5000,
        "only {candidates} candidates were compared"
    );
}

#[test]
fn world_root_facts_equal_candidate_facts_v1_for_hundreds_of_positions() {
    // The direct cross-contract check: decode the world model's root section with the
    // production decoder and compare with the GameState-based CandidateFactsV1 function.
    let (w_cap, r_cap) = (128, 64);
    let mut n = 0usize;
    let mut seen = std::collections::HashSet::new();
    for s in probe_positions(500, 70) {
        let fresh = GameState::from_fen(&s.to_fen()).unwrap();
        if fresh.is_terminal()
            || fresh.legal_actions().len() > w_cap
            || !seen.insert(fresh.to_fen())
        {
            continue;
        }
        let bytes = compute_world_bytes(
            std::slice::from_ref(&fresh),
            &NativeWorldModel,
            w_cap,
            r_cap,
            WorldHorizon::Root,
        )
        .unwrap();
        let decoded = recur64_runtime::v2_inputs::root_facts_from_bytes(&bytes[0], w_cap);
        let want = facts_for(&fresh);
        for (k, w) in want.iter().enumerate() {
            assert!(
                (decoded[k] - w).abs() < 1e-6,
                "{}: fact {k}: world {} vs CandidateFactsV1 {w}",
                fresh.to_fen(),
                decoded[k]
            );
        }
        // Padded candidates decode to exactly zero.
        assert!(decoded[want.len()..].iter().all(|v| *v == 0.0));
        n += 1;
        if n >= 300 {
            break;
        }
    }
    assert!(n >= 200);
}
