//! Legal-candidate contract: every legal move maps to exactly one action, and
//! every action decodes back to a legal move.

use std::collections::BTreeSet;

use recur64_core::{ActionId, GameState, PromotionCode, StandardMove};

/// Reference positions (CPW perft suite plus edge cases).
const POSITIONS: &[&str] = &[
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", // startpos
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", // Kiwipete
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",                // position 3
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", // position 4
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", // position 5
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", // position 6
    "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3", // en passant
    "8/P7/8/8/8/8/8/k6K w - - 0 1",                             // promotion
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",                     // castling both sides
    "4k3/8/8/8/8/8/8/4K2R w K - 0 1",                           // white kingside only
];

fn check(fen: &str) {
    let g = GameState::from_fen(fen).unwrap();
    let p = g.perspective();
    let moves = g.legal_standard_moves();
    let actions = g.legal_actions();

    assert_eq!(moves.len(), actions.len(), "count mismatch: {fen}");

    let from_moves: BTreeSet<ActionId> = moves
        .iter()
        .map(|m| {
            ActionId::from_physical(m.from, m.to, m.promotion.unwrap_or(PromotionCode::NONE), p)
        })
        .collect();
    let from_actions: BTreeSet<ActionId> = actions.iter().copied().collect();
    assert_eq!(from_moves, from_actions, "action set mismatch: {fen}");
    assert_eq!(
        from_actions.len(),
        actions.len(),
        "duplicate ActionId in {fen}"
    );

    // Every action decodes to a legal move.
    for id in &actions {
        let (f, t, pr) = id.to_physical(p);
        let promo = if pr.is_none() { None } else { Some(pr) };
        let mv = StandardMove::new(f, t, promo);
        let cozy = mv.to_cozy(g.board()).unwrap();
        assert!(
            g.board().is_legal(cozy),
            "action {id:?} decoded illegal: {fen}"
        );
    }

    // The fixed-capacity list matches the Vec.
    let list = g.legal_action_list();
    assert_eq!(
        list.as_slice(),
        actions.as_slice(),
        "fixed list mismatch: {fen}"
    );
}

#[test]
fn legal_candidate_invariants() {
    for fen in POSITIONS {
        check(fen);
    }
}

#[test]
fn terminal_positions_have_no_actions() {
    for fen in [
        // Checkmate.
        "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
        // Stalemate.
        "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",
    ] {
        let g = GameState::from_fen(fen).unwrap();
        assert!(g.is_terminal(), "{fen}");
        assert!(g.legal_actions().is_empty(), "{fen}");
        assert!(g.legal_action_list().is_empty(), "{fen}");
    }
}

#[test]
fn promotions_are_distinct_actions() {
    let g = GameState::from_fen("8/P7/8/8/8/8/8/k6K w - - 0 1").unwrap();
    let actions = g.legal_actions();
    // a7a8 with N/B/R/Q and a7b8 (no black piece) etc.
    let queens = actions
        .iter()
        .filter(|id| id.promo() == PromotionCode::Q)
        .count();
    let knights = actions
        .iter()
        .filter(|id| id.promo() == PromotionCode::N)
        .count();
    assert_eq!(queens, 1, "one queen promotion");
    assert_eq!(knights, 1, "one knight promotion");
    assert!(actions.len() >= 4, "four promotion types present");
}
