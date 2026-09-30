//! Visible refusals, exact accounting, terminal handling, determinism.

use recur64_core::{ActionId, GameState, PromotionCode, Square, StandardMove};
use recur64_statequery::{QueryError, QueryManager, StatePacketV1, semantic_state_id};

fn action(gs: &GameState, uci: &str) -> u16 {
    let mv = StandardMove::from_uci(gs.board(), uci).unwrap();
    ActionId::from_physical(
        mv.from,
        mv.to,
        mv.promotion.unwrap_or(PromotionCode::NONE),
        gs.perspective(),
    )
    .index() as u16
}

fn play(m: &mut QueryManager, ucis: &[&str]) -> Vec<StatePacketV1> {
    let mut node = 0;
    let mut out = Vec::new();
    for u in ucis {
        let a = action(m.state(node).unwrap(), u);
        let p = m.query(node, a).unwrap();
        node = p.node_id;
        out.push(p);
    }
    out
}

#[test]
fn one_query_one_unit_and_exact_counters() {
    let mut m = QueryManager::new(GameState::startpos()).unwrap();
    assert_eq!(m.successful_queries(), 0);
    let root = m.packet(0).unwrap();
    assert_eq!(root.legal_actions.len(), 20);
    let a = root.legal_actions[0];
    let p = m.query(0, a).unwrap();
    assert_eq!(m.successful_queries(), 1);
    assert_eq!(m.node_count(), 2);
    assert_eq!(p.parent_id, Some(0));
    assert_eq!(p.incoming_action, Some(a));
    assert_eq!(p.ply_from_root, 1);
    assert_eq!(m.legal_moves_generated(), p.legal_actions.len() as u64);
}

#[test]
fn duplicate_edge_refuses_and_counts_nothing() {
    let mut m = QueryManager::new(GameState::startpos()).unwrap();
    let a = m.unqueried(0).unwrap()[3];
    m.query(0, a).unwrap();
    let e = m.query(0, a).unwrap_err();
    assert_eq!(e, QueryError::DuplicateEdge { node: 0, action: a });
    assert_eq!(m.successful_queries(), 1);
    assert_eq!(m.node_count(), 2);
    assert!(!m.unqueried(0).unwrap().contains(&a));
}

#[test]
fn illegal_action_and_unknown_node_refuse() {
    let mut m = QueryManager::new(GameState::startpos()).unwrap();
    let bogus = ActionId::encode(Square::A1, Square::A8, PromotionCode::NONE).index() as u16;
    assert_eq!(
        m.query(0, bogus).unwrap_err(),
        QueryError::IllegalAction {
            node: 0,
            action: bogus
        }
    );
    assert_eq!(m.query(99, 0).unwrap_err(), QueryError::UnknownNode(99));
    assert_eq!(m.successful_queries(), 0);
    assert_eq!(m.node_count(), 1);
}

#[test]
fn terminal_nodes_have_no_frontier_and_refuse_queries() {
    let mut m = QueryManager::new(GameState::startpos()).unwrap();
    let ps = play(&mut m, &["f2f3", "e7e5", "g2g4", "d8h4"]);
    let mate = ps.last().unwrap();
    assert!(mate.terminal);
    assert!(mate.legal_actions.is_empty());
    assert!(m.unqueried(mate.node_id).unwrap().is_empty());
    assert_eq!(
        m.query(mate.node_id, 0).unwrap_err(),
        QueryError::TerminalParent(mate.node_id)
    );
}

#[test]
fn stalemate_and_insufficient_material_roots_have_no_frontier() {
    for fen in [
        "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",
        "8/8/4k3/8/8/4K3/8/8 w - - 0 1",
    ] {
        let m = QueryManager::new(GameState::from_fen(fen).unwrap()).unwrap();
        let p = m.packet(0).unwrap();
        assert!(p.terminal, "{fen}");
        assert!(p.legal_actions.is_empty(), "{fen}");
    }
}

#[test]
fn threefold_repetition_terminal_child_hides_its_legal_moves() {
    let mut m = QueryManager::new(GameState::startpos()).unwrap();
    let seq = [
        "g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8",
    ];
    let ps = play(&mut m, &seq);
    let last = ps.last().unwrap();
    assert_eq!(last.repetition_count, 3);
    assert!(last.terminal);
    assert_eq!(last.terminal_reason, Some("threefold_repetition"));
    // The board still has legal moves, but the tool exposes no frontier.
    assert!(last.legal_actions.is_empty());
    assert!(ps[..ps.len() - 1].iter().all(|p| !p.terminal));
}

#[test]
fn fifty_move_rule_child_is_terminal() {
    let gs = GameState::from_fen("8/8/4k3/8/8/4K2R/8/8 w - - 99 80").unwrap();
    let mut m = QueryManager::new(gs).unwrap();
    let ps = play(&mut m, &["h3h4"]);
    assert!(ps[0].terminal);
    assert_eq!(ps[0].terminal_reason, Some("fifty_move_rule"));
    assert_eq!(ps[0].halfmove_clock, 100);
}

#[test]
fn check_flag_castling_and_en_passant_fields() {
    let mut m = QueryManager::new(GameState::startpos()).unwrap();
    let ps = play(&mut m, &["e2e4", "f7f6", "d1h5"]);
    assert!(ps[2].in_check);
    assert_eq!(ps[2].side_to_move, "black");
    // After 1.e4 the en-passant target is e3; black is to move so it is
    // rank-flipped to canonical e6 = 44.
    assert_eq!(ps[0].ep_square, Some(44));
    assert_eq!(ps[1].ep_square, None);
    assert_eq!(ps[0].castling, [true; 4]);
    // Castling rights are mover-relative: after Ke1e2 white has none.
    let mut m2 = QueryManager::new(GameState::startpos()).unwrap();
    let ps2 = play(&mut m2, &["e2e4", "e7e5", "e1e2"]);
    assert_eq!(ps2[2].castling, [true, true, false, false]);
}

#[test]
fn budget_exhaustion_refuses_visibly() {
    let mut m = QueryManager::new(GameState::startpos())
        .unwrap()
        .with_budget(2);
    let a = m.unqueried(0).unwrap();
    m.query(0, a[0]).unwrap();
    m.query(0, a[1]).unwrap();
    assert_eq!(
        m.query(0, a[2]).unwrap_err(),
        QueryError::BudgetExhausted { budget: 2 }
    );
    assert_eq!(m.successful_queries(), 2);
}

#[test]
fn packets_and_hashes_are_deterministic_and_state_keyed() {
    let run = || {
        let mut m = QueryManager::new(GameState::startpos()).unwrap();
        play(&mut m, &["e2e4", "e7e5", "g1f3"])
    };
    let (a, b) = (run(), run());
    for (x, y) in a.iter().zip(&b) {
        assert_eq!(x.semantic_id, y.semantic_id);
        assert_eq!(x.state_digest(), y.state_digest());
        assert_eq!(x.observation.as_slice(), y.observation.as_slice());
        assert_eq!(
            serde_json::to_string(x).unwrap(),
            serde_json::to_string(y).unwrap()
        );
    }
    assert_ne!(a[0].semantic_id, a[1].semantic_id);
    assert_eq!(a[0].semantic_id.len(), 64);
}

#[test]
fn promotion_edges_are_distinct_actions() {
    let gs = GameState::from_fen("8/P6k/8/8/8/8/p6K/8 w - - 0 1").unwrap();
    let m = QueryManager::new(gs).unwrap();
    let p = m.packet(0).unwrap();
    let promos = p
        .legal_actions
        .iter()
        .filter(|a| ActionId::from_index(u32::from(**a)).unwrap().promo() != PromotionCode::NONE)
        .count();
    assert_eq!(promos, 4);
}

fn id_of(fen: &str) -> String {
    semantic_state_id(&GameState::from_fen(fen).unwrap())
}

#[test]
fn identity_distinguishes_castling_ep_clock_and_side_to_move() {
    let base = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1";
    assert_eq!(id_of(base), id_of(base));
    assert_ne!(
        id_of(base),
        id_of("r3k2r/8/8/8/8/8/8/R3K2R w Kkq - 0 1"),
        "castling"
    );
    assert_ne!(
        id_of(base),
        id_of("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1"),
        "side to move"
    );
    assert_ne!(
        id_of(base),
        id_of("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 37 1"),
        "halfmove clock"
    );
    let ep = "rnbqkbnr/ppp1p1pp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 3";
    let no_ep = "rnbqkbnr/ppp1p1pp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 3";
    assert_ne!(id_of(ep), id_of(no_ep), "en passant");
}

#[test]
fn identity_ignores_the_fullmove_number() {
    let a = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1";
    let b = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 57";
    assert_eq!(id_of(a), id_of(b));
}

#[test]
fn identity_distinguishes_repetition_history_of_identical_placements() {
    // Same placement, side to move and castling rights as the start position,
    // but one earlier occurrence exists: their futures differ (threefold).
    let mut m = QueryManager::new(GameState::startpos()).unwrap();
    let ps = play(&mut m, &["g1f3", "g8f6", "f3g1", "f6g8"]);
    let back_home = ps.last().unwrap();
    assert_eq!(back_home.repetition_count, 2);
    let fresh = QueryManager::new(GameState::startpos())
        .unwrap()
        .packet(0)
        .unwrap();
    assert_ne!(back_home.semantic_id, fresh.semantic_id);
    // Two full cycles: a third distinct identity.
    let mut m2 = QueryManager::new(GameState::startpos()).unwrap();
    let ps2 = play(
        &mut m2,
        &["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"],
    );
    assert_ne!(ps2.last().unwrap().semantic_id, back_home.semantic_id);
}

#[test]
fn identity_distinguishes_reversible_histories_that_reach_the_same_placement() {
    // Same placement, clock 4: but the earlier positions that could still
    // recur differ (Nf3/Nf6 then Nc3/Nc6 versus the reverse order is one
    // history; Nf3/Nf6 then Ng1/Ng8 is another), so identities differ.
    let mut a = QueryManager::new(GameState::startpos()).unwrap();
    let pa = play(&mut a, &["g1f3", "g8f6", "b1c3", "b8c6"]);
    let mut b = QueryManager::new(GameState::startpos()).unwrap();
    let pb = play(&mut b, &["b1c3", "b8c6", "g1f3", "g8f6"]);
    // Identical placement and clock but different reversible history.
    assert_eq!(pa[3].halfmove_clock, pb[3].halfmove_clock);
    assert_ne!(pa[3].semantic_id, pb[3].semantic_id);
    // The same move order reproduces the identity exactly.
    let mut c = QueryManager::new(GameState::startpos()).unwrap();
    let pc = play(&mut c, &["g1f3", "g8f6", "b1c3", "b8c6"]);
    assert_eq!(pa[3].semantic_id, pc[3].semantic_id);
}

#[test]
fn identity_ignores_positions_before_the_last_irreversible_move() {
    // After a pawn move nothing earlier can recur, so the same post-pawn-move
    // continuation has the same identity however the game began.
    let mut a = QueryManager::new(GameState::startpos()).unwrap();
    let pa = play(&mut a, &["g1f3", "g8f6", "f3g1", "f6g8", "e2e4"]);
    let mut b = QueryManager::new(GameState::startpos()).unwrap();
    let pb = play(&mut b, &["b1c3", "b8c6", "c3b1", "c6b8", "e2e4"]);
    assert_eq!(pa[4].halfmove_clock, 0);
    assert_eq!(pa[4].semantic_id, pb[4].semantic_id);
}

#[test]
fn identity_includes_the_ply_cap_only_when_one_is_set() {
    let plain = GameState::startpos();
    let capped = GameState::startpos().with_max_plies(200);
    assert_ne!(semantic_state_id(&plain), semantic_state_id(&capped));
}
