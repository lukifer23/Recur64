//! Visible refusals, exact accounting, terminal handling, determinism.

use recur64_core::{ActionId, GameState, PromotionCode, Square, StandardMove};
use recur64_statequery::{QueryError, QueryManager, StatePacketV1};

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
        assert_eq!(x.state_hash, y.state_hash);
        assert_eq!(x.observation.as_slice(), y.observation.as_slice());
        assert_eq!(
            serde_json::to_string(x).unwrap(),
            serde_json::to_string(y).unwrap()
        );
    }
    assert_ne!(a[0].state_hash, a[1].state_hash);
    assert_eq!(a[0].state_hash.len(), 64);
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
