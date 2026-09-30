//! Digest and identity contracts: what each digest covers, proved by mutation,
//! and the one-query operation story proved by counters.

use recur64_core::{ActionId, GameState, PromotionCode, StandardMove};
use recur64_statequery::{
    EPHEMERAL_FIELDS, PACKET_FIELDS, QUERY_IDENTITY_PACKET_FIELDS, QueryError, QueryManager,
    STATE_DIGEST_FIELDS, StatePacketV1,
};

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

/// A non-terminal packet whose optional fields are populated where possible.
fn sample() -> StatePacketV1 {
    let mut m = QueryManager::new(GameState::startpos()).unwrap();
    // After 1.e4 an en-passant square is set and every castling right remains.
    play(&mut m, &["e2e4"]).remove(0)
}

fn mutate(p: &StatePacketV1, field: &str) -> StatePacketV1 {
    let mut q = p.clone();
    match field {
        "node_id" => q.node_id += 1,
        "parent_id" => q.parent_id = Some(q.parent_id.map_or(7, |x| x + 1)),
        "incoming_action" => q.incoming_action = Some(q.incoming_action.map_or(1, |x| x + 1)),
        "ply_from_root" => q.ply_from_root += 1,
        "observation" => {
            let mut fresh = recur64_core::ObservationV1::zeroed();
            fresh.0.copy_from_slice(p.observation.as_slice());
            fresh.0[123] += 1.0;
            q.observation = fresh;
        }
        "legal_actions" => q.legal_actions.push(u16::MAX),
        "side_to_move" => {
            q.side_to_move = if q.side_to_move == "white" {
                "black"
            } else {
                "white"
            }
        }
        "in_check" => q.in_check = !q.in_check,
        "terminal" => q.terminal = !q.terminal,
        "terminal_reason" => {
            q.terminal_reason = match q.terminal_reason {
                None => Some("checkmate"),
                Some(_) => None,
            }
        }
        "castling" => q.castling[0] = !q.castling[0],
        "ep_square" => q.ep_square = Some(q.ep_square.map_or(9, |x| x ^ 1)),
        "halfmove_clock" => q.halfmove_clock += 1,
        "repetition_count" => q.repetition_count += 1,
        "semantic_id" => q.semantic_id.push('0'),
        other => panic!("unknown field {other}"),
    }
    q
}

#[test]
fn every_packet_field_is_classified_exactly_once() {
    for f in PACKET_FIELDS {
        let n = [
            STATE_DIGEST_FIELDS.contains(&f),
            QUERY_IDENTITY_PACKET_FIELDS.contains(&f),
            EPHEMERAL_FIELDS.contains(&f),
        ]
        .iter()
        .filter(|&&x| x)
        .count();
        assert_eq!(
            n, 1,
            "packet field {f:?} must be in exactly one digest class"
        );
    }
    let classified =
        STATE_DIGEST_FIELDS.len() + QUERY_IDENTITY_PACKET_FIELDS.len() + EPHEMERAL_FIELDS.len();
    assert_eq!(
        classified,
        PACKET_FIELDS.len(),
        "no class names a field that does not exist"
    );
}

#[test]
fn changing_any_state_content_field_changes_the_state_digest() {
    let p = sample();
    let base = p.state_digest();
    for f in STATE_DIGEST_FIELDS {
        assert_ne!(
            mutate(&p, f).state_digest(),
            base,
            "state digest ignores declared field {f:?}"
        );
    }
}

#[test]
fn ephemeral_and_edge_fields_do_not_change_the_state_digest() {
    let p = sample();
    let base = p.state_digest();
    for f in EPHEMERAL_FIELDS.iter().chain(&QUERY_IDENTITY_PACKET_FIELDS) {
        assert_eq!(
            mutate(&p, f).state_digest(),
            base,
            "state digest must not depend on {f:?}"
        );
    }
}

#[test]
fn changing_any_query_identity_component_changes_its_digest() {
    let p = sample();
    let parent = "a".repeat(64);
    let id = p.query_identity(&parent).unwrap();
    let base = id.digest();
    // parent semantic identity
    assert_ne!(p.query_identity(&"b".repeat(64)).unwrap().digest(), base);
    // incoming action
    assert_ne!(
        mutate(&p, "incoming_action")
            .query_identity(&parent)
            .unwrap()
            .digest(),
        base
    );
    // ply from root
    assert_ne!(
        mutate(&p, "ply_from_root")
            .query_identity(&parent)
            .unwrap()
            .digest(),
        base
    );
    // child state digest: every state-content field flows into it
    for f in STATE_DIGEST_FIELDS {
        assert_ne!(
            mutate(&p, f).query_identity(&parent).unwrap().digest(),
            base,
            "query identity ignores state field {f:?}"
        );
    }
    // ephemeral handles never enter it
    for f in EPHEMERAL_FIELDS {
        assert_eq!(
            mutate(&p, f).query_identity(&parent).unwrap().digest(),
            base,
            "query identity must not depend on ephemeral {f:?}"
        );
    }
}

#[test]
fn query_identity_is_persistent_across_managers_and_detects_wrong_edge_or_depth() {
    let run = || {
        let mut m = QueryManager::new(GameState::startpos()).unwrap();
        let ps = play(&mut m, &["e2e4", "e7e5", "g1f3"]);
        ps.iter()
            .map(|p| m.query_identity(p.node_id).unwrap().unwrap())
            .collect::<Vec<_>>()
    };
    let (a, b) = (run(), run());
    assert_eq!(a, b, "identity must not depend on the manager instance");
    // Consecutive edges differ; parent semantic id chains to the previous child.
    assert_ne!(a[0].digest(), a[1].digest());
    let mut m = QueryManager::new(GameState::startpos()).unwrap();
    let ps = play(&mut m, &["e2e4", "e7e5"]);
    assert_eq!(a[1].parent_semantic_id, ps[0].semantic_id);
    // A packet cached from one edge does not validate against another edge.
    let wrong_parent = ps[1].query_identity(&ps[1].semantic_id).unwrap();
    assert_ne!(wrong_parent, a[1], "wrong parent must be detected");
    let mut shifted = a[1].clone();
    shifted.ply_from_root += 2;
    assert_ne!(
        shifted.digest(),
        a[1].digest(),
        "wrong depth must be detected"
    );
    let mut other_edge = a[1].clone();
    other_edge.incoming_action += 1;
    assert_ne!(
        other_edge.digest(),
        a[1].digest(),
        "wrong edge must be detected"
    );
    // The root has no incoming edge.
    assert!(m.query_identity(0).unwrap().is_none());
    assert_eq!(
        m.packet(0).unwrap().query_identity("x"),
        Err(QueryError::RootHasNoEdge)
    );
}

#[test]
fn one_successful_query_has_one_transition_and_one_child_legal_generation() {
    let mut m = QueryManager::new(GameState::startpos()).unwrap();
    assert_eq!(m.legal_generations(), 1, "only the given root so far");
    assert_eq!(m.state_transitions(), 0);
    let mut moves = 0u64;
    let mut node = 0;
    for (i, uci) in ["e2e4", "e7e5", "g1f3", "b8c6"].iter().enumerate() {
        let a = action(m.state(node).unwrap(), uci);
        let p = m.query(node, a).unwrap();
        moves += p.legal_actions.len() as u64;
        node = p.node_id;
        let q = (i + 1) as u64;
        assert_eq!(m.successful_queries() as u64, q);
        assert_eq!(m.state_transitions(), q);
        assert_eq!(
            m.legal_generations(),
            1 + q,
            "exactly one child generation per query"
        );
        assert_eq!(m.legal_moves_generated(), moves, "counts child lists only");
    }
}

#[test]
fn refused_queries_do_no_chess_work() {
    let mut m = QueryManager::new(GameState::startpos())
        .unwrap()
        .with_budget(1);
    let a = m.unqueried(0).unwrap();
    m.query(0, a[0]).unwrap();
    let snapshot = (
        m.successful_queries(),
        m.state_transitions(),
        m.legal_generations(),
        m.legal_moves_generated(),
        m.node_count(),
    );
    let illegal = ActionId::encode(
        recur64_core::Square::A1,
        recur64_core::Square::A8,
        PromotionCode::NONE,
    )
    .index() as u16;
    assert!(matches!(
        m.query(0, illegal),
        Err(QueryError::IllegalAction { .. })
    ));
    assert!(matches!(
        m.query(0, a[0]),
        Err(QueryError::DuplicateEdge { .. })
    ));
    assert!(matches!(
        m.query(0, a[1]),
        Err(QueryError::BudgetExhausted { .. })
    ));
    assert!(matches!(
        m.query(42, a[1]),
        Err(QueryError::UnknownNode(42))
    ));
    // A terminal parent refuses too.
    let mut t = QueryManager::new(GameState::startpos()).unwrap();
    let ps = play(&mut t, &["f2f3", "e7e5", "g2g4", "d8h4"]);
    let t_snapshot = (t.legal_generations(), t.state_transitions());
    assert!(matches!(
        t.query(ps[3].node_id, 0),
        Err(QueryError::TerminalParent(_))
    ));
    assert_eq!((t.legal_generations(), t.state_transitions()), t_snapshot);
    assert_eq!(
        snapshot,
        (
            m.successful_queries(),
            m.state_transitions(),
            m.legal_generations(),
            m.legal_moves_generated(),
            m.node_count()
        ),
        "a refused query must not transition, generate or add nodes"
    );
}

#[test]
fn an_action_that_is_not_a_legal_move_index_is_refused_not_a_panic() {
    let mut m = QueryManager::new(GameState::startpos()).unwrap();
    // Outside the 20,480-id space.
    assert!(matches!(
        m.query(0, u16::MAX),
        Err(QueryError::IllegalAction { node: 0, .. })
    ));
    assert_eq!(m.legal_generations(), 1);
}
