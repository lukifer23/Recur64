//! Differential test of StateQueryV1 against the authoritative `GameState`
//! transition path.
//!
//! The reference path decodes each ActionId to physical squares itself and calls
//! `GameState::apply` directly, never going through the manager's legal-list
//! lookup. Every queried edge must match the reference child exactly.

use recur64_core::{
    ActionId, Color, GameState, OBS_LEN, PromotionCode, StandardMove, encode_observation_v1,
};
use recur64_statequery::{NodeId, QueryManager, StatePacketV1};

struct SplitMix(u64);
impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn reference_child(parent: &GameState, action: u16) -> GameState {
    let id = ActionId::from_index(u32::from(action)).unwrap();
    let (from, to, promo) = id.to_physical(parent.perspective());
    let mv = StandardMove {
        from,
        to,
        promotion: if promo == PromotionCode::NONE {
            None
        } else {
            Some(promo)
        },
    };
    let mut child = parent.clone();
    child
        .apply(mv)
        .expect("reference transition must accept every legal edge");
    child
}

/// Compare one packet with the reference child. Returns the number of checks made.
fn assert_matches(p: &StatePacketV1, reference: &GameState, ctx: &str) {
    // Observation and canonical perspective.
    let obs = encode_observation_v1(reference);
    assert_eq!(
        p.observation.as_slice(),
        obs.as_slice(),
        "{ctx}: observation"
    );
    assert_eq!(p.observation.as_slice().len(), OBS_LEN);
    // Independent perspective check: the current-frame own-piece channels
    // (0..=5) must sum to the side-to-move's piece count.
    let side = reference.side_to_move();
    let own_count = reference.board().colors(side).len() as f32;
    let opp_count = reference.board().colors(!side).len() as f32;
    let (mut own, mut opp) = (0.0f32, 0.0f32);
    for sq in 0..64 {
        for c in 0..6 {
            own += p.observation.get(sq, c);
        }
        for c in 6..12 {
            opp += p.observation.get(sq, c);
        }
    }
    assert_eq!(
        own, own_count,
        "{ctx}: own pieces must be canonical 'white'"
    );
    assert_eq!(opp, opp_count, "{ctx}: opponent pieces");
    // Legal list correspondence (empty iff terminal).
    let terminal = reference.is_terminal();
    assert_eq!(p.terminal, terminal, "{ctx}: terminal");
    assert_eq!(
        p.terminal_reason,
        reference.termination().map(|t| t.label()),
        "{ctx}: reason"
    );
    if terminal {
        assert!(
            p.legal_actions.is_empty(),
            "{ctx}: terminal nodes expose no frontier"
        );
    } else {
        let want: Vec<u16> = reference
            .legal_actions()
            .iter()
            .map(|a| a.index() as u16)
            .collect();
        assert_eq!(p.legal_actions, want, "{ctx}: legal actions");
        assert!(
            !p.legal_actions.is_empty(),
            "{ctx}: non-terminal must have moves"
        );
        assert!(
            p.legal_actions.windows(2).all(|w| w[0] < w[1]),
            "{ctx}: sorted, unique"
        );
    }
    assert_eq!(
        p.side_to_move,
        if side == Color::White {
            "white"
        } else {
            "black"
        }
    );
    assert_eq!(
        p.in_check,
        !reference.board().checkers().is_empty(),
        "{ctx}: check"
    );
    assert_eq!(
        p.repetition_count,
        reference.repetition_count(),
        "{ctx}: repetition"
    );
    assert_eq!(
        p.halfmove_clock,
        u32::from(reference.board().halfmove_clock()),
        "{ctx}: clock"
    );
}

/// Query every edge of every expanded node, walking `depth` plies (BFS) or a
/// random descent. Returns the number of edges verified.
fn verify_subtree(root: GameState, depth: u32, random: Option<&mut SplitMix>) -> u64 {
    let mut m = QueryManager::new(root.clone()).unwrap();
    let mut frontier: Vec<(NodeId, GameState)> = vec![(0, root)];
    let mut edges = 0u64;
    match random {
        None => {
            for _ in 0..depth {
                let mut next = Vec::new();
                for (node, rs) in &frontier {
                    for a in m.unqueried(*node).unwrap() {
                        let p = m.query(*node, a).unwrap();
                        let child = reference_child(rs, a);
                        assert_matches(&p, &child, &format!("bfs n{node} a{a}"));
                        edges += 1;
                        if !p.terminal {
                            next.push((p.node_id, child));
                        }
                    }
                }
                frontier = next;
            }
        }
        Some(rng) => {
            let (mut node, mut rs) = frontier.pop().unwrap();
            for ply in 0..depth {
                let acts = m.unqueried(node).unwrap();
                if acts.is_empty() {
                    break;
                }
                let mut children = Vec::new();
                for a in acts {
                    let p = m.query(node, a).unwrap();
                    let child = reference_child(&rs, a);
                    assert_matches(&p, &child, &format!("rnd ply{ply} n{node} a{a}"));
                    edges += 1;
                    if !p.terminal {
                        children.push((p.node_id, child));
                    }
                }
                if children.is_empty() {
                    break;
                }
                let k = rng.below(children.len());
                (node, rs) = children.swap_remove(k);
            }
        }
    }
    assert_eq!(
        u64::from(m.successful_queries()),
        edges,
        "counter equals verified edges"
    );
    edges
}

const FIXTURES: [&str; 9] = [
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", // castling both sides
    "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", // black castling
    "rnbqkbnr/ppp1p1pp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 3", // en passant
    "8/P6k/8/8/8/8/p6K/8 w - - 0 1",        // promotions
    "8/P6k/8/8/8/8/p6K/8 b - - 0 1",        // black promotions
    "7k/5Q2/8/6K1/8/8/8/8 w - - 0 1",       // stalemate / mate next
    "8/8/4k3/8/8/4K2R/8/8 w - - 99 80",     // fifty-move edge
    "4k3/8/8/8/8/8/4P3/4K2R w K - 0 1",     // castle with pawn
    "r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", // mate-in-1 available
];

#[test]
fn fixtures_bfs_depth_three() {
    let mut total = 0;
    for fen in FIXTURES {
        let gs = GameState::from_fen(fen).unwrap();
        total += verify_subtree(gs, 3, None);
    }
    assert!(total > 10_000, "fixtures exercised too few edges: {total}");
    eprintln!("fixture edges verified: {total}");
}

#[test]
fn startpos_bfs_depth_three() {
    let n = verify_subtree(GameState::startpos(), 3, None);
    assert!(n > 8_000, "{n}");
}

#[test]
fn repetition_through_history_matches_reference() {
    // Shuffle knights so the third occurrence is reached inside the queried path.
    let mut m = QueryManager::new(GameState::startpos()).unwrap();
    let mut reference = GameState::startpos();
    let mut node = 0;
    for uci in [
        "g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8",
    ] {
        let mv = StandardMove::from_uci(reference.board(), uci).unwrap();
        let a = ActionId::from_physical(
            mv.from,
            mv.to,
            mv.promotion.unwrap_or(PromotionCode::NONE),
            reference.perspective(),
        )
        .index() as u16;
        let p = m.query(node, a).unwrap();
        reference.apply(mv).unwrap();
        assert_matches(&p, &reference, uci);
        node = p.node_id;
    }
}

#[test]
fn random_games_two_hundred_thousand_edges() {
    let mut rng = SplitMix(0x5EED_0001_57A7_E001);
    let mut edges = 0u64;
    let mut games = 0u64;
    while edges < 200_000 {
        edges += verify_subtree(GameState::startpos(), 120, Some(&mut rng));
        games += 1;
    }
    eprintln!("random descent: {games} games, {edges} edges verified");
    assert!(edges >= 200_000);
}

#[test]
fn determinism_across_independent_managers() {
    let run = || {
        let mut m = QueryManager::new(GameState::startpos()).unwrap();
        let mut hashes = Vec::new();
        for a in m.unqueried(0).unwrap() {
            hashes.push(m.query(0, a).unwrap().state_hash);
        }
        hashes
    };
    assert_eq!(run(), run());
}
