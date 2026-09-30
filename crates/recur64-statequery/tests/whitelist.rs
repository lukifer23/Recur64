//! The packet schema is frozen. Any added or renamed field fails here, and no
//! field name may suggest solver/search/value/aggregate-tactical information.

use recur64_core::{ActionId, GameState, PromotionCode, StandardMove};
use recur64_statequery::{PACKET_FIELDS, QueryManager};

const FROZEN: [&str; 15] = [
    "node_id",
    "parent_id",
    "incoming_action",
    "ply_from_root",
    "observation",
    "legal_actions",
    "side_to_move",
    "in_check",
    "terminal",
    "terminal_reason",
    "castling",
    "ep_square",
    "halfmove_clock",
    "repetition_count",
    "state_hash",
];

const PROHIBITED_SUBSTRINGS: [&str; 22] = [
    "forced",
    "mate",
    "dtm",
    "dtz",
    "tablebase",
    "tb_",
    "proof",
    "solver",
    "best",
    "visit",
    "puct",
    "mcts",
    "value",
    "wdl",
    "count_of",
    "num_winning",
    "winning",
    "quality",
    "score",
    "utility",
    "summary",
    "reply",
];

#[test]
fn serialized_packet_fields_equal_the_frozen_whitelist() {
    let mut m = QueryManager::new(GameState::startpos()).unwrap();
    let a = m.unqueried(0).unwrap()[0];
    let p = m.query(0, a).unwrap();
    let v = serde_json::to_value(&p).unwrap();
    let mut got: Vec<&str> = v.as_object().unwrap().keys().map(String::as_str).collect();
    let mut want: Vec<&str> = FROZEN.to_vec();
    got.sort_unstable();
    want.sort_unstable();
    assert_eq!(
        got, want,
        "serialized fields must equal the frozen whitelist"
    );
    let mut c: Vec<&str> = PACKET_FIELDS.to_vec();
    c.sort_unstable();
    assert_eq!(
        c, want,
        "PACKET_FIELDS constant must equal the frozen whitelist"
    );
}

#[test]
fn no_field_name_suggests_prohibited_information() {
    for f in FROZEN {
        for bad in PROHIBITED_SUBSTRINGS {
            assert!(
                !f.contains(bad),
                "field {f:?} contains prohibited token {bad:?}"
            );
        }
    }
}

#[test]
fn terminal_packets_serialize_the_same_schema() {
    let mut m = QueryManager::new(GameState::startpos()).unwrap();
    let mut node = 0;
    let mut last = None;
    for uci in ["f2f3", "e7e5", "g2g4", "d8h4"] {
        let gs = m.state(node).unwrap().clone();
        let mv = StandardMove::from_uci(gs.board(), uci).unwrap();
        let a = ActionId::from_physical(
            mv.from,
            mv.to,
            mv.promotion.unwrap_or(PromotionCode::NONE),
            gs.perspective(),
        )
        .index() as u16;
        let p = m.query(node, a).unwrap();
        node = p.node_id;
        last = Some(p);
    }
    let p = last.unwrap();
    assert!(p.terminal && p.in_check && p.legal_actions.is_empty());
    assert_eq!(p.terminal_reason, Some("checkmate"));
    let v = serde_json::to_value(&p).unwrap();
    assert_eq!(v.as_object().unwrap().len(), FROZEN.len());
}
