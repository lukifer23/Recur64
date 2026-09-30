//! StateQueryV1: a deliberately narrow exact transition tool.
//!
//! `QueryManager::query(parent, action)` expands exactly ONE legal tree edge and
//! returns a [`StatePacketV1`] for the child. One successful call is one unit of
//! query budget. Legal-move generation at the child is part of that call.
//!
//! The packet carries exact state facts only. It never carries solver-derived,
//! search-derived, value-derived or aggregate-tactical information; the field set
//! is frozen by `tests/whitelist.rs`, and this crate depends only on
//! `recur64-core`, so it cannot reach the mate solver or any model.
//!
//! The manager keeps an authoritative [`GameState`] (full history) per node so
//! castling, en passant, repetition and the fifty-move rule are exact.

use std::collections::BTreeMap;
use std::fmt;

use recur64_core::{
    ActionId, CoreError, GameState, MAX_LEGAL_MOVES, ObservationV1, PromotionCode, StandardMove,
    encode_observation_v1,
};
use serde::{Serialize, Serializer};
use sha2::{Digest, Sha256};

/// Contract version of the packet schema and hash definition.
pub const STATE_QUERY_VERSION: u32 = 1;

/// Version of the semantic state identity definition.
pub const STATE_IDENTITY_VERSION: u32 = 1;

/// Frozen packet field names, in serialization order.
pub const PACKET_FIELDS: [&str; 15] = [
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
    "semantic_id",
];

pub type NodeId = u32;

/// Exact facts about one queried state. No solver, search or value fields.
#[derive(Debug, Clone, Serialize)]
pub struct StatePacketV1 {
    pub node_id: NodeId,
    pub parent_id: Option<NodeId>,
    /// Canonical ActionId index taken from the parent (None for the root).
    pub incoming_action: Option<u16>,
    pub ply_from_root: u32,
    #[serde(serialize_with = "ser_observation")]
    pub observation: ObservationV1,
    /// Complete canonical legal ActionIds, ascending. Empty iff terminal.
    pub legal_actions: Vec<u16>,
    /// Physical side to move: "white" or "black".
    pub side_to_move: &'static str,
    pub in_check: bool,
    pub terminal: bool,
    /// `Termination::label()` when terminal.
    pub terminal_reason: Option<&'static str>,
    /// [own short, own long, opponent short, opponent long] (mover-relative).
    pub castling: [bool; 4],
    /// En-passant target, canonical square index.
    pub ep_square: Option<u8>,
    pub halfmove_clock: u32,
    pub repetition_count: u32,
    /// Semantic state identity (`STATE_IDENTITY_VERSION`): SHA-256 (hex) over
    /// everything Rules Profile V1 needs to decide future legality and
    /// termination. See [`semantic_state_id`]. Node ids are excluded.
    pub semantic_id: String,
}

impl StatePacketV1 {
    /// Digest of the complete packet content (observation included, node ids
    /// excluded). Used to prove cached and live packets are field-equivalent;
    /// it is NOT the identity of the state (see `semantic_id`).
    pub fn content_digest(&self) -> String {
        hash_packet(self)
    }
}

fn ser_observation<S: Serializer>(obs: &ObservationV1, s: S) -> Result<S::Ok, S::Error> {
    s.collect_seq(obs.as_slice().iter())
}

/// Visible refusals. Nothing is truncated, substituted or silently ignored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryError {
    UnknownNode(NodeId),
    /// The parent is terminal: it has no frontier.
    TerminalParent(NodeId),
    /// The ActionId is not legal at the parent.
    IllegalAction {
        node: NodeId,
        action: u16,
    },
    /// This edge was already queried.
    DuplicateEdge {
        node: NodeId,
        action: u16,
    },
    /// More legal moves than the fixed 256 capacity; never truncated.
    TooManyLegalMoves {
        node: NodeId,
        count: usize,
    },
    BudgetExhausted {
        budget: u32,
    },
    Core(CoreError),
}

impl fmt::Display for QueryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QueryError::UnknownNode(n) => write!(f, "unknown node {n}"),
            QueryError::TerminalParent(n) => write!(f, "node {n} is terminal; it has no frontier"),
            QueryError::IllegalAction { node, action } => {
                write!(f, "action {action} is not legal at node {node}")
            }
            QueryError::DuplicateEdge { node, action } => {
                write!(f, "edge ({node}, {action}) was already queried")
            }
            QueryError::TooManyLegalMoves { node, count } => {
                write!(
                    f,
                    "node {node} has {count} legal moves (> {MAX_LEGAL_MOVES})"
                )
            }
            QueryError::BudgetExhausted { budget } => {
                write!(f, "query budget {budget} exhausted")
            }
            QueryError::Core(e) => write!(f, "core error: {e}"),
        }
    }
}

impl std::error::Error for QueryError {}

impl From<CoreError> for QueryError {
    fn from(e: CoreError) -> Self {
        QueryError::Core(e)
    }
}

struct Node {
    state: GameState,
    parent: Option<NodeId>,
    incoming: Option<u16>,
    ply: u32,
    legal: Vec<ActionId>,
    /// action index -> child node
    children: BTreeMap<u16, NodeId>,
}

/// Owns the query tree. Node 0 is the root (given, not queried).
pub struct QueryManager {
    nodes: Vec<Node>,
    budget: Option<u32>,
    successful_queries: u32,
    legal_moves_generated: u64,
}

fn action_of(state: &GameState, mv: &StandardMove) -> ActionId {
    ActionId::from_physical(
        mv.from,
        mv.to,
        mv.promotion.unwrap_or(PromotionCode::NONE),
        state.perspective(),
    )
}

/// Legal ActionIds with their standard moves; empty when terminal.
fn legal_edges(state: &GameState) -> Vec<(ActionId, StandardMove)> {
    if state.is_terminal() {
        return Vec::new();
    }
    let mut v: Vec<(ActionId, StandardMove)> = state
        .legal_standard_moves()
        .into_iter()
        .map(|mv| (action_of(state, &mv), mv))
        .collect();
    v.sort_unstable_by_key(|(a, _)| *a);
    v
}

impl QueryManager {
    /// New manager rooted at `root` with no budget cap.
    pub fn new(root: GameState) -> Result<Self, QueryError> {
        let mut m = Self {
            nodes: Vec::new(),
            budget: None,
            successful_queries: 0,
            legal_moves_generated: 0,
        };
        let legal = legal_edges(&root);
        if legal.len() > MAX_LEGAL_MOVES {
            return Err(QueryError::TooManyLegalMoves {
                node: 0,
                count: legal.len(),
            });
        }
        m.nodes.push(Node {
            state: root,
            parent: None,
            incoming: None,
            ply: 0,
            legal: legal.into_iter().map(|(a, _)| a).collect(),
            children: BTreeMap::new(),
        });
        Ok(m)
    }

    /// Cap on successful queries; a further query errors visibly.
    pub fn with_budget(mut self, budget: u32) -> Self {
        self.budget = Some(budget);
        self
    }

    pub fn budget(&self) -> Option<u32> {
        self.budget
    }

    pub fn successful_queries(&self) -> u32 {
        self.successful_queries
    }

    pub fn legal_moves_generated(&self) -> u64 {
        self.legal_moves_generated
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    fn node(&self, id: NodeId) -> Result<&Node, QueryError> {
        self.nodes
            .get(id as usize)
            .ok_or(QueryError::UnknownNode(id))
    }

    /// Authoritative state of a node (read-only).
    pub fn state(&self, id: NodeId) -> Result<&GameState, QueryError> {
        Ok(&self.node(id)?.state)
    }

    /// The packet for any already-known node. The root packet is given, not queried.
    pub fn packet(&self, id: NodeId) -> Result<StatePacketV1, QueryError> {
        let n = self.node(id)?;
        Ok(build_packet(id, n))
    }

    /// Legal, not-yet-queried ActionIds at `id`, ascending. Empty if terminal.
    pub fn unqueried(&self, id: NodeId) -> Result<Vec<u16>, QueryError> {
        let n = self.node(id)?;
        Ok(n.legal
            .iter()
            .map(|a| a.index() as u16)
            .filter(|a| !n.children.contains_key(a))
            .collect())
    }

    /// Child node for an already-queried edge.
    pub fn child(&self, id: NodeId, action: u16) -> Result<Option<NodeId>, QueryError> {
        Ok(self.node(id)?.children.get(&action).copied())
    }

    /// Expand exactly one legal edge. Consumes exactly one budget unit.
    pub fn query(&mut self, parent: NodeId, action: u16) -> Result<StatePacketV1, QueryError> {
        let (child_state, ply) = {
            let p = self.node(parent)?;
            if p.state.is_terminal() {
                return Err(QueryError::TerminalParent(parent));
            }
            if p.children.contains_key(&action) {
                return Err(QueryError::DuplicateEdge {
                    node: parent,
                    action,
                });
            }
            if let Some(b) = self.budget
                && self.successful_queries >= b
            {
                return Err(QueryError::BudgetExhausted { budget: b });
            }
            let mv = legal_edges(&p.state)
                .into_iter()
                .find(|(a, _)| a.index() == u32::from(action))
                .map(|(_, mv)| mv)
                .ok_or(QueryError::IllegalAction {
                    node: parent,
                    action,
                })?;
            let mut child = p.state.clone();
            child.apply(mv)?;
            (child, p.ply + 1)
        };
        let legal = legal_edges(&child_state);
        let id = self.nodes.len() as NodeId;
        if legal.len() > MAX_LEGAL_MOVES {
            return Err(QueryError::TooManyLegalMoves {
                node: id,
                count: legal.len(),
            });
        }
        self.legal_moves_generated += legal.len() as u64;
        self.nodes.push(Node {
            state: child_state,
            parent: Some(parent),
            incoming: Some(action),
            ply,
            legal: legal.into_iter().map(|(a, _)| a).collect(),
            children: BTreeMap::new(),
        });
        self.nodes[parent as usize].children.insert(action, id);
        self.successful_queries += 1;
        Ok(build_packet(id, &self.nodes[id as usize]))
    }
}

fn build_packet(id: NodeId, n: &Node) -> StatePacketV1 {
    let s = &n.state;
    let board = s.board();
    let side = s.side_to_move();
    let own = board.castle_rights(side);
    let opp = board.castle_rights(!side);
    let castling = [
        own.short.is_some(),
        own.long.is_some(),
        opp.short.is_some(),
        opp.long.is_some(),
    ];
    let ep_square = board.en_passant().map(|file| {
        let rank = if side == recur64_core::Color::White {
            recur64_core::Rank::Sixth
        } else {
            recur64_core::Rank::Third
        };
        let sq = recur64_core::Square::new(file, rank);
        s.perspective().square(sq) as u8
    });
    let termination = s.termination();
    StatePacketV1 {
        node_id: id,
        parent_id: n.parent,
        incoming_action: n.incoming,
        ply_from_root: n.ply,
        observation: encode_observation_v1(s),
        legal_actions: n.legal.iter().map(|a| a.index() as u16).collect(),
        side_to_move: if side == recur64_core::Color::White {
            "white"
        } else {
            "black"
        },
        in_check: !board.checkers().is_empty(),
        terminal: termination.is_some(),
        terminal_reason: termination.map(|t| t.label()),
        castling,
        ep_square,
        halfmove_clock: u32::from(board.halfmove_clock()),
        repetition_count: s.repetition_count(),
        semantic_id: semantic_state_id(s),
    }
}

/// FEN without the halfmove and fullmove counters: placement, side to move,
/// castling rights and en-passant square.
fn position_key(board: &recur64_core::Board) -> String {
    let fen = board.to_string();
    fen.split(' ').take(4).collect::<Vec<_>>().join(" ")
}

/// Semantic state identity, `STATE_IDENTITY_VERSION` 1.
///
/// Two states share an id only if Rules Profile V1 treats their futures
/// identically. It therefore covers, beyond board placement:
///
/// * side to move, castling rights and the en-passant square (the position key);
/// * the halfmove clock (fifty-move rule);
/// * the repetition history: the multiset of position keys of every earlier
///   position that can still recur (those since the last irreversible move,
///   i.e. the last `halfmove_clock + 1` positions, clipped to the recorded
///   history), because the threefold count of a later position depends on it;
/// * the administrative ply cap and, when a cap is set, the ply count.
///
/// It deliberately excludes the fullmove number, the move order that led here,
/// node ids and anything the model observes but the rules ignore (the 8-frame
/// observation window). The identity is conservative: it never merges states
/// whose rule behaviour can differ, at the cost of treating some same-placement
/// states as distinct.
pub fn semantic_state_id(state: &GameState) -> String {
    let board = state.board();
    let clock = u32::from(board.halfmove_clock());
    let hist = state.history();
    let window = (clock as usize + 1).min(hist.len());
    let mut keys: Vec<String> = hist[hist.len() - window..]
        .iter()
        .map(position_key)
        .collect();
    keys.sort_unstable();
    let mut h = Sha256::new();
    h.update(b"recur64.state_identity");
    h.update(STATE_IDENTITY_VERSION.to_le_bytes());
    let cur = position_key(board);
    h.update((cur.len() as u32).to_le_bytes());
    h.update(cur.as_bytes());
    h.update(clock.to_le_bytes());
    h.update((keys.len() as u32).to_le_bytes());
    for k in &keys {
        h.update((k.len() as u32).to_le_bytes());
        h.update(k.as_bytes());
    }
    match state.max_plies() {
        Some(cap) => {
            h.update([1u8]);
            h.update(cap.to_le_bytes());
            h.update(state.ply().to_le_bytes());
        }
        None => h.update([0u8]),
    }
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// Packet content digest: fixed field order, little-endian, length-prefixed.
/// Node/parent ids and the incoming action are excluded.
fn hash_packet(p: &StatePacketV1) -> String {
    let mut h = Sha256::new();
    h.update(b"recur64.state_packet_v1");
    h.update(STATE_QUERY_VERSION.to_le_bytes());
    h.update((p.semantic_id.len() as u32).to_le_bytes());
    h.update(p.semantic_id.as_bytes());
    for f in p.observation.as_slice() {
        h.update(f.to_bits().to_le_bytes());
    }
    h.update((p.legal_actions.len() as u32).to_le_bytes());
    for a in &p.legal_actions {
        h.update(a.to_le_bytes());
    }
    h.update([
        p.side_to_move.as_bytes()[0],
        p.in_check as u8,
        p.terminal as u8,
    ]);
    match p.terminal_reason {
        Some(r) => {
            h.update([1u8]);
            h.update((r.len() as u32).to_le_bytes());
            h.update(r.as_bytes());
        }
        None => h.update([0u8]),
    }
    h.update(p.castling.map(u8::from));
    match p.ep_square {
        Some(s) => h.update([1u8, s]),
        None => h.update([0u8, 0]),
    }
    h.update(p.halfmove_clock.to_le_bytes());
    h.update(p.repetition_count.to_le_bytes());
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}
