//! StateQueryV1: a deliberately narrow exact transition tool.
//!
//! `QueryManager::query(parent, action)` expands exactly ONE legal tree edge and
//! returns a [`StatePacketV1`] for the child. One successful call is one unit of
//! query budget.
//!
//! # Operation story of one successful query
//!
//! 1. the requested ActionId is looked up in the parent's stored, sorted legal
//!    set (no move generation; the set was produced once when the parent was
//!    created);
//! 2. the ActionId is decoded directly, under the parent's canonical
//!    perspective, into the one physical move it names;
//! 3. the authoritative [`GameState::apply`] validates and performs that one
//!    transition (no sibling is transitioned or inspected);
//! 4. the child's legal list is generated exactly once.
//!
//! Refused queries (unknown node, terminal parent, illegal, duplicate, budget)
//! perform none of steps 2 to 4. The tool has no depth limit of its own: the
//! exact tree may follow any depth the budget allows. A model that can only
//! represent a bounded depth enforces that itself.
//!
//! The packet carries exact state facts only. It never carries solver-derived,
//! search-derived, value-derived or aggregate-tactical information; the field set
//! is frozen by `tests/whitelist.rs`, and this crate depends only on
//! `recur64-core`, so it cannot reach the mate solver or any model.
//!
//! The manager keeps an authoritative [`GameState`] (full history) per node so
//! castling, en passant, repetition and the fifty-move rule are exact.
//!
//! # Identities and digests
//!
//! * `semantic_id`: the rules-semantic identity of a state ([`semantic_state_id`]).
//! * [`StatePacketV1::state_digest`]: a digest over every *state-content* field
//!   of the packet (see [`STATE_DIGEST_FIELDS`]). It says what the state is and
//!   what the model observes of it; it says nothing about how it was reached.
//! * [`QueryIdentity`]: the persistent identity of one query edge: the parent's
//!   semantic id, the incoming ActionId, the ply from the root and the child's
//!   state digest. Persistent caches key on this, never on `node_id` or
//!   `parent_id`, which are ephemeral per manager.

use std::collections::BTreeMap;
use std::fmt;

use recur64_core::{
    ActionId, CoreError, GameState, MAX_LEGAL_MOVES, ObservationV1, PromotionCode, StandardMove,
    encode_observation_v1,
};
use serde::{Serialize, Serializer};
use sha2::{Digest, Sha256};

/// Contract version of the packet schema and digest definitions.
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

/// Fields covered by [`StatePacketV1::state_digest`]: every state-content field.
pub const STATE_DIGEST_FIELDS: [&str; 11] = [
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

/// Edge and path fields covered by [`QueryIdentity`] (through the packet's
/// `incoming_action` and `ply_from_root`; the parent is identified by semantic id).
pub const QUERY_IDENTITY_PACKET_FIELDS: [&str; 2] = ["incoming_action", "ply_from_root"];

/// Ephemeral per-manager handles: never part of a persistent digest or cache key.
pub const EPHEMERAL_FIELDS: [&str; 2] = ["node_id", "parent_id"];

pub type NodeId = u32;

/// Exact facts about one queried state. No solver, search or value fields.
#[derive(Debug, Clone, Serialize)]
pub struct StatePacketV1 {
    /// Ephemeral handle inside one [`QueryManager`].
    pub node_id: NodeId,
    /// Ephemeral handle inside one [`QueryManager`].
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
    /// SHA-256 (hex) over every state-content field listed in
    /// [`STATE_DIGEST_FIELDS`], observation included. It excludes the ephemeral
    /// handles and the edge/path fields (`incoming_action`, `ply_from_root`),
    /// which belong to [`QueryIdentity`]. Two packets with equal digests describe
    /// the same state and present the same observation, however they were
    /// reached.
    pub fn state_digest(&self) -> String {
        let mut h = Sha256::new();
        h.update(b"recur64.state_digest");
        h.update(STATE_QUERY_VERSION.to_le_bytes());
        h.update((self.semantic_id.len() as u32).to_le_bytes());
        h.update(self.semantic_id.as_bytes());
        for f in self.observation.as_slice() {
            h.update(f.to_bits().to_le_bytes());
        }
        h.update((self.legal_actions.len() as u32).to_le_bytes());
        for a in &self.legal_actions {
            h.update(a.to_le_bytes());
        }
        h.update((self.side_to_move.len() as u32).to_le_bytes());
        h.update(self.side_to_move.as_bytes());
        h.update([u8::from(self.in_check), u8::from(self.terminal)]);
        match self.terminal_reason {
            Some(r) => {
                h.update([1u8]);
                h.update((r.len() as u32).to_le_bytes());
                h.update(r.as_bytes());
            }
            None => h.update([0u8]),
        }
        h.update(self.castling.map(u8::from));
        match self.ep_square {
            Some(s) => h.update([1u8, s]),
            None => h.update([0u8, 0]),
        }
        h.update(self.halfmove_clock.to_le_bytes());
        h.update(self.repetition_count.to_le_bytes());
        hex(h)
    }

    /// The persistent identity of the edge that produced this packet, given the
    /// semantic id of the parent state. Errors for a root packet.
    pub fn query_identity(&self, parent_semantic_id: &str) -> Result<QueryIdentity, QueryError> {
        let incoming_action = self.incoming_action.ok_or(QueryError::RootHasNoEdge)?;
        Ok(QueryIdentity {
            parent_semantic_id: parent_semantic_id.to_string(),
            incoming_action,
            ply_from_root: self.ply_from_root,
            child_state_digest: self.state_digest(),
        })
    }
}

/// Persistent identity of one query edge, independent of ephemeral node ids.
///
/// A cached packet is valid for a live query only if the whole identity matches,
/// so a packet attached to the wrong edge or the wrong depth is detected.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct QueryIdentity {
    pub parent_semantic_id: String,
    pub incoming_action: u16,
    pub ply_from_root: u32,
    pub child_state_digest: String,
}

impl QueryIdentity {
    /// SHA-256 (hex) over all four components.
    pub fn digest(&self) -> String {
        let mut h = Sha256::new();
        h.update(b"recur64.query_identity");
        h.update(STATE_QUERY_VERSION.to_le_bytes());
        h.update((self.parent_semantic_id.len() as u32).to_le_bytes());
        h.update(self.parent_semantic_id.as_bytes());
        h.update(self.incoming_action.to_le_bytes());
        h.update(self.ply_from_root.to_le_bytes());
        h.update((self.child_state_digest.len() as u32).to_le_bytes());
        h.update(self.child_state_digest.as_bytes());
        hex(h)
    }
}

fn hex(h: Sha256) -> String {
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
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
    /// A query identity was requested for the root, which has no incoming edge.
    RootHasNoEdge,
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
            QueryError::RootHasNoEdge => write!(f, "the root has no incoming edge"),
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
    /// Sorted legal ActionIds; empty iff terminal. Generated once.
    legal: Vec<ActionId>,
    semantic_id: String,
    /// action index -> child node
    children: BTreeMap<u16, NodeId>,
}

/// Owns the query tree. Node 0 is the root (given, not queried).
pub struct QueryManager {
    nodes: Vec<Node>,
    budget: Option<u32>,
    successful_queries: u32,
    legal_moves_generated: u64,
    legal_generations: u64,
}

/// The sorted legal ActionIds of a state; empty when terminal. This is the only
/// place the tool generates legal moves: once per node.
fn legal_ids(state: &GameState) -> Vec<ActionId> {
    if state.is_terminal() {
        return Vec::new();
    }
    state.legal_actions()
}

impl QueryManager {
    /// New manager rooted at `root` with no budget cap. Generates the root's
    /// legal list (one generation; the root itself is given, not queried).
    pub fn new(root: GameState) -> Result<Self, QueryError> {
        let legal = legal_ids(&root);
        if legal.len() > MAX_LEGAL_MOVES {
            return Err(QueryError::TooManyLegalMoves {
                node: 0,
                count: legal.len(),
            });
        }
        let semantic_id = semantic_state_id(&root);
        Ok(Self {
            nodes: vec![Node {
                state: root,
                parent: None,
                incoming: None,
                ply: 0,
                legal,
                semantic_id,
                children: BTreeMap::new(),
            }],
            budget: None,
            successful_queries: 0,
            legal_moves_generated: 0,
            legal_generations: 1,
        })
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

    /// Exact transitions performed: equal to [`Self::successful_queries`].
    pub fn state_transitions(&self) -> u64 {
        u64::from(self.successful_queries)
    }

    /// Legal moves contained in the child lists generated by queries (the
    /// per-query cost). The given root's list is not counted.
    pub fn legal_moves_generated(&self) -> u64 {
        self.legal_moves_generated
    }

    /// Legal-list generations performed: one for the root plus exactly one per
    /// successful query.
    pub fn legal_generations(&self) -> u64 {
        self.legal_generations
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

    /// Persistent identity of the edge that created `id`; `None` for the root.
    pub fn query_identity(&self, id: NodeId) -> Result<Option<QueryIdentity>, QueryError> {
        let n = self.node(id)?;
        match n.parent {
            None => Ok(None),
            Some(p) => {
                let parent = self.node(p)?;
                Ok(Some(
                    build_packet(id, n).query_identity(&parent.semantic_id)?,
                ))
            }
        }
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
            // 1. membership in the stored, sorted legal set: no move generation.
            let id =
                ActionId::from_index(u32::from(action)).map_err(|_| QueryError::IllegalAction {
                    node: parent,
                    action,
                })?;
            if p.legal.binary_search(&id).is_err() {
                return Err(QueryError::IllegalAction {
                    node: parent,
                    action,
                });
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
            // 2. decode the one named move under the parent's canonical perspective.
            let (from, to, promo) = id.to_physical(p.state.perspective());
            let mv = StandardMove {
                from,
                to,
                promotion: if promo == PromotionCode::NONE {
                    None
                } else {
                    Some(promo)
                },
            };
            // 3. the authoritative transition: validates and applies that one move.
            let mut child = p.state.clone();
            child.apply(mv)?;
            (child, p.ply + 1)
        };
        // 4. the child's legal list, generated exactly once.
        let legal = legal_ids(&child_state);
        let id = self.nodes.len() as NodeId;
        if legal.len() > MAX_LEGAL_MOVES {
            return Err(QueryError::TooManyLegalMoves {
                node: id,
                count: legal.len(),
            });
        }
        self.legal_generations += 1;
        self.legal_moves_generated += legal.len() as u64;
        let semantic_id = semantic_state_id(&child_state);
        self.nodes.push(Node {
            state: child_state,
            parent: Some(parent),
            incoming: Some(action),
            ply,
            legal,
            semantic_id,
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
        semantic_id: n.semantic_id.clone(),
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
    hex(h)
}
