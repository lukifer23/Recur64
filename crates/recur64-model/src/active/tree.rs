//! Discovered-node and frontier records (`frontier_v1`).
//!
//! Pure Rust and deterministic: no tensors. Every discovered node keeps explicit
//! metadata outside the neural latent. V3.0 is a tree: transpositions (equal
//! `semantic_id`) are recorded but never merged.

use std::collections::HashMap;

use recur64_statequery::{NodeId, StatePacketV1};

use crate::config::ACTIVE_MAX_DEPTH;

/// Metadata of one discovered node. Slot 0 is the root.
#[derive(Debug, Clone)]
pub struct NodeMeta {
    /// Position in discovery order; also the index into the model's node tensors.
    pub slot: usize,
    /// Node id inside the exact query manager.
    pub id: NodeId,
    pub parent_slot: Option<usize>,
    pub incoming_action: Option<u16>,
    /// Root candidate (index into the root legal list) this node descends from;
    /// `None` only for the root itself.
    pub branch: Option<usize>,
    /// Plies below the root. Even depth: the root player is to move.
    pub depth: u32,
    pub in_check: bool,
    pub terminal: bool,
    /// Complete legal ActionIds (empty iff terminal), ascending.
    pub legal: Vec<u16>,
    queried: Vec<bool>,
    pub semantic_id: String,
}

impl NodeMeta {
    /// Root-relative turn parity: 0 when the root player moves at this node.
    pub fn parity(&self) -> u32 {
        self.depth % 2
    }

    pub fn is_queried(&self, pos: usize) -> bool {
        self.queried[pos]
    }
}

/// One legal, not-yet-queried edge of a discovered non-terminal node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EdgeRef {
    pub node_slot: usize,
    /// Index into the node's legal list.
    pub pos: usize,
    pub action: u16,
    /// Root candidate the edge belongs to (an edge at the root is its own branch).
    pub branch: usize,
    /// Ply depth of the parent node.
    pub parent_depth: u32,
}

impl EdgeRef {
    /// Ordering key of the frozen comparator `fixed_bfs_actionid_v1`.
    pub fn bfs_key(&self) -> (u32, usize, u16) {
        (self.parent_depth, self.node_slot, self.action)
    }
}

/// Discovered nodes of one search.
#[derive(Debug, Clone)]
pub struct Tree {
    nodes: Vec<NodeMeta>,
    seen: HashMap<String, usize>,
    transpositions: usize,
}

impl Tree {
    /// Start a tree from the root packet.
    pub fn new(root: &StatePacketV1) -> anyhow::Result<Self> {
        anyhow::ensure!(
            root.parent_id.is_none() && root.incoming_action.is_none(),
            "the root packet must have no parent"
        );
        let mut seen = HashMap::new();
        seen.insert(root.semantic_id.clone(), 0);
        Ok(Self {
            nodes: vec![meta_from(root, 0, None, None)],
            seen,
            transpositions: 0,
        })
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn node(&self, slot: usize) -> &NodeMeta {
        &self.nodes[slot]
    }

    pub fn nodes(&self) -> &[NodeMeta] {
        &self.nodes
    }

    /// Nodes whose `semantic_id` equals an earlier node's (recorded, not merged).
    pub fn transpositions(&self) -> usize {
        self.transpositions
    }

    pub fn terminal_count(&self) -> usize {
        self.nodes.iter().filter(|n| n.terminal).count()
    }

    /// All legal, not-yet-queried edges of every discovered non-terminal node,
    /// in (node slot, legal position) order. Deterministic.
    pub fn frontier(&self) -> Vec<EdgeRef> {
        let mut out = Vec::new();
        for n in &self.nodes {
            if n.terminal {
                continue;
            }
            for (pos, &action) in n.legal.iter().enumerate() {
                if !n.queried[pos] {
                    out.push(EdgeRef {
                        node_slot: n.slot,
                        pos,
                        action,
                        branch: n.branch.unwrap_or(pos),
                        parent_depth: n.depth,
                    });
                }
            }
        }
        out
    }

    /// Record the child returned by querying `edge`. Refuses an edge that is not
    /// on the current frontier, a packet that does not belong to the edge, and a
    /// depth beyond the supported range. Returns the child's slot.
    pub fn add_child(&mut self, edge: &EdgeRef, packet: &StatePacketV1) -> anyhow::Result<usize> {
        anyhow::ensure!(
            edge.node_slot < self.nodes.len(),
            "edge names an unknown node"
        );
        let parent = &self.nodes[edge.node_slot];
        anyhow::ensure!(!parent.terminal, "terminal nodes have no frontier");
        anyhow::ensure!(
            parent.legal.get(edge.pos) == Some(&edge.action),
            "edge action {} is not legal position {} of node {}",
            edge.action,
            edge.pos,
            edge.node_slot
        );
        anyhow::ensure!(
            !parent.queried[edge.pos],
            "edge ({}, {}) was already queried",
            edge.node_slot,
            edge.action
        );
        anyhow::ensure!(
            packet.parent_id == Some(parent.id) && packet.incoming_action == Some(edge.action),
            "packet does not belong to edge ({}, {})",
            edge.node_slot,
            edge.action
        );
        let depth = parent.depth + 1;
        anyhow::ensure!(
            depth as usize <= ACTIVE_MAX_DEPTH,
            "depth {depth} exceeds the supported range {ACTIVE_MAX_DEPTH}; refusing to clip"
        );
        let slot = self.nodes.len();
        let branch = edge.branch;
        self.nodes[edge.node_slot].queried[edge.pos] = true;
        if self.seen.contains_key(&packet.semantic_id) {
            self.transpositions += 1;
        } else {
            self.seen.insert(packet.semantic_id.clone(), slot);
        }
        self.nodes
            .push(meta_from(packet, slot, Some(edge.node_slot), Some(branch)));
        Ok(slot)
    }
}

fn meta_from(
    p: &StatePacketV1,
    slot: usize,
    parent_slot: Option<usize>,
    branch: Option<usize>,
) -> NodeMeta {
    NodeMeta {
        slot,
        id: p.node_id,
        parent_slot,
        incoming_action: p.incoming_action,
        branch,
        depth: p.ply_from_root,
        in_check: p.in_check,
        terminal: p.terminal,
        legal: p.legal_actions.clone(),
        queried: vec![false; p.legal_actions.len()],
        semantic_id: p.semantic_id.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use recur64_core::GameState;
    use recur64_statequery::QueryManager;

    fn setup(fen: &str) -> (QueryManager, Tree) {
        let m = QueryManager::new(GameState::from_fen(fen).unwrap()).unwrap();
        let t = Tree::new(&m.packet(0).unwrap()).unwrap();
        (m, t)
    }

    #[test]
    fn root_frontier_is_every_legal_move_each_its_own_branch() {
        let (_, t) = setup("4k3/8/8/8/8/8/3Q4/4K3 w - - 0 1");
        let f = t.frontier();
        assert_eq!(f.len(), t.node(0).legal.len());
        for (i, e) in f.iter().enumerate() {
            assert_eq!((e.node_slot, e.pos, e.branch, e.parent_depth), (0, i, i, 0));
        }
    }

    #[test]
    fn a_queried_edge_leaves_the_frontier_and_its_child_joins_it() {
        let (mut m, mut t) = setup("4k3/8/8/8/8/8/3Q4/4K3 w - - 0 1");
        let before = t.frontier();
        let e = before[2].clone();
        let p = m.query(0, e.action).unwrap();
        let slot = t.add_child(&e, &p).unwrap();
        assert_eq!(slot, 1);
        let after = t.frontier();
        assert!(!after.contains(&e));
        // Root edges minus one, plus the child's legal moves (depth 1, same branch).
        assert_eq!(after.len(), before.len() - 1 + t.node(1).legal.len());
        assert!(
            after
                .iter()
                .filter(|x| x.node_slot == 1)
                .all(|x| x.branch == 2 && x.parent_depth == 1)
        );
        assert_eq!(t.node(1).parity(), 1);
    }

    #[test]
    fn duplicate_and_foreign_edges_refuse() {
        let (mut m, mut t) = setup("4k3/8/8/8/8/8/3Q4/4K3 w - - 0 1");
        let f = t.frontier();
        let p = m.query(0, f[0].action).unwrap();
        t.add_child(&f[0], &p).unwrap();
        assert!(t.add_child(&f[0], &p).is_err(), "already queried");
        let p2 = m.query(0, f[1].action).unwrap();
        assert!(
            t.add_child(&f[3], &p2).is_err(),
            "packet belongs to another edge"
        );
    }

    #[test]
    fn terminal_children_expose_no_frontier() {
        // Mate in one: Qg8#.
        let (mut m, mut t) = setup("k7/8/1K6/8/8/8/8/6Q1 w - - 0 1");
        let mating = t
            .frontier()
            .into_iter()
            .find(|e| {
                let mut probe = QueryManager::new(
                    GameState::from_fen("k7/8/1K6/8/8/8/8/6Q1 w - - 0 1").unwrap(),
                )
                .unwrap();
                probe.query(0, e.action).unwrap().terminal
            })
            .expect("a mating move exists");
        let p = m.query(0, mating.action).unwrap();
        assert!(p.terminal);
        let slot = t.add_child(&mating, &p).unwrap();
        assert!(t.frontier().iter().all(|e| e.node_slot != slot));
        assert_eq!(t.terminal_count(), 1);
    }

    #[test]
    fn bfs_key_orders_by_depth_then_discovery_then_action() {
        let a = EdgeRef {
            node_slot: 3,
            pos: 0,
            action: 9,
            branch: 0,
            parent_depth: 1,
        };
        let b = EdgeRef {
            node_slot: 1,
            pos: 0,
            action: 99,
            branch: 0,
            parent_depth: 2,
        };
        let c = EdgeRef {
            node_slot: 3,
            pos: 1,
            action: 5,
            branch: 0,
            parent_depth: 1,
        };
        let mut v = [b.clone(), a.clone(), c.clone()];
        v.sort_by_key(EdgeRef::bfs_key);
        assert_eq!(v[0], c);
        assert_eq!(v[1], a);
        assert_eq!(v[2], b);
    }
}
