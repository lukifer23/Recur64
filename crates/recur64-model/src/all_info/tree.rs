//! `all_info_depth2_v1`: the exhaustive raw depth-2 tree of one root.
//!
//! Every root successor, and for every non-terminal successor every opponent reply, is
//! obtained through the same exact `StateQuery` tool the active-search model uses. There
//! is no pruning, ordering heuristic, sampling or truncation. Only raw state fields are
//! kept (`StatePacketV1`: observation, legal actions, terminal/check flags, incoming
//! action); no solver, proof, value or label information exists on a packet, and no
//! `CandidateFactsV1` is ever computed for a descendant.

use recur64_core::GameState;
use recur64_statequery::{QueryManager, StatePacketV1};

/// One root move: its successor state and every reply to it.
#[derive(Debug, Clone)]
pub struct Branch {
    /// The state after the root move (depth 1).
    pub child: StatePacketV1,
    /// Every opponent reply to `child`, in ascending reply-`ActionId` order (depth 2).
    /// Empty iff `child` is terminal.
    pub replies: Vec<StatePacketV1>,
}

/// State counts of one tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct TreeCounts {
    pub root_legal: usize,
    pub depth1: usize,
    pub depth2: usize,
    pub terminal_depth1: usize,
}

impl TreeCounts {
    /// Total future states supplied (depth 1 plus depth 2).
    pub fn states(&self) -> usize {
        self.depth1 + self.depth2
    }
}

/// The exhaustive depth-2 tree of a root position.
#[derive(Debug, Clone)]
pub struct AllInfoTree {
    pub root: StatePacketV1,
    /// One branch per root legal action, in ascending `ActionId` order (the root
    /// candidate order).
    pub branches: Vec<Branch>,
}

impl AllInfoTree {
    /// Build the full depth-2 tree. A terminal root is refused.
    pub fn build(root: &GameState) -> anyhow::Result<Self> {
        let mut mgr = QueryManager::new(root.clone())?;
        let root_pkt = mgr.packet(0)?;
        anyhow::ensure!(
            !root_pkt.legal_actions.is_empty(),
            "terminal roots have no ALL-INFO tree"
        );
        let mut branches = Vec::with_capacity(root_pkt.legal_actions.len());
        for &action in &root_pkt.legal_actions {
            let child = mgr.query(0, action)?;
            let mut replies = Vec::with_capacity(child.legal_actions.len());
            if !child.terminal {
                for &reply in &child.legal_actions {
                    replies.push(mgr.query(child.node_id, reply)?);
                }
            } else {
                anyhow::ensure!(
                    child.legal_actions.is_empty(),
                    "a terminal state reported legal actions"
                );
            }
            branches.push(Branch { child, replies });
        }
        let tree = Self {
            root: root_pkt,
            branches,
        };
        let c = tree.counts();
        // The tool's own counters prove nothing was skipped: one query per supplied state.
        anyhow::ensure!(
            mgr.successful_queries() as usize == c.states(),
            "tree holds {} states but the tool answered {} queries",
            c.states(),
            mgr.successful_queries()
        );
        Ok(tree)
    }

    pub fn counts(&self) -> TreeCounts {
        TreeCounts {
            root_legal: self.branches.len(),
            depth1: self.branches.len(),
            depth2: self.branches.iter().map(|b| b.replies.len()).sum(),
            terminal_depth1: self.branches.iter().filter(|b| b.child.terminal).count(),
        }
    }
}
