//! Frozen, answer-independent V5 acquisition schedules and graph manifests.

use recur64_core::{
    ActionId, Color, GameState, Perspective, RootRelativeObservationV1,
    encode_root_relative_observation_v1,
};
use recur64_statequery::{NodeId, QueryManager};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{ACTION_GEOMETRY, PAYLOAD_FLAGS, splitmix64};

pub const MAX_PILOT_Q: usize = 8;
pub const MAX_ENGINEERING_Q: usize = 16;
pub const RANKED_MAX_DEPTH: u8 = 5;
pub const MAX_BRANCH_EDGES: usize = 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Schedule {
    UniformFrontierV1,
    BaseRankedDepthV1,
}

impl Schedule {
    pub fn id(self) -> &'static str {
        match self {
            Self::UniformFrontierV1 => "uniform_frontier_v1",
            Self::BaseRankedDepthV1 => "base_ranked_depth_v1",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EpisodeKey {
    pub position_id: String,
    pub schedule: Schedule,
    pub run_seed: u64,
    pub occurrence_ordinal: u64,
}

impl EpisodeKey {
    pub fn seed(&self) -> u64 {
        let mut h = Sha256::new();
        h.update(b"recur64.v5.episode_key.v1\0");
        h.update(self.position_id.as_bytes());
        h.update([0]);
        h.update(self.schedule.id().as_bytes());
        h.update(self.run_seed.to_le_bytes());
        h.update(self.occurrence_ordinal.to_le_bytes());
        u64::from_le_bytes(h.finalize()[..8].try_into().expect("eight bytes"))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReturnedPayload {
    pub observation: Vec<f32>,
    pub flags: [f32; PAYLOAD_FLAGS],
    pub state_digest: String,
    pub semantic_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AcquiredNode {
    pub storage_id: u32,
    pub parent: Option<u32>,
    pub root_candidate: usize,
    pub incoming_action_root_frame: u16,
    pub action_geometry: [f32; ACTION_GEOMETRY],
    pub depth: u8,
    pub root_to_move: bool,
    pub path: Vec<u16>,
    /// Exact cumulative StateQuery counters immediately after this edge.
    pub cumulative_legal_generations: u64,
    pub cumulative_legal_moves_generated: u64,
    pub payload: ReturnedPayload,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AcquiredGraph {
    pub schema: String,
    pub config_digest: String,
    pub episode: EpisodeKey,
    pub requested_q: usize,
    pub actual_q: usize,
    pub exhausted_frontier: bool,
    pub root_player: String,
    pub nodes: Vec<AcquiredNode>,
    pub successful_queries: u64,
    pub legal_generations: u64,
    pub legal_moves_generated: u64,
    pub digest: String,
}

/// Persistence-only envelope: AcquiredGraph scientific semantics are unchanged.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphArtifact {
    pub schema: String,
    pub source_sha: String,
    pub architecture: String,
    pub config_digest: String,
    pub graph_digest: String,
    pub structure_digest: String,
    pub generation_role: String,
    pub graph: AcquiredGraph,
}
impl GraphArtifact {
    pub fn new(graph: AcquiredGraph, source_sha: &str) -> anyhow::Result<Self> {
        graph.verify()?;
        anyhow::ensure!(
            source_sha.len() == 40 && source_sha.bytes().all(|c| c.is_ascii_hexdigit()),
            "invalid graph source SHA"
        );
        Ok(Self {
            schema: "v5_graph_artifact_v1".into(),
            source_sha: source_sha.into(),
            architecture: crate::config::ARCHITECTURE.into(),
            config_digest: graph.config_digest.clone(),
            graph_digest: graph.digest.clone(),
            structure_digest: graph.compute_structure_digest()?,
            generation_role: "v5 graph generate".into(),
            graph,
        })
    }
    pub fn verify(&self, current_source: &str) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.schema == "v5_graph_artifact_v1" && self.source_sha == current_source,
            "stale or unversioned graph source provenance"
        );
        anyhow::ensure!(
            self.architecture == crate::config::ARCHITECTURE
                && self.config_digest == crate::config::V5Config::default().scientific_digest()?,
            "graph artifact configuration mismatch"
        );
        self.graph.verify()?;
        anyhow::ensure!(
            self.graph.config_digest == self.config_digest
                && self.graph.digest == self.graph_digest
                && self.graph.compute_structure_digest()? == self.structure_digest
                && self.generation_role == "v5 graph generate",
            "graph artifact metadata/content mismatch"
        );
        Ok(())
    }
    pub fn load(path: &std::path::Path, source: &str) -> anyhow::Result<Self> {
        let artifact: Self = serde_json::from_slice(&std::fs::read(path)?)?;
        artifact.verify(source)?;
        Ok(artifact)
    }
}

impl AcquiredGraph {
    pub fn prefix(&self, q: usize) -> anyhow::Result<Self> {
        anyhow::ensure!(
            q <= self.requested_q,
            "prefix Q{q} exceeds requested Q{}",
            self.requested_q
        );
        let mut out = self.clone();
        out.requested_q = q;
        out.nodes.truncate(q.min(self.nodes.len()));
        out.actual_q = out.nodes.len();
        out.successful_queries = out.actual_q as u64;
        if let Some(last) = out.nodes.last() {
            out.legal_generations = last.cumulative_legal_generations;
            out.legal_moves_generated = last.cumulative_legal_moves_generated;
        } else {
            out.legal_generations = 0;
            out.legal_moves_generated = 0;
        }
        out.exhausted_frontier = self.exhausted_frontier && out.actual_q < q;
        out.digest.clear();
        out.digest = out.compute_digest()?;
        Ok(out)
    }

    pub fn compute_digest(&self) -> anyhow::Result<String> {
        let mut copy = self.clone();
        copy.digest.clear();
        let mut h = Sha256::new();
        h.update(b"recur64.v5.graph_manifest.v1\0");
        h.update(serde_json::to_vec(&copy)?);
        Ok(format!("{:x}", h.finalize()))
    }

    pub fn compute_structure_digest(&self) -> anyhow::Result<String> {
        let nodes: Vec<_> = self
            .nodes
            .iter()
            .map(|node| {
                serde_json::json!({
                    "storage_id": node.storage_id,
                    "parent": node.parent,
                    "root_candidate": node.root_candidate,
                    "incoming_action_root_frame": node.incoming_action_root_frame,
                    "action_geometry": node.action_geometry,
                    "depth": node.depth,
                    "root_to_move": node.root_to_move,
                    "path": node.path,
                    "cumulative_legal_generations": node.cumulative_legal_generations,
                    "cumulative_legal_moves_generated": node.cumulative_legal_moves_generated,
                })
            })
            .collect();
        let structure = serde_json::json!({
            "schema": self.schema,
            "config_digest": self.config_digest,
            "episode": self.episode,
            "requested_q": self.requested_q,
            "actual_q": self.actual_q,
            "exhausted_frontier": self.exhausted_frontier,
            "root_player": self.root_player,
            "nodes": nodes,
            "successful_queries": self.successful_queries,
            "legal_generations": self.legal_generations,
            "legal_moves_generated": self.legal_moves_generated,
        });
        let mut hash = Sha256::new();
        hash.update(b"recur64.v5.graph_structure.v1\0");
        hash.update(serde_json::to_vec(&structure)?);
        Ok(format!("{:x}", hash.finalize()))
    }

    pub fn verify(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.schema == "v5_graph_manifest_v2",
            "graph schema mismatch"
        );
        anyhow::ensure!(
            self.config_digest == crate::config::V5Config::default().scientific_digest()?,
            "graph configuration digest mismatch"
        );
        anyhow::ensure!(
            self.actual_q <= self.requested_q && self.requested_q <= MAX_ENGINEERING_Q,
            "graph Q out of range"
        );
        anyhow::ensure!(
            self.actual_q == self.nodes.len(),
            "actual Q does not match node count"
        );
        anyhow::ensure!(
            self.successful_queries == self.actual_q as u64,
            "query accounting mismatch"
        );
        anyhow::ensure!(
            self.nodes.iter().all(|n| n.depth > 0
                && usize::from(n.depth) <= crate::DEPTH_FEATURES
                && usize::from(n.depth) <= self.actual_q),
            "graph depth out of range"
        );
        anyhow::ensure!(
            self.nodes
                .iter()
                .all(|n| n.payload.observation.len() == recur64_core::OBS_LEN),
            "payload observation length mismatch"
        );
        for (i, n) in self.nodes.iter().enumerate() {
            anyhow::ensure!(n.storage_id as usize == i, "storage ids are not dense");
            anyhow::ensure!(n.path.len() == usize::from(n.depth), "path/depth mismatch");
            if let Some(p) = n.parent {
                anyhow::ensure!((p as usize) < self.nodes.len(), "parent id is absent");
                anyhow::ensure!(p != n.storage_id, "node cannot parent itself");
                let parent = &self.nodes[p as usize];
                anyhow::ensure!(
                    n.depth == parent.depth + 1
                        && n.root_candidate == parent.root_candidate
                        && n.path[..n.path.len() - 1] == parent.path,
                    "parent path/depth/ownership mismatch"
                );
            } else {
                anyhow::ensure!(n.depth == 1, "root edge depth must be one");
            }
        }
        if self.episode.schedule == Schedule::BaseRankedDepthV1 {
            let mut branch_counts = std::collections::BTreeMap::new();
            for n in &self.nodes {
                anyhow::ensure!(n.depth <= RANKED_MAX_DEPTH, "ranked DFS depth exceeded");
                *branch_counts.entry(n.root_candidate).or_insert(0usize) += 1;
            }
            anyhow::ensure!(
                branch_counts.values().all(|n| *n <= MAX_BRANCH_EDGES),
                "ranked DFS branch budget exceeded"
            );
        }
        for n in &self.nodes {
            let mut cursor = n.parent;
            for _ in 0..self.nodes.len() {
                let Some(parent) = cursor else { break };
                cursor = self.nodes[parent as usize].parent;
            }
            anyhow::ensure!(cursor.is_none(), "parent relation contains a cycle");
        }
        anyhow::ensure!(
            self.digest == self.compute_digest()?,
            "graph manifest digest mismatch"
        );
        Ok(())
    }
}

#[derive(Clone)]
struct Meta {
    storage: u32,
    root_candidate: usize,
    path: Vec<u16>,
}

type FrontierEdge = (Vec<u16>, NodeId, u16, usize, Option<u32>);

// Complete current legal frontier. Depth five limits ranked DFS only; Q bounds
// the deepest possible acquired uniform path, not frontier enumeration.
fn uniform_frontier(
    qm: &QueryManager,
    metas: &[Option<Meta>],
    root_actions: &[ActionId],
) -> anyhow::Result<Vec<FrontierEdge>> {
    let mut frontier = Vec::new();
    for qid in 0..qm.node_count() as NodeId {
        let (parent_storage, root_candidate, path) = if qid == 0 {
            (None, usize::MAX, Vec::new())
        } else {
            let m = metas[qid as usize]
                .as_ref()
                .expect("metadata for acquired node");
            (Some(m.storage), m.root_candidate, m.path.clone())
        };
        for action in qm.unqueried(qid)? {
            let branch = if qid == 0 {
                root_candidate_index(root_actions, action)?
            } else {
                root_candidate
            };
            let mut child_path = path.clone();
            child_path.push(action);
            frontier.push((child_path, qid, action, branch, parent_storage));
        }
    }
    Ok(frontier)
}

/// Refuse a representation that cannot cover the complete authorized frontier.
/// Only the ranked schedule is authorized to stop at five.
pub fn validate_uniform_frontier_contract() -> anyhow::Result<()> {
    anyhow::ensure!(
        crate::DEPTH_FEATURES >= MAX_ENGINEERING_Q,
        "V5 acquisition contract violation: depth representation cannot cover the authorized uniform query budget"
    );
    Ok(())
}

fn action_geometry(id: ActionId) -> [f32; ACTION_GEOMETRY] {
    let (from, to, promo) = id.decode();
    let ff = (from as usize % 8) as f32;
    let fr = (from as usize / 8) as f32;
    let tf = (to as usize % 8) as f32;
    let tr = (to as usize / 8) as f32;
    let mut out = [0.0; ACTION_GEOMETRY];
    out[0] = ff / 7.0;
    out[1] = fr / 7.0;
    out[2] = tf / 7.0;
    out[3] = tr / 7.0;
    out[4] = (tf - ff) / 7.0;
    out[5] = (tr - fr) / 7.0;
    out[6 + promo.code() as usize] = 1.0;
    out
}

fn flags(
    in_check: bool,
    terminal: bool,
    reason: Option<&str>,
) -> anyhow::Result<[f32; PAYLOAD_FLAGS]> {
    let mut out = [0.0; PAYLOAD_FLAGS];
    out[0] = f32::from(u8::from(in_check));
    out[1] = f32::from(u8::from(terminal));
    let at = match reason {
        Some("checkmate") => Some(2),
        Some("stalemate") => Some(3),
        Some("insufficient_material") => Some(4),
        Some("threefold_repetition") => Some(5),
        Some("fifty_move_rule") => Some(6),
        Some("truncated") => Some(7),
        Some("aborted") => Some(8),
        None => None,
        Some(other) => anyhow::bail!("unknown Rules Profile termination {other}"),
    };
    if let Some(i) = at {
        out[i] = 1.0;
    }
    Ok(out)
}

fn hash_path(seed: u64, path: &[u16], action: u16) -> u64 {
    let mut h = Sha256::new();
    h.update(b"recur64.v5.path_order.v1\0");
    h.update(seed.to_le_bytes());
    for p in path {
        h.update(p.to_le_bytes());
    }
    h.update(action.to_le_bytes());
    u64::from_le_bytes(h.finalize()[..8].try_into().expect("eight bytes"))
}

#[allow(clippy::too_many_arguments)]
fn acquire_edge(
    qm: &mut QueryManager,
    metas: &mut Vec<Option<Meta>>,
    nodes: &mut Vec<AcquiredNode>,
    parent_qid: NodeId,
    action: u16,
    root_candidate: usize,
    parent_storage: Option<u32>,
    path: Vec<u16>,
    root_player: Color,
) -> anyhow::Result<NodeId> {
    let parent_perspective = qm.state(parent_qid)?.perspective();
    let root_perspective = Perspective::of(root_player);
    let root_action =
        ActionId::from_index(u32::from(action))?.reframe(parent_perspective, root_perspective);
    let packet = qm.query(parent_qid, action)?;
    let state = qm.state(packet.node_id)?;
    let observation: RootRelativeObservationV1 =
        encode_root_relative_observation_v1(state, root_player);
    let storage = nodes.len() as u32;
    nodes.push(AcquiredNode {
        storage_id: storage,
        parent: parent_storage,
        root_candidate,
        incoming_action_root_frame: root_action.index() as u16,
        action_geometry: action_geometry(root_action),
        depth: packet
            .ply_from_root
            .try_into()
            .map_err(|_| anyhow::anyhow!("depth overflow"))?,
        root_to_move: state.side_to_move() == root_player,
        path: path.clone(),
        cumulative_legal_generations: qm.legal_generations(),
        cumulative_legal_moves_generated: qm.legal_moves_generated(),
        payload: ReturnedPayload {
            observation: observation.as_slice().to_vec(),
            flags: flags(packet.in_check, packet.terminal, packet.terminal_reason)?,
            state_digest: packet.state_digest(),
            semantic_id: packet.semantic_id,
        },
    });
    while metas.len() <= packet.node_id as usize {
        metas.push(None);
    }
    metas[packet.node_id as usize] = Some(Meta {
        storage,
        root_candidate,
        path,
    });
    Ok(packet.node_id)
}

fn root_candidate_index(root_actions: &[ActionId], action: u16) -> anyhow::Result<usize> {
    root_actions
        .iter()
        .position(|a| a.index() as u16 == action)
        .ok_or_else(|| anyhow::anyhow!("root action {action} absent"))
}

pub fn acquire(
    root: &GameState,
    episode: EpisodeKey,
    requested_q: usize,
    base_logits: Option<&[f32]>,
) -> anyhow::Result<AcquiredGraph> {
    anyhow::ensure!(
        (1..=MAX_ENGINEERING_Q).contains(&requested_q),
        "Q must be 1..={MAX_ENGINEERING_Q}"
    );
    let root_actions = root.legal_actions();
    anyhow::ensure!(
        !root_actions.is_empty(),
        "cannot acquire from a terminal root"
    );
    if episode.schedule == Schedule::BaseRankedDepthV1 {
        let z = base_logits
            .ok_or_else(|| anyhow::anyhow!("base_ranked_depth_v1 requires frozen z0"))?;
        anyhow::ensure!(z.len() == root_actions.len(), "z0/legal width mismatch");
        anyhow::ensure!(
            z.iter().all(|v| v.is_finite()),
            "z0 contains non-finite values"
        );
    }
    let root_player = root.side_to_move();
    let mut qm = QueryManager::new(root.clone())?.with_budget(requested_q as u32);
    let mut metas: Vec<Option<Meta>> = vec![None];
    let mut nodes = Vec::with_capacity(requested_q);
    let seed = episode.seed();

    match episode.schedule {
        Schedule::UniformFrontierV1 => {
            let mut rng = seed;
            while nodes.len() < requested_q {
                let mut frontier = uniform_frontier(&qm, &metas, &root_actions)?;
                if frontier.is_empty() {
                    break;
                }
                frontier.sort_by(|a, b| a.0.cmp(&b.0).then(a.2.cmp(&b.2)));
                let pick = (splitmix64(&mut rng) as usize) % frontier.len();
                let (path, parent, action, branch, parent_storage) = frontier.swap_remove(pick);
                acquire_edge(
                    &mut qm,
                    &mut metas,
                    &mut nodes,
                    parent,
                    action,
                    branch,
                    parent_storage,
                    path,
                    root_player,
                )?;
            }
        }
        Schedule::BaseRankedDepthV1 => {
            let z = base_logits.expect("checked above");
            let mut order: Vec<usize> = (0..root_actions.len()).collect();
            order.sort_by(|&a, &b| {
                z[b].total_cmp(&z[a])
                    .then(root_actions[a].cmp(&root_actions[b]))
            });
            for branch in order {
                if nodes.len() >= requested_q {
                    break;
                }
                let root_action = root_actions[branch].index() as u16;
                let child = acquire_edge(
                    &mut qm,
                    &mut metas,
                    &mut nodes,
                    0,
                    root_action,
                    branch,
                    None,
                    vec![root_action],
                    root_player,
                )?;
                let mut branch_count = 1usize;
                expand_dfs(
                    &mut qm,
                    &mut metas,
                    &mut nodes,
                    child,
                    branch,
                    &mut branch_count,
                    requested_q,
                    seed,
                    root_player,
                )?;
            }
        }
    }

    let exhausted_frontier = nodes.len() < requested_q;
    let mut graph = AcquiredGraph {
        schema: "v5_graph_manifest_v2".into(),
        config_digest: crate::config::V5Config::default().scientific_digest()?,
        episode,
        requested_q,
        actual_q: nodes.len(),
        exhausted_frontier,
        root_player: if root_player == Color::White {
            "white".into()
        } else {
            "black".into()
        },
        nodes,
        successful_queries: qm.state_transitions(),
        legal_generations: qm.legal_generations(),
        legal_moves_generated: qm.legal_moves_generated(),
        digest: String::new(),
    };
    graph.digest = graph.compute_digest()?;
    graph.verify()?;
    Ok(graph)
}

#[allow(clippy::too_many_arguments)]
fn expand_dfs(
    qm: &mut QueryManager,
    metas: &mut Vec<Option<Meta>>,
    nodes: &mut Vec<AcquiredNode>,
    parent: NodeId,
    branch: usize,
    branch_count: &mut usize,
    requested_q: usize,
    seed: u64,
    root_player: Color,
) -> anyhow::Result<()> {
    if nodes.len() >= requested_q
        || *branch_count >= MAX_BRANCH_EDGES
        || qm.packet(parent)?.ply_from_root >= u32::from(RANKED_MAX_DEPTH)
    {
        return Ok(());
    }
    let meta = metas[parent as usize].as_ref().expect("metadata").clone();
    let mut actions = qm.unqueried(parent)?;
    actions.sort_by_key(|&a| (hash_path(seed, &meta.path, a), a));
    for action in actions {
        if nodes.len() >= requested_q || *branch_count >= MAX_BRANCH_EDGES {
            break;
        }
        let mut path = meta.path.clone();
        path.push(action);
        let child = acquire_edge(
            qm,
            metas,
            nodes,
            parent,
            action,
            branch,
            Some(meta.storage),
            path,
            root_player,
        )?;
        *branch_count += 1;
        expand_dfs(
            qm,
            metas,
            nodes,
            child,
            branch,
            branch_count,
            requested_q,
            seed,
            root_player,
        )?;
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn depth_chain_fixture(depth: usize) -> (GameState, AcquiredGraph) {
    // Test-only real opening, acquired via the production StateQuery wrapper.
    let root =
        GameState::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
    let moves = [
        "e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1", "f8e7", "f1e1",
        "b7b5", "a4b3", "d7d6", "c2c3", "e8g8",
    ];
    assert!((1..=moves.len()).contains(&depth));
    let root_actions = root.legal_actions();
    let mut qm = QueryManager::new(root.clone())
        .unwrap()
        .with_budget(depth as u32);
    let mut metas = vec![None];
    let mut nodes = Vec::new();
    let mut parent = 0;
    let mut path = Vec::new();
    let mut branch = None;
    for mv in &moves[..depth] {
        let perspective = qm.state(parent).unwrap().perspective();
        let action = qm
            .unqueried(parent)
            .unwrap()
            .into_iter()
            .find(|index| {
                let (from, to, _) = ActionId::from_index(u32::from(*index))
                    .unwrap()
                    .to_physical(perspective);
                format!("{from}{to}").to_lowercase() == *mv
            })
            .expect("legal test opening move");
        let branch =
            *branch.get_or_insert_with(|| root_candidate_index(&root_actions, action).unwrap());
        path.push(action);
        let parent_storage = metas[parent as usize].as_ref().map(|m: &Meta| m.storage);
        parent = acquire_edge(
            &mut qm,
            &mut metas,
            &mut nodes,
            parent,
            action,
            branch,
            parent_storage,
            path.clone(),
            root.side_to_move(),
        )
        .unwrap();
    }
    let mut graph = AcquiredGraph {
        schema: "v5_graph_manifest_v2".into(),
        config_digest: crate::config::V5Config::default()
            .scientific_digest()
            .unwrap(),
        episode: EpisodeKey {
            position_id: "test-only-depth-chain".into(),
            schedule: Schedule::UniformFrontierV1,
            run_seed: 5301,
            occurrence_ordinal: 0,
        },
        requested_q: depth,
        actual_q: nodes.len(),
        exhausted_frontier: false,
        root_player: "white".into(),
        nodes,
        successful_queries: qm.state_transitions(),
        legal_generations: qm.legal_generations(),
        legal_moves_generated: qm.legal_moves_generated(),
        digest: String::new(),
    };
    graph.digest = graph.compute_digest().unwrap();
    graph.verify().unwrap();
    (root, graph)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn root() -> GameState {
        GameState::from_fen("6k1/8/8/8/8/8/4Q3/3RK3 w - - 0 1").unwrap()
    }

    fn episode(schedule: Schedule) -> EpisodeKey {
        EpisodeKey {
            position_id: "fixture".into(),
            schedule,
            run_seed: 5301,
            occurrence_ordinal: 7,
        }
    }

    #[test]
    fn uniform_is_reproducible_and_prefixes_are_nested() {
        let a = acquire(&root(), episode(Schedule::UniformFrontierV1), 8, None).unwrap();
        let b = acquire(&root(), episode(Schedule::UniformFrontierV1), 8, None).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.prefix(4).unwrap().nodes, a.nodes[..4]);
        assert_eq!(a.successful_queries, a.actual_q as u64);
    }

    #[test]
    fn ranked_uses_z0_but_not_a_label() {
        let root = root();
        let n = root.legal_actions().len();
        let mut z = vec![0.0; n];
        z[n - 1] = 2.0;
        let g = acquire(&root, episode(Schedule::BaseRankedDepthV1), 8, Some(&z)).unwrap();
        assert_eq!(g.nodes[0].root_candidate, n - 1);
        assert!(g.nodes.iter().all(|x| x.depth <= RANKED_MAX_DEPTH));
    }

    #[test]
    fn episode_identity_has_no_r_field() {
        let e = episode(Schedule::UniformFrontierV1);
        assert!(!serde_json::to_string(&e).unwrap().contains("\"r\""));
    }

    #[test]
    fn depth_six_through_sixteen_are_valid_and_out_of_contract_graphs_are_refused() {
        validate_uniform_frontier_contract().unwrap();
        for depth in 6..=MAX_ENGINEERING_Q {
            let (_, graph) = depth_chain_fixture(depth);
            assert_eq!(usize::from(graph.nodes.last().unwrap().depth), depth);
            assert_eq!(graph.successful_queries as usize, depth);
            graph.prefix(4).unwrap().verify().unwrap();
        }
        let (_, graph) = depth_chain_fixture(16);
        let check = |mut changed: AcquiredGraph| {
            changed.digest = changed.compute_digest().unwrap();
            assert!(changed.verify().is_err());
        };
        let mut changed = graph.clone();
        changed.nodes.last_mut().unwrap().depth = 17;
        check(changed);
        let mut changed = graph.clone();
        changed.requested_q = 15;
        check(changed);
        let mut changed = graph.clone();
        changed.nodes[6].parent = Some(0);
        check(changed);
        let mut changed = graph.clone();
        changed.episode.schedule = Schedule::BaseRankedDepthV1;
        check(changed);
        let mut changed = graph.clone();
        changed.schema = "v5_graph_manifest_v1".into();
        check(changed);
        let mut changed = graph;
        changed.config_digest =
            "0f1c31d5fb3873ecca356a83c413674442633bdd9e53e9f1744523058b4fb00c".into();
        check(changed);
    }

    #[test]
    fn returned_terminal_flags_come_from_the_exact_queried_child() {
        let root = GameState::from_fen("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1").unwrap();
        let actions = root.legal_actions();
        let perspective = root.perspective();
        let mating = actions
            .iter()
            .position(|action| {
                let (from, to, _) = action.to_physical(perspective);
                format!("{from}{to}").to_lowercase() == "a1a8"
            })
            .unwrap();
        let mut z0 = vec![0.0; actions.len()];
        z0[mating] = 1.0;
        let graph = acquire(&root, episode(Schedule::BaseRankedDepthV1), 1, Some(&z0)).unwrap();
        assert_eq!(graph.nodes[0].root_candidate, mating);
        assert_eq!(graph.nodes[0].payload.flags[1], 1.0);
        assert_eq!(graph.nodes[0].payload.flags[2], 1.0);
    }

    #[test]
    fn structure_digest_is_payload_independent_but_full_digest_is_not() {
        let graph = acquire(&root(), episode(Schedule::UniformFrontierV1), 4, None).unwrap();
        let mut changed = graph.clone();
        changed.nodes[0].payload.flags[0] = 1.0 - changed.nodes[0].payload.flags[0];
        changed.digest.clear();
        changed.digest = changed.compute_digest().unwrap();
        assert_ne!(graph.digest, changed.digest);
        assert_eq!(
            graph.compute_structure_digest().unwrap(),
            changed.compute_structure_digest().unwrap()
        );
    }

    #[test]
    fn uniform_frontier_includes_legal_edges_below_a_depth_five_acquired_node() {
        // Explicitly test-only real chess path. All five successor transitions
        // use the production StateQuery wrapper; no solver/labels are involved.
        let root = GameState::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            .unwrap();
        let actions = root.legal_actions();
        let mut qm = QueryManager::new(root.clone()).unwrap().with_budget(8);
        let mut metas = vec![None];
        let mut nodes = Vec::new();
        let mut parent = 0;
        let mut path = Vec::new();
        let first_action = qm.unqueried(parent).unwrap()[0];
        let branch = root_candidate_index(&actions, first_action).unwrap();
        for _ in 0..5 {
            let action = qm.unqueried(parent).unwrap()[0];
            path.push(action);
            let parent_storage = metas[parent as usize].as_ref().map(|m: &Meta| m.storage);
            parent = acquire_edge(
                &mut qm,
                &mut metas,
                &mut nodes,
                parent,
                action,
                branch,
                parent_storage,
                path.clone(),
                root.side_to_move(),
            )
            .unwrap();
        }
        let legal_descendants = qm.unqueried(parent).unwrap().len();
        assert!(
            legal_descendants > 0,
            "depth-five fixture is unexpectedly terminal"
        );
        let frontier = uniform_frontier(&qm, &metas, &actions).unwrap();
        let included = frontier.iter().filter(|edge| edge.1 == parent).count();
        println!(
            "V5_FRONTIER_REGRESSION {}",
            serde_json::json!({
                "successful_exact_queries": qm.state_transitions(), "available_query_budget": 3,
                "parent_depth": qm.packet(parent).unwrap().ply_from_root,
                "legal_unqueried_edges": legal_descendants, "included_by_uniform_selector": included,
                "path": path,
            })
        );
        assert_eq!(
            included, legal_descendants,
            "uniform_frontier_v1 must include every current unqueried legal edge; only ranked DFS has a depth-five limit"
        );
    }
}

#[cfg(test)]
mod artifact_tests {
    use super::*;
    #[test]
    fn provenance_envelope_refuses_stale_config_tamper_and_copied_metadata() {
        let root = GameState::from_fen("6k1/8/8/8/8/8/4Q3/3RK3 w - - 0 1").unwrap();
        let episode = EpisodeKey {
            position_id: "provenance".into(),
            schedule: Schedule::UniformFrontierV1,
            run_seed: 5301,
            occurrence_ordinal: 0,
        };
        let graph = acquire(&root, episode.clone(), 8, None).unwrap();
        let sha = "a".repeat(40);
        let a = GraphArtifact::new(graph.clone(), &sha).unwrap();
        a.verify(&sha).unwrap();
        assert!(a.verify(&"b".repeat(40)).is_err());
        assert_eq!(a.graph, graph); // envelope did not change scientific graph/hash/config
        let mut bad = a.clone();
        bad.config_digest = "bad".into();
        assert!(bad.verify(&sha).is_err());
        let mut bad = a.clone();
        bad.graph.nodes[0].payload.flags[0] += 1.0;
        assert!(bad.verify(&sha).is_err());
        let mut bad = a.clone();
        bad.graph = acquire(
            &root,
            EpisodeKey {
                occurrence_ordinal: 1,
                ..episode
            },
            8,
            None,
        )
        .unwrap();
        assert!(bad.verify(&sha).is_err());
        assert!(
            serde_json::from_value::<GraphArtifact>(serde_json::to_value(graph).unwrap()).is_err()
        );
    }
}
