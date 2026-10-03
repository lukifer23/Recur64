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
pub const MAX_DEPTH: u8 = 5;
pub const MAX_BRANCH_EDGES: usize = 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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

    pub fn verify(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.schema == "v5_graph_manifest_v1",
            "graph schema mismatch"
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
            self.nodes
                .iter()
                .all(|n| n.depth > 0 && n.depth <= MAX_DEPTH),
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
            if let Some(p) = n.parent {
                anyhow::ensure!((p as usize) < self.nodes.len(), "parent id is absent");
                anyhow::ensure!(p != n.storage_id, "node cannot parent itself");
            }
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
                let mut frontier = Vec::new();
                for qid in 0..qm.node_count() as NodeId {
                    if qm.packet(qid)?.ply_from_root >= u32::from(MAX_DEPTH) {
                        continue;
                    }
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
                            root_candidate_index(&root_actions, action)?
                        } else {
                            root_candidate
                        };
                        let mut child_path = path.clone();
                        child_path.push(action);
                        frontier.push((child_path, qid, action, branch, parent_storage));
                    }
                }
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
        schema: "v5_graph_manifest_v1".into(),
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
        || qm.packet(parent)?.ply_from_root >= u32::from(MAX_DEPTH)
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
        assert!(g.nodes.iter().all(|x| x.depth <= MAX_DEPTH));
    }

    #[test]
    fn episode_identity_has_no_r_field() {
        let e = episode(Schedule::UniformFrontierV1);
        assert!(!serde_json::to_string(&e).unwrap().contains("\"r\""));
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
}
