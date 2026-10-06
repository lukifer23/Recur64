//! Label-free Q8 acquisition, typed input packets and exact frozen-base import.
use burn::tensor::backend::AutodiffBackend;
use recur64_core::{ActionId, GameState, Perspective, encode_root_relative_observation_v1};
use recur64_statequery::QueryManager;
use recur64_v5::graph::{AcquiredNode, ReturnedPayload};
use recur64_v5::{ACTION_GEOMETRY, PAYLOAD_FLAGS};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

pub const BASE_SOURCE: &str = "d11659eca0774e0064bed0ef64ead2b725886d93";
pub const BASE_MODEL: &str = "2d1c770a43a6455148b774e9ddb552b6ca33efd9fdd5d37593cefe7c8ae0bb00";
pub const BASE_OPTIM: &str = "cbef56e557f71a9205e34f65c782d9264cabd8fe960e5ab3915790a5f368f03d";
pub const BASE_RECIPE: &str = "6642579e1f2472bda955ca7ada5bb3b8a435634c023b665684da1e4676347e70";
pub const BASE_FP: &str = "12b272a941e5b29589195a65c779a60509626d2108975e4793674ad5867d75c9";
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Policy {
    ExploitTwo,
    BroadRankedHash,
}
impl Policy {
    pub fn all() -> [Self; 2] {
        [Self::ExploitTwo, Self::BroadRankedHash]
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Packet {
    pub schema: String,
    pub source_sha: String,
    pub config_digest: String,
    pub root_id: String,
    pub generation_role: String,
    pub policy: Policy,
    pub legal: Vec<u16>,
    pub nodes: Vec<AcquiredNode>,
    pub legal_counts: Vec<usize>,
    pub root_legal_count: usize,
    pub successful_queries: u64,
    pub legal_generations: u64,
    pub legal_moves_generated: u64,
    pub exhausted: bool,
    pub digest: String,
}
pub fn config() -> serde_json::Value {
    serde_json::json!({"architecture":crate::ARCHITECTURE,"width":256,"heads":8,"ffn":768,"encoder_blocks":2,"slots":4,"structure":48,"initialization":"all_owned_nodes_turn_pools_1024_768_256","refinement":"shared_local_simultaneous_1024_768_256","residual":0.1,"readout_bias":false,"auxiliary":"eligible_available_class_mean","aux_weight":0.5,"q":8,"r":[1,4],"seed":6300,"policies":"v6_ranked_two_hash_two_reply_v1","null":"payload_anchor_and_terminal_mask_zero","precision":"fp32"})
}
pub fn config_digest() -> anyhow::Result<String> {
    crate::digest(&config())
}
pub fn validate_predecessor(m: &recur64_v5::stage::CheckpointMeta) -> anyhow::Result<()> {
    use recur64_v5::stage::Stage;
    anyhow::ensure!(
        m.stage == Stage::BaselineA
            && m.recipe.stage == Stage::BaselineA
            && m.update == 1200
            && m.recipe.source_sha == BASE_SOURCE
            && m.model_hash == BASE_MODEL
            && m.optimizer_hash == BASE_OPTIM
            && m.recipe_digest == BASE_RECIPE
            && m.config_digest == recur64_v5::config::V5Config::default().scientific_digest()?
            && m.recipe.seed == 5301
            && m.backend == "cuda"
            && m.precision == "fp32"
            && m.recipe.precision == "fp32"
            && m.recipe.physical_microbatch == 2
            && m.recipe.data_contract == "v5_hp_data_v2"
            && m.recipe.train_identity == "V5_HP_TRAIN_V2"
            && m.recipe.dev_identity == "V5_HP_DEV_V2"
            && m.recipe.train_digest == recur64_v5::data::TRAIN_DIGEST
            && m.recipe.dev_digest == recur64_v5::data::DEV_DIGEST,
        "exact V6 frozen Stage A import capability refused"
    );
    Ok(())
}
pub fn import_base<B: AutodiffBackend>(
    path: &std::path::Path,
    device: &B::Device,
    backend: &str,
) -> anyhow::Result<crate::baseline::FrozenBase<B>> {
    let (m, meta) =
        recur64_v5::stage::load_finished_model(path, recur64_v5::stage::Stage::BaselineA, device)?;
    validate_predecessor(&meta)?;
    if backend == "cuda" {
        anyhow::ensure!(
            recur64_v5::stage::baseline_fingerprint(&m, device)? == BASE_FP,
            "frozen CUDA B0 fingerprint mismatch"
        );
    }
    crate::baseline::FrozenBase::from_legacy(m, device)
}
pub fn key(id: &str, path: &[u16], action: u16, seed: u64) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"v6_ranked_two_hash_two_reply_v1\0");
    h.update(seed.to_le_bytes());
    h.update(id.as_bytes());
    h.update([0]);
    for a in path {
        h.update(a.to_le_bytes());
    }
    h.update(action.to_le_bytes());
    h.finalize().into()
}
fn selected(legal: &[u16], logits: &[f32], id: &str, policy: Policy) -> anyhow::Result<Vec<usize>> {
    anyhow::ensure!(
        legal.len() == logits.len() && !legal.is_empty() && logits.iter().all(|v| v.is_finite()),
        "invalid acquisition B0/logit alignment"
    );
    let mut rank: Vec<usize> = (0..legal.len()).collect();
    rank.sort_by(|&a, &b| {
        logits[b]
            .total_cmp(&logits[a])
            .then(legal[a].cmp(&legal[b]))
    });
    let mut out = rank.into_iter().take(2).collect::<Vec<_>>();
    if policy == Policy::BroadRankedHash {
        let mut rest: Vec<_> = (0..legal.len()).filter(|a| !out.contains(a)).collect();
        rest.sort_by_key(|&a| (key(id, &[], legal[a], 0x7A60_E001), legal[a]));
        out.extend(rest.into_iter().take(2));
    }
    Ok(out)
}
pub fn acquire(
    root: &GameState,
    id: &str,
    logits: &[f32],
    policy: Policy,
    source: &str,
) -> anyhow::Result<Packet> {
    let legal: Vec<_> = root
        .legal_actions()
        .iter()
        .map(|a| a.index() as u16)
        .collect();
    // Dataset IDs contain family/depth annotations; hash chess state only.
    let selection_identity = root.to_fen();
    let initial = selected(&legal, logits, &selection_identity, policy)?;
    let mut qm = QueryManager::new(root.clone())?.with_budget(8);
    let player = root.side_to_move();
    let mut nodes: Vec<AcquiredNode> = Vec::new();
    let mut handles = Vec::new();
    let mut counts = Vec::new();
    let mut branches = Vec::new();
    let mut chosen = BTreeSet::new();
    let mut next_initial = 0;
    let mut cursor = 0;
    while nodes.len() < 8 {
        let edge = if next_initial < initial.len() {
            let a = initial[next_initial];
            next_initial += 1;
            branches.push(a);
            chosen.insert(a);
            Some((0, None, a, vec![legal[a]], legal[a]))
        } else {
            let mut pick = None;
            for _ in 0..branches.len() {
                let owner = branches[cursor % branches.len()];
                cursor += 1;
                let mut order: Vec<usize> = (0..nodes.len())
                    .filter(|&i| nodes[i].root_candidate == owner && nodes[i].depth < 16)
                    .collect();
                order.sort_by_key(|&i| (nodes[i].depth, nodes[i].path.clone()));
                for i in order {
                    let packet = qm.packet(handles[i])?;
                    let used: BTreeSet<_> = nodes
                        .iter()
                        .filter(|n| n.parent == Some(i as u32))
                        .map(|n| *n.path.last().unwrap())
                        .collect();
                    let mut actions: Vec<_> = packet
                        .legal_actions
                        .into_iter()
                        .filter(|a| !used.contains(a))
                        .collect();
                    actions.sort_by_key(|&a| {
                        (key(&selection_identity, &nodes[i].path, a, 0x7A60_E001), a)
                    });
                    if let Some(&a) = actions.first() {
                        let mut path = nodes[i].path.clone();
                        path.push(a);
                        pick = Some((handles[i], Some(i as u32), owner, path, a));
                        break;
                    }
                }
                if pick.is_some() {
                    break;
                }
            }
            if pick.is_none() {
                let mut rest: Vec<_> = (0..legal.len()).filter(|a| !chosen.contains(a)).collect();
                rest.sort_by_key(|&a| {
                    (
                        key(&selection_identity, &[], legal[a], 0x7A60_E001),
                        legal[a],
                    )
                });
                if let Some(&a) = rest.first() {
                    branches.push(a);
                    chosen.insert(a);
                    pick = Some((0, None, a, vec![legal[a]], legal[a]));
                }
            }
            pick
        };
        let Some((parent, parent_storage, owner, path, a)) = edge else {
            break;
        };
        let frame = qm.state(parent)?.perspective();
        let root_action = ActionId::from_index(a as u32)?.reframe(frame, Perspective::of(player));
        let p = qm.query(parent, a)?;
        let obs = encode_root_relative_observation_v1(qm.state(p.node_id)?, player);
        counts.push(p.legal_actions.len());
        handles.push(p.node_id);
        nodes.push(AcquiredNode {
            storage_id: nodes.len() as u32,
            parent: parent_storage,
            root_candidate: owner,
            incoming_action_root_frame: root_action.index() as u16,
            action_geometry: action_geometry(root_action),
            depth: p.ply_from_root as u8,
            root_to_move: qm.state(p.node_id)?.side_to_move() == player,
            path,
            cumulative_legal_generations: qm.legal_generations(),
            cumulative_legal_moves_generated: qm.legal_moves_generated(),
            payload: ReturnedPayload {
                observation: obs.as_slice().to_vec(),
                flags: flags(p.in_check, p.terminal, p.terminal_reason)?,
                state_digest: p.state_digest(),
                semantic_id: p.semantic_id,
            },
        });
    }
    let mut p = Packet {
        schema: "v6_evidence_packet_v2".into(),
        source_sha: source.into(),
        config_digest: config_digest()?,
        root_id: id.into(),
        generation_role: "p0_acquisition".into(),
        policy,
        root_legal_count: legal.len(),
        legal,
        successful_queries: nodes.len() as u64,
        exhausted: nodes.len() < 8,
        nodes,
        legal_counts: counts,
        legal_generations: qm.legal_generations(),
        legal_moves_generated: qm.legal_moves_generated(),
        digest: String::new(),
    };
    p.digest = p.content_digest()?;
    p.verify(source)?;
    Ok(p)
}
impl Packet {
    pub fn content_digest(&self) -> anyhow::Result<String> {
        let mut p = self.clone();
        p.digest.clear();
        crate::digest(&p)
    }
    pub fn verify(&self, source: &str) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.schema == "v6_evidence_packet_v2"
                && self.source_sha == source
                && source.len() == 40
                && self.config_digest == config_digest()?
                && self.digest == self.content_digest()?,
            "stale/tampered packet source/config/content"
        );
        anyhow::ensure!(
            !self.legal.is_empty()
                && self.legal.windows(2).all(|a| a[0] < a[1])
                && self.root_legal_count == self.legal.len()
                && self.nodes.len() <= 8
                && self.successful_queries == self.nodes.len() as u64
                && self.nodes.len() == self.legal_counts.len(),
            "packet accounting/legal alignment"
        );
        let mut paths = BTreeSet::new();
        for (i, n) in self.nodes.iter().enumerate() {
            anyhow::ensure!(
                n.storage_id == i as u32
                    && n.root_candidate < self.legal.len()
                    && !n.path.is_empty()
                    && n.path.len() == n.depth as usize
                    && n.path[0] == self.legal[n.root_candidate]
                    && paths.insert(n.path.clone())
                    && n.payload.observation.len() == 64 * 119
                    && n.payload
                        .observation
                        .iter()
                        .chain(n.payload.flags.iter())
                        .all(|x| x.is_finite()),
                "invalid node/owner/payload"
            );
            if let Some(parent) = n.parent {
                anyhow::ensure!(
                    (parent as usize) < i
                        && self.nodes[parent as usize].root_candidate == n.root_candidate
                        && self.nodes[parent as usize].path == n.path[..n.path.len() - 1],
                    "invalid parent/branch path"
                );
            } else {
                anyhow::ensure!(n.depth == 1, "missing node parent");
            }
            anyhow::ensure!(
                self.children(i).len() <= self.legal_counts[i],
                "unknown reply count underflow"
            );
        }
        Ok(())
    }
    pub fn verify_against_root(&self, root: &GameState) -> anyhow::Result<()> {
        self.verify(&self.source_sha)?;
        anyhow::ensure!(
            self.legal
                == root
                    .legal_actions()
                    .iter()
                    .map(|a| a.index() as u16)
                    .collect::<Vec<_>>(),
            "copied packet/root legal mismatch"
        );
        let mut qm = QueryManager::new(root.clone())?.with_budget(8);
        let mut handles = Vec::new();
        for (j, n) in self.nodes.iter().enumerate() {
            let parent = n.parent.map_or(0, |i| handles[i as usize]);
            let action = *n.path.last().unwrap();
            let frame = qm.state(parent)?.perspective();
            let expected_action = ActionId::from_index(action as u32)?
                .reframe(frame, Perspective::of(root.side_to_move()));
            let p = qm.query(parent, action)?;
            let state = qm.state(p.node_id)?;
            anyhow::ensure!(
                p.state_digest() == n.payload.state_digest
                    && p.semantic_id == n.payload.semantic_id
                    && encode_root_relative_observation_v1(state, root.side_to_move()).as_slice()
                        == n.payload.observation
                    && flags(p.in_check, p.terminal, p.terminal_reason)? == n.payload.flags
                    && n.root_to_move == (state.side_to_move() == root.side_to_move())
                    && p.legal_actions.len() == self.legal_counts[j]
                    && action_geometry(expected_action) == n.action_geometry,
                "packet payload/history/geometry/count mismatch"
            );
            handles.push(p.node_id);
        }
        Ok(())
    }
    pub fn children(&self, i: usize) -> Vec<usize> {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.parent == Some(i as u32))
            .map(|(j, _)| j)
            .collect()
    }
    pub fn unknown(&self, i: usize) -> usize {
        self.legal_counts[i] - self.children(i).len()
    }
    pub fn eligibility(&self) -> Vec<bool> {
        (0..self.legal.len())
            .map(|a| self.nodes.iter().any(|n| n.root_candidate == a))
            .collect()
    }
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

pub(crate) fn diagnostic_chain(root: GameState, source: &str) -> anyhow::Result<Packet> {
    let mut p = acquire(
        &root,
        "deep",
        &vec![0.; root.legal_actions().len()],
        Policy::ExploitTwo,
        source,
    )?;
    let mut qm = QueryManager::new(root.clone())?.with_budget(5);
    p.nodes.clear();
    p.legal_counts.clear();
    let mut parent = 0;
    let mut path = vec![];
    let mut owner = 0;
    for depth in 1..=5 {
        let a = *qm.packet(parent)?.legal_actions.first().ok_or_else(|| {
            anyhow::anyhow!("fixed deep qualification path terminated before depth5")
        })?;
        if depth == 1 {
            owner = p
                .legal
                .iter()
                .position(|v| *v == a)
                .ok_or_else(|| anyhow::anyhow!("root action missing"))?;
        }
        let frame = qm.state(parent)?.perspective();
        let action =
            ActionId::from_index(a as u32)?.reframe(frame, Perspective::of(root.side_to_move()));
        let v = qm.query(parent, a)?;
        path.push(a);
        p.legal_counts.push(v.legal_actions.len());
        p.nodes.push(AcquiredNode {
            storage_id: depth - 1,
            parent: if depth == 1 { None } else { Some(depth - 2) },
            root_candidate: owner,
            incoming_action_root_frame: action.index() as u16,
            action_geometry: action_geometry(action),
            depth: depth as u8,
            root_to_move: qm.state(v.node_id)?.side_to_move() == root.side_to_move(),
            path: path.clone(),
            cumulative_legal_generations: qm.legal_generations(),
            cumulative_legal_moves_generated: qm.legal_moves_generated(),
            payload: ReturnedPayload {
                observation: encode_root_relative_observation_v1(
                    qm.state(v.node_id)?,
                    root.side_to_move(),
                )
                .as_slice()
                .to_vec(),
                flags: flags(v.in_check, v.terminal, v.terminal_reason)?,
                state_digest: v.state_digest(),
                semantic_id: v.semantic_id,
            },
        });
        parent = v.node_id;
    }
    p.successful_queries = 5;
    p.exhausted = false;
    p.digest = p.content_digest()?;
    p.generation_role = "qualification_chain".into();
    p.digest = p.content_digest()?;
    p.verify(source)?;
    Ok(p)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ranked_hash_selection_is_label_free_and_unique() {
        let l = [1, 2, 3, 4, 5, 6];
        let z = [6., 5., 4., 3., 2., 1.];
        let a = selected(&l, &z, "id", Policy::BroadRankedHash).unwrap();
        assert_eq!(&a[..2], &[0, 1]);
        assert_eq!(a.len(), 4);
        assert_eq!(a.iter().collect::<BTreeSet<_>>().len(), 4);
        assert_eq!(a, selected(&l, &z, "id", Policy::BroadRankedHash).unwrap());
        assert_eq!(
            selected(&[1], &[0.], "id", Policy::BroadRankedHash).unwrap(),
            vec![0]
        );
    }
    #[test]
    fn packet_has_no_label_fields_and_refuses_tampering() {
        let r = GameState::from_fen("6k1/8/8/8/8/8/4Q3/3RK3 w - - 0 1").unwrap();
        let l = r.legal_actions().len();
        let p = acquire(
            &r,
            "test",
            &vec![0.; l],
            Policy::ExploitTwo,
            &"a".repeat(40),
        )
        .unwrap();
        let renamed = acquire(
            &r,
            "changed-family-and-depth-label",
            &vec![0.; l],
            Policy::ExploitTwo,
            &"a".repeat(40),
        )
        .unwrap();
        assert_eq!(
            p.nodes, renamed.nodes,
            "acquisition must ignore annotated IDs"
        );
        let broad = acquire(
            &r,
            "original-label",
            &vec![0.; l],
            Policy::BroadRankedHash,
            &"a".repeat(40),
        )
        .unwrap();
        let broad_renamed = acquire(
            &r,
            "changed-label",
            &vec![0.; l],
            Policy::BroadRankedHash,
            &"a".repeat(40),
        )
        .unwrap();
        assert_eq!(broad.nodes, broad_renamed.nodes);
        assert_eq!(p.nodes.len(), 8);
        let text = serde_json::to_string(&p).unwrap();
        for field in ["correct", "mate_depth", "family", "solver"] {
            assert!(!text.contains(&format!("\"{field}\":")));
        }
        let mut q = p.clone();
        q.legal_counts[0] += 1;
        assert!(q.verify(&"a".repeat(40)).is_err());
        assert!(p.verify(&"b".repeat(40)).is_err());
        assert!(p.unknown(0) > 0);
    }
}

#[cfg(test)]
mod predecessor_tests {
    use super::*;
    #[test]
    fn exact_import_and_each_mutation_refuse() {
        use recur64_v5::stage::{CheckpointMeta, Recipe, Stage};
        let r = Recipe::stage_a(BASE_SOURCE.into(), 2).unwrap();
        let m = CheckpointMeta {
            schema: recur64_v5::stage::CHECKPOINT_SCHEMA.into(),
            architecture: recur64_v5::config::ARCHITECTURE.into(),
            stage: Stage::BaselineA,
            config_digest: r.config_digest.clone(),
            recipe: r,
            recipe_digest: BASE_RECIPE.into(),
            update: 1200,
            model_hash: BASE_MODEL.into(),
            optimizer_hash: BASE_OPTIM.into(),
            init_model_hash: None,
            backend: "cuda".into(),
            precision: "fp32".into(),
            factual_null_gradient_semantics: "unused by import".into(),
            history: vec![],
            resume_events: vec![],
        };
        validate_predecessor(&m).unwrap();
        let original = serde_json::to_value(&m).unwrap();
        for path in [
            "model_hash",
            "optimizer_hash",
            "recipe_digest",
            "config_digest",
            "backend",
            "precision",
        ] {
            let mut x = original.clone();
            x[path] = serde_json::json!("wrong");
            assert!(
                validate_predecessor(&serde_json::from_value(x).unwrap()).is_err(),
                "{path}"
            );
        }
        for path in [
            "source_sha",
            "data_contract",
            "train_identity",
            "dev_identity",
            "train_digest",
            "dev_digest",
        ] {
            let mut x = original.clone();
            x["recipe"][path] = serde_json::json!("wrong");
            assert!(
                validate_predecessor(&serde_json::from_value(x).unwrap()).is_err(),
                "{path}"
            );
        }
        for (path, v) in [("seed", 6300), ("physical_microbatch", 1)] {
            let mut x = original.clone();
            x["recipe"][path] = serde_json::json!(v);
            assert!(validate_predecessor(&serde_json::from_value(x).unwrap()).is_err());
        }
        let mut x = original;
        x["update"] = serde_json::json!(1199);
        assert!(validate_predecessor(&serde_json::from_value(x).unwrap()).is_err());
    }
}
