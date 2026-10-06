//! Frozen label-free donor mappings and tensor interventions.
//!
//! These are tensor controls, not guaranteed legal chess histories. Donor choice
//! never reads correctness, predictions or losses; the recipient/donor hash keys
//! are the original V6 P0 keys. Only `payload.observation` frames (successor-only)
//! or whole payloads (complete shuffle) change; recipient structure is preserved.
use crate::{
    p0::{Entry, Plan},
    packet::{Packet, key},
};
use anyhow::{Result, ensure};
use recur64_v5::graph::AcquiredNode;
use serde::{Deserialize, Serialize};

pub const SEED_E002: u64 = 0x7A60_E002;
pub const SEED_E402: u64 = 0x7A60_E402;
pub const SEED_E502: u64 = 0x7A60_E502;
const SQUARES: usize = 64;
const CHANNELS: usize = 119;
const FRAME: usize = 14;
const FRAMES: usize = 8;
const FEATURES_AT: usize = FRAMES * FRAME; // 112..119 are non-history features

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct DonorRow {
    pub recipient_id: String,
    pub policy: usize,
    pub node: usize,
    pub recipient_path: Vec<u16>,
    pub recipient_depth: u8,
    pub turn: bool,
    pub donor_entry: usize,
    pub donor_id: String,
    pub donor_node: usize,
    pub donor_path: Vec<u16>,
    pub donor_depth: u8,
    pub depth_delta: u8,
    pub pool: usize,
}

/// `rows[entry][policy][node]`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct DonorMap {
    pub seed: u64,
    pub rows: Vec<Vec<Vec<DonorRow>>>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Maps {
    pub e002: DonorMap,
    pub e402: DonorMap,
    pub e502: DonorMap,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Condition {
    Real,
    ShuffleCompleteE002,
    AllNull,
    BoardFlagsZero,
    SuccessorFramesZero,
    SuccessorOnlyE002,
    SuccessorOnlyE402,
    SuccessorOnlyE502,
}
impl Condition {
    pub fn all() -> [Condition; 8] {
        use Condition::*;
        [
            Real,
            ShuffleCompleteE002,
            AllNull,
            BoardFlagsZero,
            SuccessorFramesZero,
            SuccessorOnlyE002,
            SuccessorOnlyE402,
            SuccessorOnlyE502,
        ]
    }
    pub fn name(self) -> &'static str {
        use Condition::*;
        match self {
            Real => "real",
            ShuffleCompleteE002 => "shuffle_complete_e002",
            AllNull => "all_null",
            BoardFlagsZero => "board_flags_zero",
            SuccessorFramesZero => "successor_frames_zero",
            SuccessorOnlyE002 => "successor_only_e002",
            SuccessorOnlyE402 => "successor_only_e402",
            SuccessorOnlyE502 => "successor_only_e502",
        }
    }
    pub fn map_seed(self) -> Option<u64> {
        use Condition::*;
        match self {
            ShuffleCompleteE002 | SuccessorOnlyE002 => Some(SEED_E002),
            SuccessorOnlyE402 => Some(SEED_E402),
            SuccessorOnlyE502 => Some(SEED_E502),
            _ => None,
        }
    }
    pub fn is_successor_only(self) -> bool {
        matches!(
            self,
            Condition::SuccessorOnlyE002
                | Condition::SuccessorOnlyE402
                | Condition::SuccessorOnlyE502
        )
    }
}
impl Maps {
    pub fn for_seed(&self, seed: u64) -> &DonorMap {
        match seed {
            SEED_E002 => &self.e002,
            SEED_E402 => &self.e402,
            _ => &self.e502,
        }
    }
}

fn row_candidates<'a>(
    plan: &'a Plan,
    e: &Entry,
    policy: usize,
    n: &AcquiredNode,
) -> Vec<(usize, &'a Entry, usize, &'a AcquiredNode)> {
    plan.entries
        .iter()
        .enumerate()
        .filter(|(_, d)| d.id != e.id && d.cell == e.cell)
        .flat_map(|(di, d)| {
            d.packets[policy]
                .nodes
                .iter()
                .enumerate()
                .filter(move |(_, v)| v.root_to_move == n.root_to_move)
                .map(move |(vi, v)| (di, d, vi, v))
        })
        .collect()
}

/// Original P0 matching/order rules with an explicit seed. Unresolved donor => Err.
pub fn donor_map(plan: &Plan, seed: u64) -> Result<DonorMap> {
    let mut rows = Vec::new();
    for e in &plan.entries {
        let mut per_policy = Vec::new();
        for policy in 0..2 {
            let mut per_node = Vec::new();
            for (ni, n) in e.packets[policy].nodes.iter().enumerate() {
                let mut donors = row_candidates(plan, e, policy, n);
                ensure!(
                    !donors.is_empty(),
                    "unresolved same-turn different-root donor for {}",
                    e.id
                );
                let distance = donors
                    .iter()
                    .map(|(_, _, _, v)| v.depth.abs_diff(n.depth))
                    .min()
                    .unwrap();
                donors.retain(|(_, _, _, v)| v.depth.abs_diff(n.depth) == distance);
                donors.sort_by_key(|(_, d, _, v)| {
                    (d.id.clone(), v.depth, v.path.clone(), v.storage_id)
                });
                let h = key(&e.id, &n.path, n.depth as u16, seed);
                let at = u64::from_le_bytes(h[..8].try_into().unwrap()) as usize % donors.len();
                let (di, d, vi, v) = donors[at];
                ensure!(d.id != e.id, "self donor");
                per_node.push(DonorRow {
                    recipient_id: e.id.clone(),
                    policy,
                    node: ni,
                    recipient_path: n.path.clone(),
                    recipient_depth: n.depth,
                    turn: n.root_to_move,
                    donor_entry: di,
                    donor_id: d.id.clone(),
                    donor_node: vi,
                    donor_path: v.path.clone(),
                    donor_depth: v.depth,
                    depth_delta: distance,
                    pool: donors.len(),
                });
            }
            per_policy.push(per_node);
        }
        rows.push(per_policy);
    }
    Ok(DonorMap { seed, rows })
}
pub fn all_maps(plan: &Plan) -> Result<Maps> {
    Ok(Maps {
        e002: donor_map(plan, SEED_E002)?,
        e402: donor_map(plan, SEED_E402)?,
        e502: donor_map(plan, SEED_E502)?,
    })
}

fn donor_node<'a>(plan: &'a Plan, r: &DonorRow, policy: usize) -> &'a AcquiredNode {
    &plan.entries[r.donor_entry].packets[policy].nodes[r.donor_node]
}

/// Successor-only replacement: frames `k < min(recipient depth, 8)` take donor
/// frame `k` when `k < donor depth`, else a zero 14-channel frame. Everything
/// else, including features 112..118 and flags, is the recipient's.
fn successor_only(n: &mut AcquiredNode, donor: &AcquiredNode) {
    let recipient_depth = (n.depth as usize).min(FRAMES);
    let donor_depth = donor.depth as usize;
    for square in 0..SQUARES {
        for k in 0..recipient_depth {
            let at = square * CHANNELS + k * FRAME;
            if k < donor_depth {
                n.payload.observation[at..at + FRAME]
                    .copy_from_slice(&donor.payload.observation[at..at + FRAME]);
            } else {
                n.payload.observation[at..at + FRAME].fill(0.);
            }
        }
    }
}

/// The condition's packet for `plan.entries[entry]`/`policy`. The digest is
/// recomputed so the typed packet is self-consistent; structure is verified
/// separately by [`verify_intervention`].
pub fn intervened(
    plan: &Plan,
    maps: &Maps,
    entry: usize,
    policy: usize,
    condition: Condition,
) -> Result<Packet> {
    let mut p = plan.entries[entry].packets[policy].clone();
    match condition {
        Condition::Real | Condition::AllNull => {}
        Condition::ShuffleCompleteE002 => {
            for (ni, n) in p.nodes.iter_mut().enumerate() {
                let r = &maps.e002.rows[entry][policy][ni];
                n.payload = donor_node(plan, r, policy).payload.clone();
            }
        }
        Condition::SuccessorOnlyE002
        | Condition::SuccessorOnlyE402
        | Condition::SuccessorOnlyE502 => {
            let map = maps.for_seed(condition.map_seed().unwrap());
            for (ni, n) in p.nodes.iter_mut().enumerate() {
                let r = &map.rows[entry][policy][ni];
                successor_only(n, donor_node(plan, r, policy));
            }
        }
        Condition::BoardFlagsZero => {
            for n in &mut p.nodes {
                n.payload.observation.fill(0.);
                n.payload.flags.fill(0.);
            }
        }
        Condition::SuccessorFramesZero => {
            for n in &mut p.nodes {
                for square in 0..SQUARES {
                    for k in 0..(n.depth as usize).min(FRAMES) {
                        let at = square * CHANNELS + k * FRAME;
                        n.payload.observation[at..at + FRAME].fill(0.);
                    }
                }
            }
        }
    }
    p.digest = p.content_digest()?;
    Ok(p)
}

fn structure(p: &Packet) -> serde_json::Value {
    let mut v = serde_json::to_value(p).unwrap();
    v.as_object_mut().unwrap().remove("digest");
    for n in v["nodes"].as_array_mut().unwrap() {
        n.as_object_mut().unwrap().remove("payload");
    }
    v
}

/// Independent channel-preservation checker. It re-derives the expected tensor
/// channel by channel (rather than reusing the frame loops above) and returns the
/// number of (node, square, channel) cells checked. Any mismatch is an Err.
pub fn verify_intervention(
    plan: &Plan,
    maps: &Maps,
    entry: usize,
    policy: usize,
    condition: Condition,
    changed: &Packet,
) -> Result<usize> {
    let real = &plan.entries[entry].packets[policy];
    ensure!(
        structure(real) == structure(changed),
        "intervention changed packet structure ({})",
        condition.name()
    );
    ensure!(
        changed.digest == changed.content_digest()?,
        "intervention digest stale"
    );
    let mut checked = 0;
    for (ni, (old, new)) in real.nodes.iter().zip(&changed.nodes).enumerate() {
        ensure!(
            new.payload.observation.len() == SQUARES * CHANNELS,
            "observation size"
        );
        let donor = if condition.map_seed().is_some() {
            let r = &maps.for_seed(condition.map_seed().unwrap()).rows[entry][policy][ni];
            ensure!(
                r.donor_id != real.root_id
                    && r.turn == old.root_to_move
                    && r.recipient_depth == old.depth
                    && donor_node(plan, r, policy).root_to_move == old.root_to_move,
                "donor row inconsistent with recipient"
            );
            Some(donor_node(plan, r, policy))
        } else {
            None
        };
        if condition.is_successor_only() {
            let donor = donor.unwrap();
            ensure!(
                new.payload.flags == old.payload.flags
                    && new.payload.state_digest == old.payload.state_digest
                    && new.payload.semantic_id == old.payload.semantic_id,
                "successor-only changed flags/identity"
            );
            for square in 0..SQUARES {
                for c in 0..CHANNELS {
                    let got = new.payload.observation[square * CHANNELS + c];
                    let was = old.payload.observation[square * CHANNELS + c];
                    let expected = if c >= FEATURES_AT {
                        was // observation features 112..118 retained
                    } else {
                        let frame = c / FRAME;
                        if frame >= old.depth as usize {
                            was // recipient root / pre-root frames retained
                        } else if frame < donor.depth as usize {
                            donor.payload.observation[square * CHANNELS + c]
                        } else {
                            0. // zero frame including validity channel
                        }
                    };
                    ensure!(
                        got.to_bits() == expected.to_bits(),
                        "successor-only channel mismatch node {ni} square {square} channel {c}"
                    );
                    checked += 1;
                }
            }
        } else if condition == Condition::ShuffleCompleteE002 {
            ensure!(
                new.payload == donor.unwrap().payload,
                "complete shuffle is not the donor payload"
            );
            checked += SQUARES * CHANNELS;
        } else if condition == Condition::SuccessorFramesZero {
            ensure!(new.payload.flags == old.payload.flags, "flags changed");
            for square in 0..SQUARES {
                for c in 0..CHANNELS {
                    let was = old.payload.observation[square * CHANNELS + c];
                    let removed = c < FEATURES_AT && c / FRAME < old.depth as usize;
                    let expected = if removed { 0. } else { was };
                    ensure!(
                        new.payload.observation[square * CHANNELS + c].to_bits()
                            == expected.to_bits(),
                        "successor-zero channel mismatch"
                    );
                    checked += 1;
                }
            }
        } else if condition == Condition::BoardFlagsZero {
            ensure!(
                new.payload.observation.iter().all(|v| *v == 0.)
                    && new.payload.flags.iter().all(|v| *v == 0.),
                "board+flags not zero"
            );
            checked += SQUARES * CHANNELS;
        } else {
            ensure!(new.payload == old.payload, "unchanged condition changed");
            checked += SQUARES * CHANNELS;
        }
    }
    Ok(checked)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packet::{Policy, acquire};
    use recur64_core::GameState;
    use recur64_v5::graph::ReturnedPayload;

    fn plan_of(fens: &[&str]) -> Plan {
        let source = "a".repeat(40);
        let entries = fens
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let root = GameState::from_fen(f).unwrap();
                let w = root.legal_actions().len();
                let packets = Policy::all()
                    .iter()
                    .map(|p| {
                        acquire(&root, &format!("root-{i}"), &vec![0.; w], *p, &source).unwrap()
                    })
                    .collect();
                Entry {
                    index: i,
                    id: format!("root-{i}"),
                    cell: "same-cell".into(),
                    baseline_correct: i % 2 == 0,
                    packets,
                }
            })
            .collect();
        Plan {
            schema: "test".into(),
            source_sha: source,
            config_digest: String::new(),
            train_digest: String::new(),
            seed: 0,
            updates: 0,
            entries,
            strata_support: Default::default(),
            initial_shapes_match: true,
            episode_digest: String::new(),
            digest: String::new(),
        }
    }
    const FENS: [&str; 3] = [
        "7k/8/8/8/8/8/3Q4/K1R5 w - - 0 1",
        "6k1/8/8/8/8/8/4Q3/3RK3 w - - 0 1",
        "6k1/8/8/8/8/2Q5/8/K2R4 w - - 0 1",
    ];

    #[test]
    fn mappings_are_deterministic_label_free_and_stop_without_a_donor() {
        let plan = plan_of(&FENS);
        let a = all_maps(&plan).unwrap();
        assert_eq!(a, all_maps(&plan).unwrap());
        assert_ne!(a.e002, a.e402);
        // Correctness bookkeeping is not an input to donor choice.
        let mut flipped = plan.clone();
        for e in &mut flipped.entries {
            e.baseline_correct = !e.baseline_correct;
        }
        assert_eq!(a, all_maps(&flipped).unwrap());
        for per_entry in &a.e002.rows {
            for rows in per_entry {
                for r in rows {
                    assert_ne!(r.donor_id, r.recipient_id);
                }
            }
        }
        let mut lone = plan.clone();
        lone.entries.truncate(1);
        assert!(donor_map(&lone, SEED_E002).is_err());
    }

    #[test]
    fn every_condition_preserves_structure_and_exact_channels() {
        let plan = plan_of(&FENS);
        let maps = all_maps(&plan).unwrap();
        for entry in 0..plan.entries.len() {
            for policy in 0..2 {
                for c in Condition::all() {
                    let p = intervened(&plan, &maps, entry, policy, c).unwrap();
                    let checked = verify_intervention(&plan, &maps, entry, policy, c, &p).unwrap();
                    assert!(checked > 0, "{}", c.name());
                }
                // Successor-only really changes something relative to real.
                let p =
                    intervened(&plan, &maps, entry, policy, Condition::SuccessorOnlyE402).unwrap();
                assert!(
                    p.nodes
                        .iter()
                        .zip(&plan.entries[entry].packets[policy].nodes)
                        .any(|(a, b)| a.payload.observation != b.payload.observation)
                );
            }
        }
    }

    #[test]
    fn successor_only_zero_pads_when_donor_is_shallower_and_keeps_root_frames() {
        let plan = plan_of(&FENS);
        let mut recipient = plan.entries[0].packets[0].nodes[0].clone();
        let mut donor = plan.entries[1].packets[0].nodes[0].clone();
        recipient.depth = 5;
        donor.depth = 2;
        recipient.payload.observation = vec![1.; SQUARES * CHANNELS];
        donor.payload.observation = vec![2.; SQUARES * CHANNELS];
        recipient.payload.flags = [1.; 9];
        let before = recipient.payload.clone();
        successor_only(&mut recipient, &donor);
        for square in 0..SQUARES {
            for c in 0..CHANNELS {
                let v = recipient.payload.observation[square * CHANNELS + c];
                let want = if c >= FEATURES_AT {
                    1.
                } else {
                    match c / FRAME {
                        0..=1 => 2., // k < donor depth: copied
                        2..=4 => 0., // donor shallower: zero frame incl. validity
                        _ => 1.,     // root / pre-root retained
                    }
                };
                assert_eq!(v, want, "square {square} channel {c}");
            }
        }
        assert_eq!(recipient.payload.flags, before.flags);
        // Depth beyond the 8-frame window replaces exactly 8 frames, never panics.
        recipient.depth = 12;
        recipient.payload = ReturnedPayload {
            observation: vec![1.; SQUARES * CHANNELS],
            ..before
        };
        donor.depth = 12;
        successor_only(&mut recipient, &donor);
        assert!(
            recipient.payload.observation[..FEATURES_AT]
                .iter()
                .all(|v| *v == 2.)
                && recipient.payload.observation[FEATURES_AT..CHANNELS]
                    .iter()
                    .all(|v| *v == 1.)
        );
    }

    #[test]
    fn checker_rejects_corrupted_interventions() {
        let plan = plan_of(&FENS);
        let maps = all_maps(&plan).unwrap();
        let c = Condition::SuccessorOnlyE002;
        let mut p = intervened(&plan, &maps, 0, 0, c).unwrap();
        p.nodes[0].payload.observation[FEATURES_AT] += 1.;
        p.digest = p.content_digest().unwrap();
        assert!(verify_intervention(&plan, &maps, 0, 0, c, &p).is_err());
        let mut p = intervened(&plan, &maps, 0, 0, c).unwrap();
        p.nodes[0].payload.flags[0] = 1. - p.nodes[0].payload.flags[0];
        p.digest = p.content_digest().unwrap();
        assert!(verify_intervention(&plan, &maps, 0, 0, c, &p).is_err());
        let mut p = intervened(&plan, &maps, 0, 0, c).unwrap();
        p.nodes[0].depth += 1;
        p.digest = p.content_digest().unwrap();
        assert!(verify_intervention(&plan, &maps, 0, 0, c, &p).is_err());
    }
}
