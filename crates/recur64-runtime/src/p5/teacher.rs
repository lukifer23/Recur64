//! `proof_teacher_seeded_v1`: the P5 training teacher.
//!
//! It is a separate, versioned schedule; the generic
//! [`crate::proof::trace_teacher::ProofTraceTeacher`] stays the reference for the
//! full `A(S)`.
//!
//! * **Before completion.** The selector target is the UNIFORM distribution over the
//!   complete tied set `A_proof(S)`. The query that is actually made is one
//!   admissible PROOF edge chosen by a deterministic seeded-uniform draw over the
//!   tied set (sorted by action path, so the choice does not depend on frontier slot
//!   numbering), never the lexicographically first and never model-dependent. While
//!   the proof is incomplete there must be at least one `A_proof` edge, and since the
//!   teacher never enters an incorrect root branch, `A_refute` must be empty.
//! * **Completion latch.** The first time the proof residual is 0 the example latches
//!   complete for the rest of the episode. From then on there is NO selector target,
//!   whatever `A(S)` would say, and the remaining forced budget is spent with the
//!   frozen `fixed_bfs_actionid_v1` order. A filler query that opens an incorrect
//!   root branch therefore cannot reactivate any process loss.
//!
//! The follow choice is a pure function of (run seed, position id, the example's
//! ordinal in its budget stream, the budget, the step) through the project's
//! deterministic `mix` and a fixed FNV hash; it never touches a process-randomised
//! hasher.

use std::collections::{BTreeMap, HashSet};

use recur64_core::GameState;
use recur64_model::active::{EdgeRef, QueryScript, ScriptStep, Tree};
use recur64_statequery::QueryManager;

use crate::proof::generator::mix;
use crate::proof::trace::{Path, PositionTrace};
use crate::proof::trace_teacher::{edge_path, node_paths};

/// FNV-1a over bytes (the same constants as the sampler's cell tag).
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = 0xCBF2_9CE4_8422_2325u64;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

/// The tie-choice key of one example occurrence.
pub fn follow_key(base: u64, position_id: &str, ordinal: u64, budget: usize) -> u64 {
    mix(base ^ mix(fnv1a(position_id.as_bytes()) ^ mix(ordinal ^ mix(budget as u64 + 0xB0D6E7))))
}

/// Per-example record of what the teacher did.
#[derive(Debug, Clone, Default)]
pub struct EpisodeRecord {
    /// Steps at which this example had a (non-empty) selector target.
    pub supervised: usize,
    /// The step at which the proof was first found complete, if ever.
    pub completed_at: Option<usize>,
    /// Queries made while latched complete.
    pub filler_queries: usize,
}

pub struct SeededProofTeacher<'a> {
    traces: Vec<&'a PositionTrace>,
    keys: Vec<u64>,
    latched: Vec<bool>,
    pub records: Vec<EpisodeRecord>,
}

impl<'a> SeededProofTeacher<'a> {
    pub fn new(traces: Vec<&'a PositionTrace>, keys: Vec<u64>) -> Self {
        assert_eq!(traces.len(), keys.len());
        let n = traces.len();
        Self {
            traces,
            keys,
            latched: vec![false; n],
            records: vec![EpisodeRecord::default(); n],
        }
    }

    pub fn is_latched(&self, example: usize) -> bool {
        self.latched[example]
    }
}

/// The frozen filler: the minimum of `(parent depth, parent slot, ActionId)`.
fn bfs_first(frontier: &[EdgeRef]) -> usize {
    frontier
        .iter()
        .enumerate()
        .min_by_key(|(_, e)| e.bfs_key())
        .map(|(i, _)| i)
        .expect("a non-empty frontier")
}

impl QueryScript for SeededProofTeacher<'_> {
    fn next(
        &mut self,
        example: usize,
        step: usize,
        frontier: &[EdgeRef],
        tree: &Tree,
    ) -> anyhow::Result<ScriptStep> {
        let trace = *self
            .traces
            .get(example)
            .ok_or_else(|| anyhow::anyhow!("no proof trace for example {example}"))?;
        if self.latched[example] {
            self.records[example].filler_queries += 1;
            return Ok(ScriptStep {
                follow: bfs_first(frontier),
                targets: Vec::new(),
            });
        }
        let paths = node_paths(tree);
        let s: HashSet<Path> = paths.iter().filter(|p| !p.is_empty()).cloned().collect();
        if trace.is_complete(&s) {
            self.latched[example] = true;
            self.records[example].completed_at = Some(step);
            self.records[example].filler_queries += 1;
            return Ok(ScriptStep {
                follow: bfs_first(frontier),
                targets: Vec::new(),
            });
        }
        let adm = trace.admissible(&s);
        anyhow::ensure!(
            !adm.proof.is_empty(),
            "{}: an incomplete proof has no admissible proof edge",
            trace.id
        );
        anyhow::ensure!(
            adm.refute.is_empty(),
            "{}: the teacher queried outside the proof before completion (refutation edges are admissible)",
            trace.id
        );
        // Map every admissible path onto the live frontier, keyed by path.
        let mut by_path: BTreeMap<Path, usize> = BTreeMap::new();
        for (i, e) in frontier.iter().enumerate() {
            let p = edge_path(&paths, e);
            if adm.proof.contains(&p) {
                by_path.insert(p, i);
            }
        }
        anyhow::ensure!(
            by_path.len() == adm.proof.len(),
            "{}: {} admissible proof edges are not on the live frontier",
            trace.id,
            adm.proof.len() - by_path.len()
        );
        let sorted: Vec<usize> = by_path.values().copied().collect(); // sorted by path
        let pick = (mix(self.keys[example] ^ mix(step as u64)) % sorted.len() as u64) as usize;
        let follow = sorted[pick];
        let mut targets: Vec<usize> = by_path.into_values().collect();
        targets.sort_unstable();
        self.records[example].supervised += 1;
        Ok(ScriptStep { follow, targets })
    }
}

/// Run one teacher episode WITHOUT the model: the exact tree semantics of
/// `ActiveSearchModel::run` (same `QueryManager`, `Tree`, frontier order), so the
/// number of supervised decisions of a whole optimizer update is known before any
/// gradient is computed.
pub fn simulate_episode(
    trace: &PositionTrace,
    root: &GameState,
    budget: usize,
    key: u64,
) -> anyhow::Result<EpisodeRecord> {
    let mut mgr = QueryManager::new(root.clone())?;
    let mut tree = Tree::new(&mgr.packet(0)?)?;
    let mut teacher = SeededProofTeacher::new(vec![trace], vec![key]);
    for step in 0..budget {
        let frontier = tree.frontier();
        if frontier.is_empty() {
            break;
        }
        let s = teacher.next(0, step, &frontier, &tree)?;
        let e = frontier[s.follow].clone();
        let pkt = mgr.query(tree.node(e.node_slot).id, e.action)?;
        tree.add_child(&e, &pkt)?;
    }
    Ok(teacher.records.pop().expect("one record"))
}
