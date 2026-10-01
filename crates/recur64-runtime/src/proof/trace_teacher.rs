//! Live adapter between `ProofTraceV1` and the model's query tree.
//!
//! `ProofTraceTeacher` implements [`QueryScript`]: at each step it reads the real
//! [`Tree`] (the same structure `ActiveSearchModel::run` queries), derives the
//! queried-edge set `S` as root-relative action paths, computes `A(S)` from the
//! trace, and maps each admissible path onto an actual frontier edge. It does not
//! invent edges and it fails loudly if an admissible edge is not on the frontier.

use std::collections::HashSet;

use recur64_model::active::{EdgeRef, QueryScript, ScriptStep, Tree};

use super::trace::{Path, PositionTrace};

/// Root-relative action path of every node, indexed by tree slot.
pub fn node_paths(tree: &Tree) -> Vec<Path> {
    let mut paths: Vec<Path> = Vec::with_capacity(tree.len());
    for n in tree.nodes() {
        match (n.parent_slot, n.incoming_action) {
            (None, _) => paths.push(Vec::new()),
            (Some(p), Some(a)) => {
                let mut path = paths[p].clone();
                path.push(a);
                paths.push(path);
            }
            (Some(_), None) => unreachable!("a non-root node has an incoming action"),
        }
    }
    paths
}

/// `S`: the path of every queried edge (one per non-root node).
pub fn queried_set(tree: &Tree) -> HashSet<Path> {
    node_paths(tree)
        .into_iter()
        .filter(|p| !p.is_empty())
        .collect()
}

/// The path identifying a frontier edge.
pub fn edge_path(paths: &[Path], e: &EdgeRef) -> Path {
    let mut p = paths[e.node_slot].clone();
    p.push(e.action);
    p
}

/// Teacher for a batch: `traces[i]` belongs to example `i`.
///
/// This is the REFERENCE adapter for the generic target `A(S) = A_proof(S) union
/// A_refute(S)`, kept unchanged for diagnostics and future off-policy work. It follows
/// the first admissible edge and does not latch completion. The versioned P5 training
/// schedule is `p5::teacher::SeededProofTeacher`, which does both differently.
pub struct ProofTraceTeacher<'a> {
    pub traces: &'a [PositionTrace],
}

impl QueryScript for ProofTraceTeacher<'_> {
    fn next(
        &mut self,
        example: usize,
        _step: usize,
        frontier: &[EdgeRef],
        tree: &Tree,
    ) -> anyhow::Result<ScriptStep> {
        let trace = self
            .traces
            .get(example)
            .ok_or_else(|| anyhow::anyhow!("no proof trace for example {example}"))?;
        let paths = node_paths(tree);
        let s: HashSet<Path> = paths.iter().filter(|p| !p.is_empty()).cloned().collect();
        let admissible = trace.admissible(&s).all();
        let mut targets = Vec::new();
        let mut matched = std::collections::BTreeSet::new();
        for (i, e) in frontier.iter().enumerate() {
            let p = edge_path(&paths, e);
            if admissible.contains(&p) {
                targets.push(i);
                matched.insert(p);
            }
        }
        anyhow::ensure!(
            matched.len() == admissible.len(),
            "{}: {} admissible edges are not on the live frontier (e.g. {:?})",
            trace.id,
            admissible.len() - matched.len(),
            admissible.difference(&matched).next()
        );
        // Follow the first admissible edge; once the proof is complete the budget
        // is still forced, so fall back to the frozen breadth-first order with no
        // selector target.
        let follow = targets.first().copied().unwrap_or_else(|| {
            frontier
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.bfs_key())
                .map(|(i, _)| i)
                .expect("a non-empty frontier")
        });
        Ok(ScriptStep { follow, targets })
    }
}
