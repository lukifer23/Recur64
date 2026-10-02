//! The V3.5 label-only target provider.
//!
//! `A(S) = A_proof(S) ∪ A_refute(S)` over the CURRENT learner-induced queried set `S`,
//! uniform over every admissible frontier edge, all ties retained. It implements
//! [`QueryTargetProvider`], whose only output is a list of frontier indices: it has no
//! method that could choose the executed edge, and it is called after the choice is fixed.
//!
//! Unlike the P5 teacher there is NO completion latch and no filler schedule. Once the
//! proof residual is 0 the target is empty (loss masking only); the learner's forward
//! action schedule is unaffected.

use std::collections::{BTreeMap, HashSet};

use recur64_model::active::{EdgeRef, QueryTargetProvider, Tree};

use crate::proof::trace::{Path, PositionTrace};
use crate::proof::trace_teacher::{edge_path, node_paths};

/// What the provider observed about one example.
#[derive(Debug, Clone, Default)]
pub struct TargetStats {
    /// Steps with a non-empty selector target.
    pub supervised: usize,
    /// Of those, steps whose target contains at least one `A_refute` edge.
    pub refute_steps: usize,
    /// First step at which the proof was found complete.
    pub completed_at: Option<usize>,
}

pub struct ProofTargetProvider<'a> {
    traces: Vec<&'a PositionTrace>,
    /// `targets[example][step]`: the frontier indices handed to the loss.
    pub targets: Vec<Vec<Vec<usize>>>,
    pub stats: Vec<TargetStats>,
}

impl<'a> ProofTargetProvider<'a> {
    pub fn new(traces: Vec<&'a PositionTrace>) -> Self {
        let n = traces.len();
        Self {
            traces,
            targets: vec![Vec::new(); n],
            stats: vec![TargetStats::default(); n],
        }
    }

    pub fn supervised_total(&self) -> usize {
        self.stats.iter().map(|s| s.supervised).sum()
    }
}

impl QueryTargetProvider for ProofTargetProvider<'_> {
    fn targets(
        &mut self,
        example: usize,
        step: usize,
        frontier: &[EdgeRef],
        tree: &Tree,
    ) -> anyhow::Result<Vec<usize>> {
        let trace = *self
            .traces
            .get(example)
            .ok_or_else(|| anyhow::anyhow!("no proof trace for example {example}"))?;
        anyhow::ensure!(
            self.targets[example].len() == step,
            "target provider called out of order for example {example}: step {step} after {}",
            self.targets[example].len()
        );
        let paths = node_paths(tree);
        let s: HashSet<Path> = paths.iter().filter(|p| !p.is_empty()).cloned().collect();
        let mut out: Vec<usize> = Vec::new();
        if trace.is_complete(&s) {
            let st = &mut self.stats[example];
            st.completed_at.get_or_insert(step);
        } else {
            let adm = trace.admissible(&s);
            let all = adm.all();
            anyhow::ensure!(
                !all.is_empty(),
                "{}: an incomplete proof has an empty admissible set",
                trace.id
            );
            let mut by_path: BTreeMap<Path, usize> = BTreeMap::new();
            for (i, e) in frontier.iter().enumerate() {
                let p = edge_path(&paths, e);
                if all.contains(&p) {
                    by_path.insert(p, i);
                }
            }
            anyhow::ensure!(
                by_path.len() == all.len(),
                "{}: {} admissible edges are not on the live frontier",
                trace.id,
                all.len() - by_path.len()
            );
            out = by_path.into_values().collect();
            out.sort_unstable();
            let st = &mut self.stats[example];
            st.supervised += 1;
            if !adm.refute.is_empty() {
                st.refute_steps += 1;
            }
        }
        self.targets[example].push(out.clone());
        Ok(out)
    }
}
