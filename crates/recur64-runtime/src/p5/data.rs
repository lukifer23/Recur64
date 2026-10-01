//! Verified P5 inputs: TRAIN and TUNE with their traces, and the per-budget
//! samplers. Anything that is not exactly the frozen artifact is refused.

use std::path::Path;

use crate::proof::custody::load_working_split;
use crate::proof::sampler::{CellSampler, SamplerStats};
use crate::proof::targets::{ProofPosition, ProofTargets, Split};
use crate::proof::trace::PositionTrace;
use crate::proof::trace_store::{AuditManifest, load_all_traces};

use super::recipe::{self, BUDGETS};

/// The identities a dataset must have. `frozen()` is the only one used outside tests.
#[derive(Debug, Clone)]
pub struct Expected {
    pub digest: String,
    pub positions: usize,
    pub trace_manifest: String,
    pub split: Split,
}

impl Expected {
    pub fn train() -> Self {
        Self {
            digest: recipe::TRAIN_DIGEST.into(),
            positions: recipe::TRAIN_POSITIONS,
            trace_manifest: recipe::TRAIN_TRACE_MANIFEST.into(),
            split: Split::Train,
        }
    }

    pub fn tune() -> Self {
        Self {
            digest: recipe::TUNE_DIGEST.into(),
            positions: recipe::TUNE_POSITIONS,
            trace_manifest: recipe::TUNE_TRACE_MANIFEST.into(),
            split: Split::Tune,
        }
    }
}

/// A verified dataset with its traces, in source order.
#[derive(Debug)]
pub struct Dataset {
    pub targets: ProofTargets,
    pub traces: Vec<PositionTrace>,
    pub trace_manifest_digest: String,
}

impl Dataset {
    pub fn positions(&self) -> &[ProofPosition] {
        &self.targets.positions
    }

    /// `(family, mate_depth)` of every position, for the samplers.
    pub fn cells(&self) -> Vec<(String, u8)> {
        self.targets
            .positions
            .iter()
            .map(|p| (p.family.clone(), p.mate_depth))
            .collect()
    }
}

/// Load a dataset and its traces and verify every identity. Refuses holdouts and
/// confirmation splits (through `load_working_split`), the HOLDOUT_C digest, a
/// mismatched source digest, count, split or trace manifest, a missing or failing
/// audit, and any trace that does not match its source position.
pub fn load_dataset(path: &Path, trace_dir: &Path, expected: &Expected) -> anyhow::Result<Dataset> {
    let allowed = [expected.split];
    anyhow::ensure!(
        matches!(expected.split, Split::Train | Split::Tune),
        "P5 uses TRAIN and TUNE only"
    );
    let targets = load_working_split(path, &allowed)?;
    anyhow::ensure!(
        targets.digest == expected.digest,
        "{}: content digest {} is not the frozen {}",
        path.display(),
        targets.digest,
        expected.digest
    );
    anyhow::ensure!(
        targets.positions.len() == expected.positions,
        "{}: {} positions, expected {}",
        path.display(),
        targets.positions.len(),
        expected.positions
    );
    // The independent audit must exist, cover everything and have no failure.
    let audit = AuditManifest::load(trace_dir)?;
    anyhow::ensure!(
        audit.ok() && audit.total_checked == targets.positions.len(),
        "the trace audit in {} is incomplete or reports failures",
        trace_dir.display()
    );
    let (tm, traces) = load_all_traces(&targets, trace_dir)?;
    anyhow::ensure!(
        tm.manifest_digest == expected.trace_manifest,
        "trace manifest {} is not the frozen {}",
        tm.manifest_digest,
        expected.trace_manifest
    );
    anyhow::ensure!(
        audit.trace_manifest_digest == tm.manifest_digest,
        "the audit belongs to a different trace manifest"
    );
    for (p, t) in targets.positions.iter().zip(&traces) {
        anyhow::ensure!(
            p.id == t.id && p.fen == t.fen,
            "trace {} does not match position {}",
            t.id,
            p.id
        );
    }
    Ok(Dataset {
        targets,
        traces,
        trace_manifest_digest: tm.manifest_digest,
    })
}

/// One independent `cell_balanced_v1` sampler per training budget.
pub struct BudgetSamplers {
    samplers: Vec<CellSampler>,
}

impl BudgetSamplers {
    pub fn new(cells: &[(String, u8)], run_seed: u64) -> Self {
        Self {
            samplers: BUDGETS
                .iter()
                .map(|&b| CellSampler::new(cells, recipe::sampler_seed(run_seed, b)))
                .collect(),
        }
    }

    fn slot(budget: usize) -> usize {
        BUDGETS
            .iter()
            .position(|&b| b == budget)
            .unwrap_or_else(|| panic!("budget {budget} is not a P5 training budget"))
    }

    /// Next position index of a budget's stream and its ordinal in that stream.
    pub fn draw(&mut self, budget: usize) -> (usize, u64) {
        let s = &mut self.samplers[Self::slot(budget)];
        let ordinal = s.examples_drawn();
        (s.next_index(), ordinal)
    }

    /// Examples drawn so far per budget (checkpointed; resume fast-forwards to it).
    pub fn draws(&self) -> Vec<u64> {
        self.samplers
            .iter()
            .map(CellSampler::examples_drawn)
            .collect()
    }

    /// Fast-forward freshly built samplers to a recorded state. The sequence is a
    /// pure function of (seed, draw index), so replaying the draws is exact.
    pub fn fast_forward(&mut self, draws: &[u64]) -> anyhow::Result<()> {
        anyhow::ensure!(
            draws.len() == self.samplers.len(),
            "sampler state has the wrong shape"
        );
        for (s, &n) in self.samplers.iter_mut().zip(draws) {
            anyhow::ensure!(
                s.examples_drawn() == 0,
                "can only fast-forward a fresh sampler"
            );
            for _ in 0..n {
                s.next_index();
            }
        }
        Ok(())
    }

    pub fn stats(&self) -> Vec<(usize, SamplerStats)> {
        BUDGETS
            .iter()
            .zip(&self.samplers)
            .map(|(&b, s)| (b, s.stats()))
            .collect()
    }
}
