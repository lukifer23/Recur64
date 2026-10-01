//! Verified P6 inputs. ALL-INFO needs positions and root policy targets only: no
//! `ProofTrace` is loaded, so no proof information can reach the model or the sampler.
//! HOLDOUT_C and confirmation splits are refused by `load_working_split`.

use std::path::Path;

use recur64_core::GameState;
use recur64_model::all_info::AllInfoTree;

use crate::p5::recipe::{TRAIN_DIGEST, TRAIN_POSITIONS, TUNE_DIGEST, TUNE_POSITIONS};
use crate::proof::custody::load_working_split;
use crate::proof::targets::{ProofPosition, ProofTargets, Split};

/// Which frozen split to load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Which {
    Train,
    Tune,
}

impl Which {
    fn expect(self) -> (&'static str, usize, Split) {
        match self {
            Which::Train => (TRAIN_DIGEST, TRAIN_POSITIONS, Split::Train),
            Which::Tune => (TUNE_DIGEST, TUNE_POSITIONS, Split::Tune),
        }
    }
}

/// Load a frozen split without its traces and verify its identity.
pub fn load_targets(path: &Path, which: Which) -> anyhow::Result<ProofTargets> {
    let (digest, n, split) = which.expect();
    let t = load_working_split(path, &[split])?;
    anyhow::ensure!(
        t.digest == digest,
        "{}: content digest {} is not the frozen {digest}",
        path.display(),
        t.digest
    );
    anyhow::ensure!(
        t.positions.len() == n,
        "{}: {} positions, expected {n}",
        path.display(),
        t.positions.len()
    );
    Ok(t)
}

/// `(family, mate_depth)` of every position, for the sampler.
pub fn cells(t: &ProofTargets) -> Vec<(String, u8)> {
    t.positions
        .iter()
        .map(|p| (p.family.clone(), p.mate_depth))
        .collect()
}

/// Parse the roots of `positions`.
pub fn roots_of(positions: &[&ProofPosition]) -> anyhow::Result<Vec<GameState>> {
    positions
        .iter()
        .map(|p| GameState::from_fen(&p.fen).map_err(|e| anyhow::anyhow!("{}: {e}", p.id)))
        .collect()
}

/// Build the exhaustive depth-2 trees of `roots`, in parallel across threads. The result
/// order equals the input order, so it is independent of scheduling.
pub fn build_trees(roots: &[GameState]) -> anyhow::Result<Vec<AllInfoTree>> {
    let threads = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(roots.len().max(1));
    if threads <= 1 {
        return roots.iter().map(AllInfoTree::build).collect();
    }
    let mut out: Vec<Option<anyhow::Result<AllInfoTree>>> =
        (0..roots.len()).map(|_| None).collect();
    let chunk = roots.len().div_ceil(threads);
    std::thread::scope(|s| {
        for (rs, os) in roots.chunks(chunk).zip(out.chunks_mut(chunk)) {
            s.spawn(move || {
                for (r, o) in rs.iter().zip(os.iter_mut()) {
                    *o = Some(AllInfoTree::build(r));
                }
            });
        }
    });
    out.into_iter()
        .map(|o| o.expect("every root was built"))
        .collect()
}
