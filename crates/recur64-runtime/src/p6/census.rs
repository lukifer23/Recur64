//! TRAIN-side depth-2 state census: how many exact future states an ALL-INFO tree holds.
//! Counts come from the authoritative move generator (`GameState`), independently of the
//! model; a test proves they equal the tree builder's counts.

use std::collections::BTreeMap;

use serde::Serialize;

use recur64_core::GameState;

use crate::proof::targets::ProofTargets;

/// Counts of one position.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub struct PositionCount {
    pub root_legal: usize,
    pub depth1: usize,
    pub depth2: usize,
    pub terminal_depth1: usize,
    /// Largest legal list among the depth-1 states (the StateQuery cap is 256).
    pub max_child_legal: usize,
}

impl PositionCount {
    pub fn states(&self) -> usize {
        self.depth1 + self.depth2
    }
}

pub fn count_position(root: &GameState) -> PositionCount {
    let moves = root.legal_standard_moves();
    let mut c = PositionCount {
        root_legal: moves.len(),
        depth1: moves.len(),
        depth2: 0,
        terminal_depth1: 0,
        max_child_legal: 0,
    };
    for mv in moves {
        let mut child = root.clone();
        child.apply(mv).expect("a legal move applies");
        if child.is_terminal() {
            c.terminal_depth1 += 1;
        } else {
            let n = child.legal_standard_moves().len();
            c.depth2 += n;
            c.max_child_legal = c.max_child_legal.max(n);
        }
    }
    c
}

#[derive(Debug, Clone, Serialize)]
pub struct Stats {
    pub n: usize,
    pub min: u64,
    pub median: u64,
    pub p90: u64,
    pub p95: u64,
    pub p99: u64,
    pub max: u64,
    pub mean: f64,
}

/// `ceil(p * n) - 1` order statistic of a sorted slice.
fn q(sorted: &[u64], p: f64) -> u64 {
    let rank = ((p * sorted.len() as f64).ceil() as usize).clamp(1, sorted.len());
    sorted[rank - 1]
}

pub fn stats(mut v: Vec<u64>) -> Stats {
    v.sort_unstable();
    let n = v.len();
    Stats {
        n,
        min: v[0],
        median: q(&v, 0.5),
        p90: q(&v, 0.9),
        p95: q(&v, 0.95),
        p99: q(&v, 0.99),
        max: v[n - 1],
        mean: v.iter().sum::<u64>() as f64 / n as f64,
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Census {
    pub positions: usize,
    pub root_legal: Stats,
    pub depth1_states: Stats,
    pub depth2_states: Stats,
    pub total_future_states: Stats,
    pub by_cell: BTreeMap<String, Stats>,
    pub terminal_depth1_fraction: f64,
    /// Exact state transitions the exhaustive trees require (one per supplied state).
    pub total_state_transitions: u64,
    pub max_child_legal: usize,
}

/// Count every position of `data` (parallel; the result does not depend on scheduling).
pub fn census(data: &ProofTargets) -> anyhow::Result<Census> {
    let roots: Vec<GameState> = data
        .positions
        .iter()
        .map(|p| GameState::from_fen(&p.fen).map_err(|e| anyhow::anyhow!("{}: {e}", p.id)))
        .collect::<anyhow::Result<_>>()?;
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let chunk = roots.len().div_ceil(threads.max(1));
    let mut counts = vec![
        PositionCount {
            root_legal: 0,
            depth1: 0,
            depth2: 0,
            terminal_depth1: 0,
            max_child_legal: 0
        };
        roots.len()
    ];
    std::thread::scope(|s| {
        for (rs, cs) in roots.chunks(chunk).zip(counts.chunks_mut(chunk)) {
            s.spawn(move || {
                for (r, c) in rs.iter().zip(cs.iter_mut()) {
                    *c = count_position(r);
                }
            });
        }
    });
    let col =
        |f: fn(&PositionCount) -> usize| counts.iter().map(|c| f(c) as u64).collect::<Vec<_>>();
    let mut by_cell: BTreeMap<String, Vec<u64>> = BTreeMap::new();
    for (p, c) in data.positions.iter().zip(&counts) {
        by_cell
            .entry(format!("{} M{}", p.family, p.mate_depth))
            .or_default()
            .push(c.states() as u64);
    }
    let term: usize = counts.iter().map(|c| c.terminal_depth1).sum();
    let d1: usize = counts.iter().map(|c| c.depth1).sum();
    Ok(Census {
        positions: counts.len(),
        root_legal: stats(col(|c| c.root_legal)),
        depth1_states: stats(col(|c| c.depth1)),
        depth2_states: stats(col(|c| c.depth2)),
        total_future_states: stats(col(PositionCount::states)),
        by_cell: by_cell.into_iter().map(|(k, v)| (k, stats(v))).collect(),
        terminal_depth1_fraction: term as f64 / d1 as f64,
        total_state_transitions: counts.iter().map(|c| c.states() as u64).sum(),
        max_child_legal: counts.iter().map(|c| c.max_child_legal).max().unwrap_or(0),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use recur64_model::all_info::AllInfoTree;

    #[test]
    fn counts_match_the_exhaustive_tree_builder() {
        for fen in [
            "4k3/8/8/8/8/8/3Q4/R3K3 w - - 0 1",
            "8/8/8/4k3/8/8/4K3/1Q5R w - - 0 1",
            "k7/8/1K6/8/8/8/8/1Q5R w - - 0 1",
        ] {
            let r = GameState::from_fen(fen).unwrap();
            let c = count_position(&r);
            let t = AllInfoTree::build(&r).unwrap().counts();
            assert_eq!(
                (c.depth1, c.depth2, c.terminal_depth1),
                (t.depth1, t.depth2, t.terminal_depth1),
                "{fen}"
            );
        }
    }

    #[test]
    fn order_statistics_are_the_ceil_rank_values() {
        let s = stats((1..=100).collect());
        assert_eq!(
            (s.min, s.median, s.p90, s.p95, s.p99, s.max),
            (1, 50, 90, 95, 99, 100)
        );
        assert!((s.mean - 50.5).abs() < 1e-12);
    }
}
