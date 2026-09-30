//! `cell_balanced_v1`: a versioned, deterministic training-example sampler.
//!
//! A CELL is `(family, mate_depth)`. Combinatorial pool size must not silently
//! choose the training objective, so every non-empty cell gets equal long-run
//! sampling probability:
//!
//! * a rotor visits the non-empty cells in a fixed order; global example `g`
//!   draws from cell `g mod K`;
//! * each cell owns an independent seeded shuffle of its positions and consumes
//!   it without replacement;
//! * when a cell's queue is exhausted it reshuffles deterministically into its
//!   next local epoch (a new permutation seeded by the local epoch number);
//! * nothing is duplicated on disk: oversampling a small cell is sampling, not
//!   extra unique data.
//!
//! The rotor advances globally across updates, so with an effective batch that is
//! not a multiple of `K` the cells that receive the extra example rotate, and the
//! long-run counts of any two cells never differ by more than one example.
//!
//! The sequence is a pure function of (model seed, [`VERSION`], global example
//! index).

use std::collections::BTreeMap;

use serde::Serialize;

use super::generator::{Rng, mix};

/// Sampler contract version, recorded with every run.
pub const VERSION: &str = "cell_balanced_v1";

/// One cell's queue state.
struct Cell {
    /// Indices into the caller's position list.
    members: Vec<usize>,
    order: Vec<usize>,
    cursor: usize,
    /// Number of times this cell's queue has been (re)built; 1 after the first draw.
    epochs_started: u64,
    consumed: u64,
}

/// The sampler.
pub struct CellSampler {
    seed: u64,
    cells: Vec<((String, u8), Cell)>,
    /// Global example counter (drives the rotor).
    drawn: u64,
}

fn permute(members: &[usize], seed: u64, cell_tag: u64, local_epoch: u64) -> Vec<usize> {
    let mut v = members.to_vec();
    let mut rng = Rng(mix(seed ^ mix(cell_tag ^ mix(local_epoch + 0x5EED))));
    for i in (1..v.len()).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
    v
}

/// Stable numeric tag of a cell key (independent of iteration order).
fn cell_tag(key: &(String, u8)) -> u64 {
    let mut h = 0xCBF2_9CE4_8422_2325u64; // FNV-1a
    for b in key.0.bytes().chain([b'|', key.1]) {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

impl CellSampler {
    /// `cells[i]` is the `(family, depth)` of position `i`.
    pub fn new(cells: &[(String, u8)], seed: u64) -> Self {
        let mut groups: BTreeMap<(String, u8), Vec<usize>> = BTreeMap::new();
        for (i, c) in cells.iter().enumerate() {
            groups.entry(c.clone()).or_default().push(i);
        }
        let cells = groups
            .into_iter()
            .filter(|(_, m)| !m.is_empty())
            .map(|(k, members)| {
                (
                    k,
                    Cell {
                        members,
                        order: Vec::new(),
                        cursor: 0,
                        epochs_started: 0,
                        consumed: 0,
                    },
                )
            })
            .collect();
        Self {
            seed,
            cells,
            drawn: 0,
        }
    }

    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }

    /// The next training position index.
    pub fn next_index(&mut self) -> usize {
        let k = self.cells.len() as u64;
        let slot = (self.drawn % k) as usize;
        self.drawn += 1;
        let seed = self.seed;
        let (key, cell) = &mut self.cells[slot];
        if cell.cursor >= cell.order.len() {
            cell.order = permute(&cell.members, seed, cell_tag(key), cell.epochs_started);
            cell.cursor = 0;
            cell.epochs_started += 1;
        }
        let i = cell.order[cell.cursor];
        cell.cursor += 1;
        cell.consumed += 1;
        i
    }

    pub fn examples_drawn(&self) -> u64 {
        self.drawn
    }

    /// Exposure statistics so far.
    pub fn stats(&self) -> SamplerStats {
        let total = self.drawn.max(1) as f64;
        let mut by_family: BTreeMap<String, u64> = BTreeMap::new();
        let mut by_depth: BTreeMap<String, u64> = BTreeMap::new();
        let mut cells = Vec::new();
        for ((family, depth), c) in &self.cells {
            *by_family.entry(family.clone()).or_default() += c.consumed;
            *by_depth.entry(format!("M{depth}")).or_default() += c.consumed;
            let unique = c.members.len() as u64;
            cells.push(CellStats {
                cell: format!("{family}-M{depth}"),
                unique_positions: unique,
                examples_consumed: c.consumed,
                local_epochs_seen: c.consumed as f64 / unique as f64,
                wrap_count: c.epochs_started.saturating_sub(1),
                fraction_of_training_examples: c.consumed as f64 / total,
            });
        }
        SamplerStats {
            version: VERSION.into(),
            examples_drawn: self.drawn,
            cells,
            fraction_by_family: by_family
                .into_iter()
                .map(|(k, v)| (k, v as f64 / total))
                .collect(),
            fraction_by_depth: by_depth
                .into_iter()
                .map(|(k, v)| (k, v as f64 / total))
                .collect(),
        }
    }
}

/// Per-cell exposure.
#[derive(Debug, Clone, Serialize)]
pub struct CellStats {
    pub cell: String,
    pub unique_positions: u64,
    pub examples_consumed: u64,
    pub local_epochs_seen: f64,
    pub wrap_count: u64,
    pub fraction_of_training_examples: f64,
}

/// Exposure summary of a run.
#[derive(Debug, Clone, Serialize)]
pub struct SamplerStats {
    pub version: String,
    pub examples_drawn: u64,
    pub cells: Vec<CellStats>,
    pub fraction_by_family: BTreeMap<String, f64>,
    pub fraction_by_depth: BTreeMap<String, f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// 3 families x depths with very different sizes.
    fn layout() -> Vec<(String, u8)> {
        let mut v = Vec::new();
        for (fam, sizes) in [
            ("KQvK", [7, 11, 13]),
            ("KRvK", [5, 9, 6]),
            ("KQQvK", [40, 50, 60]),
        ] {
            for (d, n) in sizes.iter().enumerate() {
                for _ in 0..*n {
                    v.push((fam.to_string(), d as u8 + 1));
                }
            }
        }
        v
    }

    fn draw(seed: u64, n: usize) -> Vec<usize> {
        let l = layout();
        let mut s = CellSampler::new(&l, seed);
        (0..n).map(|_| s.next_index()).collect()
    }

    #[test]
    fn same_seed_same_sequence_different_seed_different_order() {
        assert_eq!(draw(3, 2000), draw(3, 2000));
        assert_ne!(draw(3, 2000), draw(4, 2000));
        // Different seed still visits the same cell at the same global index.
        let l = layout();
        let (a, b) = (draw(3, 500), draw(4, 500));
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(l[*x], l[*y], "the rotor, not the seed, picks the cell");
        }
    }

    #[test]
    fn all_cells_represented_and_exposure_equal_within_one() {
        let l = layout();
        let mut s = CellSampler::new(&l, 9);
        assert_eq!(s.cell_count(), 9);
        // 1000 is not a multiple of 9, so the remainder must rotate.
        for _ in 0..1000 {
            s.next_index();
        }
        let st = s.stats();
        let counts: Vec<u64> = st.cells.iter().map(|c| c.examples_consumed).collect();
        assert!(
            counts.iter().max().unwrap() - counts.iter().min().unwrap() <= 1,
            "{counts:?}"
        );
        // Rotation across many "updates" of a non-multiple batch stays balanced.
        let mut s = CellSampler::new(&l, 9);
        for _update in 0..37 {
            for _ in 0..256 {
                s.next_index();
            }
        }
        let c: Vec<u64> = s
            .stats()
            .cells
            .iter()
            .map(|c| c.examples_consumed)
            .collect();
        assert!(
            c.iter().max().unwrap() - c.iter().min().unwrap() <= 1,
            "{c:?}"
        );
    }

    #[test]
    fn no_duplicate_within_a_local_epoch_and_wrap_is_deterministic() {
        let l = layout();
        let mut s = CellSampler::new(&l, 5);
        let k = s.cell_count();
        // The smallest cell has 5 positions: its first 5 draws are distinct.
        let mut seen: Vec<HashSet<usize>> = vec![HashSet::new(); k];
        let smallest = 5usize;
        for g in 0..(k * smallest) {
            let i = s.next_index();
            assert!(seen[g % k].insert(i), "duplicate inside a local epoch");
        }
        // Continue past several wraps of the small cell; two identical samplers agree.
        let a = draw(5, 5000);
        let b = draw(5, 5000);
        assert_eq!(a, b);
        let mut s = CellSampler::new(&l, 5);
        for _ in 0..5000 {
            s.next_index();
        }
        let st = s.stats();
        let small = st.cells.iter().find(|c| c.cell == "KRvK-M1").unwrap();
        assert_eq!(small.unique_positions, 5);
        assert!(small.wrap_count >= 50, "{small:?}");
        assert!((small.local_epochs_seen - small.examples_consumed as f64 / 5.0).abs() < 1e-9);
    }

    #[test]
    fn a_new_local_epoch_reshuffles_the_same_members() {
        let l = layout();
        let mut s = CellSampler::new(&l, 1);
        // Collect the draws of the 5-position cell (KRvK, depth 1) over 2 epochs.
        let mut small = Vec::new();
        while small.len() < 10 {
            let i = s.next_index();
            if l[i] == ("KRvK".to_string(), 1) {
                small.push(i);
            }
        }
        let (e1, e2) = (&small[..5], &small[5..]);
        let (a, b): (HashSet<_>, HashSet<_>) = (e1.iter().collect(), e2.iter().collect());
        assert_eq!(a.len(), 5);
        assert_eq!(a, b, "each local epoch visits exactly the cell's members");
        assert_ne!(e1, e2, "the next local epoch is a different permutation");
    }

    #[test]
    fn fractions_sum_to_one_by_family_and_depth() {
        let l = layout();
        let mut s = CellSampler::new(&l, 2);
        for _ in 0..900 {
            s.next_index();
        }
        let st = s.stats();
        assert!((st.fraction_by_family.values().sum::<f64>() - 1.0).abs() < 1e-9);
        assert!((st.fraction_by_depth.values().sum::<f64>() - 1.0).abs() < 1e-9);
        // 9 equal cells: each family holds 3 cells => 1/3 of the exposure.
        for v in st.fraction_by_family.values() {
            assert!((v - 1.0 / 3.0).abs() < 0.01);
        }
    }
}
