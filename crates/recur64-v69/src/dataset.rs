//! Root analysis (exact teacher), grouping, partition assignment, quota fill.

use crate::canon::{CanonKey, canonical_key, key_hex, key_id};
use crate::generate::Family;
use crate::oracle::{Oracle, RootDepth, Verdict, root_depth};
use crate::streams::{MasterSeed, keyed_u64};
use cozy_chess::{Board, Color};
use recur64_core::rules::{Termination, classify};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};

pub const MAX_DEPTH: u8 = 3;
/// Per-root selection cap per class (frozen design choice, see docs/v69/CONTRACT.md).
pub const CAP_PER_ROOT_PER_CLASS: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Partition {
    Fit,
    Val,
    Test,
}

impl Partition {
    pub const ALL: [Partition; 3] = [Partition::Fit, Partition::Val, Partition::Test];
    pub fn quota(self) -> usize {
        match self {
            Partition::Fit => 64,
            Partition::Val => 32,
            Partition::Test => 32,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Partition::Fit => "fit",
            Partition::Val => "val",
            Partition::Test => "test",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChildRec {
    pub mv: String,
    pub fen: String,
    pub key: String,
    /// "terminal:<label>" | "pos" | "neg"
    pub status: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RootRec {
    pub id: String,
    pub family: String,
    pub attacker: String,
    pub fen: String,
    pub key: String,
    pub depth: u8,
    pub legal_moves: usize,
    pub correct_moves: usize,
    pub terminal_children: usize,
    pub children: Vec<ChildRec>,
    pub oracle_nodes: u64,
    pub oracle_micros: u64,
}

#[derive(Debug)]
pub enum RootOutcome {
    Accepted(Box<RootRec>),
    RejectedM1,
    /// Exhaustively proven: no forced mate within MAX_DEPTH.
    RejectedBeyond,
    /// Resource-limited (UNKNOWN) at the root or at any child.
    Unresolved,
}

fn color_name(c: Color) -> &'static str {
    if c == Color::White { "white" } else { "black" }
}

pub fn parse_color(s: &str) -> Color {
    if s == "white" { Color::White } else { Color::Black }
}

/// Exact analysis of a freshly generated root; the oracle cache is cleared first.
pub fn analyze_root(oracle: &mut Oracle, family: Family, board: &Board, att: Color) -> RootOutcome {
    let t0 = std::time::Instant::now();
    let nodes0 = oracle.total_nodes;
    oracle.clear_cache();
    let m = match root_depth(oracle, board, att, MAX_DEPTH) {
        RootDepth::Exact(1) => return RootOutcome::RejectedM1,
        RootDepth::Exact(m) => m,
        RootDepth::NoMateWithin(_) => return RootOutcome::RejectedBeyond,
        RootDepth::Unknown => return RootOutcome::Unresolved,
    };
    let n = m - 1;
    let key = canonical_key(board, att);
    let mut children = Vec::new();
    let mut correct = 0usize;
    let mut terminal = 0usize;
    let mut moves = Vec::new();
    board.generate_moves(|ms| {
        moves.extend(ms);
        false
    });
    for mv in &moves {
        let mut c = board.clone();
        c.play_unchecked(*mv);
        let ckey = key_hex(&canonical_key(&c, att));
        // Terminal states are handled before generic ongoing-state evaluation.
        let status = if let Some(t) = classify(&c, 1, 0, None) {
            terminal += 1;
            // A terminal Checkmate child would mean M == 1 (already excluded).
            assert!(t != Termination::Checkmate, "mate child at M>=2 root");
            format!("terminal:{}", t.label())
        } else {
            match oracle.mate_within_child(&c, att, n) {
                Verdict::Yes => {
                    correct += 1;
                    "pos".to_string()
                }
                Verdict::No => "neg".to_string(),
                Verdict::Unknown => return RootOutcome::Unresolved,
            }
        };
        children.push(ChildRec { mv: mv.to_string(), fen: c.to_string(), key: ckey, status });
    }
    // Engineering gate: the root proof (mate within M) must be witnessed by at
    // least one positive child.
    assert!(correct >= 1, "root proven mate-in-{m} but no correct child");
    let rk = key_hex(&key);
    RootOutcome::Accepted(Box::new(RootRec {
        id: key_id(&rk),
        family: family.name().to_string(),
        attacker: color_name(att).to_string(),
        fen: board.to_string(),
        key: rk,
        depth: m,
        legal_moves: moves.len(),
        correct_moves: correct,
        terminal_children: terminal,
        children,
        oracle_nodes: oracle.total_nodes - nodes0,
        oracle_micros: t0.elapsed().as_micros() as u64,
    }))
}

// ---------------------------------------------------------------- grouping

#[derive(Clone, Debug, Serialize)]
pub struct Group {
    pub id: String,
    pub roots: Vec<usize>,
    pub family: String,
    /// Depth of the lexicographically smallest root key (stratum for assignment).
    pub stratum_depth: u8,
    pub mixed_depth: bool,
    pub partition: Partition,
}

struct Dsu(Vec<usize>);
impl Dsu {
    fn find(&mut self, x: usize) -> usize {
        let mut r = x;
        while self.0[r] != r {
            r = self.0[r];
        }
        let mut c = x;
        while self.0[c] != r {
            let nx = self.0[c];
            self.0[c] = r;
            c = nx;
        }
        r
    }
    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            self.0[ra.max(rb)] = ra.min(rb);
        }
    }
}

/// Connected components over roots sharing any canonical child position
/// (all children incl. terminal; full 65-byte canonical identity compared, never
/// a hash).
pub fn build_groups(seed: &MasterSeed, roots: &[RootRec]) -> Vec<Group> {
    let mut dsu = Dsu((0..roots.len()).collect());
    let mut owner: HashMap<&str, usize> = HashMap::new();
    for (i, r) in roots.iter().enumerate() {
        for c in &r.children {
            match owner.get(c.key.as_str()) {
                Some(&j) => dsu.union(i, j),
                None => {
                    owner.insert(c.key.as_str(), i);
                }
            }
        }
    }
    let mut comps: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for i in 0..roots.len() {
        let r = dsu.find(i);
        comps.entry(r).or_default().push(i);
    }
    let mut groups: Vec<Group> = comps
        .into_values()
        .map(|mut idx| {
            idx.sort_by(|&a, &b| roots[a].key.cmp(&roots[b].key));
            let first = &roots[idx[0]];
            Group {
                id: first.id.clone(),
                family: first.family.clone(),
                stratum_depth: first.depth,
                mixed_depth: idx.iter().any(|&i| roots[i].depth != first.depth),
                roots: idx,
                partition: Partition::Fit,
            }
        })
        .collect();
    assign_partitions(seed, &mut groups);
    groups
}

/// Deterministic, label-blind partition assignment: within each stratum
/// (family, depth) groups are ordered by a keyed hash of their id and dealt in
/// blocks of four [Fit, Fit, Val, Test] permuted by the partition stream.
fn assign_partitions(seed: &MasterSeed, groups: &mut [Group]) {
    let mut strata: BTreeMap<(String, u8), Vec<usize>> = BTreeMap::new();
    for (i, g) in groups.iter().enumerate() {
        strata.entry((g.family.clone(), g.stratum_depth)).or_default().push(i);
    }
    for ((fam, d), mut idx) in strata {
        idx.sort_by_key(|&i| keyed_u64(seed, "partition/order", groups[i].id.as_bytes()));
        for (b, chunk) in idx.chunks(4).enumerate() {
            let mut block = [Partition::Fit, Partition::Fit, Partition::Val, Partition::Test];
            let mut rng = seed.stream(&format!("partition/{fam}/{d}"), b as u64);
            for k in (1..4).rev() {
                block.swap(k, rng.below(k as u64 + 1) as usize);
            }
            for (j, &gi) in chunk.iter().enumerate() {
                groups[gi].partition = block[j];
            }
        }
    }
}

// ---------------------------------------------------------------- selection

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Example {
    pub id: String,
    pub partition: Partition,
    pub family: String,
    pub budget: u8,
    pub label: bool,
    pub fen: String,
    pub key: String,
    pub root_id: String,
    pub group_id: String,
    pub root_fen: String,
    pub root_depth: u8,
    pub mv: String,
}

pub type Cell = (String, u8, bool);

pub struct Selection {
    pub examples: Vec<Example>,
    /// Examples obtained per (partition, cell); quota is `Partition::quota`.
    pub filled: BTreeMap<(Partition, Cell), usize>,
}

impl Selection {
    pub fn feasible(&self) -> bool {
        self.shortfalls().is_empty()
    }

    pub fn shortfalls(&self) -> Vec<(Partition, Cell, usize, usize)> {
        let mut v = Vec::new();
        for p in Partition::ALL {
            for f in Family::ALL {
                for b in [1u8, 2] {
                    for c in [false, true] {
                        let cell = (f.name().to_string(), b, c);
                        let got = self.filled.get(&(p, cell.clone())).copied().unwrap_or(0);
                        if got < p.quota() {
                            v.push((p, cell, got, p.quota()));
                        }
                    }
                }
            }
        }
        v
    }
}

/// Fill every (partition, family, budget, class) cell up to quota.
pub fn select_examples(seed: &MasterSeed, roots: &[RootRec], groups: &[Group]) -> Selection {
    let mut root_group: Vec<usize> = vec![0; roots.len()];
    for (gi, g) in groups.iter().enumerate() {
        for &r in &g.roots {
            root_group[r] = gi;
        }
    }
    let mut examples = Vec::new();
    let mut filled = BTreeMap::new();
    let mut taken: HashSet<(String, u8)> = HashSet::new();
    for p in Partition::ALL {
        for f in Family::ALL {
            for b in [1u8, 2] {
                let mut rs: Vec<usize> = (0..roots.len())
                    .filter(|&i| {
                        roots[i].family == f.name()
                            && roots[i].depth == b + 1
                            && groups[root_group[i]].partition == p
                    })
                    .collect();
                rs.sort_by_key(|&i| keyed_u64(seed, "selection/root", roots[i].key.as_bytes()));
                for class in [false, true] {
                    let want = if class { "pos" } else { "neg" };
                    let mut count = 0usize;
                    for &ri in &rs {
                        if count >= p.quota() {
                            break;
                        }
                        let r = &roots[ri];
                        let mut cand: Vec<&ChildRec> = r.children.iter().filter(|c| c.status == want).collect();
                        cand.sort_by_key(|c| {
                            keyed_u64(seed, "selection/child", format!("{}|{}", r.key, c.mv).as_bytes())
                        });
                        let mut got = 0;
                        for c in cand {
                            if got >= CAP_PER_ROOT_PER_CLASS || count >= p.quota() {
                                break;
                            }
                            if !taken.insert((c.key.clone(), b)) {
                                continue; // same canonical position+budget already used
                            }
                            examples.push(Example {
                                id: format!("{}-{}-{}", p.name(), key_id(&c.key), b),
                                partition: p,
                                family: f.name().to_string(),
                                budget: b,
                                label: class,
                                fen: c.fen.clone(),
                                key: c.key.clone(),
                                root_id: r.id.clone(),
                                group_id: groups[root_group[ri]].id.clone(),
                                root_fen: r.fen.clone(),
                                root_depth: r.depth,
                                mv: c.mv.clone(),
                            });
                            got += 1;
                            count += 1;
                        }
                    }
                    filled.insert((p, (f.name().to_string(), b, class)), count);
                }
            }
        }
    }
    Selection { examples, filled }
}

pub fn key_from_hex(s: &str) -> CanonKey {
    let mut k = [0u8; 65];
    for i in 0..65 {
        k[i] = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap();
    }
    k
}
