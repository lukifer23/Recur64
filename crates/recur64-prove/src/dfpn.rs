//! Depth-first proof-number search (df-pn, Nagai 2002) for forced mates, as an established, learning-free baseline.
//!
//! Same game semantics and the same depth bound `n` (attacker moves) as [`crate::search::Prover`], so cycles cannot
//! occur and no repetition handling is needed. The OR node is the attacker, the AND node the defender. Initial
//! proof/disproof numbers are 1/1 (plain df-pn, no heuristic initialisation). The returned proof is built from the
//! transposition table and goes through the same independent checker as every other proof.

use std::collections::HashMap;
use std::sync::Arc;

use cozy_chess::{Board, GameStatus};

use crate::search::{Proof, Solved, dead};
use crate::tb::{count_moves, moves_of};

const INF: u32 = 1 << 29;

#[derive(Clone, Copy)]
struct Pd {
    pn: u32,
    dn: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Key(u64, u8, bool); // (position hash, remaining attacker moves, is OR node)

enum Child {
    /// attacker mates: proven leaf
    Win,
    /// bound exhausted / dead / nothing to prove from here
    Lose,
    Node(Board, u8, bool),
}

pub struct DfPn {
    tt: HashMap<Key, Pd>,
    nodes: u64,
    budget: u64,
    out: bool,
    /// df-pn+ style initial numbers from the move count: an AND node needs all its replies proven
    /// (`pn` = number of replies), an OR node needs all its moves refuted (`dn` = number of moves).
    plus: bool,
}

fn ongoing(b: &Board) -> bool {
    b.status() == GameStatus::Ongoing && !dead(b)
}

fn sat(a: u32, b: u32) -> u32 {
    (a + b).min(INF)
}

impl DfPn {
    pub fn new(budget: u64) -> Self {
        Self { tt: HashMap::new(), nodes: 0, budget, out: false, plus: false }
    }

    /// df-pn with move-count initial proof numbers (the usual practical variant).
    pub fn new_plus(budget: u64) -> Self {
        Self { plus: true, ..Self::new(budget) }
    }

    fn init(&self, b: &Board, or: bool) -> Pd {
        if !self.plus {
            return Pd { pn: 1, dn: 1 };
        }
        let n = (count_moves(b) as u32).max(1);
        if or { Pd { pn: 1, dn: n } } else { Pd { pn: n, dn: 1 } }
    }

    fn children(b: &Board, n: u8, or: bool) -> Vec<Child> {
        if or {
            if n == 0 || !ongoing(b) {
                return Vec::new();
            }
            moves_of(b)
                .into_iter()
                .map(|m| {
                    let mut after = b.clone();
                    after.play_unchecked(m);
                    if after.status() == GameStatus::Won {
                        Child::Win
                    } else if n >= 2 {
                        Child::Node(after, n - 1, false)
                    } else {
                        Child::Lose
                    }
                })
                .collect()
        } else {
            if n == 0 || !ongoing(b) {
                return Vec::new();
            }
            moves_of(b)
                .into_iter()
                .map(|r| {
                    let mut next = b.clone();
                    next.play_unchecked(r);
                    Child::Node(next, n, true)
                })
                .collect()
        }
    }

    fn value(&self, c: &Child) -> Pd {
        match c {
            Child::Win => Pd { pn: 0, dn: INF },
            Child::Lose => Pd { pn: INF, dn: 0 },
            Child::Node(b, n, or) => self.tt.get(&Key(b.hash(), *n, *or)).copied().unwrap_or_else(|| self.init(b, *or)),
        }
    }

    fn mid(&mut self, b: &Board, n: u8, or: bool, thpn: u32, thdn: u32) {
        let key = Key(b.hash(), n, or);
        let cur = self.tt.get(&key).copied().unwrap_or_else(|| self.init(b, or));
        if cur.pn == 0 || cur.dn == 0 || cur.pn >= thpn || cur.dn >= thdn || self.out {
            return;
        }
        self.nodes += 1;
        if self.nodes > self.budget {
            self.out = true;
            return;
        }
        let kids = Self::children(b, n, or);
        if kids.is_empty() {
            // an OR node with no way to win (bound exhausted, finished position) or an AND node that cannot continue
            self.tt.insert(key, Pd { pn: INF, dn: 0 });
            return;
        }
        loop {
            let vals: Vec<Pd> = kids.iter().map(|c| self.value(c)).collect();
            let (pn, dn) = if or {
                (vals.iter().map(|v| v.pn).min().unwrap(), vals.iter().fold(0, |a, v| sat(a, v.dn)))
            } else {
                (vals.iter().fold(0, |a, v| sat(a, v.pn)), vals.iter().map(|v| v.dn).min().unwrap())
            };
            self.tt.insert(key, Pd { pn, dn });
            if pn == 0 || dn == 0 || pn >= thpn || dn >= thdn || self.out {
                return;
            }
            // select the child to expand and its thresholds
            let (best, c_thpn, c_thdn) = if or {
                let (mut bi, mut b1, mut b2) = (0usize, INF + 1, INF + 1);
                for (i, v) in vals.iter().enumerate() {
                    if v.pn < b1 {
                        b2 = b1;
                        b1 = v.pn;
                        bi = i;
                    } else if v.pn < b2 {
                        b2 = v.pn;
                    }
                }
                (bi, thpn.min(b2.saturating_add(1)), thdn.saturating_sub(dn).saturating_add(vals[bi].dn))
            } else {
                let (mut bi, mut b1, mut b2) = (0usize, INF + 1, INF + 1);
                for (i, v) in vals.iter().enumerate() {
                    if v.dn < b1 {
                        b2 = b1;
                        b1 = v.dn;
                        bi = i;
                    } else if v.dn < b2 {
                        b2 = v.dn;
                    }
                }
                (bi, thpn.saturating_sub(pn).saturating_add(vals[bi].pn), thdn.min(b2.saturating_add(1)))
            };
            match &kids[best] {
                Child::Node(cb, cn, cor) => self.mid(cb, *cn, *cor, c_thpn, c_thdn),
                _ => return, // a leaf cannot be expanded (its value already decided the node)
            }
        }
    }

    fn proven(&self, b: &Board, n: u8, or: bool) -> bool {
        self.tt.get(&Key(b.hash(), n, or)).is_some_and(|v| v.pn == 0)
    }

    fn extract(&self, b: &Board, n: u8, memo: &mut HashMap<(u64, u8), Arc<Proof>>) -> Option<Arc<Proof>> {
        if let Some(p) = memo.get(&(b.hash(), n)) {
            return Some(p.clone());
        }
        for m in moves_of(b) {
            let mut after = b.clone();
            after.play_unchecked(m);
            let proof = if after.status() == GameStatus::Won {
                Some(Arc::new(Proof::new(m, Vec::new())))
            } else if n >= 2 && self.proven(&after, n - 1, false) {
                let mut replies = Vec::new();
                let mut ok = true;
                for r in moves_of(&after) {
                    let mut next = after.clone();
                    next.play_unchecked(r);
                    match self.extract(&next, n - 1, memo) {
                        Some(p) => replies.push((r, p)),
                        None => {
                            ok = false;
                            break;
                        }
                    }
                }
                ok.then(|| Arc::new(Proof::new(m, replies)))
            } else {
                None
            };
            if let Some(p) = proof {
                memo.insert((b.hash(), n), p.clone());
                return Some(p);
            }
        }
        None
    }

    /// Iterative deepening over the attacker-move bound `1..=max_n`, one df-pn search per bound.
    pub fn solve(mut self, root: &Board, max_n: u8) -> Solved {
        for n in 1..=max_n {
            self.mid(root, n, true, INF, INF);
            if self.out {
                return Solved { proof: None, bound: None, nodes: self.nodes, budget_exhausted: true };
            }
            if self.proven(root, n, true) {
                let proof = self.extract(root, n, &mut HashMap::new());
                return Solved { proof, bound: Some(n), nodes: self.nodes, budget_exhausted: false };
            }
        }
        Solved { proof: None, bound: None, nodes: self.nodes, budget_exhausted: false }
    }
}
