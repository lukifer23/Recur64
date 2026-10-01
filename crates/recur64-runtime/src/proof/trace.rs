//! `ProofTraceV1`: the exact adversarial proof structure of a `ProofTargetsV1`
//! position, for TRAINING-SIDE process supervision only.
//!
//! Nothing here enters `StatePacketV1`; this module depends on the exact mate
//! solver and is never imported by the query tool or the model.
//!
//! # Structure (AND/OR proof graph)
//!
//! For a position of minimal mate depth `D` (attacker moves):
//!
//! * an **OR node** `(board, n)` is an attacker decision with `n` attacker moves
//!   left (including the one to play). Its alternatives are EVERY legal attacker
//!   move that forces mate within `n`: either an immediate checkmate (a leaf) or a
//!   move after which the defender is lost within `n - 1`;
//! * an **AND node** `(board, k)` is a defender decision after an attacker move,
//!   with `k` attacker moves left. Its alternatives are EVERY legal defender
//!   reply, each leading to the OR node `(next, k)`. A certificate must resolve all
//!   of them.
//!
//! Nodes are shared between paths that reach the same `(board, n)` (the stored
//! form is a DAG), but trace *edges* are identified by the root-relative action
//! path, because V3's query tree does not merge transpositions.
//!
//! # Certificate cost `Q*` (in exact query edges)
//!
//! `OR(b, n) = min over winning m of  1 + (0 if m mates else AND(after, n-1))` and
//! `AND(a, k) = sum over every reply r of  1 + OR(next, k)`.
//! `Q*(p) = OR(root, D)`: the exact minimum edge count of a complete certificate of
//! a correct root move.
//!
//! # Set-valued target `A(S)`
//!
//! For a queried-edge set `S` (a set of root-relative action paths), with
//! `r(T, S) = |T \ S|` over certificates `T`: `T*(S)` are the certificates that
//! minimise `r`; `A_proof(S)` are the frontier edges lying in at least one of them;
//! `A_refute(S)` are, for every queried child of an incorrect root move, its
//! refuting defender replies that are still unqueried. `A(S)` is their union. It is
//! a function of the SET `S` alone, never of the order the edges were queried in.
//! The selector target is uniform over `A(S)`.

use std::collections::{BTreeSet, HashMap, HashSet};

use cozy_chess::{Board, GameStatus, Move};
use recur64_core::{ActionId, Perspective, PromotionCode, StandardMove, is_insufficient_material};
use serde::{Deserialize, Serialize};

use super::mate::MateSolver;
use super::targets::ProofPosition;

/// Schema/contract identifier.
pub const TRACE_SCHEMA: &str = "proof_trace_v1";

/// Verbatim definition recorded in every shard digest.
pub const TRACE_DEFINITION: &str = "OR(b,n)=min over winning attacker moves of 1+(0 if mate else AND(after,n-1)); \
AND(a,k)=sum over EVERY defender reply of 1+OR(next,k); Q*=OR(root,D); A(S)=A_proof(S) union A_refute(S) \
over root-relative action paths; selector target uniform over A(S)";

/// Marker for a leaf child.
pub const NO_CHILD: u32 = u32::MAX;

/// A root-relative action path (canonical ActionId index per ply). Identifies one
/// query-tree edge: the last element is the edge's own action.
pub type Path = Vec<u16>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Kind {
    Or,
    And,
}

/// One alternative at a node.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Alt {
    /// Canonical ActionId index, relative to the side to move at the node.
    pub a: u16,
    /// OR only: the move delivers checkmate (a leaf).
    pub mate: bool,
    /// Child node index (`NO_CHILD` for a mating leaf).
    pub c: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Node {
    pub k: Kind,
    /// Attacker moves left (see the module docs).
    pub n: u8,
    /// Exact minimal certificate edge count below this node, from an empty `S`.
    pub cost: u64,
    pub alts: Vec<Alt>,
}

/// An incorrect root move and the defender replies that refute it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Refutation {
    pub root_action: u16,
    pub replies: Vec<u16>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PositionTrace {
    pub id: String,
    pub fen: String,
    pub family: String,
    pub mate_depth: u8,
    /// Exact minimum certificate size in query edges.
    pub q_star: u64,
    /// Index of the root OR node.
    pub root: u32,
    pub nodes: Vec<Node>,
    pub refutations: Vec<Refutation>,
}

fn sorted_moves(b: &Board) -> Vec<(u16, Move)> {
    let mut v = Vec::with_capacity(48);
    let side = b.side_to_move();
    b.generate_moves(|ms| {
        for m in ms {
            let sm = StandardMove::from_cozy(b, m).expect("cozy move is well formed");
            let id = ActionId::from_physical(
                sm.from,
                sm.to,
                sm.promotion.unwrap_or(PromotionCode::NONE),
                Perspective::of(side),
            );
            v.push((id.index() as u16, m));
        }
        false
    });
    v.sort_unstable_by_key(|(a, _)| *a);
    v
}

fn ongoing(b: &Board) -> bool {
    b.status() == GameStatus::Ongoing && !is_insufficient_material(b)
}

struct Builder<'a> {
    solver: &'a mut MateSolver,
    nodes: Vec<Node>,
    memo: HashMap<(u64, u8, bool), u32>,
}

impl Builder<'_> {
    fn push(&mut self, node: Node) -> u32 {
        self.nodes.push(node);
        (self.nodes.len() - 1) as u32
    }

    /// OR node of the side to move in `b` with `n` attacker moves left.
    fn or_node(&mut self, b: &Board, n: u8) -> anyhow::Result<u32> {
        let key = (b.hash(), n, false);
        if let Some(&i) = self.memo.get(&key) {
            return Ok(i);
        }
        let mut alts = Vec::new();
        let mut best: Option<u64> = None;
        for (a, m) in sorted_moves(b) {
            let mut after = b.clone();
            after.play_unchecked(m);
            let (mate, child, cost) = if after.status() == GameStatus::Won {
                (true, NO_CHILD, 1u64)
            } else if n >= 2 && self.solver.defender_lost(&after, n - 1) {
                let c = self.and_node(&after, n - 1)?;
                (false, c, 1u64.saturating_add(self.nodes[c as usize].cost))
            } else {
                continue;
            };
            best = Some(best.map_or(cost, |x| x.min(cost)));
            alts.push(Alt { a, mate, c: child });
        }
        let cost = best.ok_or_else(|| {
            anyhow::anyhow!("no winning alternative at an OR node with {n} attacker moves left")
        })?;
        let i = self.push(Node {
            k: Kind::Or,
            n,
            cost,
            alts,
        });
        self.memo.insert(key, i);
        Ok(i)
    }

    /// AND node: the defender to move in `after`, `k` attacker moves left.
    fn and_node(&mut self, after: &Board, k: u8) -> anyhow::Result<u32> {
        let key = (after.hash(), k, true);
        if let Some(&i) = self.memo.get(&key) {
            return Ok(i);
        }
        let mut alts = Vec::new();
        let mut cost = 0u64;
        for (a, r) in sorted_moves(after) {
            let mut next = after.clone();
            next.play_unchecked(r);
            let c = self.or_node(&next, k)?;
            cost = cost
                .saturating_add(1)
                .saturating_add(self.nodes[c as usize].cost);
            alts.push(Alt { a, mate: false, c });
        }
        anyhow::ensure!(!alts.is_empty(), "an AND node has no defender reply");
        let i = self.push(Node {
            k: Kind::And,
            n: k,
            cost,
            alts,
        });
        self.memo.insert(key, i);
        Ok(i)
    }
}

/// Build the trace of one proof position. The root alternatives must equal the
/// stored correct set (generator-level consistency; the independent audit
/// re-derives everything separately).
pub fn build_trace(solver: &mut MateSolver, p: &ProofPosition) -> anyhow::Result<PositionTrace> {
    let board: Board = p
        .fen
        .parse()
        .map_err(|e| anyhow::anyhow!("{}: {e:?}", p.id))?;
    let d = p.mate_depth;
    anyhow::ensure!(
        solver.mate_depth(&board, d) == Some(d),
        "{}: stored mate depth {d} not reproduced",
        p.id
    );
    let mut b = Builder {
        solver,
        nodes: Vec::new(),
        memo: HashMap::new(),
    };
    let root = b.or_node(&board, d)?;
    let nodes = b.nodes;
    let q_star = nodes[root as usize].cost;

    let stored: BTreeSet<u16> = p.correct.iter().map(|&i| p.legal[i as usize]).collect();
    let got: BTreeSet<u16> = nodes[root as usize].alts.iter().map(|a| a.a).collect();
    anyhow::ensure!(
        stored == got,
        "{}: trace root alternatives {got:?} differ from the stored correct set {stored:?}",
        p.id
    );

    // Refutations of incorrect root moves.
    let mut refutations = Vec::new();
    for (a, m) in sorted_moves(&board) {
        if got.contains(&a) {
            continue;
        }
        let mut after = board.clone();
        after.play_unchecked(m);
        if !ongoing(&after) {
            continue;
        }
        let mut replies = Vec::new();
        for (ra, r) in sorted_moves(&after) {
            let mut next = after.clone();
            next.play_unchecked(r);
            // The attacker cannot force mate within the remaining horizon d - 1.
            if !(d >= 2 && b.solver.attacker_forces(&next, d - 1)) {
                replies.push(ra);
            }
        }
        if !replies.is_empty() {
            refutations.push(Refutation {
                root_action: a,
                replies,
            });
        }
    }
    Ok(PositionTrace {
        id: p.id.clone(),
        fen: p.fen.clone(),
        family: p.family.clone(),
        mate_depth: d,
        q_star,
        root,
        nodes,
        refutations,
    })
}

/// The admissible next edges for a queried set.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Admissible {
    pub proof: BTreeSet<Path>,
    pub refute: BTreeSet<Path>,
}

impl Admissible {
    /// `A(S)` as one set.
    pub fn all(&self) -> BTreeSet<Path> {
        self.proof.union(&self.refute).cloned().collect()
    }

    pub fn is_empty(&self) -> bool {
        self.proof.is_empty() && self.refute.is_empty()
    }
}

impl PositionTrace {
    fn node(&self, i: u32) -> &Node {
        &self.nodes[i as usize]
    }

    fn edge_cost(s: &HashSet<Path>, path: &Path) -> (bool, u64) {
        let in_s = s.contains(path);
        (in_s, u64::from(!in_s))
    }

    fn or_residual(&self, x: u32, path: &mut Path, s: &HashSet<Path>) -> u64 {
        let mut best = u64::MAX;
        for alt in &self.node(x).alts {
            best = best.min(self.alt_residual_or(alt, path, s));
        }
        best
    }

    fn alt_residual_or(&self, alt: &Alt, path: &mut Path, s: &HashSet<Path>) -> u64 {
        path.push(alt.a);
        let (in_s, e) = Self::edge_cost(s, path);
        let child = if alt.mate {
            0
        } else if in_s {
            self.and_residual(alt.c, path, s)
        } else {
            self.node(alt.c).cost
        };
        path.pop();
        e.saturating_add(child)
    }

    fn and_residual(&self, y: u32, path: &mut Path, s: &HashSet<Path>) -> u64 {
        let mut total = 0u64;
        for alt in &self.node(y).alts {
            path.push(alt.a);
            let (in_s, e) = Self::edge_cost(s, path);
            let child = if in_s {
                self.or_residual(alt.c, path, s)
            } else {
                self.node(alt.c).cost
            };
            path.pop();
            total = total.saturating_add(e).saturating_add(child);
        }
        total
    }

    /// Remaining edges of a minimal certificate given the queried set `s`.
    pub fn residual(&self, s: &HashSet<Path>) -> u64 {
        self.or_residual(self.root, &mut Vec::new(), s)
    }

    /// The proof is complete: a full certificate lies inside `s`.
    pub fn is_complete(&self, s: &HashSet<Path>) -> bool {
        self.residual(s) == 0
    }

    fn collect_or(&self, x: u32, path: &mut Path, s: &HashSet<Path>, out: &mut BTreeSet<Path>) {
        let node = self.node(x);
        let min = self.or_residual(x, path, s);
        for alt in &node.alts {
            if self.alt_residual_or(alt, path, s) != min {
                continue;
            }
            path.push(alt.a);
            if !s.contains(path) {
                out.insert(path.clone());
            } else if !alt.mate {
                self.collect_and(alt.c, path, s, out);
            }
            path.pop();
        }
    }

    fn collect_and(&self, y: u32, path: &mut Path, s: &HashSet<Path>, out: &mut BTreeSet<Path>) {
        for alt in &self.node(y).alts {
            path.push(alt.a);
            if !s.contains(path) {
                out.insert(path.clone());
            } else {
                self.collect_or(alt.c, path, s, out);
            }
            path.pop();
        }
    }

    /// `A(S)` for the queried-edge set `s`. A function of the set only.
    ///
    /// A complete proof contributes no `A_proof` edges. `A_refute` is kept as the
    /// frozen definition states: the unqueried refuting replies of every queried
    /// incorrect root move.
    pub fn admissible(&self, s: &HashSet<Path>) -> Admissible {
        let mut proof = BTreeSet::new();
        if self.residual(s) > 0 {
            self.collect_or(self.root, &mut Vec::new(), s, &mut proof);
        }
        let mut refute = BTreeSet::new();
        for r in &self.refutations {
            let first: Path = vec![r.root_action];
            if !s.contains(&first) {
                continue;
            }
            for &reply in &r.replies {
                let edge: Path = vec![r.root_action, reply];
                if !s.contains(&edge) {
                    refute.insert(edge);
                }
            }
        }
        Admissible { proof, refute }
    }

    /// Every node reachable from the root, in index order (for audits).
    pub fn reachable(&self) -> Vec<u32> {
        let mut seen = vec![false; self.nodes.len()];
        let mut stack = vec![self.root];
        while let Some(i) = stack.pop() {
            if std::mem::replace(&mut seen[i as usize], true) {
                continue;
            }
            for a in &self.node(i).alts {
                if !a.mate && (a.c as usize) < self.nodes.len() {
                    stack.push(a.c);
                }
            }
        }
        (0..self.nodes.len() as u32)
            .filter(|&i| seen[i as usize])
            .collect()
    }
}

/// All certificates of a trace, straight from the definition: a certificate chooses
/// one alternative at every reachable OR node and takes EVERY reply at every AND
/// node. Each is returned as its set of edge paths. Exponential: tests only.
#[cfg(test)]
pub fn reference_certificates(t: &PositionTrace) -> Vec<BTreeSet<Path>> {
    fn certs_or(t: &PositionTrace, x: u32, path: &mut Path) -> Vec<BTreeSet<Path>> {
        let mut out = Vec::new();
        for alt in &t.nodes[x as usize].alts {
            path.push(alt.a);
            let edge = path.clone();
            if alt.mate {
                out.push(BTreeSet::from([edge]));
            } else {
                for mut c in certs_and(t, alt.c, path) {
                    c.insert(edge.clone());
                    out.push(c);
                }
            }
            path.pop();
        }
        out
    }
    fn certs_and(t: &PositionTrace, y: u32, path: &mut Path) -> Vec<BTreeSet<Path>> {
        // Cartesian product over every defender reply.
        let mut acc: Vec<BTreeSet<Path>> = vec![BTreeSet::new()];
        for alt in &t.nodes[y as usize].alts {
            path.push(alt.a);
            let edge = path.clone();
            let mut options = Vec::new();
            for mut c in certs_or(t, alt.c, path) {
                c.insert(edge.clone());
                options.push(c);
            }
            path.pop();
            let mut next = Vec::new();
            for a in &acc {
                for o in &options {
                    next.push(a.union(o).cloned().collect());
                }
            }
            acc = next;
        }
        acc
    }
    certs_or(t, t.root, &mut Vec::new())
}

/// Number of certificates (saturating), to keep brute-force fixtures small.
#[cfg(test)]
pub fn count_certificates(t: &PositionTrace) -> u128 {
    fn or(t: &PositionTrace, x: u32) -> u128 {
        t.nodes[x as usize].alts.iter().fold(0u128, |acc, a| {
            acc.saturating_add(if a.mate { 1 } else { and(t, a.c) })
        })
    }
    fn and(t: &PositionTrace, y: u32) -> u128 {
        t.nodes[y as usize]
            .alts
            .iter()
            .fold(1u128, |acc, a| acc.saturating_mul(or(t, a.c)))
    }
    or(t, t.root)
}

/// Brute-force reference for `A(S)`, written straight from the definition by
/// enumerating every certificate. Exponential, so for small fixtures only; used to
/// check the dynamic-programming implementation.
#[cfg(test)]
pub fn reference_admissible(t: &PositionTrace, s: &HashSet<Path>) -> Admissible {
    let certs = reference_certificates(t);
    let residual = |c: &BTreeSet<Path>| c.iter().filter(|e| !s.contains(*e)).count();
    let min = certs.iter().map(residual).min().unwrap_or(0);
    let mut proof = BTreeSet::new();
    if min > 0 {
        for c in certs.iter().filter(|c| residual(c) == min) {
            for e in c {
                if s.contains(e) {
                    continue;
                }
                // Frontier edge: its parent (the path without its last action) is
                // the root or a queried edge.
                let parent = &e[..e.len() - 1];
                if parent.is_empty() || s.contains(parent) {
                    proof.insert(e.clone());
                }
            }
        }
    }
    let mut refute = BTreeSet::new();
    for r in &t.refutations {
        if s.contains(&vec![r.root_action]) {
            for &reply in &r.replies {
                let e = vec![r.root_action, reply];
                if !s.contains(&e) {
                    refute.insert(e);
                }
            }
        }
    }
    Admissible { proof, refute }
}
