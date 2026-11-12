//! Sound AND/OR forced-mate prover with pluggable move ordering, a width limit and a node budget,
//! plus an INDEPENDENT proof checker.
//!
//! Semantics match `recur64_runtime::proof::mate::MateSolver` (checkmate ends the game; stalemate and
//! dead positions are not mates; `n` counts attacker moves). The only differences are ordering, an optional
//! width limit `k` (only the best `k` attacker moves are tried: incomplete, never unsound) and a node budget.
//! A returned [`Proof`] is a full AND/OR tree that [`verify`] re-checks against the rules alone.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use cozy_chess::{Board, GameStatus, Move};

use crate::tb::moves_of;

/// Ordering policy. `attacker` is called at OR nodes (white to move), `defender` at AND nodes.
/// Best candidates first.
pub trait Orderer {
    fn attacker(&mut self, b: &Board, moves: &mut Vec<Move>);
    fn defender(&mut self, b: &Board, moves: &mut Vec<Move>);
}

/// Generation order, no selection: the blind baseline.
pub struct Blind;
impl Orderer for Blind {
    fn attacker(&mut self, _b: &Board, _m: &mut Vec<Move>) {}
    fn defender(&mut self, _b: &Board, _m: &mut Vec<Move>) {}
}

/// True for dead positions (bare kings, or a lone minor piece); public for orderers.
pub fn dead(b: &Board) -> bool {
    is_dead(b)
}

fn ongoing(b: &Board) -> bool {
    b.status() == GameStatus::Ongoing && !is_dead(b)
}

/// Bare kings, or a king and a single minor piece against a bare king (the only dead positions reachable here).
fn is_dead(b: &Board) -> bool {
    let all = b.occupied().len();
    if all <= 2 {
        return true;
    }
    if all == 3 {
        let minors = b.pieces(cozy_chess::Piece::Bishop) | b.pieces(cozy_chess::Piece::Knight);
        return !minors.is_empty();
    }
    false
}

/// A proof as a shared DAG: the attacker plays `mv`; if that mates, `replies` is empty, otherwise it lists EVERY
/// legal defender reply with the attacker continuation. Positions reached by different lines share one node
/// (`Arc`), so a proof whose unfolded TREE would be astronomically large stays small in memory.
#[derive(Debug, Clone)]
pub struct Proof {
    pub mv: Move,
    pub replies: Vec<(Move, Arc<Proof>)>,
    depth: u8,
}

impl Proof {
    pub fn new(mv: Move, replies: Vec<(Move, Arc<Proof>)>) -> Self {
        let depth = 1 + replies.iter().map(|(_, p)| p.depth).max().unwrap_or(0);
        Self { mv, replies, depth }
    }

    /// Attacker moves along the longest line.
    pub fn depth(&self) -> u8 {
        self.depth
    }

    /// The line in which the defender always plays the reply that lasts longest inside this proof:
    /// `(attacker move, defender reply)` pairs; the last pair has no reply.
    pub fn principal_line(&self) -> Vec<(Move, Option<Move>)> {
        let mut line = Vec::new();
        let mut cur = self;
        loop {
            match cur.replies.iter().max_by_key(|(_, p)| p.depth) {
                Some((r, next)) => {
                    line.push((cur.mv, Some(*r)));
                    cur = next;
                }
                None => {
                    line.push((cur.mv, None));
                    return line;
                }
            }
        }
    }

    /// Number of distinct DAG nodes (shared sub-proofs counted once).
    pub fn dag_size(&self) -> usize {
        fn walk(p: &Proof, seen: &mut HashSet<*const Proof>) {
            if !seen.insert(p as *const Proof) {
                return;
            }
            for (_, c) in &p.replies {
                walk(c, seen);
            }
        }
        let mut seen = HashSet::new();
        walk(self, &mut seen);
        seen.len()
    }

    /// Size of the fully unfolded tree (can be astronomically larger than the DAG).
    pub fn tree_size(&self) -> f64 {
        fn walk(p: &Proof, memo: &mut HashMap<*const Proof, f64>) -> f64 {
            if let Some(v) = memo.get(&(p as *const Proof)) {
                return *v;
            }
            let v = 1.0 + p.replies.iter().map(|(_, c)| walk(c, memo)).sum::<f64>();
            memo.insert(p as *const Proof, v);
            v
        }
        walk(self, &mut HashMap::new())
    }
}

/// Independent checker: uses only the board rules, never the prover tables or the scorer. Nodes already verified are
/// memoised by (position hash, proof node) so shared sub-proofs are checked once; validity of a node depends only on
/// its position and sub-proofs. Returns the longest attacker-move line, or the first violation.
pub fn verify(root: &Board, proof: &Proof) -> Result<u8, String> {
    fn attacker(b: &Board, p: &Proof, memo: &mut HashMap<(u64, *const Proof), u8>) -> Result<u8, String> {
        if let Some(&d) = memo.get(&(b.hash(), p as *const Proof)) {
            return Ok(d);
        }
        if b.status() != GameStatus::Ongoing || is_dead(b) {
            return Err(format!("attacker to move in a finished position: {b}"));
        }
        if !moves_of(b).contains(&p.mv) {
            return Err(format!("illegal attacker move {} in {b}", p.mv));
        }
        let mut after = b.clone();
        after.play_unchecked(p.mv);
        let d = if after.status() == GameStatus::Won {
            if !p.replies.is_empty() {
                return Err("mating move with replies attached".into());
            }
            1
        } else {
            if after.status() != GameStatus::Ongoing || is_dead(&after) {
                return Err(format!("attacker move ends the game without mate: {after}"));
            }
            let legal = moves_of(&after);
            if p.replies.len() != legal.len() {
                return Err(format!(
                    "proof covers {} of {} defender replies in {after}",
                    p.replies.len(),
                    legal.len()
                ));
            }
            let mut deepest = 0;
            for l in &legal {
                let Some((_, sub)) = p.replies.iter().find(|(m, _)| m == l) else {
                    return Err(format!("defender reply {l} not covered in {after}"));
                };
                let mut next = after.clone();
                next.play_unchecked(*l);
                deepest = deepest.max(attacker(&next, sub, memo)?);
            }
            1 + deepest
        };
        memo.insert((b.hash(), p as *const Proof), d);
        Ok(d)
    }
    attacker(root, proof, &mut HashMap::new())
}

#[derive(Debug, Clone, Default)]
pub struct SolveStats {
    pub nodes: u64,
    pub tt_hits: u64,
}

#[derive(Debug)]
pub struct Solved {
    pub proof: Option<Arc<Proof>>,
    /// Attacker moves of the first (iterative-deepening) bound at which a proof was found.
    pub bound: Option<u8>,
    pub nodes: u64,
    pub budget_exhausted: bool,
}

pub struct Prover<'a, O: Orderer + ?Sized> {
    orderer: &'a mut O,
    width: Option<usize>,
    budget: u64,
    nodes: u64,
    /// `(hash, n)` of OR nodes: Some(move) = proven with that move; None = not provable under this width.
    or_tt: HashMap<(u64, u8), Option<Move>>,
    and_tt: HashMap<(u64, u8), bool>,
}

struct Out;

impl<'a, O: Orderer + ?Sized> Prover<'a, O> {
    pub fn new(orderer: &'a mut O, width: Option<usize>, budget: u64) -> Self {
        Self {
            orderer,
            width,
            budget,
            nodes: 0,
            or_tt: HashMap::new(),
            and_tt: HashMap::new(),
        }
    }

    fn tick(&mut self) -> Result<(), Out> {
        self.nodes += 1;
        if self.nodes > self.budget {
            Err(Out)
        } else {
            Ok(())
        }
    }

    fn or(&mut self, b: &Board, n: u8) -> Result<bool, Out> {
        if n == 0 || !ongoing(b) {
            return Ok(false);
        }
        let key = (b.hash(), n);
        if let Some(e) = self.or_tt.get(&key) {
            return Ok(e.is_some());
        }
        self.tick()?;
        let mut mv = moves_of(b);
        self.orderer.attacker(b, &mut mv);
        if let Some(k) = self.width {
            mv.truncate(k.max(1));
        }
        let mut found = None;
        for m in mv {
            let mut after = b.clone();
            after.play_unchecked(m);
            if after.status() == GameStatus::Won {
                found = Some(m);
                break;
            }
            if n >= 2 && self.and(&after, n - 1)? {
                found = Some(m);
                break;
            }
        }
        self.or_tt.insert(key, found);
        Ok(found.is_some())
    }

    fn and(&mut self, after: &Board, n: u8) -> Result<bool, Out> {
        if n == 0 || !ongoing(after) {
            return Ok(false);
        }
        let key = (after.hash(), n);
        if let Some(&v) = self.and_tt.get(&key) {
            return Ok(v);
        }
        self.tick()?;
        let mut mv = moves_of(after);
        self.orderer.defender(after, &mut mv);
        let mut all = true;
        for r in mv {
            let mut next = after.clone();
            next.play_unchecked(r);
            if !self.or(&next, n)? {
                all = false;
                break;
            }
        }
        self.and_tt.insert(key, all);
        Ok(all)
    }

    fn extract(&self, b: &Board, n: u8, memo: &mut HashMap<(u64, u8), Arc<Proof>>) -> Arc<Proof> {
        if let Some(p) = memo.get(&(b.hash(), n)) {
            return p.clone();
        }
        let m = self
            .or_tt
            .get(&(b.hash(), n))
            .copied()
            .flatten()
            .expect("extract only follows proven OR nodes");
        let mut after = b.clone();
        after.play_unchecked(m);
        let p = if after.status() == GameStatus::Won {
            Arc::new(Proof::new(m, Vec::new()))
        } else {
            let replies = moves_of(&after)
                .into_iter()
                .map(|r| {
                    let mut next = after.clone();
                    next.play_unchecked(r);
                    (r, self.extract(&next, n - 1, memo))
                })
                .collect();
            Arc::new(Proof::new(m, replies))
        };
        memo.insert((b.hash(), n), p.clone());
        p
    }

    /// Iterative deepening over the attacker-move bound `1..=max_n`.
    pub fn solve(self, root: &Board, max_n: u8) -> Solved {
        self.solve_from(root, 1, max_n)
    }

    /// Iterative deepening over `from_n..=max_n` (a diagnostic can start at the optimal bound).
    pub fn solve_from(mut self, root: &Board, from_n: u8, max_n: u8) -> Solved {
        for n in from_n..=max_n {
            match self.or(root, n) {
                Ok(true) => {
                    let proof = self.extract(root, n, &mut HashMap::new());
                    return Solved {
                        proof: Some(proof),
                        bound: Some(n),
                        nodes: self.nodes,
                        budget_exhausted: false,
                    };
                }
                Ok(false) => {}
                Err(Out) => {
                    return Solved {
                        proof: None,
                        bound: None,
                        nodes: self.nodes,
                        budget_exhausted: true,
                    };
                }
            }
        }
        Solved {
            proof: None,
            bound: None,
            nodes: self.nodes,
            budget_exhausted: false,
        }
    }
}
