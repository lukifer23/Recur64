//! Sound AND/OR forced-mate prover with pluggable move ordering, a width limit and a node budget,
//! plus an INDEPENDENT proof checker.
//!
//! Semantics match `recur64_runtime::proof::mate::MateSolver` (checkmate ends the game; stalemate and
//! dead positions are not mates; `n` counts attacker moves). The only differences are ordering, an optional
//! width limit `k` (only the best `k` attacker moves are tried: incomplete, never unsound) and a node budget.
//! A returned [`Proof`] is a full AND/OR tree that [`verify`] re-checks against the rules alone.

use std::collections::HashMap;

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

/// A verified-by-construction proof tree: the attacker plays `mv`; if that mates, `replies` is empty,
/// otherwise it lists EVERY legal defender reply with the attacker's continuation.
#[derive(Debug, Clone)]
pub struct Proof {
    pub mv: Move,
    pub replies: Vec<(Move, Proof)>,
}

impl Proof {
    /// Attacker moves along the longest line.
    pub fn depth(&self) -> u8 {
        1 + self.replies.iter().map(|(_, p)| p.depth()).max().unwrap_or(0)
    }

    pub fn size(&self) -> usize {
        1 + self.replies.iter().map(|(_, p)| p.size()).sum::<usize>()
    }
}

/// Independent checker: uses only the board rules, never the prover's tables or the scorer.
/// Returns the longest attacker-move line, or an error describing the first violation.
pub fn verify(root: &Board, proof: &Proof) -> Result<u8, String> {
    fn attacker(b: &Board, p: &Proof) -> Result<u8, String> {
        if b.status() != GameStatus::Ongoing || is_dead(b) {
            return Err(format!("attacker to move in a finished position: {b}"));
        }
        if !moves_of(b).contains(&p.mv) {
            return Err(format!("illegal attacker move {} in {b}", p.mv));
        }
        let mut after = b.clone();
        after.play_unchecked(p.mv);
        if after.status() == GameStatus::Won {
            return if p.replies.is_empty() {
                Ok(1)
            } else {
                Err("mating move with replies attached".into())
            };
        }
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
            deepest = deepest.max(attacker(&next, sub)?);
        }
        Ok(1 + deepest)
    }
    attacker(root, proof)
}

#[derive(Debug, Clone, Default)]
pub struct SolveStats {
    pub nodes: u64,
    pub tt_hits: u64,
}

#[derive(Debug)]
pub struct Solved {
    pub proof: Option<Proof>,
    /// Attacker moves of the first (iterative-deepening) bound at which a proof was found.
    pub bound: Option<u8>,
    pub nodes: u64,
    pub budget_exhausted: bool,
}

pub struct Prover<'a, O: Orderer> {
    orderer: &'a mut O,
    width: Option<usize>,
    budget: u64,
    nodes: u64,
    /// `(hash, n)` of OR nodes: Some(move) = proven with that move; None = not provable under this width.
    or_tt: HashMap<(u64, u8), Option<Move>>,
    and_tt: HashMap<(u64, u8), bool>,
}

struct Out;

impl<'a, O: Orderer> Prover<'a, O> {
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

    fn extract(&self, b: &Board, n: u8) -> Proof {
        let m = self
            .or_tt
            .get(&(b.hash(), n))
            .copied()
            .flatten()
            .expect("extract only follows proven OR nodes");
        let mut after = b.clone();
        after.play_unchecked(m);
        if after.status() == GameStatus::Won {
            return Proof { mv: m, replies: Vec::new() };
        }
        let replies = moves_of(&after)
            .into_iter()
            .map(|r| {
                let mut next = after.clone();
                next.play_unchecked(r);
                (r, self.extract(&next, n - 1))
            })
            .collect();
        Proof { mv: m, replies }
    }

    /// Iterative deepening over the attacker-move bound `1..=max_n`.
    pub fn solve(mut self, root: &Board, max_n: u8) -> Solved {
        for n in 1..=max_n {
            match self.or(root, n) {
                Ok(true) => {
                    let proof = self.extract(root, n);
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
