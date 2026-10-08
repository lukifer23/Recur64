//! Exact bounded forced-mate teacher.
//!
//! Semantics ("mate within n"): the attacker forces checkmate using at most `n`
//! of its own moves; every defender reply is universally quantified. Terminal
//! states are classified by the Recur64 rules authority
//! (`recur64_core::rules::classify`) before any generic evaluation. Positions
//! carry no history (fresh_no_history domain), so repetition count is 1.
//!
//! Resource limits: each query has a node budget. On exhaustion the query
//! returns `Unknown`; nothing computed inside an aborted query is stored (cache
//! writes happen only when a subtree returns a completed exact value).

use cozy_chess::{Board, Color, Move};
use recur64_core::rules::{Termination, classify};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Yes,
    No,
    /// Node budget exhausted. Never to be read as `No`.
    Unknown,
}

struct Abort;

pub struct Oracle {
    node_limit: u64,
    nodes: u64,
    /// key: (position, remaining attacker moves, attacker colour, attacker to move)
    cache: HashMap<(Board, u8, Color, bool), bool>,
    cache_cap: usize,
    pub total_nodes: u64,
    pub aborted_queries: u64,
}

fn term(b: &Board) -> Option<Termination> {
    classify(b, 1, 0, None)
}

fn legal_moves(b: &Board) -> Vec<Move> {
    let mut v = Vec::with_capacity(64);
    b.generate_moves(|ms| {
        v.extend(ms);
        false
    });
    v
}

impl Oracle {
    pub fn new(node_limit: u64) -> Self {
        Self { node_limit, nodes: 0, cache: HashMap::new(), cache_cap: 2_000_000, total_nodes: 0, aborted_queries: 0 }
    }

    pub fn set_node_limit(&mut self, n: u64) {
        self.node_limit = n;
    }

    /// Drop all cached proofs (used between roots so every root starts empty).
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    fn tick(&mut self) -> Result<(), Abort> {
        self.nodes += 1;
        if self.nodes > self.node_limit { Err(Abort) } else { Ok(()) }
    }

    fn store(&mut self, key: (Board, u8, Color, bool), v: bool) {
        if self.cache.len() >= self.cache_cap {
            self.cache.clear();
        }
        self.cache.insert(key, v);
    }

    /// Attacker to move at `b`: can the attacker force mate within `n` moves?
    fn att(&mut self, b: &Board, att: Color, n: u8) -> Result<bool, Abort> {
        self.tick()?;
        let key = (b.clone(), n, att, true);
        if let Some(&v) = self.cache.get(&key) {
            return Ok(v);
        }
        let mut result = false;
        for mv in legal_moves(b) {
            let mut c = b.clone();
            c.play_unchecked(mv);
            // A move can only mate if it gives check; avoids a full
            // classification of quiet children when n == 1 (equivalent:
            // Checkmate requires a non-empty checker set).
            let in_check = !c.checkers().is_empty();
            if n == 1 && !in_check {
                continue;
            }
            match term(&c) {
                Some(Termination::Checkmate) => {
                    result = true;
                    break;
                }
                Some(_) => continue, // draw: this move does not win
                None => {}
            }
            if n >= 2 && self.def(&c, att, n - 1)? {
                result = true;
                break;
            }
        }
        self.store(key, result);
        Ok(result)
    }

    /// Defender to move at nonterminal `b`; attacker has `n` moves remaining.
    /// True iff every defender reply still allows the attacker to mate in `n`.
    fn def(&mut self, b: &Board, att: Color, n: u8) -> Result<bool, Abort> {
        self.tick()?;
        let key = (b.clone(), n, att, false);
        if let Some(&v) = self.cache.get(&key) {
            return Ok(v);
        }
        let mut result = true;
        for mv in legal_moves(b) {
            let mut c = b.clone();
            c.play_unchecked(mv);
            if term(&c).is_some() {
                // Defender reached a terminal non-win for the attacker
                // (stalemate/dead/anything but an attacker checkmate, which a
                // defender move cannot deliver in this domain).
                result = false;
                break;
            }
            if !self.att(&c, att, n)? {
                result = false;
                break;
            }
        }
        self.store(key, result);
        Ok(result)
    }

    fn run<F: FnOnce(&mut Self) -> Result<bool, Abort>>(&mut self, f: F) -> Verdict {
        self.nodes = 0;
        let r = f(self);
        self.total_nodes += self.nodes.min(self.node_limit);
        match r {
            Ok(true) => Verdict::Yes,
            Ok(false) => Verdict::No,
            Err(Abort) => {
                self.aborted_queries += 1;
                Verdict::Unknown
            }
        }
    }

    /// Attacker to move at root `b`: mate within `n` attacker moves?
    /// Precondition: `b` is nonterminal with the attacker to move.
    pub fn mate_within_root(&mut self, b: &Board, att: Color, n: u8) -> Verdict {
        debug_assert_eq!(b.side_to_move(), att);
        self.run(|o| o.att(b, att, n))
    }

    /// Defender to move at nonterminal child `b`: the attacker forces mate within
    /// `n` remaining attacker moves?
    pub fn mate_within_child(&mut self, b: &Board, att: Color, n: u8) -> Verdict {
        debug_assert_ne!(b.side_to_move(), att);
        self.run(|o| o.def(b, att, n))
    }
}

/// Result of exact minimal-depth determination for a root.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RootDepth {
    /// Minimal forced-mate depth (1..=max_depth), with absence below proven.
    Exact(u8),
    /// Proven: no forced mate within `max_depth`.
    NoMateWithin(u8),
    Unknown,
}

/// Minimal forced-mate depth of attacker-to-move root, up to `max_depth`.
pub fn root_depth(o: &mut Oracle, b: &Board, att: Color, max_depth: u8) -> RootDepth {
    for d in 1..=max_depth {
        match o.mate_within_root(b, att, d) {
            Verdict::Yes => return RootDepth::Exact(d),
            Verdict::No => {}
            Verdict::Unknown => return RootDepth::Unknown,
        }
    }
    RootDepth::NoMateWithin(max_depth)
}
