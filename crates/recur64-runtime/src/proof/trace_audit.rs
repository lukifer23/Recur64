//! Independent audit of `ProofTraceV1`.
//!
//! The generator ([`super::trace`]) works on raw `cozy_chess` boards with the
//! memoizing [`super::mate::MateSolver`]. This audit re-derives every claim through
//! a different path: `GameState` values, `GameState::apply` with its full
//! termination classification, `legal_actions()`, and the independent
//! [`super::audit`] solver memo. A trace is accepted only if both paths agree on
//! every node; nothing in the generator is called.
//!
//! Checked per trace: source identity; the root legal list and correct set against
//! `ProofTargets`; for every reachable OR node that its alternatives are EXACTLY the
//! winning attacker moves (none missing, none extra, mate flags right); for every
//! AND node that EVERY legal reply is present; that a node shared between paths is
//! the same `(position, n)`; the incorrect-root refutation sets; that costs and
//! `Q*` recompute from the verified structure; and that the graph is acyclic and has
//! no unreachable nodes.

use std::collections::{BTreeSet, HashMap};

use recur64_core::{ActionId, GameState, Termination};

use super::audit::{Auditor, apply, key};
use super::targets::ProofPosition;
use super::trace::{Kind, NO_CHILD, PositionTrace};

/// Memo shared by one audit thread.
#[derive(Default)]
pub struct TraceAuditMemo(Auditor);

fn action(id: u16) -> ActionId {
    ActionId::from_index(u32::from(id)).expect("stored ActionId is in range")
}

/// Audit one trace against its source position. `Err` carries the disagreement.
pub fn audit_trace(
    t: &PositionTrace,
    p: &ProofPosition,
    memo: &mut TraceAuditMemo,
) -> Result<(), String> {
    let tag = |m: String| format!("{}: {m}", p.id);
    if t.id != p.id || t.fen != p.fen || t.family != p.family || t.mate_depth != p.mate_depth {
        return Err(tag("trace identity differs from its source position".into()));
    }
    let root_state = GameState::from_fen(&p.fen).map_err(|e| tag(format!("fen: {e}")))?;
    let legal = root_state.legal_actions();
    let stored_legal: Vec<u32> = p.legal.iter().map(|&i| u32::from(i)).collect();
    if legal.iter().map(|a| a.index()).collect::<Vec<_>>() != stored_legal {
        return Err(tag("root legal list differs from GameState".into()));
    }
    let d = p.mate_depth;
    if t.root as usize >= t.nodes.len() {
        return Err(tag("root index out of range".into()));
    }
    let root = &t.nodes[t.root as usize];
    if root.k != Kind::Or || root.n != d {
        return Err(tag(format!(
            "root node is {:?}/{} not Or/{d}",
            root.k, root.n
        )));
    }

    // Structural: indices in range, strictly decreasing (n, kind), no orphans.
    for (i, node) in t.nodes.iter().enumerate() {
        if node.alts.is_empty() {
            return Err(tag(format!("node {i} has no alternatives")));
        }
        for a in &node.alts {
            if node.k == Kind::And && a.mate {
                return Err(tag(format!("AND node {i} has a mate alternative")));
            }
            if a.mate {
                if a.c != NO_CHILD {
                    return Err(tag(format!("mating leaf at node {i} has a child")));
                }
                continue;
            }
            let Some(child) = t.nodes.get(a.c as usize) else {
                return Err(tag(format!("node {i}: child {} out of range", a.c)));
            };
            let ok = match node.k {
                // OR(n) -> AND(n-1); AND(k) -> OR(k): acyclic by construction.
                Kind::Or => child.k == Kind::And && child.n + 1 == node.n,
                Kind::And => child.k == Kind::Or && child.n == node.n,
            };
            if !ok {
                return Err(tag(format!(
                    "node {i} -> {} breaks the (n, kind) order",
                    a.c
                )));
            }
        }
    }
    let reach = t.reachable();
    if reach.len() != t.nodes.len() {
        return Err(tag(format!(
            "{} of {} nodes are unreachable from the root",
            t.nodes.len() - reach.len(),
            t.nodes.len()
        )));
    }

    // Semantic walk: each node is verified once, against the first state that
    // reaches it; every later arrival must be the same (position, n).
    let a = &mut memo.0;
    let mut first: HashMap<u32, (String, u8, Kind)> = HashMap::new();
    let mut verified: HashMap<u32, u64> = HashMap::new();
    let cost = walk(t, t.root, &root_state, a, &mut first, &mut verified, &tag)?;
    if cost != t.q_star || t.nodes[t.root as usize].cost != t.q_star {
        return Err(tag(format!(
            "Q* {} differs from the independently recomputed cost {cost}",
            t.q_star
        )));
    }

    // The root alternatives are exactly the stored correct set.
    let stored_correct: BTreeSet<u16> = p.correct.iter().map(|&i| p.legal[i as usize]).collect();
    let alts: BTreeSet<u16> = root.alts.iter().map(|x| x.a).collect();
    if stored_correct != alts {
        return Err(tag(
            "root alternatives differ from the stored correct set".into()
        ));
    }

    // Refutations of incorrect root moves, re-derived from scratch.
    let mut expected: Vec<(u16, Vec<u16>)> = Vec::new();
    for id in &legal {
        let ia = id.index() as u16;
        if stored_correct.contains(&ia) {
            continue;
        }
        let after = apply(&root_state, *id);
        if after.is_terminal() {
            continue;
        }
        let mut replies = Vec::new();
        for r in after.legal_actions() {
            let next = apply(&after, r);
            // Refuting: the attacker cannot force mate within the remaining d - 1.
            if next.is_terminal() || !(d >= 2 && a.attacker_wins(&next, d - 1)) {
                replies.push(r.index() as u16);
            }
        }
        if !replies.is_empty() {
            expected.push((ia, replies));
        }
    }
    let got: Vec<(u16, Vec<u16>)> = t
        .refutations
        .iter()
        .map(|r| (r.root_action, r.replies.clone()))
        .collect();
    if expected != got {
        return Err(tag(
            "refutation sets differ from the independent derivation".into(),
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn walk(
    t: &PositionTrace,
    idx: u32,
    state: &GameState,
    a: &mut Auditor,
    first: &mut HashMap<u32, (String, u8, Kind)>,
    verified: &mut HashMap<u32, u64>,
    tag: &dyn Fn(String) -> String,
) -> Result<u64, String> {
    let node = &t.nodes[idx as usize];
    let ident = (key(state), node.n, node.k);
    if let Some(prev) = first.get(&idx) {
        if *prev != ident {
            return Err(tag(format!(
                "node {idx} is shared by two different (position, n, kind): {prev:?} vs {ident:?}"
            )));
        }
        return Ok(verified[&idx]);
    }
    first.insert(idx, ident);
    let n = node.n;
    let cost = match node.k {
        Kind::Or => {
            // Independent win test of EVERY legal attacker move.
            let mut winners: Vec<(u16, bool)> = Vec::new();
            let mut afters = HashMap::new();
            for id in state.legal_actions() {
                let after = apply(state, id);
                let mate = after.termination() == Some(Termination::Checkmate);
                let wins = mate || (n >= 2 && a.defender_all(&after, n - 1));
                if wins {
                    winners.push((id.index() as u16, mate));
                    afters.insert(id.index() as u16, after);
                }
            }
            let claimed: Vec<(u16, bool)> = node.alts.iter().map(|x| (x.a, x.mate)).collect();
            if winners != claimed {
                return Err(tag(format!(
                    "OR node {idx} (n={n}) alternatives {claimed:?} differ from the independent winners {winners:?}"
                )));
            }
            let mut best = u64::MAX;
            for alt in &node.alts {
                let c = if alt.mate {
                    1
                } else {
                    let after = &afters[&alt.a];
                    1 + walk(t, alt.c, after, a, first, verified, tag)?
                };
                best = best.min(c);
            }
            best
        }
        Kind::And => {
            // EVERY legal reply must be present, in ActionId order.
            let replies: Vec<u16> = state
                .legal_actions()
                .iter()
                .map(|r| r.index() as u16)
                .collect();
            let claimed: Vec<u16> = node.alts.iter().map(|x| x.a).collect();
            if replies != claimed {
                return Err(tag(format!(
                    "AND node {idx}: replies {claimed:?} differ from the legal replies {replies:?} (a reply is missing or extra)"
                )));
            }
            let mut total = 0u64;
            for alt in &node.alts {
                let next = apply(state, action(alt.a));
                if next.is_terminal() || !a.attacker_wins(&next, n) {
                    return Err(tag(format!(
                        "AND node {idx}: after reply {} the attacker does not force mate within {n}",
                        alt.a
                    )));
                }
                total += 1 + walk(t, alt.c, &next, a, first, verified, tag)?;
            }
            total
        }
    };
    if cost != node.cost {
        return Err(tag(format!(
            "node {idx}: stored cost {} differs from the recomputed {cost}",
            node.cost
        )));
    }
    verified.insert(idx, cost);
    Ok(cost)
}
