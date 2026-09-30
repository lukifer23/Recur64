//! Independent audit of `ProofTargetsV1` labels.
//!
//! The generator ([`super::generator`]) solves on raw `cozy_chess` boards with the
//! memoizing [`super::mate::MateSolver`]. This audit re-derives every label
//! through a different path: `GameState` values, `GameState::apply` and its full
//! termination classification (every draw rule), `legal_actions()` ordering, and
//! its own memo keyed by position text. A label is accepted only if both paths
//! agree.

use std::collections::{HashMap, HashSet};

use recur64_core::{GameState, StandardMove, Termination, candidate_facts};

use super::generator::MAX_CORRECT_FRACTION;
use super::targets::{ProofPosition, ProofTargets};

fn apply(state: &GameState, id: recur64_core::ActionId) -> GameState {
    let (f, t, p) = id.to_physical(state.perspective());
    let mut next = state.clone();
    next.apply(StandardMove::new(f, t, (!p.is_none()).then_some(p)))
        .expect("a legal action applies");
    next
}

/// Position identity for the audit memo: placement + side (fresh-history contract).
fn key(state: &GameState) -> String {
    let fen = state.to_fen();
    let mut it = fen.split(' ');
    format!("{} {}", it.next().unwrap_or(""), it.next().unwrap_or(""))
}

#[derive(Default)]
struct Auditor {
    memo_attacker: HashMap<(String, u8), bool>,
    memo_defender: HashMap<(String, u8), bool>,
}

impl Auditor {
    /// The side to move forces checkmate within `n` of its own moves.
    fn attacker_wins(&mut self, s: &GameState, n: u8) -> bool {
        if n == 0 || s.is_terminal() {
            return false;
        }
        let k = (key(s), n);
        if let Some(&v) = self.memo_attacker.get(&k) {
            return v;
        }
        let mut wins = false;
        for id in s.legal_actions() {
            let after = apply(s, id);
            if after.termination() == Some(Termination::Checkmate)
                || (n >= 2 && self.defender_all(&after, n - 1))
            {
                wins = true;
                break;
            }
        }
        self.memo_attacker.insert(k, wins);
        wins
    }

    /// `after` has the defender to move; every reply leaves the attacker able to
    /// force mate within `n` more moves.
    fn defender_all(&mut self, after: &GameState, n: u8) -> bool {
        if n == 0 || after.is_terminal() {
            return false;
        }
        let k = (key(after), n);
        if let Some(&v) = self.memo_defender.get(&k) {
            return v;
        }
        let replies = after.legal_actions();
        let mut all = !replies.is_empty();
        for id in replies {
            let next = apply(after, id);
            if next.is_terminal() || !self.attacker_wins(&next, n) {
                all = false;
                break;
            }
        }
        self.memo_defender.insert(k, all);
        all
    }
}

/// Audit one position; `Err` carries the disagreement.
pub fn audit_position(p: &ProofPosition, auditor_memo: &mut AuditMemo) -> Result<(), String> {
    let state = GameState::from_fen(&p.fen).map_err(|e| format!("{}: fen: {e}", p.id))?;
    let clock_zero = state.to_fen().split(' ').nth(4) == Some("0");
    if !clock_zero || state.is_terminal() || state.side_to_move() != recur64_core::Color::White {
        return Err(format!(
            "{}: violates fresh_no_history_v1 (clock, terminal or side)",
            p.id
        ));
    }
    let legal = state.legal_actions();
    let stored: Vec<u32> = p.legal.iter().map(|&i| i as u32).collect();
    let actual: Vec<u32> = legal.iter().map(|a| a.index()).collect();
    if stored != actual {
        return Err(format!(
            "{}: legal action list differs from GameState",
            p.id
        ));
    }
    let a = &mut auditor_memo.0;
    let d = p.mate_depth;
    if !a.attacker_wins(&state, d) {
        return Err(format!("{}: no forced mate within {d}", p.id));
    }
    if d >= 2 && a.attacker_wins(&state, d - 1) {
        return Err(format!("{}: a forced mate exists within {}", p.id, d - 1));
    }
    let mut correct = Vec::new();
    for (i, id) in legal.iter().enumerate() {
        let after = apply(&state, *id);
        let wins = if after.termination() == Some(Termination::Checkmate) {
            d == 1
        } else {
            d >= 2 && a.defender_all(&after, d - 1)
        };
        if wins {
            correct.push(i as u32);
        }
    }
    if correct != p.correct {
        return Err(format!(
            "{}: correct set {:?} differs from the audit {:?}",
            p.id, p.correct, correct
        ));
    }
    let chance = correct.len() as f32 / legal.len() as f32;
    if (chance - p.chance_top1).abs() > 1e-6 {
        return Err(format!("{}: chance mismatch", p.id));
    }
    if d >= 2 {
        if chance > MAX_CORRECT_FRACTION + 1e-6 {
            return Err(format!(
                "{}: correct fraction {chance} above the filter",
                p.id
            ));
        }
        let facts = candidate_facts(&state);
        let idx: Vec<usize> = correct.iter().map(|&c| c as usize).collect();
        let ambiguous = idx
            .iter()
            .any(|&c| (0..facts.len()).any(|w| !idx.contains(&w) && facts[w] == facts[c]));
        if !ambiguous {
            return Err(format!(
                "{}: no correct move shares facts with an incorrect one",
                p.id
            ));
        }
    }
    Ok(())
}

/// Memo reused across positions of one audit thread.
#[derive(Default)]
pub struct AuditMemo(Auditor);

/// Outcome of auditing a whole dataset.
#[derive(Debug, Default, serde::Serialize)]
pub struct AuditReport {
    pub checked: usize,
    pub failures: Vec<String>,
}

/// Audit every position of `t` (parallel, independent memo per thread).
pub fn audit_targets(t: &ProofTargets, threads: usize) -> AuditReport {
    let threads = threads.max(1);
    let chunk = t.positions.len().div_ceil(threads).max(1);
    let parts: Vec<Vec<String>> = std::thread::scope(|scope| {
        let hs: Vec<_> = t
            .positions
            .chunks(chunk)
            .map(|c| {
                scope.spawn(move || {
                    let mut memo = AuditMemo::default();
                    c.iter()
                        .filter_map(|p| audit_position(p, &mut memo).err())
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        hs.into_iter()
            .map(|h| h.join().expect("audit thread"))
            .collect()
    });
    AuditReport {
        checked: t.positions.len(),
        failures: parts.into_iter().flatten().collect(),
    }
}

/// Hard-disjointness across splits: no exact FEN and no symmetry-canonical key
/// appears in more than one dataset. `Err` names the first overlap.
pub fn check_disjoint(sets: &[&ProofTargets]) -> Result<(), String> {
    let mut fens: HashMap<&str, &str> = HashMap::new();
    let mut canons: HashMap<&str, &str> = HashMap::new();
    for t in sets {
        let mut own_fens: HashSet<&str> = HashSet::new();
        let mut own_canons: HashSet<&str> = HashSet::new();
        for p in &t.positions {
            if let Some(other) = fens.get(p.fen.as_str())
                && *other != t.split.label()
            {
                return Err(format!(
                    "exact FEN overlap {} between {other} and {}",
                    p.fen,
                    t.split.label()
                ));
            }
            if let Some(other) = canons.get(p.canon.as_str())
                && *other != t.split.label()
            {
                return Err(format!(
                    "canonical overlap {} between {other} and {}",
                    p.id,
                    t.split.label()
                ));
            }
            own_fens.insert(&p.fen);
            own_canons.insert(&p.canon);
        }
        for f in own_fens {
            fens.insert(f, t.split.label());
        }
        for c in own_canons {
            canons.insert(c, t.split.label());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::generator::{GenSpec, generate};
    use super::super::targets::Split;
    use super::*;

    fn small(split: Split, seed: u64, forbid: &[&ProofTargets]) -> ProofTargets {
        let mut spec = GenSpec {
            split,
            seed,
            per_cell: 2,
            depths: vec![1, 2],
            threads: 2,
            max_tries_per_thread: 5_000_000,
            forbid_fen: HashSet::new(),
            forbid_canon: HashSet::new(),
        };
        for t in forbid {
            for p in &t.positions {
                spec.forbid_fen.insert(p.fen.clone());
                spec.forbid_canon.insert(p.canon.clone());
            }
        }
        generate(&spec).unwrap().0
    }

    #[test]
    fn generated_labels_pass_the_independent_audit_and_splits_are_disjoint() {
        let confirm = small(Split::Confirm, 21, &[]);
        let train = small(Split::Train, 22, &[&confirm]);
        for t in [&confirm, &train] {
            let r = audit_targets(t, 2);
            assert!(r.failures.is_empty(), "{:?}", r.failures);
            assert_eq!(r.checked, t.positions.len());
        }
        check_disjoint(&[&confirm, &train]).unwrap();
    }

    #[test]
    fn a_corrupted_label_is_caught() {
        let mut t = small(Split::Confirm, 31, &[]);
        let p = t
            .positions
            .iter_mut()
            .find(|p| p.mate_depth == 2)
            .expect("a depth-2 position");
        // Claim one more correct move than the truth.
        let extra = (0..p.legal.len() as u32)
            .find(|i| !p.correct.contains(i))
            .unwrap();
        p.correct.push(extra);
        p.correct.sort_unstable();
        p.chance_top1 = p.correct.len() as f32 / p.legal.len() as f32;
        let r = audit_targets(&t, 1);
        assert!(
            !r.failures.is_empty(),
            "the audit must reject a wrong label"
        );
    }

    #[test]
    fn overlapping_splits_are_refused() {
        let a = small(Split::Confirm, 41, &[]);
        let mut b = small(Split::Train, 42, &[&a]);
        b.positions.push(ProofPosition {
            split: Split::Train,
            id: "dup".into(),
            ..a.positions[0].clone()
        });
        assert!(check_disjoint(&[&a, &b]).is_err());
    }
}
