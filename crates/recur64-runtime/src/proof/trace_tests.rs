//! Tests for `ProofTraceV1`: structure, the exact cost, the set-valued target, the
//! independent audit, sharded storage, the frozen threshold and the custody guards.
//!
//! Fixtures are found by a deterministic seeded search over real exact positions;
//! nothing here is canned. A fixture that cannot be found within the search bound
//! fails the test rather than weakening it.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::PathBuf;

use cozy_chess::GameStatus;
use recur64_core::GameState;
use recur64_model::active::{QueryScript, Tree};
use recur64_statequery::QueryManager;

use super::custody::{
    ConfirmAuthorization, HOLDOUT_C_DIGEST, load_sealed_confirmation, load_working_split,
    verify_holdout_c,
};
use super::generator::{Rng, canonical_key, legal_cozy_moves, sample_position};
use super::mate::MateSolver;
use super::targets::{ProofPosition, ProofTargets, Split};
use super::trace::*;
use super::trace_audit::{TraceAuditMemo, audit_trace};
use super::trace_store::{
    AuditManifest, TraceManifest, audit_traces, ensure_traceable, feasibility, generate_traces,
    threshold_pass,
};
use super::trace_teacher::{ProofTraceTeacher, edge_path, node_paths};

const KQR: &[char] = &['K', 'Q', 'R'];
const KQ: &[char] = &['K', 'Q'];
const KR: &[char] = &['K', 'R'];

fn position(id: &str, fen: &str, family: &str, solver: &mut MateSolver) -> Option<ProofPosition> {
    let state = GameState::from_fen(fen).ok()?;
    let d = solver.mate_depth(state.board(), 3)?;
    let moves = legal_cozy_moves(&state);
    let correct = solver.correct_moves(state.board(), d, &moves);
    let legal: Vec<u16> = state
        .legal_actions()
        .iter()
        .map(|a| a.index() as u16)
        .collect();
    Some(ProofPosition {
        id: id.to_string(),
        fen: fen.to_string(),
        split: Split::Train,
        family: family.to_string(),
        mate_depth: d,
        canon: canonical_key(fen),
        chance_top1: correct.len() as f32 / legal.len() as f32,
        legal,
        correct: correct.iter().map(|&i| i as u32).collect(),
        generator_seed: 0,
    })
}

/// Deterministic search for a position of `depth` whose trace satisfies `pred`.
fn find(
    white: &[char],
    family: &str,
    depth: u8,
    seed: u64,
    tries: usize,
    pred: impl Fn(&PositionTrace) -> bool,
) -> (ProofPosition, PositionTrace) {
    let mut rng = Rng(seed);
    let mut solver = MateSolver::new();
    for i in 0..tries {
        let Some((fen, _)) = sample_position(&mut rng, white) else {
            continue;
        };
        if solver.table_size() > 2_000_000 {
            solver = MateSolver::new();
        }
        let Some(p) = position(
            &format!("fx-{family}-{depth}-{i}"),
            &fen,
            family,
            &mut solver,
        ) else {
            continue;
        };
        if p.mate_depth != depth {
            continue;
        }
        let t = build_trace(&mut solver, &p).expect("trace builds");
        if pred(&t) {
            return (p, t);
        }
    }
    panic!("no {family} M{depth} fixture satisfying the predicate within {tries} samples");
}

fn small(t: &PositionTrace) -> bool {
    count_certificates(t) <= 6_000
}

fn and_nodes(t: &PositionTrace) -> impl Iterator<Item = &Node> {
    t.reachable()
        .into_iter()
        .map(|i| &t.nodes[i as usize])
        .filter(|n| n.k == Kind::And)
}

fn or_nodes(t: &PositionTrace) -> impl Iterator<Item = &Node> {
    t.reachable()
        .into_iter()
        .map(|i| &t.nodes[i as usize])
        .filter(|n| n.k == Kind::Or)
}

/// A random prefix-closed set of known edges: proof edges, plus incorrect root
/// moves and their replies (waste queries).
fn random_s(t: &PositionTrace, rng: &mut Rng, k: usize) -> HashSet<Path> {
    let mut queried: HashSet<Path> = HashSet::new();
    // path -> node index for queried nodes that lead somewhere in the proof graph.
    let mut at: HashMap<Path, u32> = HashMap::from([(Vec::new(), t.root)]);
    for _ in 0..k {
        let mut cands: Vec<(Path, Option<u32>)> = Vec::new();
        for (p, &x) in &at {
            for alt in &t.nodes[x as usize].alts {
                let mut e = p.clone();
                e.push(alt.a);
                if !queried.contains(&e) {
                    cands.push((e, (!alt.mate).then_some(alt.c)));
                }
            }
        }
        for r in &t.refutations {
            let m: Path = vec![r.root_action];
            if !queried.contains(&m) {
                cands.push((m.clone(), None));
            } else {
                for &reply in &r.replies {
                    let e: Path = vec![r.root_action, reply];
                    if !queried.contains(&e) {
                        cands.push((e, None));
                    }
                }
            }
        }
        if cands.is_empty() {
            break;
        }
        cands.sort();
        let (e, node) = cands[(rng.next_u64() % cands.len() as u64) as usize].clone();
        queried.insert(e.clone());
        if let Some(c) = node {
            at.insert(e, c);
        }
    }
    queried
}

// ---------------------------------------------------------------------------
// Structure, cost and the set-valued target
// ---------------------------------------------------------------------------

#[test]
fn q_star_equals_the_smallest_explicitly_enumerated_certificate() {
    let mut checked = 0;
    for (white, fam, depth, seed) in [
        (KQ, "KQvK", 1u8, 11u64),
        (KR, "KRvK", 2, 12),
        (KQ, "KQvK", 2, 13),
        (KQR, "KQRvK", 2, 14),
    ] {
        for off in 0..3 {
            let (_, t) = find(white, fam, depth, seed + off * 100, 6_000, small);
            let certs = reference_certificates(&t);
            let min = certs.iter().map(BTreeSet::len).min().unwrap() as u64;
            assert_eq!(t.q_star, min, "{} {}", t.id, t.fen);
            assert_eq!(t.nodes[t.root as usize].cost, t.q_star);
            checked += 1;
        }
    }
    assert!(checked >= 12);
}

#[test]
fn the_dp_target_equals_the_definition_on_random_queried_sets() {
    let mut rng = Rng(2024);
    let mut sets = 0;
    let mut nonempty = 0;
    for (white, fam, depth, seed) in [
        (KQ, "KQvK", 1u8, 21u64),
        (KQ, "KQvK", 2, 22),
        (KR, "KRvK", 2, 23),
        (KQR, "KQRvK", 2, 24),
        (KQR, "KQRvK", 1, 25),
    ] {
        for off in 0..2 {
            let (_, t) = find(white, fam, depth, seed + off * 100, 6_000, small);
            for k in 0..14 {
                for _ in 0..6 {
                    let s = random_s(&t, &mut rng, k);
                    let got = admissible_for_test(&t, &s);
                    let want = reference_admissible(&t, &s);
                    assert_eq!(got, want, "{} k={k} S={s:?}", t.id);
                    sets += 1;
                    nonempty += usize::from(!got.is_empty());
                }
            }
        }
    }
    assert!(sets >= 500, "{sets}");
    assert!(nonempty >= 250, "coverage: {nonempty} non-empty targets");
}

fn admissible_for_test(t: &PositionTrace, s: &HashSet<Path>) -> Admissible {
    t.admissible(s)
}

#[test]
fn the_empty_set_admits_exactly_the_minimal_certificate_root_edges() {
    let (_, t) = find(KQR, "KQRvK", 2, 31, 6_000, small);
    let a = t.admissible(&HashSet::new());
    assert!(
        a.refute.is_empty(),
        "nothing is queried, so nothing is refutable yet"
    );
    // Only root edges are on the frontier, each a tied-minimum alternative.
    let min = t.nodes[t.root as usize]
        .alts
        .iter()
        .map(|alt| {
            1 + if alt.mate {
                0
            } else {
                t.nodes[alt.c as usize].cost
            }
        })
        .min()
        .unwrap();
    let want: BTreeSet<Path> = t.nodes[t.root as usize]
        .alts
        .iter()
        .filter(|alt| {
            1 + if alt.mate {
                0
            } else {
                t.nodes[alt.c as usize].cost
            } == min
        })
        .map(|alt| vec![alt.a])
        .collect();
    assert_eq!(a.proof, want);
    assert!(a.proof.iter().all(|p| p.len() == 1));
}

#[test]
fn required_fixture_structures_exist_and_behave() {
    // M1 with several correct root moves (interchangeable mating leaves).
    let (_, m1) = find(KQ, "KQvK", 1, 41, 6_000, |t| {
        t.nodes[t.root as usize].alts.len() >= 2
    });
    assert!(m1.nodes[m1.root as usize].alts.iter().all(|a| a.mate));
    assert_eq!(m1.q_star, 1);
    let a = m1.admissible(&HashSet::new());
    assert_eq!(
        a.proof.len(),
        m1.nodes[m1.root as usize].alts.len(),
        "every correct root move is admissible"
    );

    // M2 with multiple tied attacker alternatives at the root.
    let tied = |t: &PositionTrace| {
        let costs: Vec<u64> = t.nodes[t.root as usize]
            .alts
            .iter()
            .map(|alt| {
                1 + if alt.mate {
                    0
                } else {
                    t.nodes[alt.c as usize].cost
                }
            })
            .collect();
        let min = *costs.iter().min().unwrap();
        costs.iter().filter(|&&c| c == min).count() >= 2
    };
    let (_, m2) = find(KQR, "KQRvK", 2, 42, 20_000, tied);
    assert!(m2.admissible(&HashSet::new()).proof.len() >= 2);

    // A defender node with several replies, some interchangeable (equal cost).
    let interchangeable = |t: &PositionTrace| {
        and_nodes(t).any(|n| {
            n.alts.len() >= 2 && {
                let costs: Vec<u64> = n.alts.iter().map(|a| t.nodes[a.c as usize].cost).collect();
                costs
                    .iter()
                    .enumerate()
                    .any(|(i, c)| costs[i + 1..].contains(c))
            }
        })
    };
    let (_, m3) = find(KQR, "KQRvK", 3, 43, 40_000, interchangeable);
    assert_eq!(m3.mate_depth, 3);
    let widest = and_nodes(&m3).map(|n| n.alts.len()).max().unwrap();
    assert!(widest >= 2);
    // The certificate must resolve EVERY reply: its size exceeds the number of
    // replies of any single defender node.
    assert!(m3.q_star as usize > widest);

    // An incorrect root move with several valid refutations.
    let (_, multi_ref) = find(KQR, "KQRvK", 2, 44, 20_000, |t| {
        t.refutations.iter().any(|r| r.replies.len() >= 2)
    });
    let r = multi_ref
        .refutations
        .iter()
        .find(|r| r.replies.len() >= 2)
        .unwrap();
    let s: HashSet<Path> = HashSet::from([vec![r.root_action]]);
    let a = multi_ref.admissible(&s);
    for &reply in &r.replies {
        assert!(
            a.refute.contains(&vec![r.root_action, reply]),
            "every refutation is admissible"
        );
    }

    // A terminal mate edge exists in every proof (a leaf with no child).
    assert!(or_nodes(&m3).any(|n| n.alts.iter().any(|a| a.mate && a.c == NO_CHILD)));
}

#[test]
fn a_stalemating_move_is_neither_a_win_nor_a_refutation_target() {
    let stale = |p: &ProofPosition| {
        let b: cozy_chess::Board = p.fen.parse().unwrap();
        let mut found = Vec::new();
        b.generate_moves(|ms| {
            for m in ms {
                let mut a = b.clone();
                a.play_unchecked(m);
                if a.status() == GameStatus::Drawn {
                    found.push(m);
                }
            }
            false
        });
        !found.is_empty()
    };
    let mut rng = Rng(51);
    let mut solver = MateSolver::new();
    for i in 0..60_000 {
        let Some((fen, _)) = sample_position(&mut rng, KQ) else {
            continue;
        };
        let Some(p) = position(&format!("st-{i}"), &fen, "KQvK", &mut solver) else {
            continue;
        };
        if p.mate_depth > 2 || !stale(&p) {
            continue;
        }
        let t = build_trace(&mut solver, &p).unwrap();
        let state = GameState::from_fen(&p.fen).unwrap();
        let b = state.board();
        for (idx, id) in state.legal_actions().iter().enumerate() {
            let mv = legal_cozy_moves(&state)[idx];
            let mut after = b.clone();
            after.play_unchecked(mv);
            if after.status() == GameStatus::Drawn {
                let a = id.index() as u16;
                assert!(
                    t.nodes[t.root as usize].alts.iter().all(|x| x.a != a),
                    "stalemate is not a win"
                );
                assert!(
                    t.refutations.iter().all(|r| r.root_action != a),
                    "a stalemated child has no replies"
                );
                return;
            }
        }
    }
    panic!("no fixture with a stalemating root move");
}

#[test]
fn trace_edges_are_path_specific_not_state_keyed() {
    // A node of the stored DAG reached by two different root-relative paths
    // (a transposition-like pair) must remain two distinct trace edges.
    let shared = |t: &PositionTrace| {
        let mut indeg: HashMap<u32, usize> = HashMap::new();
        for i in t.reachable() {
            for a in &t.nodes[i as usize].alts {
                if !a.mate {
                    *indeg.entry(a.c).or_default() += 1;
                }
            }
        }
        indeg.values().any(|&d| d >= 2)
    };
    let (_, t) = find(KQR, "KQRvK", 3, 61, 40_000, shared);
    let mut indeg: HashMap<u32, usize> = HashMap::new();
    for i in t.reachable() {
        for a in &t.nodes[i as usize].alts {
            if !a.mate {
                *indeg.entry(a.c).or_default() += 1;
            }
        }
    }
    let target = *indeg
        .iter()
        .find(|(_, d)| **d >= 2)
        .map(|(n, _)| n)
        .unwrap();
    // Enumerate every action path to `target`.
    fn paths_to(t: &PositionTrace, x: u32, target: u32, path: &mut Path, out: &mut Vec<Path>) {
        if x == target {
            out.push(path.clone());
        }
        for a in &t.nodes[x as usize].alts {
            if a.mate {
                continue;
            }
            path.push(a.a);
            paths_to(t, a.c, target, path, out);
            path.pop();
        }
    }
    let mut paths = Vec::new();
    paths_to(&t, t.root, target, &mut Vec::new(), &mut paths);
    assert!(
        paths.len() >= 2,
        "{} paths reach the shared node",
        paths.len()
    );
    let distinct: BTreeSet<&Path> = paths.iter().collect();
    assert_eq!(
        distinct.len(),
        paths.len(),
        "the paths are distinct identities"
    );
    // Admissible edges below the shared node are reported under the path that was
    // actually queried, never under the other one.
    let p1 = &paths[0];
    let s: HashSet<Path> = (1..=p1.len()).map(|k| p1[..k].to_vec()).collect();
    let a = t.admissible(&s).all();
    for e in &a {
        let parent = &e[..e.len() - 1];
        assert!(
            parent.is_empty() || s.contains(parent),
            "{e:?} hangs off an unqueried edge"
        );
    }
}

// ---------------------------------------------------------------------------
// Order invariance and the live adapter
// ---------------------------------------------------------------------------

/// A random topological order of the prefix-closed set `s`.
fn random_order(s: &HashSet<Path>, rng: &mut Rng) -> Vec<Path> {
    let mut placed: HashSet<Path> = HashSet::new();
    let mut order = Vec::new();
    while order.len() < s.len() {
        let mut ready: Vec<&Path> = s
            .iter()
            .filter(|p| {
                !placed.contains(*p) && (p.len() == 1 || placed.contains(&p[..p.len() - 1]))
            })
            .collect();
        ready.sort();
        let p = ready[(rng.next_u64() % ready.len() as u64) as usize].clone();
        placed.insert(p.clone());
        order.push(p);
    }
    order
}

fn replay(root: &GameState, order: &[Path]) -> (QueryManager, Tree) {
    let mut m = QueryManager::new(root.clone()).unwrap();
    let mut tree = Tree::new(&m.packet(0).unwrap()).unwrap();
    let mut slots: HashMap<Path, usize> = HashMap::from([(Vec::new(), 0)]);
    for p in order {
        let slot = slots[&p[..p.len() - 1].to_vec()];
        let edge = tree
            .frontier()
            .into_iter()
            .find(|e| e.node_slot == slot && e.action == *p.last().unwrap())
            .unwrap_or_else(|| panic!("edge {p:?} is not on the live frontier"));
        let pkt = m.query(tree.node(slot).id, edge.action).unwrap();
        let new_slot = tree.add_child(&edge, &pkt).unwrap();
        slots.insert(p.clone(), new_slot);
    }
    (m, tree)
}

#[test]
fn the_target_depends_on_the_queried_set_not_on_query_order() {
    let mut rng = Rng(77);
    let mut permutations = 0;
    for (white, fam, depth, seed) in [
        (KQR, "KQRvK", 2u8, 71u64),
        (KR, "KRvK", 3, 72),
        (KQR, "KQRvK", 3, 73),
    ] {
        let (p, t) = find(white, fam, depth, seed, 40_000, |t| t.q_star <= 40);
        let root = GameState::from_fen(&p.fen).unwrap();
        for k in [0usize, 1, 3, 6, 10] {
            let s = random_s(&t, &mut rng, k);
            let direct = t.admissible(&s).all();
            let mut seen: Option<BTreeSet<Path>> = None;
            for _ in 0..8 {
                let order = random_order(&s, &mut rng);
                let (_m, tree) = replay(&root, &order);
                let paths = node_paths(&tree);
                let mut teacher = ProofTraceTeacher {
                    traces: std::slice::from_ref(&t),
                };
                let frontier = tree.frontier();
                let step = teacher.next(0, 0, &frontier, &tree).unwrap();
                let got: BTreeSet<Path> = step
                    .targets
                    .iter()
                    .map(|&i| edge_path(&paths, &frontier[i]))
                    .collect();
                assert_eq!(
                    got, direct,
                    "{} k={k}: target changed with query order",
                    t.id
                );
                if let Some(prev) = &seen {
                    assert_eq!(prev, &got);
                }
                seen = Some(got);
                permutations += 1;
            }
        }
    }
    assert!(permutations >= 100);
}

/// Scope of the claim (clarified in P4.1): "completion in exactly Q* queries"
/// holds for an ON-PROOF trajectory that starts from the EMPTY queried set and
/// follows only `A_proof` choices, each of which removes exactly one residual
/// edge. It is not a statement about an arbitrary queried set: prior off-proof or
/// refutation queries do not reduce the proof residual (see the next test).
#[test]
fn an_on_proof_trajectory_from_the_empty_set_completes_in_exactly_q_star_queries() {
    for (white, fam, depth, seed, take_last) in [
        (KQ, "KQvK", 2u8, 81u64, false),
        (KQR, "KQRvK", 2, 82, true),
        (KR, "KRvK", 3, 83, false),
        (KQR, "KQRvK", 3, 84, true),
    ] {
        let (p, t) = find(white, fam, depth, seed, 40_000, |t| t.q_star <= 60);
        let root = GameState::from_fen(&p.fen).unwrap();
        let mut order: Vec<Path> = Vec::new();
        let mut steps = 0u64;
        loop {
            let (_m, tree) = replay(&root, &order);
            let s: HashSet<Path> = order.iter().cloned().collect();
            if t.is_complete(&s) {
                break;
            }
            let frontier = tree.frontier();
            let paths = node_paths(&tree);
            let mut teacher = ProofTraceTeacher {
                traces: std::slice::from_ref(&t),
            };
            let step = teacher.next(0, steps as usize, &frontier, &tree).unwrap();
            assert!(
                !step.targets.is_empty(),
                "an incomplete proof always has admissible edges"
            );
            // Every target is a real frontier edge, and the probabilities sum to 1.
            assert!(step.targets.iter().all(|&i| i < frontier.len()));
            let p_each = 1.0 / step.targets.len() as f64;
            assert!((p_each * step.targets.len() as f64 - 1.0).abs() < 1e-12);
            // Choosing ANY admissible edge is legitimate: take the first or the last.
            let pick = if take_last {
                *step.targets.last().unwrap()
            } else {
                step.follow
            };
            assert!(step.targets.contains(&pick));
            order.push(edge_path(&paths, &frontier[pick]));
            steps += 1;
            assert!(
                steps <= t.q_star,
                "more queries than the minimal certificate"
            );
        }
        assert_eq!(
            steps, t.q_star,
            "{}: each admissible query removes exactly one residual edge",
            t.id
        );
    }
}

/// The other side of the clarification: refutation queries are useful off-policy
/// supervision but never reduce the correct proof's residual, so a trajectory with
/// off-proof prefix queries is NOT Q* queries long.
#[test]
fn off_proof_queries_do_not_reduce_the_residual_and_lengthen_the_trajectory() {
    let (_, t) = find(KQR, "KQRvK", 2, 95, 20_000, |t| {
        t.refutations.iter().any(|r| r.replies.len() >= 2)
    });
    let r = t.refutations.iter().find(|r| r.replies.len() >= 2).unwrap();
    let mut s: HashSet<Path> = HashSet::new();
    assert_eq!(t.residual(&s), t.q_star);
    // Query an incorrect root move, then one of its refutations.
    s.insert(vec![r.root_action]);
    assert_eq!(
        t.residual(&s),
        t.q_star,
        "an incorrect root edge removes no proof edge"
    );
    let a = t.admissible(&s);
    assert!(
        !a.refute.is_empty(),
        "refutation edges are admissible after that query"
    );
    s.insert(vec![r.root_action, r.replies[0]]);
    assert_eq!(t.residual(&s), t.q_star, "nor does its refutation");
    // Finish the proof from there by following A_proof only.
    let mut proof_queries = 0u64;
    while !t.is_complete(&s) {
        let e = t
            .admissible(&s)
            .proof
            .iter()
            .next()
            .cloned()
            .expect("proof edges remain");
        s.insert(e);
        proof_queries += 1;
    }
    assert_eq!(
        proof_queries, t.q_star,
        "the proof itself still needs exactly Q* proof edges"
    );
    assert_eq!(
        s.len() as u64,
        t.q_star + 2,
        "total trajectory length is Q* plus the two off-proof queries"
    );
}

#[test]
fn once_the_proof_is_complete_the_teacher_gives_no_target_but_still_follows_a_legal_edge() {
    let (p, t) = find(KQ, "KQvK", 1, 91, 6_000, |_| true);
    let root = GameState::from_fen(&p.fen).unwrap();
    let mate: Path = vec![t.nodes[t.root as usize].alts[0].a];
    let (_m, tree) = replay(&root, &[mate]);
    let mut teacher = ProofTraceTeacher {
        traces: std::slice::from_ref(&t),
    };
    let frontier = tree.frontier();
    let step = teacher.next(0, 1, &frontier, &tree).unwrap();
    assert!(step.targets.is_empty());
    assert!(step.follow < frontier.len());
}

// ---------------------------------------------------------------------------
// Independent audit
// ---------------------------------------------------------------------------

#[test]
fn the_audit_accepts_generated_traces_and_rejects_every_corruption() {
    let (p, t) = find(KQR, "KQRvK", 2, 101, 20_000, |t| {
        !t.refutations.is_empty() && and_nodes(t).count() >= 1
    });
    audit_corruption_suite(&p, &t);
}

/// The cell that decides the frozen gate is mate-in-3: the same ten corruptions
/// must be rejected on a real KQRvK M3 trace.
#[test]
fn the_audit_rejects_every_corruption_of_a_mate_in_three_trace() {
    let (p, t) = find(KQR, "KQRvK", 3, 102, 40_000, |t| {
        t.refutations.iter().any(|r| r.replies.len() >= 2) && and_nodes(t).count() >= 2
    });
    assert_eq!(t.mate_depth, 3);
    audit_corruption_suite(&p, &t);
}

fn audit_corruption_suite(p: &ProofPosition, t: &PositionTrace) {
    let (p, t) = (p.clone(), t.clone());
    let mut memo = TraceAuditMemo::default();
    audit_trace(&t, &p, &mut memo).expect("a generated trace passes the independent audit");

    let first_and = t.nodes[t.root as usize]
        .alts
        .iter()
        .find(|a| !a.mate)
        .map(|a| a.c as usize)
        .expect("an AND node below the root");
    type Corruption = Box<dyn Fn(&mut PositionTrace)>;
    let corruptions: Vec<(&str, Corruption)> = vec![
        (
            "omitted defender reply",
            Box::new(move |t| {
                t.nodes[first_and].alts.pop();
            }),
        ),
        (
            "extra non-winning attacker alternative",
            Box::new(|t| {
                let missing = (0..t.nodes.len() as u32)
                    .find(|&i| t.nodes[i as usize].k == Kind::And)
                    .unwrap();
                let used: HashSet<u16> =
                    t.nodes[t.root as usize].alts.iter().map(|a| a.a).collect();
                let extra = (0u16..20_480).find(|a| !used.contains(a)).unwrap();
                let n = t.nodes[t.root as usize].n;
                let child = (0..t.nodes.len() as u32)
                    .find(|&i| t.nodes[i as usize].k == Kind::And && t.nodes[i as usize].n + 1 == n)
                    .unwrap_or(missing);
                t.nodes[t.root as usize].alts.push(Alt {
                    a: extra,
                    mate: false,
                    c: child,
                });
            }),
        ),
        (
            "flipped mate flag",
            Box::new(|t| {
                if let Some(a) = t
                    .nodes
                    .iter_mut()
                    .flat_map(|n| n.alts.iter_mut())
                    .find(|a| a.mate)
                {
                    a.mate = false;
                    a.c = 0;
                }
            }),
        ),
        ("wrong Q*", Box::new(|t| t.q_star += 1)),
        (
            "wrong node cost",
            Box::new(move |t| t.nodes[first_and].cost += 1),
        ),
        (
            "dropped refutation entry",
            Box::new(|t| {
                t.refutations.pop();
            }),
        ),
        (
            "shortened refutation set",
            Box::new(|t| {
                if let Some(r) = t.refutations.iter_mut().find(|r| r.replies.len() >= 2) {
                    r.replies.pop();
                } else {
                    t.refutations[0].replies.clear();
                }
            }),
        ),
        (
            "orphan node",
            Box::new(|t| {
                let extra = t.nodes[t.root as usize].clone();
                t.nodes.push(extra);
            }),
        ),
        ("wrong source id", Box::new(|t| t.id.push('x'))),
        ("wrong mate depth", Box::new(|t| t.mate_depth += 1)),
    ];
    for (name, corrupt) in corruptions {
        let mut bad = t.clone();
        corrupt(&mut bad);
        let mut memo = TraceAuditMemo::default();
        assert!(
            audit_trace(&bad, &p, &mut memo).is_err(),
            "the independent audit accepted a corrupted trace: {name}"
        );
    }
}

#[test]
fn the_audit_cross_checks_the_stored_correct_set() {
    let (mut p, t) = find(KQ, "KQvK", 2, 111, 6_000, |_| true);
    let mut memo = TraceAuditMemo::default();
    audit_trace(&t, &p, &mut memo).unwrap();
    // A source position whose labels were tampered with no longer matches the trace.
    let extra = (0..p.legal.len() as u32)
        .find(|i| !p.correct.contains(i))
        .unwrap();
    p.correct.push(extra);
    p.correct.sort_unstable();
    assert!(audit_trace(&t, &p, &mut TraceAuditMemo::default()).is_err());
}

// ---------------------------------------------------------------------------
// Storage, determinism, resume, feasibility
// ---------------------------------------------------------------------------

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("recur64_trace_{name}"));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn tiny_source() -> ProofTargets {
    let mut solver = MateSolver::new();
    let mut positions = Vec::new();
    let mut rng = Rng(5150);
    let mut seen = HashSet::new();
    let mut i = 0;
    while positions.len() < 7 {
        let white = [KQ, KR, KQR][positions.len() % 3];
        let Some((fen, _)) = sample_position(&mut rng, white) else {
            continue;
        };
        if !seen.insert(canonical_key(&fen)) {
            continue;
        }
        let fam = match white.len() {
            2 if white[1] == 'Q' => "KQvK",
            2 => "KRvK",
            _ => "KQRvK",
        };
        if let Some(p) = position(&format!("tiny-{i}"), &fen, fam, &mut solver)
            && p.mate_depth <= 2
        {
            positions.push(p);
        }
        i += 1;
    }
    ProofTargets::new(
        Split::Train,
        1,
        serde_json::json!({"fixture": "tiny"}),
        positions,
    )
}

#[test]
fn shards_are_deterministic_across_thread_counts_and_resumable() {
    let src = tiny_source();
    let (a, b) = (tmp("det_a"), tmp("det_b"));
    let ma = generate_traces(&src, &a, 3, 1, &|_, _, _| {}).unwrap();
    let mb = generate_traces(&src, &b, 3, 4, &|_, _, _| {}).unwrap();
    assert_eq!(
        ma.manifest_digest, mb.manifest_digest,
        "thread count must not change the output"
    );
    assert_eq!(ma.shards.len(), 3);
    for r in &ma.shards {
        assert_eq!(
            std::fs::read(a.join(&r.file)).unwrap(),
            std::fs::read(b.join(&r.file)).unwrap(),
            "shard {} differs between 1 and 4 threads",
            r.index
        );
    }
    // Resume: delete one shard, rerun; the others are validated and reused.
    std::fs::remove_file(a.join(&ma.shards[1].file)).unwrap();
    let reused = std::cell::RefCell::new(Vec::new());
    let again =
        generate_traces(&src, &a, 3, 2, &|i, _, r| reused.borrow_mut().push((i, r))).unwrap();
    assert_eq!(again.manifest_digest, ma.manifest_digest);
    assert_eq!(*reused.borrow(), vec![(0, true), (1, false), (2, true)]);
}

#[test]
fn corrupted_or_mismatched_shards_refuse_instead_of_being_reused() {
    let src = tiny_source();
    let d = tmp("corrupt");
    let m = generate_traces(&src, &d, 3, 2, &|_, _, _| {}).unwrap();
    let path = d.join(&m.shards[0].file);
    let original = std::fs::read(&path).unwrap();

    // Stale digest after an edit.
    let mut s: super::trace_store::TraceShard = serde_json::from_slice(&original).unwrap();
    s.positions[0].q_star += 1;
    std::fs::write(&path, serde_json::to_vec(&s).unwrap()).unwrap();
    let e = generate_traces(&src, &d, 3, 2, &|_, _, _| {})
        .unwrap_err()
        .to_string();
    assert!(e.contains("invalid"), "{e}");

    // A shard of a different source dataset.
    let mut s: super::trace_store::TraceShard = serde_json::from_slice(&original).unwrap();
    s.source_dataset_digest = "0".repeat(64);
    s.digest = s.compute_digest();
    std::fs::write(&path, serde_json::to_vec(&s).unwrap()).unwrap();
    assert!(generate_traces(&src, &d, 3, 2, &|_, _, _| {}).is_err());

    // A shard from the wrong range.
    let other: super::trace_store::TraceShard =
        serde_json::from_slice(&std::fs::read(d.join(&m.shards[1].file)).unwrap()).unwrap();
    std::fs::write(&path, serde_json::to_vec(&other).unwrap()).unwrap();
    assert!(generate_traces(&src, &d, 3, 2, &|_, _, _| {}).is_err());

    // Restoring the original bytes makes the run succeed again.
    std::fs::write(&path, &original).unwrap();
    generate_traces(&src, &d, 3, 2, &|_, _, _| {}).unwrap();
}

#[test]
fn audit_is_complete_resumable_and_gates_the_feasibility_table() {
    let src = tiny_source();
    let d = tmp("audit");
    let tm = generate_traces(&src, &d, 3, 2, &|_, _, _| {}).unwrap();
    // No audit yet: the feasibility measurement is refused.
    assert!(feasibility(&src, &d, "diagnostic").is_err());
    let am = audit_traces(&src, &d, 2, &|_, _, _| {}).unwrap();
    assert!(am.ok());
    assert_eq!(am.total_checked, src.positions.len());
    assert_eq!(am.trace_manifest_digest, tm.manifest_digest);
    let reused = std::cell::RefCell::new(Vec::new());
    let am2 = audit_traces(&src, &d, 3, &|i, _, r| reused.borrow_mut().push((i, r))).unwrap();
    assert_eq!(am2.manifest_digest, am.manifest_digest);
    assert!(
        reused.borrow().iter().all(|(_, r)| *r),
        "completed audit shards are reused"
    );
    let f = feasibility(&src, &d, "diagnostic").unwrap();
    assert_eq!(f.positions, src.positions.len() as u64);
    assert!(f.primary.is_none());
    // The primary role needs the TRAIN split AND the primary cell.
    assert!(
        feasibility(&src, &d, "primary").is_err(),
        "KQRvK M3 is absent from the fixture"
    );

    // A trace that fails the audit blocks the table.
    let bad = tmp("audit_bad");
    generate_traces(&src, &bad, 3, 2, &|_, _, _| {}).unwrap();
    let path = bad.join("trace-shard-00000.json");
    let mut s: super::trace_store::TraceShard =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    s.positions[0].q_star += 5;
    s.digest = s.compute_digest(); // a self-consistent shard that is nevertheless wrong
    std::fs::write(&path, serde_json::to_vec(&s).unwrap()).unwrap();
    // Re-point the manifest at the edited shard so only the audit can catch it.
    let mut m = TraceManifest::load(&bad).unwrap();
    m.shards[0].digest = s.digest.clone();
    m.manifest_digest = m.compute_digest();
    std::fs::write(
        bad.join("trace-manifest.json"),
        serde_json::to_vec_pretty(&m).unwrap(),
    )
    .unwrap();
    let am = audit_traces(&src, &bad, 2, &|_, _, _| {}).unwrap();
    assert!(!am.ok() && am.total_failures >= 1, "{am:?}");
    let e = feasibility(&src, &bad, "diagnostic")
        .unwrap_err()
        .to_string();
    assert!(e.contains("refused"), "{e}");
    // The audit manifest of one run cannot be used with another run's traces.
    std::fs::copy(
        d.join("audit-manifest.json"),
        bad.join("audit-manifest.json"),
    )
    .unwrap();
    assert!(AuditManifest::load(&bad).is_ok());
    assert!(feasibility(&src, &bad, "diagnostic").is_err());
}

#[test]
fn the_frozen_threshold_is_exact_at_its_boundaries() {
    assert!(threshold_pass(100, 25));
    assert!(!threshold_pass(100, 24));
    assert!(threshold_pass(4, 1));
    assert!(!threshold_pass(5, 1));
    assert!(threshold_pass(8, 2));
    assert!(!threshold_pass(0, 0));
    assert!(threshold_pass(3, 1), "1/3 >= 1/4");
    assert!(!threshold_pass(4000, 999));
    assert!(threshold_pass(4000, 1000));
    // No float rounding can move the line: 0.25 * n is exact for every n.
    for n in 1..2000u64 {
        let le = n.div_ceil(4);
        assert!(threshold_pass(n, le));
        if le > 0 {
            assert!(!threshold_pass(n, le - 1) || (le - 1) * 4 >= n);
        }
    }
}

// ---------------------------------------------------------------------------
// Custody
// ---------------------------------------------------------------------------

fn synthetic_holdout_c(dir: &std::path::Path) -> PathBuf {
    let mut solver = MateSolver::new();
    let mut rng = Rng(9);
    let mut p = loop {
        let Some((fen, _)) = sample_position(&mut rng, KQ) else {
            continue;
        };
        if let Some(p) = position("hc-0", &fen, "KQvK", &mut solver) {
            break p;
        }
    };
    p.split = Split::HoldoutC;
    let t = ProofTargets::new(
        Split::HoldoutC,
        1,
        serde_json::json!({"fixture": true}),
        vec![p],
    );
    let path = dir.join("proof-holdout_c.json");
    t.save(&path).unwrap();
    path
}

#[test]
fn sealed_confirmation_data_cannot_enter_working_paths() {
    let d = tmp("seal");
    let c = synthetic_holdout_c(&d);
    // A holdout_c split is refused by the working loader even though it parses.
    let e = load_working_split(&c, &[Split::Train, Split::Tune])
        .unwrap_err()
        .to_string();
    assert!(e.contains("sealed"), "{e}");
    // It is also refused by the tracer.
    let t = ProofTargets::load(&c).unwrap();
    assert!(ensure_traceable(&t).is_err());
    // Its (non-frozen) digest fails custody verification, loudly.
    let e = verify_holdout_c(&c).unwrap_err().to_string();
    assert!(e.contains("STOP") && e.contains(HOLDOUT_C_DIGEST), "{e}");
    // Every non-working split is untraceable.
    for split in [
        Split::Confirm,
        Split::HoldoutA,
        Split::HoldoutB,
        Split::HoldoutC,
    ] {
        let mut x = tiny_source();
        x.split = split;
        x.digest = x.compute_digest();
        assert!(ensure_traceable(&x).is_err(), "{split:?}");
    }
    // A working split of the right kind is accepted; a Tune file is refused where
    // only Train is allowed.
    let tiny = tiny_source();
    let tp = d.join("proof-train.json");
    tiny.save(&tp).unwrap();
    load_working_split(&tp, &[Split::Train]).unwrap();
    assert!(load_working_split(&tp, &[Split::Tune]).is_err());
}

#[test]
fn confirmation_access_needs_explicit_authorization_and_is_logged() {
    assert!(ConfirmAuthorization::request("V3-P4", "x").is_err());
    assert!(ConfirmAuthorization::request("V3-P5", "x").is_err());
    assert!(ConfirmAuthorization::request("V3-P8", "").is_err());
    assert!(ConfirmAuthorization::request("V3-P8", "   ").is_err());
    let auth = ConfirmAuthorization::request("V3-P8", "owner approval 2026-xx-xx").unwrap();
    let d = tmp("auth");
    let c = synthetic_holdout_c(&d);
    let log = d.join("v3-confirm-exposure.log");
    // No valid seal: refused, and nothing is logged.
    let bad_seal = d.join("seal.json");
    std::fs::write(
        &bad_seal,
        serde_json::json!({"schema": "v3_confirm_seal_v1", "split": "holdout_c", "file": "",
            "expected_digest": HOLDOUT_C_DIGEST, "actual_digest": HOLDOUT_C_DIGEST, "positions": 1,
            "verified": true, "sealed": true, "evaluated": true,
            "permitted_p4_use": "", "future_access": ""})
        .to_string(),
    )
    .unwrap();
    assert!(
        load_sealed_confirmation(&c, &bad_seal, &auth, &log).is_err(),
        "already evaluated"
    );
    assert!(!log.exists(), "a refused access must not write an exposure");
    // A seal that is not evaluated but whose data digest differs is refused too.
    std::fs::write(
        &bad_seal,
        serde_json::json!({"schema": "v3_confirm_seal_v1", "split": "holdout_c", "file": "",
            "expected_digest": HOLDOUT_C_DIGEST, "actual_digest": HOLDOUT_C_DIGEST, "positions": 1,
            "verified": true, "sealed": true, "evaluated": false,
            "permitted_p4_use": "", "future_access": ""})
        .to_string(),
    )
    .unwrap();
    assert!(
        load_sealed_confirmation(&c, &bad_seal, &auth, &log).is_err(),
        "digest differs"
    );
    assert!(!log.exists());
}

#[test]
fn the_state_query_packet_stays_answer_free() {
    // Proof structure lives only in this module; the query tool's frozen field set
    // contains nothing derived from it.
    let forbidden = [
        "trace",
        "admissible",
        "proof",
        "refut",
        "q_star",
        "cost",
        "mate",
        "depth_to",
    ];
    for f in recur64_statequery::PACKET_FIELDS {
        for bad in forbidden {
            assert!(
                !f.contains(bad),
                "packet field {f} looks like proof information ({bad})"
            );
        }
    }
    assert_eq!(recur64_statequery::PACKET_FIELDS.len(), 15);
}
