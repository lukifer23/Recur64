//! V69 semantic tests. Fixtures are constructed here (hand-built positions and a
//! TEST-ONLY seed unrelated to the experiment master seed); no historical or
//! experiment data is read.

use cozy_chess::{Board, Color, Move, Piece, Square};
use recur64_core::rules::{Termination, classify};
use recur64_v69::canon::canonical_key;
use recur64_v69::custody::Custody;
use recur64_v69::dataset::*;
use recur64_v69::generate::{Family, sample_root};
use recur64_v69::oracle::{Oracle, RootDepth, Verdict, root_depth};
use recur64_v69::reference;
use recur64_v69::streams::MasterSeed;
use std::path::Path;

const TEST_SEED_HEX: &str = "5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed";

fn test_seed() -> MasterSeed {
    MasterSeed::from_hex(TEST_SEED_HEX).unwrap()
}

fn board(fen: &str) -> Board {
    fen.parse().unwrap()
}

fn legal(b: &Board) -> Vec<Move> {
    let mut v = Vec::new();
    b.generate_moves(|ms| {
        v.extend(ms);
        false
    });
    v
}

// ---------------------------------------------------------------- rules

#[test]
fn movegen_matches_bruteforce_is_legal() {
    let seed = test_seed();
    let mut checked = 0;
    for fam in Family::ALL {
        for i in 0..60 {
            if let Ok(r) = sample_root(&seed, fam, i) {
                let mut generated: Vec<Move> = legal(&r.board);
                generated.sort_by_key(|m| (m.from as usize, m.to as usize));
                let mut brute = Vec::new();
                for f in Square::ALL {
                    for t in Square::ALL {
                        let m = Move { from: f, to: t, promotion: None };
                        if r.board.is_legal(m) {
                            brute.push(m);
                        }
                    }
                }
                brute.sort_by_key(|m| (m.from as usize, m.to as usize));
                assert_eq!(generated, brute);
                checked += 1;
            }
        }
    }
    assert!(checked > 50);
}

#[test]
fn terminal_boundaries_use_rules_authority() {
    // Stalemate after Qg6 (hand-built).
    let b = board("7k/8/5K2/8/8/8/8/6Q1 w - - 0 1");
    let mut c = b.clone();
    c.play(Move { from: Square::G1, to: Square::G6, promotion: None });
    assert_eq!(classify(&c, 1, 0, None), Some(Termination::Stalemate));
    // Qg7 is mate (child terminal Checkmate).
    let mut c = b.clone();
    c.play(Move { from: Square::G1, to: Square::G7, promotion: None });
    assert_eq!(classify(&c, 1, 0, None), Some(Termination::Checkmate));
    // The solver: mate in 1 exists, the stalemating move is not a mate.
    let mut o = Oracle::new(1_000_000);
    assert_eq!(o.mate_within_root(&b, Color::White, 1), Verdict::Yes);
    assert_eq!(root_depth(&mut o, &b, Color::White, 3), RootDepth::Exact(1));
}

// ---------------------------------------------------------------- canonical identity

fn mirror_color_swap(b: &Board, t: usize) -> (Board, Color) {
    // Apply dihedral transform t and swap colours (attacker identity relabelled).
    let mut cells: Vec<(Square, Piece, Color)> = Vec::new();
    for sq in Square::ALL {
        if let Some(p) = b.piece_on(sq) {
            cells.push((sq, p, b.color_on(sq).unwrap()));
        }
    }
    let tf = |sq: Square| -> Square {
        let (f, r) = (sq.file() as usize, sq.rank() as usize);
        let (f, r) = match t {
            0 => (f, r),
            1 => (7 - f, r),
            2 => (f, 7 - r),
            3 => (7 - f, 7 - r),
            4 => (r, f),
            5 => (7 - r, f),
            6 => (r, 7 - f),
            _ => (7 - r, 7 - f),
        };
        Square::index(r * 8 + f)
    };
    let mut fen_cells = [[' '; 8]; 8];
    for (sq, p, c) in cells {
        let s = tf(sq);
        let ch = match p {
            Piece::King => 'k',
            Piece::Queen => 'q',
            Piece::Rook => 'r',
            _ => unreachable!(),
        };
        let nc = !c;
        fen_cells[s.rank() as usize][s.file() as usize] = if nc == Color::White { ch.to_ascii_uppercase() } else { ch };
    }
    let mut fen = String::new();
    for r in (0..8).rev() {
        let mut e = 0;
        for f in 0..8 {
            if fen_cells[r][f] == ' ' {
                e += 1;
            } else {
                if e > 0 {
                    fen.push_str(&e.to_string());
                    e = 0;
                }
                fen.push(fen_cells[r][f]);
            }
        }
        if e > 0 {
            fen.push_str(&e.to_string());
        }
        if r > 0 {
            fen.push('/');
        }
    }
    let stm = !b.side_to_move();
    fen.push_str(if stm == Color::White { " w - - 0 1" } else { " b - - 0 1" });
    (fen.parse().unwrap(), !Color::White)
}

#[test]
fn canonical_key_invariant_under_all_16_symmetries_and_discriminates() {
    let seed = test_seed();
    let mut n = 0;
    let mut keys = std::collections::HashSet::new();
    for fam in Family::ALL {
        for i in 0..300 {
            let Ok(r) = sample_root(&seed, fam, i) else { continue };
            let k0 = canonical_key(&r.board, r.attacker);
            keys.insert(k0);
            for t in 0..8 {
                let (b2, _) = mirror_color_swap(&r.board, t);
                // attacker colour swapped with the relabelling
                let k = canonical_key(&b2, !r.attacker);
                assert_eq!(k, k0, "colour-swapped image t={t}");
                // pure dihedral image: swap twice (colour swap is an involution)
                let (b3, _) = mirror_color_swap(&b2, 5);
                assert_eq!(canonical_key(&b3, r.attacker), k0);
            }
            n += 1;
        }
    }
    assert!(n > 100);
    assert!(keys.len() > 100, "distinct positions must keep distinct keys");
    // Side-to-move discrimination: same pieces, different mover.
    let b = board("7k/8/5K2/8/8/8/8/6Q1 w - - 0 1");
    let b2 = board("7k/8/5K2/8/8/8/8/6Q1 b - - 0 1");
    assert_ne!(canonical_key(&b, Color::White), canonical_key(&b2, Color::White));
}

// ---------------------------------------------------------------- oracle vs reference

fn gen_roots(fam: Family, count: u64) -> Vec<(Board, Color)> {
    let seed = test_seed();
    (0..count).filter_map(|i| sample_root(&seed, fam, i).ok().map(|r| (r.board, r.attacker))).collect()
}

#[test]
fn root_depth_matches_independent_reference_both_colors() {
    let mut cmp = 0;
    let mut colors = std::collections::HashSet::new();
    let mut depth_seen = std::collections::HashSet::new();
    for fam in Family::ALL {
        let mut o = Oracle::new(5_000_000);
        for (b, att) in gen_roots(fam, 120) {
            // Reference limited to depth 2 for tractability (M3 is covered at child level below).
            let mine = match root_depth(&mut o, &b, att, 2) {
                RootDepth::Exact(d) => Some(d),
                RootDepth::NoMateWithin(_) => None,
                RootDepth::Unknown => panic!("unknown at test scale"),
            };
            let theirs = reference::root_min_depth(&b, att, 2);
            assert_eq!(mine, theirs, "{}", b);
            colors.insert(att);
            depth_seen.insert(mine);
            cmp += 1;
        }
    }
    assert!(cmp > 100);
    assert_eq!(colors.len(), 2);
    assert!(depth_seen.contains(&Some(1)) && depth_seen.contains(&Some(2)) && depth_seen.contains(&None));
}

#[test]
fn child_targets_match_reference_at_both_budgets_with_both_classes() {
    let mut seen = std::collections::HashSet::new(); // (budget, label, color)
    let mut compared = 0;
    for fam in Family::ALL {
        let mut o = Oracle::new(5_000_000);
        let mut roots_done = [0usize; 2];
        for (b, att) in gen_roots(fam, 400) {
            let m = match root_depth(&mut o, &b, att, 3) {
                RootDepth::Exact(m @ 2..=3) => m,
                _ => continue,
            };
            let idx = (m - 2) as usize;
            // keep the reference cost bounded: few M3 roots, more M2 roots
            if roots_done[idx] >= if m == 2 { 6 } else { 2 } {
                continue;
            }
            roots_done[idx] += 1;
            let n = m - 1;
            for mv in legal(&b) {
                let mut c = b.clone();
                c.play_unchecked(mv);
                if classify(&c, 1, 0, None).is_some() {
                    continue; // terminal children are excluded from the task
                }
                let mine = o.mate_within_child(&c, att, n);
                assert_ne!(mine, Verdict::Unknown);
                let theirs = reference::child_target(&c, att, n);
                assert_eq!(mine == Verdict::Yes, theirs, "child {c} n={n}");
                seen.insert((n, theirs, att));
                compared += 1;
            }
        }
    }
    assert!(compared > 200, "compared {compared}");
    for n in [1u8, 2] {
        for label in [false, true] {
            for color in [Color::White, Color::Black] {
                assert!(seen.contains(&(n, label, color)), "missing coverage n={n} label={label} color={color:?}");
            }
        }
    }
}

#[test]
fn negative_target_is_budget_relative_not_eventual_loss() {
    // A child that is negative at n=1 but positive at n=2 must exist and be handled as
    // a budget-relative negative.
    let mut found = false;
    'outer: for fam in Family::ALL {
        let mut o = Oracle::new(5_000_000);
        for (b, att) in gen_roots(fam, 300) {
            if !matches!(root_depth(&mut o, &b, att, 3), RootDepth::Exact(3)) {
                continue;
            }
            for mv in legal(&b) {
                let mut c = b.clone();
                c.play_unchecked(mv);
                if classify(&c, 1, 0, None).is_some() {
                    continue;
                }
                if o.mate_within_child(&c, att, 1) == Verdict::No && o.mate_within_child(&c, att, 2) == Verdict::Yes {
                    assert!(!reference::child_target(&c, att, 1) && reference::child_target(&c, att, 2));
                    found = true;
                    break 'outer;
                }
            }
        }
    }
    assert!(found);
}

#[test]
fn node_limit_yields_unknown_and_never_poisons_the_cache() {
    let seed = test_seed();
    // find an M3 root
    let mut o_ref = Oracle::new(50_000_000);
    let mut target = None;
    'o: for fam in Family::ALL {
        for i in 0..400 {
            if let Ok(r) = sample_root(&seed, fam, i) {
                if matches!(root_depth(&mut o_ref, &r.board, r.attacker, 3), RootDepth::Exact(3)) {
                    target = Some(r);
                    break 'o;
                }
            }
        }
    }
    let r = target.expect("an M3 root");
    let mut tiny = Oracle::new(25);
    assert_eq!(tiny.mate_within_root(&r.board, r.attacker, 3), Verdict::Unknown);
    assert_eq!(tiny.mate_within_root(&r.board, r.attacker, 2), Verdict::Unknown);
    assert!(tiny.aborted_queries >= 2);
    // Same oracle, after an abort, with a big budget agrees with a pristine oracle.
    tiny.set_node_limit(50_000_000);
    let mut fresh = Oracle::new(50_000_000);
    for n in [1u8, 2, 3] {
        assert_eq!(tiny.mate_within_root(&r.board, r.attacker, n), fresh.mate_within_root(&r.board, r.attacker, n), "n={n}");
    }
    assert_eq!(tiny.mate_within_root(&r.board, r.attacker, 2), Verdict::No);
    assert_eq!(tiny.mate_within_root(&r.board, r.attacker, 3), Verdict::Yes);
}

// ---------------------------------------------------------------- generator, streams

#[test]
fn sampler_is_pure_and_respects_domain() {
    let seed = test_seed();
    let mut ok = 0;
    for fam in Family::ALL {
        for i in 0..1200 {
            let a = sample_root(&seed, fam, i);
            let b = sample_root(&seed, fam, i);
            match (a, b) {
                (Ok(x), Ok(y)) => {
                    assert_eq!(x.board.to_string(), y.board.to_string());
                    assert_eq!(x.board.side_to_move(), x.attacker);
                    assert_eq!(x.board.halfmove_clock(), 0);
                    assert!(x.board.checkers().is_empty());
                    let att = x.board.colors(x.attacker);
                    let n = |p: Piece| (x.board.pieces(p) & att).len();
                    match fam {
                        Family::Kqq => assert_eq!((n(Piece::Queen), n(Piece::Rook)), (2, 0)),
                        Family::Kqr => assert_eq!((n(Piece::Queen), n(Piece::Rook)), (1, 1)),
                        Family::Krr => assert_eq!((n(Piece::Queen), n(Piece::Rook)), (0, 2)),
                    }
                    assert_eq!(x.board.colors(!x.attacker).len(), 1);
                    ok += 1;
                }
                (Err(_), Err(_)) => {}
                _ => panic!("sampler not deterministic"),
            }
        }
    }
    assert!(ok > 500);
}

#[test]
fn streams_are_deterministic_and_disjoint() {
    let s = test_seed();
    let mut a = s.stream("generation/KQQvK", 7);
    let mut b = s.stream("generation/KQQvK", 7);
    assert_eq!(a.next_u64(), b.next_u64());
    let labels = ["generation/KQQvK", "partition", "model_init", "train_order", "intervention"];
    let mut firsts = std::collections::HashSet::new();
    for l in labels {
        assert!(firsts.insert(s.stream(l, 0).next_u64()));
    }
    let other = MasterSeed::from_hex(&"ab".repeat(32)).unwrap();
    assert_ne!(s.stream("partition", 0).next_u64(), other.stream("partition", 0).next_u64());
    let mut r = s.stream("x", 0);
    for _ in 0..1000 {
        assert!(r.below(7) < 7);
    }
}

// ---------------------------------------------------------------- custody

#[test]
fn custody_refuses_escapes_and_historical_components() {
    let dir = std::env::temp_dir().join(format!("v69-custody-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let root = dir.join("artifacts").join("v69");
    let c = Custody::new(&root).unwrap();
    assert!(c.resolve(Path::new("run-a/data/fit.jsonl")).is_ok());
    assert!(c.resolve(Path::new("../outside.json")).is_err());
    assert!(c.resolve(&dir.join("elsewhere").join("x")).is_err());
    assert!(c.resolve(Path::new("runs/old")).is_err());
    assert!(c.resolve(Path::new("run-a/evidence/x")).is_err());
    // a historical worktree-shaped location is refused as an artifact root
    let hist = dir.join("Recur64").join("artifacts");
    assert!(Custody::new(&hist).unwrap().resolve(Path::new("a")).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------- grouping

fn fake_root(id: &str, fam: &str, depth: u8, child_keys: &[&str]) -> RootRec {
    let hex = |s: &str| format!("{:0<130}", s.bytes().map(|b| format!("{b:02x}")).collect::<String>());
    RootRec {
        id: id.to_string(),
        family: fam.to_string(),
        attacker: "white".into(),
        fen: String::new(),
        key: hex(id),
        depth,
        legal_moves: child_keys.len(),
        correct_moves: 1,
        terminal_children: 0,
        children: child_keys
            .iter()
            .map(|k| ChildRec { mv: format!("m{k}"), fen: String::new(), key: hex(k), status: "neg".into() })
            .collect(),
        oracle_nodes: 0,
        oracle_micros: 0,
    }
}

#[test]
fn roots_sharing_a_child_are_one_group_and_one_partition_across_budgets() {
    let seed = test_seed();
    let mut roots = vec![
        fake_root("r1", "KQQvK", 2, &["a", "b"]),
        fake_root("r2", "KQQvK", 3, &["b", "c"]), // shares b with r1, different budget
        fake_root("r3", "KQQvK", 2, &["c", "d"]), // chains through c
        fake_root("r4", "KQQvK", 2, &["z"]),
    ];
    for i in 0..40 {
        roots.push(fake_root(&format!("s{i}"), "KRRvK", 2, &[&format!("u{i}")]));
    }
    let g1 = build_groups(&seed, &roots);
    let g2 = build_groups(&seed, &roots);
    let comp = g1.iter().find(|g| g.roots.contains(&0)).unwrap();
    assert!(comp.roots.contains(&1) && comp.roots.contains(&2));
    assert!(!comp.roots.contains(&3));
    assert!(comp.mixed_depth);
    // determinism
    assert_eq!(g1.len(), g2.len());
    for (a, b) in g1.iter().zip(g2.iter()) {
        assert_eq!((a.partition, &a.roots), (b.partition, &b.roots));
    }
    // 2:1:1 dealing on the 40-group KRR stratum
    let mut counts = std::collections::HashMap::new();
    for g in g1.iter().filter(|g| g.family == "KRRvK") {
        *counts.entry(g.partition).or_insert(0) += 1;
    }
    assert_eq!(counts[&Partition::Fit], 20);
    assert_eq!(counts[&Partition::Val], 10);
    assert_eq!(counts[&Partition::Test], 10);
}
