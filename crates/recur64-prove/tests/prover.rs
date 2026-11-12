//! G0 (part 2): the prover is sound and, at full width, optimal; the checker is independent and strict.

use cozy_chess::Piece;
use recur64_prove::classical::Classical;
use recur64_prove::search::{Blind, Orderer, Prover, Solved, verify};
use recur64_prove::split::{Part, canon_key, part_of};
use recur64_prove::tb::{Tablebase, moves_of};

fn sample(tb: &Tablebase, depth: u8, take: usize) -> Vec<cozy_chess::Board> {
    tb.won_white_positions()
        .into_iter()
        .filter(|(_, d)| *d == depth)
        .map(|(b, _)| b)
        .step_by(37)
        .take(take)
        .collect()
}

fn run<O: Orderer>(o: &mut O, width: Option<usize>, b: &cozy_chess::Board, max_n: u8) -> Solved {
    Prover::new(o, width, 5_000_000).solve(b, max_n)
}

#[test]
fn full_width_search_is_optimal_and_every_proof_passes_the_independent_checker() {
    for (name, piece) in [("KQvK", Piece::Queen), ("KRvK", Piece::Rook)] {
        let tb = Tablebase::build(name, &[piece]).unwrap();
        for depth in 1..=3u8 {
            let boards = sample(&tb, depth, 60);
            assert!(boards.len() >= 20, "{name} depth {depth}: {} samples", boards.len());
            for b in &boards {
                for solved in [run(&mut Blind, None, b, 4), run(&mut Classical, None, b, 4)] {
                    let proof = solved.proof.expect("a forced mate within the bound exists");
                    assert_eq!(solved.bound, Some(depth), "{name} {b}: iterative deepening must stop at the optimal depth");
                    assert_eq!(verify(b, &proof).unwrap(), proof.depth());
                    assert_eq!(proof.depth(), depth, "{name} {b}");
                }
            }
        }
    }
}

#[test]
fn width_limited_search_never_returns_an_unsound_proof() {
    let tb = Tablebase::build("KRvK", &[Piece::Rook]).unwrap();
    let mut proofs = 0;
    for depth in [2u8, 4, 6, 8] {
        for b in sample(&tb, depth, 25) {
            for width in [1usize, 2, 3] {
                let s = Prover::new(&mut Classical, Some(width), 300_000).solve(&b, 12);
                if let Some(p) = s.proof {
                    let d = verify(&b, &p).expect("a width-limited proof must still be a real proof");
                    assert!(d >= depth, "a proof cannot be shorter than the optimum: {d} < {depth} in {b}");
                    proofs += 1;
                }
            }
        }
    }
    assert!(proofs > 20, "only {proofs} width-limited proofs were found; the test needs coverage");
}

#[test]
fn the_checker_rejects_tampered_proofs() {
    let tb = Tablebase::build("KQvK", &[Piece::Queen]).unwrap();
    let b = sample(&tb, 3, 1).remove(0);
    let good = run(&mut Classical, None, &b, 4).proof.unwrap();
    assert!(verify(&b, &good).is_ok());
    // Drop one defender reply somewhere: the proof must be rejected.
    let mut missing = (*good).clone();
    let mut tampered = false;
    fn drop_one(p: &mut recur64_prove::search::Proof) -> bool {
        if !p.replies.is_empty() {
            p.replies.pop();
            return true;
        }
        false
    }
    if drop_one(&mut missing) {
        tampered = true;
    }
    assert!(tampered && verify(&b, &missing).is_err(), "a proof missing a defender reply must be rejected");
    // A different first move cannot be answered by the original reply tree.
    let mut wrong = (*good).clone();
    wrong.mv = moves_of(&b).into_iter().find(|m| *m != good.mv).unwrap();
    assert!(verify(&b, &wrong).is_err());
    // An illegal move is rejected.
    let mut illegal = (*good).clone();
    illegal.mv = cozy_chess::Move {
        from: cozy_chess::Square::A1,
        to: cozy_chess::Square::H8,
        promotion: None,
    };
    assert!(verify(&b, &illegal).is_err());
}

#[test]
fn the_split_is_symmetry_invariant_and_roughly_80_10_10() {
    let tb = Tablebase::build("KRvK", &[Piece::Rook]).unwrap();
    let mut counts = [0usize; 3];
    for b in tb.positions.iter().step_by(7) {
        let idx = match part_of(b) {
            Part::Train => 0,
            Part::Val => 1,
            Part::Test => 2,
        };
        counts[idx] += 1;
    }
    let n = counts.iter().sum::<usize>() as f64;
    assert!((counts[0] as f64 / n - 0.8).abs() < 0.03, "train share {counts:?}");
    assert!((counts[2] as f64 / n - 0.1).abs() < 0.03, "test share {counts:?}");
    let b: cozy_chess::Board = "8/8/8/4k3/8/8/8/R3K3 w - - 0 1".parse().unwrap();
    let m: cozy_chess::Board = "8/8/8/3k4/8/8/8/3K3R w - - 0 1".parse().unwrap();
    assert_eq!(canon_key(&b), canon_key(&m));
}

#[test]
fn network_tokens_are_in_range_and_translation_invariant_for_relations() {
    use recur64_prove::net::{MAXP, SQ_TOKENS, TOKENS, tokens};
    let tb = Tablebase::build("KRvK", &[Piece::Rook]).unwrap();
    for b in tb.positions.iter().step_by(997) {
        let (t, n) = tokens(b);
        assert!(n <= MAXP && n == 6, "KRvK has 3 piece-square + 3 relative tokens, got {n}");
        assert!(t[..n].iter().all(|&v| (v as usize) < TOKENS));
        assert!(t[..3].iter().all(|&v| (v as usize) < SQ_TOKENS) && t[3..n].iter().all(|&v| (v as usize) >= SQ_TOKENS));
    }
    // The same king-rook geometry shifted across the board gives the same relative tokens.
    let a: cozy_chess::Board = "8/8/8/8/2k5/8/1R6/K7 w - - 0 1".parse().unwrap();
    let b: cozy_chess::Board = "8/8/8/8/3k4/8/2R5/1K6 w - - 0 1".parse().unwrap();
    let (ta, _) = tokens(&a);
    let (tb2, _) = tokens(&b);
    assert_eq!(ta[3..6], tb2[3..6], "relative tokens must not depend on the absolute position");
    assert_ne!(ta[..3], tb2[..3]);
}


#[test]
fn the_df_pn_baseline_is_optimal_sound_and_checked() {
    use recur64_prove::dfpn::DfPn;
    for (name, piece) in [("KQvK", Piece::Queen), ("KRvK", Piece::Rook)] {
        let tb = Tablebase::build(name, &[piece]).unwrap();
        for depth in 1..=5u8 {
            let boards = sample(&tb, depth, 25);
            assert!(boards.len() >= 10, "{name} depth {depth}: {} samples", boards.len());
            for b in &boards {
                let s = DfPn::new(5_000_000).solve(b, depth + 2);
                let proof = s.proof.expect("df-pn must find the forced mate within the bound");
                assert_eq!(s.bound, Some(depth), "{name} {b}: iterative deepening stops at the optimal depth");
                assert_eq!(verify(b, &proof).unwrap(), depth, "{name} {b}");
            }
        }
    }
}


#[test]
fn every_token_slot_is_a_valid_id_for_every_token_set() {
    use recur64_prove::net::{MAXP, tokens_for, vocab};
    let kr = Tablebase::build("KRvK", &[Piece::Rook]).unwrap();
    for set in ["v2", "v3", "sq"] {
        for b in kr.positions.iter().step_by(1013) {
            let (t, n) = tokens_for(set, b);
            assert!(n <= MAXP);
            // masked slots are still fed to the embedding lookup, so ALL MAXP ids must be inside the vocabulary
            assert!(t.iter().all(|&v| (v as usize) < vocab(set)), "{set}: out-of-vocabulary id in {t:?} (n={n}, vocab {})", vocab(set));
        }
    }
}
