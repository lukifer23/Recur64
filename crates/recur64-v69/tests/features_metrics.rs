//! Feature-encoding and independent-aggregator tests (hand-built fixtures only).

use recur64_v69::dataset::{Example, Partition};
use recur64_v69::features::{FamilyKind, featurize, read_rows};
use recur64_v69::metrics::{PredRow, aggregate, n_readouts};
use recur64_v69::streams::MasterSeed;

// White attacker (Kf6, Qf1, Qg1) vs lone black king h8, black to move.
const W_ATT: &str = "7k/8/5K2/8/8/8/8/5QQ1 b - - 1 1";
// Colour-swapped + rank-flipped image: black attacker (Kf3, Qf8, Qg8), white king h1 to move.
const B_ATT: &str = "5qq1/8/8/8/8/5k2/8/7K w - - 1 1";

#[test]
fn both_attacker_colours_give_identical_features() {
    let (a, fa) = featurize(W_ATT, 1).unwrap();
    let (b, fb) = featurize(B_ATT, 1).unwrap();
    assert_eq!(fa, FamilyKind::Kqq);
    assert_eq!(fb, FamilyKind::Kqq);
    assert_eq!(a, b, "colour-relative + rank-flipped encodings must coincide");
    assert_eq!(a.attacker_to_move, 0);
    // explicit ownership: codes 1..=6 are attacker, 7..=12 defender
    assert!(a.piece.iter().any(|&p| p == 6)); // attacker king
    assert!(a.piece.iter().any(|&p| p == 12)); // defender king
    assert_eq!(a.piece.iter().filter(|&&p| p == 5).count(), 2); // two attacker queens
}

#[test]
fn budget_is_part_of_input_and_unsupported_values_fail() {
    assert_eq!(featurize(W_ATT, 2).unwrap().0.budget, 2);
    assert!(featurize(W_ATT, 0).is_err());
    assert!(featurize(W_ATT, 3).is_err());
}

#[test]
fn invalid_or_out_of_domain_inputs_fail_visibly() {
    // 5-field FEN
    assert!(featurize("7k/8/5K2/8/8/8/8/5QQ1 b - - 1", 1).is_err());
    // garbage
    assert!(featurize("not a fen at all x y", 1).is_err());
    // pawn present
    assert!(featurize("7k/8/5K2/8/8/8/4P3/5QQ1 b - - 1 1", 1).is_err());
    // wrong material (KQvK)
    assert!(featurize("7k/8/5K2/8/8/8/8/6Q1 b - - 1 1", 1).is_err());
    // halfmove outside domain
    assert!(featurize("7k/8/5K2/8/8/8/8/5QQ1 b - - 7 1", 1).is_err());
    // terminal (checkmate) child: Qg7# style
    assert!(featurize("7k/6Q1/5K2/8/8/8/8/6Q1 b - - 1 1", 1).is_err());
}

#[test]
fn features_do_not_depend_on_labels_ids_or_row_order() {
    let rows = read_rows(&format!(
        "{{\"id\":\"a\",\"fen\":\"{W_ATT}\",\"budget\":1,\"label\":true}}\n{{\"id\":\"b\",\"fen\":\"{W_ATT}\",\"budget\":1,\"label\":false}}\n"
    ))
    .unwrap();
    let f0 = featurize(&rows[0].fen, rows[0].budget).unwrap().0;
    let f1 = featurize(&rows[1].fen, rows[1].budget).unwrap().0;
    assert_eq!(f0, f1); // featurize takes (fen, budget) only: no label/id channel exists
    // duplicates / unknown fields rejected
    assert!(read_rows(&format!("{{\"id\":\"a\",\"fen\":\"{W_ATT}\",\"budget\":1,\"label\":true}}\n{{\"id\":\"a\",\"fen\":\"{W_ATT}\",\"budget\":1,\"label\":true}}\n")).is_err());
    assert!(read_rows(&format!("{{\"id\":\"a\",\"fen\":\"{W_ATT}\",\"budget\":1,\"label\":true,\"root\":\"x\"}}\n")).is_err());
}

#[test]
fn erasure_clears_board_but_keeps_task_metadata() {
    let (f, _) = featurize(W_ATT, 2).unwrap();
    let e = f.erased();
    assert!(e.piece.iter().all(|&p| p == 0));
    assert_eq!((e.budget, e.attacker_to_move, e.scalars), (f.budget, f.attacker_to_move, f.scalars));
}

// ------------------------------------------------------------------ aggregator

fn ex(id: &str, label: bool, group: &str, budget: u8) -> Example {
    Example {
        id: id.into(),
        partition: Partition::Val,
        family: "KQQvK".into(),
        budget,
        label,
        fen: String::new(),
        key: String::new(),
        root_id: String::new(),
        group_id: group.into(),
        root_fen: String::new(),
        root_depth: budget + 1,
        mv: String::new(),
    }
}

fn pr(id: &str, mode: &str, z: f32, donor: Option<&str>) -> PredRow {
    PredRow { id: id.into(), mode: mode.into(), logits: vec![z], donor_id: donor.map(|s| s.to_string()) }
}

#[test]
fn aggregator_matches_hand_computed_values_and_gates() {
    assert_eq!(n_readouts("A"), 1);
    assert_eq!(n_readouts("B"), 3);
    let seed = MasterSeed::from_hex(&"11".repeat(32)).unwrap();
    // 4 examples: labels T,T,F,F ; logits chosen so 3/4 correct (one false positive)
    let metas = vec![ex("e0", true, "g0", 1), ex("e1", true, "g1", 1), ex("e2", false, "g2", 1), ex("e3", false, "g3", 1)];
    let zs = [2.0f32, 1.0, -1.5, 0.5];
    let mut real: Vec<PredRow> = (0..4).map(|i| pr(&format!("e{i}"), "real", zs[i], None)).collect();
    // derangement e0->e1, e1->e2, e2->e3, e3->e0; deranged logits equal donor's real logits
    let donors = ["e1", "e2", "e3", "e0"];
    for i in 0..4 {
        let dz = zs[(i + 1) % 4];
        real.push(pr(&format!("e{i}"), "derange", dz, Some(donors[i])));
    }
    let rep = aggregate("A", &real, &real, &metas, &metas, &seed).unwrap();
    // BA: tp=2 fn=0 tn=1 fp=1 -> 0.5*(1+0.5)=0.75 ; acc 0.75
    assert!((rep.val.real_bal_acc - 0.75).abs() < 1e-12);
    assert!((rep.val.real_acc - 0.75).abs() < 1e-12);
    let bce = |z: f64, y: f64| z.max(0.0) - z * y + (-z.abs()).exp().ln_1p();
    let want = (bce(2.0, 1.0) + bce(1.0, 1.0) + bce(-1.5, 0.0) + bce(0.5, 0.0)) / 4.0;
    assert!((rep.val.real_final_bce - want).abs() < 1e-12);
    let sig = |z: f64| 1.0 / (1.0 + (-z).exp());
    let br = ((sig(2.0) - 1.0).powi(2) + (sig(1.0) - 1.0).powi(2) + sig(-1.5).powi(2) + sig(0.5).powi(2)) / 4.0;
    assert!((rep.val.brier - br).abs() < 1e-12);
    // derange: preds for recipients = donor logits (1.0,-1.5,0.5,2.0) vs recipient labels (T,T,F,F): pos,neg,pos,pos
    let d = rep.val.derange.as_ref().unwrap();
    // recipient labels: e0 T pred+ ok; e1 T pred- wrong; e2 F pred+ wrong; e3 F pred+ wrong -> acc .25
    assert!((d.acc_vs_recipient_labels - 0.25).abs() < 1e-12);
    // donor labels: donors e1 T, e2 F, e3 F, e0 T vs preds + - + + -> ok, ok, wrong, ok = .75
    assert!((d.acc_vs_donor_labels - 0.75).abs() < 1e-12);
    // donor-label agreement: (T,T)=agree, (T,F), (F,F)=agree, (F,T) -> 0.5
    assert!((d.donor_label_agreement_rate - 0.5).abs() < 1e-12);
    assert!(d.max_abs_logit_diff_vs_donor_real < 1e-6);
    assert_eq!(rep.bootstrap_groups, 4);
    // gates: fit/val BA .75 < .95 -> metric gates fail on fit; val BA >= .75 true
    assert!(!rep.gates.fit_bal_acc_ge_95);
    assert!(rep.gates.val_bal_acc_ge_75);
    assert!(!rep.gates.metric_gates_pass);
    let ci = &rep.val_bootstrap["val_bal_acc"];
    assert!(ci.lo95 <= ci.point + 1e-9 && ci.hi95 >= ci.point - 1e-9);
}

#[test]
fn aggregator_rejects_bad_inputs() {
    let seed = MasterSeed::from_hex(&"11".repeat(32)).unwrap();
    let metas = vec![ex("e0", true, "g0", 1), ex("e1", false, "g1", 1)];
    let good = vec![pr("e0", "real", 1.0, None), pr("e1", "real", -1.0, None)];
    // missing real prediction
    assert!(aggregate("A", &good[..1], &good, &metas, &metas, &seed).is_err());
    // non-finite logit
    let nan = vec![pr("e0", "real", f32::NAN, None), pr("e1", "real", -1.0, None)];
    assert!(aggregate("A", &nan, &good, &metas, &metas, &seed).is_err());
    // wrong readout count for arm B
    assert!(aggregate("B", &good, &good, &metas, &metas, &seed).is_err());
    // donor == recipient is not a derangement
    let mut bad = good.clone();
    bad.push(pr("e0", "derange", 0.0, Some("e0")));
    bad.push(pr("e1", "derange", 0.0, Some("e0")));
    assert!(aggregate("A", &bad, &good, &metas, &metas, &seed).is_err());
}

#[test]
fn d8_augmentation_orbit_matches_transformed_boards_for_both_attacker_colours() {
    use recur64_v69::features::{d8_square, transform_fen};
    for fen in [W_ATT, B_ATT, "7k/8/5K2/8/8/8/8/R5Q1 b - - 1 1"] {
        let f = featurize(fen, 1).unwrap().0;
        // identity and group closure on the grid
        assert_eq!(f.d8(0), f);
        let mut orbit_feat: Vec<[u8; 64]> = (0..8).map(|t| f.d8(t).piece).collect();
        let mut orbit_board: Vec<[u8; 64]> = (0..8).map(|t| featurize(&transform_fen(fen, t).unwrap(), 1).unwrap().0.piece).collect();
        orbit_feat.sort();
        orbit_board.sort();
        assert_eq!(orbit_feat, orbit_board, "feature-grid orbit must equal the orbit of real transformed boards: {fen}");
        for t in 0..8 {
            let g = f.d8(t);
            assert_eq!((g.budget, g.attacker_to_move, g.scalars), (f.budget, f.attacker_to_move, f.scalars));
        }
    }
    // d8_square is a bijection for every t
    for t in 0..8 {
        let mut seen = [false; 64];
        for s in 0..64 {
            seen[d8_square(s, t)] = true;
        }
        assert!(seen.iter().all(|x| *x));
    }
}
