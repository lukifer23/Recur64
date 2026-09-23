//! Hand-verified tactical edge cases (T1.9).

use recur64_core::{GameState, StandardMove, Termination};

fn has_move(fen: &str, uci: &str) -> bool {
    let g = GameState::from_fen(fen).unwrap();
    g.legal_standard_moves().iter().any(|m| m.to_uci() == uci)
}

#[test]
fn en_passant_exposing_own_king_is_illegal() {
    // Black pawn e4 could capture d3 e.p., but that opens the 4th rank to the
    // white queen on h4 and exposes the black king on a4.
    let fen = "8/8/8/8/k2Pp2Q/8/8/4K3 b - d3 0 1";
    assert!(!has_move(fen, "e4d3"), "pinned en passant must be illegal");
    // Black pawns move toward rank 1; the plain push e4e3 is still legal.
    assert!(has_move(fen, "e4e3"));
}

#[test]
fn castling_through_check_is_illegal() {
    // Black rook on f2 attacks f1, the square the king crosses for O-O.
    let fen = "r3k2r/8/8/8/8/8/5r2/R3K2R w KQkq - 0 1";
    assert!(!has_move(fen, "e1g1"), "cannot castle through check");
    assert!(has_move(fen, "e1c1"), "queenside castle is legal here");
}

#[test]
fn castling_into_check_is_illegal() {
    // Black rook on g2 attacks g1, the king's destination.
    let fen = "r3k2r/8/8/8/8/8/6r1/R3K2R w KQkq - 0 1";
    assert!(!has_move(fen, "e1g1"), "cannot castle into check");
}

#[test]
fn castling_while_in_check_is_illegal() {
    // Black rook on e2 gives check on the e-file.
    let fen = "r3k2r/8/8/8/8/8/4r3/R3K2R w KQkq - 0 1";
    assert!(!has_move(fen, "e1g1"));
    assert!(!has_move(fen, "e1c1"));
}

#[test]
fn castling_rights_lost_after_king_move() {
    let mut g = GameState::from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1").unwrap();
    g.apply_uci("e1e2").unwrap();
    let rights = g.board().castle_rights(recur64_core::Color::White);
    assert!(rights.short.is_none() && rights.long.is_none());
}

#[test]
fn castling_rights_lost_after_rook_move() {
    let mut g = GameState::from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1").unwrap();
    g.apply_uci("a1a2").unwrap();
    let rights = g.board().castle_rights(recur64_core::Color::White);
    assert!(
        rights.long.is_none(),
        "queenside right lost after rook move"
    );
    assert!(rights.short.is_some(), "kingside right retained");
}

#[test]
fn all_four_promotions_available() {
    let fen = "8/P6k/8/8/8/8/8/K7 w - - 0 1";
    let g = GameState::from_fen(fen).unwrap();
    let ups: Vec<String> = g
        .legal_standard_moves()
        .iter()
        .filter(|m| m.from == recur64_core::Square::A7)
        .map(|m| m.to_uci())
        .collect();
    for suffix in ["a8q", "a8r", "a8b", "a8n"] {
        assert!(ups.iter().any(|m| m.ends_with(suffix)), "missing {suffix}");
    }
}

#[test]
fn promotion_capture_available() {
    let fen = "1n5k/P7/8/8/8/8/8/K7 w - - 0 1";
    let g = GameState::from_fen(fen).unwrap();
    let captures: Vec<String> = g
        .legal_standard_moves()
        .iter()
        .filter(|m| m.from == recur64_core::Square::A7 && m.to == recur64_core::Square::B8)
        .map(|m| m.to_uci())
        .collect();
    assert_eq!(captures.len(), 4, "four promotion-capture choices");
}

#[test]
fn double_check_only_king_moves() {
    // Black king h8 in double check from Rh1 (file) and Bb2 (diagonal).
    let fen = "7k/8/8/8/8/8/1B6/6KR b - - 0 1";
    let g = GameState::from_fen(fen).unwrap();
    assert!(g.board().checkers().len() >= 2, "expected double check");
    for m in g.legal_standard_moves() {
        assert_eq!(m.from, recur64_core::Square::H8, "only the king may move");
    }
}

#[test]
fn back_rank_mate() {
    let mut g = GameState::from_fen("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1").unwrap();
    g.apply_uci("a1a8").unwrap();
    assert_eq!(g.termination(), Some(Termination::Checkmate));
}

#[test]
fn stalemate_detected() {
    let g = GameState::from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1").unwrap();
    assert_eq!(g.termination(), Some(Termination::Stalemate));
    assert!(g.legal_standard_moves().is_empty());
}

#[test]
fn same_placement_different_rights_are_different_states() {
    let a = GameState::from_fen("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1").unwrap();
    let b = GameState::from_fen("4k3/8/8/8/8/8/8/R3K2R w K - 0 1").unwrap();
    assert!(!a.board().same_position(b.board()));
    assert_ne!(
        a.legal_standard_moves().len(),
        b.legal_standard_moves().len()
    );
}

#[test]
fn promotion_roundtrips_through_action_space() {
    let g = GameState::from_fen("8/P6k/8/8/8/8/8/K7 w - - 0 1").unwrap();
    let p = g.perspective();
    for m in g.legal_standard_moves() {
        let id = recur64_core::ActionId::from_physical(
            m.from,
            m.to,
            m.promotion.unwrap_or(recur64_core::PromotionCode::NONE),
            p,
        );
        let (f, t, pr) = id.to_physical(p);
        let promo = if pr.is_none() { None } else { Some(pr) };
        assert_eq!(StandardMove::new(f, t, promo), m);
    }
}
