//! Contract test pinning the exact cozy-chess 0.3.4 behaviors Recur64 relies on.
//!
//! If a cozy-chess upgrade changes any of these, this test fails and the change
//! must be reviewed before it can silently alter Recur64 semantics.

use cozy_chess::{Board, Color, File, GameStatus, Square, util};

#[test]
fn square_indexing_is_rank_major_a1_zero() {
    assert_eq!(Square::A1 as usize, 0);
    assert_eq!(Square::H1 as usize, 7);
    assert_eq!(Square::A8 as usize, 56);
    assert_eq!(Square::H8 as usize, 63);
    // Recur64 uses a1=0 rank-major; cozy matches exactly.
    assert_eq!(Square::B3 as usize, 1 + 2 * 8);
}

#[test]
fn black_relative_is_rank_reflection_xor_56() {
    for sq in 0u8..64 {
        let square = Square::index(sq as usize);
        let reflected = square.relative_to(Color::Black);
        assert_eq!(
            reflected as usize,
            (sq as usize) ^ 56,
            "black relative must be rank reflection (XOR 56)"
        );
    }
}

#[test]
fn startpos_has_20_legal_moves() {
    let board = Board::default();
    let mut n = 0;
    board.generate_moves(|moves| {
        n += moves.len();
        false
    });
    assert_eq!(n, 20);
}

#[test]
fn castling_is_king_captures_rook_but_util_roundtrips_uci() {
    let board = Board::default();
    // Standard UCI O-O is e1g1; cozy stores it as king-to-rook e1h1.
    let mv = util::parse_uci_move(&board, "e1g1").unwrap();
    assert_eq!(mv.from, Square::E1);
    assert_eq!(mv.to, Square::H1);
    assert_eq!(util::display_uci_move(&board, mv).to_string(), "e1g1");

    // O-O-O: UCI e1c1 -> cozy e1a1.
    let mv = util::parse_uci_move(&board, "e1c1").unwrap();
    assert_eq!(mv.from, Square::E1);
    assert_eq!(mv.to, Square::A1);
    assert_eq!(util::display_uci_move(&board, mv).to_string(), "e1c1");
}

#[test]
fn en_passant_is_fen_style_present_after_any_double_push() {
    // After 1.e4 there is no black pawn able to capture en passant, but the EP
    // target square is still reported (FEN semantics). This is what
    // Observation V1 uses for its EP indicator.
    let mut board = Board::default();
    board.play("e2e4".parse().unwrap());
    assert_eq!(board.en_passant(), Some(File::E));
}

#[test]
fn same_position_ignores_clocks_and_nonapplicable_ep() {
    let a: Board = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        .parse()
        .unwrap();
    let b: Board = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 4 3"
        .parse()
        .unwrap();
    assert!(a.same_position(&b));

    // Here a legal en passant capture exists, so EP is legally relevant.
    let c: Board = "rnbqkb1r/ppp1pppp/5n2/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3"
        .parse()
        .unwrap();
    let d: Board = "rnbqkb1r/ppp1pppp/5n2/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq - 4 5"
        .parse()
        .unwrap();
    assert!(!c.same_position(&d));
}

#[test]
fn status_detects_checkmate_and_stalemate() {
    // Fool's mate: 1.f3 e5 2.g4 Qh4#.
    let mut board = Board::default();
    for mv in ["f2f3", "e7e5", "g2g4", "d8h4"] {
        board.play(mv.parse().unwrap());
    }
    assert_eq!(board.status(), GameStatus::Won);
    assert_eq!(board.side_to_move(), Color::White);

    // Classic stalemate.
    let stalemate: Board = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1".parse().unwrap();
    assert_eq!(stalemate.status(), GameStatus::Drawn);
}
