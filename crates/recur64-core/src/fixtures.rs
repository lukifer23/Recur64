//! Reference fixtures with provenance.
//!
//! Perft counts are from the Chess Programming Wiki "Perft Results" page
//! (<https://www.chessprogramming.org/Perft_Results>). They are an independent
//! oracle: they do not come from cozy-chess's own tests.

/// A perft reference position with published node counts.
pub struct PerftFixture {
    pub name: &'static str,
    pub fen: &'static str,
    pub source: &'static str,
    /// `counts[d - 1]` is the perft node count at depth `d`.
    pub counts: &'static [u64],
}

/// The six standard CPW perft positions.
pub const PERFT_FIXTURES: &[PerftFixture] = &[
    PerftFixture {
        name: "startpos",
        fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        source: "CPW Perft Results",
        counts: &[20, 400, 8902, 197281, 4865609, 119060324],
    },
    PerftFixture {
        name: "kiwipete",
        fen: "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        source: "CPW Perft Results (Position 2)",
        counts: &[48, 2039, 97862, 4085603, 193690690],
    },
    PerftFixture {
        name: "position3",
        fen: "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        source: "CPW Perft Results (Position 3)",
        counts: &[14, 191, 2812, 43238, 674624, 11030083],
    },
    PerftFixture {
        name: "position4",
        fen: "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        source: "CPW Perft Results (Position 4)",
        counts: &[6, 264, 9467, 422333, 15833292],
    },
    PerftFixture {
        name: "position5",
        fen: "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        source: "CPW Perft Results (Position 5)",
        counts: &[44, 1486, 62379, 2103487, 89941194],
    },
    PerftFixture {
        name: "position6",
        fen: "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        source: "CPW Perft Results (Position 6)",
        counts: &[46, 2079, 89890, 3894594, 164075551],
    },
];

/// A named tactical/edge-case position.
pub struct EdgeCase {
    pub name: &'static str,
    pub fen: &'static str,
    pub note: &'static str,
}

/// Hand-verified tactical edge cases. FENs are standard and cross-checked by
/// parsing with cozy-chess and by the assertions in `tests/edge_cases.rs`.
pub const EDGE_CASES: &[EdgeCase] = &[
    EdgeCase {
        name: "ep_pinned_pawn",
        fen: "8/8/8/8/k2Pp2Q/8/8/4K3 b - d3 0 1",
        note: "en passant that would expose the black king is illegal",
    },
    EdgeCase {
        name: "castling_through_check",
        fen: "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        note: "castling is available; through-check is filtered by movegen",
    },
    EdgeCase {
        name: "castling_into_check",
        fen: "4k3/8/8/8/8/8/8/4K2R w K - 0 1",
        note: "king on e1 may castle kingside when path is clear",
    },
    EdgeCase {
        name: "promotion_all",
        fen: "8/P6k/8/8/8/8/8/K7 w - - 0 1",
        note: "pawn on a7 with four promotion choices",
    },
    EdgeCase {
        name: "promotion_capture",
        fen: "1n5k/P7/8/8/8/8/8/K7 w - - 0 1",
        note: "promotion capture onto b8",
    },
    EdgeCase {
        name: "double_check",
        fen: "4k3/8/8/8/8/8/4R3/4RK2 b - - 0 1",
        note: "black king in double check",
    },
    EdgeCase {
        name: "checkmate_back_rank",
        fen: "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1",
        note: "back-rank mate available",
    },
    EdgeCase {
        name: "stalemate",
        fen: "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",
        note: "black to move is stalemated",
    },
    EdgeCase {
        name: "same_board_diff_rights",
        fen: "4k3/8/8/8/8/8/8/R3K2R w K - 0 1",
        note: "same placement as castling_into_check but different rights",
    },
    EdgeCase {
        name: "same_board_black_to_move",
        fen: "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
        note: "same placement as castling_through_check, black to move",
    },
];
