//! Perft against published Chess Programming Wiki counts.
//!
//! These counts are an independent oracle (not cozy-chess's own tests). CI uses
//! bounded depths; deeper depths are marked `#[ignore]` for manual runs.

use cozy_chess::Board;

use recur64_core::fixtures::PERFT_FIXTURES;
use recur64_core::perft::perft;

fn ci_depth(name: &str) -> u32 {
    match name {
        "startpos" => 4,
        "kiwipete" => 3,
        "position3" => 4,
        "position4" => 3,
        "position5" => 3,
        "position6" => 3,
        _ => 2,
    }
}

#[test]
fn perft_matches_published_counts_ci() {
    for f in PERFT_FIXTURES {
        let b: Board = f.fen.parse().unwrap();
        let d = ci_depth(f.name);
        let expected = f.counts[(d - 1) as usize];
        let got = perft(&b, d);
        assert_eq!(
            got, expected,
            "{} at depth {d} (source: {})",
            f.name, f.source
        );
    }
}

#[test]
#[ignore = "deeper perft; run with --ignored --release"]
fn perft_deep_startpos() {
    let b = Board::default();
    assert_eq!(perft(&b, 5), 4_865_609);
}
