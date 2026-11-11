//! G0 (part 1): the retrograde tablebase is exact. Checked against known maxima and, independently,
//! against the repo's `MateSolver` (a different algorithm: memoised forward search).

use cozy_chess::{Color, Piece};
use recur64_prove::tb::Tablebase;
use recur64_runtime::proof::mate::MateSolver;

#[test]
fn krvk_has_the_known_maximum_of_16_moves() {
    let tb = Tablebase::build("KRvK", &[Piece::Rook]).unwrap();
    assert_eq!(tb.max_plies_white, 31, "KRvK longest mate is 16 moves = 31 plies");
    let won = tb.won_white_positions();
    assert!(won.len() > 50_000, "{} won KRvK positions", won.len());
    assert_eq!(won.iter().map(|(_, d)| *d).max(), Some(16));
}

#[test]
fn kqvk_has_the_known_maximum_of_10_moves() {
    let tb = Tablebase::build("KQvK", &[Piece::Queen]).unwrap();
    assert_eq!(tb.max_plies_white, 19, "KQvK longest mate is 10 moves = 19 plies");
    assert_eq!(tb.won_white_positions().iter().map(|(_, d)| *d).max(), Some(10));
}

#[test]
fn the_tablebase_agrees_with_the_independent_forward_solver_up_to_mate_in_3() {
    for (name, piece) in [("KQvK", Piece::Queen), ("KRvK", Piece::Rook)] {
        let tb = Tablebase::build(name, &[piece]).unwrap();
        let mut solver = MateSolver::new();
        let (mut checked, mut shallow) = (0usize, 0usize);
        for (b, d) in tb.won_white_positions().into_iter().step_by(17) {
            assert_eq!(b.side_to_move(), Color::White);
            let forward = solver.mate_depth(&b, 3);
            let expect = if d <= 3 { Some(d) } else { None };
            assert_eq!(forward, expect, "{name}: {b} tablebase depth {d}");
            checked += 1;
            shallow += usize::from(d <= 3);
        }
        assert!(checked > 1_000 && shallow > 100, "{name}: coverage {checked} checked, {shallow} shallow");
        // White to move always wins in these endings; the drawn cases are black-to-move positions where the piece can be taken.
        assert!(tb
            .positions
            .iter()
            .filter(|b| b.side_to_move() == Color::White && b.status() == cozy_chess::GameStatus::Ongoing)
            .all(|b| tb.mate_in(b).is_some()), "{name}: a white-to-move position was not a win");
        let (mut lost, mut held) = (0, 0);
        for b in tb.positions.iter().filter(|b| b.side_to_move() == Color::Black && b.status() == cozy_chess::GameStatus::Ongoing).step_by(97) {
            let forward = solver.defender_lost(b, 3);
            let expect = tb.plies_to_mate(b).is_some_and(|p| p / 2 <= 3);
            assert_eq!(forward, expect, "{name}: black to move {b}");
            lost += usize::from(forward);
            held += usize::from(!forward);
        }
        assert!(lost > 20 && held > 20, "{name}: coverage lost {lost}, held {held}");
    }
}
