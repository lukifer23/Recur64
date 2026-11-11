//! Exact retrograde tablebase for "white pieces + king vs bare king" (distance to mate in plies).
//!
//! This is the LABEL SOURCE and the GRADER for V500. It is never consulted by the search under test.
//! Rules match the repo's `MateSolver` contract: checkmate ends the game; stalemate and insufficient
//! material are draws; there is no history, repetition or fifty-move accounting (so distances are the
//! standard "DTM" values, e.g. KRvK 16 moves = 31 plies, KQvK 10 moves = 19 plies).

use std::collections::HashMap;

use cozy_chess::{Board, BoardBuilder, Color, GameStatus, Move, Piece};


const UNKNOWN: u8 = 255;

pub fn moves_of(b: &Board) -> Vec<Move> {
    let mut v = Vec::with_capacity(48);
    b.generate_moves(|m| {
        v.extend(m);
        false
    });
    v
}

/// Plies to mate for every position in which the side to move is lost. `Attacker` = white.
pub struct Tablebase {
    pub name: String,
    /// `board.hash()` -> plies to checkmate (white to move: odd, black to move: even, 0 = checkmated).
    pub plies: HashMap<u64, u8>,
    /// Every legal position of the material (both sides to move), in enumeration order.
    pub positions: Vec<Board>,
    pub max_plies_white: u8,
}

fn place(
    pieces: &[Piece],
    idx: usize,
    used: &mut Vec<usize>,
    squares: &mut Vec<usize>,
    out: &mut Vec<Vec<usize>>,
) {
    if idx == pieces.len() {
        out.push(squares.clone());
        return;
    }
    for sq in 0..64 {
        if used.contains(&sq) {
            continue;
        }
        used.push(sq);
        squares.push(sq);
        place(pieces, idx + 1, used, squares, out);
        squares.pop();
        used.pop();
    }
}

impl Tablebase {
    /// Build the tablebase for white `extra` pieces (plus the white king) against the black king.
    pub fn build(name: &str, extra: &[Piece]) -> anyhow::Result<Self> {
        // Piece list: white king, extras, black king.
        let mut pieces = vec![Piece::King];
        pieces.extend_from_slice(extra);
        pieces.push(Piece::King);
        let colors: Vec<Color> = (0..pieces.len())
            .map(|i| if i + 1 == pieces.len() { Color::Black } else { Color::White })
            .collect();
        let mut placements = Vec::new();
        place(&pieces, 0, &mut Vec::new(), &mut Vec::new(), &mut placements);

        let mut positions: Vec<Board> = Vec::new();
        for pl in &placements {
            for stm in [Color::White, Color::Black] {
                let mut bb = BoardBuilder::empty();
                for (i, &sq) in pl.iter().enumerate() {
                    bb.board[sq] = Some((pieces[i], colors[i]));
                }
                bb.side_to_move = stm;
                // cozy-chess accepts adjacent kings and positions where the side NOT to move is in check, so
                // legality is enforced here: kings must not touch and the opponent of the mover must not be attacked.
                let (wk, bk) = (pl[0], pl[pl.len() - 1]);
                let (dr, df) = ((wk / 8) as i32 - (bk / 8) as i32, (wk % 8) as i32 - (bk % 8) as i32);
                if dr.abs() <= 1 && df.abs() <= 1 {
                    continue;
                }
                let Ok(b) = bb.build() else { continue };
                let mut flipped = bb.clone();
                flipped.side_to_move = !stm;
                if let Ok(f) = flipped.build() {
                    if !f.checkers().is_empty() {
                        continue; // the side that just moved left its king attacked? i.e. the non-mover is in check
                    }
                }
                positions.push(b);
            }
        }
        let index: HashMap<u64, u32> = positions
            .iter()
            .enumerate()
            .map(|(i, b)| (b.hash(), i as u32))
            .collect();
        anyhow::ensure!(index.len() == positions.len(), "board hash collision inside {name}");

        // Children (positions after each legal move) as indices; a missing child is a drawn dead position.
        let n = positions.len();
        let mut children: Vec<Vec<u32>> = Vec::with_capacity(n);
        let mut complete: Vec<bool> = Vec::with_capacity(n);
        let mut val = vec![UNKNOWN; n];
        for (i, b) in positions.iter().enumerate() {
            let mv = moves_of(b);
            let mut ch = Vec::with_capacity(mv.len());
            let mut all_known = true;
            for m in &mv {
                let mut a = b.clone();
                a.play_unchecked(*m);
                match index.get(&a.hash()) {
                    Some(&j) => ch.push(j),
                    None => all_known = false, // capture to a dead position: never a forced mate
                }
            }
            if b.side_to_move() == Color::Black && b.status() == GameStatus::Won {
                val[i] = 0; // black is checkmated
            }
            children.push(ch);
            complete.push(all_known && !mv.is_empty());
        }

        // Level-synchronous retrograde analysis. Odd levels: white to move (mate in `level` plies if some move
        // reaches a black position of value level-1). Even levels: black to move (every reply is a known white
        // mate; value = 1 + max child). Stop after two consecutive empty levels (one of each parity).
        let mut level = 0u8;
        let mut empty = 0;
        while empty < 2 {
            level += 1;
            anyhow::ensure!(level < 250, "retrograde analysis did not terminate");
            let mut assign: Vec<usize> = Vec::new();
            for i in 0..n {
                if val[i] != UNKNOWN {
                    continue;
                }
                if level % 2 == 1 {
                    if positions[i].side_to_move() == Color::White
                        && children[i].iter().any(|&j| val[j as usize] == level - 1)
                    {
                        assign.push(i);
                    }
                } else if positions[i].side_to_move() == Color::Black && complete[i] {
                    let mx = children[i].iter().map(|&j| val[j as usize]).max().unwrap_or(UNKNOWN);
                    if mx != UNKNOWN && mx + 1 == level {
                        assign.push(i);
                    }
                }
            }
            empty = if assign.is_empty() { empty + 1 } else { 0 };
            for i in assign {
                val[i] = level;
            }
        }

        let mut plies = HashMap::new();
        let mut max_w = 0u8;
        for (i, b) in positions.iter().enumerate() {
            if val[i] != UNKNOWN {
                plies.insert(b.hash(), val[i]);
                if b.side_to_move() == Color::White {
                    max_w = max_w.max(val[i]);
                }
            }
        }
        Ok(Self {
            name: name.to_string(),
            plies,
            positions,
            max_plies_white: max_w,
        })
    }

    /// Plies to mate with best play from both sides, or `None` if the side to move cannot be forced to lose.
    pub fn plies_to_mate(&self, b: &Board) -> Option<u8> {
        self.plies.get(&b.hash()).copied()
    }

    /// Attacker moves needed (white to move): `(plies + 1) / 2`.
    pub fn mate_in(&self, b: &Board) -> Option<u8> {
        debug_assert_eq!(b.side_to_move(), Color::White);
        self.plies_to_mate(b).map(|p| p.div_ceil(2))
    }

    /// White-to-move positions that are won, with their optimal attacker-move depth.
    pub fn won_white_positions(&self) -> Vec<(Board, u8)> {
        self.positions
            .iter()
            .filter(|b| b.side_to_move() == Color::White && b.status() == GameStatus::Ongoing)
            .filter_map(|b| self.mate_in(b).map(|d| (b.clone(), d)))
            .collect()
    }
}
