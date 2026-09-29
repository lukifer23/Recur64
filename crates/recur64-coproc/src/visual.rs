//! `VisualBoardV1`: a canonical, top-down, procedurally drawn board image.
//!
//! Properties (all enforced by tests):
//!
//! * top-down, no perspective, no lighting, no shadows to speak of;
//! * **canonical current-player orientation** — the renderer draws the
//!   canonical Observation V1 board, so the side to move is always at the
//!   bottom of the image and both colours render the same way;
//! * deterministic: the same position always produces the same pixels;
//! * no system fonts, no textures, no random rotation and no network assets —
//!   every shape is integer arithmetic;
//! * piece *type* changes the silhouette and piece *ownership* changes the fill
//!   luminance, so the two are separable in grayscale.

use cozy_chess::{Color, Piece, Square};

use crate::CoprocError;
use crate::board::reconstruct;
use crate::input::CoprocInput;

/// Image sizes the renderer supports (pixels per side).
pub const SUPPORTED_SIZES: [usize; 2] = [64, 96];

/// Bytes per `size x size` RGB image.
pub const fn image_bytes(size: usize) -> usize {
    size * size * 3
}

const LIGHT_SQUARE: [u8; 3] = [238, 238, 208];
const DARK_SQUARE: [u8; 3] = [112, 148, 96];

/// Piece fill/outline for one side. Own pieces are light with a dark outline;
/// opponent pieces are dark with a light outline.
fn palette(color: Color) -> ([u8; 3], [u8; 3]) {
    match color {
        Color::White => ([242, 242, 234], [26, 26, 30]),
        Color::Black => ([30, 30, 36], [232, 232, 224]),
    }
}

/// Whether the local point `(x, y)` (cell-relative, `-1..1` across the cell) is
/// inside the silhouette of `piece` grown by `s`.
fn inside(piece: Piece, x: f32, y: f32, s: f32) -> bool {
    // Shared pedestal so every piece has a base.
    let pedestal = x.abs() <= 0.80 * s && y >= 0.50 * s && y <= 0.92 * s;
    let body = match piece {
        Piece::Pawn => (x * x + y * y).sqrt() <= 0.62 * s,
        Piece::Knight => {
            // A slanted head: box trimmed on the top-right and bottom-left.
            x.abs() <= 0.72 * s
                && y.abs() <= 0.80 * s
                && (x + y) <= 0.55 * s
                && (x - y) >= -0.95 * s
        }
        Piece::Bishop => x.abs() + y.abs() <= 0.80 * s,
        Piece::Rook => x.abs() <= 0.72 * s && y.abs() <= 0.80 * s,
        Piece::Queen => {
            let disc = (x * x + y * y).sqrt() <= 0.66 * s;
            // Three crown points above the disc.
            let crown = [(0.0f32, -0.85f32), (-0.55, -0.70), (0.55, -0.70)]
                .iter()
                .any(|(cx, cy)| {
                    let (dx, dy) = (x - cx * s, y - cy * s);
                    (dx * dx + dy * dy).sqrt() <= 0.30 * s
                });
            disc || crown
        }
        Piece::King => {
            let disc = (x * x + y * y).sqrt() <= 0.62 * s;
            let vertical = x.abs() <= 0.16 * s && y.abs() <= 0.90 * s;
            let horizontal = y.abs() <= 0.16 * s && x.abs() <= 0.55 * s;
            disc || vertical || horizontal
        }
    };
    body || pedestal
}

/// Render the canonical board of `input` into an RGB buffer.
pub fn render_board(input: &[u8], out: &mut [u8], size: usize) -> Result<(), CoprocError> {
    if !SUPPORTED_SIZES.contains(&size) {
        return Err(CoprocError::BadImageSize(size));
    }
    if out.len() != image_bytes(size) {
        return Err(CoprocError::BadOutputLength(out.len()));
    }
    let inp = CoprocInput::new(input)?;
    let rec = reconstruct(&inp)?;
    let cell = size / 8;
    let half = cell as f32 / 2.0;

    for py in 0..size {
        for px in 0..size {
            let file = px / cell;
            // Canonical orientation: rank 0 (own back rank) at the bottom.
            let rank = 7 - py / cell;
            let dark = (file + rank).is_multiple_of(2);
            let mut rgb = if dark { DARK_SQUARE } else { LIGHT_SQUARE };

            let square = Square::new(file_of(file), rank_of(rank));
            if let (Some(piece), Some(color)) =
                (rec.board.piece_on(square), rec.board.color_on(square))
            {
                let cx = (px % cell) as f32 + 0.5 - half;
                let cy = (py % cell) as f32 + 0.5 - half;
                let x = cx / half;
                let y = cy / half;
                let (fill, outline) = palette(color);
                if inside(piece, x, y, 1.16) {
                    rgb = if inside(piece, x, y, 1.0) {
                        fill
                    } else {
                        outline
                    };
                }
            }

            let o = (py * size + px) * 3;
            out[o..o + 3].copy_from_slice(&rgb);
        }
    }
    Ok(())
}

fn file_of(index: usize) -> cozy_chess::File {
    const FILES: [cozy_chess::File; 8] = [
        cozy_chess::File::A,
        cozy_chess::File::B,
        cozy_chess::File::C,
        cozy_chess::File::D,
        cozy_chess::File::E,
        cozy_chess::File::F,
        cozy_chess::File::G,
        cozy_chess::File::H,
    ];
    FILES[index]
}

fn rank_of(index: usize) -> cozy_chess::Rank {
    const RANKS: [cozy_chess::Rank; 8] = [
        cozy_chess::Rank::First,
        cozy_chess::Rank::Second,
        cozy_chess::Rank::Third,
        cozy_chess::Rank::Fourth,
        cozy_chess::Rank::Fifth,
        cozy_chess::Rank::Sixth,
        cozy_chess::Rank::Seventh,
        cozy_chess::Rank::Eighth,
    ];
    RANKS[index]
}

/// The canonical-square index a pixel belongs to (used by tests and probes).
pub fn square_at(pixel: usize, size: usize) -> usize {
    let cell = size / 8;
    let (px, py) = (pixel % size, pixel / size);
    let file = px / cell;
    let rank = 7 - py / cell;
    rank * 8 + file
}

/// Mean luminance of the pixels belonging to one canonical square.
pub fn square_luminance(image: &[u8], size: usize, square: usize) -> f32 {
    let cell = size / 8;
    let file = square % 8;
    let rank = square / 8;
    let (x0, y0) = (file * cell, (7 - rank) * cell);
    let mut sum = 0u64;
    let mut n = 0u64;
    for y in y0..y0 + cell {
        for x in x0..x0 + cell {
            let o = (y * size + x) * 3;
            sum += image[o] as u64 + image[o + 1] as u64 + image[o + 2] as u64;
            n += 3;
        }
    }
    sum as f32 / n as f32 / 255.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{encode_observation, legal_moves, state_from_fen};

    fn input_for(fen: &str) -> Vec<u8> {
        let state = state_from_fen(fen);
        crate::input::write_input(&encode_observation(&state), &legal_moves(&state), 0).unwrap()
    }

    fn render(fen: &str, size: usize) -> Vec<u8> {
        let input = input_for(fen);
        let mut out = vec![0u8; image_bytes(size)];
        render_board(&input, &mut out, size).unwrap();
        out
    }

    #[test]
    fn same_position_same_pixels() {
        let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        for size in SUPPORTED_SIZES {
            assert_eq!(render(fen, size), render(fen, size));
        }
    }

    #[test]
    fn image_size_is_exact_and_finite() {
        for size in SUPPORTED_SIZES {
            let img = render("8/8/8/4k3/8/8/3Q4/4K3 w - - 0 1", size);
            assert_eq!(img.len(), image_bytes(size));
        }
        let input = input_for("8/8/8/4k3/8/8/3Q4/4K3 w - - 0 1");
        let mut out = vec![0u8; 100];
        assert!(matches!(
            render_board(&input, &mut out, 64),
            Err(CoprocError::BadOutputLength(_))
        ));
        let mut out = vec![0u8; image_bytes(48)];
        assert!(matches!(
            render_board(&input, &mut out, 48),
            Err(CoprocError::BadImageSize(48))
        ));
    }

    #[test]
    fn board_orientation_is_canonical() {
        // Own back rank (rank 0) is at the bottom of the image. A lone own rook
        // on a1 and a lone opponent rook on a8: the bottom-left square must be
        // brighter than the top-left one (own pieces are the light fill).
        let fen = "r6k/8/8/8/8/8/8/R6K w - - 0 1";
        let size = 64;
        let img = render(fen, size);
        let bottom_left = square_luminance(&img, size, 0); // a1, own rook
        let top_left = square_luminance(&img, size, 56); // a8, opponent rook
        assert!(
            bottom_left > top_left,
            "own (bottom) piece must be brighter than the opponent (top) piece: {bottom_left} vs {top_left}"
        );
    }

    #[test]
    fn both_colours_render_the_same_canonical_image() {
        // A position and its colour-flipped twin have the same *canonical*
        // observation, so they must render identical pixels.
        let as_white = input_for("4k3/8/8/8/8/8/8/4K2Q w - - 0 1");
        let flipped = input_for("4k2q/8/8/8/8/8/8/4K3 b - - 0 1");
        let mut a = vec![0u8; image_bytes(64)];
        let mut b = vec![0u8; image_bytes(64)];
        render_board(&as_white, &mut a, 64).unwrap();
        render_board(&flipped, &mut b, 64).unwrap();
        assert_eq!(a, b, "canonical orientation must be colour-independent");
    }

    #[test]
    fn piece_types_render_differently() {
        // One own piece on d4 per variant; the silhouettes must differ.
        let variants = [
            "8/8/8/8/3P4/8/8/K6k w - - 0 1",
            "8/8/8/8/3N4/8/8/K6k w - - 0 1",
            "8/8/8/8/3B4/8/8/K6k w - - 0 1",
            "8/8/8/8/3R4/8/8/K6k w - - 0 1",
            "8/8/8/8/3Q4/8/8/K6k w - - 0 1",
        ];
        let d4 = 27usize; // rank 3 (index 3) * 8 + file 3
        let mut seen: Vec<Vec<u8>> = Vec::new();
        for fen in variants {
            let img = render(fen, 64);
            let cell = 64 / 8;
            let x0 = (d4 % 8) * cell;
            let y0 = (7 - d4 / 8) * cell;
            let mut patch = Vec::new();
            for y in y0..y0 + cell {
                for x in x0..x0 + cell {
                    let o = (y * 64 + x) * 3;
                    patch.extend_from_slice(&img[o..o + 3]);
                }
            }
            assert!(
                !seen.contains(&patch),
                "piece type must render a distinct silhouette: {fen}"
            );
            seen.push(patch);
        }
    }

    #[test]
    fn side_ownership_is_visible_in_grayscale() {
        // Same canonical square, same side to move, opposite owners: the images
        // differ and the own piece is the brighter one.
        let own = render("8/8/8/8/3Q4/8/8/K6k w - - 0 1", 64);
        let opp = render("8/8/8/8/3q4/8/8/K6k w - - 0 1", 64);
        assert_ne!(own, opp);
        let size = 64;
        let own_l = square_luminance(&own, size, 27);
        let opp_l = square_luminance(&opp, size, 27);
        assert!(own_l > opp_l, "{own_l} vs {opp_l}");
    }

    #[test]
    fn pixel_to_square_mapping_is_consistent() {
        for size in SUPPORTED_SIZES {
            let cell = size / 8;
            for sq in 0..crate::SQUARES {
                let px = (sq % 8) * cell;
                let py = (7 - sq / 8) * cell;
                assert_eq!(square_at(py * size + px, size), sq);
            }
        }
    }
}
