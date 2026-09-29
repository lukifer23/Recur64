//! Index-to-square conversion shared by the compute bank and the renderer.

use cozy_chess::Square;

/// `cozy-chess` square from an index.
pub trait SquareIndex {
    /// The square for this canonical index.
    #[allow(clippy::wrong_self_convention)]
    fn from_index(self) -> Square;
}

impl SquareIndex for usize {
    #[inline]
    fn from_index(self) -> Square {
        Square::index(self)
    }
}

impl SquareIndex for u8 {
    #[inline]
    fn from_index(self) -> Square {
        Square::index(self as usize)
    }
}

impl SquareIndex for i32 {
    #[inline]
    fn from_index(self) -> Square {
        Square::index(self as usize)
    }
}

/// The canonical index of a square (`a1 = 0`).
#[inline]
pub fn index_of(square: Square) -> usize {
    square as usize
}
