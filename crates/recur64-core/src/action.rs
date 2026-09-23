//! Action V1 identity.
//!
//! `ActionId = ((from * 64 + to) * 5 + promotion_code)`, promotion codes
//! `{0=none,1=N,2=B,3=R,4=Q}`, total space 20,480. This is a storage/indexing
//! convention, **not** authorization for a dense `hidden -> 20,480` layer.
//!
//! The network's actions use the **same canonical (current-side-to-move)
//! orientation** as Observation V1. `ActionId` itself is orientation-neutral;
//! the `from_physical`/`to_physical` helpers apply the perspective.

use crate::error::CoreError;
use crate::square::{NUM_SQUARES, Perspective, Piece, Square};

/// Promotion code: no promotion.
pub const PROMO_NONE: u8 = 0;
/// Promotion code: knight.
pub const PROMO_N: u8 = 1;
/// Promotion code: bishop.
pub const PROMO_B: u8 = 2;
/// Promotion code: rook.
pub const PROMO_R: u8 = 3;
/// Promotion code: queen.
pub const PROMO_Q: u8 = 4;
/// Number of promotion codes, including "none".
pub const PROMO_CODES: u8 = 5;
/// Total action-index space: `64 * 64 * 5`.
pub const ACTION_SPACE: u32 = 64 * 64 * PROMO_CODES as u32; // 20,480

/// A validated promotion code in `0..5`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PromotionCode(u8);

impl PromotionCode {
    pub const NONE: PromotionCode = PromotionCode(PROMO_NONE);
    pub const N: PromotionCode = PromotionCode(PROMO_N);
    pub const B: PromotionCode = PromotionCode(PROMO_B);
    pub const R: PromotionCode = PromotionCode(PROMO_R);
    pub const Q: PromotionCode = PromotionCode(PROMO_Q);

    /// All codes in canonical order.
    pub const ALL: [PromotionCode; 5] = [
        PromotionCode::NONE,
        PromotionCode::N,
        PromotionCode::B,
        PromotionCode::R,
        PromotionCode::Q,
    ];

    /// Validate a raw code.
    pub fn new(code: u8) -> Result<Self, CoreError> {
        if (code as u32) < PROMO_CODES as u32 {
            Ok(PromotionCode(code))
        } else {
            Err(CoreError::InvalidPromotion(format!(
                "promotion code {code} out of range 0..5"
            )))
        }
    }

    pub const fn code(self) -> u8 {
        self.0
    }

    pub const fn is_none(self) -> bool {
        self.0 == PROMO_NONE
    }

    /// The promoted-to piece, if this is a real promotion.
    pub const fn piece(self) -> Option<Piece> {
        match self.0 {
            PROMO_N => Some(Piece::Knight),
            PROMO_B => Some(Piece::Bishop),
            PROMO_R => Some(Piece::Rook),
            PROMO_Q => Some(Piece::Queen),
            _ => None,
        }
    }

    /// Build from a piece; only N/B/R/Q are valid promotions.
    pub fn from_piece(piece: Piece) -> Result<Self, CoreError> {
        match piece {
            Piece::Knight => Ok(PromotionCode::N),
            Piece::Bishop => Ok(PromotionCode::B),
            Piece::Rook => Ok(PromotionCode::R),
            Piece::Queen => Ok(PromotionCode::Q),
            other => Err(CoreError::InvalidPromotion(format!(
                "piece {other:?} cannot be a promotion target"
            ))),
        }
    }
}

/// A validated action index in `0..20480`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ActionId(u16);

impl ActionId {
    /// The action-index space size.
    pub const SPACE: u32 = ACTION_SPACE;

    /// Encode squares in whatever orientation they are supplied. Prefer
    /// `from_physical` for chess moves.
    pub fn encode(from: Square, to: Square, promo: PromotionCode) -> Self {
        let idx = ((from as u32) * 64 + (to as u32)) * 5 + promo.code() as u32;
        debug_assert!(idx < ACTION_SPACE);
        ActionId(idx as u16)
    }

    /// Build from a raw index, validating the range.
    pub fn from_index(index: u32) -> Result<Self, CoreError> {
        if index < ACTION_SPACE {
            Ok(ActionId(index as u16))
        } else {
            Err(CoreError::InvalidAction(index))
        }
    }

    /// Encode a **physical** move from the given perspective into canonical
    /// action space.
    pub fn from_physical(
        from: Square,
        to: Square,
        promo: PromotionCode,
        perspective: Perspective,
    ) -> Self {
        Self::encode(perspective.square(from), perspective.square(to), promo)
    }

    pub const fn index(self) -> u32 {
        self.0 as u32
    }

    /// Decode into canonical squares and promotion code. Infallible for a
    /// constructed `ActionId`.
    pub fn decode(self) -> (Square, Square, PromotionCode) {
        let idx = self.index();
        let promo = PromotionCode((idx % 5) as u8);
        let rest = idx / 5;
        let to = Square::index((rest % 64) as usize);
        let from = Square::index((rest / 64) as usize);
        (from, to, promo)
    }

    /// Decode canonical action space into a **physical** move.
    pub fn to_physical(self, perspective: Perspective) -> (Square, Square, PromotionCode) {
        let (from, to, promo) = self.decode();
        (perspective.square(from), perspective.square(to), promo)
    }

    /// Convenience: the promotion code.
    pub const fn promo(self) -> PromotionCode {
        PromotionCode((self.0 as u32 % 5) as u8)
    }
}

/// Number of squares as `u32` (action-space arithmetic).
pub const SQUARES_U32: u32 = NUM_SQUARES as u32;

/// Maximum legal moves in any standard chess position is 218; 256 is a safe
/// fixed capacity so legal-candidate generation does not heap-allocate.
pub const MAX_LEGAL_MOVES: usize = 256;

/// A fixed-capacity list of canonical `ActionId`s.
#[derive(Clone)]
pub struct ActionList {
    ids: [ActionId; MAX_LEGAL_MOVES],
    len: u16,
}

impl Default for ActionList {
    fn default() -> Self {
        Self::new()
    }
}

impl ActionList {
    pub fn new() -> Self {
        Self {
            ids: [ActionId(0); MAX_LEGAL_MOVES],
            len: 0,
        }
    }

    pub fn push(&mut self, id: ActionId) {
        debug_assert!((self.len as usize) < MAX_LEGAL_MOVES);
        self.ids[self.len as usize] = id;
        self.len += 1;
    }

    pub fn len(&self) -> usize {
        self.len as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_slice(&self) -> &[ActionId] {
        &self.ids[..self.len as usize]
    }

    pub fn sort(&mut self) {
        self.ids[..self.len as usize].sort_unstable();
    }

    pub fn iter(&self) -> impl Iterator<Item = ActionId> + '_ {
        self.as_slice().iter().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::square::Color;

    #[test]
    fn action_space_is_20480() {
        assert_eq!(ACTION_SPACE, 20_480);
        let last = ActionId::encode(Square::H8, Square::H8, PromotionCode::Q);
        assert_eq!(last.index(), ACTION_SPACE - 1);
        assert!(
            ActionId::from_index(ACTION_SPACE)
                .unwrap_err()
                .to_string()
                .contains("20480")
        );
    }

    #[test]
    fn encode_decode_roundtrip_full_space() {
        for i in 0..ACTION_SPACE {
            let id = ActionId::from_index(i).unwrap();
            let (f, t, p) = id.decode();
            assert_eq!(ActionId::encode(f, t, p), id, "roundtrip at {i}");
        }
    }

    #[test]
    fn promotion_codes_distinct() {
        let quiet = ActionId::encode(Square::A7, Square::A8, PromotionCode::NONE);
        let queen = ActionId::encode(Square::A7, Square::A8, PromotionCode::Q);
        let knight = ActionId::encode(Square::A7, Square::A8, PromotionCode::N);
        assert_ne!(quiet, queen);
        assert_ne!(queen, knight);
        assert_ne!(knight, quiet);
    }

    #[test]
    fn invalid_promotion_code_fails() {
        assert!(PromotionCode::new(5).is_err());
        assert!(PromotionCode::from_piece(Piece::King).is_err());
        assert!(PromotionCode::from_piece(Piece::Pawn).is_err());
    }

    #[test]
    fn physical_canonical_roundtrip_both_colors() {
        for i in 0..64u8 {
            let from = Square::index(i as usize);
            for j in 0..64u8 {
                let to = Square::index(j as usize);
                for promo in PromotionCode::ALL {
                    for side in [Color::White, Color::Black] {
                        let p = Perspective::of(side);
                        let id = ActionId::from_physical(from, to, promo, p);
                        let (f, t, pr) = id.to_physical(p);
                        assert_eq!((f, t, pr), (from, to, promo));
                    }
                }
            }
        }
    }

    #[test]
    fn castling_actions_symmetric_across_colors() {
        // White O-O (e1->g1) and Black O-O (e8->g8) canonicalize to the same ID.
        let white = ActionId::from_physical(
            Square::E1,
            Square::G1,
            PromotionCode::NONE,
            Perspective::white(),
        );
        let black = ActionId::from_physical(
            Square::E8,
            Square::G8,
            PromotionCode::NONE,
            Perspective::black(),
        );
        assert_eq!(white, black);
    }
}
