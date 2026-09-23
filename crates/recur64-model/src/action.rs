//! Action-index convention and legal-candidate batching for the probe model.
//!
//! The action ID is a *storage/indexing* convention only. It is deliberately
//! **not** backed by a dense `hidden -> 20480` output layer. The policy path
//! produces scores from source/destination square representations and gathers
//! only the supplied legal candidates.
//!
//! Phase 1 made `recur64-core` the single source of truth for the action
//! encoding; the constants and functions below delegate to it while preserving
//! the Phase 0 public API.

use recur64_core::{ActionId, PromotionCode, Square};

/// Promotion code: no promotion.
pub const PROMO_NONE: u8 = recur64_core::PROMO_NONE;
/// Promotion code: knight.
pub const PROMO_N: u8 = recur64_core::PROMO_N;
/// Promotion code: bishop.
pub const PROMO_B: u8 = recur64_core::PROMO_B;
/// Promotion code: rook.
pub const PROMO_R: u8 = recur64_core::PROMO_R;
/// Promotion code: queen.
pub const PROMO_Q: u8 = recur64_core::PROMO_Q;

/// Number of promotion codes, including "none".
pub const PROMO_CODES: u32 = recur64_core::PROMO_CODES as u32;
/// Number of squares per side.
pub const SQUARES: u32 = recur64_core::NUM_SQUARES as u32;
/// Total action-index space: `64 * 64 * 5`.
pub const ACTION_SPACE: u32 = recur64_core::ACTION_SPACE;

/// Encode a move into the Recur64 action-index space.
///
/// Panics visibly on out-of-range squares or promotion codes.
pub fn action_id(from: u32, to: u32, promo: u8) -> u32 {
    let f = Square::index(from as usize);
    let t = Square::index(to as usize);
    let p = PromotionCode::new(promo).expect("valid promotion code");
    ActionId::encode(f, t, p).index()
}

/// Decode an action index back into `(from, to, promo)`.
///
/// Panics visibly on an out-of-range id.
pub fn decode_action_id(id: u32) -> (u32, u32, u8) {
    let a = ActionId::from_index(id).expect("valid action id");
    let (f, t, p) = a.decode();
    (f as u32, t as u32, p.code())
}

/// A padded batch of legal candidate lists.
///
/// Each position has `len` legal candidates. Entries at index `>= len` are
/// padding and must receive exactly zero probability. The legal list is never
/// truncated.
#[derive(Debug, Clone)]
pub struct CandidateBatch {
    /// Number of positions (rows).
    pub batch: usize,
    /// Padded candidate-list width (max legal count in this batch).
    pub width: usize,
    /// `from` square per candidate, row-major `[batch][width]`.
    pub from: Vec<u32>,
    /// `to` square per candidate.
    pub to: Vec<u32>,
    /// Promotion code per candidate (0..4).
    pub promo: Vec<u8>,
    /// Valid-candidate mask.
    pub mask: Vec<bool>,
    /// Legal candidate count per position.
    pub lens: Vec<usize>,
    /// Terminal positions (no legal candidates): policy softmax must be bypassed.
    pub terminal: Vec<bool>,
}

impl CandidateBatch {
    /// Build from per-position candidate vectors.
    pub fn from_lists(lists: &[Vec<(u32, u32, u8)>]) -> Self {
        let batch = lists.len();
        let width = lists.iter().map(|l| l.len()).max().unwrap_or(0);
        let mut from = vec![0u32; batch * width];
        let mut to = vec![0u32; batch * width];
        let mut promo = vec![0u8; batch * width];
        let mut mask = vec![false; batch * width];
        let mut lens = vec![0usize; batch];
        let mut terminal = vec![false; batch];
        for (b, list) in lists.iter().enumerate() {
            lens[b] = list.len();
            terminal[b] = list.is_empty();
            for (k, &(f, t, p)) in list.iter().enumerate() {
                let i = b * width + k;
                from[i] = f;
                to[i] = t;
                promo[i] = p;
                mask[i] = true;
            }
        }
        Self {
            batch,
            width,
            from,
            to,
            promo,
            mask,
            lens,
            terminal,
        }
    }

    /// Flat `from * 64 + to` index per candidate (used to gather base scores).
    pub fn base_index(&self) -> Vec<i32> {
        self.from
            .iter()
            .zip(self.to.iter())
            .map(|(&f, &t)| (f * SQUARES + t) as i32)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn action_roundtrip_and_promotion_distinctness() {
        // A non-promotion queen-side move and its queen-promotion counterpart
        // must map to distinct IDs.
        let quiet = action_id(12, 4, PROMO_NONE);
        let queen = action_id(12, 4, PROMO_Q);
        let knight = action_id(12, 4, PROMO_N);
        assert_ne!(quiet, queen);
        assert_ne!(queen, knight);
        for id in [quiet, queen, knight] {
            let (f, t, p) = decode_action_id(id);
            assert_eq!(action_id(f, t, p), id);
        }
    }

    #[test]
    fn action_space_is_20480() {
        assert_eq!(ACTION_SPACE, 20_480);
        assert_eq!(action_id(63, 63, PROMO_Q), ACTION_SPACE - 1);
    }

    #[test]
    fn matches_core_encoding() {
        // Drift guard: the model's encoding equals recur64-core's.
        for id in 0..ACTION_SPACE {
            let (f, t, p) = decode_action_id(id);
            assert_eq!(action_id(f, t, p), id);
            assert_eq!(id, recur64_core::ActionId::from_index(id).unwrap().index());
        }
    }

    #[test]
    fn candidate_batch_pads_without_truncating() {
        let lists = vec![vec![(0, 8, PROMO_NONE), (8, 16, PROMO_NONE)], vec![]];
        let cb = CandidateBatch::from_lists(&lists);
        assert_eq!(cb.width, 2);
        assert_eq!(cb.lens, vec![2, 0]);
        assert_eq!(cb.terminal, vec![false, true]);
        assert!(!cb.mask[2] && !cb.mask[3]);
    }
}
