//! Property/metamorphic tests (proptest + seeded random games).

use proptest::prelude::*;

use recur64_core::{
    ActionId, Color, GameState, OBS_LEN, Perspective, PromotionCode, Square, StandardMove,
    encode_observation_v1,
};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    #[test]
    fn action_encode_decode_roundtrip(a in 0u8..64, b in 0u8..64, p in 0u8..5) {
        let from = Square::index(a as usize);
        let to = Square::index(b as usize);
        let promo = PromotionCode::new(p).unwrap();
        let id = ActionId::encode(from, to, promo);
        let (f, t, pr) = id.decode();
        prop_assert_eq!((f, t, pr), (from, to, promo));
        prop_assert_eq!(ActionId::encode(f, t, pr), id);
    }

    #[test]
    fn physical_canonical_roundtrip(a in 0u8..64, b in 0u8..64, p in 0u8..5, black in any::<bool>()) {
        let from = Square::index(a as usize);
        let to = Square::index(b as usize);
        let promo = PromotionCode::new(p).unwrap();
        let side = if black { Color::Black } else { Color::White };
        let perspective = Perspective::of(side);
        let id = ActionId::from_physical(from, to, promo, perspective);
        let (f, t, pr) = id.to_physical(perspective);
        prop_assert_eq!((f, t, pr), (from, to, promo));
    }
}

struct SplitMix64(u64);
impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

fn pick_action(g: &GameState, rng: &mut SplitMix64) -> Option<StandardMove> {
    let actions = g.legal_actions();
    if actions.is_empty() {
        return None;
    }
    let id = actions[(rng.next_u64() as usize) % actions.len()];
    let (f, t, pr) = id.to_physical(g.perspective());
    let promo = if pr.is_none() { None } else { Some(pr) };
    Some(StandardMove::new(f, t, promo))
}

fn check_position(g: &GameState) {
    let moves = g.legal_standard_moves();
    let actions = g.legal_actions();
    assert_eq!(moves.len(), actions.len(), "action count mismatch");

    let mut unique = actions.clone();
    unique.dedup();
    assert_eq!(unique.len(), actions.len(), "duplicate actions");

    for id in &actions {
        let (f, t, pr) = id.to_physical(g.perspective());
        let promo = if pr.is_none() { None } else { Some(pr) };
        let mv = StandardMove::new(f, t, promo);
        let cozy = mv.to_cozy(g.board()).unwrap();
        assert!(g.board().is_legal(cozy), "action decoded to illegal move");
    }

    let obs = encode_observation_v1(g);
    assert_eq!(obs.as_slice().len(), OBS_LEN);
    assert!(obs.as_slice().iter().all(|v| v.is_finite()));
}

#[test]
fn random_games_are_self_consistent() {
    for seed in 0..400u64 {
        let mut rng = SplitMix64::new(seed);
        let mut g = GameState::startpos();
        for _ in 0..80 {
            check_position(&g);
            if g.is_terminal() {
                break;
            }
            match pick_action(&g, &mut rng) {
                Some(mv) => g.apply(mv).unwrap(),
                None => break,
            }
        }
    }
}
