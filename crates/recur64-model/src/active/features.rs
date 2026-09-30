//! Extrapolatable scalar features for the active-search planner and selector.
//!
//! The model is trained through budget 8 and evaluated at 16, so no feature may
//! be a learned table with a fixed vocabulary. Every quantity (ply depth,
//! remaining budget) enters through fixed functions of the scalar: a linear
//! term, a log term and Fourier terms. Values beyond the supported range are
//! refused, never clipped.

use crate::config::{ACTIVE_MAX_BUDGET, ACTIVE_MAX_DEPTH};

/// Fixed features per scalar.
pub const SCALAR_FEATS: usize = 8;
/// Selector edge features: child depth, remaining budget, parent parity.
pub const EDGE_FEATS: usize = SCALAR_FEATS * 2 + 1;
/// Planner event features: edge features plus child terminal and child in-check.
pub const EVENT_FEATS: usize = EDGE_FEATS + 2;
/// STOP head features: remaining budget only.
pub const STOP_FEATS: usize = SCALAR_FEATS;

/// Fixed, parameter-free encoding of a non-negative scalar.
pub fn scalar(x: f32) -> [f32; SCALAR_FEATS] {
    use std::f32::consts::PI;
    [
        x / 16.0,
        (1.0 + x).ln() / 17.0_f32.ln(),
        (PI * x / 32.0).sin(),
        (PI * x / 32.0).cos(),
        (PI * x / 8.0).sin(),
        (PI * x / 8.0).cos(),
        (PI * x / 2.0).sin(),
        (PI * x / 2.0).cos(),
    ]
}

/// Features of a frontier edge: the depth of the node it would create, the
/// remaining budget, and the parity of the parent's ply (who moves there,
/// relative to the root player: 0 = the root player).
pub fn edge_features(
    child_depth: u32,
    remaining: usize,
    parent_parity: u32,
) -> anyhow::Result<[f32; EDGE_FEATS]> {
    anyhow::ensure!(
        child_depth as usize <= ACTIVE_MAX_DEPTH,
        "query depth {child_depth} exceeds the supported range {ACTIVE_MAX_DEPTH}; refusing to clip"
    );
    anyhow::ensure!(
        remaining <= ACTIVE_MAX_BUDGET,
        "remaining budget {remaining} exceeds the supported range {ACTIVE_MAX_BUDGET}"
    );
    let mut out = [0.0f32; EDGE_FEATS];
    out[..SCALAR_FEATS].copy_from_slice(&scalar(child_depth as f32));
    out[SCALAR_FEATS..2 * SCALAR_FEATS].copy_from_slice(&scalar(remaining as f32));
    out[2 * SCALAR_FEATS] = (parent_parity % 2) as f32;
    Ok(out)
}

/// Planner event features: edge features plus the child's terminal and check flags.
pub fn event_features(
    child_depth: u32,
    remaining: usize,
    parent_parity: u32,
    child_terminal: bool,
    child_in_check: bool,
) -> anyhow::Result<[f32; EVENT_FEATS]> {
    let e = edge_features(child_depth, remaining, parent_parity)?;
    let mut out = [0.0f32; EVENT_FEATS];
    out[..EDGE_FEATS].copy_from_slice(&e);
    out[EDGE_FEATS] = f32::from(u8::from(child_terminal));
    out[EDGE_FEATS + 1] = f32::from(u8::from(child_in_check));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn features_are_finite_and_extrapolate_past_the_training_range() {
        for x in 0..=64 {
            assert!(scalar(x as f32).iter().all(|v| v.is_finite()));
        }
        // 16 was never trained on, yet has a well-defined encoding that is
        // continuous with 8.
        let (a, b) = (scalar(8.0), scalar(16.0));
        assert_ne!(a, b);
    }

    #[test]
    fn out_of_range_inputs_refuse_instead_of_clipping() {
        assert!(edge_features(ACTIVE_MAX_DEPTH as u32, 0, 0).is_ok());
        assert!(edge_features(ACTIVE_MAX_DEPTH as u32 + 1, 0, 0).is_err());
        assert!(edge_features(1, ACTIVE_MAX_BUDGET + 1, 0).is_err());
    }

    #[test]
    fn parity_is_root_relative_and_binary() {
        assert_eq!(edge_features(1, 4, 0).unwrap()[2 * SCALAR_FEATS], 0.0);
        assert_eq!(edge_features(2, 4, 1).unwrap()[2 * SCALAR_FEATS], 1.0);
        assert_eq!(edge_features(3, 4, 2).unwrap()[2 * SCALAR_FEATS], 0.0);
    }
}
