//! The frozen V420 gates (`docs/V420_RESEARCH_PLAN.md` section 5, S4 as amended by V420-E1).
//! Pure functions over per-position top-1 vectors, so they are tested without a GPU and applied
//! exactly once by `recur64 v420 measure`.

use recur64_v4::stats::{Ci, paired_bootstrap};
use serde_json::{Value, json};

/// Pre-registered bootstrap seed for the S1 and S3 intervals.
pub const BOOT_S1: u64 = 0x7420_0001;
pub const BOOT_S3: u64 = 0x7420_0002;

pub const S1_MIN_GAIN: f64 = 0.03;
pub const S2_K8_TOLERANCE: f64 = 0.01;
pub const S2_K16_TOLERANCE: f64 = 0.02;
pub const S3_FLOOR: f64 = -0.02;
pub const S4_TOLERANCE: f64 = 0.01;
/// V4 Stage A, seed 5101, all of V4_TRAIN_DEV (`docs/evidence/v4/stage-a-seed5101.json`).
pub const S4_V4_STAGE_A_TOP1_SEED5101: f64 = 0.8403361344537815;

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len().max(1) as f64
}

/// Pooled (over seeds) mean of per-seed per-position vectors.
pub fn pooled_mean(per_seed: &[Vec<f64>]) -> f64 {
    mean(&per_seed.iter().map(|v| mean(v)).collect::<Vec<_>>())
}

/// Element-wise difference `a - b` per seed.
pub fn diff(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    a.iter()
        .zip(b)
        .map(|(x, y)| x.iter().zip(y).map(|(p, q)| p - q).collect())
        .collect()
}

fn ci_json(ci: &Ci) -> Value {
    json!({"mean": ci.mean, "lo": ci.lo, "hi": ci.hi, "per_seed": ci.per_seed,
           "wholly_positive": ci.wholly_positive()})
}

#[derive(Debug, Clone)]
pub struct GateInputs<'a> {
    /// Headline-cell per-position top-1, `[seed][position]`, same positions in the same order.
    pub r1: &'a [Vec<f64>],
    pub rk4: &'a [Vec<f64>],
    pub rv_k4: &'a [Vec<f64>],
    pub rv_k8: &'a [Vec<f64>],
    pub rv_k16: &'a [Vec<f64>],
    pub wide: &'a [Vec<f64>],
    /// BASE8 seed-5101 top-1 over all of V4_TRAIN_DEV, if it was run.
    pub base8_top1_all_dev: Option<f64>,
    /// S0: every run completed, finite, healthy.
    pub s0_ok: bool,
}

/// Apply S0-S4 once. Returns the machine-readable gate report and the verdict string.
pub fn apply(g: &GateInputs<'_>) -> anyhow::Result<(Value, &'static str)> {
    let s1_ci = paired_bootstrap(&diff(g.rk4, g.r1), BOOT_S1)?;
    let s1_pass = s1_ci.wholly_positive() && s1_ci.mean >= S1_MIN_GAIN;

    let (rv4, rv8, rv16) = (pooled_mean(g.rv_k4), pooled_mean(g.rv_k8), pooled_mean(g.rv_k16));
    let s2_k8 = rv8 >= rv4 - S2_K8_TOLERANCE;
    let s2_k16 = rv16 >= rv4 - S2_K16_TOLERANCE;
    let s2_pass = s2_k8 && s2_k16;

    let s3_ci = paired_bootstrap(&diff(g.rk4, g.wide), BOOT_S3)?;
    let s3_ok = s3_ci.mean >= S3_FLOOR;

    let s4 = g
        .base8_top1_all_dev
        .map(|t| (t, (t - S4_V4_STAGE_A_TOP1_SEED5101).abs() <= S4_TOLERANCE));

    let verdict = if !g.s0_ok {
        "S0_FAIL_FIX_PLUMBING_ONCE"
    } else if s4.is_some_and(|(_, ok)| !ok) {
        "S4_FAIL_PIPELINE_NOT_VALIDATED"
    } else if !s1_pass {
        "STOP_DEPTH_DOES_NOT_HELP"
    } else if !s2_pass {
        "HELPS_AT_TRAINED_DEPTH_NO_EXTRAPOLATION"
    } else {
        "PASS_S1_S2"
    };
    let report = json!({
        "schema": "v420_gates_v1",
        "S0": {"pass": g.s0_ok},
        "S1": {"rule": "CI(RK4 - R1) wholly > 0, every seed > 0, pooled gain >= 0.03", "ci": ci_json(&s1_ci),
               "min_gain": S1_MIN_GAIN, "pass": s1_pass},
        "S2": {"rule": "RV top-1 at K=8 >= K=4 - 0.01 and K=16 >= K=4 - 0.02 (pooled)",
               "rv_k4": rv4, "rv_k8": rv8, "rv_k16": rv16, "k8_ok": s2_k8, "k16_ok": s2_k16, "pass": s2_pass},
        "S3": {"rule": "RK4 - WIDE >= -0.02 (reported, not a kill)", "ci": ci_json(&s3_ci),
               "floor": S3_FLOOR, "ok": s3_ok},
        "S4": s4.map(|(t, ok)| json!({"base8_top1": t, "v4_stage_a_5101": S4_V4_STAGE_A_TOP1_SEED5101,
                                    "tolerance": S4_TOLERANCE, "pass": ok})),
        "verdict": verdict,
    });
    Ok((report, verdict))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seeds(base: f64, n: usize, bump: &[f64]) -> Vec<Vec<f64>> {
        bump.iter()
            .map(|b| {
                (0..n)
                    .map(|i| if (i as f64) < (base + b) * n as f64 { 1.0 } else { 0.0 })
                    .collect()
            })
            .collect()
    }

    fn inputs<'a>(
        r1: &'a [Vec<f64>],
        rk4: &'a [Vec<f64>],
        flat: &'a [Vec<f64>],
        base8: Option<f64>,
    ) -> GateInputs<'a> {
        GateInputs {
            r1,
            rk4,
            rv_k4: flat,
            rv_k8: flat,
            rv_k16: flat,
            wide: flat,
            base8_top1_all_dev: base8,
            s0_ok: true,
        }
    }

    #[test]
    fn a_flat_depth_curve_stops_v420() {
        let r1 = seeds(0.60, 500, &[0.0, 0.0, 0.0]);
        let (rep, v) = apply(&inputs(&r1, &r1, &r1, Some(0.84))).unwrap();
        assert_eq!(v, "STOP_DEPTH_DOES_NOT_HELP");
        assert_eq!(rep["S1"]["pass"], false);
    }

    #[test]
    fn a_clear_gain_in_every_seed_passes_s1_and_s2() {
        let r1 = seeds(0.60, 500, &[0.0, 0.0, 0.0]);
        let rk4 = seeds(0.60, 500, &[0.08, 0.07, 0.09]);
        let (rep, v) = apply(&inputs(&r1, &rk4, &rk4, Some(0.845))).unwrap();
        assert_eq!(v, "PASS_S1_S2", "{rep}");
    }

    #[test]
    fn one_seed_without_a_gain_fails_s1_even_if_the_pool_is_large() {
        let r1 = seeds(0.60, 500, &[0.0, 0.0, 0.0]);
        let rk4 = seeds(0.60, 500, &[0.15, 0.15, -0.01]);
        let (_, v) = apply(&inputs(&r1, &rk4, &rk4, Some(0.84))).unwrap();
        assert_eq!(v, "STOP_DEPTH_DOES_NOT_HELP");
    }

    #[test]
    fn collapse_past_the_trained_depth_fails_s2() {
        let r1 = seeds(0.60, 500, &[0.0, 0.0, 0.0]);
        let rk4 = seeds(0.60, 500, &[0.08, 0.08, 0.08]);
        let k16 = seeds(0.60, 500, &[0.0, 0.0, 0.0]);
        let mut g = inputs(&r1, &rk4, &rk4, Some(0.84));
        g.rv_k16 = &k16;
        let (_, v) = apply(&g).unwrap();
        assert_eq!(v, "HELPS_AT_TRAINED_DEPTH_NO_EXTRAPOLATION");
    }

    #[test]
    fn a_failed_pipeline_check_blocks_any_claim_and_s0_comes_first() {
        let r1 = seeds(0.60, 500, &[0.0, 0.0, 0.0]);
        let rk4 = seeds(0.60, 500, &[0.08, 0.08, 0.08]);
        let (_, v) = apply(&inputs(&r1, &rk4, &rk4, Some(0.70))).unwrap();
        assert_eq!(v, "S4_FAIL_PIPELINE_NOT_VALIDATED");
        let mut g = inputs(&r1, &rk4, &rk4, Some(0.84));
        g.s0_ok = false;
        assert_eq!(apply(&g).unwrap().1, "S0_FAIL_FIX_PLUMBING_ONCE");
    }
}
