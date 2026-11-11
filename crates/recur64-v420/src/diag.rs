//! Gradient health check (plan section 6): per-group weight and gradient norms, computed from the
//! SAME gradients the optimiser is about to consume. Read-only: it changes no weight or gradient.

use std::collections::BTreeMap;

use burn::module::{Module, ModuleVisitor, Param};
use burn::optim::GradientsParams;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use serde::{Deserialize, Serialize};

use crate::model::DepthModel;

/// Plan section 6 thresholds (frozen).
pub const MIN_GRAD_OVER_WEIGHT: f64 = 1.0e-6;
pub const MAX_DROP_FACTOR: f64 = 1.0e3;
/// Updates at which the check is logged. The abort rule applies from `ABORT_FROM`.
pub const CHECK_UPDATES: [u64; 3] = [0, 50, 200];
pub const ABORT_FROM: u64 = 50;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GroupNorm {
    pub params: usize,
    pub weight_norm: f64,
    pub grad_norm: f64,
    pub tensors: usize,
    pub tensors_with_grad: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthRecord {
    pub update: u64,
    pub k: usize,
    pub groups: BTreeMap<String, GroupNorm>,
}

struct V<'a, B: AutodiffBackend> {
    grads: &'a GradientsParams,
    stack: Vec<String>,
    rows: BTreeMap<String, (f64, f64, usize, usize, usize)>,
    _b: std::marker::PhantomData<B>,
}

fn sumsq<B: Backend, const D: usize>(t: Tensor<B, D>) -> f64 {
    let v = t.flatten::<1>(0, D - 1).into_data().to_vec::<f32>().expect("f32 tensor");
    v.iter().map(|x| f64::from(*x) * f64::from(*x)).sum()
}

impl<B: AutodiffBackend> ModuleVisitor<B> for V<'_, B> {
    fn enter_module(&mut self, name: &str, _t: &str) {
        self.stack.push(name.to_string());
    }
    fn exit_module(&mut self, _name: &str, _t: &str) {
        self.stack.pop();
    }
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        let group = self.stack.first().cloned().unwrap_or_default();
        let w = param.val().inner();
        let n = w.shape().num_elements();
        let r = self.rows.entry(group).or_default();
        r.2 += n;
        r.3 += 1;
        r.0 += sumsq(w);
        if let Some(g) = self.grads.get::<B::InnerBackend, D>(param.id) {
            r.4 += 1;
            r.1 += sumsq(g);
        }
    }
}

/// Weight and gradient norms per top-level module group (`blocks` = the board block(s)).
pub fn group_norms<B: AutodiffBackend>(
    model: &DepthModel<B>,
    grads: &GradientsParams,
) -> BTreeMap<String, GroupNorm> {
    let mut v = V::<B> {
        grads,
        stack: Vec::new(),
        rows: BTreeMap::new(),
        _b: std::marker::PhantomData,
    };
    model.visit(&mut v);
    v.rows
        .into_iter()
        .map(|(k, (w, g, p, t, tg))| {
            (
                k,
                GroupNorm {
                    params: p,
                    weight_norm: w.sqrt(),
                    grad_norm: g.sqrt(),
                    tensors: t,
                    tensors_with_grad: tg,
                },
            )
        })
        .collect()
}

/// Verdict of the plan-section-6 abort rule on the `blocks` group, given update-0 and current records.
/// `Ok(())` means healthy (or too early to judge); `Err` carries the reason.
pub fn abort_rule(first: &HealthRecord, now: &HealthRecord) -> Result<(), String> {
    if now.update < ABORT_FROM {
        return Ok(());
    }
    let (Some(g0), Some(g1)) = (first.groups.get("blocks"), now.groups.get("blocks")) else {
        return Err("no `blocks` group in the health record".into());
    };
    if g1.tensors_with_grad < g1.tensors {
        return Err(format!(
            "update {}: only {}/{} board-block tensors received a gradient",
            now.update, g1.tensors_with_grad, g1.tensors
        ));
    }
    if !(g1.grad_norm.is_finite() && g1.weight_norm.is_finite()) {
        return Err(format!("update {}: non-finite block norms", now.update));
    }
    if g1.weight_norm > 0.0 && g1.grad_norm / g1.weight_norm < MIN_GRAD_OVER_WEIGHT {
        return Err(format!(
            "update {}: board-block grad/weight {:.3e} is below {:.0e}",
            now.update,
            g1.grad_norm / g1.weight_norm,
            MIN_GRAD_OVER_WEIGHT
        ));
    }
    if g0.grad_norm > 0.0 && g1.grad_norm < g0.grad_norm / MAX_DROP_FACTOR {
        return Err(format!(
            "update {}: board-block gradient fell from {:.3e} to {:.3e} (more than {:.0e}x)",
            now.update, g0.grad_norm, g1.grad_norm, MAX_DROP_FACTOR
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(update: u64, g: f64, w: f64, with: usize) -> HealthRecord {
        let mut groups = BTreeMap::new();
        groups.insert(
            "blocks".into(),
            GroupNorm {
                params: 10,
                weight_norm: w,
                grad_norm: g,
                tensors: 4,
                tensors_with_grad: with,
            },
        );
        HealthRecord { update, k: 1, groups }
    }

    #[test]
    fn the_abort_rule_matches_the_frozen_thresholds() {
        let first = rec(0, 1.0, 10.0, 4);
        assert!(abort_rule(&first, &rec(50, 0.5, 10.0, 4)).is_ok());
        assert!(abort_rule(&first, &rec(10, 0.0, 10.0, 0)).is_ok(), "too early to judge");
        assert!(abort_rule(&first, &rec(50, 1e-8, 10.0, 4)).is_err(), "grad/weight below 1e-6");
        assert!(abort_rule(&first, &rec(50, 5e-4, 10.0, 4)).is_err(), "fell more than 1e3x");
        assert!(abort_rule(&first, &rec(200, 0.5, 10.0, 3)).is_err(), "a tensor got no gradient");
    }
}
