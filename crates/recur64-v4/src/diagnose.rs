//! V4-P2 step 1: READ-ONLY saddle diagnostic on finished Stage B models.
//!
//! Question: is the evidence path stuck at a near-zero saddle (vanishing trust and gradients, an
//! optimisation dead-start) or does it carry gradient that training simply failed to use? This module
//! loads a finished model, computes ONE Stage-B-shaped gradient on `V4_TRAIN_DEV` positions (no
//! optimiser step, no weight change) and reports per-layer weight and gradient norms, plus the
//! trust / message-norm / delta magnitudes seen at query time. It evaluates nothing on `v4_tune_v1`
//! or HOLDOUT_C and writes no checkpoint.

use std::collections::BTreeMap;

use burn::module::{AutodiffModule, Module, ModuleVisitor, Param};
use burn::optim::GradientsParams;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use serde_json::{Value, json};

use crate::data::V4Data;
use crate::model::EvidenceBeliefModel;
use crate::session::{Freeze, RunOptions, Selection};
use crate::train::{Sel, ce_update};

#[derive(Default, Clone)]
struct Row {
    w_sq: f64,
    g_sq: f64,
    params: usize,
    with_grad: usize,
    tensors: usize,
}

struct NormVisitor<'a, B: AutodiffBackend> {
    grads: Option<&'a GradientsParams>,
    rows: BTreeMap<String, Row>,
    stack: Vec<String>,
    _b: std::marker::PhantomData<B>,
}

fn sumsq<B: Backend, const D: usize>(t: Tensor<B, D>) -> f64 {
    let flat = t.flatten::<1>(0, D - 1);
    let v = flat.into_data().to_vec::<f32>().expect("f32 tensor");
    v.iter().map(|x| f64::from(*x) * f64::from(*x)).sum()
}

impl<B: AutodiffBackend> ModuleVisitor<B> for NormVisitor<'_, B> {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.stack.push(name.to_string());
    }
    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.stack.pop();
    }
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        let layer = self.stack.join(".");
        let id = param.id;
        let w = param.val().inner();
        let n = w.shape().num_elements();
        let row = self.rows.entry(layer).or_default();
        row.tensors += 1;
        row.params += n;
        row.w_sq += sumsq(w);
        if let Some(g) = self.grads.and_then(|g| g.get::<B::InnerBackend, D>(id)) {
            row.with_grad += 1;
            row.g_sq += sumsq(g);
        }
    }
}

fn norms<B: AutodiffBackend>(model: &EvidenceBeliefModel<B>, grads: Option<&GradientsParams>) -> BTreeMap<String, Row> {
    let mut v = NormVisitor::<B> {
        grads,
        rows: BTreeMap::new(),
        stack: Vec::new(),
        _b: std::marker::PhantomData,
    };
    model.visit(&mut v);
    v.rows
}

fn stats(v: &[f64]) -> Value {
    if v.is_empty() {
        return json!({"n": 0});
    }
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |p: f64| s[((s.len() - 1) as f64 * p).round() as usize];
    json!({"n": v.len(), "mean": mean, "min": s[0], "p10": q(0.1), "median": q(0.5), "p90": q(0.9), "max": s[s.len() - 1]})
}

/// One Stage-B-shaped gradient plus query-time magnitudes. `idx` are positions of `V4_TRAIN_DEV`.
pub fn saddle_report<B: AutodiffBackend>(
    model: &EvidenceBeliefModel<B>,
    data: &V4Data,
    idx: &[usize],
    budget: usize,
    micro: usize,
    device: &B::Device,
) -> anyhow::Result<Value> {
    // 1. One accumulated CE gradient, exactly as Stage B computes it (base detached, FIXED schedule).
    let opts = RunOptions::new(budget).with_freeze(Freeze::BASE);
    let (grads, loss) = ce_update(model, data, idx, micro, &opts, Sel::Fixed, device)?;
    let weights_and_grads = norms(model, Some(&grads));
    // Freshly initialised weights of the SAME geometry, for a "how far did training move" reference.
    let fresh = EvidenceBeliefModel::<B>::new(model.config().clone(), device);
    let fresh_rows = norms(&fresh, None);

    let mut layers = Vec::new();
    let mut groups: BTreeMap<String, (f64, f64, usize)> = BTreeMap::new();
    for (name, r) in &weights_and_grads {
        let wn = r.w_sq.sqrt();
        let gn = r.g_sq.sqrt();
        let w0 = fresh_rows.get(name).map_or(f64::NAN, |f| f.w_sq.sqrt());
        layers.push(json!({
            "layer": name, "params": r.params, "tensors": r.tensors, "tensors_with_grad": r.with_grad,
            "weight_norm": wn, "weight_norm_fresh_init": w0, "grad_norm": gn,
            "grad_over_weight": if wn > 0.0 { gn / wn } else { f64::NAN },
        }));
        let top = name.split('.').next().unwrap_or("").to_string();
        let g = groups.entry(top).or_default();
        g.0 += r.w_sq;
        g.1 += r.g_sq;
        g.2 += r.params;
    }
    let groups: Vec<Value> = groups
        .into_iter()
        .map(|(k, (w, g, p))| json!({"group": k, "params": p, "weight_norm": w.sqrt(), "grad_norm": g.sqrt()}))
        .collect();

    // 2. Query-time magnitudes on the graph-free copy.
    let inf = model.valid();
    let idev = Default::default();
    let mut trust = Vec::new();
    let mut norm = Vec::new();
    let mut delta_abs = Vec::new();
    let mut delta_max_per_pos = Vec::new();
    for chunk in idx.chunks(micro.max(1)) {
        let roots = data.roots(chunk)?;
        let mut o = RunOptions::new(budget);
        o.diagnostics = true;
        let out = inf.run(&roots, &o, Selection::Fixed, 0, &idev)?;
        for m in &out.messages {
            if let Some(t) = m.trust {
                trust.push(f64::from(t));
            }
            if let Some(n) = m.norm {
                norm.push(f64::from(n));
            }
        }
        let [b, w] = out.delta.dims();
        let d = out.delta.clone().into_data().to_vec::<f32>().expect("f32");
        let mask = out.mask.clone().into_data().to_vec::<bool>().unwrap_or_default();
        for e in 0..b {
            let mut mx = 0.0f64;
            for c in 0..w {
                if mask.get(e * w + c).copied().unwrap_or(true) {
                    let a = f64::from(d[e * w + c].abs());
                    delta_abs.push(a);
                    mx = mx.max(a);
                }
            }
            delta_max_per_pos.push(mx);
        }
    }

    Ok(json!({
        "schema": "v4_saddle_report_v1",
        "positions": idx.len(), "budget": budget,
        "ce_loss_this_batch": loss,
        "groups": groups,
        "layers": layers,
        "trust": stats(&trust),
        "message_norm": stats(&norm),
        "abs_delta_all_candidates": stats(&delta_abs),
        "max_abs_delta_per_position": stats(&delta_max_per_pos),
        "note": "READ-ONLY: one gradient, no optimiser step, no checkpoint written, v4_tune_v1 and HOLDOUT_C untouched",
    }))
}
