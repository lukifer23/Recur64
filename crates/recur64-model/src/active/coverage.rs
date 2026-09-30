//! Per-parameter gradient coverage: which parameters received a finite,
//! non-zero gradient from one backward pass. Used by the engineering tests and
//! by `recur64 v3-qual` so both check the same thing.

use burn::module::{Module, ModuleVisitor, Param, ParamId};
use burn::optim::GradientsParams;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

/// One parameter tensor's gradient status.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CoverageRow {
    /// Dotted module path, for example `planner.gate.weight`.
    pub name: String,
    pub numel: usize,
    pub has_grad: bool,
    pub finite: bool,
    pub nonzero: bool,
    /// Largest absolute gradient component (0 when there is none).
    pub max_abs: f32,
}

struct Visitor<'a, B: AutodiffBackend> {
    grads: &'a GradientsParams,
    path: Vec<String>,
    rows: Vec<CoverageRow>,
    _p: std::marker::PhantomData<B>,
}

impl<B: AutodiffBackend> Visitor<'_, B> {
    fn record<const D: usize>(&mut self, id: ParamId, numel: usize) {
        let name = self.path.join(".");
        let row = match self.grads.get::<B::InnerBackend, D>(id) {
            None => CoverageRow {
                name,
                numel,
                has_grad: false,
                finite: true,
                nonzero: false,
                max_abs: 0.0,
            },
            Some(g) => {
                let v = g.into_data().to_vec::<f32>().unwrap_or_default();
                let max_abs = v.iter().fold(0.0f32, |m, x| m.max(x.abs()));
                CoverageRow {
                    name,
                    numel,
                    has_grad: true,
                    finite: v.iter().all(|x| x.is_finite()),
                    nonzero: v.iter().any(|x| x.abs() > 1e-12),
                    max_abs,
                }
            }
        };
        self.rows.push(row);
    }
}

impl<B: AutodiffBackend> ModuleVisitor<B> for Visitor<'_, B> {
    fn enter_module(&mut self, name: &str, _container: &str) {
        self.path.push(name.to_string());
    }
    fn exit_module(&mut self, _name: &str, _container: &str) {
        self.path.pop();
    }
    fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<B, D>>) {
        let numel = p.val().shape().num_elements();
        self.record::<D>(p.id, numel);
    }
}

/// Gradient status of every float parameter of `model`.
pub fn gradient_coverage<B: AutodiffBackend, M: Module<B>>(
    model: &M,
    grads: &GradientsParams,
) -> Vec<CoverageRow> {
    let mut v = Visitor::<B> {
        grads,
        path: Vec::new(),
        rows: Vec::new(),
        _p: std::marker::PhantomData,
    };
    model.visit(&mut v);
    v.rows
}

/// True for the STOP head, whose gradient is exactly zero while STOP is masked.
pub fn is_stop_head(name: &str) -> bool {
    name.contains("stop_hidden") || name.contains("stop_out")
}

/// Largest gradient a mathematically inert parameter may show: float noise only.
pub const INERT_NOISE_BOUND: f32 = 1e-5;

/// The key-projection bias of a V2.5 `Block` / `CandidateBlock`, inherited
/// unchanged by the root encoder (contract `v25_root_encoder_v1`) and by the
/// query-state encoder. A constant added to every key of a query cancels in the
/// softmax, so its true gradient is exactly zero; whether floating-point noise
/// lands above any fixed threshold varies with the initialisation, so it cannot
/// be required to be "non-zero". It is exempt from the non-zero requirement and
/// instead required to stay at noise level ([`INERT_NOISE_BOUND`]). The V3
/// planner has no such parameter: its key projection has no bias.
pub fn is_inherited_inert_key_bias(name: &str) -> bool {
    name.ends_with(".k_proj.bias") && (name.starts_with("root.") || name.starts_with("query."))
}
