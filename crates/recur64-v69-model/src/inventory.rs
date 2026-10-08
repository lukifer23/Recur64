//! Parameter inventory (names, shapes, weight-decay group) via Burn visitors.

use burn::module::{Module, ModuleMapper, ModuleVisitor, Param, ParamId};
use burn::prelude::*;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct ParamInfo {
    pub index: usize,
    /// Dotted module path, e.g. `fast.cross.q`.
    pub path: String,
    /// `weight` | `bias` | `gamma` | `beta`
    pub leaf: String,
    /// Top-level component (first path element).
    pub component: String,
    pub shape: Vec<usize>,
    pub numel: usize,
    /// Receives weight decay (rank >= 2). Biases and normalization parameters do not.
    pub decay: bool,
    #[serde(skip)]
    pub id: ParamId,
}

struct Walker {
    stack: Vec<(String, String)>,
    counters: Vec<usize>,
}

impl Walker {
    fn new() -> Self {
        Self { stack: Vec::new(), counters: vec![0] }
    }
    fn enter(&mut self, name: &str, container: &str) {
        if std::env::var("V69_DEBUG_INV").is_ok() {
            eprintln!("enter {name} {container}");
        }
        self.stack.push((name.to_string(), container.to_string()));
        self.counters.push(0);
    }
    fn exit(&mut self) {
        self.stack.pop();
        self.counters.pop();
    }
    /// (dotted path of the owning layer, leaf name): the visitor enters every field,
    /// parameters included, so the top of the stack is the leaf.
    fn next_leaf(&mut self) -> (String, String) {
        let (leaf, _) = self.stack.last().expect("parameter outside any module").clone();
        let path = self.stack[..self.stack.len() - 1].iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>().join(".");
        assert!(matches!(leaf.as_str(), "weight" | "bias" | "gamma" | "beta"), "unexpected parameter leaf {leaf} at {path}");
        (path, leaf)
    }
}

struct Inv {
    w: Walker,
    out: Vec<ParamInfo>,
}

impl<B: Backend> ModuleVisitor<B> for Inv {
    fn enter_module(&mut self, name: &str, container_type: &str) {
        self.w.enter(name, container_type);
    }
    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.w.exit();
    }
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        let (path, leaf) = self.w.next_leaf();
        let shape: Vec<usize> = param.val().dims().to_vec();
        let numel = shape.iter().product();
        let decay = D >= 2;
        let component = path.split('.').next().unwrap_or("").to_string();
        let index = self.out.len();
        self.out.push(ParamInfo { index, path, leaf, component, shape, numel, decay, id: param.id });
    }
}

pub fn inventory<B: Backend, M: Module<B>>(m: &M) -> Vec<ParamInfo> {
    let mut v = Inv { w: Walker::new(), out: Vec::new() };
    m.visit(&mut v);
    // Group sanity: biases and normalization parameters are exactly the rank-1 ones.
    for p in &v.out {
        let rank1 = matches!(p.leaf.as_str(), "bias" | "gamma" | "beta");
        assert_eq!(rank1, !p.decay, "decay grouping inconsistent for {}.{}", p.path, p.leaf);
    }
    v.out
}

/// Replace every float parameter, in traversal order, with `values[i]`.
struct Loader<'a, B: Backend> {
    values: &'a [(Vec<usize>, Vec<f32>)],
    next: usize,
    device: B::Device,
}

impl<B: Backend> ModuleMapper<B> for Loader<'_, B> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let (shape, data) = &self.values[self.next];
        self.next += 1;
        let cur: Vec<usize> = param.val().dims().to_vec();
        assert_eq!(&cur, shape, "shape mismatch at parameter {}", self.next - 1);
        let t = Tensor::<B, D>::from_data(burn::tensor::TensorData::new(data.clone(), cur), &self.device);
        param.map(|_| t).set_require_grad(true)
    }
}

pub fn load_values<B: Backend, M: Module<B>>(m: M, values: &[(Vec<usize>, Vec<f32>)], device: &B::Device) -> M {
    let mut l = Loader::<B> { values, next: 0, device: device.clone() };
    let m = m.map(&mut l);
    assert_eq!(l.next, values.len(), "parameter count mismatch while loading");
    m
}

/// Host copy of every float parameter in traversal order.
struct Dumper {
    out: Vec<(Vec<usize>, Vec<f32>)>,
}

impl<B: Backend> ModuleVisitor<B> for Dumper {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        let t = param.val();
        let shape = t.dims().to_vec();
        let data = t.into_data().to_vec::<f32>().unwrap();
        self.out.push((shape, data));
    }
}

pub fn dump_values<B: Backend, M: Module<B>>(m: &M) -> Vec<(Vec<usize>, Vec<f32>)> {
    let mut d = Dumper { out: Vec::new() };
    m.visit(&mut d);
    d.out
}
