//! D1: direct-board MLP, a generic trainer shared by D1-A (E1 one-pass model) and D1-M,
//! fixed-horizon fit with panel snapshots. Mechanics are those of E1 with the 2,000-update
//! horizon defined in docs/v69/D1_CONTRACT.md. E1 code paths are not modified.

use crate::inventory::{ParamInfo, dump_values, inventory};
use crate::model::{Arm, Batch, Model, bce_mean};
use crate::train::{CLIP_EPS, CLIP_NORM, FINAL_LR, PEAK_LR, StepStats, WARMUP, WEIGHT_DECAY};
use burn::module::{AutodiffModule, Module, ModuleVisitor, Param};
use burn::nn::{Linear, LinearConfig};
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{AdamW, AdamWConfig, GradientsAccumulator, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
use burn::tensor::activation;
use burn::tensor::backend::AutodiffBackend;
use recur64_v69::features::{Features, N_PIECE_CODES, N_SCALARS};
use serde::Serialize;
use std::path::Path;

pub const D1_UPDATES: usize = 2000;
pub const D1_MICRO: usize = 2;
pub const D1_ACCUM: usize = 8;
pub const MEASURE_UPDATES: [usize; 5] = [0, 100, 500, 1000, 2000];
pub const MLP_IN: usize = 64 * N_PIECE_CODES + N_SCALARS + 2;

/// LR for 0-based update u in 0..1999 (see D1 contract §4).
pub fn lr_d1(u: usize) -> f64 {
    if u < WARMUP {
        PEAK_LR * (u + 1) as f64 / WARMUP as f64
    } else {
        let p = (u - WARMUP) as f64 / (D1_UPDATES - 1 - WARMUP) as f64;
        FINAL_LR + 0.5 * (PEAK_LR - FINAL_LR) * (1.0 + (std::f64::consts::PI * p.min(1.0)).cos())
    }
}

// ----------------------------------------------------------------------------- MLP

#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    fc3: Linear<B>,
}

impl<B: Backend> Mlp<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            fc1: LinearConfig::new(MLP_IN, 128).with_bias(true).init(device),
            fc2: LinearConfig::new(128, 64).with_bias(true).init(device),
            fc3: LinearConfig::new(64, 1).with_bias(true).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 1> {
        let n = x.dims()[0];
        let h = activation::gelu(self.fc1.forward(x));
        let h = activation::gelu(self.fc2.forward(h));
        self.fc3.forward(h).reshape([n])
    }
}

/// 64 x 13 one-hot (index square*13+code) ++ 8 rule scalars ++ budget one-hot [b==1, b==2].
pub fn mlp_input_vec(f: &Features) -> Vec<f32> {
    let mut v = vec![0.0f32; MLP_IN];
    for sq in 0..64 {
        v[sq * N_PIECE_CODES + f.piece[sq] as usize] = 1.0;
    }
    for (i, s) in f.scalars.iter().enumerate() {
        v[64 * N_PIECE_CODES + i] = *s;
    }
    v[64 * N_PIECE_CODES + N_SCALARS + (f.budget as usize - 1)] = 1.0;
    v
}

pub fn mlp_input<B: Backend>(feats: &[&Features], device: &B::Device) -> Tensor<B, 2> {
    let data: Vec<f32> = feats.iter().flat_map(|f| mlp_input_vec(f)).collect();
    Tensor::from_data(burn::tensor::TensorData::new(data, [feats.len(), MLP_IN]), device)
}

pub trait DiagModel<B: Backend>: Module<B> + Clone {
    /// Logits [n] for the given examples (inputs only; no labels).
    fn logits(&self, feats: &[&Features], device: &B::Device) -> Tensor<B, 1>;
}

impl<B: Backend> DiagModel<B> for Mlp<B> {
    fn logits(&self, feats: &[&Features], device: &B::Device) -> Tensor<B, 1> {
        self.forward(mlp_input::<B>(feats, device))
    }
}

/// D1-A: the unchanged E1 one-pass model (arm A, single readout).
impl<B: Backend> DiagModel<B> for Model<B> {
    fn logits(&self, feats: &[&Features], device: &B::Device) -> Tensor<B, 1> {
        let b = Batch::<B>::from_features(feats, device);
        self.forward(&b, Arm::A).pop().unwrap()
    }
}

// ----------------------------------------------------------------------------- generic trainer

type Opt<B, M> = OptimizerAdaptor<AdamW, M, B>;

pub fn make_opt<B: AutodiffBackend, M: AutodiffModule<B>>(wd: f64) -> Opt<B, M> {
    AdamWConfig::new().with_beta_1(0.9).with_beta_2(0.999).with_epsilon(1e-8).with_weight_decay(wd as f32).init::<B, M>()
}

struct NormV<'a, B: AutodiffBackend> {
    grads: &'a GradientsParams,
    sumsq: f64,
    missing: usize,
    _b: std::marker::PhantomData<B>,
}

impl<B: AutodiffBackend> ModuleVisitor<B> for NormV<'_, B> {
    fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<B, D>>) {
        match self.grads.get::<B::InnerBackend, D>(p.id) {
            Some(g) => self.sumsq += g.powf_scalar(2.0).sum().into_data().to_vec::<f32>().unwrap()[0] as f64,
            None => self.missing += 1,
        }
    }
}

struct SplitV<'a, B: AutodiffBackend> {
    src: &'a mut GradientsParams,
    decay: GradientsParams,
    nodecay: GradientsParams,
    scale: Option<f32>,
    _b: std::marker::PhantomData<B>,
}

impl<B: AutodiffBackend> ModuleVisitor<B> for SplitV<'_, B> {
    fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<B, D>>) {
        if let Some(mut g) = self.src.remove::<B::InnerBackend, D>(p.id) {
            if let Some(s) = self.scale {
                g = g.mul_scalar(s);
            }
            if D >= 2 {
                self.decay.register::<B::InnerBackend, D>(p.id, g);
            } else {
                self.nodecay.register::<B::InnerBackend, D>(p.id, g);
            }
        }
    }
}

pub struct DTrainer<B: AutodiffBackend, M: DiagModel<B> + AutodiffModule<B>> {
    pub model: M,
    pub opt_decay: Opt<B, M>,
    pub opt_nodecay: Opt<B, M>,
    pub update: usize,
    pub clip_calls: u64,
    pub device: B::Device,
}

pub fn global_norm<B: AutodiffBackend, M: Module<B>>(model: &M, grads: &GradientsParams) -> (f64, usize) {
    let mut v = NormV::<B> { grads, sumsq: 0.0, missing: 0, _b: Default::default() };
    model.visit(&mut v);
    (v.sumsq.sqrt(), v.missing)
}

impl<B: AutodiffBackend, M: DiagModel<B> + AutodiffModule<B>> DTrainer<B, M> {
    pub fn new(model: M, device: &B::Device) -> Self {
        Self { model, opt_decay: make_opt::<B, M>(WEIGHT_DECAY), opt_nodecay: make_opt::<B, M>(0.0), update: 0, clip_calls: 0, device: device.clone() }
    }

    pub fn micro_grads(&self, feats: &[&Features], labels: &[bool], scale: f32) -> (GradientsParams, f64) {
        let z = self.model.logits(feats, &self.device);
        let y = Tensor::<B, 1>::from_floats(labels.iter().map(|&l| l as u8 as f32).collect::<Vec<_>>().as_slice(), &self.device);
        let loss = bce_mean(z, y);
        let lv = loss.clone().into_data().to_vec::<f32>().unwrap()[0] as f64;
        let grads = GradientsParams::from_grads(loss.mul_scalar(scale).backward(), &self.model);
        (grads, lv)
    }

    pub fn clip_and_split(&self, acc: GradientsParams) -> (GradientsParams, GradientsParams, f64, f64, bool) {
        let (norm, missing) = global_norm::<B, M>(&self.model, &acc);
        assert_eq!(missing, 0, "{missing} parameters received no gradient");
        let factor = (CLIP_NORM / (norm + CLIP_EPS)).min(1.0);
        let clipped = norm + CLIP_EPS > CLIP_NORM;
        let mut acc = acc;
        let mut sv = SplitV::<B> { src: &mut acc, decay: GradientsParams::new(), nodecay: GradientsParams::new(), scale: if clipped { Some(factor as f32) } else { None }, _b: Default::default() };
        self.model.visit(&mut sv);
        (sv.decay, sv.nodecay, norm, factor, clipped)
    }

    pub fn apply(&mut self, acc: GradientsParams, loss: f64, t0: std::time::Instant) -> StepStats {
        let (gd, gn, norm, factor, clipped) = self.clip_and_split(acc);
        self.clip_calls += 1;
        let lr = lr_d1(self.update);
        let m = self.model.clone();
        let m = self.opt_decay.step(lr, m, gd);
        self.model = self.opt_nodecay.step(lr, m, gn);
        let st = StepStats { update: self.update, lr, loss, final_bce: loss, grad_norm_pre_clip: norm, clip_scale: if clipped { factor } else { 1.0 }, clipped, millis: t0.elapsed().as_secs_f64() * 1e3 };
        self.update += 1;
        st
    }

    /// One update from `D1_ACCUM` microbatches of `D1_MICRO` examples (loss/8 each).
    pub fn step(&mut self, micros: &[(Vec<&Features>, Vec<bool>)]) -> StepStats {
        assert_eq!(micros.len(), D1_ACCUM);
        let t0 = std::time::Instant::now();
        let mut acc = GradientsAccumulator::<M>::new();
        let mut loss = 0.0;
        for (f, l) in micros {
            let (g, lv) = self.micro_grads(f, l, 1.0 / D1_ACCUM as f32);
            acc.accumulate(&self.model, g);
            loss += lv / D1_ACCUM as f64;
        }
        self.apply(acc.grads(), loss, t0)
    }
}

// ----------------------------------------------------------------------------- checkpoints

fn rec() -> NamedMpkFileRecorder<FullPrecisionSettings> {
    NamedMpkFileRecorder::<FullPrecisionSettings>::new()
}

pub fn save_ckpt<B: AutodiffBackend, M: DiagModel<B> + AutodiffModule<B>>(dir: &Path, t: &DTrainer<B, M>) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir)?;
    let r = rec();
    t.model.clone().save_file(dir.join("model"), &r).map_err(|e| anyhow::anyhow!("save model: {e}"))?;
    r.record(t.opt_decay.to_record(), dir.join("opt_decay")).map_err(|e| anyhow::anyhow!("save opt_decay: {e}"))?;
    r.record(t.opt_nodecay.to_record(), dir.join("opt_nodecay")).map_err(|e| anyhow::anyhow!("save opt_nodecay: {e}"))?;
    std::fs::write(dir.join("meta.json"), serde_json::to_vec_pretty(&serde_json::json!({"completed_updates": t.update, "clip_calls": t.clip_calls, "next_lr": lr_d1(t.update.min(D1_UPDATES - 1))}))?)?;
    Ok(())
}

pub fn load_ckpt<B: AutodiffBackend, M: DiagModel<B> + AutodiffModule<B>>(dir: &Path, template: M, device: &B::Device) -> anyhow::Result<DTrainer<B, M>> {
    let r = rec();
    let model = template.load_file(dir.join("model"), &r, device).map_err(|e| anyhow::anyhow!("load model: {e}"))?;
    let mut t = DTrainer::<B, M>::new(model, device);
    let rd: <Opt<B, M> as Optimizer<M, B>>::Record = r.load(dir.join("opt_decay"), device).map_err(|e| anyhow::anyhow!("load opt_decay: {e}"))?;
    let rn: <Opt<B, M> as Optimizer<M, B>>::Record = r.load(dir.join("opt_nodecay"), device).map_err(|e| anyhow::anyhow!("load opt_nodecay: {e}"))?;
    t.opt_decay = t.opt_decay.load_record(rd);
    t.opt_nodecay = t.opt_nodecay.load_record(rn);
    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("meta.json"))?)?;
    t.update = meta["completed_updates"].as_u64().unwrap() as usize;
    t.clip_calls = meta["clip_calls"].as_u64().unwrap();
    Ok(t)
}

// ----------------------------------------------------------------------------- panel measurement

/// Inference-mode logits on the panel (batches of up to 16; no gradient).
pub fn panel_logits<B: AutodiffBackend, M: DiagModel<B> + AutodiffModule<B>>(model: &M, feats: &[Features], device: &B::Device) -> Vec<f32>
where
    M::InnerModule: DiagModel<B::InnerBackend>,
{
    let inner = model.valid();
    let mut out = Vec::new();
    for chunk in feats.chunks(16) {
        let refs: Vec<&Features> = chunk.iter().collect();
        out.extend(inner.logits(&refs, device).into_data().to_vec::<f32>().unwrap());
    }
    out
}

#[derive(Serialize, Clone, Debug)]
pub struct Movement {
    pub update: usize,
    /// component -> (||theta_t - theta_0||_2, ||theta_t - theta_0||_2 / ||theta_0||_2)
    pub by_component: std::collections::BTreeMap<String, (f64, f64)>,
    pub total: (f64, f64),
}

pub fn movement<B: AutodiffBackend, M: Module<B>>(model: &M, theta0: &[(Vec<usize>, Vec<f32>)], inv: &[ParamInfo], update: usize) -> Movement {
    let now = dump_values::<B, M>(model);
    let mut by: std::collections::BTreeMap<String, (f64, f64)> = Default::default();
    let (mut td, mut t0) = (0f64, 0f64);
    for ((p, a), b) in inv.iter().zip(&now).zip(theta0) {
        let d: f64 = a.1.iter().zip(&b.1).map(|(x, y)| ((*x - *y) as f64).powi(2)).sum();
        let n0: f64 = b.1.iter().map(|y| (*y as f64).powi(2)).sum();
        let e = by.entry(p.component.clone()).or_insert((0.0, 0.0));
        e.0 += d;
        e.1 += n0;
        td += d;
        t0 += n0;
    }
    let by = by.into_iter().map(|(k, (d, n0))| (k, (d.sqrt(), d.sqrt() / n0.sqrt().max(1e-30)))).collect();
    Movement { update, by_component: by, total: (td.sqrt(), td.sqrt() / t0.sqrt().max(1e-30)) }
}

pub fn inv_of<B: Backend, M: Module<B>>(m: &M) -> Vec<ParamInfo> {
    inventory::<B, M>(m)
}
