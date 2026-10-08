//! Frozen training mechanics: accumulation, global clipping, parameter groups,
//! LR schedule, checkpoints. See docs/v69/MODEL_SPEC.md for the equations.
//!
//! Update u (0-based), effective batch 16 = 8 accumulation steps x microbatch 2:
//!   g_u = sum_{m=0..7} grad( L_m / 8 )        (L_m = mean over 2 examples of the
//!                                              mean-over-readouts stable BCE)
//!   n_u = ||g_u||_2 over ALL parameters (global, once, after accumulation)
//!   g_u <- g_u * min(1, 1/(n_u + 1e-6))
//!   params <- AdamW(lr_u; decay group wd=1e-4, bias/norm group wd=0)(g_u)

use crate::inventory::inventory;
use crate::model::{Arm, Batch, Model, arm_loss, bce_mean};
use burn::module::{Module, ModuleVisitor, Param, ParamId};
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{AdamW, AdamWConfig, GradientsAccumulator, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
use burn::tensor::backend::AutodiffBackend;
use recur64_v69::features::Features;
use serde::Serialize;
use std::collections::HashSet;
use std::path::Path;

pub const PEAK_LR: f64 = 5e-4;
pub const FINAL_LR: f64 = 5e-5;
pub const WARMUP: usize = 20;
pub const TOTAL_UPDATES: usize = 600;
pub const WEIGHT_DECAY: f64 = 1e-4;
pub const CLIP_NORM: f64 = 1.0;
pub const CLIP_EPS: f64 = 1e-6;
pub const MICRO: usize = 2;
pub const ACCUM: usize = 8;

/// LR for the 0-based update index `u`: linear warmup `PEAK*(u+1)/20` for u < 20,
/// then cosine from PEAK (u = 20) to FINAL (u = 599).
pub fn lr_at(u: usize) -> f64 {
    if u < WARMUP {
        PEAK_LR * (u + 1) as f64 / WARMUP as f64
    } else {
        let p = (u - WARMUP) as f64 / (TOTAL_UPDATES - 1 - WARMUP) as f64;
        FINAL_LR + 0.5 * (PEAK_LR - FINAL_LR) * (1.0 + (std::f64::consts::PI * p.min(1.0)).cos())
    }
}

type Opt<B> = OptimizerAdaptor<AdamW, Model<B>, B>;

pub fn make_optimizer<B: AutodiffBackend>(weight_decay: f64) -> Opt<B> {
    AdamWConfig::new()
        .with_beta_1(0.9)
        .with_beta_2(0.999)
        .with_epsilon(1e-8)
        .with_weight_decay(weight_decay as f32)
        .init::<B, Model<B>>()
}

#[derive(Debug, Clone, Serialize)]
pub struct StepStats {
    pub update: usize,
    pub lr: f64,
    pub loss: f64,
    pub final_bce: f64,
    pub grad_norm_pre_clip: f64,
    pub clip_scale: f64,
    pub clipped: bool,
    pub millis: f64,
}

struct NormVisitor<'a, B: AutodiffBackend> {
    grads: &'a GradientsParams,
    sumsq: f64,
    missing: usize,
    _b: std::marker::PhantomData<B>,
}

impl<B: AutodiffBackend> ModuleVisitor<B> for NormVisitor<'_, B> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        match self.grads.get::<B::InnerBackend, D>(param.id) {
            Some(g) => {
                let s = g.powf_scalar(2.0).sum().into_data().to_vec::<f32>().unwrap()[0];
                self.sumsq += s as f64;
            }
            None => self.missing += 1,
        }
    }
}

struct SplitVisitor<'a, B: AutodiffBackend> {
    src: &'a mut GradientsParams,
    decay: GradientsParams,
    nodecay: GradientsParams,
    scale: Option<f32>,
    _b: std::marker::PhantomData<B>,
}

impl<B: AutodiffBackend> ModuleVisitor<B> for SplitVisitor<'_, B> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        if let Some(mut g) = self.src.remove::<B::InnerBackend, D>(param.id) {
            if let Some(s) = self.scale {
                g = g.mul_scalar(s);
            }
            if D >= 2 {
                self.decay.register::<B::InnerBackend, D>(param.id, g);
            } else {
                self.nodecay.register::<B::InnerBackend, D>(param.id, g);
            }
        }
    }
}

pub fn global_grad_norm<B: AutodiffBackend>(model: &Model<B>, grads: &GradientsParams) -> (f64, usize) {
    let mut v = NormVisitor::<B> { grads, sumsq: 0.0, missing: 0, _b: Default::default() };
    model.visit(&mut v);
    (v.sumsq.sqrt(), v.missing)
}

/// Clip factor from the global accumulated norm (applied once per update).
pub fn clip_factor(norm: f64) -> f64 {
    (CLIP_NORM / (norm + CLIP_EPS)).min(1.0)
}

pub struct Trainer<B: AutodiffBackend> {
    pub model: Model<B>,
    pub opt_decay: Opt<B>,
    pub opt_nodecay: Opt<B>,
    pub update: usize,
    pub clip_calls: u64,
    pub arm: Arm,
    pub device: B::Device,
}

pub struct Micro<'a> {
    pub feats: Vec<&'a Features>,
    pub labels: Vec<bool>,
}

impl<B: AutodiffBackend> Trainer<B> {
    pub fn new(model: Model<B>, arm: Arm, device: &B::Device) -> Self {
        Self {
            model,
            opt_decay: make_optimizer::<B>(WEIGHT_DECAY),
            opt_nodecay: make_optimizer::<B>(0.0),
            update: 0,
            clip_calls: 0,
            arm,
            device: device.clone(),
        }
    }

    /// Forward+backward for one microbatch; returns (gradients, scaled loss value, final-readout BCE).
    pub fn micro_grads(&self, mb: &Micro, scale: f32) -> (GradientsParams, f64, f64) {
        let batch = Batch::<B>::from_features(&mb.feats, &self.device);
        let y = Tensor::<B, 1>::from_floats(mb.labels.iter().map(|&l| if l { 1.0f32 } else { 0.0 }).collect::<Vec<_>>().as_slice(), &self.device);
        let outs = self.model.forward(&batch, self.arm);
        let loss = arm_loss(&outs, &y);
        let final_bce = bce_mean(outs.last().unwrap().clone().detach(), y).into_data().to_vec::<f32>().unwrap()[0] as f64;
        let lv = loss.clone().into_data().to_vec::<f32>().unwrap()[0] as f64;
        let scaled = loss.mul_scalar(scale);
        let grads = GradientsParams::from_grads(scaled.backward(), &self.model);
        (grads, lv, final_bce)
    }

    /// Global-norm clip (once, on the accumulated gradient) and split into the
    /// weight-decay and no-decay groups. Returns (decay, nodecay, norm, factor, clipped).
    pub fn clip_and_split(&self, acc: GradientsParams) -> (GradientsParams, GradientsParams, f64, f64, bool) {
        let (norm, missing) = global_grad_norm(&self.model, &acc);
        assert_eq!(missing, 0, "{missing} parameters received no gradient");
        let factor = clip_factor(norm);
        let clipped = norm + CLIP_EPS > CLIP_NORM;
        let mut acc = acc;
        let mut sv = SplitVisitor::<B> {
            src: &mut acc,
            decay: GradientsParams::new(),
            nodecay: GradientsParams::new(),
            scale: if clipped { Some(factor as f32) } else { None },
            _b: Default::default(),
        };
        self.model.visit(&mut sv);
        (sv.decay, sv.nodecay, norm, factor, clipped)
    }

    /// Accumulate already-computed microbatch gradients, clip once, step both groups.
    pub fn apply(&mut self, acc: GradientsParams, loss: f64, final_bce: f64, t0: std::time::Instant) -> StepStats {
        let (gd, gn, norm, factor, clipped) = self.clip_and_split(acc);
        self.clip_calls += 1;
        let lr = lr_at(self.update);
        let model = self.model.clone();
        let model = self.opt_decay.step(lr, model, gd);
        self.model = self.opt_nodecay.step(lr, model, gn);
        let st = StepStats {
            update: self.update,
            lr,
            loss,
            final_bce,
            grad_norm_pre_clip: norm,
            clip_scale: if clipped { factor } else { 1.0 },
            clipped,
            millis: t0.elapsed().as_secs_f64() * 1e3,
        };
        self.update += 1;
        st
    }

    /// One full optimizer update from ACCUM microbatches.
    pub fn step(&mut self, micros: &[Micro]) -> StepStats {
        assert_eq!(micros.len(), ACCUM, "an update needs exactly {ACCUM} microbatches");
        let t0 = std::time::Instant::now();
        let mut accum = GradientsAccumulator::<Model<B>>::new();
        let (mut loss, mut fb) = (0.0, 0.0);
        for mb in micros {
            let (g, l, f) = self.micro_grads(mb, 1.0 / ACCUM as f32);
            accum.accumulate(&self.model, g);
            loss += l / ACCUM as f64;
            fb += f / ACCUM as f64;
        }
        self.apply(accum.grads(), loss, fb, t0)
    }

    pub fn param_ids(&self) -> (HashSet<ParamId>, HashSet<ParamId>) {
        let inv = inventory::<B, _>(&self.model);
        (
            inv.iter().filter(|p| p.decay).map(|p| p.id).collect(),
            inv.iter().filter(|p| !p.decay).map(|p| p.id).collect(),
        )
    }
}

// ----------------------------------------------------------------- checkpoints

#[derive(Serialize, serde::Deserialize, Debug, Clone)]
pub struct CheckpointMeta {
    pub arm: String,
    pub completed_updates: usize,
    pub lr_schedule_position: usize,
    pub next_lr: f64,
    pub clip_calls: u64,
    pub provenance: serde_json::Value,
}

fn rec() -> NamedMpkFileRecorder<FullPrecisionSettings> {
    NamedMpkFileRecorder::<FullPrecisionSettings>::new()
}

/// Full training checkpoint: weights + both optimizer states (moments and step
/// counters) + schedule position + metadata. `dir` must already be validated.
pub fn save_checkpoint<B: AutodiffBackend>(dir: &Path, t: &Trainer<B>, provenance: serde_json::Value) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir)?;
    let r = rec();
    t.model.clone().save_file(dir.join("model"), &r).map_err(|e| anyhow::anyhow!("save model: {e}"))?;
    r.record(t.opt_decay.to_record(), dir.join("opt_decay")).map_err(|e| anyhow::anyhow!("save opt_decay: {e}"))?;
    r.record(t.opt_nodecay.to_record(), dir.join("opt_nodecay")).map_err(|e| anyhow::anyhow!("save opt_nodecay: {e}"))?;
    let meta = CheckpointMeta {
        arm: t.arm.name().to_string(),
        completed_updates: t.update,
        lr_schedule_position: t.update,
        next_lr: lr_at(t.update),
        clip_calls: t.clip_calls,
        provenance,
    };
    std::fs::write(dir.join("meta.json"), serde_json::to_vec_pretty(&meta)?)?;
    Ok(())
}

pub fn load_checkpoint<B: AutodiffBackend>(dir: &Path, arm: Arm, device: &B::Device) -> anyhow::Result<(Trainer<B>, CheckpointMeta)> {
    let meta: CheckpointMeta = serde_json::from_slice(&std::fs::read(dir.join("meta.json"))?)?;
    anyhow::ensure!(meta.arm == arm.name(), "checkpoint arm {} != {}", meta.arm, arm.name());
    let r = rec();
    let model = Model::<B>::new(device).load_file(dir.join("model"), &r, device).map_err(|e| anyhow::anyhow!("load model: {e}"))?;
    let mut t = Trainer::<B>::new(model, arm, device);
    let rd: <Opt<B> as Optimizer<Model<B>, B>>::Record = r.load(dir.join("opt_decay"), device).map_err(|e| anyhow::anyhow!("load opt_decay: {e}"))?;
    let rn: <Opt<B> as Optimizer<Model<B>, B>>::Record = r.load(dir.join("opt_nodecay"), device).map_err(|e| anyhow::anyhow!("load opt_nodecay: {e}"))?;
    t.opt_decay = t.opt_decay.load_record(rd);
    t.opt_nodecay = t.opt_nodecay.load_record(rn);
    t.update = meta.completed_updates;
    t.clip_calls = meta.clip_calls;
    Ok((t, meta))
}

/// Inference-only weights export loader (evaluation): model weights from a checkpoint dir.
pub fn load_model_for_eval<B: Backend>(dir: &Path, device: &B::Device) -> anyhow::Result<Model<B>> {
    Model::<B>::new(device).load_file(dir.join("model"), &rec(), device).map_err(|e| anyhow::anyhow!("load model: {e}"))
}

/// Scientific training sample order: epoch e is a Fisher-Yates permutation of
/// 0..n driven by `train_order` stream index e; the stream of samples is the
/// concatenation of epochs.
pub fn sample_order(seed: &recur64_v69::streams::MasterSeed, n: usize, count: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(count);
    let mut epoch = 0u64;
    while out.len() < count {
        let mut perm: Vec<usize> = (0..n).collect();
        let mut rng = seed.stream(recur64_v69::streams::STREAM_TRAIN_ORDER, epoch);
        for i in (1..n).rev() {
            perm.swap(i, rng.below(i as u64 + 1) as usize);
        }
        out.extend(perm);
        epoch += 1;
    }
    out.truncate(count);
    out
}
