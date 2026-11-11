//! Arms, recipe, trainer and evaluation for V420.
//!
//! The recipe is V4 Stage A's (owner decision 2026-11-11): 2,000 updates, peak LR 3e-4, warmup 80,
//! batch 128, micro 64, the `per_update_cell_balanced_v1` sampler on V4_TRAIN_FIT, CUDA FP32, the
//! shared AdamW contract. Every arm sees the SAME batches for a given seed. Run directories hold
//! final WEIGHTS and a history; they are NOT resumable (no optimiser state is saved) and say so.

use std::path::Path;
use std::time::Instant;

use burn::module::Module;
use burn::optim::{GradientsAccumulator, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::TensorData;
use burn::tensor::activation;
use burn::tensor::backend::AutodiffBackend;
use recur64_model::candidate::CandidateInputs;
use recur64_model::train::{OPTIMIZER_CONTRACT, adamw};
use recur64_runtime::learner::lr_at;
use recur64_v4::data::{PartSampler, V4Data};
use recur64_v4::session::root_ce;
use recur64_v4::train::mix;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::diag::{CHECK_UPDATES, HealthRecord, abort_rule, group_norms};
use crate::model::{DepthModel, Geometry};

pub const RECIPE_SCHEMA: &str = "v420_recipe_v1";
pub const SAMPLER: &str = "per_update_cell_balanced_v1";
/// Stage-A-identical training constants.
pub const UPDATES: u64 = 2000;
pub const WARMUP: u64 = 80;
pub const PEAK_LR: f64 = 3e-4;
pub const BATCH: usize = 128;
pub const MICRO: usize = 64;
/// Maximum depth sampled by the RV arm during training.
pub const RV_MAX_K: usize = 4;
/// Evaluation depths for the shared arms (the RV extrapolation diagnostic and gates).
pub const EVAL_KS: [usize; 5] = [1, 2, 4, 8, 16];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KSpec {
    Fixed(usize),
    /// Uniform in `1..=max`, one K per update (whole batch), derived from the seed and update.
    Sampled(usize),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Arm {
    pub name: String,
    pub unique_blocks: usize,
    pub k: KSpec,
}

/// The frozen arm table (`docs/V420_RESEARCH_PLAN.md` section 4 as amended by V420-E1).
pub fn arm(name: &str) -> anyhow::Result<Arm> {
    let (unique_blocks, k) = match name {
        "R1" => (1, KSpec::Fixed(1)),
        "RK2" => (1, KSpec::Fixed(2)),
        "RK4" => (1, KSpec::Fixed(4)),
        "RV" => (1, KSpec::Sampled(RV_MAX_K)),
        "WIDE" => (4, KSpec::Fixed(4)),
        "BASE8" => (8, KSpec::Fixed(8)),
        other => anyhow::bail!("unknown V420 arm '{other}' (R1, RK2, RK4, RV, WIDE, BASE8)"),
    };
    Ok(Arm {
        name: name.into(),
        unique_blocks,
        k,
    })
}

impl Arm {
    /// Depth used at update `u` of a run with `seed`.
    pub fn k_at(&self, seed: u64, u: u64) -> usize {
        match self.k {
            KSpec::Fixed(k) => k,
            KSpec::Sampled(max) => 1 + (mix(0x4200_0001 ^ seed, u) % max as u64) as usize,
        }
    }

    /// Depths this arm is evaluated at.
    pub fn eval_ks(&self) -> Vec<usize> {
        if self.unique_blocks == 1 {
            EVAL_KS.to_vec()
        } else {
            vec![self.unique_blocks]
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recipe {
    pub schema: String,
    pub identity: String,
    pub arm: Arm,
    pub seed: u64,
    pub updates: u64,
    pub warmup: u64,
    pub peak_lr: f64,
    pub batch: usize,
    pub micro: usize,
    pub sampler: String,
    pub optimizer_contract: String,
    pub fit_digest: String,
    pub dev_digest: String,
    pub train_digest: String,
    pub geometry: Geometry,
}

impl Recipe {
    pub fn new(arm: Arm, seed: u64, data: &V4Data) -> Self {
        let geometry = Geometry::v4(arm.unique_blocks);
        Self {
            schema: RECIPE_SCHEMA.into(),
            identity: crate::model::IDENTITY.into(),
            arm,
            seed,
            updates: UPDATES,
            warmup: WARMUP,
            peak_lr: PEAK_LR,
            batch: BATCH,
            micro: MICRO,
            sampler: SAMPLER.into(),
            optimizer_contract: OPTIMIZER_CONTRACT.into(),
            fit_digest: data.fit_digest.clone(),
            dev_digest: data.dev_digest.clone(),
            train_digest: data.targets.digest.clone(),
            geometry,
        }
    }

    pub fn digest(&self) -> String {
        let mut h = Sha256::new();
        h.update(serde_json::to_vec(self).expect("recipe serialises"));
        format!("{:x}", h.finalize())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepRecord {
    pub update: u64,
    pub k: usize,
    pub lr: f64,
    pub loss: f64,
    pub wall_s: f64,
}

fn scalar<B: Backend>(t: Tensor<B, 1>) -> f64 {
    f64::from(t.into_data().to_vec::<f32>().expect("f32")[0])
}

/// Soft-target cross-entropy of masked logits, summed over the batch (divide by the batch outside).
fn ce_sum<B: Backend>(z: Tensor<B, 2>, tgt: Tensor<B, 2>, valid: Tensor<B, 2, Bool>) -> Tensor<B, 1> {
    let logp = activation::log_softmax(z, 1).mask_fill(valid.bool_not(), 0.0);
    (logp * tgt).sum().neg()
}

pub struct Trainer<B: AutodiffBackend> {
    pub model: DepthModel<B>,
    optim: burn::optim::adaptor::OptimizerAdaptor<burn::optim::AdamW, DepthModel<B>, B>,
    pub recipe: Recipe,
    pub updates_done: u64,
    pub history: Vec<StepRecord>,
    pub health: Vec<HealthRecord>,
}

impl<B: AutodiffBackend> Trainer<B> {
    pub fn new(recipe: Recipe, device: &B::Device) -> Self {
        let model = DepthModel::<B>::new(&recipe.geometry, device);
        Self {
            model,
            optim: adamw::<B, DepthModel<B>>(),
            recipe,
            updates_done: 0,
            history: Vec::new(),
            health: Vec::new(),
        }
    }

    /// One optimiser update. Errors (visibly) on a non-finite loss or a failed health rule.
    pub fn step(&mut self, data: &V4Data, device: &B::Device) -> anyhow::Result<StepRecord> {
        let r = &self.recipe;
        anyhow::ensure!(self.updates_done < r.updates, "the run is complete");
        let u = self.updates_done;
        let t0 = Instant::now();
        let lr = lr_at(u, r.peak_lr, r.warmup, r.updates);
        let k = r.arm.k_at(r.seed, u);
        let mut sampler = PartSampler::new(data, &data.fit, mix(r.seed, u));
        let batch = sampler.next_batch(r.batch);
        let n = batch.len() as f32;
        let mut acc = GradientsAccumulator::<DepthModel<B>>::new();
        let mut loss_sum = 0.0;
        for chunk in batch.chunks(r.micro.max(1)) {
            let roots = data.roots(chunk)?;
            let inputs = CandidateInputs::<B>::from_states(&roots, device)?;
            let z = self.model.logits(&inputs, k)?;
            let w = z.dims()[1];
            let tgt = Tensor::<B, 2>::from_data(
                TensorData::new(data.target_rows(chunk, w), [chunk.len(), w]),
                device,
            );
            let loss = ce_sum(z, tgt, inputs.cands.mask.clone()).div_scalar(n);
            loss_sum += scalar(loss.clone());
            acc.accumulate(
                &self.model,
                GradientsParams::from_grads(loss.backward(), &self.model),
            );
        }
        let grads = acc.grads();
        anyhow::ensure!(loss_sum.is_finite(), "non-finite loss at update {u}");
        if CHECK_UPDATES.contains(&u) {
            let rec = HealthRecord {
                update: u,
                k,
                groups: group_norms(&self.model, &grads),
            };
            self.health.push(rec.clone());
            if let Some(first) = self.health.first() {
                abort_rule(first, &rec).map_err(|e| {
                    anyhow::anyhow!("V420 gradient-health abort (plan section 6): {e}")
                })?;
            }
        }
        self.model = self.optim.step(lr, self.model.clone(), grads);
        let rec = StepRecord {
            update: u,
            k,
            lr,
            loss: loss_sum,
            wall_s: t0.elapsed().as_secs_f64(),
        };
        self.updates_done += 1;
        self.history.push(rec.clone());
        Ok(rec)
    }

    /// Save final weights and the history. NOT resumable: no optimiser state is written.
    pub fn save(&self, dir: &Path, device_label: &str) -> anyhow::Result<()> {
        std::fs::create_dir_all(dir)?;
        let rec = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        self.model
            .clone()
            .save_file(dir.join("model"), &rec)
            .map_err(|e| anyhow::anyhow!("saving weights: {e:?}"))?;
        let tail: Vec<f64> = self.history.iter().rev().take(25).map(|h| h.loss).collect();
        let summary = serde_json::json!({
            "schema": "v420_run_summary_v1",
            "recipe": self.recipe, "recipe_digest": self.recipe.digest(),
            "updates": self.updates_done,
            "parameters": self.model.num_params_total(),
            "first_loss": self.history.first().map(|h| h.loss),
            "last25_mean_loss": tail.iter().sum::<f64>() / tail.len().max(1) as f64,
            "all_losses_finite": self.history.iter().all(|h| h.loss.is_finite()),
            "health": self.health,
            "health_ok": true,
            "device": device_label,
            "resumable": false,
            "weights_only": true,
            "v4_tune_v1_evaluated": false, "holdout_c_evaluated": false,
        });
        std::fs::write(dir.join("run-summary.json"), serde_json::to_vec_pretty(&summary)?)?;
        std::fs::write(
            dir.join("history.json"),
            serde_json::to_vec(&self.history)?,
        )?;
        Ok(())
    }
}

/// Load weights saved by [`Trainer::save`] into an inference copy.
pub fn load_weights<B: AutodiffBackend>(
    dir: &Path,
    geometry: &Geometry,
    device: &B::Device,
) -> anyhow::Result<DepthModel<B::InnerBackend>> {
    let rec = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let idev = Default::default();
    let _ = device;
    DepthModel::<B::InnerBackend>::new(geometry, &idev)
        .load_file(dir.join("model"), &rec, &idev)
        .map_err(|e| anyhow::anyhow!("{}: loading weights: {e:?}", dir.display()))
}

/// Per-position evaluation (inner backend; no autodiff).
#[derive(Debug, Clone, Default, Serialize)]
pub struct PosEval {
    pub ce: Vec<f64>,
    pub top1: Vec<f64>,
    pub mass: Vec<f64>,
}

pub fn eval_positions<B: Backend>(
    model: &DepthModel<B>,
    data: &V4Data,
    idx: &[usize],
    k: usize,
    micro: usize,
    device: &B::Device,
) -> anyhow::Result<PosEval> {
    let mut out = PosEval::default();
    for chunk in idx.chunks(micro.max(1)) {
        let roots = data.roots(chunk)?;
        let inputs = CandidateInputs::<B>::from_states(&roots, device)?;
        let z = model.logits(&inputs, k)?;
        let w = z.dims()[1];
        let z = z.into_data().to_vec::<f32>().map_err(|e| anyhow::anyhow!("{e:?}"))?;
        for (r, &i) in chunk.iter().enumerate() {
            let vw = data.position(i).legal.len();
            let row = &z[r * w..r * w + vw];
            let correct = data.correct(i);
            out.ce.push(f64::from(root_ce(row, &correct)));
            let (mut best, mut arg) = (f32::NEG_INFINITY, 0usize);
            for (c, &v) in row.iter().enumerate() {
                if v > best {
                    best = v;
                    arg = c;
                }
            }
            out.top1.push(f64::from(u8::from(correct.contains(&arg))));
            let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let den: f32 = row.iter().map(|v| (v - mx).exp()).sum();
            out.mass.push(f64::from(
                correct.iter().map(|&c| (row[c] - mx).exp()).sum::<f32>() / den,
            ));
        }
    }
    Ok(out)
}
