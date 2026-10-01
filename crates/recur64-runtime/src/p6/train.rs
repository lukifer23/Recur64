//! P6 ALL-INFO training: root-policy cross-entropy only, exact accumulation over the
//! physical layout, strict resumable checkpoints (same discipline as P5.1).

use std::path::Path;
use std::time::Instant;

use burn::module::AutodiffModule;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{AdamW, GradientsAccumulator, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::TensorData;
use burn::tensor::backend::AutodiffBackend;
use serde::{Deserialize, Serialize};

use recur64_model::all_info::AllInfoModel;
use recur64_model::candidate::CandidateInputs;
use recur64_model::checkpoint::{CheckpointMeta, load_training, save_training};
use recur64_model::train::{adamw, global_grad_norm};

use crate::learner::lr_at;
use crate::proof::sampler::CellSampler;
use crate::proof::targets::ProofTargets;

use super::data::{build_trees, cells, roots_of};
use super::recipe::{P6Recipe, sampler_seed};

pub type Optim<B> = OptimizerAdaptor<AdamW, AllInfoModel<B>, B>;

pub const STATE_SCHEMA: &str = "v3_p6_state_v1";
pub const BACKEND_TAG: &str = "v3-p6";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateReport {
    pub policy_loss: f64,
    pub grad_norm: f32,
    pub examples: usize,
    /// Future states (depth 1 + depth 2) supplied to the model in this update.
    pub states_supplied: usize,
    pub max_micro_states: usize,
    /// Wall seconds spent building the exhaustive trees (CPU) and in forward + backward.
    pub tree_build_s: f64,
    pub model_s: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateRecord {
    pub update: u64,
    pub lr: f64,
    pub wall_s: f64,
    pub report: UpdateReport,
}

/// Strict sidecar written next to every checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct P6State {
    pub schema: String,
    pub recipe_digest: String,
    pub recipe: P6Recipe,
    pub updates_done: u64,
    pub sampler_draws: u64,
    pub history: Vec<UpdateRecord>,
    #[serde(default)]
    pub resumptions: u32,
}

/// Compute the gradient and report of one optimizer update over `indices` (the update's
/// `effective_batch` position indices, in draw order).
pub fn compute_update<B: AutodiffBackend>(
    model: &AllInfoModel<B>,
    data: &ProofTargets,
    indices: &[usize],
    recipe: &P6Recipe,
    device: &B::Device,
) -> anyhow::Result<(GradientsParams, UpdateReport)> {
    anyhow::ensure!(
        indices.len() == recipe.effective_batch,
        "an update draws {} examples, got {}",
        recipe.effective_batch,
        indices.len()
    );
    let n_total = indices.len();
    let mut acc = GradientsAccumulator::<AllInfoModel<B>>::new();
    let (mut policy_sum_all, mut states, mut max_micro) = (0.0f64, 0usize, 0usize);
    let (mut build_s, mut model_s) = (0.0f64, 0.0f64);
    for chunk in indices.chunks(recipe.micro) {
        let positions: Vec<&_> = chunk.iter().map(|&i| &data.positions[i]).collect();
        let t0 = Instant::now();
        let roots = roots_of(&positions)?;
        let trees = build_trees(&roots)?;
        build_s += t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let inputs = CandidateInputs::<B>::from_states(&roots, device)?;
        let out = model.forward_trees(&inputs, &trees, device)?;
        let [b, w] = out.readout.policy.mask.dims();
        let mut tgt = vec![0.0f32; b * w];
        for (i, p) in positions.iter().enumerate() {
            let t = p.target();
            tgt[i * w..i * w + t.len()].copy_from_slice(&t);
        }
        let target = Tensor::<B, 2>::from_data(TensorData::new(tgt, [b, w]), device);
        let policy_sum = (out.readout.policy.log_probs.clone() * target).sum().neg();
        let v: f64 = policy_sum
            .clone()
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("{e:?}"))?[0]
            .into();
        anyhow::ensure!(v.is_finite(), "non-finite policy loss");
        policy_sum_all += v;
        let micro_states: usize = out.accounting.states.iter().map(|(a, b)| a + b).sum();
        states += micro_states;
        max_micro = max_micro.max(micro_states);
        let loss = policy_sum * (recipe.policy_weight as f32 / n_total as f32);
        let grads = GradientsParams::from_grads(loss.backward(), model);
        acc.accumulate(model, grads);
        model_s += t1.elapsed().as_secs_f64();
    }
    let grads = acc.grads();
    let grad_norm = global_grad_norm::<B, AllInfoModel<B>>(&grads, model);
    anyhow::ensure!(grad_norm.is_finite(), "non-finite gradient norm");
    Ok((
        grads,
        UpdateReport {
            policy_loss: recipe.policy_weight * policy_sum_all / n_total as f64,
            grad_norm,
            examples: n_total,
            states_supplied: states,
            max_micro_states: max_micro,
            tree_build_s: build_s,
            model_s,
        },
    ))
}

pub struct P6Trainer<B: AutodiffBackend> {
    pub recipe: P6Recipe,
    pub model: AllInfoModel<B>,
    pub optim: Optim<B>,
    pub sampler: CellSampler,
    pub updates_done: u64,
    pub history: Vec<UpdateRecord>,
    /// Resumptions including the current load (0 for a fresh run).
    pub resumptions: u32,
}

impl<B: AutodiffBackend> P6Trainer<B> {
    pub fn new(recipe: P6Recipe, train: &ProofTargets, device: &B::Device) -> anyhow::Result<Self> {
        recipe.validate_for_training()?;
        let seed = recipe
            .seed
            .ok_or_else(|| anyhow::anyhow!("a training recipe needs a seed"))?;
        anyhow::ensure!(
            recipe.peak_lr.is_some(),
            "a training recipe needs a peak LR"
        );
        B::seed(device, seed);
        let model = AllInfoModel::<B>::new(recipe.model.clone(), device);
        Ok(Self {
            sampler: CellSampler::new(&cells(train), sampler_seed(seed)),
            optim: adamw::<B, AllInfoModel<B>>(),
            model,
            recipe,
            updates_done: 0,
            history: Vec::new(),
            resumptions: 0,
        })
    }

    /// Draw the next update's examples.
    pub fn plan_next(&mut self) -> Vec<usize> {
        (0..self.recipe.effective_batch)
            .map(|_| self.sampler.next_index())
            .collect()
    }

    pub fn step(
        &mut self,
        data: &ProofTargets,
        device: &B::Device,
    ) -> anyhow::Result<UpdateRecord> {
        anyhow::ensure!(
            self.updates_done < self.recipe.updates,
            "the run is complete"
        );
        let t0 = Instant::now();
        let lr = lr_at(
            self.updates_done,
            self.recipe.peak_lr.expect("checked in new"),
            self.recipe.warmup,
            self.recipe.updates,
        );
        let plan = self.plan_next();
        let (grads, report) = compute_update(&self.model, data, &plan, &self.recipe, device)?;
        anyhow::ensure!(
            report.policy_loss.is_finite(),
            "non-finite loss at update {}",
            self.updates_done
        );
        self.model = self.optim.step(lr, self.model.clone(), grads);
        let rec = UpdateRecord {
            update: self.updates_done,
            lr,
            wall_s: t0.elapsed().as_secs_f64(),
            report,
        };
        self.updates_done += 1;
        self.history.push(rec.clone());
        Ok(rec)
    }

    pub fn save(&self, dir: &Path) -> anyhow::Result<()> {
        let ck = dir.join("checkpoint");
        let meta = CheckpointMeta::new(
            self.recipe.model.clone(),
            1,
            false,
            self.updates_done,
            self.recipe.peak_lr.unwrap_or(0.0),
            self.recipe.seed.unwrap_or(0),
            0,
            BACKEND_TAG,
            "fp32",
        );
        save_training::<B, _, _>(&ck, &self.model, &self.optim, &meta)?;
        let state = P6State {
            schema: STATE_SCHEMA.into(),
            recipe_digest: self.recipe.digest(),
            recipe: self.recipe.clone(),
            updates_done: self.updates_done,
            sampler_draws: self.sampler.examples_drawn(),
            history: self.history.clone(),
            resumptions: self.resumptions,
        };
        let tmp = dir.join("p6-state.json.tmp");
        std::fs::write(&tmp, serde_json::to_vec(&state)?)?;
        std::fs::rename(&tmp, dir.join("p6-state.json"))?;
        Ok(())
    }

    /// Resume. Refuses a sidecar whose digest is not `recipe`'s, an internally inconsistent
    /// sidecar, and a checkpoint whose metadata disagrees with it.
    pub fn load(
        dir: &Path,
        recipe: P6Recipe,
        train: &ProofTargets,
        device: &B::Device,
    ) -> anyhow::Result<Self> {
        recipe.validate_for_training()?;
        let state: P6State = serde_json::from_slice(&std::fs::read(dir.join("p6-state.json"))?)?;
        anyhow::ensure!(state.schema == STATE_SCHEMA, "unknown P6 state schema");
        anyhow::ensure!(
            state.recipe_digest == recipe.digest() && state.recipe == recipe,
            "the checkpoint was written under recipe {} but this run uses {}: refusing to resume",
            state.recipe_digest,
            recipe.digest()
        );
        let seed = recipe
            .seed
            .ok_or_else(|| anyhow::anyhow!("a training recipe needs a seed"))?;
        let peak_lr = recipe
            .peak_lr
            .ok_or_else(|| anyhow::anyhow!("a training recipe needs a peak LR"))?;
        Self::check_state(&state, &recipe, peak_lr)?;
        B::seed(device, seed);
        let template = AllInfoModel::<B>::new(recipe.model.clone(), device);
        let (model, optim, meta) = load_training::<B, _, _>(
            &dir.join("checkpoint"),
            template,
            adamw::<B, AllInfoModel<B>>(),
            device,
        )?;
        anyhow::ensure!(
            meta.architecture == "all_info_v1"
                && meta.backend == BACKEND_TAG
                && meta.precision == "fp32"
                && meta.recurrence == 1
                && !meta.deep_supervision,
            "checkpoint is not a P6 fp32 all_info_v1 checkpoint ({} / {} / {})",
            meta.architecture,
            meta.backend,
            meta.precision
        );
        anyhow::ensure!(
            meta.step == state.updates_done
                && meta.update_counter == state.updates_done
                && meta.lr_schedule_step == state.updates_done,
            "checkpoint metadata disagrees with the sidecar (step {} / update_counter {} / lr_schedule_step {} vs {})",
            meta.step,
            meta.update_counter,
            meta.lr_schedule_step,
            state.updates_done
        );
        anyhow::ensure!(
            meta.seed == seed && meta.lr == peak_lr,
            "checkpoint was written for seed {} / peak lr {} but this run is seed {seed} / peak lr {peak_lr}",
            meta.seed,
            meta.lr
        );
        let mut sampler = CellSampler::new(&cells(train), sampler_seed(seed));
        for _ in 0..state.sampler_draws {
            sampler.next_index();
        }
        Ok(Self {
            recipe,
            model,
            optim,
            sampler,
            updates_done: state.updates_done,
            history: state.history,
            resumptions: state.resumptions + 1,
        })
    }

    fn check_state(state: &P6State, recipe: &P6Recipe, peak_lr: f64) -> anyhow::Result<()> {
        anyhow::ensure!(
            state.updates_done <= recipe.updates,
            "sidecar records {} updates but the recipe has {}",
            state.updates_done,
            recipe.updates
        );
        anyhow::ensure!(
            state.history.len() as u64 == state.updates_done,
            "sidecar history has {} records for {} updates",
            state.history.len(),
            state.updates_done
        );
        for (i, h) in state.history.iter().enumerate() {
            let r = &h.report;
            anyhow::ensure!(
                h.update == i as u64,
                "sidecar history record {i} is labelled update {}",
                h.update
            );
            anyhow::ensure!(
                h.lr.is_finite()
                    && h.wall_s.is_finite()
                    && r.policy_loss.is_finite()
                    && r.grad_norm.is_finite(),
                "sidecar history record {i} holds a non-finite value"
            );
            let want = lr_at(i as u64, peak_lr, recipe.warmup, recipe.updates);
            anyhow::ensure!(
                (h.lr - want).abs() <= 1e-12 * want.abs().max(1e-12),
                "sidecar history record {i} has lr {} but the schedule gives {want}",
                h.lr
            );
        }
        let want = state.updates_done * recipe.effective_batch as u64;
        anyhow::ensure!(
            state.sampler_draws == want,
            "sampler recorded {} draws but {} updates imply {want}",
            state.sampler_draws,
            state.updates_done
        );
        Ok(())
    }

    /// The inference-side copy of the current weights.
    pub fn inference_model(&self) -> AllInfoModel<B::InnerBackend> {
        self.model.valid()
    }
}
