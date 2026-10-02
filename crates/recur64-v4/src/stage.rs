//! Stage recipes and the resumable stage trainer (A: base, B: evidence, C: utility).
//!
//! A run directory holds a full training checkpoint (`checkpoint/`: weights, AdamW state, meta)
//! plus `v4-state.json` (recipe, recipe digest, updates done, history). A weights-only file is
//! never presented as resumable: `Trainer::load` needs both and refuses a different recipe.

use std::path::Path;
use std::time::Instant;

use burn::module::AutodiffModule;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{AdamW, GradientsAccumulator, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use recur64_model::checkpoint::{CheckpointMeta, hash_file, load_training, save_training};
use recur64_model::config::ModelConfig;
use recur64_model::train::{OPTIMIZER_CONTRACT, adamw};
use recur64_runtime::learner::lr_at;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::data::{PartSampler, V4Data};
use crate::model::EvidenceBeliefModel;
use crate::session::{Freeze, RunOptions};
use crate::train::{Sel, UtilityLoss, ce_update, chunks_min2, mix, probe_batch_ad, utility_loss};

pub const RECIPE_SCHEMA: &str = "v4_stage_recipe_v1";
pub const STATE_SCHEMA: &str = "v4_state_v1";
pub const SAMPLER: &str = "per_update_cell_balanced_v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Stage {
    /// Base tower, budget 0 only.
    A,
    /// Evidence path, base detached.
    B,
    /// Utility head, base and evidence detached.
    C,
}

impl Stage {
    pub fn label(self) -> &'static str {
        match self {
            Stage::A => "a",
            Stage::B => "b",
            Stage::C => "c",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recipe {
    pub schema: String,
    pub stage: Stage,
    pub seed: u64,
    pub updates: u64,
    pub warmup: u64,
    pub peak_lr: f64,
    pub batch: usize,
    pub micro: usize,
    /// Stage B: budget per update, cycled.
    pub budgets: Vec<usize>,
    /// Stage C: probe count and the label-independent prefix lengths, cycled.
    pub probe_k: usize,
    pub prefixes: Vec<usize>,
    pub utility_loss: Option<UtilityLoss>,
    /// Content hash (`meta.model_id`) of the weights this stage started from; `None` for stage A.
    pub init_model_id: Option<String>,
    pub fit_digest: String,
    pub dev_digest: String,
    pub sampler: String,
    pub optimizer_contract: String,
    pub model: ModelConfig,
}

impl Recipe {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        stage: Stage,
        seed: u64,
        updates: u64,
        peak_lr: f64,
        data: &V4Data,
        model: &ModelConfig,
        init_model_id: Option<String>,
    ) -> Self {
        let (batch, micro, warmup) = match stage {
            Stage::A => (128, 64, 80),
            Stage::B => (128, 32, 80),
            Stage::C => (32, 16, 30),
        };
        Self {
            schema: RECIPE_SCHEMA.into(),
            stage,
            seed,
            updates,
            warmup: warmup.min(updates / 2).max(1),
            peak_lr,
            batch,
            micro,
            budgets: if stage == Stage::B { vec![2, 4, 8] } else { Vec::new() },
            probe_k: if stage == Stage::C { crate::train::PROBE_K } else { 0 },
            prefixes: if stage == Stage::C { vec![0, 1, 2, 3] } else { Vec::new() },
            utility_loss: None,
            init_model_id,
            fit_digest: data.fit_digest.clone(),
            dev_digest: data.dev_digest.clone(),
            sampler: SAMPLER.into(),
            optimizer_contract: OPTIMIZER_CONTRACT.into(),
            model: model.clone(),
        }
    }

    pub fn with_loss(mut self, loss: UtilityLoss) -> Self {
        self.utility_loss = Some(loss);
        self
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(self.schema == RECIPE_SCHEMA, "recipe schema {}", self.schema);
        anyhow::ensure!(self.updates > 0 && self.peak_lr > 0.0 && self.batch > 0, "empty recipe");
        anyhow::ensure!(self.model.validate().is_ok(), "invalid model config");
        match self.stage {
            Stage::A => anyhow::ensure!(self.init_model_id.is_none(), "stage A starts from scratch"),
            Stage::B => {
                anyhow::ensure!(self.init_model_id.is_some(), "stage B needs a stage A init");
                anyhow::ensure!(
                    self.budgets.iter().all(|&b| (1..=8).contains(&b)),
                    "stage B budgets must be in 1..=8"
                );
            }
            Stage::C => {
                anyhow::ensure!(self.init_model_id.is_some(), "stage C needs a stage B init");
                anyhow::ensure!(self.utility_loss.is_some(), "stage C needs a utility loss");
                anyhow::ensure!(self.probe_k >= 2 && !self.prefixes.is_empty(), "bad probe plan");
            }
        }
        Ok(())
    }

    /// SHA-256 over the canonical JSON of the whole recipe.
    pub fn digest(&self) -> String {
        let mut h = Sha256::new();
        h.update(serde_json::to_vec(self).expect("recipe serialises"));
        format!("{:x}", h.finalize())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepRecord {
    pub update: u64,
    pub lr: f64,
    pub loss: f64,
    pub wall_s: f64,
    /// Stage B budget / stage C prefix of this update.
    pub detail: u64,
}

#[derive(Serialize, Deserialize)]
struct State {
    schema: String,
    recipe_digest: String,
    recipe: Recipe,
    updates_done: u64,
    history: Vec<StepRecord>,
}

type Opt<B> = OptimizerAdaptor<AdamW, EvidenceBeliefModel<B>, B>;

pub struct Trainer<B: AutodiffBackend> {
    pub model: EvidenceBeliefModel<B>,
    optim: Opt<B>,
    pub recipe: Recipe,
    pub updates_done: u64,
    pub history: Vec<StepRecord>,
}

fn scalar<B: Backend>(t: Tensor<B, 1>) -> f64 {
    f64::from(t.into_data().to_vec::<f32>().expect("f32")[0])
}

impl<B: AutodiffBackend> Trainer<B> {
    pub fn new(recipe: Recipe, model: EvidenceBeliefModel<B>) -> anyhow::Result<Self> {
        recipe.validate()?;
        Ok(Self {
            model,
            optim: adamw::<B, EvidenceBeliefModel<B>>(),
            recipe,
            updates_done: 0,
            history: Vec::new(),
        })
    }

    /// One optimiser update. Which parameters may move is fixed by the stage:
    /// A everything (budget 0 only), B the evidence path (base detached), C the utility path.
    pub fn step(&mut self, data: &V4Data, device: &B::Device) -> anyhow::Result<StepRecord> {
        anyhow::ensure!(self.updates_done < self.recipe.updates, "the run is complete");
        let r = &self.recipe;
        let u = self.updates_done;
        let t0 = Instant::now();
        let lr = lr_at(u, r.peak_lr, r.warmup, r.updates);
        let mut sampler = PartSampler::new(data, &data.fit, mix(r.seed, u));
        let batch = sampler.next_batch(r.batch);
        let (grads, loss, detail) = match r.stage {
            Stage::A => {
                let opts = RunOptions::new(0);
                let (g, l) = ce_update(&self.model, data, &batch, r.micro, &opts, Sel::Fixed, device)?;
                (g, l, 0)
            }
            Stage::B => {
                let budget = r.budgets[(u as usize) % r.budgets.len()];
                let opts = RunOptions::new(budget).with_freeze(Freeze::BASE);
                let sel = if u % 2 == 0 {
                    Sel::Fixed
                } else {
                    Sel::Random(mix(r.seed, u))
                };
                let (g, l) = ce_update(&self.model, data, &batch, r.micro, &opts, sel, device)?;
                (g, l, budget as u64)
            }
            Stage::C => {
                let prefix = r.prefixes[(u as usize) % r.prefixes.len()];
                let kind = r.utility_loss.expect("validated");
                let n = batch.len() as f32;
                let mut acc = GradientsAccumulator::<EvidenceBeliefModel<B>>::new();
                let mut loss_sum = 0.0;
                for (mi, chunk) in chunks_min2(&batch, r.micro).into_iter().enumerate() {
                    let (scores, samples) = probe_batch_ad(
                        &self.model,
                        data,
                        chunk,
                        prefix,
                        r.probe_k,
                        mix(r.seed, u * 1_000 + mi as u64),
                        device,
                    )?;
                    let loss = utility_loss(scores, &samples, kind)
                        .mul_scalar(chunk.len() as f32 / n);
                    loss_sum += scalar(loss.clone());
                    acc.accumulate(
                        &self.model,
                        GradientsParams::from_grads(loss.backward(), &self.model),
                    );
                }
                (acc.grads(), loss_sum, prefix as u64)
            }
        };
        anyhow::ensure!(loss.is_finite(), "non-finite loss at update {u}");
        self.model = self.optim.step(lr, self.model.clone(), grads);
        let rec = StepRecord {
            update: u,
            lr,
            loss,
            wall_s: t0.elapsed().as_secs_f64(),
            detail,
        };
        self.updates_done += 1;
        self.history.push(rec.clone());
        Ok(rec)
    }

    /// Save the full training checkpoint and the state sidecar (atomically).
    pub fn save(&self, dir: &Path, backend_tag: &str) -> anyhow::Result<String> {
        let meta = CheckpointMeta::new(
            self.recipe.model.clone(),
            1,
            false,
            self.updates_done,
            self.recipe.peak_lr,
            self.recipe.seed,
            0,
            backend_tag,
            "fp32",
        );
        save_training::<B, _, _>(&dir.join("checkpoint"), &self.model, &self.optim, &meta)?;
        let model_id = model_id_of(&dir.join("checkpoint"))?;
        let state = State {
            schema: STATE_SCHEMA.into(),
            recipe_digest: self.recipe.digest(),
            recipe: self.recipe.clone(),
            updates_done: self.updates_done,
            history: self.history.clone(),
        };
        let path = dir.join("v4-state.json");
        let tmp = dir.join("v4-state.json.tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(&state)?)?;
        std::fs::rename(&tmp, &path)?;
        Ok(model_id)
    }

    /// Resume: needs the checkpoint AND the sidecar, and refuses a different recipe.
    pub fn load(dir: &Path, recipe: Recipe, device: &B::Device) -> anyhow::Result<Self> {
        recipe.validate()?;
        let state: State = serde_json::from_slice(
            &std::fs::read(dir.join("v4-state.json"))
                .map_err(|e| anyhow::anyhow!("{}: no resumable state ({e})", dir.display()))?,
        )?;
        anyhow::ensure!(state.schema == STATE_SCHEMA, "state schema {}", state.schema);
        anyhow::ensure!(
            state.recipe_digest == recipe.digest() && state.recipe_digest == state.recipe.digest(),
            "{}: the saved recipe differs from the requested recipe; resuming is refused",
            dir.display()
        );
        let template = EvidenceBeliefModel::<B>::new(recipe.model.clone(), device);
        let (model, optim, meta) = load_training::<B, _, _>(
            &dir.join("checkpoint"),
            template,
            adamw::<B, EvidenceBeliefModel<B>>(),
            device,
        )?;
        anyhow::ensure!(
            meta.step == state.updates_done,
            "checkpoint step {} differs from the state's {} updates",
            meta.step,
            state.updates_done
        );
        Ok(Self {
            model,
            optim,
            recipe,
            updates_done: state.updates_done,
            history: state.history,
        })
    }
}

/// Content hash of the saved weights of a checkpoint directory (`meta.model_id`).
pub fn model_id_of(checkpoint_dir: &Path) -> anyhow::Result<String> {
    let meta: CheckpointMeta = serde_json::from_slice(&std::fs::read(checkpoint_dir.join("meta.json"))?)?;
    anyhow::ensure!(!meta.model_id.is_empty(), "checkpoint has no model id");
    let _ = hash_file; // identity is recorded by `save_training`; re-exported for callers
    Ok(meta.model_id)
}

/// Load the model weights of a finished stage directory (fresh optimiser is the caller's job):
/// the checkpoint must be a complete V4 training checkpoint.
pub fn load_model<B: AutodiffBackend>(
    dir: &Path,
    cfg: &ModelConfig,
    device: &B::Device,
) -> anyhow::Result<(EvidenceBeliefModel<B>, CheckpointMeta)> {
    let template = EvidenceBeliefModel::<B>::new(cfg.clone(), device);
    let (m, _o, meta) = load_training::<B, _, _>(
        &dir.join("checkpoint"),
        template,
        adamw::<B, EvidenceBeliefModel<B>>(),
        device,
    )?;
    Ok((m, meta))
}

/// Inference copy (no autodiff).
pub fn inference<B: AutodiffBackend>(m: &EvidenceBeliefModel<B>) -> EvidenceBeliefModel<B::InnerBackend> {
    m.valid()
}
