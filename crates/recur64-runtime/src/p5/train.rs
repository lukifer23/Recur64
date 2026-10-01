//! The P5 multi-budget trainer.
//!
//! One set of weights is trained on budgets {0,2,4,8}. Per optimizer update the
//! microbatches follow the frozen budget sequence, each microbatch drawing from that
//! budget's own `cell_balanced_v1` sampler; the teacher is `proof_teacher_seeded_v1`.
//!
//! # Normalisation (the scientific objective)
//!
//! `L = sum(policy CE)/N_examples + 1.0 * sum(selector NLL)/N_supervised`, where both
//! normalisers count the WHOLE optimizer update. Because the teacher is independent of
//! the model, the number of supervised decisions of the update is known before any
//! gradient is computed (a model-free simulation of the same tree), so every
//! microbatch is back-propagated once with the final weights and the accumulated
//! gradient is already the monolithic gradient. The selector weight therefore never
//! depends on how many B0 microbatches exist, on the physical microbatch size, or on
//! the accumulation count.

use std::path::Path;
use std::time::Instant;

use burn::module::AutodiffModule;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{AdamW, GradientsAccumulator, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::TensorData;
use burn::tensor::backend::AutodiffBackend;
use serde::{Deserialize, Serialize};

use recur64_core::GameState;
use recur64_model::active::loss::selector_nll_sum;
use recur64_model::active::{ActiveSearchModel, RunOptions, Selection};
use recur64_model::checkpoint::{CheckpointMeta, load_training, save_training};
use recur64_model::train::{adamw, global_grad_norm};

use crate::learner::lr_at;

use super::data::{BudgetSamplers, Dataset};
use super::recipe::{BUDGETS, Recipe, teacher_key_base};
use super::teacher::{SeededProofTeacher, follow_key, simulate_episode};

pub type Optim<B> = OptimizerAdaptor<AdamW, ActiveSearchModel<B>, B>;

/// One training example occurrence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Item {
    /// Index into the dataset.
    pub index: usize,
    /// Position of this draw in its budget's stream.
    pub ordinal: u64,
}

/// One microbatch: a single budget.
#[derive(Debug, Clone)]
pub struct Micro {
    pub budget: usize,
    pub items: Vec<Item>,
}

/// Everything one optimizer update consumes.
#[derive(Debug, Clone)]
pub struct UpdatePlan {
    pub micros: Vec<Micro>,
}

impl UpdatePlan {
    pub fn examples(&self) -> usize {
        self.micros.iter().map(|m| m.items.len()).sum()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetReport {
    pub budget: usize,
    pub examples: usize,
    pub policy_loss: f64,
    pub supervised_decisions: usize,
    pub proofs_completed: usize,
    pub filler_queries: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateReport {
    pub policy_loss: f64,
    pub selector_loss: f64,
    pub total_loss: f64,
    pub grad_norm: f32,
    pub examples: usize,
    pub supervised_decisions: usize,
    pub per_budget: Vec<BudgetReport>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateRecord {
    pub update: u64,
    pub lr: f64,
    pub wall_s: f64,
    pub report: UpdateReport,
}

fn scalar<B: Backend>(t: Tensor<B, 1>) -> f64 {
    t.into_data()
        .to_vec::<f32>()
        .ok()
        .and_then(|v| v.first().copied())
        .map_or(f64::NAN, f64::from)
}

fn roots_of(data: &Dataset, items: &[Item]) -> anyhow::Result<Vec<GameState>> {
    items
        .iter()
        .map(|it| {
            GameState::from_fen(&data.positions()[it.index].fen)
                .map_err(|e| anyhow::anyhow!("{}: {e}", data.positions()[it.index].id))
        })
        .collect()
}

/// Compute the gradient and report of one optimizer update for `plan`. No
/// optimizer is involved; this is the quantity the reference tests check.
pub fn compute_update<B: AutodiffBackend>(
    model: &ActiveSearchModel<B>,
    data: &Dataset,
    plan: &UpdatePlan,
    recipe: &Recipe,
    device: &B::Device,
) -> anyhow::Result<(GradientsParams, UpdateReport)> {
    let seed = recipe
        .seed
        .ok_or_else(|| anyhow::anyhow!("a training recipe needs a seed"))?;
    let base = teacher_key_base(seed);
    let n_total = plan.examples();
    anyhow::ensure!(n_total > 0, "empty update");

    // Model-free pass: the supervised decisions of the whole update.
    let mut keys: Vec<Vec<u64>> = Vec::new();
    let mut dry: Vec<Vec<usize>> = Vec::new();
    let mut sup_total = 0usize;
    for m in &plan.micros {
        let mut ks = Vec::new();
        let mut ds = Vec::new();
        for it in &m.items {
            let p = &data.positions()[it.index];
            let key = follow_key(base, &p.id, it.ordinal, m.budget);
            let root = GameState::from_fen(&p.fen)?;
            let rec = simulate_episode(&data.traces[it.index], &root, m.budget, key)?;
            sup_total += rec.supervised;
            ks.push(key);
            ds.push(rec.supervised);
        }
        keys.push(ks);
        dry.push(ds);
    }

    let mut acc = GradientsAccumulator::<ActiveSearchModel<B>>::new();
    let (mut policy_sum_all, mut sel_sum_all) = (0.0f64, 0.0f64);
    let mut per_budget: Vec<BudgetReport> = BUDGETS
        .iter()
        .map(|&b| BudgetReport {
            budget: b,
            examples: 0,
            policy_loss: 0.0,
            supervised_decisions: 0,
            proofs_completed: 0,
            filler_queries: 0,
        })
        .collect();
    for (mi, m) in plan.micros.iter().enumerate() {
        let roots = roots_of(data, &m.items)?;
        let traces: Vec<&_> = m.items.iter().map(|it| &data.traces[it.index]).collect();
        let mut teacher = SeededProofTeacher::new(traces, keys[mi].clone());
        let mut opts = RunOptions::forced(m.budget);
        opts.health_checks = recipe.health_checks;
        let out = model.run(&roots, &opts, Selection::Script(&mut teacher), device)?;

        // The teacher must reproduce the model-free pass exactly.
        for (i, rec) in teacher.records.iter().enumerate() {
            anyhow::ensure!(
                rec.supervised == dry[mi][i],
                "teacher determinism violated at micro {mi} example {i}: {} vs {}",
                rec.supervised,
                dry[mi][i]
            );
        }

        // Exact uniform root target over the correct set.
        let [b, w] = out.readout.policy.mask.dims();
        let mut tgt = vec![0.0f32; b * w];
        for (i, it) in m.items.iter().enumerate() {
            let t = data.positions()[it.index].target();
            tgt[i * w..i * w + t.len()].copy_from_slice(&t);
        }
        let target = Tensor::<B, 2>::from_data(TensorData::new(tgt, [b, w]), device);
        let policy_sum = (out.readout.policy.log_probs.clone() * target).sum().neg();
        let policy_value = scalar(policy_sum.clone());
        anyhow::ensure!(
            policy_value.is_finite(),
            "non-finite policy loss at micro {mi}"
        );

        let mut loss = policy_sum * (recipe.policy_weight as f32 / n_total as f32);
        let mut sel_value = 0.0f64;
        if let Some((sel_sum, count)) = selector_nll_sum(&out.selector_steps) {
            let expected: usize = teacher.records.iter().map(|r| r.supervised).sum();
            anyhow::ensure!(
                count == expected,
                "micro {mi}: {count} selector decisions recorded but the teacher supervised {expected}"
            );
            sel_value = scalar(sel_sum.clone());
            anyhow::ensure!(
                sel_value.is_finite(),
                "non-finite selector loss at micro {mi}"
            );
            loss = loss + sel_sum * (recipe.selector_weight as f32 / sup_total.max(1) as f32);
        } else {
            anyhow::ensure!(
                teacher.records.iter().all(|r| r.supervised == 0),
                "micro {mi}: supervised decisions without selector steps"
            );
        }
        policy_sum_all += policy_value;
        sel_sum_all += sel_value;
        let slot = BUDGETS
            .iter()
            .position(|&x| x == m.budget)
            .expect("a training budget");
        let pb = &mut per_budget[slot];
        pb.examples += m.items.len();
        pb.policy_loss += policy_value;
        pb.supervised_decisions += teacher.records.iter().map(|r| r.supervised).sum::<usize>();
        pb.proofs_completed += teacher
            .records
            .iter()
            .filter(|r| r.completed_at.is_some())
            .count();
        pb.filler_queries += teacher
            .records
            .iter()
            .map(|r| r.filler_queries)
            .sum::<usize>();

        let grads = GradientsParams::from_grads(loss.backward(), model);
        acc.accumulate(model, grads);
    }
    for pb in &mut per_budget {
        if pb.examples > 0 {
            pb.policy_loss /= pb.examples as f64;
        }
    }
    let grads = acc.grads();
    let grad_norm = global_grad_norm::<B, ActiveSearchModel<B>>(&grads, model);
    anyhow::ensure!(grad_norm.is_finite(), "non-finite gradient norm");
    let policy_loss = recipe.policy_weight * policy_sum_all / n_total as f64;
    let selector_loss = if sup_total > 0 {
        sel_sum_all / sup_total as f64
    } else {
        0.0
    };
    Ok((
        grads,
        UpdateReport {
            policy_loss,
            selector_loss,
            total_loss: policy_loss + recipe.selector_weight * selector_loss,
            grad_norm,
            examples: n_total,
            supervised_decisions: sup_total,
            per_budget,
        },
    ))
}

/// Strict sidecar written next to every checkpoint. A loader refuses it unless the
/// recipe digest equals the one it was asked to run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct P5State {
    pub schema: String,
    pub recipe_digest: String,
    pub recipe: Recipe,
    pub updates_done: u64,
    pub sampler_draws: Vec<u64>,
    pub history: Vec<UpdateRecord>,
}

pub const STATE_SCHEMA: &str = "v3_p5_state_v1";

pub struct Trainer<B: AutodiffBackend> {
    pub recipe: Recipe,
    pub model: ActiveSearchModel<B>,
    pub optim: Optim<B>,
    pub samplers: BudgetSamplers,
    pub updates_done: u64,
    pub history: Vec<UpdateRecord>,
}

impl<B: AutodiffBackend> Trainer<B> {
    /// A fresh run: the backend RNG is seeded with the run seed before the model is
    /// built, so every learning rate of one seed starts from identical weights.
    pub fn new(recipe: Recipe, train: &Dataset, device: &B::Device) -> anyhow::Result<Self> {
        recipe.validate_for_training()?;
        let seed = recipe
            .seed
            .ok_or_else(|| anyhow::anyhow!("a training recipe needs a seed"))?;
        anyhow::ensure!(
            recipe.peak_lr.is_some(),
            "a training recipe needs a peak learning rate"
        );
        B::seed(device, seed);
        let model = ActiveSearchModel::<B>::new(recipe.model.clone(), device);
        Ok(Self {
            samplers: BudgetSamplers::new(&train.cells(), seed),
            optim: adamw::<B, ActiveSearchModel<B>>(),
            model,
            recipe,
            updates_done: 0,
            history: Vec::new(),
        })
    }

    /// Draw the next update's examples from the per-budget samplers.
    pub fn plan_next(&mut self) -> UpdatePlan {
        let micro = self.recipe.micro;
        let micros = self
            .recipe
            .budget_sequence
            .clone()
            .into_iter()
            .map(|budget| Micro {
                budget,
                items: (0..micro)
                    .map(|_| {
                        let (index, ordinal) = self.samplers.draw(budget);
                        Item { index, ordinal }
                    })
                    .collect(),
            })
            .collect();
        UpdatePlan { micros }
    }

    /// One optimizer update.
    pub fn step(&mut self, data: &Dataset, device: &B::Device) -> anyhow::Result<UpdateRecord> {
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
            report.total_loss.is_finite(),
            "non-finite total loss at update {}",
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

    /// Save model, optimizer, metadata and the strict sidecar.
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
            "v3-p5",
            "fp32",
        );
        save_training::<B, _, _>(&ck, &self.model, &self.optim, &meta)?;
        let state = P5State {
            schema: STATE_SCHEMA.into(),
            recipe_digest: self.recipe.digest(),
            recipe: self.recipe.clone(),
            updates_done: self.updates_done,
            sampler_draws: self.samplers.draws(),
            history: self.history.clone(),
        };
        let tmp = dir.join("p5-state.json.tmp");
        std::fs::write(&tmp, serde_json::to_vec(&state)?)?;
        std::fs::rename(&tmp, dir.join("p5-state.json"))?;
        Ok(())
    }

    /// Resume. Refuses a sidecar whose recipe digest is not `recipe`'s, a checkpoint
    /// whose metadata disagrees with the sidecar, and any model/contract mismatch.
    pub fn load(
        dir: &Path,
        recipe: Recipe,
        train: &Dataset,
        device: &B::Device,
    ) -> anyhow::Result<Self> {
        recipe.validate_for_training()?;
        let state: P5State = serde_json::from_slice(&std::fs::read(dir.join("p5-state.json"))?)?;
        anyhow::ensure!(state.schema == STATE_SCHEMA, "unknown P5 state schema");
        anyhow::ensure!(
            state.recipe_digest == recipe.digest() && state.recipe == recipe,
            "the checkpoint was written under recipe {} but this run uses {}: refusing to resume",
            state.recipe_digest,
            recipe.digest()
        );
        let seed = recipe
            .seed
            .ok_or_else(|| anyhow::anyhow!("a training recipe needs a seed"))?;
        B::seed(device, seed);
        let template = ActiveSearchModel::<B>::new(recipe.model.clone(), device);
        let (model, optim, meta) = load_training::<B, _, _>(
            &dir.join("checkpoint"),
            template,
            adamw::<B, ActiveSearchModel<B>>(),
            device,
        )?;
        anyhow::ensure!(
            meta.step == state.updates_done && meta.architecture == "active_search_v3",
            "checkpoint metadata disagrees with the sidecar (step {} vs {})",
            meta.step,
            state.updates_done
        );
        let mut samplers = BudgetSamplers::new(&train.cells(), seed);
        samplers.fast_forward(&state.sampler_draws)?;
        Ok(Self {
            recipe,
            model,
            optim,
            samplers,
            updates_done: state.updates_done,
            history: state.history,
        })
    }

    /// The inference-side copy of the current weights.
    pub fn inference_model(&self) -> ActiveSearchModel<B::InnerBackend> {
        self.model.valid()
    }
}
