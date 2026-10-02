//! The V3.5 trainer: detached ACTIVE rollout (Pass A) + autodiff replay (Pass B).
//!
//! Per optimizer update, with the weights frozen:
//!
//! * **Pass A.** For every micro-batch with budget > 0 the inference copy of the model
//!   runs `Selection::ActiveLabelled`: the learner picks every edge, the real
//!   `StateQuery` executes it, and ProofTrace labels each visited prefix. The learner's
//!   edges, the labels, the accounting and the final policy are recorded.
//! * **Pass B.** The autodiff model replays the recorded edges exactly (`Selection::Script`
//!   with the recorded follow and targets), receiving real state content. It is refused
//!   unless it reproduces Pass A.
//!
//! `L = sum(policy CE)/N_examples + 1.0 * sum(selector NLL)/N_supervised`, both
//! normalisers whole-update counts; Pass A gives `N_supervised` before any backward.

use std::path::Path;
use std::time::Instant;

use burn::module::AutodiffModule;
use burn::optim::{GradientsAccumulator, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::TensorData;
use burn::tensor::backend::AutodiffBackend;
use serde::{Deserialize, Serialize};

use recur64_core::GameState;
use recur64_model::active::loss::selector_nll_sum;
use recur64_model::active::{
    ActiveOutput, ActiveSearchModel, EdgeRef, QueryScript, RunOptions, ScriptStep, Selection, Tree,
};
use recur64_model::checkpoint::{CheckpointMeta, load_training, save_training};
use recur64_model::train::{adamw, global_grad_norm};

use crate::learner::lr_at;
use crate::p5::data::{BudgetSamplers, Dataset};
use crate::p5::recipe::BUDGETS;
use crate::p5::train::{Item, Micro, Optim, P5State, UpdatePlan};

use super::recipe::{Recipe35, SELECTED_P5_DIGEST};
use super::target::ProofTargetProvider;

pub const STATE_SCHEMA: &str = "v35_state_v1";
pub const BACKEND_TAG: &str = "v3-p35";

/// Examples one optimizer update draws from budget `budget`'s sampler.
pub fn draws_per_update(recipe: &Recipe35, budget: usize) -> u64 {
    (recipe
        .base
        .budget_sequence
        .iter()
        .filter(|&&b| b == budget)
        .count()
        * recipe.base.micro) as u64
}

/// Replays a recorded learner trajectory and its labels (Pass B).
pub struct RecordedScript {
    pub follow: Vec<Vec<usize>>,
    pub targets: Vec<Vec<Vec<usize>>>,
}

impl QueryScript for RecordedScript {
    fn next(
        &mut self,
        example: usize,
        step: usize,
        frontier: &[EdgeRef],
        _tree: &Tree,
    ) -> anyhow::Result<ScriptStep> {
        let follow = *self
            .follow
            .get(example)
            .and_then(|f| f.get(step))
            .ok_or_else(|| anyhow::anyhow!("recorded trajectory {example} has no step {step}"))?;
        anyhow::ensure!(
            follow < frontier.len(),
            "recorded edge {follow} is outside the replay frontier ({})",
            frontier.len()
        );
        Ok(ScriptStep {
            follow,
            targets: self.targets[example][step].clone(),
        })
    }
}

type PathRow = (usize, u16, usize, u32, bool, usize);

/// Everything Pass A records for one micro-batch.
pub struct Rollout {
    pub follow: Vec<Vec<usize>>,
    pub targets: Vec<Vec<Vec<usize>>>,
    pub supervised: usize,
    pub refute_steps: usize,
    pub proofs_completed: usize,
    pub policy: Vec<f32>,
    pub paths: Vec<Vec<PathRow>>,
    pub successful: Vec<usize>,
    pub depths: Vec<Vec<u32>>,
    pub unique_nodes: usize,
    pub terminal_nodes: usize,
    pub transpositions: usize,
}

fn host<B: Backend>(t: Tensor<B, 2>) -> anyhow::Result<Vec<f32>> {
    t.into_data()
        .to_vec::<f32>()
        .map_err(|e| anyhow::anyhow!("reading policy to host: {e:?}"))
}

fn paths_of<B: Backend>(out: &ActiveOutput<B>) -> Vec<Vec<PathRow>> {
    out.traces
        .iter()
        .map(|t| {
            t.iter()
                .map(|r| {
                    (
                        r.parent_slot,
                        r.action,
                        r.branch,
                        r.depth,
                        r.terminal,
                        r.frontier_size,
                    )
                })
                .collect()
        })
        .collect()
}

/// Pass A for one micro-batch on any backend (the trainer uses the inference copy).
pub fn rollout<B: Backend>(
    model: &ActiveSearchModel<B>,
    data: &Dataset,
    roots: &[GameState],
    items: &[Item],
    budget: usize,
    health_checks: bool,
    device: &B::Device,
) -> anyhow::Result<Rollout> {
    let traces: Vec<&_> = items.iter().map(|it| &data.traces[it.index]).collect();
    let mut provider = ProofTargetProvider::new(traces);
    let mut opts = RunOptions::forced(budget);
    opts.health_checks = health_checks;
    let out = model.run(
        roots,
        &opts,
        Selection::ActiveLabelled(&mut provider),
        device,
    )?;
    let supervised = provider.supervised_total();
    let refute_steps = provider.stats.iter().map(|s| s.refute_steps).sum();
    let proofs_completed = provider
        .stats
        .iter()
        .filter(|s| s.completed_at.is_some())
        .count();
    Ok(Rollout {
        follow: out.chosen.clone(),
        targets: provider.targets,
        supervised,
        refute_steps,
        proofs_completed,
        policy: host(out.readout.policy.log_probs.clone())?,
        paths: paths_of(&out),
        successful: out.accounting.successful_queries.clone(),
        depths: out.accounting.query_depths.clone(),
        unique_nodes: out.accounting.unique_nodes,
        terminal_nodes: out.accounting.terminal_nodes,
        transpositions: out.accounting.transpositions_detected,
    })
}

/// Largest |log-prob| difference; errors on any structural disagreement.
pub fn check_replay<B: Backend>(
    a: &Rollout,
    b: &ActiveOutput<B>,
    tolerance: f64,
) -> anyhow::Result<f64> {
    anyhow::ensure!(
        a.paths == paths_of(b),
        "replay query paths differ from the rollout"
    );
    anyhow::ensure!(
        a.successful == b.accounting.successful_queries && a.depths == b.accounting.query_depths,
        "replay successful-query accounting differs from the rollout"
    );
    anyhow::ensure!(
        a.unique_nodes == b.accounting.unique_nodes
            && a.terminal_nodes == b.accounting.terminal_nodes
            && a.transpositions == b.accounting.transpositions_detected,
        "replay tree accounting differs from the rollout"
    );
    let pb = host(b.readout.policy.log_probs.clone())?;
    anyhow::ensure!(pb.len() == a.policy.len(), "replay policy shape differs");
    let mut worst = 0.0f64;
    for (x, y) in a.policy.iter().zip(&pb) {
        let d = f64::from((x - y).abs());
        anyhow::ensure!(d.is_finite(), "non-finite replay policy difference");
        worst = worst.max(d);
    }
    anyhow::ensure!(
        worst <= tolerance,
        "the autodiff replay differs from the detached rollout by {worst:e} (> {tolerance:e})"
    );
    Ok(worst)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetReport35 {
    pub budget: usize,
    pub examples: usize,
    pub policy_loss: f64,
    pub supervised_decisions: usize,
    pub refute_steps: usize,
    pub proofs_completed: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateReport35 {
    pub policy_loss: f64,
    pub selector_loss: f64,
    pub total_loss: f64,
    pub grad_norm: f32,
    pub examples: usize,
    pub supervised_decisions: usize,
    pub max_replay_policy_diff: f64,
    pub rollout_s: f64,
    pub replay_backward_s: f64,
    pub per_budget: Vec<BudgetReport35>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateRecord35 {
    pub update: u64,
    pub lr: f64,
    pub wall_s: f64,
    pub report: UpdateReport35,
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

/// Gradient and report of one optimizer update (no optimizer involved).
pub fn compute_update<B: AutodiffBackend>(
    model: &ActiveSearchModel<B>,
    data: &Dataset,
    plan: &UpdatePlan,
    recipe: &Recipe35,
    device: &B::Device,
) -> anyhow::Result<(GradientsParams, UpdateReport35)> {
    let base = &recipe.base;
    let n_total = plan.examples();
    anyhow::ensure!(n_total > 0, "empty update");
    let t_a = Instant::now();
    let inference = model.valid();
    let mut all_roots = Vec::with_capacity(plan.micros.len());
    let mut rollouts: Vec<Option<Rollout>> = Vec::with_capacity(plan.micros.len());
    let mut sup_total = 0usize;
    for m in &plan.micros {
        let roots = roots_of(data, &m.items)?;
        if m.budget == 0 {
            rollouts.push(None);
        } else {
            let r = rollout(
                &inference,
                data,
                &roots,
                &m.items,
                m.budget,
                base.health_checks,
                device,
            )?;
            sup_total += r.supervised;
            rollouts.push(Some(r));
        }
        all_roots.push(roots);
    }
    let rollout_s = t_a.elapsed().as_secs_f64();

    let t_b = Instant::now();
    let mut acc = GradientsAccumulator::<ActiveSearchModel<B>>::new();
    let (mut policy_sum_all, mut sel_sum_all) = (0.0f64, 0.0f64);
    let mut max_diff = 0.0f64;
    let mut per_budget: Vec<BudgetReport35> = BUDGETS
        .iter()
        .map(|&b| BudgetReport35 {
            budget: b,
            examples: 0,
            policy_loss: 0.0,
            supervised_decisions: 0,
            refute_steps: 0,
            proofs_completed: 0,
        })
        .collect();
    for (mi, m) in plan.micros.iter().enumerate() {
        let roots = &all_roots[mi];
        let mut opts = RunOptions::forced(m.budget);
        opts.health_checks = base.health_checks;
        let (out, supervised, refute, proofs) = match &rollouts[mi] {
            None => {
                let mut none = RecordedScript {
                    follow: vec![Vec::new(); roots.len()],
                    targets: vec![Vec::new(); roots.len()],
                };
                (
                    model.run(roots, &opts, Selection::Script(&mut none), device)?,
                    0,
                    0,
                    0,
                )
            }
            Some(r) => {
                let mut script = RecordedScript {
                    follow: r.follow.clone(),
                    targets: r.targets.clone(),
                };
                let out = model.run(roots, &opts, Selection::Script(&mut script), device)?;
                let d = check_replay(r, &out, recipe.replay_policy_tolerance)?;
                max_diff = max_diff.max(d);
                (out, r.supervised, r.refute_steps, r.proofs_completed)
            }
        };

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
        let mut loss = policy_sum * (base.policy_weight as f32 / n_total as f32);
        let mut sel_value = 0.0f64;
        if let Some((sel_sum, count)) = selector_nll_sum(&out.selector_steps) {
            anyhow::ensure!(
                count == supervised,
                "micro {mi}: {count} selector decisions recorded but Pass A supervised {supervised}"
            );
            sel_value = scalar(sel_sum.clone());
            anyhow::ensure!(
                sel_value.is_finite(),
                "non-finite selector loss at micro {mi}"
            );
            loss = loss + sel_sum * (base.selector_weight as f32 / sup_total.max(1) as f32);
        } else {
            anyhow::ensure!(
                supervised == 0,
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
        pb.supervised_decisions += supervised;
        pb.refute_steps += refute;
        pb.proofs_completed += proofs;

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
    let policy_loss = base.policy_weight * policy_sum_all / n_total as f64;
    let selector_loss = if sup_total > 0 {
        sel_sum_all / sup_total as f64
    } else {
        0.0
    };
    Ok((
        grads,
        UpdateReport35 {
            policy_loss,
            selector_loss,
            total_loss: policy_loss + base.selector_weight * selector_loss,
            grad_norm,
            examples: n_total,
            supervised_decisions: sup_total,
            max_replay_policy_diff: max_diff,
            rollout_s,
            replay_backward_s: t_b.elapsed().as_secs_f64(),
            per_budget,
        },
    ))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct P35State {
    pub schema: String,
    pub recipe_digest: String,
    pub recipe: Recipe35,
    pub updates_done: u64,
    pub sampler_draws: Vec<u64>,
    pub history: Vec<UpdateRecord35>,
    #[serde(default)]
    pub resumptions: u32,
}

pub struct Trainer35<B: AutodiffBackend> {
    pub recipe: Recipe35,
    pub model: ActiveSearchModel<B>,
    pub optim: Optim<B>,
    pub samplers: BudgetSamplers,
    pub updates_done: u64,
    pub history: Vec<UpdateRecord35>,
    pub resumptions: u32,
}

/// Verify an init directory is the seed's selected P5 final checkpoint and load its
/// WEIGHTS ONLY (the optimizer state is discarded).
pub fn load_init_weights<B: AutodiffBackend>(
    recipe: &Recipe35,
    dir: &Path,
    device: &B::Device,
) -> anyhow::Result<ActiveSearchModel<B>> {
    let seed = recipe
        .seed()
        .ok_or_else(|| anyhow::anyhow!("recipe has no seed"))?;
    let st: P5State = serde_json::from_slice(&std::fs::read(dir.join("p5-state.json"))?)?;
    anyhow::ensure!(
        st.recipe_digest == recipe.init_p5_recipe_digest && st.recipe.digest() == st.recipe_digest,
        "init checkpoint carries P5 recipe {} but seed {seed} requires {}",
        st.recipe_digest,
        recipe.init_p5_recipe_digest
    );
    let mut cleared = st.recipe.clone();
    cleared.seed = None;
    anyhow::ensure!(
        cleared.digest() == SELECTED_P5_DIGEST
            && st.recipe.seed == Some(seed)
            && st.recipe.peak_lr == recipe.base.peak_lr,
        "init checkpoint is not the selected P5 recipe at the selected LR for seed {seed}"
    );
    anyhow::ensure!(
        st.updates_done == st.recipe.updates && st.recipe.model == recipe.base.model,
        "init checkpoint is not a complete P5 run of the frozen model config"
    );
    let template = ActiveSearchModel::<B>::new(recipe.base.model.clone(), device);
    let (model, _discarded_optimizer, meta) = load_training::<B, _, _>(
        &dir.join("checkpoint"),
        template,
        adamw::<B, ActiveSearchModel<B>>(),
        device,
    )?;
    anyhow::ensure!(
        meta.architecture == "active_search_v3"
            && meta.precision == "fp32"
            && meta.recurrence == 1
            && !meta.deep_supervision
            && meta.step == st.updates_done,
        "init checkpoint metadata is not a final P5 fp32 active_search_v3 checkpoint"
    );
    Ok(model)
}

impl<B: AutodiffBackend> Trainer35<B> {
    /// A fresh V3.5 run from the seed's P5 final weights and a fresh optimizer.
    pub fn new(
        recipe: Recipe35,
        train: &Dataset,
        init_dir: &Path,
        device: &B::Device,
    ) -> anyhow::Result<Self> {
        recipe.validate_for_training()?;
        let seed = recipe.seed().expect("validated");
        let model = load_init_weights::<B>(&recipe, init_dir, device)?;
        Ok(Self {
            samplers: BudgetSamplers::new(&train.cells(), seed),
            optim: adamw::<B, ActiveSearchModel<B>>(),
            model,
            recipe,
            updates_done: 0,
            history: Vec::new(),
            resumptions: 0,
        })
    }

    pub fn plan_next(&mut self) -> UpdatePlan {
        let micro = self.recipe.base.micro;
        let micros = self
            .recipe
            .base
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

    pub fn step(&mut self, data: &Dataset, device: &B::Device) -> anyhow::Result<UpdateRecord35> {
        anyhow::ensure!(
            self.updates_done < self.recipe.base.updates,
            "the run is complete"
        );
        let t0 = Instant::now();
        let lr = lr_at(
            self.updates_done,
            self.recipe.base.peak_lr.expect("validated"),
            self.recipe.base.warmup,
            self.recipe.base.updates,
        );
        let plan = self.plan_next();
        let (grads, report) = compute_update(&self.model, data, &plan, &self.recipe, device)?;
        anyhow::ensure!(
            report.total_loss.is_finite(),
            "non-finite total loss at update {}",
            self.updates_done
        );
        self.model = self.optim.step(lr, self.model.clone(), grads);
        let rec = UpdateRecord35 {
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
        let meta = CheckpointMeta::new(
            self.recipe.base.model.clone(),
            1,
            false,
            self.updates_done,
            self.recipe.base.peak_lr.unwrap_or(0.0),
            self.recipe.seed().unwrap_or(0),
            0,
            BACKEND_TAG,
            "fp32",
        );
        save_training::<B, _, _>(&dir.join("checkpoint"), &self.model, &self.optim, &meta)?;
        let state = P35State {
            schema: STATE_SCHEMA.into(),
            recipe_digest: self.recipe.digest(),
            recipe: self.recipe.clone(),
            updates_done: self.updates_done,
            sampler_draws: self.samplers.draws(),
            history: self.history.clone(),
            resumptions: self.resumptions,
        };
        let tmp = dir.join("p35-state.json.tmp");
        std::fs::write(&tmp, serde_json::to_vec(&state)?)?;
        std::fs::rename(&tmp, dir.join("p35-state.json"))?;
        Ok(())
    }

    pub fn load(
        dir: &Path,
        recipe: Recipe35,
        train: &Dataset,
        device: &B::Device,
    ) -> anyhow::Result<Self> {
        recipe.validate_for_training()?;
        let state: P35State = serde_json::from_slice(&std::fs::read(dir.join("p35-state.json"))?)?;
        anyhow::ensure!(state.schema == STATE_SCHEMA, "unknown V3.5 state schema");
        anyhow::ensure!(
            state.recipe_digest == recipe.digest() && state.recipe == recipe,
            "the checkpoint was written under recipe {} but this run uses {}: refusing to resume",
            state.recipe_digest,
            recipe.digest()
        );
        let seed = recipe.seed().expect("validated");
        let peak_lr = recipe.base.peak_lr.expect("validated");
        Self::check_state(&state, &recipe, peak_lr)?;
        let template = ActiveSearchModel::<B>::new(recipe.base.model.clone(), device);
        let (model, optim, meta) = load_training::<B, _, _>(
            &dir.join("checkpoint"),
            template,
            adamw::<B, ActiveSearchModel<B>>(),
            device,
        )?;
        anyhow::ensure!(
            meta.architecture == "active_search_v3"
                && meta.backend == BACKEND_TAG
                && meta.precision == "fp32"
                && meta.recurrence == 1
                && !meta.deep_supervision,
            "checkpoint is not a V3.5 fp32 active_search_v3 checkpoint ({} / {} / {})",
            meta.architecture,
            meta.backend,
            meta.precision
        );
        anyhow::ensure!(
            meta.step == state.updates_done
                && meta.update_counter == state.updates_done
                && meta.lr_schedule_step == state.updates_done,
            "checkpoint metadata disagrees with the sidecar"
        );
        anyhow::ensure!(
            meta.seed == seed && meta.lr == peak_lr,
            "checkpoint was written for another seed / peak LR"
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
            resumptions: state.resumptions + 1,
        })
    }

    fn check_state(state: &P35State, recipe: &Recipe35, peak_lr: f64) -> anyhow::Result<()> {
        let b = &recipe.base;
        anyhow::ensure!(
            state.updates_done <= b.updates,
            "sidecar records {} updates but the recipe has {}",
            state.updates_done,
            b.updates
        );
        anyhow::ensure!(
            state.history.len() as u64 == state.updates_done,
            "sidecar history has {} records for {} updates",
            state.history.len(),
            state.updates_done
        );
        for (i, h) in state.history.iter().enumerate() {
            let r = &h.report;
            anyhow::ensure!(h.update == i as u64, "history record {i} mislabelled");
            anyhow::ensure!(
                h.lr.is_finite()
                    && h.wall_s.is_finite()
                    && r.policy_loss.is_finite()
                    && r.selector_loss.is_finite()
                    && r.total_loss.is_finite()
                    && r.grad_norm.is_finite(),
                "history record {i} holds a non-finite value"
            );
            let want = lr_at(i as u64, peak_lr, b.warmup, b.updates);
            anyhow::ensure!(
                (h.lr - want).abs() <= 1e-12 * want.abs().max(1e-12),
                "history record {i} has lr {} but the schedule gives {want}",
                h.lr
            );
        }
        anyhow::ensure!(
            state.sampler_draws.len() == BUDGETS.len(),
            "sampler state has the wrong shape"
        );
        for (&bd, &n) in BUDGETS.iter().zip(&state.sampler_draws) {
            let want = state.updates_done * draws_per_update(recipe, bd);
            anyhow::ensure!(
                n == want,
                "budget {bd} sampler recorded {n} draws but {} updates imply {want}",
                state.updates_done
            );
        }
        Ok(())
    }

    pub fn inference_model(&self) -> ActiveSearchModel<B::InnerBackend> {
        self.model.valid()
    }
}
