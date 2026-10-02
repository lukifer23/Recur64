//! The frozen V3.5 recipe and its identity (`on_policy_proof_relabel_v1`).

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::p5::recipe::{BUDGETS, EFFECTIVE_BATCH, Layout, Recipe, TEACHER_ID, UPDATES, WARMUP};

pub const RECIPE_SCHEMA: &str = "v35_recipe_v1";
pub const TRAINING_ID: &str = "on_policy_proof_relabel_v1";
pub const SELECTED_LR: f64 = 3.0e-4;
/// Digest (seed cleared) of the selected P5 recipe every init checkpoint must carry.
pub const SELECTED_P5_DIGEST: &str =
    "a069ba9d18befed65f970aca253b47780365fd7019be38283f270d79d6c1db33";

/// Seed -> digest of the P5 selected-recipe final checkpoint it initialises from.
pub const INIT_CHECKPOINTS: [(u64, &str); 3] = [
    (
        5101,
        "c9d90281f82f813925a6ef373eab77a98b84eaa96ed6b2111f4d6c60c6d961a1",
    ),
    (
        5102,
        "20b30924b86afb50b4f828861c8a271b44eabc1dcf83645145d37c5a20120802",
    ),
    (
        5103,
        "9238476339a90801067b9ebee0f52310eca982d8d8ef827a4d2a42a569eab0f4",
    ),
];

pub const ROLLOUT_SEMANTICS: &str = "the learner selects EVERY query (ACTIVE argmax); ProofTrace labels the \
learner-induced queried set S with the uniform A_proof(S) U A_refute(S) and never chooses, overrides or \
perturbs the executed edge; no latch, no filler schedule, no teacher mixing; proof completion only masks \
the selector loss";

pub const UPDATE_SEMANTICS: &str = "two-pass: Pass A is a detached ACTIVE rollout on the inference copy of \
the current weights recording learner edges and oracle targets; Pass B replays the recorded edges through \
the autodiff model and is refused unless it reproduces Pass A (paths, counts, accounting, policy)";

pub const LOSS_NORMALISATION: &str = "L = sum(policy CE over every example of the update)/N_examples + 1.0 * \
sum(selector NLL over every supervised learner-visited decision of the update)/N_supervised_decisions; \
each normaliser is the whole optimizer update's count, known exactly after Pass A";

pub const INIT_SEMANTICS: &str = "weights only from the matching seed's selected P5 final checkpoint; fresh \
adamw-v1 optimizer state (momentum is not inherited)";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Recipe35 {
    pub schema: String,
    pub training: String,
    /// The shared P5 fields (data, sampler, budgets, layout, schedule, loss weights,
    /// optimizer, precision). `teacher`, `latch` and `normalisation` carry V3.5 text.
    pub base: Recipe,
    pub rollout: String,
    pub update: String,
    pub init: String,
    pub init_p5_recipe_digest: String,
    pub selected_p5_digest: String,
    pub replay_policy_tolerance: f64,
}

/// Maximum |log-prob| difference between Pass A and Pass B (same weights, same path).
pub const REPLAY_POLICY_TOLERANCE: f64 = 1e-4;

pub fn init_digest_for(seed: u64) -> Option<&'static str> {
    INIT_CHECKPOINTS
        .iter()
        .find(|(s, _)| *s == seed)
        .map(|(_, d)| *d)
}

impl Recipe35 {
    pub fn contract(layout: Layout, health_checks: bool) -> Self {
        let mut base = Recipe::screen_contract(layout, health_checks);
        base.teacher = TRAINING_ID.into();
        base.latch = "none: proof completion masks the selector loss only and never alters the forward schedule".into();
        base.normalisation = LOSS_NORMALISATION.into();
        Self {
            schema: RECIPE_SCHEMA.into(),
            training: TRAINING_ID.into(),
            base,
            rollout: ROLLOUT_SEMANTICS.into(),
            update: UPDATE_SEMANTICS.into(),
            init: INIT_SEMANTICS.into(),
            init_p5_recipe_digest: String::new(),
            selected_p5_digest: SELECTED_P5_DIGEST.into(),
            replay_policy_tolerance: REPLAY_POLICY_TOLERANCE,
        }
    }

    pub fn for_seed(mut self, seed: u64) -> anyhow::Result<Self> {
        let d = init_digest_for(seed)
            .ok_or_else(|| anyhow::anyhow!("seed {seed} is not a V3.5 seed (5101/5102/5103)"))?;
        self.base = self.base.for_run(SELECTED_LR, seed);
        self.init_p5_recipe_digest = d.into();
        Ok(self)
    }

    pub fn seed(&self) -> Option<u64> {
        self.base.seed
    }

    pub fn digest(&self) -> String {
        let bytes = serde_json::to_vec(self).expect("a recipe serialises");
        format!("{:x}", Sha256::digest(bytes))
    }

    /// Identity with the seed (and the seed-bound init digest) removed.
    pub fn contract_digest(&self) -> String {
        let mut c = self.clone();
        c.base.seed = None;
        c.init_p5_recipe_digest.clear();
        c.digest()
    }

    pub fn layout(&self) -> Layout {
        self.base.layout()
    }

    pub fn validate_for_training(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.schema == RECIPE_SCHEMA && self.training == TRAINING_ID,
            "unknown V3.5 recipe schema"
        );
        anyhow::ensure!(
            self.base.teacher == TRAINING_ID && self.base.teacher != TEACHER_ID,
            "the V3.5 recipe must not name the P5 teacher"
        );
        anyhow::ensure!(
            self.base.budgets == BUDGETS && !self.base.budget_sequence.contains(&16),
            "V3.5 trains budgets {{0,2,4,8}} only; B16 is never trained"
        );
        anyhow::ensure!(
            self.base.micro * self.base.accum == self.base.effective_batch
                && self.base.budget_sequence.len() == self.base.accum,
            "inconsistent layout"
        );
        anyhow::ensure!(self.base.precision == "fp32", "FP32 only");
        anyhow::ensure!(
            self.base.policy_weight == 1.0
                && self.base.selector_weight == 1.0
                && self.base.wdl_weight == 0.0,
            "loss weights are frozen: policy 1, selector 1, WDL 0"
        );
        anyhow::ensure!(
            self.replay_policy_tolerance == REPLAY_POLICY_TOLERANCE,
            "the replay tolerance is frozen"
        );
        let seed = self
            .base
            .seed
            .ok_or_else(|| anyhow::anyhow!("a V3.5 recipe needs a seed"))?;
        anyhow::ensure!(
            self.base.peak_lr == Some(SELECTED_LR),
            "V3.5 uses the selected peak LR {SELECTED_LR:e} only (no LR screen)"
        );
        anyhow::ensure!(
            init_digest_for(seed) == Some(self.init_p5_recipe_digest.as_str()),
            "the init checkpoint digest is not the frozen one for seed {seed}"
        );
        anyhow::ensure!(
            self.selected_p5_digest == SELECTED_P5_DIGEST,
            "the selected P5 recipe identity changed"
        );
        #[cfg(not(test))]
        {
            self.base.validate()?;
            anyhow::ensure!(
                self.base.effective_batch == EFFECTIVE_BATCH
                    && self.base.updates == UPDATES
                    && self.base.warmup == WARMUP,
                "updates / warmup / batch are frozen"
            );
        }
        #[cfg(test)]
        {
            let _ = (EFFECTIVE_BATCH, UPDATES, WARMUP);
            self.base.model.validate()?;
        }
        Ok(())
    }
}
