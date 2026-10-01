//! The frozen P5 training recipe and its identity.
//!
//! The recipe is a plain, fully serialised description of everything that
//! determines a P5 run. Its SHA-256 is the run identity: it is written into every
//! checkpoint sidecar and verified on every load, so a checkpoint can never be
//! resumed or evaluated under a different recipe by accident.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use recur64_model::config::ModelConfig;
use recur64_model::train::OPTIMIZER_CONTRACT;

use crate::proof::generator::mix;
use crate::proof::sampler;

pub const RECIPE_SCHEMA: &str = "v3_p5_recipe_v1";

/// The versioned P5 training teacher (see `teacher.rs`).
pub const TEACHER_ID: &str = "proof_teacher_seeded_v1";

/// Semantics of the proof-completion latch, recorded in the identity.
pub const LATCH_SEMANTICS: &str = "once the proof residual is 0 the example latches complete for the rest of the \
episode: no selector target ever again, remaining forced budget consumed by fixed_bfs_actionid_v1";

/// Normalisation, recorded in the identity.
pub const LOSS_NORMALISATION: &str = "L = sum(policy CE over every example of the update)/N_examples + 1.0 * \
sum(selector NLL over every supervised decision of the update)/N_supervised_decisions; each normaliser is \
the whole optimizer update's count, never a microbatch's";

pub const LR_SCHEDULE: &str = "linear_warmup_then_cosine_v1 (recur64_runtime::learner::lr_at)";

/// Training budgets. B16 is never trained.
pub const BUDGETS: [usize; 4] = [0, 2, 4, 8];

/// Frozen expected inputs.
pub const TRAIN_DIGEST: &str = "3b25dc8549dd2fc9d47c30e294c273b3306aecb3eba91b964715326ddf74f2e6";
pub const TRAIN_POSITIONS: usize = 44_332;
pub const TRAIN_TRACE_MANIFEST: &str =
    "8160734ed5e3a145c12dd72d8e9dc49cc893984fc1cf714ea93eba904f58c488";
pub const TUNE_DIGEST: &str = "c66018657009c9c5eade58369b5f451466d6662c910f76b8aacddcac99921b53";
pub const TUNE_POSITIONS: usize = 4_500;
pub const TUNE_TRACE_MANIFEST: &str =
    "acb786fad82573c9b4427652e5e1b060f7b312c5f5e9121e02980c21cfaa6d25";

/// The preregistered candidate learning rates and screen seeds.
pub const CANDIDATE_LRS: [f64; 3] = [7.5e-5, 1.5e-4, 3.0e-4];
pub const SCREEN_SEEDS: [u64; 2] = [5101, 5102];
pub const UPDATES: u64 = 800;
pub const WARMUP: u64 = 80;
pub const EFFECTIVE_BATCH: usize = 128;
pub const EVAL_UPDATES: [u64; 5] = [0, 200, 400, 600, 800];

/// The physical layout of one optimizer update.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Layout {
    pub micro: usize,
    pub accum: usize,
}

impl Layout {
    /// The preregistered default.
    pub const DEFAULT: Layout = Layout {
        micro: 16,
        accum: 8,
    };
    /// The one pre-authorised hardware-only fallback (OOM or device instability).
    pub const FALLBACK: Layout = Layout {
        micro: 8,
        accum: 16,
    };

    pub fn effective_batch(&self) -> usize {
        self.micro * self.accum
    }

    /// The budget of each microbatch of an update: `[0,2,4,8]` repeated, so every
    /// budget gets exactly one quarter of the effective batch.
    pub fn budget_sequence(&self) -> Vec<usize> {
        (0..self.accum).map(|j| BUDGETS[j % 4]).collect()
    }
}

/// Seed of the independent cell-balanced sampler of one budget.
pub fn sampler_seed(run_seed: u64, budget: usize) -> u64 {
    mix(run_seed ^ mix(0x5A3D_0000 + budget as u64))
}

/// Base key of the teacher's tie choices.
pub fn teacher_key_base(run_seed: u64) -> u64 {
    mix(run_seed ^ 0x7EAC_4E21_0000_0001)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Recipe {
    pub schema: String,
    pub model: ModelConfig,
    pub p25_data_digest: String,
    pub train_trace_manifest: String,
    pub v3_tune_digest: String,
    pub tune_trace_manifest: String,
    pub sampler: String,
    pub sampler_seed_rule: String,
    pub teacher: String,
    pub latch: String,
    pub budgets: Vec<usize>,
    pub budget_sequence: Vec<usize>,
    pub micro: usize,
    pub accum: usize,
    pub effective_batch: usize,
    pub updates: u64,
    pub warmup: u64,
    pub lr_schedule: String,
    /// `None` in the pre-screen contract; set per run and in the selected recipe.
    pub peak_lr: Option<f64>,
    pub policy_weight: f64,
    pub selector_weight: f64,
    pub wdl_weight: f64,
    pub normalisation: String,
    pub optimizer_contract: String,
    /// `None` in the pre-screen contract.
    pub seed: Option<u64>,
    pub precision: String,
    pub health_checks: bool,
    pub eval_updates: Vec<u64>,
}

impl Recipe {
    /// The frozen pre-screen contract for a physical layout: everything except the
    /// peak learning rate and the seed.
    pub fn screen_contract(layout: Layout, health_checks: bool) -> Self {
        Self {
            schema: RECIPE_SCHEMA.into(),
            model: ModelConfig::active_search_v3(),
            p25_data_digest: TRAIN_DIGEST.into(),
            train_trace_manifest: TRAIN_TRACE_MANIFEST.into(),
            v3_tune_digest: TUNE_DIGEST.into(),
            tune_trace_manifest: TUNE_TRACE_MANIFEST.into(),
            sampler: sampler::VERSION.into(),
            sampler_seed_rule: "one independent cell_balanced_v1 sampler per budget, seed = mix(run_seed ^ mix(0x5A3D0000 + budget))"
                .into(),
            teacher: TEACHER_ID.into(),
            latch: LATCH_SEMANTICS.into(),
            budgets: BUDGETS.to_vec(),
            budget_sequence: layout.budget_sequence(),
            micro: layout.micro,
            accum: layout.accum,
            effective_batch: layout.effective_batch(),
            updates: UPDATES,
            warmup: WARMUP,
            lr_schedule: LR_SCHEDULE.into(),
            peak_lr: None,
            policy_weight: 1.0,
            selector_weight: 1.0,
            wdl_weight: 0.0,
            normalisation: LOSS_NORMALISATION.into(),
            optimizer_contract: OPTIMIZER_CONTRACT.into(),
            seed: None,
            precision: "fp32".into(),
            health_checks,
            eval_updates: EVAL_UPDATES.to_vec(),
        }
    }

    /// The recipe of one run.
    pub fn for_run(mut self, lr: f64, seed: u64) -> Self {
        self.peak_lr = Some(lr);
        self.seed = Some(seed);
        self
    }

    /// Run identity: SHA-256 over the canonical JSON of the whole recipe.
    pub fn digest(&self) -> String {
        let bytes = serde_json::to_vec(self).expect("a recipe serialises");
        format!("{:x}", Sha256::digest(bytes))
    }

    /// Identity of the recipe with the learning rate and seed removed: what the six
    /// screen runs share.
    pub fn contract_digest(&self) -> String {
        let mut c = self.clone();
        c.peak_lr = None;
        c.seed = None;
        c.digest()
    }

    pub fn layout(&self) -> Layout {
        Layout {
            micro: self.micro,
            accum: self.accum,
        }
    }

    /// What the trainer checks. Production builds enforce the full frozen contract;
    /// unit tests run tiny geometries and keep only the structural rules (budgets,
    /// loss weights, precision, model).
    pub fn validate_for_training(&self) -> anyhow::Result<()> {
        #[cfg(not(test))]
        {
            self.validate()
        }
        #[cfg(test)]
        {
            anyhow::ensure!(self.schema == RECIPE_SCHEMA, "unknown recipe schema");
            anyhow::ensure!(
                self.budgets == BUDGETS && !self.budget_sequence.contains(&16),
                "P5 trains budgets {{0,2,4,8}} only"
            );
            anyhow::ensure!(
                self.micro * self.accum == self.effective_batch
                    && self.budget_sequence.len() == self.accum,
                "inconsistent layout"
            );
            anyhow::ensure!(self.precision == "fp32", "FP32 only");
            self.model.validate()
        }
    }

    /// Internal consistency; the recipe must describe an authorised configuration.
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(self.schema == RECIPE_SCHEMA, "unknown recipe schema");
        anyhow::ensure!(
            self.budgets == BUDGETS && !self.budgets.contains(&16),
            "P5 trains budgets {{0,2,4,8}} only; B16 is never trained"
        );
        anyhow::ensure!(
            self.effective_batch == EFFECTIVE_BATCH
                && self.micro * self.accum == EFFECTIVE_BATCH
                && self.budget_sequence == self.layout().budget_sequence(),
            "the layout must give an effective batch of {EFFECTIVE_BATCH} with equal budget exposure"
        );
        let l = self.layout();
        anyhow::ensure!(
            l == Layout::DEFAULT || l == Layout::FALLBACK,
            "only micro16/accum8 and the pre-authorised micro8/accum16 fallback are allowed"
        );
        anyhow::ensure!(
            self.updates == UPDATES && self.warmup == WARMUP,
            "updates and warmup are frozen at {UPDATES}/{WARMUP}"
        );
        anyhow::ensure!(
            self.policy_weight == 1.0 && self.selector_weight == 1.0 && self.wdl_weight == 0.0,
            "loss weights are frozen: policy 1, selector 1, WDL 0"
        );
        anyhow::ensure!(self.precision == "fp32", "FP32 only");
        anyhow::ensure!(
            self.p25_data_digest == TRAIN_DIGEST
                && self.train_trace_manifest == TRAIN_TRACE_MANIFEST
                && self.v3_tune_digest == TUNE_DIGEST
                && self.tune_trace_manifest == TUNE_TRACE_MANIFEST,
            "the recipe names data or trace identities other than the frozen ones"
        );
        self.model.validate()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouts_give_equal_budget_exposure_and_the_frozen_batch() {
        for l in [Layout::DEFAULT, Layout::FALLBACK] {
            assert_eq!(l.effective_batch(), 128);
            let seq = l.budget_sequence();
            for b in BUDGETS {
                let micros = seq.iter().filter(|&&x| x == b).count();
                assert_eq!(micros * l.micro, 32, "{l:?} budget {b}");
            }
        }
        assert_eq!(
            Layout::DEFAULT.budget_sequence(),
            vec![0, 2, 4, 8, 0, 2, 4, 8]
        );
    }

    #[test]
    fn the_run_identity_changes_with_every_scientific_field() {
        let base = Recipe::screen_contract(Layout::DEFAULT, true).for_run(1.5e-4, 5101);
        let d = base.digest();
        let mut v = base.clone();
        v.peak_lr = Some(3.0e-4);
        assert_ne!(v.digest(), d);
        let mut v = base.clone();
        v.seed = Some(5102);
        assert_ne!(v.digest(), d);
        let mut v = base.clone();
        v.selector_weight = 0.75;
        assert_ne!(v.digest(), d);
        let mut v = base.clone();
        v.teacher = "other".into();
        assert_ne!(v.digest(), d);
        let mut v = base.clone();
        v.model.active.as_mut().unwrap().workspace_tokens = 9;
        assert_ne!(v.digest(), d);
        let mut v = base.clone();
        v.model.active.as_mut().unwrap().contracts.planner = "x".into();
        assert_ne!(v.digest(), d);
        // The shared contract ignores LR and seed only.
        let other = Recipe::screen_contract(Layout::DEFAULT, true).for_run(7.5e-5, 5102);
        assert_eq!(base.contract_digest(), other.contract_digest());
        assert_ne!(
            base.contract_digest(),
            Recipe::screen_contract(Layout::FALLBACK, true).contract_digest()
        );
    }

    #[test]
    fn invalid_recipes_are_refused() {
        let ok = Recipe::screen_contract(Layout::DEFAULT, true).for_run(1.5e-4, 5101);
        ok.validate().unwrap();
        let mut v = ok.clone();
        v.budgets.push(16);
        assert!(v.validate().is_err(), "B16 must not be trainable");
        let mut v = ok.clone();
        v.micro = 32;
        v.accum = 4;
        assert!(v.validate().is_err());
        let mut v = ok.clone();
        v.selector_weight = 0.5;
        assert!(v.validate().is_err());
        let mut v = ok.clone();
        v.wdl_weight = 0.1;
        assert!(v.validate().is_err());
        let mut v = ok.clone();
        v.p25_data_digest = "0".repeat(64);
        assert!(v.validate().is_err());
        let mut v = ok.clone();
        v.updates = 400;
        assert!(v.validate().is_err());
    }

    #[test]
    fn sampler_seeds_are_distinct_per_budget_and_seed() {
        let s: std::collections::HashSet<u64> =
            BUDGETS.iter().map(|&b| sampler_seed(5101, b)).collect();
        assert_eq!(s.len(), 4);
        assert_ne!(sampler_seed(5101, 0), sampler_seed(5102, 0));
        assert_ne!(teacher_key_base(5101), teacher_key_base(5102));
    }
}
