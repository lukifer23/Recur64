//! The frozen P6 ALL-INFO training recipe and its identity (`docs/V3_P6_PLAN.md`).

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use recur64_model::config::{Architecture, ModelConfig};
use recur64_model::train::OPTIMIZER_CONTRACT;

use crate::p5::recipe::{
    EFFECTIVE_BATCH, LR_SCHEDULE, TRAIN_DIGEST, TRAIN_POSITIONS, TUNE_DIGEST, TUNE_POSITIONS,
    UPDATES, WARMUP,
};
use crate::proof::generator::mix;
use crate::proof::sampler;

pub const RECIPE_SCHEMA: &str = "v3_p6_recipe_v1";

/// The peak LR selected by the P5 screen (V3-D21). No P6 LR screen exists.
pub const SELECTED_LR: f64 = 3.0e-4;
/// The three paired model seeds of the information-sufficiency control.
pub const P6_SEEDS: [u64; 3] = [5101, 5102, 5103];
/// The Gate I cell.
pub const GATE_CELL: &str = "KQRvK M3";
/// ALL-INFO is evaluated at the final update only; no checkpoint is selected on TUNE.
pub const EVAL_UPDATES: [u64; 1] = [800];

pub const LOSS: &str = "root_policy_ce_only";
pub const LOSS_NORMALISATION: &str = "L = sum(policy CE over every example of the update)/N_examples, N = 128; \\
the normaliser is the whole optimizer update's count, never a microbatch's. No selector loss, no WDL loss";
pub const SAMPLER_SEED_RULE: &str = "one cell_balanced_v1 sampler over the 15 P25 TRAIN cells, seed = mix(run_seed ^ mix(0x5A3D6000))";

/// Frozen fallback ladder of physical layouts, all with effective batch 128: the first
/// that runs, fits, is stable and projects under two hours per run is chosen from
/// TRAIN-side systems evidence only.
pub const LAYOUT_LADDER: [(usize, usize); 4] = [(16, 8), (8, 16), (4, 32), (2, 64)];

/// Seed of the single cell-balanced sampler of a run.
pub fn sampler_seed(run_seed: u64) -> u64 {
    mix(run_seed ^ mix(0x5A3D_6000))
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct P6Recipe {
    pub schema: String,
    pub model: ModelConfig,
    pub p25_data_digest: String,
    pub v3_tune_digest: String,
    pub input_contract: String,
    pub sampler: String,
    pub sampler_seed_rule: String,
    pub micro: usize,
    pub accum: usize,
    pub effective_batch: usize,
    pub updates: u64,
    pub warmup: u64,
    pub lr_schedule: String,
    pub peak_lr: Option<f64>,
    pub seed: Option<u64>,
    pub loss: String,
    pub policy_weight: f64,
    pub selector_weight: f64,
    pub wdl_weight: f64,
    pub normalisation: String,
    pub optimizer_contract: String,
    pub precision: String,
    pub eval_updates: Vec<u64>,
}

impl P6Recipe {
    /// The frozen contract for a physical layout: everything except the seed.
    pub fn contract(micro: usize, accum: usize) -> Self {
        Self {
            schema: RECIPE_SCHEMA.into(),
            model: ModelConfig::all_info_v1(),
            p25_data_digest: TRAIN_DIGEST.into(),
            v3_tune_digest: TUNE_DIGEST.into(),
            input_contract: recur64_model::config::ALL_INFO_INPUT.into(),
            sampler: sampler::VERSION.into(),
            sampler_seed_rule: SAMPLER_SEED_RULE.into(),
            micro,
            accum,
            effective_batch: micro * accum,
            updates: UPDATES,
            warmup: WARMUP,
            lr_schedule: LR_SCHEDULE.into(),
            peak_lr: Some(SELECTED_LR),
            seed: None,
            loss: LOSS.into(),
            policy_weight: 1.0,
            selector_weight: 0.0,
            wdl_weight: 0.0,
            normalisation: LOSS_NORMALISATION.into(),
            optimizer_contract: OPTIMIZER_CONTRACT.into(),
            precision: "fp32".into(),
            eval_updates: EVAL_UPDATES.to_vec(),
        }
    }

    pub fn for_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn digest(&self) -> String {
        let bytes = serde_json::to_vec(self).expect("a recipe serialises");
        format!("{:x}", Sha256::digest(bytes))
    }

    /// Identity with the seed removed: what the three ALL-INFO runs share.
    pub fn contract_digest(&self) -> String {
        let mut c = self.clone();
        c.seed = None;
        c.digest()
    }

    /// Structural validation used by the trainer (production builds enforce the full frozen
    /// contract; unit tests run tiny geometries and keep the structural rules).
    pub fn validate_for_training(&self) -> anyhow::Result<()> {
        #[cfg(not(test))]
        {
            self.validate()
        }
        #[cfg(test)]
        {
            anyhow::ensure!(self.schema == RECIPE_SCHEMA, "unknown recipe schema");
            anyhow::ensure!(
                self.micro * self.accum == self.effective_batch,
                "inconsistent layout"
            );
            anyhow::ensure!(self.precision == "fp32", "FP32 only");
            anyhow::ensure!(
                self.model.architecture == Architecture::AllInfoV1,
                "P6 trains all_info_v1"
            );
            self.model.validate()
        }
    }

    /// The recipe must describe the authorised P6 configuration exactly.
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(self.schema == RECIPE_SCHEMA, "unknown recipe schema");
        anyhow::ensure!(
            self.model.architecture == Architecture::AllInfoV1,
            "P6 trains all_info_v1"
        );
        anyhow::ensure!(
            LAYOUT_LADDER.contains(&(self.micro, self.accum))
                && self.effective_batch == EFFECTIVE_BATCH
                && self.micro * self.accum == EFFECTIVE_BATCH,
            "the physical layout must be on the frozen ladder with effective batch {EFFECTIVE_BATCH}"
        );
        anyhow::ensure!(
            self.updates == UPDATES && self.warmup == WARMUP,
            "updates and warmup are frozen at {UPDATES}/{WARMUP}"
        );
        anyhow::ensure!(
            self.peak_lr == Some(SELECTED_LR),
            "P6 uses the selected P5 LR {SELECTED_LR:e}; there is no P6 LR screen"
        );
        anyhow::ensure!(
            self.policy_weight == 1.0 && self.selector_weight == 0.0 && self.wdl_weight == 0.0,
            "loss weights are frozen: policy 1, selector 0, WDL 0"
        );
        anyhow::ensure!(self.precision == "fp32", "FP32 only");
        anyhow::ensure!(
            self.p25_data_digest == TRAIN_DIGEST && self.v3_tune_digest == TUNE_DIGEST,
            "the recipe names data identities other than the frozen ones"
        );
        anyhow::ensure!(
            self.eval_updates == EVAL_UPDATES,
            "ALL-INFO is evaluated at update {UPDATES} only"
        );
        anyhow::ensure!(
            self.input_contract == recur64_model::config::ALL_INFO_INPUT
                && self.loss == LOSS
                && self.model == ModelConfig::all_info_v1(),
            "the recipe names a model, input or loss other than the frozen ones"
        );
        self.model.validate()
    }
}

/// Frozen dataset sizes (re-exported for the loaders).
pub const TRAIN_N: usize = TRAIN_POSITIONS;
pub const TUNE_N: usize = TUNE_POSITIONS;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_contract_validates_and_every_scientific_field_changes_the_digest() {
        let c = P6Recipe::contract(16, 8);
        c.clone().for_seed(5101).validate().unwrap();
        let base = c.clone().for_seed(5101).digest();
        assert_ne!(base, c.clone().for_seed(5102).digest());
        assert_eq!(
            c.clone().for_seed(5101).contract_digest(),
            c.contract_digest()
        );
        for mutate in [
            (|r: &mut P6Recipe| r.peak_lr = Some(1.5e-4)) as fn(&mut P6Recipe),
            |r| r.selector_weight = 1.0,
            |r| r.wdl_weight = 0.5,
            |r| r.updates = 801,
            |r| r.model.all_info.as_mut().unwrap().set_ffn = 512,
            |r| r.loss = "x".into(),
        ] {
            let mut r = c.clone().for_seed(5101);
            mutate(&mut r);
            assert_ne!(r.digest(), base);
            assert!(r.validate().is_err(), "a changed recipe must be refused");
        }
    }

    #[test]
    fn only_ladder_layouts_with_effective_batch_128_are_valid() {
        for (m, a) in LAYOUT_LADDER {
            P6Recipe::contract(m, a).for_seed(1).validate().unwrap();
        }
        assert!(P6Recipe::contract(32, 4).for_seed(1).validate().is_err());
        assert!(P6Recipe::contract(16, 4).for_seed(1).validate().is_err());
    }

    #[test]
    fn sampler_seeds_differ_per_run_seed() {
        assert_ne!(sampler_seed(5101), sampler_seed(5102));
        assert_ne!(sampler_seed(5102), sampler_seed(5103));
    }
}
