//! Frozen V5 stage recipes, deterministic samplers and full resumable checkpoints.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use burn::module::AutodiffModule;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{AdamW, GradientsAccumulator, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Bool, TensorData};
use recur64_model::train::{OPTIMIZER_CONTRACT, adamw};
use recur64_runtime::learner::lr_at;
use recur64_runtime::proof::sampler::{CellSampler, VERSION as CELL_SAMPLER};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::config::{ARCHITECTURE, V5Config};
use crate::data::{DEV_DIGEST, FIT_DIGEST, TRAIN_DIGEST, V5Data};
use crate::graph::{EpisodeKey, Schedule, acquire};
use crate::loss::correct_set_loss;
use crate::model::{BaseOutput, CounterfactualRelationalLoop, RootInputs, Treatment, V5Inputs};

pub const RECIPE_SCHEMA: &str = "v5_stage_recipe_v3";
pub const CHECKPOINT_SCHEMA: &str = "v5_training_checkpoint_v1";
pub const STAGE_A_UPDATES: u64 = 1_200;
pub const STAGE_B_UPDATES: u64 = 800;
pub const PILOT_SEED: u64 = 5_301;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    BaselineA,
    ReaderB,
}

impl Stage {
    pub fn label(self) -> &'static str {
        match self {
            Self::BaselineA => "stage_a",
            Self::ReaderB => "stage_b",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Condition {
    pub schedule: Schedule,
    pub q: usize,
    pub r: usize,
}

pub fn reader_conditions() -> Vec<Condition> {
    let mut out = Vec::with_capacity(18);
    for schedule in [Schedule::UniformFrontierV1, Schedule::BaseRankedDepthV1] {
        for q in [2, 4, 8] {
            for r in [1, 2, 4] {
                out.push(Condition { schedule, q, r });
            }
        }
    }
    out
}

/// Preregistered contract identity; future Stage B weight bindings do not exist
/// until the owner separately authorizes and completes Stage A.
pub fn frozen_recipe_contract(source: &str) -> anyhow::Result<serde_json::Value> {
    crate::data::verify_preregistered_bindings()?;
    Ok(serde_json::json!({
        "schema":"v5_stage_recipe_v3_contract", "scientific_recipe":RECIPE_SCHEMA,
        "source_sha":source,"architecture":ARCHITECTURE,
        "config_digest":V5Config::default().scientific_digest()?,
        "data_contract":crate::native_data_v2::CONTRACT,
        "train_identity":crate::native_data_v2::Role::Train.identity(),
        "dev_identity":crate::native_data_v2::Role::Dev.identity(),
        "train_digest":TRAIN_DIGEST,"train_record_content_digest":crate::data::TRAIN_CONTENT_DIGEST,
        "dev_digest":DEV_DIGEST,"dev_record_content_digest":crate::data::DEV_CONTENT_DIGEST,
        "train_count":27000,"train_cells":crate::data::binding(crate::native_data_v2::Role::Train)?.manifest.cell_counts,
        "dev_count":4500,"primary_cell_count":750,"sampler":CELL_SAMPLER,
        "seed":PILOT_SEED,"precision":"fp32","optimizer":OPTIMIZER_CONTRACT,
        "loss":crate::config::CORRECT_SET_LOSS,"warmup":80,"peak_lr":3e-4,
        "physical_microbatch":2,
        "stage_a":{"updates":1200,"effective_batch":64,"q":0,"reader_execution":false},
        "stage_b":{"updates":800,"effective_batch":36,"conditions":reader_conditions(),"examples_per_condition":2}
    }))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recipe {
    pub schema: String,
    pub architecture: String,
    pub config: V5Config,
    pub config_digest: String,
    pub source_sha: String,
    pub stage: Stage,
    pub seed: u64,
    pub updates: u64,
    pub warmup: u64,
    pub peak_lr: f64,
    pub effective_batch: usize,
    pub physical_microbatch: usize,
    pub accumulation_steps: usize,
    pub precision: String,
    pub optimizer: String,
    pub loss: String,
    pub sampler: String,
    pub data_contract: String,
    pub train_identity: String,
    pub dev_identity: String,
    pub train_cell_count: usize,
    pub train_digest: String,
    pub fit_digest: String,
    pub dev_digest: String,
    pub init_model_hash: Option<String>,
    pub baseline_fingerprint: Option<String>,
    pub conditions: Vec<Condition>,
}

impl Recipe {
    pub fn stage_a(source_sha: String, physical_microbatch: usize) -> anyhow::Result<Self> {
        Self::new(
            Stage::BaselineA,
            source_sha,
            physical_microbatch,
            None,
            None,
        )
    }

    pub fn stage_b(
        source_sha: String,
        physical_microbatch: usize,
        init_model_hash: String,
        baseline_fingerprint: String,
    ) -> anyhow::Result<Self> {
        Self::new(
            Stage::ReaderB,
            source_sha,
            physical_microbatch,
            Some(init_model_hash),
            Some(baseline_fingerprint),
        )
    }

    fn new(
        stage: Stage,
        source_sha: String,
        physical_microbatch: usize,
        init_model_hash: Option<String>,
        baseline_fingerprint: Option<String>,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            matches!(physical_microbatch, 1 | 2),
            "only the qualified physical microbatch 2 or fallback 1 is permitted"
        );
        let config = V5Config::default();
        let (updates, effective_batch) = match stage {
            Stage::BaselineA => (STAGE_A_UPDATES, 64),
            Stage::ReaderB => (STAGE_B_UPDATES, 36),
        };
        let recipe = Self {
            schema: RECIPE_SCHEMA.into(),
            architecture: ARCHITECTURE.into(),
            config_digest: config.scientific_digest()?,
            config,
            source_sha,
            stage,
            seed: PILOT_SEED,
            updates,
            warmup: 80,
            peak_lr: 3.0e-4,
            effective_batch,
            physical_microbatch,
            accumulation_steps: effective_batch / physical_microbatch,
            precision: "fp32".into(),
            optimizer: OPTIMIZER_CONTRACT.into(),
            loss: crate::config::CORRECT_SET_LOSS.into(),
            sampler: CELL_SAMPLER.into(),
            data_contract: crate::native_data_v2::CONTRACT.into(),
            train_identity: crate::native_data_v2::Role::Train.identity().into(),
            dev_identity: crate::native_data_v2::Role::Dev.identity().into(),
            train_cell_count: 9,
            train_digest: TRAIN_DIGEST.into(),
            fit_digest: FIT_DIGEST.into(),
            dev_digest: DEV_DIGEST.into(),
            init_model_hash,
            baseline_fingerprint,
            conditions: if stage == Stage::ReaderB {
                reader_conditions()
            } else {
                Vec::new()
            },
        };
        recipe.validate()?;
        Ok(recipe)
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.schema == RECIPE_SCHEMA && self.architecture == ARCHITECTURE,
            "V5 recipe identity mismatch"
        );
        self.config.validate()?;
        anyhow::ensure!(
            self.config.scientific_digest()? == self.config_digest,
            "V5 recipe config digest mismatch"
        );
        anyhow::ensure!(
            self.seed == PILOT_SEED,
            "only pilot seed 5301 is authorized"
        );
        anyhow::ensure!(
            self.warmup == 80
                && self.peak_lr == 3.0e-4
                && self.precision == "fp32"
                && self.optimizer == OPTIMIZER_CONTRACT
                && self.loss == crate::config::CORRECT_SET_LOSS
                && self.sampler == CELL_SAMPLER,
            "optimizer, schedule, precision, loss or sampler differs from the frozen recipe"
        );
        anyhow::ensure!(
            self.data_contract == crate::native_data_v2::CONTRACT
                && self.train_identity == crate::native_data_v2::Role::Train.identity()
                && self.dev_identity == crate::native_data_v2::Role::Dev.identity()
                && self.train_cell_count == 9
                && self.train_digest == TRAIN_DIGEST
                && self.fit_digest == FIT_DIGEST
                && self.dev_digest == DEV_DIGEST,
            "dataset identity differs from the frozen recipe"
        );
        anyhow::ensure!(
            matches!(self.physical_microbatch, 1 | 2)
                && self
                    .effective_batch
                    .is_multiple_of(self.physical_microbatch)
                && self.accumulation_steps == self.effective_batch / self.physical_microbatch,
            "invalid physical layout"
        );
        match self.stage {
            Stage::BaselineA => anyhow::ensure!(
                self.updates == STAGE_A_UPDATES
                    && self.effective_batch == 64
                    && self.init_model_hash.is_none()
                    && self.baseline_fingerprint.is_none()
                    && self.conditions.is_empty(),
                "Stage A recipe differs from the preregistration"
            ),
            Stage::ReaderB => anyhow::ensure!(
                self.updates == STAGE_B_UPDATES
                    && self.effective_batch == 36
                    && self.init_model_hash.is_some()
                    && self.baseline_fingerprint.is_some()
                    && self.conditions == reader_conditions(),
                "Stage B recipe differs from the preregistration"
            ),
        }
        Ok(())
    }

    pub fn digest(&self) -> anyhow::Result<String> {
        self.validate()?;
        let mut hash = Sha256::new();
        hash.update(b"recur64.v5.stage_recipe.v3\0");
        hash.update(serde_json::to_vec(self)?);
        Ok(format!("{:x}", hash.finalize()))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepRecord {
    pub update: u64,
    pub lr: f64,
    pub loss: f64,
    pub wall_seconds: f64,
    pub graph_manifest_digests: Vec<String>,
    pub cell_exposure: std::collections::BTreeMap<String, u64>,
    pub condition_cell_exposure:
        std::collections::BTreeMap<String, std::collections::BTreeMap<String, u64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMeta {
    pub schema: String,
    pub architecture: String,
    pub stage: Stage,
    pub recipe: Recipe,
    pub recipe_digest: String,
    pub config_digest: String,
    pub update: u64,
    pub model_hash: String,
    pub optimizer_hash: String,
    pub init_model_hash: Option<String>,
    pub backend: String,
    pub precision: String,
    pub factual_null_gradient_semantics: String,
    pub history: Vec<StepRecord>,
    pub resume_events: Vec<String>,
}

type Opt<B> = OptimizerAdaptor<AdamW, CounterfactualRelationalLoop<B>, B>;

pub struct Trainer<B: AutodiffBackend> {
    pub model: CounterfactualRelationalLoop<B>,
    optim: Opt<B>,
    pub recipe: Recipe,
    pub updates_done: u64,
    pub history: Vec<StepRecord>,
    pub resume_events: Vec<String>,
}

fn mix(a: u64, b: u64) -> u64 {
    let mut x = a ^ b.wrapping_add(0x9e37_79b9_7f4a_7c15);
    crate::splitmix64(&mut x)
}

fn correct_mask<B: Backend>(
    data: &V5Data,
    indices: &[usize],
    width: usize,
    device: &B::Device,
) -> anyhow::Result<Tensor<B, 2, Bool>> {
    let mut mask = vec![false; indices.len() * width];
    for (row, &index) in indices.iter().enumerate() {
        let position = data.position(index);
        anyhow::ensure!(
            !position.correct.is_empty() && position.legal.len() <= width,
            "{}: malformed correct/legal set",
            position.id
        );
        for &column in &position.correct {
            anyhow::ensure!(
                (column as usize) < position.legal.len(),
                "{}: correct index is outside legal actions",
                position.id
            );
            mask[row * width + column as usize] = true;
        }
    }
    Ok(Tensor::from_data(
        TensorData::new(mask, [indices.len(), width]),
        device,
    ))
}

fn scalar<B: Backend>(value: Tensor<B, 1>) -> f64 {
    f64::from(value.into_data().to_vec::<f32>().expect("f32 scalar")[0])
}

pub fn baseline_fingerprint<B: AutodiffBackend>(
    model: &CounterfactualRelationalLoop<B>,
    device: &B::Device,
) -> anyhow::Result<String> {
    let roots = [
        recur64_core::GameState::from_fen("6k1/8/8/8/8/8/4Q3/3RK3 w - - 0 1")?,
        recur64_core::GameState::from_fen("3rk3/4q3/8/8/8/8/8/6K1 b - - 0 1")?,
        recur64_core::GameState::from_fen("4k3/P7/8/8/8/8/7r/4K3 w - - 0 1")?,
    ];
    let refs: Vec<&recur64_core::GameState> = roots.iter().collect();
    let inner = model.valid();
    let inputs = RootInputs::<B::InnerBackend>::from_roots(&refs, device)?;
    let output = inner.base_root(&inputs);
    let mut hash = Sha256::new();
    hash.update(b"recur64.v5.baseline_fingerprint.v1\0");
    for values in [
        output.context.into_data().to_vec::<f32>()?,
        output.hypotheses.into_data().to_vec::<f32>()?,
        output.z0.into_data().to_vec::<f32>()?,
    ] {
        for value in values {
            hash.update(value.to_bits().to_le_bytes());
        }
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn stage_a_indices(data: &V5Data, update: u64, seed: u64, count: usize) -> Vec<usize> {
    let mut sampler = CellSampler::new(&data.cells(&data.fit), seed);
    for _ in 0..(update as usize * count) {
        let _ = sampler.next_index();
    }
    (0..count).map(|_| data.fit[sampler.next_index()]).collect()
}

fn condition_indices(data: &V5Data, update: u64, seed: u64, condition: usize) -> Vec<(usize, u64)> {
    let condition_seed = mix(seed, condition as u64);
    let mut sampler = CellSampler::new(&data.cells(&data.fit), condition_seed);
    for _ in 0..(update * 2) {
        let _ = sampler.next_index();
    }
    (0..2)
        .map(|_| {
            let ordinal = sampler.examples_drawn();
            (data.fit[sampler.next_index()], ordinal)
        })
        .collect()
}

fn acquisition_seed(seed: u64, condition: &Condition) -> u64 {
    let schedule = match condition.schedule {
        Schedule::UniformFrontierV1 => 0x55AA_0001,
        Schedule::BaseRankedDepthV1 => 0x55AA_0002,
    };
    // R is deliberately absent: acquisition identity is independent of the
    // number of shared reader-loop applications.
    mix(seed, schedule ^ condition.q as u64)
}

impl<B: AutodiffBackend> Trainer<B> {
    pub fn new(recipe: Recipe, model: CounterfactualRelationalLoop<B>) -> anyhow::Result<Self> {
        recipe.validate()?;
        Ok(Self {
            model,
            optim: adamw::<B, CounterfactualRelationalLoop<B>>(),
            recipe,
            updates_done: 0,
            history: Vec::new(),
            resume_events: Vec::new(),
        })
    }

    pub fn step(&mut self, data: &V5Data, device: &B::Device) -> anyhow::Result<StepRecord> {
        use std::time::Instant;
        anyhow::ensure!(
            self.updates_done < self.recipe.updates,
            "the frozen stage is complete"
        );
        data.require_role(crate::native_data_v2::Role::Train)?;
        data.verify_custody()?;
        let started = Instant::now();
        let update = self.updates_done;
        let lr = lr_at(
            update,
            self.recipe.peak_lr,
            self.recipe.warmup,
            self.recipe.updates,
        );
        let mut accumulator = GradientsAccumulator::<CounterfactualRelationalLoop<B>>::new();
        let mut loss_total = 0.0;
        let mut graph_digests = Vec::new();
        let mut cell_exposure = std::collections::BTreeMap::new();
        let mut condition_cell_exposure = std::collections::BTreeMap::new();

        match self.recipe.stage {
            Stage::BaselineA => {
                let indices =
                    stage_a_indices(data, update, self.recipe.seed, self.recipe.effective_batch);
                for &index in &indices {
                    let p = data.position(index);
                    *cell_exposure
                        .entry(format!("{} M{}", p.family, p.mate_depth))
                        .or_insert(0) += 1;
                }
                for chunk in indices.chunks(self.recipe.physical_microbatch) {
                    let roots = data.roots(chunk)?;
                    for (&index, root) in chunk.iter().zip(&roots) {
                        data.validate_root_alignment(index, root)?;
                    }
                    let refs: Vec<&recur64_core::GameState> = roots.iter().collect();
                    let input = RootInputs::<B>::from_roots(&refs, device)?;
                    let output = self.model.base_root(&input);
                    let correct = correct_mask::<B>(data, chunk, input.cands.width, device)?;
                    let loss = correct_set_loss(output.z0, input.cands.mask.clone(), correct)
                        .mul_scalar(chunk.len() as f32 / self.recipe.effective_batch as f32);
                    loss_total += scalar(loss.clone());
                    accumulator.accumulate(
                        &self.model,
                        GradientsParams::from_grads(loss.backward(), &self.model),
                    );
                }
            }
            Stage::ReaderB => {
                let inner = self.model.valid();
                for (condition_index, condition) in self.recipe.conditions.iter().enumerate() {
                    let selected =
                        condition_indices(data, update, self.recipe.seed, condition_index);
                    let mut exposure = std::collections::BTreeMap::new();
                    for &(index, _) in &selected {
                        let p = data.position(index);
                        let cell = format!("{} M{}", p.family, p.mate_depth);
                        *cell_exposure.entry(cell.clone()).or_insert(0) += 1;
                        *exposure.entry(cell).or_insert(0) += 1;
                    }
                    condition_cell_exposure.insert(format!("{condition_index:02}"), exposure);
                    for selected_chunk in selected.chunks(self.recipe.physical_microbatch) {
                        let indices: Vec<usize> = selected_chunk.iter().map(|x| x.0).collect();
                        let roots = data.roots(&indices)?;
                        for (&index, root) in indices.iter().zip(&roots) {
                            data.validate_root_alignment(index, root)?;
                        }
                        let refs: Vec<&recur64_core::GameState> = roots.iter().collect();

                        // The complete baseline path runs on the graph-free backend.
                        let root_inner = RootInputs::<B::InnerBackend>::from_roots(&refs, device)?;
                        let base_inner = inner.base_root(&root_inner);
                        let width = root_inner.cands.width;
                        let z0: Vec<f32> = base_inner
                            .z0
                            .clone()
                            .into_data()
                            .to_vec()
                            .expect("f32 logits");
                        let mut graphs = Vec::with_capacity(selected_chunk.len());
                        for (row, (&index, &(_, ordinal))) in
                            indices.iter().zip(selected_chunk).enumerate()
                        {
                            let legal = data.position(index).legal.len();
                            let row_logits = &z0[row * width..row * width + legal];
                            let graph = acquire(
                                &roots[row],
                                EpisodeKey {
                                    position_id: data.position(index).id.clone(),
                                    schedule: condition.schedule,
                                    run_seed: acquisition_seed(self.recipe.seed, condition),
                                    occurrence_ordinal: ordinal,
                                },
                                condition.q,
                                (condition.schedule == Schedule::BaseRankedDepthV1)
                                    .then_some(row_logits),
                            )?;
                            graph_digests.push(graph.digest.clone());
                            graphs.push(graph);
                        }
                        let examples: Vec<(
                            &recur64_core::GameState,
                            &crate::graph::AcquiredGraph,
                        )> = roots.iter().zip(&graphs).collect();
                        let input = V5Inputs::<B>::from_examples(&examples, device)?;
                        let base = BaseOutput {
                            context: Tensor::from_inner(base_inner.context),
                            pooled: Tensor::from_inner(base_inner.pooled),
                            hypotheses: Tensor::from_inner(base_inner.hypotheses),
                            z0: Tensor::from_inner(base_inner.z0),
                        };
                        let output = self.model.paired_with_base(
                            &input,
                            base,
                            condition.r,
                            Treatment::Normal,
                        );
                        let correct = correct_mask::<B>(data, &indices, input.cands.width, device)?;
                        let loss =
                            correct_set_loss(output.logits, input.cands.mask.clone(), correct)
                                .mul_scalar(
                                    selected_chunk.len() as f32
                                        / self.recipe.effective_batch as f32,
                                );
                        loss_total += scalar(loss.clone());
                        accumulator.accumulate(
                            &self.model,
                            GradientsParams::from_grads(loss.backward(), &self.model),
                        );
                    }
                }
            }
        }
        anyhow::ensure!(loss_total.is_finite(), "non-finite loss at update {update}");
        self.model = self.optim.step(lr, self.model.clone(), accumulator.grads());
        let record = StepRecord {
            update,
            lr,
            loss: loss_total,
            wall_seconds: started.elapsed().as_secs_f64(),
            graph_manifest_digests: graph_digests,
            cell_exposure,
            condition_cell_exposure,
        };
        self.updates_done += 1;
        self.history.push(record.clone());
        Ok(record)
    }

    pub fn save(
        &self,
        run_dir: &Path,
        backend: &str,
        device: &B::Device,
    ) -> anyhow::Result<String> {
        self.recipe.validate()?;
        if self.recipe.stage == Stage::ReaderB {
            let actual = baseline_fingerprint(&self.model, device)?;
            anyhow::ensure!(
                self.recipe.baseline_fingerprint.as_deref() == Some(actual.as_str()),
                "Stage B changed the frozen baseline path"
            );
        }
        std::fs::create_dir_all(run_dir)?;
        let generation = run_dir
            .join("checkpoints")
            .join(format!("update-{:012}", self.updates_done));
        if generation.exists() {
            if generation.join("state.json").exists() {
                anyhow::bail!(
                    "{} already contains a completed checkpoint",
                    generation.display()
                );
            }
            let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
            let quarantine = run_dir.join("checkpoints").join(format!(
                "quarantine-update-{:012}-{stamp}",
                self.updates_done
            ));
            std::fs::rename(&generation, &quarantine)?;
        }
        std::fs::create_dir_all(&generation)?;
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        let model_path = generation.join("model");
        let optimizer_path = generation.join("optimizer");
        self.model
            .clone()
            .save_file(model_path.clone(), &recorder)?;
        recorder.record(self.optim.to_record(), optimizer_path.clone())?;
        let model_file = model_path.with_extension("mpk");
        let model_hash = hash_file(&model_file)?;
        let meta = CheckpointMeta {
            schema: CHECKPOINT_SCHEMA.into(),
            architecture: ARCHITECTURE.into(),
            stage: self.recipe.stage,
            recipe_digest: self.recipe.digest()?,
            config_digest: self.recipe.config_digest.clone(),
            update: self.updates_done,
            model_hash: model_hash.clone(),
            optimizer_hash: hash_file(&optimizer_path.with_extension("mpk"))?,
            init_model_hash: self.recipe.init_model_hash.clone(),
            backend: backend.into(),
            precision: "fp32".into(),
            factual_null_gradient_semantics: "both_streams_differentiated_v1".into(),
            recipe: self.recipe.clone(),
            history: self.history.clone(),
            resume_events: self.resume_events.clone(),
        };
        let state_tmp = generation.join("state.json.tmp");
        std::fs::write(&state_tmp, serde_json::to_vec_pretty(&meta)?)?;
        std::fs::rename(state_tmp, generation.join("state.json"))?;
        Ok(model_hash)
    }

    pub fn load_latest(run_dir: &Path, recipe: Recipe, device: &B::Device) -> anyhow::Result<Self> {
        recipe.validate()?;
        let generation = latest_generation(run_dir)?;
        let meta: CheckpointMeta =
            serde_json::from_slice(&std::fs::read(generation.join("state.json"))?)?;
        anyhow::ensure!(
            meta.schema == CHECKPOINT_SCHEMA
                && meta.architecture == ARCHITECTURE
                && meta.recipe_digest == recipe.digest()?
                && meta.recipe.digest()? == meta.recipe_digest
                && meta.config_digest == recipe.config_digest
                && meta.stage == recipe.stage
                && meta.update <= recipe.updates,
            "checkpoint identity, recipe, stage or update mismatch"
        );
        let model_path = generation.join("model");
        anyhow::ensure!(
            hash_file(&model_path.with_extension("mpk"))? == meta.model_hash,
            "checkpoint model content hash mismatch"
        );
        anyhow::ensure!(
            hash_file(&generation.join("optimizer.mpk"))? == meta.optimizer_hash,
            "checkpoint optimizer content hash mismatch"
        );
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        let template = CounterfactualRelationalLoop::<B>::new(recipe.config.clone(), device);
        let model = template.load_file(model_path, &recorder, device)?;
        let optim_record = recorder.load(generation.join("optimizer"), device)?;
        let optim = adamw::<B, CounterfactualRelationalLoop<B>>().load_record(optim_record);
        let mut resume_events = meta.resume_events;
        resume_events.push(format!("resumed_from_update_{}", meta.update));
        Ok(Self {
            model,
            optim,
            recipe,
            updates_done: meta.update,
            history: meta.history,
            resume_events,
        })
    }
}

pub fn load_finished_model<B: AutodiffBackend>(
    run_dir: &Path,
    expected_stage: Stage,
    device: &B::Device,
) -> anyhow::Result<(CounterfactualRelationalLoop<B>, CheckpointMeta)> {
    let generation = latest_generation(run_dir)?;
    let meta: CheckpointMeta =
        serde_json::from_slice(&std::fs::read(generation.join("state.json"))?)?;
    meta.recipe.validate()?;
    anyhow::ensure!(
        meta.schema == CHECKPOINT_SCHEMA
            && meta.architecture == ARCHITECTURE
            && meta.stage == expected_stage
            && meta.recipe.stage == expected_stage
            && meta.update == meta.recipe.updates
            && meta.recipe_digest == meta.recipe.digest()?
            && meta.config_digest == meta.recipe.config_digest,
        "{}: checkpoint is not a completed, internally consistent {}",
        run_dir.display(),
        expected_stage.label()
    );
    let model_path = generation.join("model");
    anyhow::ensure!(
        hash_file(&model_path.with_extension("mpk"))? == meta.model_hash,
        "checkpoint model content hash mismatch"
    );
    anyhow::ensure!(
        hash_file(&generation.join("optimizer.mpk"))? == meta.optimizer_hash,
        "checkpoint optimizer content hash mismatch"
    );
    let template = CounterfactualRelationalLoop::<B>::new(meta.recipe.config.clone(), device);
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let model = template.load_file(model_path, &recorder, device)?;
    Ok((model, meta))
}

/// Load one exact immutable checkpoint generation for fixed-update evaluation.
pub fn load_model_at<B: AutodiffBackend>(
    run_dir: &Path,
    expected_stage: Stage,
    update: u64,
    device: &B::Device,
) -> anyhow::Result<(CounterfactualRelationalLoop<B>, CheckpointMeta)> {
    let generation = run_dir
        .join("checkpoints")
        .join(format!("update-{update:012}"));
    let meta: CheckpointMeta =
        serde_json::from_slice(&std::fs::read(generation.join("state.json"))?)?;
    meta.recipe.validate()?;
    anyhow::ensure!(
        meta.schema == CHECKPOINT_SCHEMA
            && meta.architecture == ARCHITECTURE
            && meta.stage == expected_stage
            && meta.recipe.stage == expected_stage
            && meta.update == update
            && meta.recipe_digest == meta.recipe.digest()?
            && meta.config_digest == meta.recipe.config_digest,
        "{}: checkpoint is not the exact, internally consistent {} update {}",
        generation.display(),
        expected_stage.label(),
        update
    );
    let model_path = generation.join("model");
    anyhow::ensure!(
        hash_file(&model_path.with_extension("mpk"))? == meta.model_hash,
        "checkpoint model content hash mismatch"
    );
    anyhow::ensure!(
        hash_file(&generation.join("optimizer.mpk"))? == meta.optimizer_hash,
        "checkpoint optimizer content hash mismatch"
    );
    let template = CounterfactualRelationalLoop::<B>::new(meta.recipe.config.clone(), device);
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let model = template.load_file(model_path, &recorder, device)?;
    if expected_stage == Stage::ReaderB {
        let actual = baseline_fingerprint(&model, device)?;
        anyhow::ensure!(
            meta.recipe.baseline_fingerprint.as_deref() == Some(actual.as_str()),
            "Stage B checkpoint baseline fingerprint differs from its frozen Stage A identity"
        );
    }
    Ok((model, meta))
}

pub fn latest_generation(run_dir: &Path) -> anyhow::Result<PathBuf> {
    let checkpoints = run_dir.join("checkpoints");
    let mut valid = Vec::new();
    for entry in std::fs::read_dir(&checkpoints)
        .map_err(|e| anyhow::anyhow!("{}: no checkpoint generations ({e})", run_dir.display()))?
    {
        let path = entry?.path();
        let Some(name) = path.file_name().and_then(|x| x.to_str()) else {
            continue;
        };
        if name.starts_with("update-") && path.join("state.json").is_file() {
            valid.push(path);
        }
    }
    valid.sort();
    valid
        .pop()
        .ok_or_else(|| anyhow::anyhow!("{}: no complete checkpoint", run_dir.display()))
}

pub fn hash_file(path: &Path) -> anyhow::Result<String> {
    let mut hash = Sha256::new();
    hash.update(std::fs::read(path)?);
    Ok(format!("{:x}", hash.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{EpisodeKey, acquire};
    use burn::optim::{GradientsParams, Optimizer};

    type B = burn::backend::Autodiff<burn::backend::Flex>;

    #[test]
    fn recipe_v3_and_nine_cell_exposure_are_frozen_without_training() {
        let a = Recipe::stage_a("test-source".into(), 2).unwrap();
        assert_eq!(a.schema, "v5_stage_recipe_v3");
        assert_eq!(a.train_cell_count, 9);
        assert_eq!((a.updates, a.effective_batch, a.warmup), (1200, 64, 80));
        assert_eq!(a.peak_lr, 3e-4);
        assert!(a.conditions.is_empty());
        let b =
            Recipe::stage_b("test-source".into(), 2, "initial".into(), "baseline".into()).unwrap();
        assert_eq!(
            (b.updates, b.effective_batch, b.conditions.len()),
            (800, 36, 18)
        );
        assert_eq!(b.peak_lr, a.peak_lr);
        let mut old = a.clone();
        old.schema = "v5_stage_recipe_v1".into();
        assert!(old.validate().is_err());
        let cells: Vec<_> = crate::native_data_v2::FAMILIES
            .iter()
            .flat_map(|f| (1..=3).flat_map(move |d| std::iter::repeat_n((f.to_string(), d), 3000)))
            .collect();
        for condition in 0..18 {
            let mut sampler = CellSampler::new(&cells, mix(PILOT_SEED, condition));
            let mut counts = std::collections::BTreeMap::new();
            for _ in 0..1000 {
                *counts
                    .entry(cells[sampler.next_index()].clone())
                    .or_insert(0u64) += 1;
            }
            assert_eq!(counts.len(), 9);
            assert!(counts.values().max().unwrap() - counts.values().min().unwrap() <= 1);
        }
    }

    #[test]
    fn training_acquisition_seed_is_independent_of_r() {
        let a = Condition {
            schedule: Schedule::UniformFrontierV1,
            q: 8,
            r: 1,
        };
        let b = Condition { r: 4, ..a.clone() };
        assert_eq!(
            acquisition_seed(PILOT_SEED, &a),
            acquisition_seed(PILOT_SEED, &b)
        );
    }

    fn fixture_update(
        trainer: &mut Trainer<B>,
        device: &burn::backend::flex::FlexDevice,
    ) -> (Vec<f32>, String, f64) {
        // Test-only real-chess fixtures: use the production schedule and
        // cell sampler, never claim these are the missing scientific dataset.
        let fixtures = [
            "6k1/8/8/8/8/8/4Q3/3RK3 w - - 0 1",
            "3rk3/4q3/8/8/8/8/8/6K1 b - - 0 1",
        ];
        let mut sampler = CellSampler::new(
            &[("test-white".into(), 1), ("test-black".into(), 1)],
            trainer.recipe.seed,
        );
        for _ in 0..trainer.updates_done {
            let _ = sampler.next_index();
        }
        let ordinal = sampler.examples_drawn();
        let selected = sampler.next_index();
        let root = recur64_core::GameState::from_fen(fixtures[selected]).unwrap();
        let graph = acquire(
            &root,
            EpisodeKey {
                position_id: format!("resume-fixture-{selected}"),
                schedule: Schedule::UniformFrontierV1,
                run_seed: 5301,
                occurrence_ordinal: ordinal,
            },
            2,
            None,
        )
        .unwrap();
        let examples = [(&root, &graph)];
        let input = V5Inputs::<B>::from_examples(&examples, device).unwrap();
        let base = trainer.model.base_frozen(&examples, device).unwrap();
        let output = trainer
            .model
            .paired_with_base(&input, base, 2, Treatment::Normal);
        let objective = output.centered_delta.clone().slice([0..1, 0..1]).sum()
            - output.centered_delta.slice([0..1, 1..2]).sum();
        let gradients = GradientsParams::from_grads(objective.backward(), &trainer.model);
        let lr = lr_at(
            trainer.updates_done,
            trainer.recipe.peak_lr,
            trainer.recipe.warmup,
            trainer.recipe.updates,
        );
        trainer.model = trainer.optim.step(lr, trainer.model.clone(), gradients);
        trainer.updates_done += 1;
        let input = V5Inputs::<B>::from_examples(&examples, device).unwrap();
        let logits = trainer
            .model
            .paired(&input, 2, Treatment::Normal)
            .logits
            .into_data()
            .to_vec()
            .unwrap();
        (logits, graph.digest, lr)
    }

    #[test]
    fn full_optimizer_checkpoint_resumes_exactly_on_cpu() {
        let _guard = crate::CPU_TEST_RNG
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let device = Default::default();
        <B as Backend>::seed(&device, 5301);
        let model = CounterfactualRelationalLoop::<B>::new(V5Config::default(), &device);
        let recipe = Recipe::stage_b(
            "test-source".into(),
            2,
            "test-initial-model".into(),
            baseline_fingerprint(&model, &device).unwrap(),
        )
        .unwrap();
        let mut uninterrupted = Trainer::new(recipe.clone(), model).unwrap();
        let _ = fixture_update(&mut uninterrupted, &device);

        let dir = std::env::temp_dir().join(format!("recur64-v5-resume-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        uninterrupted.save(&dir, "cpu-test", &device).unwrap();
        let mut resumed = Trainer::<B>::load_latest(&dir, recipe.clone(), &device).unwrap();
        assert_eq!(resumed.updates_done, 1);
        assert_eq!(resumed.recipe.digest().unwrap(), recipe.digest().unwrap());

        let expected = fixture_update(&mut uninterrupted, &device);
        let actual = fixture_update(&mut resumed, &device);
        assert_eq!(actual, expected);
        assert_eq!(resumed.updates_done, uninterrupted.updates_done);
        assert_eq!(resumed.updates_done, 2);
        assert_eq!(
            uninterrupted.model.parameter_digest().unwrap(),
            resumed.model.parameter_digest().unwrap()
        );

        let mut incompatible = recipe;
        incompatible.source_sha = "different-source".into();
        assert!(Trainer::<B>::load_latest(&dir, incompatible, &device).is_err());
        let generation = latest_generation(&dir).unwrap();
        // A syntactically valid replacement optimizer is still a partial-state
        // inconsistency and must be refused before loading any model tensors.
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        recorder
            .record(
                adamw::<B, CounterfactualRelationalLoop<B>>().to_record(),
                generation.join("optimizer"),
            )
            .unwrap();
        let error = Trainer::<B>::load_latest(&dir, resumed.recipe.clone(), &device)
            .err()
            .expect("replaced optimizer must be refused");
        assert!(
            error
                .to_string()
                .contains("optimizer content hash mismatch")
        );
        let _ = std::fs::remove_dir_all(dir);
    }
}
