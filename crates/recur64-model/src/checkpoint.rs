//! Training checkpoints for the Phase 0 probe.
//!
//! A *training checkpoint* preserves enough state to genuinely resume: model
//! parameters, optimizer moments/step, recurrence configuration, RNG state, and
//! metadata. A weights-only file is an *inference export*, not a resumable
//! checkpoint, and must not be presented as one.

use std::path::{Path, PathBuf};

use burn::optim::Optimizer;
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
use burn::tensor::backend::AutodiffBackend;

use crate::config::{
    ACTIVE_HEAD_VERSION, ALL_INFO_HEAD_VERSION, ActiveContracts, AllInfoContracts, Architecture,
    CANDIDATE_BLOCK_CONTRACT, CANDIDATE_FACTS_VERSION, CANDIDATE_HEAD_VERSION,
    CANDIDATE_TOKEN_CONTRACT, FACT_DELTA_CONTRACT, ModelConfig,
};
use crate::net::NeuralModel;

/// Current checkpoint schema version. Bump on any breaking change.
///
/// * v1 — Phase 0 probe checkpoints (no chess contract metadata).
/// * v2 — Phase 2: records observation/action/rules/replay contract versions,
///   model identity, and run identity.
pub const SCHEMA_VERSION: u32 = 2;

/// Metadata stored alongside a training checkpoint.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CheckpointMeta {
    pub schema_version: u32,
    pub recur64_version: String,
    pub git_revision: Option<String>,
    pub backend: String,
    pub precision: String,
    pub model: ModelConfig,
    pub recurrence: usize,
    pub deep_supervision: bool,
    pub step: u64,
    pub lr: f64,
    pub seed: u64,
    /// Sampler / fixture RNG state needed to resume the exact data sequence.
    pub rng_state: u64,
    // --- Phase 2 chess contract + identity metadata (defaulted for reading
    //     older metadata; a schema mismatch is refused before use). ---
    #[serde(default)]
    pub observation_version: u32,
    #[serde(default)]
    pub action_version: u32,
    #[serde(default)]
    pub rules_profile_version: u32,
    #[serde(default)]
    pub replay_schema_version: u32,
    /// Stable content-derived identity of the saved model.
    #[serde(default)]
    pub model_id: String,
    #[serde(default)]
    pub run_id: String,
    /// Number of optimizer updates applied.
    #[serde(default)]
    pub update_counter: u64,
    /// Position of the LR schedule.
    #[serde(default)]
    pub lr_schedule_step: u64,
    /// Readout-head function version ([`crate::model::HEAD_VERSION`]).
    /// Metadata written before the field existed is head v1.
    #[serde(default = "legacy_head_version")]
    pub head_version: u32,
    /// Architecture id (`probe_v1` for metadata written before the field existed).
    #[serde(default = "legacy_architecture")]
    pub architecture: String,
    /// `CandidateFactsV1` layout version the network consumes (0 = none).
    #[serde(default)]
    pub candidate_facts_version: u32,
    /// Candidate-token construction contract (0 = none).
    #[serde(default)]
    pub candidate_token_contract: u32,
    /// Candidate-block contract (0 = none).
    #[serde(default)]
    pub candidate_block_contract: u32,
    /// LF fact-delta contract (0 = none).
    #[serde(default)]
    pub fact_delta_contract: u32,
    /// V3 scientific contracts (present iff `active_search_v3`). A checkpoint
    /// written under one set of contracts refuses another.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_contracts: Option<ActiveContracts>,
    /// P6 ALL-INFO scientific contracts (present iff `all_info_v1`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub all_info_contracts: Option<AllInfoContracts>,
}

fn legacy_architecture() -> String {
    Architecture::ProbeV1.id().to_string()
}

fn legacy_head_version() -> u32 {
    1
}

impl CheckpointMeta {
    /// Build metadata with the current contract versions and empty identity.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: ModelConfig,
        recurrence: usize,
        deep_supervision: bool,
        step: u64,
        lr: f64,
        seed: u64,
        rng_state: u64,
        backend: impl Into<String>,
        precision: impl Into<String>,
    ) -> Self {
        let v = recur64_core::ContractVersions::V1;
        let cand = model.architecture == Architecture::CandidateV25;
        let lf = model.architecture == Architecture::LegacyFactsV25;
        let act = model.architecture == Architecture::ActiveSearchV3;
        let ai = model.architecture == Architecture::AllInfoV1;
        Self {
            active_contracts: act.then(ActiveContracts::default),
            all_info_contracts: ai.then(AllInfoContracts::default),
            architecture: model.architecture.id().to_string(),
            fact_delta_contract: if lf { FACT_DELTA_CONTRACT } else { 0 },
            candidate_facts_version: if cand || lf || act || ai {
                CANDIDATE_FACTS_VERSION
            } else {
                0
            },
            candidate_token_contract: if cand { CANDIDATE_TOKEN_CONTRACT } else { 0 },
            candidate_block_contract: if cand { CANDIDATE_BLOCK_CONTRACT } else { 0 },
            schema_version: SCHEMA_VERSION,
            recur64_version: crate::VERSION.to_string(),
            git_revision: None,
            backend: backend.into(),
            precision: precision.into(),
            model,
            recurrence,
            deep_supervision,
            step,
            lr,
            seed,
            rng_state,
            observation_version: v.observation,
            action_version: v.action,
            rules_profile_version: v.rules_profile,
            replay_schema_version: 1,
            model_id: String::new(),
            run_id: String::new(),
            update_counter: step,
            lr_schedule_step: step,
            head_version: if act {
                ACTIVE_HEAD_VERSION
            } else if ai {
                ALL_INFO_HEAD_VERSION
            } else if cand {
                CANDIDATE_HEAD_VERSION
            } else {
                crate::model::HEAD_VERSION
            },
        }
    }

    /// Refuse a checkpoint whose recorded model configuration differs from
    /// the configuration it is being loaded into. Tensor shapes alone do not
    /// prove compatibility: same-shape settings such as `rms_eps` change the
    /// function. Recurrence is deliberately NOT checked: it is a runtime
    /// choice, and the same R10 weights are evaluated at R1/R2/R4.
    pub fn check_model(&self, requested: &ModelConfig) -> anyhow::Result<()> {
        // Explicit architecture refusal (never left to a tensor-shape mismatch).
        anyhow::ensure!(
            self.architecture == requested.architecture.id()
                && self.model.architecture == requested.architecture,
            "checkpoint architecture '{}' is not the requested architecture '{}': cross-architecture loads are refused",
            self.architecture,
            requested.architecture.id()
        );
        let recorded = serde_json::to_value(&self.model)?;
        let wanted = serde_json::to_value(requested)?;
        anyhow::ensure!(
            recorded == wanted,
            "checkpoint model config {recorded} differs from the requested model config {wanted}"
        );
        Ok(())
    }

    /// Verify the model-function contracts: head version and the chess
    /// observation / action / rules versions. `replay_schema_version` is
    /// training-data provenance, not part of the network's function, so it
    /// is recorded but does not block loading.
    pub fn check_contracts(&self) -> anyhow::Result<()> {
        if self.architecture == Architecture::CandidateV25.id() {
            anyhow::ensure!(
                self.head_version == CANDIDATE_HEAD_VERSION
                    && self.candidate_facts_version == CANDIDATE_FACTS_VERSION
                    && self.candidate_token_contract == CANDIDATE_TOKEN_CONTRACT
                    && self.candidate_block_contract == CANDIDATE_BLOCK_CONTRACT,
                "candidate_v25 checkpoint contracts (head {}, facts {}, token {}, block {}) differ from the current (head {CANDIDATE_HEAD_VERSION}, facts {CANDIDATE_FACTS_VERSION}, token {CANDIDATE_TOKEN_CONTRACT}, block {CANDIDATE_BLOCK_CONTRACT})",
                self.head_version,
                self.candidate_facts_version,
                self.candidate_token_contract,
                self.candidate_block_contract
            );
        } else if self.architecture == Architecture::ActiveSearchV3.id() {
            anyhow::ensure!(
                self.head_version == ACTIVE_HEAD_VERSION
                    && self.candidate_facts_version == CANDIDATE_FACTS_VERSION,
                "active_search_v3 checkpoint head {} / facts {} differ from the current (head {ACTIVE_HEAD_VERSION}, facts {CANDIDATE_FACTS_VERSION})",
                self.head_version,
                self.candidate_facts_version
            );
            anyhow::ensure!(
                self.active_contracts.as_ref() == Some(&ActiveContracts::default()),
                "active_search_v3 checkpoint contracts {:?} differ from the current contracts {:?}",
                self.active_contracts,
                ActiveContracts::default()
            );
        } else if self.architecture == Architecture::AllInfoV1.id() {
            anyhow::ensure!(
                self.head_version == ALL_INFO_HEAD_VERSION
                    && self.candidate_facts_version == CANDIDATE_FACTS_VERSION,
                "all_info_v1 checkpoint head {} / facts {} differ from the current (head {ALL_INFO_HEAD_VERSION}, facts {CANDIDATE_FACTS_VERSION})",
                self.head_version,
                self.candidate_facts_version
            );
            anyhow::ensure!(
                self.all_info_contracts.as_ref() == Some(&AllInfoContracts::default()),
                "all_info_v1 checkpoint contracts {:?} differ from the current contracts {:?}",
                self.all_info_contracts,
                AllInfoContracts::default()
            );
        } else if self.architecture == Architecture::LegacyFactsV25.id() {
            anyhow::ensure!(
                self.head_version == crate::model::HEAD_VERSION
                    && self.candidate_facts_version == CANDIDATE_FACTS_VERSION
                    && self.fact_delta_contract == FACT_DELTA_CONTRACT,
                "legacy_facts_v25 checkpoint contracts (head {}, facts {}, fact-delta {}) differ from the current (head {}, facts {CANDIDATE_FACTS_VERSION}, fact-delta {FACT_DELTA_CONTRACT})",
                self.head_version,
                self.candidate_facts_version,
                self.fact_delta_contract,
                crate::model::HEAD_VERSION
            );
        } else {
            anyhow::ensure!(
                self.architecture == Architecture::ProbeV1.id(),
                "unknown checkpoint architecture '{}'",
                self.architecture
            );
            anyhow::ensure!(
                self.head_version == crate::model::HEAD_VERSION,
                "checkpoint head version {} is not the current head version {}: its weights were trained for a different readout function and are refused",
                self.head_version,
                crate::model::HEAD_VERSION
            );
        }
        let v = recur64_core::ContractVersions::V1;
        anyhow::ensure!(
            self.observation_version == v.observation
                && self.action_version == v.action
                && self.rules_profile_version == v.rules_profile,
            "checkpoint contract mismatch: obs {} action {} rules {} (expected {}/{}/{})",
            self.observation_version,
            self.action_version,
            self.rules_profile_version,
            v.observation,
            v.action,
            v.rules_profile
        );
        Ok(())
    }
}

fn paths(dir: &Path) -> (PathBuf, PathBuf, PathBuf) {
    (
        dir.join("model"),
        dir.join("optimizer"),
        dir.join("meta.json"),
    )
}

/// Content hash of a file (used for stable model identity).
pub fn hash_file(path: &Path) -> anyhow::Result<String> {
    use sha2::{Digest, Sha256};
    let bytes = std::fs::read(path)?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

/// Save a full training checkpoint.
pub fn save_training<B, M, O>(
    dir: &Path,
    model: &M,
    optim: &O,
    meta: &CheckpointMeta,
) -> anyhow::Result<()>
where
    B: AutodiffBackend,
    M: NeuralModel<B> + burn::module::AutodiffModule<B>,
    O: Optimizer<M, B>,
{
    anyhow::ensure!(
        meta.architecture == M::ARCHITECTURE.id() && meta.model.architecture == M::ARCHITECTURE,
        "refusing to save: checkpoint metadata says architecture '{}' but the model is '{}'",
        meta.architecture,
        M::ARCHITECTURE.id()
    );
    std::fs::create_dir_all(dir)?;
    let (model_path, optim_path, meta_path) = paths(dir);
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    model
        .clone()
        .save_file(model_path.clone(), &recorder)
        .map_err(|e| anyhow::anyhow!("save model to {}: {e}", model_path.display()))?;
    recorder
        .record(optim.to_record(), optim_path.clone())
        .map_err(|e| anyhow::anyhow!("save optimizer to {}: {e}", optim_path.display()))?;
    // Model identity is the content hash of the saved weights. The MPK recorder
    // writes `<path>.mpk`; fall back to the bare path if that convention changes.
    let mut meta = meta.clone();
    let with_mpk = model_path.with_extension("mpk");
    let model_file = if with_mpk.exists() {
        with_mpk
    } else {
        model_path.clone()
    };
    meta.model_id = hash_file(&model_file)
        .map_err(|e| anyhow::anyhow!("hash {}: {e}", model_file.display()))?;
    std::fs::write(&meta_path, serde_json::to_vec_pretty(&meta)?)
        .map_err(|e| anyhow::anyhow!("write {}: {e}", meta_path.display()))?;
    Ok(())
}

/// Load a full training checkpoint into a freshly built model and optimizer.
///
/// Refuses to load a checkpoint whose schema version does not match.
pub fn load_training<B, M, O>(
    dir: &Path,
    template: M,
    optim: O,
    device: &B::Device,
) -> anyhow::Result<(M, O, CheckpointMeta)>
where
    B: AutodiffBackend,
    M: NeuralModel<B> + burn::module::AutodiffModule<B>,
    O: Optimizer<M, B>,
{
    let (model_path, optim_path, meta_path) = paths(dir);
    let meta: CheckpointMeta = serde_json::from_slice(&std::fs::read(&meta_path)?)?;
    anyhow::ensure!(
        meta.schema_version == SCHEMA_VERSION,
        "checkpoint schema mismatch: found {} expected {}",
        meta.schema_version,
        SCHEMA_VERSION
    );
    meta.check_contracts()?;
    meta.check_model(template.model_config())?;
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let model = template.load_file(model_path, &recorder, device)?;
    let optim_record = recorder.load(optim_path, device)?;
    let optim = optim.load_record(optim_record);
    Ok((model, optim, meta))
}

/// Save a weights-only inference export (NOT a resumable checkpoint).
pub fn save_inference_export<B: Backend, M: NeuralModel<B>>(
    dir: &Path,
    model: &M,
) -> anyhow::Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let path = dir.join("weights");
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    model.clone().save_file(path.clone(), &recorder)?;
    Ok(path)
}

#[cfg(test)]
mod architecture_identity_tests {
    use super::*;

    fn probe_cfg() -> ModelConfig {
        serde_json::from_str(
            r#"{"width":384,"heads":12,"ffn":768,"input_blocks":0,"core_blocks":8,"output_blocks":0}"#,
        )
        .unwrap()
    }

    fn meta(model: ModelConfig) -> CheckpointMeta {
        CheckpointMeta::new(model, 1, false, 0, 0.0, 1, 0, "flex", "fp32")
    }

    #[test]
    fn metadata_written_before_the_architecture_fields_loads_as_probe() {
        let mut v = serde_json::to_value(meta(probe_cfg())).unwrap();
        let o = v.as_object_mut().unwrap();
        for k in [
            "architecture",
            "candidate_facts_version",
            "candidate_token_contract",
            "candidate_block_contract",
        ] {
            o.remove(k);
        }
        let m: CheckpointMeta = serde_json::from_value(v).unwrap();
        assert_eq!(m.architecture, "probe_v1");
        m.check_contracts().unwrap();
        m.check_model(&probe_cfg()).unwrap();
    }

    #[test]
    fn probe_and_candidate_checkpoints_refuse_each_other_explicitly() {
        let probe = meta(probe_cfg());
        let cand = meta(ModelConfig::candidate_v25(true));
        cand.check_contracts().unwrap();
        cand.check_model(&ModelConfig::candidate_v25(true)).unwrap();
        let e = probe
            .check_model(&ModelConfig::candidate_v25(true))
            .unwrap_err();
        assert!(e.to_string().contains("cross-architecture"), "{e}");
        let e = cand.check_model(&probe_cfg()).unwrap_err();
        assert!(e.to_string().contains("cross-architecture"), "{e}");
        // C0 weights are not CF weights.
        assert!(
            cand.check_model(&ModelConfig::candidate_v25(false))
                .is_err()
        );
    }

    #[test]
    fn lf_contract_one_checkpoints_are_refused_under_contract_two() {
        let mut lf = ModelConfig::legacy_facts_v25();
        lf.width = 32;
        lf.heads = 4;
        lf.ffn = 64;
        lf.core_blocks = 2;
        let mut m = meta(lf);
        assert_eq!(m.fact_delta_contract, FACT_DELTA_CONTRACT);
        assert_eq!(FACT_DELTA_CONTRACT, 2);
        m.check_contracts().unwrap();
        m.fact_delta_contract = 1; // an engineering-only pre-fix checkpoint
        let e = m.check_contracts().unwrap_err().to_string();
        assert!(e.contains("fact-delta"), "{e}");
        m.fact_delta_contract = 0; // or none at all
        assert!(m.check_contracts().is_err());
    }
}
