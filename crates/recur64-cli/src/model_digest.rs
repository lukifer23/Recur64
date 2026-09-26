//! `recur64 model-digest`: artifact identity vs semantic weight identity.
//!
//! `model_id` is the SHA-256 of `model.mpk` (artifact identity; it includes
//! Burn's generated ParamIds). `semantic_weights_digest` hashes only config,
//! head version and FP32 values (D50 amendment). Loads on the CPU backend, so
//! the result is device-independent. Read-only.

use std::path::{Path, PathBuf};

use clap::Args;

use recur64_model::checkpoint::{CheckpointMeta, hash_file};
use recur64_model::digest::{compare_weights, semantic_weights_digest};
use recur64_model::model::ProbeModel;
use recur64_runtime::model_io;

type Cpu = burn::backend::Flex;

#[derive(Args, Debug)]
pub struct ModelDigestArgs {
    /// Checkpoint directory (`model.mpk` + `meta.json`).
    #[arg(long)]
    pub checkpoint: PathBuf,
    /// Optional second checkpoint: report element-wise value differences.
    #[arg(long)]
    pub compare: Option<PathBuf>,
    /// Optional JSON output path.
    #[arg(long)]
    pub output: Option<PathBuf>,
}

fn load(dir: &Path) -> anyhow::Result<(CheckpointMeta, ProbeModel<Cpu>, serde_json::Value)> {
    let meta: CheckpointMeta = serde_json::from_slice(&std::fs::read(dir.join("meta.json"))?)?;
    let device = Default::default();
    let model = model_io::load::<Cpu>(dir, &meta.model, &device)?;
    let digest = semantic_weights_digest(&model)?;
    let artifact = hash_file(&dir.join("model.mpk"))?;
    let v = serde_json::json!({
        "checkpoint": dir.display().to_string(),
        "model_id_recorded": meta.model_id,
        "model_id_file_sha256": artifact,
        "semantic_weights_digest": digest.digest,
        "semantic_digest_version": digest.version,
        "head_version": digest.head_version,
        "tensor_count": digest.tensor_count,
        "param_count": digest.element_count,
        "model": meta.model,
    });
    Ok((meta, model, v))
}

pub fn run(args: ModelDigestArgs) -> anyhow::Result<()> {
    let (_, a, mut out) = load(&args.checkpoint)?;
    if let Some(other) = &args.compare {
        let (_, b, other_v) = load(other)?;
        let cmp = compare_weights(&a, &b)?;
        out = serde_json::json!({
            "a": out,
            "b": other_v,
            "artifact_ids_equal": out["model_id_file_sha256"] == other_v["model_id_file_sha256"],
            "semantic_digests_equal":
                out["semantic_weights_digest"] == other_v["semantic_weights_digest"],
            "comparison": cmp,
        });
    }
    let text = serde_json::to_string_pretty(&out)?;
    if let Some(path) = &args.output {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, &text)?;
    }
    println!("{text}");
    Ok(())
}
