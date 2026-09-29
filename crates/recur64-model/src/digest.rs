//! Semantic weight identity (H3.5B / D50 amendment).
//!
//! `model_id` is the SHA-256 of the serialized `.mpk` file: an *artifact*
//! identity. Burn records a generated `ParamId` (a random u64 from OS entropy,
//! not from `Backend::seed`) next to every tensor, so two models with identical
//! values can have different `model_id`s. The digest here hashes only what the
//! network computes with: the model config, the head version, and every float
//! parameter's name, shape and FP32 values in an explicit, stable order.
//!
//! It is a diagnostic alongside `model_id`, not a replacement for it.

use burn::prelude::*;
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::model::{HEAD_VERSION, NamedParam, ProbeModel};

/// Version tag of the digest encoding. Any change to the encoding below must
/// bump it.
pub const SEMANTIC_DIGEST_VERSION: &str = "recur64-semantic-weights-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SemanticDigest {
    pub version: &'static str,
    pub digest: String,
    pub head_version: u32,
    pub tensor_count: usize,
    pub element_count: usize,
}

fn put_u64(h: &mut Sha256, v: u64) {
    h.update(v.to_le_bytes());
}

fn put_str(h: &mut Sha256, s: &str) {
    put_u64(h, s.len() as u64);
    h.update(s.as_bytes());
}

/// Digest of already-extracted parameters (exposed for tests).
pub fn digest_named_params(config_json: &str, params: &[NamedParam]) -> SemanticDigest {
    digest_named_params_with(SEMANTIC_DIGEST_VERSION, HEAD_VERSION, config_json, params)
}

/// Digest of already-extracted parameters under an explicit encoding version
/// and head version.
pub fn digest_named_params_with(
    version: &'static str,
    head_version: u32,
    config_json: &str,
    params: &[NamedParam],
) -> SemanticDigest {
    let mut h = Sha256::new();
    put_str(&mut h, version);
    put_u64(&mut h, head_version as u64);
    put_str(&mut h, config_json);
    put_u64(&mut h, params.len() as u64);
    let mut tensors = 0;
    let mut elements = 0;
    for p in params {
        put_str(&mut h, &p.name);
        match &p.values {
            None => h.update([0u8]),
            Some(values) => {
                h.update([1u8]);
                put_str(&mut h, "f32");
                put_u64(&mut h, p.dims.len() as u64);
                for &d in &p.dims {
                    put_u64(&mut h, d as u64);
                }
                put_u64(&mut h, values.len() as u64);
                for v in values {
                    h.update(v.to_le_bytes());
                }
                tensors += 1;
                elements += values.len();
            }
        }
    }
    SemanticDigest {
        version,
        digest: format!("{:x}", h.finalize()),
        head_version,
        tensor_count: tensors,
        element_count: elements,
    }
}

/// Semantic weight digest of a model (excludes `ParamId`s and recorder bytes).
pub fn semantic_weights_digest<B: Backend>(
    model: &ProbeModel<B>,
) -> anyhow::Result<SemanticDigest> {
    let config_json = serde_json::to_string(model.config())?;
    Ok(digest_named_params(
        &config_json,
        &model.named_float_params()?,
    ))
}

/// Version tag of the Chimera semantic digest encoding.
pub const CHIMERA_SEMANTIC_DIGEST_VERSION: &str = "recur64-semantic-weights-chimera-v1";

/// Semantic weight digest of an X15 / Chimera model. The hashed config is the
/// pair `(trunk geometry, experimental contract)`, and the head version is the
/// Chimera one, so a Chimera digest can never be confused with a probe digest.
pub fn semantic_weights_digest_chimera<B: Backend>(
    model: &crate::chimera::ChimeraModel<B>,
) -> anyhow::Result<SemanticDigest> {
    let config_json =
        serde_json::to_string(&(model.config().clone(), model.experimental().clone()))?;
    Ok(digest_named_params_with(
        CHIMERA_SEMANTIC_DIGEST_VERSION,
        crate::experimental::CHIMERA_HEAD_VERSION,
        &config_json,
        &model.named_float_params()?,
    ))
}

/// Per-tensor value comparison of two models with the same layout.
#[derive(Debug, Clone, Serialize)]
pub struct TensorDelta {
    pub name: String,
    pub elements: usize,
    pub differing: usize,
    pub max_abs_delta: f64,
    pub mean_abs_delta: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct WeightComparison {
    pub tensors: usize,
    pub tensors_differing: usize,
    pub elements: usize,
    pub elements_differing: usize,
    pub fraction_differing: f64,
    pub max_abs_delta: f64,
    pub mean_abs_delta: f64,
    /// Only tensors with at least one differing element.
    pub differing_tensors: Vec<TensorDelta>,
}

/// Compare two models element by element. Fails if names or shapes differ.
pub fn compare_weights<B: Backend>(
    a: &ProbeModel<B>,
    b: &ProbeModel<B>,
) -> anyhow::Result<WeightComparison> {
    let (pa, pb) = (a.named_float_params()?, b.named_float_params()?);
    anyhow::ensure!(pa.len() == pb.len(), "parameter lists differ in length");
    let (mut elements, mut differing, mut sum, mut max) = (0usize, 0usize, 0f64, 0f64);
    let mut out = Vec::new();
    for (x, y) in pa.iter().zip(&pb) {
        anyhow::ensure!(
            x.name == y.name && x.dims == y.dims,
            "layout mismatch at {} {:?} vs {} {:?}",
            x.name,
            x.dims,
            y.name,
            y.dims
        );
        let (Some(xv), Some(yv)) = (&x.values, &y.values) else {
            anyhow::ensure!(
                x.values.is_none() && y.values.is_none(),
                "presence mismatch at {}",
                x.name
            );
            continue;
        };
        let (mut d, mut s, mut m) = (0usize, 0f64, 0f64);
        for (u, v) in xv.iter().zip(yv) {
            // Bitwise inequality, so -0.0 vs 0.0 or NaN payloads also count.
            if u.to_bits() != v.to_bits() {
                d += 1;
                let delta = (*u as f64 - *v as f64).abs();
                s += delta;
                m = m.max(delta);
            }
        }
        elements += xv.len();
        differing += d;
        sum += s;
        max = max.max(m);
        if d > 0 {
            out.push(TensorDelta {
                name: x.name.clone(),
                elements: xv.len(),
                differing: d,
                max_abs_delta: m,
                mean_abs_delta: s / xv.len().max(1) as f64,
            });
        }
    }
    Ok(WeightComparison {
        tensors: pa.iter().filter(|p| p.values.is_some()).count(),
        tensors_differing: out.len(),
        elements,
        elements_differing: differing,
        fraction_differing: differing as f64 / elements.max(1) as f64,
        max_abs_delta: max,
        mean_abs_delta: sum / elements.max(1) as f64,
        differing_tensors: out,
    })
}
