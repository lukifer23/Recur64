//! Canonical initialization from the reserved `model_init` stream.
//!
//! Values are generated on the host from SplitMix64 (Box-Muller normals), written
//! once to a hash-pinned file, and loaded into every arm (and every audit model)
//! by traversal order. Burn's own RNG is never used for scientific weights.
//! Rules: Linear weights ~ N(0, 2/(fan_in+fan_out)) (Xavier-normal); embedding
//! tables ~ N(0, 0.02^2); Linear biases 0; LayerNorm gamma 1, beta 0.

use crate::inventory::ParamInfo;
use recur64_v69::streams::{MasterSeed, SplitMix64};
use serde::{Deserialize, Serialize};

fn normal(rng: &mut SplitMix64) -> f64 {
    // Box-Muller; u1 in (0,1], u2 in [0,1)
    let u1 = ((rng.next_u64() >> 11) as f64 + 1.0) / 9007199254740993.0;
    let u2 = (rng.next_u64() >> 11) as f64 / 9007199254740992.0;
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

pub fn generate(seed: &MasterSeed, label: &str, inv: &[ParamInfo]) -> Vec<(Vec<usize>, Vec<f32>)> {
    let mut rng = seed.stream(label, 0);
    inv.iter()
        .map(|p| {
            let vals: Vec<f32> = match (p.leaf.as_str(), p.path.rsplit('.').next().unwrap_or("")) {
                ("bias", _) | ("beta", _) => vec![0.0; p.numel],
                ("gamma", _) => vec![1.0; p.numel],
                ("weight", _) if p.shape.len() == 2 && is_embedding(p) => (0..p.numel).map(|_| (0.02 * normal(&mut rng)) as f32).collect(),
                ("weight", _) => {
                    let std = (2.0 / (p.shape[0] + p.shape[1]) as f64).sqrt();
                    (0..p.numel).map(|_| (std * normal(&mut rng)) as f32).collect()
                }
                other => panic!("unhandled parameter kind {other:?}"),
            };
            (p.shape.clone(), vals)
        })
        .collect()
}

fn is_embedding(p: &ParamInfo) -> bool {
    matches!(p.path.as_str(), "piece_emb" | "square_emb" | "budget_emb" | "att_emb" | "slot_emb")
}

#[derive(Serialize, Deserialize, Debug)]
pub struct InitHeader {
    pub version: u32,
    pub stream_label: String,
    pub seed_fingerprint: String,
    pub tensors: Vec<InitTensor>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct InitTensor {
    pub path: String,
    pub leaf: String,
    pub shape: Vec<usize>,
}

/// Serialize: one JSON header line, then raw little-endian f32 in traversal order.
pub fn to_bytes(header: &InitHeader, values: &[(Vec<usize>, Vec<f32>)]) -> Vec<u8> {
    let mut out = serde_json::to_vec(header).unwrap();
    out.push(b'\n');
    for (_, v) in values {
        for x in v {
            out.extend_from_slice(&x.to_le_bytes());
        }
    }
    out
}

pub fn from_bytes(bytes: &[u8]) -> anyhow::Result<(InitHeader, Vec<(Vec<usize>, Vec<f32>)>)> {
    let nl = bytes.iter().position(|&b| b == b'\n').ok_or_else(|| anyhow::anyhow!("init file has no header"))?;
    let header: InitHeader = serde_json::from_slice(&bytes[..nl])?;
    let mut off = nl + 1;
    let mut vals = Vec::new();
    for t in &header.tensors {
        let n: usize = t.shape.iter().product();
        anyhow::ensure!(off + 4 * n <= bytes.len(), "init file truncated");
        let v: Vec<f32> = bytes[off..off + 4 * n].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        off += 4 * n;
        vals.push((t.shape.clone(), v));
    }
    anyhow::ensure!(off == bytes.len(), "init file has trailing bytes");
    Ok((header, vals))
}

/// SHA-256 over the exact tensor bytes (shape-prefixed), independent of file framing.
pub fn tensors_hash(values: &[(Vec<usize>, Vec<f32>)]) -> String {
    let mut buf = Vec::new();
    for (s, v) in values {
        for d in s {
            buf.extend_from_slice(&(*d as u64).to_le_bytes());
        }
        for x in v {
            buf.extend_from_slice(&x.to_le_bytes());
        }
    }
    recur64_v69::provenance::sha256_hex(&buf)
}
