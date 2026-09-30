//! Evaluation files and paired comparison between two sets of evaluations.
//!
//! Seed-level pairing is by model-seed IDENTITY, never by file order: both sides
//! must carry unique model seeds and the same seed set, otherwise the comparison
//! is refused.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use super::train::{EvalSummary, PosResult, paired_bootstrap};

/// One saved evaluation of one model on one split.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalFile {
    pub split: String,
    pub dataset_digest: String,
    pub model: String,
    pub model_id: String,
    /// Authoritative model-initialization seed (from checkpoint metadata for
    /// saved models, the requested seed for fresh ones). Files written before
    /// this field existed read as `None` and cannot be seed-paired.
    #[serde(default)]
    pub model_seed: Option<u64>,
    #[serde(default)]
    pub architecture: Option<String>,
    #[serde(default)]
    pub training_updates: Option<usize>,
    #[serde(default)]
    pub peak_lr: Option<f64>,
    #[serde(default)]
    pub sampler_version: Option<String>,
    pub summary: EvalSummary,
    pub results: Vec<PosResult>,
}

/// Pair the files of A and B by model seed. Errors (refuses) on a missing or
/// duplicate seed, or when the seed sets differ. Output is sorted by seed, so it
/// does not depend on the order of the inputs.
pub fn align_by_seed<'a>(
    a: &'a [EvalFile],
    b: &'a [EvalFile],
) -> Result<Vec<(u64, &'a EvalFile, &'a EvalFile)>, String> {
    fn index<'f>(side: &str, files: &'f [EvalFile]) -> Result<BTreeMap<u64, &'f EvalFile>, String> {
        let mut m = BTreeMap::new();
        for f in files {
            let seed = f
                .model_seed
                .ok_or_else(|| format!("side {side}: '{}' carries no model_seed", f.model))?;
            if m.insert(seed, f).is_some() {
                return Err(format!("side {side}: duplicate model seed {seed}"));
            }
        }
        Ok(m)
    }
    let (ia, ib) = (index("A", a)?, index("B", b)?);
    let (sa, sb): (BTreeSet<_>, BTreeSet<_>) = (ia.keys().collect(), ib.keys().collect());
    if sa != sb {
        return Err(format!(
            "model seed sets differ: A {:?} vs B {:?}",
            ia.keys().collect::<Vec<_>>(),
            ib.keys().collect::<Vec<_>>()
        ));
    }
    Ok(ia.iter().map(|(s, fa)| (*s, *fa, ib[s])).collect())
}

/// A per-position metric extractor.
type Metric = fn(&PosResult) -> f64;

/// Per-position value averaged over the files of one side, keyed by position id.
fn per_position(files: &[&EvalFile], f: Metric) -> BTreeMap<String, (u8, f64)> {
    let mut acc: BTreeMap<String, (u8, f64)> = BTreeMap::new();
    for file in files {
        for r in &file.results {
            let e = acc.entry(r.id.clone()).or_insert((r.depth, 0.0));
            e.1 += f(r) / files.len() as f64;
        }
    }
    acc
}

fn ci(values: Vec<f64>, resamples: usize, seed: u64) -> serde_json::Value {
    let (m, lo, hi) = paired_bootstrap(&values, resamples, seed);
    serde_json::json!({ "n": values.len(), "mean_diff_b_minus_a": m, "ci95": [lo, hi] })
}

/// Paired per-position comparison (B minus A) of top-1, correct-set mass and
/// negative CE, pooled over positions and by depth. With `per_seed`, also each
/// seed separately, paired by model-seed identity (refused if the seed sets do
/// not match).
pub fn compare(
    a: &[EvalFile],
    b: &[EvalFile],
    resamples: usize,
    seed: u64,
    per_seed: bool,
) -> Result<serde_json::Value, String> {
    let first = a.first().ok_or("side A is empty")?;
    if a.iter()
        .chain(b)
        .any(|f| f.dataset_digest != first.dataset_digest || f.split != first.split)
    {
        return Err("compared evaluations must be on the same split and dataset digest".into());
    }
    let pairs = if per_seed {
        Some(align_by_seed(a, b)?)
    } else {
        None
    };
    let (ra, rb): (Vec<&EvalFile>, Vec<&EvalFile>) = (a.iter().collect(), b.iter().collect());
    let metrics: [(&str, Metric); 3] = [
        ("top1", |r| f64::from(u8::from(r.top1))),
        ("mass", |r| r.mass as f64),
        ("neg_ce", |r| -(r.ce as f64)),
    ];
    let mut out = serde_json::Map::new();
    for (name, f) in metrics {
        let (pa, pb) = (per_position(&ra, f), per_position(&rb, f));
        if pa.keys().ne(pb.keys()) {
            return Err("position sets differ between A and B".into());
        }
        let diff = |sel: &dyn Fn(u8) -> bool| -> Vec<f64> {
            pa.iter()
                .filter(|(_, v)| sel(v.0))
                .map(|(k, v)| pb[k].1 - v.1)
                .collect()
        };
        let mut groups = serde_json::Map::new();
        groups.insert("all".into(), ci(diff(&|_| true), resamples, seed));
        for d in 1..=5u8 {
            let v = diff(&|x| x == d);
            if !v.is_empty() {
                groups.insert(format!("M{d}"), ci(v, resamples, seed));
            }
        }
        groups.insert(
            "M2+M3".into(),
            ci(diff(&|x| x == 2 || x == 3), resamples, seed),
        );
        if let Some(pairs) = &pairs {
            let mut seeds = serde_json::Map::new();
            for (s, fa, fb) in pairs {
                let (sa, sb) = (per_position(&[fa], f), per_position(&[fb], f));
                let d: Vec<f64> = sa.iter().map(|(k, v)| sb[k].1 - v.1).collect();
                seeds.insert(format!("seed_{s}"), ci(d, resamples, seed));
            }
            groups.insert("per_seed_all".into(), serde_json::Value::Object(seeds));
        }
        out.insert(name.into(), serde_json::Value::Object(groups));
    }
    Ok(serde_json::Value::Object(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(model: &str, seed: Option<u64>, shift: f32) -> EvalFile {
        let results: Vec<PosResult> = (0..40)
            .map(|i| PosResult {
                id: format!("p{i}"),
                family: "KQvK".into(),
                depth: 1 + (i % 3) as u8,
                top1: (i as f32 / 40.0) < 0.4 + shift,
                mass: 0.3 + shift,
                ce: 2.0 - shift,
                entropy: 1.0,
                chance: 0.1,
            })
            .collect();
        EvalFile {
            split: "tune".into(),
            dataset_digest: "d".into(),
            model: model.into(),
            model_id: "m".into(),
            model_seed: seed,
            architecture: None,
            training_updates: None,
            peak_lr: None,
            sampler_version: None,
            summary: EvalSummary::default(),
            results,
        }
    }

    #[test]
    fn matching_seed_sets_pass_and_file_order_is_irrelevant() {
        let a = vec![file("a1", Some(1), 0.0), file("a2", Some(2), 0.05)];
        let b = vec![file("b1", Some(1), 0.2), file("b2", Some(2), 0.25)];
        let b_rev = vec![b[1].clone(), b[0].clone()];
        let a_rev = vec![a[1].clone(), a[0].clone()];
        let x = compare(&a, &b, 500, 3, true).unwrap();
        let y = compare(&a_rev, &b_rev, 500, 3, true).unwrap();
        let z = compare(&a, &b_rev, 500, 3, true).unwrap();
        assert_eq!(x["top1"]["per_seed_all"], y["top1"]["per_seed_all"]);
        assert_eq!(x["top1"]["per_seed_all"], z["top1"]["per_seed_all"]);
        assert!(
            x["top1"]["per_seed_all"]["seed_1"]["mean_diff_b_minus_a"]
                .as_f64()
                .unwrap()
                > 0.0
        );
    }

    #[test]
    fn mismatched_duplicate_or_missing_seeds_are_refused() {
        let a = vec![file("a1", Some(1), 0.0), file("a2", Some(2), 0.0)];
        let b_other = vec![file("b1", Some(1), 0.1), file("b3", Some(3), 0.1)];
        assert!(
            compare(&a, &b_other, 100, 1, true)
                .unwrap_err()
                .contains("seed sets differ")
        );
        let dup = vec![file("a1", Some(1), 0.0), file("a1b", Some(1), 0.0)];
        assert!(
            compare(&dup, &dup, 100, 1, true)
                .unwrap_err()
                .contains("duplicate model seed")
        );
        let missing = vec![file("old", None, 0.0)];
        assert!(
            compare(&missing, &missing, 100, 1, true)
                .unwrap_err()
                .contains("no model_seed")
        );
        // Pooled (non-seed-level) comparison does not require seed identity.
        assert!(compare(&missing, &missing, 100, 1, false).is_ok());
    }

    #[test]
    fn different_datasets_are_refused() {
        let a = vec![file("a", Some(1), 0.0)];
        let mut b = vec![file("b", Some(1), 0.0)];
        b[0].dataset_digest = "other".into();
        assert!(compare(&a, &b, 100, 1, true).is_err());
    }
}
