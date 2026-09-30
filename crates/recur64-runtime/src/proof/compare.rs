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

fn refs(v: &[EvalFile]) -> Vec<&EvalFile> {
    v.iter().collect()
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

/// Whether a position of mate depth `depth` belongs to the named group.
fn in_group(group: &str, depth: u8) -> bool {
    match group {
        "all" => true,
        "M1" => depth == 1,
        "M2" => depth == 2,
        "M3" => depth == 3,
        "M4" => depth == 4,
        "M5" => depth == 5,
        "M2+M3" => depth == 2 || depth == 3,
        _ => false,
    }
}

/// Every group reported for every metric, pooled AND per seed. Both paths call
/// [`grouped`], so the per-seed numbers use exactly the same grouping and
/// bootstrap as the pooled ones.
const GROUPS: [&str; 7] = ["all", "M1", "M2", "M3", "M2+M3", "M4", "M5"];

/// Paired (B minus A) bootstrap summaries of every non-empty group, from two
/// per-position maps over the same position ids.
fn grouped(
    pa: &BTreeMap<String, (u8, f64)>,
    pb: &BTreeMap<String, (u8, f64)>,
    resamples: usize,
    seed: u64,
) -> Result<serde_json::Map<String, serde_json::Value>, String> {
    if pa.keys().ne(pb.keys()) {
        return Err("position sets differ between A and B".into());
    }
    let mut out = serde_json::Map::new();
    for g in GROUPS {
        let diffs: Vec<f64> = pa
            .iter()
            .filter(|(_, v)| in_group(g, v.0))
            .map(|(k, v)| pb[k].1 - v.1)
            .collect();
        if !diffs.is_empty() {
            out.insert(g.into(), ci(diffs, resamples, seed));
        }
    }
    Ok(out)
}

/// Paired per-position comparison (B minus A) of top-1, correct-set mass and
/// negative CE. Each metric reports the pooled groups (all, M1, M2, M3, M2+M3)
/// and, with `per_seed`, a `per_seed.seed_N` entry holding the SAME groups for
/// each seed separately, paired by model-seed identity (refused if the seed sets
/// do not match). A seed's overall difference can therefore never be mistaken
/// for that seed's M1 difference.
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
        let mut groups = grouped(
            &per_position(&ra, f),
            &per_position(&rb, f),
            resamples,
            seed,
        )?;
        if let Some(pairs) = &pairs {
            let mut seeds = serde_json::Map::new();
            for (s, fa, fb) in pairs {
                let g = grouped(
                    &per_position(&[fa], f),
                    &per_position(&[fb], f),
                    resamples,
                    seed,
                )?;
                seeds.insert(format!("seed_{s}"), serde_json::Value::Object(g));
            }
            groups.insert("per_seed".into(), serde_json::Value::Object(seeds));
        }
        out.insert(name.into(), serde_json::Value::Object(groups));
    }
    Ok(serde_json::Value::Object(out))
}

/// The 2x2 factorial interaction `(CF - C0) - (LF - L)` (equivalently
/// `(CF - LF) - (C0 - L)`): does the architecture effect change when facts are
/// present? Per position, each model's value is averaged over seeds, then the
/// interaction is bootstrapped with the SAME grouped paired bootstrap as every
/// other comparison. With `per_seed`, each model seed is also reported separately,
/// requiring the same unique seed set in all four cells.
pub fn interaction(
    c0: &[EvalFile],
    cf: &[EvalFile],
    l: &[EvalFile],
    lf: &[EvalFile],
    resamples: usize,
    seed: u64,
    per_seed: bool,
) -> Result<serde_json::Value, String> {
    let first = c0.first().ok_or("C0 side is empty")?;
    if [c0, cf, l, lf]
        .iter()
        .flat_map(|side| side.iter())
        .any(|f| f.dataset_digest != first.dataset_digest || f.split != first.split)
    {
        return Err("all four cells must be evaluated on the same split and dataset digest".into());
    }
    // The four cells must carry identical unique seed sets, matched by identity.
    let quad: Vec<(u64, &EvalFile, &EvalFile, &EvalFile, &EvalFile)> = {
        let base = align_by_seed(c0, cf)?;
        let against_l = align_by_seed(c0, l)?;
        let against_lf = align_by_seed(c0, lf)?;
        base.iter()
            .zip(&against_l)
            .zip(&against_lf)
            .map(|(((s, a, b), (_, _, bl)), (_, _, blf))| (*s, *a, *b, *bl, *blf))
            .collect()
    };
    let metrics: [(&str, Metric); 3] = [
        ("top1", |r| f64::from(u8::from(r.top1))),
        ("mass", |r| r.mass as f64),
        ("neg_ce", |r| -(r.ce as f64)),
    ];
    // interaction map pair: A = (LF - L), B = (CF - C0), so B - A is the interaction.
    let diff_map = |hi: &[&EvalFile], lo: &[&EvalFile], f: Metric| {
        let (ph, pl) = (per_position(hi, f), per_position(lo, f));
        ph.into_iter()
            .map(|(k, (d, v))| {
                let base = pl.get(&k).map_or(f64::NAN, |x| x.1);
                (k, (d, v - base))
            })
            .collect::<BTreeMap<_, _>>()
    };
    let mut out = serde_json::Map::new();
    for (name, f) in metrics {
        let facts_in_legacy = diff_map(&refs(lf), &refs(l), f);
        let facts_in_candidate = diff_map(&refs(cf), &refs(c0), f);
        let mut groups = grouped(&facts_in_legacy, &facts_in_candidate, resamples, seed)?;
        if per_seed {
            let mut seeds = serde_json::Map::new();
            for (s, a_c0, a_cf, a_l, a_lf) in &quad {
                let legacy = diff_map(&[a_lf], &[a_l], f);
                let candidate = diff_map(&[a_cf], &[a_c0], f);
                seeds.insert(
                    format!("seed_{s}"),
                    serde_json::Value::Object(grouped(&legacy, &candidate, resamples, seed)?),
                );
            }
            groups.insert("per_seed".into(), serde_json::Value::Object(seeds));
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

    /// A file whose top-1 is decided per depth: `correct[d-1]` for depth d.
    fn by_depth(model: &str, seed: u64, correct: [bool; 3]) -> EvalFile {
        let mut f = file(model, Some(seed), 0.0);
        for r in &mut f.results {
            r.top1 = correct[(r.depth - 1) as usize];
        }
        f
    }

    fn diff(v: &serde_json::Value, path: &[&str]) -> f64 {
        let mut cur = v;
        for p in path {
            cur = &cur[*p];
        }
        cur["mean_diff_b_minus_a"]
            .as_f64()
            .unwrap_or_else(|| panic!("missing {path:?}"))
    }

    #[test]
    fn per_seed_output_has_every_preregistered_group_for_every_metric() {
        let a = vec![file("a1", Some(1), 0.0), file("a2", Some(2), 0.05)];
        let b = vec![file("b1", Some(1), 0.2), file("b2", Some(2), 0.25)];
        let out = compare(&a, &b, 200, 3, true).unwrap();
        for metric in ["top1", "mass", "neg_ce"] {
            for g in ["all", "M1", "M2", "M3", "M2+M3"] {
                assert!(out[metric][g]["ci95"].is_array(), "{metric} pooled {g}");
                for s in ["seed_1", "seed_2"] {
                    assert!(
                        out[metric]["per_seed"][s][g]["ci95"].is_array(),
                        "{metric} {s} {g}"
                    );
                }
            }
        }
    }

    #[test]
    fn per_seed_groups_keep_their_own_signs() {
        // seed 1: B better on M1, worse on M2; seed 2: B better on both.
        let a = vec![
            by_depth("a1", 1, [false, true, true]),
            by_depth("a2", 2, [false, false, true]),
        ];
        let b = vec![
            by_depth("b1", 1, [true, false, true]),
            by_depth("b2", 2, [true, true, true]),
        ];
        let out = compare(&a, &b, 200, 3, true).unwrap();
        let ps = |s: &str, g: &str| diff(&out, &["top1", "per_seed", s, g]);
        assert!(ps("seed_1", "M1") > 0.0);
        assert!(ps("seed_1", "M2") < 0.0);
        assert_eq!(ps("seed_1", "M3"), 0.0);
        assert!(ps("seed_2", "M1") > 0.0);
        assert!(ps("seed_2", "M2") > 0.0);
        // M2+M3 and all are different groups from M1: a seed's overall sign
        // cannot stand in for its M1 sign.
        assert!(ps("seed_1", "M2+M3") < 0.0);
        assert!((ps("seed_1", "all") - ps("seed_1", "M1")).abs() > 1e-9);
        // The pooled M1 is the mean of the two seeds' M1 differences (both +1).
        assert!((diff(&out, &["top1", "M1"]) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn file_order_is_irrelevant_for_pooled_and_per_seed_output() {
        let a = vec![file("a1", Some(1), 0.0), file("a2", Some(2), 0.05)];
        let b = vec![file("b1", Some(1), 0.2), file("b2", Some(2), 0.25)];
        let rev = |v: &[EvalFile]| vec![v[1].clone(), v[0].clone()];
        let base = compare(&a, &b, 300, 3, true).unwrap();
        assert_eq!(base, compare(&rev(&a), &b, 300, 3, true).unwrap());
        assert_eq!(base, compare(&a, &rev(&b), 300, 3, true).unwrap());
        assert_eq!(base, compare(&rev(&a), &rev(&b), 300, 3, true).unwrap());
    }

    #[test]
    fn seed_contract_refusals_remain() {
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
    fn dataset_split_and_position_mismatches_are_refused() {
        let a = vec![file("a", Some(1), 0.0)];
        let mut other_digest = vec![file("b", Some(1), 0.0)];
        other_digest[0].dataset_digest = "other".into();
        assert!(compare(&a, &other_digest, 100, 1, true).is_err());
        let mut other_split = vec![file("b", Some(1), 0.0)];
        other_split[0].split = "confirm".into();
        assert!(compare(&a, &other_split, 100, 1, true).is_err());
        let mut other_ids = vec![file("b", Some(1), 0.0)];
        other_ids[0].results[0].id = "not-a-shared-position".into();
        let e = compare(&a, &other_ids, 100, 1, true).unwrap_err();
        assert!(e.contains("position sets differ"), "{e}");
        let mut fewer = vec![file("b", Some(1), 0.0)];
        fewer[0].results.pop();
        assert!(compare(&a, &fewer, 100, 1, true).is_err());
    }

    #[test]
    fn interaction_is_the_difference_of_the_two_facts_effects() {
        // top-1 by depth. Facts help the candidate model on M1 and M2 but the legacy
        // model only on M1, so the interaction is + on M2 and 0 on M1.
        let mk = |m: &str, seed: u64, c: [bool; 3]| by_depth(m, seed, c);
        let c0 = vec![
            mk("c0a", 1, [false, false, false]),
            mk("c0b", 2, [false, false, false]),
        ];
        let cf = vec![
            mk("cfa", 1, [true, true, false]),
            mk("cfb", 2, [true, true, false]),
        ];
        let l = vec![
            mk("la", 1, [false, false, false]),
            mk("lb", 2, [false, false, false]),
        ];
        let lf = vec![
            mk("lfa", 1, [true, false, false]),
            mk("lfb", 2, [true, false, false]),
        ];
        let out = interaction(&c0, &cf, &l, &lf, 200, 3, true).unwrap();
        let d = |g: &str| diff(&out, &["top1", g]);
        assert!(d("M1").abs() < 1e-12, "same facts effect on M1");
        assert!(
            (d("M2") - 1.0).abs() < 1e-12,
            "candidate gains M2 from facts, legacy does not"
        );
        assert_eq!(d("M3"), 0.0);
        assert!(diff(&out, &["top1", "per_seed", "seed_1", "M2"]) > 0.0);
        assert!(diff(&out, &["top1", "per_seed", "seed_2", "M2"]) > 0.0);
        // Order of files is irrelevant; mismatched seed sets are refused.
        let rev = |v: &[EvalFile]| vec![v[1].clone(), v[0].clone()];
        assert_eq!(
            out,
            interaction(&rev(&c0), &cf, &rev(&l), &rev(&lf), 200, 3, true).unwrap()
        );
        let bad = vec![mk("x", 1, [true; 3]), mk("y", 3, [true; 3])];
        assert!(
            interaction(&c0, &cf, &l, &bad, 200, 3, true)
                .unwrap_err()
                .contains("seed sets differ")
        );
    }
}
