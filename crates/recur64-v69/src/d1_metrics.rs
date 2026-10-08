//! Independent (f64) metric aggregation for D1 from serialized predictions.
//! Shares no code with the model crate or the baseline fitter.

use crate::dataset::Example;
use crate::streams::MasterSeed;
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct D1Pred {
    pub id: String,
    pub update: u32,
    pub logit: f64,
}

pub fn bce(z: f64, y: bool) -> f64 {
    z.max(0.0) - z * (y as u8 as f64) + (-z.abs()).exp().ln_1p()
}

/// AUROC via average ranks (ties handled); NaN when one class is absent.
pub fn auroc(scores: &[f64], labels: &[bool]) -> f64 {
    let n = scores.len();
    let np = labels.iter().filter(|l| **l).count();
    let nn = n - np;
    if np == 0 || nn == 0 {
        return f64::NAN;
    }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| scores[a].partial_cmp(&scores[b]).unwrap());
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && scores[idx[j + 1]] == scores[idx[i]] {
            j += 1;
        }
        let r = (i + j) as f64 / 2.0 + 1.0;
        for k in i..=j {
            ranks[idx[k]] = r;
        }
        i = j + 1;
    }
    let sum_pos: f64 = (0..n).filter(|&k| labels[k]).map(|k| ranks[k]).sum();
    (sum_pos - np as f64 * (np as f64 + 1.0) / 2.0) / (np as f64 * nn as f64)
}

#[derive(Serialize, Debug, Clone)]
pub struct Dist {
    pub n: usize,
    pub mean: f64,
    pub sd: f64,
    pub min: f64,
    pub q25: f64,
    pub median: f64,
    pub q75: f64,
    pub max: f64,
}

fn dist(v: &mut Vec<f64>) -> Dist {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n == 0 {
        return Dist { n: 0, mean: f64::NAN, sd: f64::NAN, min: f64::NAN, q25: f64::NAN, median: f64::NAN, q75: f64::NAN, max: f64::NAN };
    }
    let mean = v.iter().sum::<f64>() / n as f64;
    let sd = (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
    let q = |p: f64| v[((n as f64 - 1.0) * p).round() as usize];
    Dist { n, mean, sd, min: v[0], q25: q(0.25), median: q(0.5), q75: q(0.75), max: v[n - 1] }
}

#[derive(Serialize, Debug, Clone)]
pub struct CellStat {
    pub n: usize,
    pub acc: f64,
    pub mean_bce: f64,
}

#[derive(Serialize, Debug, Clone)]
pub struct Ci {
    pub point: f64,
    pub lo95: f64,
    pub hi95: f64,
}

#[derive(Serialize, Debug, Clone)]
pub struct Summary {
    pub n: usize,
    pub n_correct: usize,
    pub acc: f64,
    pub bal_acc: f64,
    pub bce: f64,
    pub brier: f64,
    pub auroc: f64,
    pub tn: usize,
    pub fp: usize,
    pub fn_: usize,
    pub tp: usize,
    pub logit_all: Dist,
    pub logit_pos: Dist,
    pub logit_neg: Dist,
    pub per_cell: BTreeMap<String, CellStat>,
    pub bootstrap: Option<BTreeMap<String, Ci>>,
}

fn core(rows: &[(f64, bool)]) -> (f64, f64, f64, f64, f64) {
    // (acc, bal_acc, bce, brier, auroc)
    let n = rows.len() as f64;
    let (mut tp, mut tn, mut fp, mut fnn) = (0f64, 0f64, 0f64, 0f64);
    let (mut b, mut br) = (0.0, 0.0);
    for (z, y) in rows {
        match (*y, *z > 0.0) {
            (true, true) => tp += 1.0,
            (true, false) => fnn += 1.0,
            (false, false) => tn += 1.0,
            (false, true) => fp += 1.0,
        }
        b += bce(*z, *y);
        br += (1.0 / (1.0 + (-z).exp()) - *y as u8 as f64).powi(2);
    }
    let tpr = tp / (tp + fnn).max(1.0);
    let tnr = tn / (tn + fp).max(1.0);
    let zs: Vec<f64> = rows.iter().map(|r| r.0).collect();
    let ys: Vec<bool> = rows.iter().map(|r| r.1).collect();
    ((tp + tn) / n, 0.5 * (tpr + tnr), b / n, br / n, auroc(&zs, &ys))
}

/// `preds` must cover exactly the ids of `meta` once each. Bootstrap (optional) resamples
/// connected group_ids with replacement.
pub fn summarize(preds: &[(String, f64)], meta: &HashMap<&str, &Example>, bootstrap: Option<(&MasterSeed, &str, usize)>) -> Result<Summary> {
    let mut seen = HashSet::new();
    for (id, z) in preds {
        ensure!(meta.contains_key(id.as_str()), "prediction for unknown id {id}");
        ensure!(seen.insert(id.as_str()), "duplicate prediction id {id}");
        ensure!(z.is_finite(), "non-finite logit for {id}");
    }
    ensure!(seen.len() == meta.len(), "predictions {} != examples {}", seen.len(), meta.len());
    let rows: Vec<(f64, bool, &Example)> = preds.iter().map(|(id, z)| (*z, meta[id.as_str()].label, meta[id.as_str()])).collect();
    let flat: Vec<(f64, bool)> = rows.iter().map(|r| (r.0, r.1)).collect();
    let (acc, ba, b, br, au) = core(&flat);
    let (mut tn, mut fp, mut fnn, mut tp) = (0, 0, 0, 0);
    for (z, y) in &flat {
        match (*y, *z > 0.0) {
            (true, true) => tp += 1,
            (true, false) => fnn += 1,
            (false, false) => tn += 1,
            (false, true) => fp += 1,
        }
    }
    let mut all: Vec<f64> = flat.iter().map(|r| r.0).collect();
    let mut pos: Vec<f64> = flat.iter().filter(|r| r.1).map(|r| r.0).collect();
    let mut neg: Vec<f64> = flat.iter().filter(|r| !r.1).map(|r| r.0).collect();
    let mut cells: BTreeMap<String, Vec<(f64, bool)>> = BTreeMap::new();
    for (z, y, e) in &rows {
        cells.entry(format!("{}/n{}/{}", e.family, e.budget, if *y { "pos" } else { "neg" })).or_default().push((*z, *y));
    }
    let per_cell = cells
        .into_iter()
        .map(|(k, v)| {
            let n = v.len();
            let ok = v.iter().filter(|(z, y)| (*z > 0.0) == *y).count();
            (k, CellStat { n, acc: ok as f64 / n as f64, mean_bce: v.iter().map(|(z, y)| bce(*z, *y)).sum::<f64>() / n as f64 })
        })
        .collect();
    let bootstrap = if let Some((seed, label, resamples)) = bootstrap {
        let mut groups: BTreeMap<&str, Vec<(f64, bool)>> = BTreeMap::new();
        for (z, y, e) in &rows {
            groups.entry(e.group_id.as_str()).or_default().push((*z, *y));
        }
        let gv: Vec<&Vec<(f64, bool)>> = groups.values().collect();
        let mut rng = seed.stream(&format!("{}/{}", crate::d1::STREAM_D1_BOOTSTRAP, label), 0);
        let mut s: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
        for _ in 0..resamples {
            let mut rs: Vec<(f64, bool)> = Vec::new();
            for _ in 0..gv.len() {
                rs.extend(gv[rng.below(gv.len() as u64) as usize].iter().cloned());
            }
            let (_, bab, bb, _, bau) = core(&rs);
            s.entry("bal_acc").or_default().push(bab);
            s.entry("bce").or_default().push(bb);
            if bau.is_finite() {
                s.entry("auroc").or_default().push(bau);
            }
        }
        let mut out = BTreeMap::new();
        for (k, mut v) in s {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let q = |p: f64| v[((v.len() as f64 - 1.0) * p).round() as usize];
            let point = match k {
                "bal_acc" => ba,
                "bce" => b,
                _ => au,
            };
            out.insert(k.to_string(), Ci { point, lo95: q(0.025), hi95: q(0.975) });
        }
        out.insert("n_groups".to_string(), Ci { point: gv.len() as f64, lo95: gv.len() as f64, hi95: gv.len() as f64 });
        Some(out)
    } else {
        None
    };
    Ok(Summary { n: flat.len(), n_correct: flat.iter().filter(|(z, y)| (*z > 0.0) == *y).count(), acc, bal_acc: ba, bce: b, brier: br, auroc: au, tn, fp, fn_: fnn, tp, logit_all: dist(&mut all), logit_pos: dist(&mut pos), logit_neg: dist(&mut neg), per_cell, bootstrap })
}
