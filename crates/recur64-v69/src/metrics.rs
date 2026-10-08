//! Independent metric/gate aggregation from serialized predictions (f64).
//!
//! This module shares no code with the model crate's in-process evaluation: it
//! reads only prediction rows and fit/val metadata, recomputes everything in f64
//! and applies the pre-declared gates. Uncertainty is a cluster bootstrap over
//! connected `group_id`s (rows are never resampled as independent).

use crate::dataset::Example;
use crate::streams::MasterSeed;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredRow {
    pub id: String,
    /// "real" | "derange" | "erase"
    pub mode: String,
    /// Logits at each prescribed readout, in order; the last is the final readout.
    pub logits: Vec<f32>,
    /// For mode == "derange": id of the donor whose board was used.
    pub donor_id: Option<String>,
}

pub fn n_readouts(arm: &str) -> usize {
    match arm {
        "A" => 1,
        _ => 3,
    }
}

fn bce(z: f64, y: bool) -> f64 {
    let yv = if y { 1.0 } else { 0.0 };
    z.max(0.0) - z * yv + (-z.abs()).exp().ln_1p()
}

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

#[derive(Default, Clone, Copy, Serialize, Debug)]
pub struct Conf {
    pub tn: u64,
    pub fp: u64,
    pub fn_: u64,
    pub tp: u64,
}

impl Conf {
    fn add(&mut self, label: bool, pred_pos: bool) {
        match (label, pred_pos) {
            (false, false) => self.tn += 1,
            (false, true) => self.fp += 1,
            (true, false) => self.fn_ += 1,
            (true, true) => self.tp += 1,
        }
    }
    fn merge(&mut self, o: &Conf) {
        self.tn += o.tn;
        self.fp += o.fp;
        self.fn_ += o.fn_;
        self.tp += o.tp;
    }
    pub fn n(&self) -> u64 {
        self.tn + self.fp + self.fn_ + self.tp
    }
    pub fn acc(&self) -> f64 {
        (self.tn + self.tp) as f64 / self.n().max(1) as f64
    }
    pub fn bal_acc(&self) -> f64 {
        let tpr = self.tp as f64 / (self.tp + self.fn_).max(1) as f64;
        let tnr = self.tn as f64 / (self.tn + self.fp).max(1) as f64;
        0.5 * (tpr + tnr)
    }
}

#[derive(Default, Clone, Copy)]
struct GroupStats {
    real: Conf,
    der_recipient: Conf,
    der_donor: Conf,
    bce_sum: f64,
    bce_n: u64,
}

#[derive(Serialize, Debug, Clone)]
pub struct Ci {
    pub point: f64,
    pub lo95: f64,
    pub hi95: f64,
}

#[derive(Serialize, Debug)]
pub struct PartitionReport {
    pub n: usize,
    pub real_final_bce: f64,
    pub real_bal_acc: f64,
    pub real_acc: f64,
    pub brier: f64,
    pub confusion: Conf,
    pub per_readout_bce: Vec<f64>,
    pub per_readout_bal_acc: Vec<f64>,
    pub per_cell: BTreeMap<String, CellMetrics>,
    pub derange: Option<DerangeReport>,
    pub erase: Option<EraseReport>,
}

#[derive(Serialize, Debug)]
pub struct CellMetrics {
    pub n: u64,
    pub acc: f64,
    pub mean_final_bce: f64,
    pub pred_pos_rate: f64,
}

#[derive(Serialize, Debug)]
pub struct DerangeReport {
    pub acc_vs_recipient_labels: f64,
    pub bal_acc_vs_recipient_labels: f64,
    pub acc_vs_donor_labels: f64,
    pub bal_acc_vs_donor_labels: f64,
    pub donor_label_agreement_rate: f64,
    pub max_abs_logit_diff_vs_donor_real: f64,
    pub n_donor_pred_checked: usize,
}

#[derive(Serialize, Debug)]
pub struct EraseReport {
    pub acc: f64,
    pub bal_acc: f64,
    pub pred_pos_rate: f64,
    pub mean_final_bce: f64,
}

#[derive(Serialize, Debug)]
pub struct Gates {
    pub fit_bal_acc_ge_95: bool,
    pub val_bal_acc_ge_75: bool,
    pub val_final_bce_le_055: bool,
    pub derange_drop_ge_15pp: bool,
    pub derange_drop_pp: f64,
    pub metric_gates_pass: bool,
}

#[derive(Serialize, Debug)]
pub struct Report {
    pub arm: String,
    pub fit: PartitionReport,
    pub val: PartitionReport,
    pub val_bootstrap: BTreeMap<String, Ci>,
    pub bootstrap_resamples: usize,
    pub bootstrap_groups: usize,
    pub gates: Gates,
}

pub const BOOTSTRAP_RESAMPLES: usize = 2000;
/// Tolerance for "donor input predicts the same as the donor's ordinary input".
pub const DONOR_LOGIT_TOL: f64 = 1e-3;

fn partition_report(
    arm: &str,
    preds: &[PredRow],
    metas: &BTreeMap<String, &Example>,
    want_groups: bool,
) -> anyhow::Result<(PartitionReport, BTreeMap<String, GroupStats>)> {
    let k = n_readouts(arm);
    let mut real: BTreeMap<&str, &PredRow> = BTreeMap::new();
    let mut der: Vec<&PredRow> = Vec::new();
    let mut ers: Vec<&PredRow> = Vec::new();
    for p in preds {
        anyhow::ensure!(p.logits.len() == k, "{} readouts expected {k}, got {}", p.id, p.logits.len());
        anyhow::ensure!(p.logits.iter().all(|z| z.is_finite()), "non-finite logit {}", p.id);
        anyhow::ensure!(metas.contains_key(&p.id), "prediction for unknown id {}", p.id);
        match p.mode.as_str() {
            "real" => {
                anyhow::ensure!(real.insert(&p.id, p).is_none(), "duplicate real prediction {}", p.id);
            }
            "derange" => der.push(p),
            "erase" => ers.push(p),
            m => anyhow::bail!("unknown mode {m}"),
        }
    }
    anyhow::ensure!(real.len() == metas.len(), "real predictions {} != examples {}", real.len(), metas.len());
    let mut conf = Conf::default();
    let mut bce_sum = 0.0;
    let mut brier = 0.0;
    let mut ro_bce = vec![0.0; k];
    let mut ro_conf = vec![Conf::default(); k];
    let mut cells: BTreeMap<String, (u64, u64, f64, u64)> = BTreeMap::new();
    let mut groups: BTreeMap<String, GroupStats> = BTreeMap::new();
    for (id, p) in &real {
        let m = metas[*id];
        let z = *p.logits.last().unwrap() as f64;
        conf.add(m.label, z > 0.0);
        bce_sum += bce(z, m.label);
        let pr = sigmoid(z);
        brier += (pr - if m.label { 1.0 } else { 0.0 }).powi(2);
        for (i, l) in p.logits.iter().enumerate() {
            ro_bce[i] += bce(*l as f64, m.label);
            ro_conf[i].add(m.label, (*l as f64) > 0.0);
        }
        let key = format!("{}/n{}/{}", m.family, m.budget, if m.label { "pos" } else { "neg" });
        let c = cells.entry(key).or_default();
        c.0 += 1;
        c.1 += ((z > 0.0) == m.label) as u64;
        c.2 += bce(z, m.label);
        c.3 += (z > 0.0) as u64;
        if want_groups {
            let g = groups.entry(m.group_id.clone()).or_default();
            g.real.add(m.label, z > 0.0);
            g.bce_sum += bce(z, m.label);
            g.bce_n += 1;
        }
    }
    let n = real.len() as f64;
    // derangement
    let derange = if der.is_empty() {
        None
    } else {
        anyhow::ensure!(der.len() == metas.len(), "derange rows {} != {}", der.len(), metas.len());
        let (mut rc, mut dc) = (Conf::default(), Conf::default());
        let mut agree = 0usize;
        let mut max_diff = 0f64;
        for p in &der {
            let m = metas[p.id.as_str()];
            let donor_id = p.donor_id.as_deref().ok_or_else(|| anyhow::anyhow!("derange row without donor"))?;
            anyhow::ensure!(donor_id != p.id, "donor == recipient for {} (not a derangement)", p.id);
            let dm = metas
                .get(donor_id)
                .ok_or_else(|| anyhow::anyhow!("donor {donor_id} not in partition"))?;
            anyhow::ensure!(dm.family == m.family && dm.budget == m.budget, "donor outside family/budget cell for {}", p.id);
            let z = *p.logits.last().unwrap() as f64;
            rc.add(m.label, z > 0.0);
            dc.add(dm.label, z > 0.0);
            agree += (m.label == dm.label) as usize;
            let dz = *real[donor_id].logits.last().unwrap() as f64;
            max_diff = max_diff.max((dz - z).abs());
            if want_groups {
                let g = groups.entry(m.group_id.clone()).or_default();
                g.der_recipient.add(m.label, z > 0.0);
                g.der_donor.add(dm.label, z > 0.0);
            }
        }
        Some(DerangeReport {
            acc_vs_recipient_labels: rc.acc(),
            bal_acc_vs_recipient_labels: rc.bal_acc(),
            acc_vs_donor_labels: dc.acc(),
            bal_acc_vs_donor_labels: dc.bal_acc(),
            donor_label_agreement_rate: agree as f64 / der.len() as f64,
            max_abs_logit_diff_vs_donor_real: max_diff,
            n_donor_pred_checked: der.len(),
        })
    };
    let erase = if ers.is_empty() {
        None
    } else {
        let mut c = Conf::default();
        let mut b = 0.0;
        let mut pos = 0u64;
        for p in &ers {
            let m = metas[p.id.as_str()];
            let z = *p.logits.last().unwrap() as f64;
            c.add(m.label, z > 0.0);
            b += bce(z, m.label);
            pos += (z > 0.0) as u64;
        }
        Some(EraseReport { acc: c.acc(), bal_acc: c.bal_acc(), pred_pos_rate: pos as f64 / ers.len() as f64, mean_final_bce: b / ers.len() as f64 })
    };
    let per_cell = cells
        .into_iter()
        .map(|(k, (cn, ok, b, pos))| (k, CellMetrics { n: cn, acc: ok as f64 / cn as f64, mean_final_bce: b / cn as f64, pred_pos_rate: pos as f64 / cn as f64 }))
        .collect();
    Ok((
        PartitionReport {
            n: real.len(),
            real_final_bce: bce_sum / n,
            real_bal_acc: conf.bal_acc(),
            real_acc: conf.acc(),
            brier: brier / n,
            confusion: conf,
            per_readout_bce: ro_bce.iter().map(|b| b / n).collect(),
            per_readout_bal_acc: ro_conf.iter().map(|c| c.bal_acc()).collect(),
            per_cell,
            derange,
            erase,
        },
        groups,
    ))
}

pub fn aggregate(
    arm: &str,
    fit_preds: &[PredRow],
    val_preds: &[PredRow],
    fit_meta: &[Example],
    val_meta: &[Example],
    seed: &MasterSeed,
) -> anyhow::Result<Report> {
    let fm: BTreeMap<String, &Example> = fit_meta.iter().map(|e| (e.id.clone(), e)).collect();
    let vm: BTreeMap<String, &Example> = val_meta.iter().map(|e| (e.id.clone(), e)).collect();
    let (fit, _) = partition_report(arm, fit_preds, &fm, false)?;
    let (val, groups) = partition_report(arm, val_preds, &vm, true)?;
    // cluster bootstrap over connected groups (validation)
    let gvec: Vec<&GroupStats> = groups.values().collect();
    let ng = gvec.len();
    let mut rng = seed.stream(&format!("bootstrap/{arm}"), 0);
    let mut samples: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    for _ in 0..BOOTSTRAP_RESAMPLES {
        let (mut real, mut der) = (Conf::default(), Conf::default());
        let (mut bs, mut bn) = (0.0, 0u64);
        for _ in 0..ng {
            let g = gvec[rng.below(ng as u64) as usize];
            real.merge(&g.real);
            der.merge(&g.der_recipient);
            bs += g.bce_sum;
            bn += g.bce_n;
        }
        samples.entry("val_bal_acc").or_default().push(real.bal_acc());
        samples.entry("val_final_bce").or_default().push(bs / bn.max(1) as f64);
        if val.derange.is_some() {
            samples.entry("val_derange_bal_acc_vs_recipient").or_default().push(der.bal_acc());
            samples.entry("val_gap_pp").or_default().push((real.bal_acc() - der.bal_acc()) * 100.0);
        }
    }
    let mut boot = BTreeMap::new();
    for (k, mut v) in samples {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = |p: f64| v[((v.len() as f64 - 1.0) * p).round() as usize];
        let point = match k {
            "val_bal_acc" => val.real_bal_acc,
            "val_final_bce" => val.real_final_bce,
            "val_derange_bal_acc_vs_recipient" => val.derange.as_ref().unwrap().bal_acc_vs_recipient_labels,
            _ => (val.real_bal_acc - val.derange.as_ref().unwrap().bal_acc_vs_recipient_labels) * 100.0,
        };
        boot.insert(k.to_string(), Ci { point, lo95: q(0.025), hi95: q(0.975) });
    }
    let drop_pp = val.derange.as_ref().map(|d| (val.real_bal_acc - d.bal_acc_vs_recipient_labels) * 100.0).unwrap_or(f64::NAN);
    let g = Gates {
        fit_bal_acc_ge_95: fit.real_bal_acc >= 0.95,
        val_bal_acc_ge_75: val.real_bal_acc >= 0.75,
        val_final_bce_le_055: val.real_final_bce <= 0.55,
        derange_drop_ge_15pp: drop_pp >= 15.0,
        derange_drop_pp: drop_pp,
        metric_gates_pass: false,
    };
    let pass = g.fit_bal_acc_ge_95 && g.val_bal_acc_ge_75 && g.val_final_bce_le_055 && g.derange_drop_ge_15pp;
    Ok(Report {
        arm: arm.to_string(),
        fit,
        val,
        val_bootstrap: boot,
        bootstrap_resamples: BOOTSTRAP_RESAMPLES,
        bootstrap_groups: ng,
        gates: Gates { metric_gates_pass: pass, ..g },
    })
}
