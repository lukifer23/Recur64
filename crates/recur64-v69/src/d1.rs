//! D1 (bounded diagnostic of learning failure): panel selection, frozen example
//! stream, hash-enforcing launch verification, and the shallow feature baseline.
//! Everything here is burn-free host code. See docs/v69/D1_CONTRACT.md.

use crate::access::Access;
use crate::dataset::Example;
use crate::features::{FamilyKind, ModelRow, featurize, read_rows};
use crate::provenance::sha256_hex;
use crate::streams::{MasterSeed, keyed_u64};
use anyhow::{Context, Result, bail, ensure};
use cozy_chess::{Board, Color, Piece, Square};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::path::Path;

pub const D1_UPDATES: usize = 2000;
pub const D1_BATCH: usize = 16;
pub const STREAM_D1_PANEL: &str = "d1/panel";
pub const STREAM_D1_ORDER: &str = "d1_train_order";
pub const STREAM_D1_MLP_INIT: &str = "d1_mlp_init";
pub const STREAM_D1_BOOTSTRAP: &str = "d1_bootstrap";


// ------------------------------------------------------------------ panel

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PanelSelection {
    pub algorithm: String,
    pub ids: Vec<String>,
    pub groups: Vec<String>,
    pub extra_strata: Vec<String>,
    pub positives: usize,
    pub negatives: usize,
    pub per_cell: BTreeMap<String, usize>,
    pub seed_fingerprint: String,
}

/// Deterministic panel: 2 per family x budget x class cell (24) + one positive and one
/// negative in each of four of the six family x budget strata chosen by a keyed ordering (8).
/// Candidate order within a cell/stratum is a keyed hash of the id under the D1 panel stream
/// label; no model output is consulted. Requires 32 distinct connected group_ids.
pub fn select_panel(seed: &MasterSeed, fit_meta: &[Example]) -> Result<(Vec<Example>, PanelSelection)> {
    let key = |e: &Example| keyed_u64(seed, STREAM_D1_PANEL, e.id.as_bytes());
    let mut used_groups: HashSet<String> = HashSet::new();
    let mut chosen: Vec<Example> = Vec::new();
    let mut chosen_ids: HashSet<String> = HashSet::new();
    let fams = ["KQQvK", "KQRvK", "KRRvK"];
    let pick = |chosen: &mut Vec<Example>, chosen_ids: &mut HashSet<String>, used: &mut HashSet<String>, fam: &str, b: u8, label: bool, n: usize| -> Result<()> {
        let mut cand: Vec<&Example> = fit_meta.iter().filter(|e| e.family == fam && e.budget == b && e.label == label && !chosen_ids.contains(&e.id)).collect();
        cand.sort_by_key(|e| key(e));
        let mut got = 0;
        for e in cand {
            if got == n {
                break;
            }
            if used.contains(&e.group_id) {
                continue;
            }
            used.insert(e.group_id.clone());
            chosen_ids.insert(e.id.clone());
            chosen.push(e.clone());
            got += 1;
        }
        ensure!(got == n, "capacity: cell {fam}/n{b}/{label} cannot supply {n} examples with distinct groups");
        Ok(())
    };
    for fam in fams {
        for b in [1u8, 2] {
            for label in [false, true] {
                pick(&mut chosen, &mut chosen_ids, &mut used_groups, fam, b, label, 2)?;
            }
        }
    }
    ensure!(chosen.len() == 24, "phase 1 produced {}", chosen.len());
    let mut strata: Vec<(String, u8)> = fams.iter().flat_map(|f| [1u8, 2].map(|b| (f.to_string(), b))).collect();
    strata.sort_by_key(|(f, b)| keyed_u64(seed, "d1/panel/stratum", format!("{f}/{b}").as_bytes()));
    let extra: Vec<(String, u8)> = strata.into_iter().take(4).collect();
    for (f, b) in &extra {
        for label in [true, false] {
            pick(&mut chosen, &mut chosen_ids, &mut used_groups, f, *b, label, 1)?;
        }
    }
    ensure!(chosen.len() == 32, "panel size {}", chosen.len());
    let positives = chosen.iter().filter(|e| e.label).count();
    ensure!(positives == 16, "positives {positives}");
    ensure!(used_groups.len() == 32, "groups not distinct");
    chosen.sort_by(|a, b| a.id.cmp(&b.id));
    let mut per_cell: BTreeMap<String, usize> = BTreeMap::new();
    for e in &chosen {
        *per_cell.entry(format!("{}/n{}/{}", e.family, e.budget, if e.label { "pos" } else { "neg" })).or_default() += 1;
    }
    let sel = PanelSelection {
        algorithm: "phase1: 2 per family x budget x class (cells in fixed order KQQ,KQR,KRR x n1,n2 x neg,pos), candidates by keyed hash of id under label d1/panel, skipping used group_ids; phase2: strata ordered by keyed hash under d1/panel/stratum, first 4, one positive then one negative each (same candidate rule)".into(),
        ids: chosen.iter().map(|e| e.id.clone()).collect(),
        groups: chosen.iter().map(|e| e.group_id.clone()).collect(),
        extra_strata: extra.iter().map(|(f, b)| format!("{f}/n{b}")).collect(),
        positives,
        negatives: 32 - positives,
        per_cell,
        seed_fingerprint: seed.fingerprint(),
    };
    Ok((chosen, sel))
}

/// Frozen training-example stream: epoch e is a Fisher-Yates permutation of 0..32 driven by
/// stream `d1_train_order/e`; epochs are concatenated to UPDATES*BATCH samples. Indices refer
/// to the panel rows sorted by id.
pub fn d1_order(seed: &MasterSeed, panel_len: usize) -> Vec<usize> {
    let total = D1_UPDATES * D1_BATCH;
    let mut out = Vec::with_capacity(total);
    let mut e = 0u64;
    while out.len() < total {
        let mut p: Vec<usize> = (0..panel_len).collect();
        let mut rng = seed.stream(STREAM_D1_ORDER, e);
        for i in (1..panel_len).rev() {
            p.swap(i, rng.below(i as u64 + 1) as usize);
        }
        out.extend(p);
        e += 1;
    }
    out.truncate(total);
    out
}

// ------------------------------------------------------------------ frozen verification

#[derive(Serialize, Deserialize, Debug)]
pub struct FrozenD1 {
    pub created_utc: String,
    pub run: String,
    pub seed_fingerprint: String,
    /// group -> (path relative to the artifact root, or "dataset:<rel>" for dataset files) -> sha256
    pub groups: BTreeMap<String, BTreeMap<String, String>>,
}

pub fn read_frozen(access: &Access) -> Result<FrozenD1> {
    Ok(serde_json::from_slice(&access.read(Path::new("d1/frozen_d1.json"))?)?)
}

/// Enforce (not merely record) the expected hashes of one group; every open goes through `access`.
pub fn verify_group(access: &Access, run_rel: &Path, frozen: &FrozenD1, group: &str) -> Result<usize> {
    let g = frozen.groups.get(group).with_context(|| format!("frozen group {group} missing"))?;
    for (rel, want) in g {
        let p = match rel.strip_prefix("dataset:") {
            Some(d) => run_rel.join(d),
            None => Path::new(rel).to_path_buf(),
        };
        let got = sha256_hex(&access.read(&p).with_context(|| format!("verify {rel}"))?);
        if &got != want {
            bail!("FROZEN HASH MISMATCH ({group}) {rel}: expected {want}, got {got}");
        }
    }
    Ok(g.len())
}

// ------------------------------------------------------------------ shallow baseline

pub const BASE_FEATURES: [&str; 11] = [
    "n_attacker_queens",
    "n_attacker_rooks",
    "def_king_dist_to_edge",
    "def_king_cheb_dist_to_nearest_corner",
    "king_king_cheb_dist",
    "major_min_cheb_dist_to_def_king",
    "major_max_cheb_dist_to_def_king",
    "n_majors_adjacent_to_def_king",
    "def_in_check",
    "n_defender_legal_moves",
    "n_legal_captures_of_attacker_majors",
];

/// 23 raw features: 11 base, budget==2 indicator, 11 base x budget==2 interactions.
pub const N_BASELINE: usize = 23;

pub fn baseline_features(fen: &str, budget: u8) -> Result<[f64; N_BASELINE]> {
    let (_, fam) = featurize(fen, budget)?; // domain validation (defender to move, etc.)
    let b: Board = fen.parse().map_err(|e| anyhow::anyhow!("fen: {e}"))?;
    let defender = b.side_to_move();
    let attacker = !defender;
    let dk = (b.pieces(Piece::King) & b.colors(defender)).into_iter().next().unwrap();
    let ak = (b.pieces(Piece::King) & b.colors(attacker)).into_iter().next().unwrap();
    let cheb = |a: Square, c: Square| -> f64 {
        let (df, dr) = ((a.file() as i32 - c.file() as i32).abs(), (a.rank() as i32 - c.rank() as i32).abs());
        df.max(dr) as f64
    };
    let majors: Vec<Square> = ((b.pieces(Piece::Queen) | b.pieces(Piece::Rook)) & b.colors(attacker)).into_iter().collect();
    ensure!(majors.len() == 2, "expected two attacker majors");
    let (f, r) = (dk.file() as i32, dk.rank() as i32);
    let edge = f.min(7 - f).min(r.min(7 - r)) as f64;
    let corner = [Square::A1, Square::H1, Square::A8, Square::H8].iter().map(|c| cheb(dk, *c)).fold(f64::INFINITY, f64::min);
    let dists: Vec<f64> = majors.iter().map(|m| cheb(*m, dk)).collect();
    let adjacent = dists.iter().filter(|d| **d <= 1.0).count() as f64;
    let in_check = !b.checkers().is_empty() as u8 as f64;
    let mut n_moves = 0f64;
    let mut n_caps = 0f64;
    let maj_set = (b.pieces(Piece::Queen) | b.pieces(Piece::Rook)) & b.colors(attacker);
    b.generate_moves(|ms| {
        for m in ms {
            n_moves += 1.0;
            if maj_set.has(m.to) {
                n_caps += 1.0;
            }
        }
        false
    });
    let (q, rk) = match fam {
        FamilyKind::Kqq => (2.0, 0.0),
        FamilyKind::Kqr => (1.0, 1.0),
        FamilyKind::Krr => (0.0, 2.0),
    };
    let _ = ak;
    let base = [q, rk, edge, corner, cheb(dk, ak), dists.iter().cloned().fold(f64::INFINITY, f64::min), dists.iter().cloned().fold(0.0, f64::max), adjacent, in_check, n_moves, n_caps];
    let b2 = (budget == 2) as u8 as f64;
    let mut out = [0.0; N_BASELINE];
    out[..11].copy_from_slice(&base);
    out[11] = b2;
    for i in 0..11 {
        out[12 + i] = base[i] * b2;
    }
    let _ = Color::White;
    Ok(out)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct BaselineModel {
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
    pub weights: Vec<f64>,
    pub intercept: f64,
    pub steps: usize,
    pub lr: f64,
    pub l2: f64,
    pub final_train_objective: f64,
    pub objective_first_step: f64,
}

pub const BASE_STEPS: usize = 1000;
pub const BASE_LR: f64 = 0.05;
pub const BASE_L2: f64 = 0.01;

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

fn bce(z: f64, y: f64) -> f64 {
    z.max(0.0) - z * y + (-z.abs()).exp().ln_1p()
}

/// Objective: mean_i BCE(z_i, y_i) + (L2/2) * sum_j w_j^2 (intercept excluded).
/// Gradient: (1/n) X^T (sigmoid(z) - y) + L2 * w ; intercept: (1/n) sum (sigmoid(z) - y).
/// Full batch, zero initialization, plain gradient descent, standardization from fitting rows only.
pub fn fit_baseline(x: &[[f64; N_BASELINE]], y: &[bool]) -> BaselineModel {
    let n = x.len() as f64;
    let mut mean = vec![0.0; N_BASELINE];
    let mut std = vec![1.0; N_BASELINE];
    for j in 0..N_BASELINE {
        mean[j] = x.iter().map(|r| r[j]).sum::<f64>() / n;
        let v = x.iter().map(|r| (r[j] - mean[j]).powi(2)).sum::<f64>() / n;
        std[j] = if v > 1e-12 { v.sqrt() } else { 1.0 };
    }
    let xs: Vec<Vec<f64>> = x.iter().map(|r| (0..N_BASELINE).map(|j| (r[j] - mean[j]) / std[j]).collect()).collect();
    let yv: Vec<f64> = y.iter().map(|&l| l as u8 as f64).collect();
    let mut w = vec![0.0; N_BASELINE];
    let mut b = 0.0;
    let obj = |w: &Vec<f64>, b: f64| -> f64 {
        let l: f64 = xs.iter().zip(&yv).map(|(r, y)| bce(r.iter().zip(w).map(|(a, c)| a * c).sum::<f64>() + b, *y)).sum::<f64>() / n;
        l + 0.5 * BASE_L2 * w.iter().map(|v| v * v).sum::<f64>()
    };
    let first = obj(&w, b);
    for _ in 0..BASE_STEPS {
        let mut gw = vec![0.0; N_BASELINE];
        let mut gb = 0.0;
        for (r, y) in xs.iter().zip(&yv) {
            let z = r.iter().zip(&w).map(|(a, c)| a * c).sum::<f64>() + b;
            let d = sigmoid(z) - y;
            for j in 0..N_BASELINE {
                gw[j] += d * r[j];
            }
            gb += d;
        }
        for j in 0..N_BASELINE {
            w[j] -= BASE_LR * (gw[j] / n + BASE_L2 * w[j]);
        }
        b -= BASE_LR * gb / n;
    }
    let last = obj(&w, b);
    BaselineModel { mean, std, weights: w, intercept: b, steps: BASE_STEPS, lr: BASE_LR, l2: BASE_L2, final_train_objective: last, objective_first_step: first }
}

impl BaselineModel {
    pub fn logit(&self, raw: &[f64; N_BASELINE]) -> f64 {
        (0..N_BASELINE).map(|j| (raw[j] - self.mean[j]) / self.std[j] * self.weights[j]).sum::<f64>() + self.intercept
    }
}

/// Load a manifest-verified data split through `access` (rows only; no metadata).
pub fn load_rows_verified(access: &Access, run_rel: &Path, name: &str) -> Result<Vec<ModelRow>> {
    let rel = format!("data/{name}.jsonl");
    let manifest: BTreeMap<String, String> = serde_json::from_slice(&access.read(&run_rel.join("MANIFEST.sha256.json"))?)?;
    let bytes = access.read(&run_rel.join(&rel))?;
    ensure!(manifest.get(&rel) == Some(&sha256_hex(&bytes)), "hash mismatch for {rel}");
    read_rows(std::str::from_utf8(&bytes)?)
}
