//! TRAIN-only mechanism measurements (questions A-G of `docs/V4_RESEARCH_PLAN.md`), computed with
//! the rules and seeds pre-registered in `docs/V4_EXPERIMENTS.md` V4-E2. All measurement is on
//! `V4_TRAIN_DEV` (or its fixed `DEV1000` subset), never on `v4_tune_v1`.

use burn::prelude::*;
use serde_json::{Value, json};

use crate::data::V4Data;
use crate::model::EvidenceBeliefModel;
use crate::session::ContentMode;
use crate::stats::{Ci, paired_bootstrap};
use crate::train::{
    EvalSel, PROBE_K, PosEval, ProbeSample, RANK_MARGIN, chunks_min2, eval_policy, mix, probe_batch,
    utility_report,
};

/// Bootstrap seeds (pre-registered).
pub const BOOT_A: u64 = 0x7A40_0101;
pub const BOOT_B_ZERO: u64 = 0x7A40_0102;
pub const BOOT_B_SHUFFLED: u64 = 0x7A40_0103;
pub const BOOT_G_TOP1: u64 = 0x7A40_0104;
pub const BOOT_E_SPEARMAN: u64 = 0x7A40_0105;
pub const BOOT_E_PAIRWISE: u64 = 0x7A40_0106;
pub const BOOT_F: u64 = 0x7A40_0107;
/// Seed of the RANDOM evaluation schedule and of the probe states (pre-registered).
pub const EVAL_RANDOM_SEED: u64 = 0x7A40_E001;
pub const PROBE_SEED: u64 = 0x7A40_E002;
pub const BUDGETS: [usize; 3] = [2, 4, 8];
pub const PREFIXES: [usize; 4] = [0, 1, 2, 3];

pub struct SeedModel<'a, B: Backend> {
    pub seed: u64,
    pub model: &'a EvidenceBeliefModel<B>,
}

fn fnv(s: &str) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in s.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// `DEV1000`: the 1,000 `V4_TRAIN_DEV` positions with the smallest `fnv1a(id)` (frozen).
pub fn dev1000(data: &V4Data) -> Vec<usize> {
    let mut v: Vec<usize> = data.dev.clone();
    v.sort_by_key(|&i| fnv(&data.position(i).id));
    v.truncate(1000);
    v
}

fn ci_json(ci: &Ci) -> Value {
    json!({"mean": ci.mean, "lo": ci.lo, "hi": ci.hi, "per_seed": ci.per_seed,
           "wholly_positive": ci.wholly_positive()})
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len().max(1) as f64
}

fn std(v: &[f64]) -> f64 {
    let m = mean(v);
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len().max(1) as f64).sqrt()
}

fn eval_json(e: &PosEval) -> Value {
    json!({"ce": mean(&e.ce), "top1": mean(&e.top1), "correct_mass": mean(&e.mass),
           "delta_norm_mean": mean(&e.delta_norm), "positions": e.ce.len()})
}

/// Stage A: B0 on `V4_TRAIN_DEV` (recorded, not gated).
pub fn stage_a_report<B: Backend>(
    model: &EvidenceBeliefModel<B>,
    data: &V4Data,
    micro: usize,
    device: &B::Device,
) -> anyhow::Result<Value> {
    let e = eval_policy(model, data, &data.dev, 0, EvalSel::Fixed, micro, device)?;
    let mut cells: std::collections::BTreeMap<String, Vec<usize>> = Default::default();
    for (r, &i) in data.dev.iter().enumerate() {
        let p = data.position(i);
        cells.entry(format!("{} M{}", p.family, p.mate_depth)).or_default().push(r);
    }
    let by_cell: Value = cells
        .iter()
        .map(|(k, rows)| {
            let pick = |v: &[f64]| mean(&rows.iter().map(|&r| v[r]).collect::<Vec<_>>());
            (
                k.clone(),
                json!({"n": rows.len(), "ce": pick(&e.ce), "top1": pick(&e.top1), "correct_mass": pick(&e.mass)}),
            )
        })
        .collect::<serde_json::Map<_, _>>()
        .into();
    Ok(json!({"measure": "stage_a_b0_on_v4_train_dev", "overall": eval_json(&e), "by_cell": by_cell,
              "chance_top1_mean": mean(&data.dev.iter().map(|&i| f64::from(data.position(i).chance_top1)).collect::<Vec<_>>())}))
}

struct Variants {
    normal: PosEval,
    zero: PosEval,
    shuffled: PosEval,
}

/// Stage B: questions A, B, C and G on `V4_TRAIN_DEV`, pooled over the seeds' models.
pub fn mechanism_report<B: Backend>(
    models: &[SeedModel<'_, B>],
    data: &V4Data,
    micro: usize,
    device: &B::Device,
) -> anyhow::Result<Value> {
    anyhow::ensure!(!models.is_empty(), "no models");
    let dev = &data.dev;
    let b0: Vec<PosEval> = models
        .iter()
        .map(|m| eval_policy(m.model, data, dev, 0, EvalSel::Fixed, micro, device))
        .collect::<Result<_, _>>()?;
    let scheds = ["fixed", "random"];
    // [sched][budget_index][seed]
    let mut runs: Vec<Vec<Vec<Variants>>> = Vec::new();
    for sched in scheds {
        let mut per_budget = Vec::new();
        for &b in &BUDGETS {
            let mut per_seed = Vec::new();
            for m in models {
                let sel = || match sched {
                    "fixed" => EvalSel::Fixed,
                    _ => EvalSel::Random(EVAL_RANDOM_SEED),
                };
                let normal = eval_policy(m.model, data, dev, b, sel(), micro, device)?;
                let replay = |content| {
                    eval_policy(
                        m.model,
                        data,
                        dev,
                        b,
                        EvalSel::Replay {
                            chosen: &normal.chosen,
                            content,
                        },
                        micro,
                        device,
                    )
                };
                let zero = replay(ContentMode::Zero)?;
                let shuffled = replay(ContentMode::Shuffled)?;
                per_seed.push(Variants {
                    normal,
                    zero,
                    shuffled,
                });
            }
            per_budget.push(per_seed);
        }
        runs.push(per_budget);
    }

    // C: zero-content replay is exactly B0 for every schedule, budget, seed and position.
    let mut c_exact = true;
    let mut c_max_diff = 0.0f64;
    for per_budget in &runs {
        for per_seed in per_budget {
            for (s, v) in per_seed.iter().enumerate() {
                for (a, b) in v.zero.ce.iter().zip(&b0[s].ce) {
                    c_max_diff = c_max_diff.max((a - b).abs());
                    c_exact &= a == b;
                }
            }
        }
    }

    let last = BUDGETS.len() - 1; // B8
    let mut per_sched = serde_json::Map::new();
    let (mut a_all, mut b_all, mut g_all) = (true, true, true);
    for (si, sched) in scheds.iter().enumerate() {
        let b8 = &runs[si][last];
        let paired = |f: &dyn Fn(usize, &Variants) -> Vec<f64>| -> Vec<Vec<f64>> {
            b8.iter().enumerate().map(|(s, v)| f(s, v)).collect()
        };
        // A
        let a_vals = paired(&|s, v| b0[s].ce.iter().zip(&v.normal.ce).map(|(x, y)| x - y).collect());
        let a_ci = paired_bootstrap(&a_vals, BOOT_A)?;
        let norms: Vec<f64> = b8.iter().flat_map(|v| v.normal.delta_norm.iter().copied()).collect();
        let nondegenerate = mean(&norms) > 0.0 && std(&norms) > 0.0;
        let a_pass = a_ci.wholly_positive() && nondegenerate;
        // B
        let bz = paired_bootstrap(
            &paired(&|_, v| v.zero.ce.iter().zip(&v.normal.ce).map(|(x, y)| x - y).collect()),
            BOOT_B_ZERO,
        )?;
        let bs = paired_bootstrap(
            &paired(&|_, v| v.shuffled.ce.iter().zip(&v.normal.ce).map(|(x, y)| x - y).collect()),
            BOOT_B_SHUFFLED,
        )?;
        let b_pass = bz.wholly_positive() && bs.wholly_positive();
        // G
        let g_top1 = paired_bootstrap(
            &paired(&|s, v| v.normal.top1.iter().zip(&b0[s].top1).map(|(x, y)| x - y).collect()),
            BOOT_G_TOP1,
        )?;
        let mass_by_budget: Vec<f64> = std::iter::once(mean(
            &b0.iter().flat_map(|e| e.mass.iter().copied()).collect::<Vec<_>>(),
        ))
        .chain(runs[si].iter().map(|per_seed| {
            mean(
                &per_seed
                    .iter()
                    .flat_map(|v| v.normal.mass.iter().copied())
                    .collect::<Vec<_>>(),
            )
        }))
        .collect();
        let monotone = mass_by_budget.windows(2).all(|w| w[1] > w[0]);
        let per_seed_mono: Vec<bool> = (0..models.len())
            .map(|s| {
                let mut m = vec![mean(&b0[s].mass)];
                m.extend(runs[si].iter().map(|ps| mean(&ps[s].normal.mass)));
                m.windows(2).all(|w| w[1] >= w[0])
            })
            .collect();
        let g_pass = g_top1.wholly_positive() && monotone;
        let confidence_only = a_ci.wholly_positive() && !g_top1.wholly_positive();
        a_all &= a_pass;
        b_all &= b_pass;
        g_all &= g_pass;
        let table: Vec<Value> = BUDGETS
            .iter()
            .enumerate()
            .map(|(bi, &b)| {
                json!({"budget": b, "per_seed": runs[si][bi].iter().enumerate().map(|(s, v)| json!({
                    "seed": models[s].seed,
                    "normal": eval_json(&v.normal), "zero_content": eval_json(&v.zero),
                    "shuffled_content": eval_json(&v.shuffled)})).collect::<Vec<_>>()})
            })
            .collect();
        per_sched.insert(
            (*sched).to_string(),
            json!({
                "A": {"ce_b0_minus_ce_b8": ci_json(&a_ci), "delta_norm_mean": mean(&norms),
                      "delta_norm_std": std(&norms), "nondegenerate": nondegenerate, "pass": a_pass},
                "B": {"ce_zero_minus_normal_b8": ci_json(&bz), "ce_shuffled_minus_normal_b8": ci_json(&bs), "pass": b_pass},
                "G": {"top1_b8_minus_b0": ci_json(&g_top1), "correct_mass_b0_b2_b4_b8": mass_by_budget,
                      "pooled_monotone": monotone, "per_seed_non_decreasing": per_seed_mono,
                      "confidence_only_gain": confidence_only, "pass": g_pass},
                "table": table,
            }),
        );
    }
    let verdict_stop = !(a_all && b_all && c_exact);
    Ok(json!({
        "measure": "stage_b_mechanism_on_v4_train_dev",
        "positions": dev.len(),
        "seeds": models.iter().map(|m| m.seed).collect::<Vec<_>>(),
        "b0": models.iter().enumerate().map(|(s, m)| json!({"seed": m.seed, "eval": eval_json(&b0[s])})).collect::<Vec<_>>(),
        "C": {"zero_content_replay_equals_b0_bitwise": c_exact, "max_abs_ce_difference": c_max_diff},
        "per_schedule": per_sched,
        "A_pass_both_schedules": a_all,
        "B_pass_both_schedules": b_all,
        "C_pass": c_exact,
        "G_pass_both_schedules": g_all,
        "stop_architecture_development": verdict_stop,
        "rule": "A, B, G must hold under BOTH the FIXED and RANDOM schedules; C is bitwise; any of A, B, C failing stops architecture development",
    }))
}

/// Probe samples of one model on `idx` at every prefix.
pub fn collect_probes<B: Backend>(
    model: &EvidenceBeliefModel<B>,
    data: &V4Data,
    idx: &[usize],
    micro: usize,
    device: &B::Device,
) -> anyhow::Result<Vec<ProbeSample>> {
    let mut out = Vec::new();
    for &prefix in &PREFIXES {
        for (ci, chunk) in chunks_min2(idx, micro).into_iter().enumerate() {
            let (_, s) = probe_batch(
                model,
                data,
                chunk,
                prefix,
                PROBE_K,
                mix(PROBE_SEED, (prefix * 10_000 + ci) as u64),
                device,
            )?;
            out.extend(s);
        }
    }
    Ok(out)
}

/// Stage C: questions D, E, F from probe samples of each seed's model (same states for every
/// seed, since the label-independent prefix does not depend on the model).
pub fn utility_study(
    per_seed: &[(u64, Vec<ProbeSample>)],
    noise_check_exact: bool,
) -> anyhow::Result<Value> {
    anyhow::ensure!(!per_seed.is_empty(), "no seeds");
    let n_states = per_seed[0].1.len();
    anyhow::ensure!(
        per_seed.iter().all(|(_, s)| s.len() == n_states
            && s.iter().zip(&per_seed[0].1).all(|(a, b)| a.position == b.position && a.prefix == b.prefix)),
        "the seeds' probe states are not aligned"
    );
    let reports: Vec<_> = per_seed.iter().map(|(_, s)| utility_report(s)).collect();
    // D, pooled over seeds.
    let pooled: Vec<ProbeSample> = per_seed.iter().flat_map(|(_, s)| s.iter().cloned()).collect();
    let d = utility_report(&pooled);
    let d_pass = d.frac_positive >= 0.05 && d.frac_negative >= 0.05 && d.std_u > 1e-4 && noise_check_exact;
    // E: per-state Spearman and per-state pairwise accuracy, states valid for every seed.
    let sp: Vec<Vec<Option<f64>>> = per_seed
        .iter()
        .map(|(_, s)| s.iter().map(|x| crate::stats::spearman(&x.scores, &x.u)).collect())
        .collect();
    let pw: Vec<Vec<Option<f64>>> = per_seed
        .iter()
        .map(|(_, s)| {
            s.iter()
                .map(|x| {
                    let (ok, n) = crate::stats::pairwise_agreement(&x.scores, &x.u, RANK_MARGIN);
                    (n > 0).then(|| ok as f64 / n as f64 - 0.5)
                })
                .collect()
        })
        .collect();
    let keep = |v: &[Vec<Option<f64>>]| -> Vec<Vec<f64>> {
        let rows: Vec<usize> = (0..n_states).filter(|&i| v.iter().all(|s| s[i].is_some())).collect();
        v.iter().map(|s| rows.iter().map(|&i| s[i].expect("kept")).collect()).collect()
    };
    let (sp_k, pw_k) = (keep(&sp), keep(&pw));
    let sp_ci = paired_bootstrap(&sp_k, BOOT_E_SPEARMAN)?;
    let pw_ci = paired_bootstrap(&pw_k, BOOT_E_PAIRWISE)?;
    let e_pass = sp_ci.wholly_positive() && pw_ci.wholly_positive();
    // F: U(argmax predicted) minus mean U(other probed edges), per state.
    let f_vals: Vec<Vec<f64>> = per_seed
        .iter()
        .map(|(_, s)| {
            s.iter()
                .map(|x| x.u[0] - x.u[1..].iter().sum::<f64>() / x.u[1..].len().max(1) as f64)
                .collect()
        })
        .collect();
    let f_ci = paired_bootstrap(&f_vals, BOOT_F)?;
    let f_pass = f_ci.wholly_positive();
    let by_prefix: Vec<Value> = crate::study::PREFIXES
        .iter()
        .map(|&p| {
            let sel: Vec<ProbeSample> = pooled.iter().filter(|s| s.prefix == p).cloned().collect();
            let r = utility_report(&sel);
            json!({"prefix": p, "states": r.states, "frac_positive": r.frac_positive,
                   "frac_negative": r.frac_negative, "mean_u": r.mean_u, "spearman_mean": r.spearman_mean,
                   "mean_u_preferred": r.mean_u_preferred, "mean_u_random_probed": r.mean_u_random_probed})
        })
        .collect();
    Ok(json!({
        "measure": "stage_c_utility_on_dev1000",
        "seeds": per_seed.iter().map(|(s, _)| *s).collect::<Vec<_>>(),
        "states_per_seed": n_states,
        "D": {"frac_positive": d.frac_positive, "frac_negative": d.frac_negative, "frac_zero": d.frac_zero,
              "mean_u": d.mean_u, "std_u": d.std_u, "probes": d.probes,
              "repeat_probe_noise_exactly_zero": noise_check_exact, "pass": d_pass},
        "E": {"spearman_per_state": ci_json(&sp_ci), "pairwise_accuracy_minus_half": ci_json(&pw_ci),
              "pairwise_ok": d.pairwise_ok, "pairwise_n": d.pairwise_n,
              "states_with_spearman": sp_k[0].len(), "pass": e_pass},
        "F": {"u_preferred_minus_u_random_probed": ci_json(&f_ci),
              "mean_u_preferred": d.mean_u_preferred, "mean_u_random_probed": d.mean_u_random_probed,
              "frac_preferred_positive": d.frac_preferred_positive,
              "frac_random_positive": d.frac_random_positive,
              "mean_regret_vs_best_probed": d.mean_regret_vs_best_probed, "pass": f_pass},
        "per_seed": reports.iter().zip(per_seed).map(|(r, (s, _))| json!({
            "seed": s, "frac_positive": r.frac_positive, "frac_negative": r.frac_negative,
            "spearman_mean": r.spearman_mean, "mean_u_preferred": r.mean_u_preferred,
            "mean_u_random_probed": r.mean_u_random_probed})).collect::<Vec<_>>(),
        "by_prefix": by_prefix,
        "multi_step_bmps_diagnostic": "NOT RUN",
        "rule": "D: >=5% positive and >=5% negative probes, std(U) > 1e-4, repeat-probe label noise exactly 0; E and F: CI wholly above null",
        "reconsider_utility_formulation": !(d_pass && e_pass && f_pass),
    }))
}

/// Stage D: a smoke-scale on-policy integration report (diagnostic, no gate).
pub fn integration_report<B: Backend>(
    models: &[SeedModel<'_, B>],
    data: &V4Data,
    micro: usize,
    device: &B::Device,
) -> anyhow::Result<Value> {
    let dev = &data.dev;
    let mut rows = Vec::new();
    for &b in &BUDGETS {
        let (mut util_fixed, mut util_random) = (Vec::new(), Vec::new());
        let mut table = Vec::new();
        for m in models {
            let u = eval_policy(m.model, data, dev, b, EvalSel::Utility, micro, device)?;
            let f = eval_policy(m.model, data, dev, b, EvalSel::Fixed, micro, device)?;
            let r = eval_policy(m.model, data, dev, b, EvalSel::Random(EVAL_RANDOM_SEED), micro, device)?;
            util_fixed.push(f.ce.iter().zip(&u.ce).map(|(x, y)| x - y).collect::<Vec<_>>());
            util_random.push(r.ce.iter().zip(&u.ce).map(|(x, y)| x - y).collect::<Vec<_>>());
            table.push(json!({"seed": m.seed, "utility": eval_json(&u), "fixed": eval_json(&f), "random": eval_json(&r)}));
        }
        rows.push(json!({
            "budget": b,
            "ce_fixed_minus_ce_utility": ci_json(&paired_bootstrap(&util_fixed, BOOT_F ^ 0x10)?),
            "ce_random_minus_ce_utility": ci_json(&paired_bootstrap(&util_random, BOOT_F ^ 0x20)?),
            "per_seed": table,
        }));
    }
    Ok(json!({"measure": "stage_d_integration_smoke_on_v4_train_dev", "gate": "none (diagnostic)", "budgets": rows}))
}
