//! `recur64 x15 compare` - the pre-registered E11 decision statistics.
//!
//! Compares two groups of checkpoints (A vs B; typically the same variant over
//! seeds) on
//!   (a) top-1 accuracy on the mate fixtures of the tactical suite (higher is
//!       better), and
//!   (c) mean KL to the deepest teacher rung on a targets split (lower is better).
//!
//! The experimental unit is the fixture / position. Within a group, results are
//! averaged over checkpoints (seeds) per unit; only then is the difference
//! bootstrapped over units (paired, deterministic, 2000 resamples). Seeds are
//! never treated as independent observations. Per-seed differences (A_i vs B_i,
//! matched by order) are reported so the sign can be checked in each seed.

use std::path::PathBuf;

use burn::prelude::*;
use clap::Args;

use recur64_model::config::ProbeConfig;
use recur64_runtime::reasoning_targets::ReasoningTargetsV1;

use crate::x15_tactics::{FixtureFile, tactic_vector};
use crate::x15_train::{ckpt_rows, mean, paired_bootstrap, split};

#[derive(Args, Debug)]
pub struct CompareArgs {
    #[arg(long, default_value = "configs/x15_cuda.toml")]
    pub config: PathBuf,
    /// Group A checkpoints (e.g. the seeds of one variant).
    #[arg(long, required = true)]
    pub a: Vec<PathBuf>,
    /// Group B checkpoints, matched to A by order for the per-seed check.
    #[arg(long, required = true)]
    pub b: Vec<PathBuf>,
    /// Thoughts for group A / B (a reasoning-disabled checkpoint always uses 1).
    #[arg(long, default_value_t = 1)]
    pub t_a: usize,
    #[arg(long, default_value_t = 1)]
    pub t_b: usize,
    #[arg(long)]
    pub fixtures: Option<PathBuf>,
    /// Fixture kinds starting with this prefix form the primary tactic metric.
    #[arg(long, default_value = "mate_")]
    pub mate_prefix: String,
    #[arg(long)]
    pub targets: Option<PathBuf>,
    #[arg(long, default_value = "confirm")]
    pub split: String,
    #[arg(long, default_value = "A")]
    pub label_a: String,
    #[arg(long, default_value = "B")]
    pub label_b: String,
}

struct Group {
    /// Per checkpoint, per unit.
    per_ckpt: Vec<Vec<f32>>,
}

impl Group {
    fn pooled(&self) -> Vec<f32> {
        let n = self.per_ckpt[0].len();
        (0..n)
            .map(|i| self.per_ckpt.iter().map(|c| c[i]).sum::<f32>() / self.per_ckpt.len() as f32)
            .collect()
    }
}

fn report(
    name: &str,
    higher_is_better: bool,
    la: &str,
    lb: &str,
    a: &Group,
    b: &Group,
) -> (bool, serde_json::Value) {
    let (pa, pb) = (a.pooled(), b.pooled());
    let (d, lo, hi) = paired_bootstrap(&pa, &pb);
    let seeds = a.per_ckpt.len().min(b.per_ckpt.len());
    let per_seed: Vec<f32> = (0..seeds)
        .map(|i| mean(&a.per_ckpt[i]) - mean(&b.per_ckpt[i]))
        .collect();
    let favourable = |x: f32| if higher_is_better { x > 0.0 } else { x < 0.0 };
    let ci_ok = if higher_is_better { lo > 0.0 } else { hi < 0.0 };
    let seeds_ok = per_seed.iter().all(|x| favourable(*x));
    println!(
        "  {name}: {la} {:.4} vs {lb} {:.4}  diff(A-B) {:+.4}  95% CI [{:+.4},{:+.4}]  per-seed diffs {:?}  -> CI favourable: {ci_ok}, seeds agree: {seeds_ok}",
        mean(&pa),
        mean(&pb),
        d,
        lo,
        hi,
        per_seed
            .iter()
            .map(|x| (x * 1e4).round() / 1e4)
            .collect::<Vec<_>>()
    );
    (
        ci_ok && seeds_ok,
        serde_json::json!({
            "metric": name, "higher_is_better": higher_is_better,
            "a_mean": mean(&pa), "b_mean": mean(&pb),
            "diff": d, "lo95": lo, "hi95": hi, "per_seed_diff": per_seed,
            "ci_favourable": ci_ok, "seeds_agree": seeds_ok,
        }),
    )
}

fn run_impl<B: Backend>(cfg: &ProbeConfig, args: &CompareArgs) -> anyhow::Result<()> {
    let device: B::Device = Default::default();
    println!(
        "compare {} ({} ckpt, T={}) vs {} ({} ckpt, T={})",
        args.label_a,
        args.a.len(),
        args.t_a,
        args.label_b,
        args.b.len(),
        args.t_b
    );
    let mut verdicts = Vec::new();
    let mut json = serde_json::Map::new();

    if let Some(fx) = &args.fixtures {
        let file: FixtureFile = serde_json::from_slice(&std::fs::read(fx)?)?;
        let vecs = |cks: &[PathBuf], t: usize| -> anyhow::Result<Vec<Vec<(String, f32)>>> {
            cks.iter()
                .map(|c| tactic_vector::<B>(cfg, &file.fixtures, c, t, &device))
                .collect()
        };
        let (va, vb) = (vecs(&args.a, args.t_a)?, vecs(&args.b, args.t_b)?);
        let pick = |v: &Vec<Vec<(String, f32)>>, want: &dyn Fn(&str) -> bool| -> Group {
            Group {
                per_ckpt: v
                    .iter()
                    .map(|c| c.iter().filter(|(k, _)| want(k)).map(|(_, x)| *x).collect())
                    .collect(),
            }
        };
        let prefix = args.mate_prefix.clone();
        let is_mate = move |k: &str| k.starts_with(&prefix);
        let (ok, j) = report(
            "mate top-1 accuracy",
            true,
            &args.label_a,
            &args.label_b,
            &pick(&va, &is_mate),
            &pick(&vb, &is_mate),
        );
        verdicts.push(("mate", ok));
        json.insert("mate".into(), j);
        // Secondary, reported per non-mate kind.
        let mut kinds: Vec<String> = va[0].iter().map(|(k, _)| k.clone()).collect();
        kinds.sort();
        kinds.dedup();
        for k in kinds.iter().filter(|k| !k.starts_with(&args.mate_prefix)) {
            let kk = k.clone();
            let f = move |x: &str| x == kk;
            let _ = report(
                &format!("{k} top-1"),
                true,
                &args.label_a,
                &args.label_b,
                &pick(&va, &f),
                &pick(&vb, &f),
            );
        }
        // Per mate material set, descriptive only.
        for k in kinds.iter().filter(|k| k.starts_with(&args.mate_prefix)) {
            let kk = k.clone();
            let f = move |x: &str| x == kk;
            let (pa, pb) = (pick(&va, &f).pooled(), pick(&vb, &f).pooled());
            println!(
                "    {k}: {} {:.2}  {} {:.2}",
                args.label_a,
                mean(&pa),
                args.label_b,
                mean(&pb)
            );
        }
    }

    if let Some(tp) = &args.targets {
        let targets = ReasoningTargetsV1::load(tp)?;
        let pos = split(&targets, &args.split);
        anyhow::ensure!(!pos.is_empty(), "no positions in split {:?}", args.split);
        let rows = |cks: &[PathBuf], t: usize| -> anyhow::Result<Group> {
            let mut per_ckpt = Vec::new();
            for c in cks {
                let (r, _) = ckpt_rows::<B>(cfg, c, &pos, t, &device)?;
                let tt = t.min(r.len());
                per_ckpt.push(r[tt - 1].kl_deep.clone());
            }
            Ok(Group { per_ckpt })
        };
        let (ga, gb) = (rows(&args.a, args.t_a)?, rows(&args.b, args.t_b)?);
        let (ok, j) = report(
            &format!("teacher KL ({} n={})", args.split, pos.len()),
            false,
            &args.label_a,
            &args.label_b,
            &ga,
            &gb,
        );
        verdicts.push(("kl", ok));
        json.insert("kl".into(), j);
    }
    println!(
        "  VERDICT (each metric wholly favourable to A, all seeds agree): {}",
        verdicts
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    Ok(())
}

pub fn run(args: CompareArgs) -> anyhow::Result<()> {
    let cfg = ProbeConfig::from_toml_str(&std::fs::read_to_string(&args.config)?)?;
    match cfg.device {
        recur64_model::config::DeviceKind::Cpu => run_impl::<burn::backend::Flex>(&cfg, &args),
        recur64_model::config::DeviceKind::Cuda => {
            #[cfg(feature = "cuda")]
            {
                run_impl::<burn::backend::Cuda>(&cfg, &args)
            }
            #[cfg(not(feature = "cuda"))]
            {
                anyhow::bail!("CUDA support is not compiled; rebuild with --features cuda")
            }
        }
    }
}
