//! `recur64 value-diag` — does the value head learn (and keep) a lesson?
//!
//! Training-only diagnostic, minutes per run: from a training checkpoint,
//! train through a sequence of phases (`--phase <replay dir>:<updates>`), and
//! every `--eval-every` updates evaluate each `--eval <name>=<replay dir>`:
//! WDL cross-entropy against the game outcomes (uniform = ln 3) and the value
//! head by material advantage from the leading side's view. Writes the
//! curve to `value-diag.json`. Nothing is saved or promoted; no search runs.

use std::collections::BTreeMap;
use std::path::PathBuf;

use burn::module::AutodiffModule;
use burn::tensor::backend::AutodiffBackend;
use clap::Args;

use recur64_core::{ActionId, Color, GameState, ObservationV1, StandardMove, material_balance};
use recur64_model::checkpoint::load_training;
use recur64_model::train::adamw;
use recur64_runtime::replay::{ReplayReader, ReplayStore, sampler::example_for_ply};
use recur64_runtime::{
    BatchEvaluator, BatchedModel, LearnerConfig, RunConfig, model_io, train_from_store,
};

type CpuTrain = burn::backend::Autodiff<burn::backend::Flex>;

#[derive(Args, Debug)]
pub struct ValueDiagArgs {
    #[arg(long)]
    pub config: PathBuf,
    /// Full training checkpoint to start from (weights + optimizer).
    #[arg(long)]
    pub checkpoint: PathBuf,
    /// Training phases in order, `<replay dir>:<updates>` (repeatable).
    #[arg(long = "phase", required = true)]
    pub phases: Vec<String>,
    /// Evaluation sets, `<name>=<replay dir>` (repeatable).
    #[arg(long = "eval", required = true)]
    pub evals: Vec<String>,
    #[arg(long, default_value_t = 25)]
    pub eval_every: usize,
    /// Peak LR (default: the config's `lr`). Warmup 10 updates, cosine over
    /// the total updates of all phases.
    #[arg(long)]
    pub lr: Option<f64>,
    #[arg(long)]
    pub output: PathBuf,
}

/// Material buckets (leader's advantage, pawn units), as in `search-gain`.
const BUCKETS: [(i32, i32, &str); 4] = [(0, 2, "0-2"), (3, 4, "3-4"), (5, 8, "5-8"), (9, 99, "9+")];

struct EvalSet {
    name: String,
    obs: Vec<ObservationV1>,
    legal: Vec<Vec<ActionId>>,
    /// WDL class from the side to move (0 win, 1 draw, 2 loss).
    class: Vec<usize>,
    /// Material balance from the side to move.
    balance: Vec<i32>,
}

fn load_eval(name: &str, dir: &str) -> anyhow::Result<EvalSet> {
    let mut set = EvalSet {
        name: name.to_string(),
        obs: Vec::new(),
        legal: Vec::new(),
        class: Vec::new(),
        balance: Vec::new(),
    };
    for game in ReplayReader::open(std::path::Path::new(dir))?.read_all_games()? {
        let Some(outcome) = game.outcome else {
            continue; // truncated games have no value label
        };
        let mut state = GameState::from_fen(&game.start_fen)?;
        for ply in &game.plies {
            let ex = example_for_ply(&state, outcome, ply).map_err(|e| anyhow::anyhow!(e))?;
            let white = material_balance(state.board());
            set.balance.push(if state.side_to_move() == Color::White {
                white
            } else {
                -white
            });
            set.class.push(usize::try_from(ex.wdl)?);
            set.obs.push(ex.observation);
            set.legal.push(ex.legal);
            let id = ActionId::from_index(ply.selected as u32)?;
            let (from, to, promo) = id.to_physical(state.perspective());
            let promotion = if promo.is_none() { None } else { Some(promo) };
            state.apply(StandardMove::new(from, to, promotion))?;
        }
    }
    anyhow::ensure!(
        !set.obs.is_empty(),
        "eval set {name} has no labelled positions"
    );
    Ok(set)
}

fn evaluate(model: &dyn BatchEvaluator, set: &EvalSet) -> anyhow::Result<serde_json::Value> {
    let mut ce = 0f64;
    // Per bucket: n, P(win for leader), P(draw), P(loss for leader), and the
    // outcome share (win/draw/loss for the leader) for reference.
    let mut b = [(0u64, 0f64, 0f64, 0f64, [0u64; 3]); 4];
    for start in (0..set.obs.len()).step_by(64) {
        let end = (start + 64).min(set.obs.len());
        let out = model
            .evaluate_batch(&set.obs[start..end], &set.legal[start..end])
            .map_err(|e| anyhow::anyhow!("evaluation failed: {e}"))?;
        for (k, r) in out.iter().enumerate() {
            let i = start + k;
            ce -= (r.wdl[set.class[i]].max(1e-12) as f64).ln();
            let bal = set.balance[i];
            if let Some(j) = BUCKETS
                .iter()
                .position(|(lo, hi, _)| bal.abs() >= *lo && bal.abs() <= *hi)
            {
                // Leader's view: flip win/loss when the side to move is behind.
                let (w, l, cls) = if bal >= 0 {
                    (r.wdl[0], r.wdl[2], set.class[i])
                } else {
                    (r.wdl[2], r.wdl[0], 2 - set.class[i])
                };
                let e = &mut b[j];
                e.0 += 1;
                e.1 += w as f64;
                e.2 += r.wdl[1] as f64;
                e.3 += l as f64;
                e.4[cls] += 1;
            }
        }
    }
    let n = set.obs.len() as f64;
    let buckets: Vec<serde_json::Value> = BUCKETS
        .iter()
        .zip(b)
        .map(|((_, _, label), (m, w, d, l, o))| {
            let k = m.max(1) as f64;
            serde_json::json!({
                "bucket": label, "positions": m,
                "p_win_leader": w / k, "p_draw": d / k, "p_loss_leader": l / k,
                "outcome_win_leader": o[0] as f64 / k,
                "outcome_draw": o[1] as f64 / k,
                "outcome_loss_leader": o[2] as f64 / k,
            })
        })
        .collect();
    Ok(serde_json::json!({
        "set": set.name,
        "positions": set.obs.len(),
        "wdl_ce": ce / n,
        "uniform_ce": 3f64.ln(),
        "value_by_material": buckets,
    }))
}

fn run_impl<B: AutodiffBackend>(cfg: &RunConfig, args: &ValueDiagArgs) -> anyhow::Result<()> {
    let device: B::Device = Default::default();
    let phases: Vec<(String, usize)> = args
        .phases
        .iter()
        .map(|p| {
            let (dir, n) = p
                .rsplit_once(':')
                .ok_or_else(|| anyhow::anyhow!("phase {p:?}: expected <replay dir>:<updates>"))?;
            Ok((dir.to_string(), n.parse()?))
        })
        .collect::<anyhow::Result<_>>()?;
    let evals: Vec<EvalSet> = args
        .evals
        .iter()
        .map(|e| {
            let (name, dir) = e
                .split_once('=')
                .ok_or_else(|| anyhow::anyhow!("eval {e:?}: expected <name>=<replay dir>"))?;
            load_eval(name, dir)
        })
        .collect::<anyhow::Result<_>>()?;
    let mut stores: BTreeMap<String, ReplayStore> = BTreeMap::new();
    for (dir, _) in &phases {
        if !stores.contains_key(dir) {
            let store = ReplayStore::open(std::path::Path::new(dir))?;
            anyhow::ensure!(
                store.sampleable() > 0,
                "phase replay {dir} has no trainable positions"
            );
            stores.insert(dir.clone(), store);
        }
    }
    let total: usize = phases.iter().map(|p| p.1).sum();
    let lr = args.lr.unwrap_or(cfg.lr);
    let (mut model, mut optim, _) = load_training(
        &args.checkpoint,
        model_io::build::<B>(&cfg.model, &device)?,
        adamw::<B, _>(),
        &device,
    )?;

    let eval_all = |model: &recur64_model::model::ProbeModel<B>,
                    at: usize,
                    phase: &str,
                    loss: Option<(f32, f32)>|
     -> anyhow::Result<serde_json::Value> {
        let batched = BatchedModel::new(model.valid(), cfg.recurrence, device.clone());
        let sets = evals
            .iter()
            .map(|s| evaluate(&batched, s))
            .collect::<anyhow::Result<Vec<_>>>()?;
        for s in &sets {
            let vbm = s["value_by_material"].as_array().expect("array");
            println!(
                "update {at:>4} [{phase}] {:<14} ce {:.4} | win|lead 5-8 {:.3} 9+ {:.3} | draw 5-8 {:.3}",
                s["set"].as_str().unwrap_or(""),
                s["wdl_ce"].as_f64().unwrap_or(f64::NAN),
                vbm[2]["p_win_leader"].as_f64().unwrap_or(f64::NAN),
                vbm[3]["p_win_leader"].as_f64().unwrap_or(f64::NAN),
                vbm[2]["p_draw"].as_f64().unwrap_or(f64::NAN),
            );
        }
        Ok(serde_json::json!({
            "update": at, "phase": phase,
            "train_loss_first_last": loss, "evals": sets,
        }))
    };

    let mut curve = vec![eval_all(&model, 0, "start", None)?];
    let mut done = 0usize;
    for (dir, updates) in &phases {
        let store = &stores[dir];
        let mut left = *updates;
        while left > 0 {
            let chunk = left.min(args.eval_every.max(1));
            let learner = LearnerConfig {
                batch_size: cfg.train_batch,
                accumulation_steps: cfg.accumulation_steps,
                max_updates: chunk,
                lr,
                warmup_updates: 10,
                planned_updates: total as u64,
                start_update: done as u64,
                recurrence: cfg.recurrence,
                seed: cfg.seed.wrapping_add(done as u64),
                deadline: None,
                current_cycle_first_game_id: None,
                games_per_cycle: 0,
            };
            let (m, report) = train_from_store(store, model, &mut optim, &learner, &device)
                .map_err(|e| anyhow::anyhow!("training failed: {e}"))?;
            model = m;
            done += report.updates;
            left -= chunk;
            curve.push(eval_all(
                &model,
                done,
                dir,
                Some((report.first_loss, report.last_loss)),
            )?);
        }
    }
    std::fs::create_dir_all(&args.output)?;
    std::fs::write(
        args.output.join("value-diag.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "checkpoint": args.checkpoint,
            "phases": args.phases,
            "lr": lr,
            "effective_batch": cfg.train_batch * cfg.accumulation_steps,
            "git_revision": option_env!("RECUR64_GIT_SHA"),
            "curve": curve,
        }))?,
    )?;
    println!("wrote {}", args.output.join("value-diag.json").display());
    Ok(())
}

pub fn run(args: ValueDiagArgs) -> anyhow::Result<()> {
    let cfg = RunConfig::from_toml_str(&std::fs::read_to_string(&args.config)?)?;
    cfg.ensure_supported()?;
    match cfg.device.as_str() {
        "cpu" => run_impl::<CpuTrain>(&cfg, &args),
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                run_impl::<burn::backend::Autodiff<burn::backend::Cuda>>(&cfg, &args)
            }
            #[cfg(not(feature = "cuda"))]
            {
                anyhow::bail!("CUDA support is not compiled; rebuild with --features cuda")
            }
        }
        other => anyhow::bail!("unknown device '{other}'"),
    }
}
