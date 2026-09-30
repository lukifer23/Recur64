//! `recur64 x15 gen-targets | audit-targets` - the fixed-data reasoning set.
//!
//! The teacher is a Recur64 probe checkpoint searched with deterministic PUCT
//! (no root noise, one leaf in flight). Positions come from real self-play
//! replay games as `start_fen + exact prefix`, so repetition state is exact.

use std::path::PathBuf;
use std::time::Instant;

use burn::prelude::*;
use clap::Args;

use recur64_runtime::evaluator::SyncEvaluator;
use recur64_runtime::reasoning_targets::{
    self as rt, Provenance, ReasoningTargetsV1, TeacherContract, select_positions,
};
use recur64_runtime::replay::ReplayReader;
use recur64_runtime::{RunConfig, model_io};

#[derive(Args, Debug)]
pub struct GenTargetsArgs {
    /// RunConfig of the teacher's run (model geometry, recurrence, device).
    #[arg(long)]
    pub teacher_config: PathBuf,
    /// Teacher checkpoint directory.
    #[arg(long)]
    pub teacher_checkpoint: PathBuf,
    /// Replay directory to draw exact-history positions from.
    #[arg(long)]
    pub replay: PathBuf,
    #[arg(long)]
    pub out: PathBuf,
    #[arg(long, default_value_t = 32)]
    pub positions: usize,
    /// Simulation ladder, shallowest first (last = deep teacher).
    #[arg(long, value_delimiter = ',', default_value = "16,32,64,128")]
    pub ladder: Vec<u32>,
    #[arg(long, default_value_t = 20260929)]
    pub seed: u64,
    #[arg(long, default_value_t = 1.0)]
    pub c_puct: f32,
    /// Worker threads sharing the one teacher model (labels are identical for any count).
    #[arg(long, default_value_t = 4)]
    pub threads: usize,
    /// Exclude every replay game that any of these targets files drew from.
    #[arg(long)]
    pub exclude_targets: Vec<PathBuf>,
    /// Label every position with this split instead of the per-game train/val hash.
    #[arg(long)]
    pub split_label: Option<String>,
    /// Only exclude source games of these splits from each excluded targets file
    /// (default: every split).
    #[arg(long)]
    pub exclude_only_splits: Vec<String>,
    /// Exclude synthetic positions whose FEN appears in these fixtures files.
    #[arg(long)]
    pub exclude_fens: Vec<PathBuf>,
    /// Synthetic tactic positions per kind (7 kinds) added as training data.
    #[arg(long, default_value_t = 0)]
    pub synthetic_tactics: usize,
    #[arg(long, default_value_t = 20261002)]
    pub tactics_seed: u64,
    /// Serve the teacher through the shared batched inference owner.
    #[arg(long, default_value_t = false)]
    pub batched: bool,
    #[arg(long, default_value_t = 48)]
    pub max_batch: usize,
    #[arg(long, default_value_t = 1)]
    pub owners: usize,
}

#[derive(Args, Debug)]
pub struct AuditTargetsArgs {
    #[arg(long)]
    pub targets: PathBuf,
    /// Also require zero shared source games with each of these targets files.
    #[arg(long)]
    pub disjoint_from: Vec<PathBuf>,
    /// Print the ladder-informativeness report.
    #[arg(long, default_value_t = false)]
    pub report: bool,
    /// Compare against another targets file over the shared position ids.
    #[arg(long)]
    pub compare: Option<PathBuf>,
}

fn generate<B: Backend>(cfg: &RunConfig, args: &GenTargetsArgs) -> anyhow::Result<()> {
    anyhow::ensure!(
        args.ladder.windows(2).all(|w| w[0] < w[1]) && !args.ladder.is_empty(),
        "ladder must be strictly increasing"
    );
    let device: B::Device = Default::default();
    let model_id = std::fs::read(args.teacher_checkpoint.join("meta.json"))
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .and_then(|v| {
            v.get("model_id")
                .and_then(|m| m.as_str())
                .map(str::to_owned)
        })
        .ok_or_else(|| anyhow::anyhow!("teacher checkpoint has no model_id"))?;

    // --- candidates: replay positions (exact history) + synthetic tactics ---
    let mut candidates = Vec::new();
    let mut games_left = 0usize;
    let mut excluded: Vec<ReasoningTargetsV1> = Vec::new();
    if args.positions > 0 {
        let mut games = ReplayReader::open(&args.replay)?.read_all_games()?;
        anyhow::ensure!(!games.is_empty(), "replay has no games");
        let mut exclude_ids = std::collections::BTreeSet::new();
        for p in &args.exclude_targets {
            let mut t = ReasoningTargetsV1::load(p)?;
            if !args.exclude_only_splits.is_empty() {
                t.positions
                    .retain(|q| args.exclude_only_splits.contains(&q.split));
            }
            exclude_ids.extend(rt::source_game_ids(&t));
            excluded.push(t);
        }
        let before = games.len();
        games = rt::exclude_games(games, &exclude_ids);
        println!(
            "excluded {} source games ({} of {} replay games removed); {} remain",
            exclude_ids.len(),
            before - games.len(),
            before,
            games.len()
        );
        anyhow::ensure!(!games.is_empty(), "no replay games left after exclusion");
        games_left = games.len();
        candidates.extend(select_positions(&games, args.positions, args.seed)?);
    }
    let mut synth = 0usize;
    if args.synthetic_tactics > 0 {
        let mut exclude = std::collections::HashSet::new();
        for p in &args.exclude_fens {
            exclude.extend(crate::x15_tactics::fixture_fens(p)?);
        }
        let fens =
            crate::x15_tactics::synth_fens(args.tactics_seed, args.synthetic_tactics, &exclude)?;
        // Hard check: nothing from the evaluation suite may be in the training mix.
        anyhow::ensure!(
            fens.iter().all(|(_, f)| !exclude.contains(f)),
            "synthetic tactic overlaps an excluded FEN"
        );
        synth = fens.len();
        for (i, (_, fen)) in fens.into_iter().enumerate() {
            candidates.push(rt::Candidate {
                game_id: 9_000_000 + i as u64,
                ply: 0,
                start_fen: fen,
                prefix: Vec::new(),
                category: "tactic_synth",
            });
        }
    }
    anyhow::ensure!(!candidates.is_empty(), "no candidates requested");

    let mut teacher = TeacherContract {
        checkpoint: args.teacher_checkpoint.display().to_string(),
        model_id,
        architecture: "probe_v1".into(),
        recurrence: cfg.recurrence,
        c_puct: args.c_puct,
        leaves_in_flight: 1,
        root_noise: false,
        ladder: args.ladder.clone(),
        evaluator: String::new(),
    };
    println!(
        "teacher {} (model_id {}...), ladder {:?}, {} candidates ({} replay games left, {} synthetic tactics), evaluator {}",
        teacher.checkpoint,
        &teacher.model_id[..12.min(teacher.model_id.len())],
        teacher.ladder,
        candidates.len(),
        games_left,
        synth,
        if args.batched { "batched" } else { "sync" }
    );
    let start = Instant::now();
    let progress = |done: usize, total: usize| {
        if done.is_multiple_of(25) || done == total {
            println!(
                "  labelled {done}/{total} ({:.0}s)",
                start.elapsed().as_secs_f64()
            );
        }
    };
    let positions = if args.batched {
        // One shared, batched inference owner (the pilot's serving path); the
        // worker threads submit leaves and the owner batches them.
        let mut owner_cfg = cfg.clone();
        owner_cfg.max_inference_batch = args.max_batch;
        owner_cfg.inference_owners = args.owners.max(1);
        teacher.evaluator = format!(
            "batched_owner_v1:max_batch={},owners={}",
            args.max_batch, owner_cfg.inference_owners
        );
        let owner = recur64_runtime::pilot::spawn_selfplay_owner::<B>(
            &args.teacher_checkpoint,
            &owner_cfg,
            &device,
        )?;
        let evaluator = owner.evaluator();
        let result = rt::label(
            candidates,
            &evaluator,
            &teacher,
            args.seed,
            args.threads,
            &progress,
        );
        let m = owner.metrics().snapshot();
        drop(evaluator);
        owner.shutdown();
        println!("  inference: {m:?}");
        result?
    } else {
        let model = model_io::load::<B>(&args.teacher_checkpoint, &cfg.model, &device)?;
        let evaluator = SyncEvaluator::new(model, cfg.recurrence, device);
        rt::label(
            candidates,
            &evaluator,
            &teacher,
            args.seed,
            args.threads,
            &progress,
        )?
    };
    let provenance = Provenance {
        git_rev: recur64_runtime::provenance::git_revision()
            .unwrap_or("unknown")
            .into(),
        created_unix_s: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
    };
    let mut positions = positions;
    if let Some(label) = &args.split_label {
        for p in &mut positions {
            p.split = label.clone();
        }
    }
    let targets = ReasoningTargetsV1::new(teacher, args.seed, provenance, positions);
    // Hard check, not a filter: no shared source game with the excluded games
    // (synthetic positions have no source game and are checked by FEN above).
    let mut replay_only = targets.clone();
    replay_only
        .positions
        .retain(|p| p.source_game_id < 9_000_000);
    for other in &excluded {
        rt::ensure_disjoint(&replay_only, other)?;
    }
    if !excluded.is_empty() {
        println!(
            "verified disjoint from {} excluded targets file(s) at the source-game level",
            excluded.len()
        );
    }
    let n = rt::audit(&targets)?;
    targets.save(&args.out)?;
    let mut cats: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    for p in &targets.positions {
        *cats.entry(p.category.clone()).or_insert(0) += 1;
    }
    println!(
        "wrote {} ({} positions audited move-for-move); digest {}",
        args.out.display(),
        n,
        targets.digest
    );
    println!("  categories: {cats:?}");
    Ok(())
}

pub fn run_gen(args: GenTargetsArgs) -> anyhow::Result<()> {
    let cfg = RunConfig::from_toml_str(&std::fs::read_to_string(&args.teacher_config)?)?;
    cfg.ensure_supported()?;
    match cfg.device.as_str() {
        "cpu" => generate::<burn::backend::Flex>(&cfg, &args),
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                // In-process scoped sampler (joined on exit): the RTX's own
                // utilization, so "is the GPU actually working" is a number.
                let (result, gpu) = recur64_runtime::gpu_telemetry::monitor(true, || {
                    generate::<burn::backend::Cuda>(&cfg, &args)
                });
                println!(
                    "gpu telemetry   : peak_vram={:?} MiB util_max={:?}% util_busy_mean={:?} temp_max={:?} C ({} samples)",
                    gpu.peak_vram_mb, gpu.util_max, gpu.util_busy_mean, gpu.temp_max_c, gpu.samples
                );
                result
            }
            #[cfg(not(feature = "cuda"))]
            {
                anyhow::bail!("CUDA support is not compiled; rebuild with --features cuda")
            }
        }
        other => anyhow::bail!("unsupported device {other:?}"),
    }
}

pub fn run_audit(args: AuditTargetsArgs) -> anyhow::Result<()> {
    let targets = ReasoningTargetsV1::load(&args.targets)?;
    let n = rt::audit(&targets)?;
    println!(
        "OK: {n} positions reconstruct move-for-move; digest {}",
        targets.digest
    );
    for other in &args.disjoint_from {
        let o = ReasoningTargetsV1::load(other)?;
        rt::ensure_disjoint(&targets, &o)?;
        println!("OK: zero shared source games with {}", other.display());
    }
    if let Some(other) = &args.compare {
        let o = ReasoningTargetsV1::load(other)?;
        let map: std::collections::HashMap<_, _> =
            o.positions.iter().map(|p| (p.id.clone(), p)).collect();
        let mut shared = 0usize;
        let rungs = targets.teacher.ladder.len();
        let mut same_best = vec![0usize; rungs];
        let mut js = vec![0.0f32; rungs];
        for p in &targets.positions {
            if let Some(q) = map.get(&p.id) {
                shared += 1;
                for r in 0..rungs.min(q.rungs.len()) {
                    same_best[r] += usize::from(p.rungs[r].best == q.rungs[r].best);
                    js[r] += rt::js_divergence(&p.rungs[r].policy, &q.rungs[r].policy);
                }
            }
        }
        println!("compare vs {}: {shared} shared positions", other.display());
        for r in 0..rungs {
            println!(
                "  rung {} sims: same best move {}/{}  mean JS {:.5}",
                targets.teacher.ladder[r],
                same_best[r],
                shared,
                js[r] / shared.max(1) as f32
            );
        }
    }
    if args.report {
        let all: Vec<_> = targets.positions.iter().collect();
        println!(
            "ladder report (all): {}",
            serde_json::to_string_pretty(&rt::ladder_report(&all))?
        );
        let mut splits: Vec<String> = targets.positions.iter().map(|p| p.split.clone()).collect();
        splits.sort();
        splits.dedup();
        for s in splits {
            let sub: Vec<_> = targets.positions.iter().filter(|p| p.split == s).collect();
            println!(
                "ladder report ({s}): {}",
                serde_json::to_string_pretty(&rt::ladder_report(&sub))?
            );
        }
    }
    Ok(())
}
