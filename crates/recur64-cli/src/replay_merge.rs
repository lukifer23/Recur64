//! `recur64 replay-merge`: concatenate replays into one fixed dataset (single
//! shard by default, so recency-weighted samplers treat every game equally).
//! Game ids are renumbered 0..n in input order; games are otherwise copied
//! unchanged. Used to build larger fixed training sets for the fast loop.

use std::path::PathBuf;

use clap::Args;

use recur64_runtime::replay::{ReplayHeader, ReplayReader, ReplayWriter};

#[derive(Args, Debug)]
pub struct ReplayMergeArgs {
    /// Comma-separated input replay directories.
    #[arg(long, value_delimiter = ',')]
    pub inputs: Vec<PathBuf>,
    #[arg(long)]
    pub output: PathBuf,
    /// Games per output shard (default: all in one shard).
    #[arg(long)]
    pub shard_max_games: Option<usize>,
}

pub fn run(args: ReplayMergeArgs) -> anyhow::Result<()> {
    anyhow::ensure!(!args.inputs.is_empty(), "no --inputs");
    anyhow::ensure!(
        !args.output.exists(),
        "output {} exists",
        args.output.display()
    );
    let mut games = Vec::new();
    let mut sources = Vec::new();
    for dir in &args.inputs {
        let reader = ReplayReader::open(dir)?;
        let g = reader.read_all_games()?;
        sources.push(serde_json::json!({
            "dir": dir.display().to_string(),
            "model_id": reader.manifest().header.model_id,
            "games": g.len(),
        }));
        games.extend(g);
    }
    let first = ReplayReader::open(&args.inputs[0])?;
    let h = &first.manifest().header;
    let header = ReplayHeader::new(
        format!("replay-merge:{}", args.inputs.len()),
        format!("merged:{}", h.model_id),
        h.backend.clone(),
        h.precision.clone(),
    );
    let n = games.len();
    let mut w = ReplayWriter::new(
        &args.output,
        header,
        args.shard_max_games.unwrap_or(n.max(1)),
    )?;
    for (i, mut g) in games.into_iter().enumerate() {
        g.game_id = i as u64;
        w.push(g)?;
    }
    let manifest = w.finish()?;
    std::fs::write(
        args.output.join("merge-sources.json"),
        serde_json::to_vec_pretty(&serde_json::json!({ "sources": sources, "games": n }))?,
    )?;
    println!(
        "merged {n} games into {} shard(s) at {}",
        manifest.shards.len(),
        args.output.display()
    );
    Ok(())
}
