//! Recur64 Phase 0 CLI. Systems probe only: no chess rules, search, or runtime.

use std::path::PathBuf;

use clap::{Parser, Subcommand};

mod bench;
#[cfg(feature = "cuda")]
mod cuda_smoke;
mod doctor;
mod model_info;

#[derive(Parser)]
#[command(
    name = "recur64",
    version,
    about = "Recur64 Phase 0 probe CLI (systems test only; not a chess engine)"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Read-only environment report (DETECTED vs TESTED).
    Doctor,
    /// Exact parameter counts and executed-block accounting for a config.
    ModelInfo {
        #[arg(long)]
        config: PathBuf,
    },
    /// Bounded benchmark matrix; writes raw data to --output.
    Bench(bench::BenchArgs),
    /// GPU backend proof: forward/backward/AdamW/checkpoint on the CUDA device.
    #[cfg(feature = "cuda")]
    CudaSmoke(cuda_smoke::CudaSmokeArgs),
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Doctor => doctor::run_doctor(),
        Commands::ModelInfo { config } => model_info::run_model_info(&config),
        Commands::Bench(args) => bench::run_bench(args),
        #[cfg(feature = "cuda")]
        Commands::CudaSmoke(args) => cuda_smoke::run_cuda_smoke(args),
    }
}
