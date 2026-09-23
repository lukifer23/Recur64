//! Recur64 Phase 0 CLI. Systems probe only: no chess rules, search, or runtime.

use std::path::PathBuf;

use clap::{Parser, Subcommand};

mod bench;
mod bench_core;
#[cfg(feature = "cuda")]
mod cuda_smoke;
mod doctor;
mod inspect;
mod model_info;
mod perft;
mod phase2;

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
    /// Count legal move tree nodes from a FEN (validates move generation).
    Perft(perft::PerftArgs),
    /// Print termination/rules/legal-candidate facts for a FEN.
    ValidatePosition(inspect::ValidateArgs),
    /// Encode a FEN as Observation V1 and list canonical legal actions.
    Encode(inspect::EncodeArgs),
    /// Phase 1 CPU baselines: movegen/apply/encode/perft throughput.
    BenchCore(bench_core::BenchCoreArgs),
    /// Generate self-play games with PUCT and write replay shards.
    Selfplay(phase2::SelfplayArgs),
    /// Verify a replay directory (checksums, legality, targets, provenance).
    ReplayAudit(phase2::ReplayAuditArgs),
    /// Train the Micro model from replay into a candidate checkpoint.
    Train(phase2::TrainArgs),
    /// Run a candidate-vs-reference systems arena.
    Arena(phase2::ArenaArgs),
    /// Bounded collect -> audit -> train -> evaluate -> report cycle.
    Run(phase2::RunArgs),
    /// Print a run's metadata and report.
    Report(phase2::ReportArgs),
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
        Commands::Perft(args) => perft::run_perft(args),
        Commands::ValidatePosition(args) => inspect::run_validate(args),
        Commands::Encode(args) => inspect::run_encode(args),
        Commands::BenchCore(args) => bench_core::run_bench_core(args),
        Commands::Selfplay(args) => phase2::run_selfplay(args),
        Commands::ReplayAudit(args) => phase2::run_replay_audit(args),
        Commands::Train(args) => phase2::run_train(args),
        Commands::Arena(args) => phase2::run_arena(args),
        Commands::Run(args) => phase2::run_run(args),
        Commands::Report(args) => phase2::run_report(args),
        #[cfg(feature = "cuda")]
        Commands::CudaSmoke(args) => cuda_smoke::run_cuda_smoke(args),
    }
}
