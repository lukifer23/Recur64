//! Replay V1: schema, writer, reader, audit.

pub mod audit;
pub mod reader;
pub mod sampler;
pub mod schema;
pub mod writer;

pub use audit::{AuditReport, audit_dir, audit_games};
pub use reader::{ReplayReader, parse_shard_bytes};
pub use sampler::{
    CapacityReport, ReplayStore, TrainingExample, enforce_capacity, example_for_ply, wdl_class,
};
pub use schema::{
    GameRecord, Manifest, PlyRecord, REPLAY_SCHEMA_VERSION, ReplayHeader, SHARD_MAGIC,
    SearchRecord, Shard, ShardInfo,
};
pub use writer::{ReplayWriter, write_manifest_atomic};
