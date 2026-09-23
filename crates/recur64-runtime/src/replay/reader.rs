//! Replay reader and shard parser.
//!
//! Parsing validates magic, schema version, declared length, and CRC before
//! decoding. A corrupt or truncated shard fails visibly.

use std::path::{Path, PathBuf};

use super::schema::{GameRecord, Manifest, REPLAY_SCHEMA_VERSION, SHARD_MAGIC, Shard};

/// Parse a shard from raw bytes, validating integrity.
pub fn parse_shard_bytes(bytes: &[u8]) -> Result<Shard, String> {
    if bytes.len() < 16 {
        return Err(format!("shard too short: {} bytes", bytes.len()));
    }
    if bytes[0..4] != SHARD_MAGIC {
        return Err("bad shard magic".into());
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    if version != REPLAY_SCHEMA_VERSION {
        return Err(format!("shard schema {version} != {REPLAY_SCHEMA_VERSION}"));
    }
    let payload_len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    let crc = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
    if bytes.len() != 16 + payload_len {
        return Err(format!(
            "shard length mismatch: header says {payload_len}, file has {}",
            bytes.len() - 16
        ));
    }
    let payload = &bytes[16..];
    let actual = crc32fast::hash(payload);
    if actual != crc {
        return Err(format!("checksum mismatch: {actual:#010x} != {crc:#010x}"));
    }
    let (shard, consumed): (Shard, usize) =
        bincode::serde::decode_from_slice(payload, bincode::config::standard())
            .map_err(|e| format!("decode error: {e}"))?;
    if consumed != payload.len() {
        return Err(format!(
            "trailing bytes: consumed {consumed} of {}",
            payload.len()
        ));
    }
    shard
        .header
        .check_contracts()
        .map_err(|e| format!("shard header contract error: {e}"))?;
    Ok(shard)
}

/// Reads a replay directory.
pub struct ReplayReader {
    dir: PathBuf,
    manifest: Manifest,
}

impl ReplayReader {
    pub fn open(dir: &Path) -> anyhow::Result<Self> {
        let manifest_path = dir.join("manifest.json");
        let manifest: Manifest = serde_json::from_slice(&std::fs::read(&manifest_path)?)?;
        manifest
            .header
            .check_contracts()
            .map_err(|e| anyhow::anyhow!("manifest contract error: {e}"))?;
        Ok(Self {
            dir: dir.to_path_buf(),
            manifest,
        })
    }

    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    pub fn read_shard(&self, file: &str) -> anyhow::Result<Shard> {
        let bytes = std::fs::read(self.dir.join(file))?;
        parse_shard_bytes(&bytes).map_err(|e| anyhow::anyhow!("{file}: {e}"))
    }

    /// Read every game in manifest order.
    pub fn read_all_games(&self) -> anyhow::Result<Vec<GameRecord>> {
        let mut out = Vec::new();
        for info in &self.manifest.shards {
            let shard = self.read_shard(&info.file)?;
            out.extend(shard.games);
        }
        Ok(out)
    }
}
