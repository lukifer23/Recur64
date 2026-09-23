//! Versioned, checksummed, atomically-published replay shards.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use super::schema::{
    GameRecord, Manifest, REPLAY_SCHEMA_VERSION, ReplayHeader, SHARD_MAGIC, Shard, ShardInfo,
};

fn bincode_config() -> bincode::config::Configuration {
    bincode::config::standard()
}

/// Writes replay games into bounded shards under a directory.
pub struct ReplayWriter {
    dir: PathBuf,
    header: ReplayHeader,
    shard_max_games: usize,
    buffer: Vec<GameRecord>,
    shard_index: u32,
    shards: Vec<ShardInfo>,
    games_written: u64,
    bytes_written: u64,
}

impl ReplayWriter {
    /// Create a writer. Refuses to reuse a directory that already has a manifest.
    pub fn new(dir: &Path, header: ReplayHeader, shard_max_games: usize) -> anyhow::Result<Self> {
        anyhow::ensure!(shard_max_games >= 1, "shard_max_games must be >= 1");
        fs::create_dir_all(dir)?;
        anyhow::ensure!(
            !dir.join("manifest.json").exists(),
            "replay directory already contains a manifest: {}",
            dir.display()
        );
        Ok(Self {
            dir: dir.to_path_buf(),
            header,
            shard_max_games,
            buffer: Vec::new(),
            shard_index: 0,
            shards: Vec::new(),
            games_written: 0,
            bytes_written: 0,
        })
    }

    /// Buffer a game, flushing a shard when the buffer is full.
    pub fn push(&mut self, game: GameRecord) -> anyhow::Result<()> {
        self.buffer.push(game);
        if self.buffer.len() >= self.shard_max_games {
            self.flush_shard()?;
        }
        Ok(())
    }

    pub fn games_written(&self) -> u64 {
        self.games_written
    }

    /// Write the current buffer as a committed shard (atomic temp + rename).
    pub fn flush_shard(&mut self) -> anyhow::Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        let games = std::mem::take(&mut self.buffer);
        let shard = Shard {
            header: self.header.clone(),
            games,
        };
        let payload = bincode::serde::encode_to_vec(&shard, bincode_config())?;
        let crc = crc32fast::hash(&payload);

        let mut bytes = Vec::with_capacity(payload.len() + 16);
        bytes.extend_from_slice(&SHARD_MAGIC);
        bytes.extend_from_slice(&REPLAY_SCHEMA_VERSION.to_le_bytes());
        bytes.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&crc.to_le_bytes());
        bytes.extend_from_slice(&payload);

        let name = format!("shard-{:06}.r64shard", self.shard_index);
        let tmp = self.dir.join(format!("{name}.tmp"));
        let final_path = self.dir.join(&name);
        {
            let mut f = fs::File::create(&tmp)?;
            f.write_all(&bytes)?;
            f.sync_all()?;
        }
        fs::rename(&tmp, &final_path)?;

        self.shards.push(ShardInfo {
            file: name,
            games: shard.games.len() as u64,
            bytes: bytes.len() as u64,
            crc32: crc,
        });
        self.games_written += shard.games.len() as u64;
        self.bytes_written += bytes.len() as u64;
        self.shard_index += 1;
        Ok(())
    }

    /// Flush remaining games and atomically publish the manifest.
    pub fn finish(mut self) -> anyhow::Result<Manifest> {
        self.flush_shard()?;
        let manifest = Manifest {
            header: self.header.clone(),
            shards: self.shards.clone(),
            games: self.games_written,
            bytes: self.bytes_written,
        };
        write_manifest_atomic(&self.dir, &manifest)?;
        Ok(manifest)
    }
}

/// Write the manifest atomically (temp + rename).
pub fn write_manifest_atomic(dir: &Path, manifest: &Manifest) -> anyhow::Result<()> {
    let tmp = dir.join("manifest.json.tmp");
    let final_path = dir.join("manifest.json");
    let bytes = serde_json::to_vec_pretty(manifest)?;
    {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(&bytes)?;
        f.sync_all()?;
    }
    fs::rename(&tmp, &final_path)?;
    Ok(())
}
