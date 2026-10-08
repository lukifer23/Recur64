//! Experiment master seed and deterministic stream derivation.
//!
//! `stream(label, index)` = SplitMix64 seeded with the first 8 bytes of
//! SHA-256("recur64-v69/stream/v1" || master(32 bytes) || len(label) || label || index_le).
//! Streams are disjoint by construction of the label. No seed search is
//! performed anywhere in this crate.

use sha2::{Digest, Sha256};

pub const STREAM_GENERATION: &str = "generation";
pub const STREAM_PARTITION: &str = "partition";
pub const STREAM_SELECTION: &str = "selection";
pub const STREAM_MODEL_INIT: &str = "model_init";
pub const STREAM_TRAIN_ORDER: &str = "train_order";
pub const STREAM_INTERVENTION: &str = "intervention";

#[derive(Clone, Copy, Debug)]
pub struct MasterSeed(pub [u8; 32]);

impl MasterSeed {
    pub fn from_hex(s: &str) -> anyhow::Result<Self> {
        let s = s.trim();
        anyhow::ensure!(s.len() == 64, "master seed must be 64 hex chars");
        let mut b = [0u8; 32];
        for i in 0..32 {
            b[i] = u8::from_str_radix(&s[2 * i..2 * i + 2], 16)?;
        }
        Ok(Self(b))
    }

    pub fn to_hex(&self) -> String {
        hex(&self.0)
    }

    /// Hash of the seed (what artifacts may cite without repeating the seed).
    pub fn fingerprint(&self) -> String {
        hex(&Sha256::digest(self.0))[..16].to_string()
    }

    pub fn stream(&self, label: &str, index: u64) -> SplitMix64 {
        let mut h = Sha256::new();
        h.update(b"recur64-v69/stream/v1");
        h.update(self.0);
        h.update((label.len() as u64).to_le_bytes());
        h.update(label.as_bytes());
        h.update(index.to_le_bytes());
        let d = h.finalize();
        SplitMix64(u64::from_le_bytes(d[..8].try_into().unwrap()))
    }
}

pub fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

#[derive(Clone, Debug)]
pub struct SplitMix64(u64);

impl SplitMix64 {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Unbiased value in `0..n` (rejection sampling).
    pub fn below(&mut self, n: u64) -> u64 {
        assert!(n > 0);
        let zone = u64::MAX - (u64::MAX % n);
        loop {
            let x = self.next_u64();
            if x < zone {
                return x % n;
            }
        }
    }
}

/// Stable 64-bit key from arbitrary bytes under a stream label (for
/// content-addressed, order-independent random keys).
pub fn keyed_u64(master: &MasterSeed, label: &str, content: &[u8]) -> u64 {
    let mut h = Sha256::new();
    h.update(b"recur64-v69/keyed/v1");
    h.update(master.0);
    h.update((label.len() as u64).to_le_bytes());
    h.update(label.as_bytes());
    h.update(content);
    let d = h.finalize();
    u64::from_le_bytes(d[..8].try_into().unwrap())
}
