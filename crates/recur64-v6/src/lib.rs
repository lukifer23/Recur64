//! V6 P0: full-information branch initialization and shared learned backup.
pub mod intervene;
pub mod loss;
pub mod model;
pub mod objective;
pub mod p0;
pub mod packet;
pub mod probe_eval;
pub mod probe_qual;
pub mod qualify;
pub const SOURCE: &str = env!("RECUR64_V6_BUILD_SOURCE_SHA");
pub const ARCHITECTURE: &str = "v6_branch_full_information_backup_v1";
pub fn digest<T: serde::Serialize>(x: &T) -> anyhow::Result<String> {
    use sha2::{Digest, Sha256};
    Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(x)?)))
}
pub mod baseline;
mod encoder;

#[cfg(test)]
pub(crate) static SEEDED_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
