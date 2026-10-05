//! Measured V5 heavy-family V2 role-aware data and exact local-byte custody.
use crate::native_data_v2::{CONTRACT, Dataset, Manifest, Role};
use anyhow::Context;
use recur64_runtime::proof::targets::{ProofPosition, ProofTargets};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::path::Path;

pub const TRAIN_POSITIONS: usize = 27_000;
pub const FIT_POSITIONS: usize = TRAIN_POSITIONS;
pub const DEV_POSITIONS: usize = 4_500;
pub const GENERATOR_SOURCE: &str = "738db983084a998664ce962f87fa3c4a6153f526";
pub const TRAIN_DIGEST: &str = "d7918b7a138b9aab17de24842dd5bf511cb0b95aeccbcbf07cb48921a81d86b9";
pub const TRAIN_CONTENT_DIGEST: &str =
    "7ca17f824d24a629d8b54e0a2be48590f6cf78e2bd42d6c0d5db62ddeadf2867";
pub const TRAIN_FEN_DIGEST: &str =
    "b7ebc90955fd8959c8e7266d1afac7a913c94959f3bca7a353b80e9cbc30f61a";
pub const TRAIN_CANONICAL_DIGEST: &str =
    "6170bee242a7af46d4b3959f34d52a01e189c3a6b7d06f85925673e218c5f987";
const TRAIN_RAW_SHA: &str = "ab34a0d098c5c2a1985f76967c62cd803b04419b31cda41166e819d389b77823";
pub const DEV_DIGEST: &str = "82d4578a62d0727ebca51f412d45e2dcf4e9461838f83148aa70d98b0a69d2a0";
pub const DEV_CONTENT_DIGEST: &str =
    "aac1be9d74af801ea2fb32e88a1f3eb9e4e93062f6a49ada8512b953c91847c5";
pub const DEV_FEN_DIGEST: &str = "4c2f3bb3bd2b594ee0921974a151ad71a19e149457b59c9061977fffd96f0e06";
pub const DEV_CANONICAL_DIGEST: &str =
    "1201ea0c6dc8789b1d9cf0212bfb088590b4473948e28355c9ecd0523d4e37ec";
const DEV_RAW_SHA: &str = "15a645f4dc94e263ec55009989f4f052cf4fadebed27f3d824e15d73f6ca75ec";
pub const CONFIRM_DIGEST: &str = "e4d2831d5c8c9ec4119b4b584eb10430be56f8e8a29db2cb437d40bf6294fc1b";
pub const CONFIRM_CONTENT_DIGEST: &str =
    "6a4e355fdb38d46cefb8b25974938432dd2979b893c788a537ada9e1b60aa4f0";
pub const CONFIRM_FEN_DIGEST: &str =
    "6f3cb6f27932dee45c328cc17531bd99d0bd6d98063c18d8e1ab30bf1fad084c";
pub const CONFIRM_CANONICAL_DIGEST: &str =
    "2860b55ad1c581f90691a52907fed779bd68a6f820bda83a264f73cfbbe37be4";
const CONFIRM_RAW_SHA: &str = "825fae2b6da27678597e46527dc2a729abad7c648637f0b9113031215c1de9bf";
pub const FIT_DIGEST: &str = "2adc94d4725acd34959578fd31c23af852a4fe3f10f56aa1259dbcb55dcaa726";

#[derive(Deserialize)]
pub struct Binding {
    pub schema: String,
    pub raw_sha256: String,
    pub record_id_digest: String,
    pub manifest: Manifest,
}
fn binding_text(role: Role) -> &'static str {
    match role {
        Role::Train => include_str!("../../../docs/evidence/v5/data/v2/train-binding.json"),
        Role::Dev => include_str!("../../../docs/evidence/v5/data/v2/dev-binding.json"),
        Role::Confirm => include_str!("../../../docs/evidence/v5/data/v2/confirm-binding.json"),
    }
}
fn require_scientific_role(actual: Role, requested: Role) -> anyhow::Result<()> {
    anyhow::ensure!(
        requested != Role::Confirm && actual == requested,
        "ordinary V5 commands refuse CONFIRM and wrong TRAIN/DEV roles"
    );
    Ok(())
}
pub fn binding(role: Role) -> anyhow::Result<Binding> {
    let b: Binding = serde_json::from_str(binding_text(role))?;
    let (target, content, fen, canon, raw) = match role {
        Role::Train => (
            TRAIN_DIGEST,
            TRAIN_CONTENT_DIGEST,
            TRAIN_FEN_DIGEST,
            TRAIN_CANONICAL_DIGEST,
            TRAIN_RAW_SHA,
        ),
        Role::Dev => (
            DEV_DIGEST,
            DEV_CONTENT_DIGEST,
            DEV_FEN_DIGEST,
            DEV_CANONICAL_DIGEST,
            DEV_RAW_SHA,
        ),
        Role::Confirm => (
            CONFIRM_DIGEST,
            CONFIRM_CONTENT_DIGEST,
            CONFIRM_FEN_DIGEST,
            CONFIRM_CANONICAL_DIGEST,
            CONFIRM_RAW_SHA,
        ),
    };
    anyhow::ensure!(
        b.schema == "v5_hp_measured_binding_v2"
            && b.raw_sha256 == raw
            && b.manifest.role == role
            && b.manifest.identity == role.identity()
            && b.manifest.source_sha == GENERATOR_SOURCE
            && b.manifest.data_contract == CONTRACT
            && b.manifest.target_digest == target
            && b.manifest.content_digest == content
            && b.manifest.fen_set_digest == fen
            && b.manifest.canonical_set_digest == canon
            && b.manifest.config_digest
                == crate::config::V5Config::default().scientific_digest()?,
        "committed measured binding differs from scientific constants"
    );
    Ok(b)
}
pub fn verify_preregistered_bindings() -> anyhow::Result<()> {
    for role in [Role::Train, Role::Dev, Role::Confirm] {
        binding(role)?;
    }
    Ok(())
}
fn verify_raw(bytes: &[u8], expected: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        format!("{:x}", Sha256::digest(bytes)) == expected,
        "local raw bytes differ from committed measured digest"
    );
    Ok(())
}
fn verify_manifest(actual: &Manifest, expected: &Manifest, role: Role) -> anyhow::Result<()> {
    anyhow::ensure!(
        actual == expected && actual.role == role,
        "wrong split or tampered source/config/content/FEN/canonical manifest"
    );
    Ok(())
}
fn load_custody(path: &Path, role: Role) -> anyhow::Result<Dataset> {
    let bytes = std::fs::read(path)?;
    let data: Dataset = serde_json::from_slice(&bytes).context(
        "retired/unbound inputs: loaders are locked to the measured V5 native V2 artifact",
    )?;
    let expected = binding(role)?;
    anyhow::ensure!(
        data.manifest.role == role,
        "wrong split role; filenames never establish identity"
    );
    verify_raw(&bytes, &expected.raw_sha256)?;
    verify_manifest(&data.manifest, &expected.manifest, role)?;
    data.custody(&expected.manifest, role)?;
    if role == Role::Confirm {
        let seal: serde_json::Value = serde_json::from_str(include_str!(
            "../../../docs/evidence/v5/data/v2/confirm-seal.json"
        ))?;
        anyhow::ensure!(
            seal["schema"] == "v5_hp_confirmation_seal_v2"
                && seal["sealed"] == true
                && seal["evaluated"] == false
                && seal["raw_sha256"] == expected.raw_sha256
                && serde_json::from_value::<Manifest>(seal["manifest"].clone())?
                    == expected.manifest,
            "confirmation seal mismatch"
        );
    }
    Ok(data)
}
/// Custody exposes metadata only; it is not a confirmation evaluation capability.
pub fn custody(path: &Path, role: Role) -> anyhow::Result<Manifest> {
    Ok(load_custody(path, role)?.manifest)
}
pub fn lineage_custody(
    train: &Path,
    dev: &Path,
    confirm: &Path,
) -> anyhow::Result<serde_json::Value> {
    let a = load_custody(train, Role::Train)?;
    let b = load_custody(dev, Role::Dev)?;
    let c = load_custody(confirm, Role::Confirm)?;
    let overlaps = vec![
        crate::native_data_v2::overlap(&a, &b),
        crate::native_data_v2::overlap(&a, &c),
        crate::native_data_v2::overlap(&b, &c),
    ];
    anyhow::ensure!(
        overlaps
            .iter()
            .all(|o| o.exact_fen == 0 && o.canonical == 0),
        "local split overlap"
    );
    Ok(
        serde_json::json!({"schema":"v5_hp_lineage_custody_v2","data_contract":CONTRACT,"train":a.manifest,"dev":b.manifest,"confirm":c.manifest,"pairwise":overlaps,"pass":true,"sealed_inputs_evaluated":false}),
    )
}

/// Engineering-only tests against the actual local artifacts. No model exists.
pub fn verify_local_boundaries(
    train: &Path,
    dev: &Path,
    confirm: &Path,
) -> anyhow::Result<serde_json::Value> {
    let custody = lineage_custody(train, dev, confirm)?;
    let refusals = [
        ("TRAIN refuses DEV", V5Data::load_train(dev).is_err()),
        (
            "TRAIN refuses CONFIRM",
            V5Data::load_train(confirm).is_err(),
        ),
        ("DEV refuses TRAIN", V5Data::load_dev(train).is_err()),
        ("DEV refuses CONFIRM", V5Data::load_dev(confirm).is_err()),
    ];
    anyhow::ensure!(
        refusals.iter().all(|(_, refused)| *refused),
        "scientific role refusal failed"
    );
    let original = std::fs::read(train)?;
    let mut changed = original.clone();
    changed.push(b'\n');
    anyhow::ensure!(
        verify_raw(&changed, TRAIN_RAW_SHA).is_err(),
        "changed local bytes accepted"
    );
    let expected = binding(Role::Train)?;
    let parsed: Dataset = serde_json::from_slice(&original)?;
    let mut mutations = Vec::new();
    for field in [
        "source",
        "config",
        "content",
        "fen",
        "canonical",
        "count",
        "cell_count",
        "audit",
    ] {
        let mut m = parsed.manifest.clone();
        match field {
            "source" => m.source_sha = "0".repeat(40),
            "config" => m.config_digest = "wrong".into(),
            "content" => m.content_digest = "wrong".into(),
            "fen" => m.fen_set_digest = "wrong".into(),
            "canonical" => m.canonical_set_digest = "wrong".into(),
            "count" => m.records -= 1,
            "cell_count" => {
                m.cell_counts.insert("KQQvK M1".into(), 2999);
            }
            _ => m.audit_failures = 1,
        }
        let refused = verify_manifest(&m, &expected.manifest, Role::Train).is_err();
        anyhow::ensure!(refused, "manifest mutation accepted: {field}");
        mutations.push((field, refused));
    }
    Ok(
        serde_json::json!({"schema":"v5_hp_actual_boundary_tests_v2","custody":custody,"role_refusals":refusals,"manifest_mutations_refused":mutations,"changed_raw_bytes_refused":true,"ordinary_confirmation_access":false,"models_initialized":0,"pass":true}),
    )
}
pub struct V5Data {
    pub targets: ProofTargets,
    /// All TRAIN indices; retained field name is compatibility, not a partition.
    pub fit: Vec<usize>,
    /// Independent DEV indices, present only in a DEV instance.
    pub dev: Vec<usize>,
    role: Role,
}
impl V5Data {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        Self::load_train(path)
    }
    pub fn load_train(path: &Path) -> anyhow::Result<Self> {
        Self::load_role(path, Role::Train)
    }
    pub fn load_dev(path: &Path) -> anyhow::Result<Self> {
        Self::load_role(path, Role::Dev)
    }
    fn load_role(path: &Path, role: Role) -> anyhow::Result<Self> {
        let data = load_custody(path, role)?;
        data.scientific_access(role)?;
        let indices = (0..data.targets.positions.len()).collect();
        let (fit, dev) = if role == Role::Train {
            (indices, vec![])
        } else {
            (vec![], indices)
        };
        Ok(Self {
            targets: data.targets,
            fit,
            dev,
            role,
        })
    }
    pub fn require_role(&self, role: Role) -> anyhow::Result<()> {
        require_scientific_role(self.role, role)
    }
    pub fn verify_custody(&self) -> anyhow::Result<()> {
        let expected = binding(self.role)?;
        let data = Dataset {
            schema: crate::native_data_v2::RAW_SCHEMA.into(),
            manifest: expected.manifest.clone(),
            targets: self.targets.clone(),
        };
        data.custody(&expected.manifest, self.role)?;
        let expected_indices: Vec<_> = (0..self.targets.positions.len()).collect();
        anyhow::ensure!(
            if self.role == Role::Train {
                self.fit == expected_indices && self.dev.is_empty()
            } else {
                self.dev == expected_indices && self.fit.is_empty()
            },
            "index membership differs from role"
        );
        Ok(())
    }
    pub fn position(&self, index: usize) -> &ProofPosition {
        &self.targets.positions[index]
    }

    pub fn roots(&self, indices: &[usize]) -> anyhow::Result<Vec<recur64_core::GameState>> {
        indices
            .iter()
            .map(|&index| {
                recur64_core::GameState::from_fen(&self.position(index).fen)
                    .map_err(|e| anyhow::anyhow!("{}: {e:?}", self.position(index).id))
            })
            .collect()
    }

    pub fn cells(&self, part: &[usize]) -> Vec<(String, u8)> {
        part.iter()
            .map(|&index| {
                let position = self.position(index);
                (position.family.clone(), position.mate_depth)
            })
            .collect()
    }

    pub fn validate_root_alignment(
        &self,
        index: usize,
        root: &recur64_core::GameState,
    ) -> anyhow::Result<()> {
        let observed: Vec<u16> = root
            .legal_actions()
            .iter()
            .map(|action| action.index() as u16)
            .collect();
        anyhow::ensure!(
            observed == self.position(index).legal,
            "{}: legal action alignment differs from the dataset",
            self.position(index).id
        );
        anyhow::ensure!(
            !observed.is_empty() && !self.position(index).correct.is_empty(),
            "{}: legal and correct sets must be nonempty",
            self.position(index).id
        );
        anyhow::ensure!(
            self.position(index)
                .correct
                .iter()
                .all(|&correct| (correct as usize) < observed.len()),
            "{}: correct index is outside the legal set",
            self.position(index).id
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn measured_bindings_and_role_refusals() {
        verify_preregistered_bindings().unwrap();
        for actual in [Role::Train, Role::Dev, Role::Confirm] {
            for requested in [Role::Train, Role::Dev, Role::Confirm] {
                assert_eq!(
                    require_scientific_role(actual, requested).is_ok(),
                    actual == requested && requested != Role::Confirm
                );
            }
        }
        assert_eq!(binding(Role::Train).unwrap().manifest.records, 27000);
        assert_eq!(binding(Role::Dev).unwrap().manifest.records, 4500);
        let confirm = binding(Role::Confirm).unwrap();
        assert!(confirm.manifest.sealed);
        assert!(!confirm.manifest.evaluated);
    }
    #[test]
    fn custody_refuses_manifest_source_config_role_and_digest_tampering() {
        let good = binding(Role::Train).unwrap().manifest;
        verify_manifest(&good, &good, Role::Train).unwrap();
        assert!(verify_manifest(&good, &good, Role::Dev).is_err());
        for mutate in 0..8 {
            let mut bad = good.clone();
            match mutate {
                0 => bad.source_sha = "b".repeat(40),
                1 => bad.config_digest = "stale".into(),
                2 => bad.content_digest = "bad".into(),
                3 => bad.canonical_set_digest = "bad".into(),
                4 => bad.fen_set_digest = "bad".into(),
                5 => bad.records -= 1,
                6 => {
                    bad.cell_counts.insert("KQQvK M1".into(), 2999);
                }
                _ => bad.audit_failures = 1,
            }
            assert!(verify_manifest(&bad, &good, Role::Train).is_err());
        }
        let bytes = b"real bytes";
        let expected = format!("{:x}", Sha256::digest(bytes));
        verify_raw(bytes, &expected).unwrap();
        assert!(verify_raw(b"real byteS", &expected).is_err());
        assert!(verify_raw(bytes, "wrong").is_err());
    }
}
