//! Synchronized engineering timings, not an alternate reader implementation.

use std::time::Instant;

use burn::tensor::backend::Backend;
use serde::Serialize;

pub const CONTRACT: &str = "v5_synchronized_execution_profile_v1";

pub fn contract_digest() -> String {
    use sha2::{Digest, Sha256};
    // Operational protocol, not a new learned/scientific architecture.
    let declaration = concat!(
        "v5_synchronized_execution_profile_v1|burn=0.21.0|fence=Backend::sync|",
        "fp32|host_and_upload_combined|candidate_facts_cpu_separate|",
        "root_once|state_once|factual_then_null|each_initialize_and_E_H_iteration|",
        "paired_readout|final_set_loss_scalar_read|full_both_stream_backward|",
        "all_parameter_gradient_health_readback|adamw_final_fence|",
        "checkpoint_save_load_all_contents_two_exact_continuations|",
        "same_device_all_logits_payload_gradients_parameter_gradients_adamw_parity|",
        "current_raw_input_and_root_facts_preparation_twice|barrier_overhead_included"
    );
    format!("{:x}", Sha256::digest(declaration.as_bytes()))
}

#[derive(Debug, Clone, Serialize)]
pub struct PhaseTiming {
    pub phase: String,
    pub seconds: f64,
}

/// Observer callbacks never inspect, detach or replace model tensors. Each
/// interval ends with the pinned backend's completion fence. Errors are retained
/// and MUST be checked with `finish` before publishing any timings.
pub(crate) struct SynchronizedProfile<'a, B: Backend> {
    device: &'a B::Device,
    started: Instant,
    phases: Vec<PhaseTiming>,
    failure: Option<String>,
}

impl<'a, B: Backend> SynchronizedProfile<'a, B> {
    pub fn new(device: &'a B::Device) -> anyhow::Result<Self> {
        B::sync(device).map_err(|e| anyhow::anyhow!("initial profile fence failed: {e:?}"))?;
        Ok(Self {
            device,
            started: Instant::now(),
            phases: Vec::new(),
            failure: None,
        })
    }

    pub fn mark(&mut self, phase: &str) {
        if self.failure.is_some() {
            return;
        }
        if let Err(error) = B::sync(self.device) {
            self.failure = Some(format!("profile fence at {phase} failed: {error:?}"));
            return;
        }
        self.phases.push(PhaseTiming {
            phase: phase.into(),
            seconds: self.started.elapsed().as_secs_f64(),
        });
        self.started = Instant::now();
    }

    pub fn finish(self) -> anyhow::Result<Vec<PhaseTiming>> {
        anyhow::ensure!(
            self.failure.is_none(),
            "{}",
            self.failure.unwrap_or_default()
        );
        Ok(self.phases)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failed_fence_observer_cannot_publish_timings() {
        let device = Default::default();
        let mut profile = SynchronizedProfile::<burn::backend::Flex>::new(&device).unwrap();
        // Isolated observer error-handling test, not a fake device qualification.
        profile.failure = Some("test-only injected fence error".into());
        profile.mark("unpublishable");
        assert!(profile.finish().is_err());
        assert_eq!(contract_digest().len(), 64);
    }
}
