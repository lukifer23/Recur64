//! V3.5 `on_policy_proof_relabel_v1`: the on-policy information-acquisition rescue.
//!
//! The learner selects EVERY query. ProofTrace only LABELS the learner-visited states;
//! it can never choose, override or perturb the executed edge.
//!
//! * [`recipe`]: the frozen V3.5 recipe and identity;
//! * [`target`]: the label-only `A(S) = A_proof ∪ A_refute` provider;
//! * [`train`]: detached ACTIVE rollout + autodiff replay trainer, checkpoint and resume;
//! * [`gate`]: the pre-registered Gate II / III / Content-Use estimators.
//!
//! Nothing here changes the model, `StateQueryV1` or `proof_trace_v1`.

pub mod gate;
pub mod recipe;
pub mod target;
pub mod train;

#[cfg(test)]
pub(crate) mod tests;
