//! V3 P5: the bounded learning-rate screen's training and evaluation machinery.
//!
//! * [`recipe`]: the frozen recipe and its identity digest;
//! * [`data`]: verified TRAIN/TUNE inputs and the per-budget samplers;
//! * [`teacher`]: `proof_teacher_seeded_v1` with the completion latch;
//! * [`train`]: the multi-budget trainer, loss normalisation, checkpoint and resume;
//! * [`eval`]: TUNE evaluation, the screen score and the offline selector diagnostics.
//!
//! Nothing here changes the model, `StateQueryV1` or `proof_trace_v1`.

pub mod data;
pub mod eval;
pub mod recipe;
pub mod teacher;
pub mod train;

#[cfg(test)]
pub(crate) mod tests;
