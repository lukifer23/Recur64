//! Recur64 V69 model crate (CUDA FP32 only). All dataset access goes through
//! recur64_v69::access::Access with the learner/evaluator roles.

pub mod d1;
pub mod d1_qualify;
pub mod data;
pub mod init;
pub mod inventory;
pub mod model;
pub mod qualify;
pub mod reference;
pub mod train;
