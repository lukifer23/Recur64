//! `active_search_v3`: budgeted exact state queries with learned selection.
//!
//! See `docs/V3_ARCHITECTURE.md`. The exact query tool lives in
//! `recur64-statequery`; nothing here can reach the mate solver.

pub mod accounting;
pub mod counters;
pub mod coverage;
pub mod features;
pub mod loss;
pub mod model;
pub mod modules;
pub mod run;
pub mod tree;

pub use accounting::{Accounting, QueryRecord, StepDiag};
pub use model::ActiveSearchModel;
pub use run::{
    ActiveOutput, QueryContentAblation, QueryScript, RunOptions, ScriptStep, Selection,
    SelectorStep,
};
pub use tree::{EdgeRef, NodeMeta, Tree};
