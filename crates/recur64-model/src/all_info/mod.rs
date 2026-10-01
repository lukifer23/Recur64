//! P6 `all_info_v1`: the information-sufficiency control (see `docs/V3_P6_PLAN.md`).

pub mod batch;
pub mod model;
pub mod tree;

pub use batch::TreeBatch;
pub use model::{AllInfoAccounting, AllInfoModel, AllInfoOutput, TreeIntegrator};
pub use tree::{AllInfoTree, Branch, TreeCounts};
