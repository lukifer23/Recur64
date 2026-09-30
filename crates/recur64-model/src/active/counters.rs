//! Independent execution counters.
//!
//! [`super::accounting::Accounting`] is filled by the driver. These counters are
//! incremented inside the model functions themselves, so a test can compare the
//! driver's claims with what actually executed (for example that the heavy root
//! encoder ran exactly once). They are thread-local: a run executes on the
//! calling thread, and parallel test threads do not interfere.

use std::cell::Cell;

thread_local! {
    static ROOT_STAGE: Cell<usize> = const { Cell::new(0) };
    static QUERY_ENCODER: Cell<usize> = const { Cell::new(0) };
    static PLANNER_UPDATE: Cell<usize> = const { Cell::new(0) };
    static ROOT_FACTS: Cell<usize> = const { Cell::new(0) };
}

/// Executions since the thread started (take a snapshot before and after a run).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Counts {
    pub root_stage: usize,
    pub query_encoder: usize,
    pub planner_update: usize,
    /// Positions whose root CandidateFacts were computed.
    pub root_facts: usize,
}

impl Counts {
    pub fn since(self, earlier: Counts) -> Counts {
        Counts {
            root_stage: self.root_stage - earlier.root_stage,
            query_encoder: self.query_encoder - earlier.query_encoder,
            planner_update: self.planner_update - earlier.planner_update,
            root_facts: self.root_facts - earlier.root_facts,
        }
    }
}

pub fn snapshot() -> Counts {
    Counts {
        root_stage: ROOT_STAGE.with(Cell::get),
        query_encoder: QUERY_ENCODER.with(Cell::get),
        planner_update: PLANNER_UPDATE.with(Cell::get),
        root_facts: ROOT_FACTS.with(Cell::get),
    }
}

pub(crate) fn note_root_stage() {
    ROOT_STAGE.with(|c| c.set(c.get() + 1));
}

pub(crate) fn note_query_encoder() {
    QUERY_ENCODER.with(|c| c.set(c.get() + 1));
}

pub(crate) fn note_planner_update() {
    PLANNER_UPDATE.with(|c| c.set(c.get() + 1));
}

pub(crate) fn note_root_facts(positions: usize) {
    ROOT_FACTS.with(|c| c.set(c.get() + positions));
}
