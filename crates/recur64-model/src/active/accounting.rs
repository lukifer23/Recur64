//! Compute accounting for one active-search inference (spec section 14).
//!
//! Root CandidateFacts are common baseline work and are reported separately:
//! they never consume query budget. All counters are measured, not derived from
//! the requested budget.

use serde::Serialize;

/// Counters and wall times of one batched run.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Accounting {
    pub requested_budget: usize,
    pub batch: usize,
    /// Batched query rounds actually executed (less than the budget only when
    /// every example ran out of frontier).
    pub steps_executed: usize,
    /// Successful queries per example.
    pub successful_queries: Vec<usize>,
    pub total_successful_queries: usize,
    /// STOP is masked in the primary experiment.
    pub stop_calls: usize,
    /// Exact transitions performed by the tool (one per successful query).
    pub state_transitions: usize,
    /// Legal moves generated at returned children (part of each query cost).
    pub legal_moves_generated: u64,
    /// Ply depth of every queried node, in query order per example.
    pub query_depths: Vec<Vec<u32>>,
    pub unique_nodes: usize,
    pub terminal_nodes: usize,
    pub transpositions_detected: usize,
    /// Examples whose frontier emptied before the budget was spent.
    pub exhausted_examples: usize,
    /// Executions of the heavy root board encoder (once per run, not per example).
    pub root_encoder_runs: usize,
    /// Examples processed by the root encoder.
    pub root_encoder_examples: usize,
    /// Batched query-encoder calls.
    pub query_encoder_calls: usize,
    /// Examples processed by the query encoder; equals successful queries.
    pub query_encoder_examples: usize,
    /// Batched planner updates.
    pub planner_update_calls: usize,
    /// Examples that received a planner update; equals successful queries.
    pub planner_update_examples: usize,
    /// Positions with root CandidateFacts computed (baseline work, not budget).
    pub root_facts_positions: usize,
    // --- wall time, seconds ---
    pub root_facts_s: f64,
    pub cpu_query_s: f64,
    pub root_encoder_s: f64,
    pub query_encoder_s: f64,
    pub planner_selector_s: f64,
    pub total_s: f64,
    /// Whether GPU sections were synchronised before each clock read. Without it
    /// the GPU wall times measure launch time, not completion.
    pub timing_synchronised: bool,
}

impl Accounting {
    /// The invariants every scientific run must satisfy (Gate VI). Returns the
    /// first violated invariant.
    pub fn check_invariants(&self) -> Result<(), String> {
        if self.root_encoder_runs != 1 {
            return Err(format!(
                "root encoder ran {} times, expected exactly once",
                self.root_encoder_runs
            ));
        }
        if self.root_encoder_examples != self.batch {
            return Err(format!(
                "root encoder processed {} examples for a batch of {}",
                self.root_encoder_examples, self.batch
            ));
        }
        let q = self.total_successful_queries;
        if self.query_encoder_examples != q || self.planner_update_examples != q {
            return Err(format!(
                "query encoder {} / planner {} examples differ from {q} successful queries",
                self.query_encoder_examples, self.planner_update_examples
            ));
        }
        if self.state_transitions != q {
            return Err(format!(
                "{} transitions for {q} successful queries",
                self.state_transitions
            ));
        }
        if self.stop_calls != 0 {
            return Err("STOP was called while masked".into());
        }
        if self.requested_budget == 0
            && (self.query_encoder_calls != 0 || self.planner_update_calls != 0)
        {
            return Err("budget 0 executed query or planner work".into());
        }
        if self.exhausted_examples == 0
            && self
                .successful_queries
                .iter()
                .any(|&n| n != self.requested_budget)
        {
            return Err("forced budget not spent exactly".into());
        }
        Ok(())
    }
}

/// One query decision, for traces.
#[derive(Debug, Clone, Serialize)]
pub struct QueryRecord {
    pub step: usize,
    /// Slot of the node the edge leaves.
    pub parent_slot: usize,
    pub action: u16,
    /// Root candidate (index into the root legal list) the edge belongs to.
    pub branch: usize,
    /// Ply depth of the node created.
    pub depth: u32,
    pub terminal: bool,
    pub frontier_size: usize,
    /// Selector entropy over the frontier; present for ACTIVE or diagnostics.
    pub selector_entropy: Option<f32>,
    /// Gap between the two best selector logits; present for ACTIVE or diagnostics.
    pub selector_margin: Option<f32>,
}

/// Per-step planner diagnostics (only when requested; they synchronise).
#[derive(Debug, Clone, Serialize)]
pub struct StepDiag {
    pub step: usize,
    pub workspace_rms: f32,
    pub workspace_delta: f32,
    pub branch_rms: f32,
    pub branch_delta: f32,
    pub gate_mean: f32,
    pub gate_std: f32,
    pub branch_gate_mean: f32,
    pub remaining: usize,
}
