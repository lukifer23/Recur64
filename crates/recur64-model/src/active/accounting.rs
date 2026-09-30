//! Compute accounting for one active-search inference (spec section 14).
//!
//! Root CandidateFacts are common baseline work and are reported separately:
//! they never consume query budget. All counters are measured, not derived from
//! the requested budget.

use serde::Serialize;

/// Counters and wall times of one batched run.
///
/// "Examples" counts positions that genuinely received the work; "rows executed"
/// counts what the device physically processed. With compaction the two are
/// equal for the query encoder and the planner; the selector runs over the whole
/// batch and its padding is reported separately, never hidden.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Accounting {
    pub requested_budget: usize,
    /// True when the run used `RunOptions::engineering_stress`: a budget above
    /// the V3.0 scientific maximum. Such a run is engineering only.
    pub engineering_only: bool,
    pub batch: usize,
    /// Batched query rounds actually executed (less than the budget only when
    /// every example ran out of frontier).
    pub steps_executed: usize,
    /// Successful queries per example.
    pub successful_queries: Vec<usize>,
    pub total_successful_queries: usize,
    /// Whether each example's final frontier is genuinely empty.
    pub final_frontier_empty: Vec<bool>,
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
    /// Examples that could not spend the whole budget because the frontier
    /// emptied (successful queries below the requested budget).
    pub exhausted_examples: usize,
    /// Executions of the heavy root board encoder (once per run, not per example).
    pub root_encoder_runs: usize,
    /// Examples processed by the root encoder.
    pub root_encoder_examples: usize,
    /// Batched query-encoder calls.
    pub query_encoder_calls: usize,
    /// Examples that received a query-encoder execution; equals successful queries.
    pub query_encoder_examples: usize,
    /// Rows the query encoder physically processed (compacted: equals examples).
    pub query_encoder_rows_executed: usize,
    /// Action slots the query encoder physically built (rows x padded width).
    pub query_action_slots_executed: usize,
    /// Legal actions actually present in those slots.
    pub query_legal_actions_valid: usize,
    /// Padded legal-action width of every round (dynamic per round).
    pub query_action_widths: Vec<usize>,
    /// Batched planner updates.
    pub planner_update_calls: usize,
    /// Examples that received a planner update; equals successful queries.
    pub planner_update_examples: usize,
    /// Rows the planner physically processed (compacted: equals examples).
    pub planner_rows_executed: usize,
    /// Rows processed that were padding or inactive. Zero by construction.
    pub inactive_rows_executed: usize,
    /// Rows the selector processed (the whole batch each round).
    pub selector_rows_executed: usize,
    /// Edge slots the selector scored (batch x padded frontier width per round).
    pub selector_edge_slots_executed: usize,
    /// Real frontier edges among those slots.
    pub selector_valid_edges: usize,
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
    /// The structural invariants every run must satisfy (Gate VI). `run` checks
    /// them before returning, so a caller cannot forget. Returns the first
    /// violated invariant.
    pub fn check_invariants(&self) -> Result<(), String> {
        let b = self.batch;
        if self.successful_queries.len() != b
            || self.query_depths.len() != b
            || self.final_frontier_empty.len() != b
        {
            return Err(format!(
                "per-example vectors ({}, {}, {}) do not match the batch {b}",
                self.successful_queries.len(),
                self.query_depths.len(),
                self.final_frontier_empty.len()
            ));
        }
        let total: usize = self.successful_queries.iter().sum();
        if total != self.total_successful_queries {
            return Err(format!(
                "sum of successful queries {total} != recorded total {}",
                self.total_successful_queries
            ));
        }
        for e in 0..b {
            if self.query_depths[e].len() != self.successful_queries[e] {
                return Err(format!(
                    "example {e}: {} recorded depths for {} successful queries",
                    self.query_depths[e].len(),
                    self.successful_queries[e]
                ));
            }
            let q = self.successful_queries[e];
            if q > self.requested_budget {
                return Err(format!("example {e}: {q} queries exceed the budget"));
            }
            // Exactly one of: budget spent, or the frontier is genuinely empty.
            if q < self.requested_budget && !self.final_frontier_empty[e] {
                return Err(format!(
                    "example {e}: {q} of {} queries spent but the frontier is not empty",
                    self.requested_budget
                ));
            }
        }
        let short = self
            .successful_queries
            .iter()
            .filter(|&&q| q < self.requested_budget)
            .count();
        if short != self.exhausted_examples {
            return Err(format!(
                "exhausted_examples {} != examples below budget {short}",
                self.exhausted_examples
            ));
        }
        if self.state_transitions != total {
            return Err(format!(
                "{} transitions for {total} successful queries",
                self.state_transitions
            ));
        }
        if self.unique_nodes != b + total {
            return Err(format!(
                "unique nodes {} != batch {b} + successful queries {total}",
                self.unique_nodes
            ));
        }
        if self.root_encoder_runs != 1 || self.root_encoder_examples != b {
            return Err(format!(
                "root encoder ran {} times over {} examples; expected once over {b}",
                self.root_encoder_runs, self.root_encoder_examples
            ));
        }
        if self.query_encoder_examples != total
            || self.query_encoder_rows_executed != total
            || self.planner_update_examples != total
            || self.planner_rows_executed != total
        {
            return Err(format!(
                "query encoder examples/rows {}/{} and planner examples/rows {}/{} must all equal {total} successful queries",
                self.query_encoder_examples,
                self.query_encoder_rows_executed,
                self.planner_update_examples,
                self.planner_rows_executed
            ));
        }
        if self.inactive_rows_executed != 0 {
            return Err(format!(
                "{} inactive/padded rows were executed",
                self.inactive_rows_executed
            ));
        }
        if self.query_encoder_calls != self.planner_update_calls
            || self.query_encoder_calls != self.steps_executed
            || self.query_action_widths.len() != self.steps_executed
            || self.steps_executed > self.requested_budget
        {
            return Err(format!(
                "rounds: encoder {} planner {} steps {} widths {} budget {}",
                self.query_encoder_calls,
                self.planner_update_calls,
                self.steps_executed,
                self.query_action_widths.len(),
                self.requested_budget
            ));
        }
        if self.exhausted_examples == 0 && self.steps_executed != self.requested_budget {
            return Err("forced budget not spent: fewer rounds than requested".into());
        }
        if self.query_legal_actions_valid > self.query_action_slots_executed
            || self.selector_valid_edges > self.selector_edge_slots_executed
        {
            return Err("valid work exceeds executed slots".into());
        }
        if self.stop_calls != 0 {
            return Err("STOP was called while masked".into());
        }
        if self.requested_budget == 0
            && (self.query_encoder_calls != 0
                || self.planner_update_calls != 0
                || self.selector_rows_executed != 0
                || total != 0)
        {
            return Err("budget 0 executed query, selector or planner work".into());
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
