//! Exact accounting of one V4 run (invariant 11 / 12). Counted independently of the model and
//! reconciled against the exact `QueryManager` counters before any run returns.

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Accounting {
    pub requested_budget: usize,
    pub batch: usize,
    /// Successful real queries per example.
    pub successful_queries: Vec<usize>,
    pub total_successful_queries: usize,
    /// `QueryManager::state_transitions` summed over the real managers.
    pub state_transitions: u64,
    pub legal_moves_generated: u64,
    /// Root board executions (must be exactly 1 per batch).
    pub root_encoder_runs: usize,
    /// Evidence-encoder calls and examples encoded.
    pub evidence_encoder_calls: usize,
    pub evidence_encoder_examples: usize,
    /// Parent/action state-encoder calls (utility path only).
    pub state_encoder_calls: usize,
    /// Messages sitting in the ledger at the end, per example.
    pub ledger_messages: Vec<usize>,
    /// Utility-head executions (one per query step, only for the Utility selection).
    pub utility_rows_executed: usize,
    /// Exact queries executed in isolated counterfactual forks. NEVER part of the real budget.
    pub probe_queries: u64,
    pub exhausted_examples: usize,
}

impl Accounting {
    /// Structural invariants; called before any successful return from a run.
    pub fn check_invariants(&self) -> Result<(), String> {
        let total: usize = self.successful_queries.iter().sum();
        if total != self.total_successful_queries {
            return Err(format!(
                "per-example queries sum to {total}, total says {}",
                self.total_successful_queries
            ));
        }
        if self.state_transitions != total as u64 {
            return Err(format!(
                "tool state transitions {} != successful queries {total}: a state reached the model outside StateQuery or a probe leaked into the real budget",
                self.state_transitions
            ));
        }
        if self.root_encoder_runs != 1 {
            return Err(format!("root encoder ran {} times", self.root_encoder_runs));
        }
        if self.evidence_encoder_examples != total {
            return Err(format!(
                "{} messages encoded for {total} queries",
                self.evidence_encoder_examples
            ));
        }
        if self.ledger_messages != self.successful_queries {
            return Err(format!(
                "ledger holds {:?}, queries were {:?}",
                self.ledger_messages, self.successful_queries
            ));
        }
        if let Some(&q) = self
            .successful_queries
            .iter()
            .find(|&&q| q > self.requested_budget)
        {
            return Err(format!(
                "{q} queries exceed the requested budget {}",
                self.requested_budget
            ));
        }
        Ok(())
    }
}
