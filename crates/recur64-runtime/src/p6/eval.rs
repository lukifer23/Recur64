//! ALL-INFO TUNE evaluation. Scored by the same per-position function as every other P5/P6
//! evaluation (`p5::eval::example_result`), so ACTIVE B0 and ALL-INFO are comparable.

use burn::prelude::*;

use recur64_model::all_info::AllInfoModel;
use recur64_model::candidate::CandidateInputs;

use crate::p5::eval::{EvalSummary, ExampleResult, example_result, summarise_positions};
use crate::proof::targets::ProofTargets;

use super::data::{build_trees, roots_of};

/// Selection label recorded in ALL-INFO summaries.
pub const ALL_INFO_LABEL: &str = "all_info_depth2_v1";

/// Evaluate `model` on every position of `data` (V3_TUNE_V1). Returns the per-position
/// results in dataset order and their summary, plus the total future states supplied.
pub fn evaluate_all_info<B: Backend>(
    model: &AllInfoModel<B>,
    data: &ProofTargets,
    batch: usize,
    device: &B::Device,
) -> anyhow::Result<(Vec<ExampleResult>, EvalSummary, usize)> {
    let n = data.positions.len();
    let mut results = Vec::with_capacity(n);
    let mut states = 0usize;
    let mut start = 0usize;
    while start < n {
        let end = (start + batch).min(n);
        let positions: Vec<&_> = data.positions[start..end].iter().collect();
        let roots = roots_of(&positions)?;
        let trees = build_trees(&roots)?;
        let inputs = CandidateInputs::<B>::from_states(&roots, device)?;
        let out = model.forward_trees(&inputs, &trees, device)?;
        states += out
            .accounting
            .states
            .iter()
            .map(|(a, b)| a + b)
            .sum::<usize>();
        let w = out.readout.policy.mask.dims()[1];
        let lp: Vec<f32> = out
            .readout
            .policy
            .log_probs
            .into_data()
            .to_vec()
            .map_err(|e| anyhow::anyhow!("reading policy: {e:?}"))?;
        anyhow::ensure!(
            lp.iter().all(|v| v.is_finite()),
            "non-finite policy output during ALL-INFO evaluation"
        );
        for (i, p) in positions.iter().enumerate() {
            results.push(example_result(&lp[i * w..i * w + p.legal.len()], p));
        }
        start = end;
    }
    let summary = summarise_positions(&data.positions, &results, 0, ALL_INFO_LABEL.to_string());
    Ok((results, summary, states))
}
