//! Narrow neural-evaluation interface for search.
//!
//! Search is deliberately independent of Burn and of `recur64-model`. It sees
//! only this trait, so it can be driven by a batched GPU owner, a synchronous
//! model evaluator, or a deterministic test evaluator without any change to
//! search logic.

use recur64_core::{ActionId, Color, ObservationV1};

/// Errors surfaced by an evaluator. Every request must produce either an
/// `Ok` result or an `Err`; an evaluator must never drop a request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvalError {
    /// The evaluator is shutting down; the request was not processed.
    Shutdown,
    /// A backend/model failure with a human-readable reason.
    Backend(String),
    /// The request was malformed (e.g. empty candidate list, wrong policy len).
    Invalid(String),
    /// A wall-clock deadline passed before every requested game could start
    /// (D38). No partial result is returned: an incomplete evaluation must
    /// never inform a decision.
    DeadlineExceeded(String),
}

impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EvalError::Shutdown => write!(f, "evaluator is shutting down"),
            EvalError::Backend(m) => write!(f, "evaluator backend error: {m}"),
            EvalError::Invalid(m) => write!(f, "invalid evaluation request: {m}"),
            EvalError::DeadlineExceeded(m) => write!(f, "evaluation deadline exceeded: {m}"),
        }
    }
}

impl std::error::Error for EvalError {}

/// A single-position evaluation request.
///
/// The observation is canonical (current side to move). `legal` is the canonical
/// legal action list in a deterministic (sorted) order.
pub struct EvalRequest<'a> {
    pub observation: &'a ObservationV1,
    pub legal: &'a [ActionId],
    /// Side to move at this position. Most evaluators ignore this (the
    /// observation is already canonical), but the arena uses it to route each
    /// ply to the correct model.
    pub side_to_move: Color,
}

/// Evaluation output, all from the **side-to-move perspective**.
#[derive(Debug, Clone, PartialEq)]
pub struct EvalResult {
    /// Probability per legal action, aligned to `EvalRequest::legal`, summing to 1.
    pub policy: Vec<f32>,
    /// Scalar value `P(win) - P(loss)`.
    pub value: f32,
    /// `[win, draw, loss]` probabilities.
    pub wdl: [f32; 3],
}

impl EvalResult {
    /// Uniform policy and a fixed value (used by deterministic tests).
    pub fn uniform(n: usize, value: f32) -> Self {
        let p = if n == 0 { 0.0 } else { 1.0 / n as f32 };
        let wdl = value_to_wdl(value);
        Self {
            policy: vec![p; n],
            value,
            wdl,
        }
    }
}

/// Map a scalar value in `[-1, 1]` to a `[win, draw, loss]` distribution.
pub fn value_to_wdl(value: f32) -> [f32; 3] {
    let v = value.clamp(-1.0, 1.0);
    // win - loss = v, win + loss = 1 - draw, symmetric draw mass.
    let draw = (1.0 - v.abs()).clamp(0.0, 1.0);
    let decisive = 1.0 - draw;
    let win = if v >= 0.0 { decisive } else { 0.0 };
    let loss = if v < 0.0 { decisive } else { 0.0 };
    [win, draw, loss]
}

/// The evaluation interface consumed by search.
pub trait Evaluator: Send + Sync {
    fn evaluate(&self, request: EvalRequest<'_>) -> Result<EvalResult, EvalError>;
}

/// A deterministic evaluator with a constant value and either uniform or
/// supplied priors. Test infrastructure, not a production mock.
pub struct FixedEvaluator {
    pub value: f32,
    pub policy: Option<Vec<f32>>,
}

impl FixedEvaluator {
    pub fn uniform(value: f32) -> Self {
        Self {
            value,
            policy: None,
        }
    }
    pub fn with_policy(value: f32, policy: Vec<f32>) -> Self {
        Self {
            value,
            policy: Some(policy),
        }
    }
}

impl Evaluator for FixedEvaluator {
    fn evaluate(&self, request: EvalRequest<'_>) -> Result<EvalResult, EvalError> {
        let n = request.legal.len();
        let policy = match &self.policy {
            Some(p) => {
                if p.len() != n {
                    return Err(EvalError::Invalid(format!(
                        "policy len {} != legal len {n}",
                        p.len()
                    )));
                }
                p.clone()
            }
            None => vec![if n == 0 { 0.0 } else { 1.0 / n as f32 }; n],
        };
        Ok(EvalResult {
            policy,
            value: self.value,
            wdl: value_to_wdl(self.value),
        })
    }
}

/// An evaluator that returns values from a scripted sequence (by call index) and
/// counts calls. Used to assert budget contracts.
pub struct ScriptedEvaluator {
    values: Vec<f32>,
    calls: std::sync::atomic::AtomicUsize,
}

impl ScriptedEvaluator {
    pub fn new(values: Vec<f32>) -> Self {
        Self {
            values,
            calls: std::sync::atomic::AtomicUsize::new(0),
        }
    }
    pub fn calls(&self) -> usize {
        self.calls.load(std::sync::atomic::Ordering::SeqCst)
    }
}

impl Evaluator for ScriptedEvaluator {
    fn evaluate(&self, request: EvalRequest<'_>) -> Result<EvalResult, EvalError> {
        let i = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let value = *self
            .values
            .get(i)
            .unwrap_or_else(|| self.values.last().unwrap_or(&0.0));
        Ok(EvalResult::uniform(request.legal.len(), value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_uniform_sums_to_one() {
        let ev = FixedEvaluator::uniform(0.5);
        let obs = ObservationV1::zeroed();
        let legal: Vec<ActionId> = (0..5).map(|i| ActionId::from_index(i).unwrap()).collect();
        let r = ev
            .evaluate(EvalRequest {
                observation: &obs,
                legal: &legal,
                side_to_move: recur64_core::Color::White,
            })
            .unwrap();
        assert_eq!(r.policy.len(), 5);
        assert!((r.policy.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert_eq!(r.value, 0.5);
    }

    #[test]
    fn value_to_wdl_is_consistent() {
        let w = value_to_wdl(1.0);
        assert_eq!(w, [1.0, 0.0, 0.0]);
        let l = value_to_wdl(-1.0);
        assert_eq!(l, [0.0, 0.0, 1.0]);
        let d = value_to_wdl(0.0);
        assert_eq!(d, [0.0, 1.0, 0.0]);
        let w = value_to_wdl(0.5);
        assert!((w[0] - w[2] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn scripted_counts_calls() {
        let ev = ScriptedEvaluator::new(vec![0.1, 0.2, 0.3]);
        let obs = ObservationV1::zeroed();
        let legal = vec![ActionId::from_index(0).unwrap()];
        for _ in 0..3 {
            ev.evaluate(EvalRequest {
                observation: &obs,
                legal: &legal,
                side_to_move: recur64_core::Color::White,
            })
            .unwrap();
        }
        assert_eq!(ev.calls(), 3);
    }
}
