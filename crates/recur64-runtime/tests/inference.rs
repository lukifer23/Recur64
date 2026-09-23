//! T2.5: single inference owner + batcher behavior.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use recur64_core::{ActionId, ObservationV1};
use recur64_runtime::{BatchEvaluator, InferenceConfig, InferenceOwner};
use recur64_search::{EvalError, EvalRequest, EvalResult, Evaluator};

struct FakeModel {
    calls: Arc<AtomicU64>,
    batch_sizes: Arc<Mutex<Vec<usize>>>,
    fail: bool,
}

impl BatchEvaluator for FakeModel {
    fn evaluate_batch(
        &self,
        observations: &[ObservationV1],
        legal: &[Vec<ActionId>],
    ) -> Result<Vec<EvalResult>, EvalError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.batch_sizes.lock().unwrap().push(observations.len());
        if self.fail {
            return Err(EvalError::Backend("boom".into()));
        }
        Ok(legal
            .iter()
            .map(|l| EvalResult::uniform(l.len(), 0.0))
            .collect())
    }
}

fn fake() -> (FakeModel, Arc<AtomicU64>, Arc<Mutex<Vec<usize>>>) {
    let calls = Arc::new(AtomicU64::new(0));
    let sizes = Arc::new(Mutex::new(Vec::new()));
    (
        FakeModel {
            calls: calls.clone(),
            batch_sizes: sizes.clone(),
            fail: false,
        },
        calls,
        sizes,
    )
}

fn legal3() -> Vec<ActionId> {
    (0..3).map(|i| ActionId::from_index(i).unwrap()).collect()
}

fn config() -> InferenceConfig {
    InferenceConfig {
        max_batch: 8,
        batch_timeout: Duration::from_millis(5),
        channel_bound: 1024,
        idle_timeout: Duration::from_millis(20),
    }
}

#[test]
fn concurrent_requests_all_answered() {
    let (model, _calls, _sizes) = fake();
    let owner = InferenceOwner::spawn(model, config());
    let ev = owner.evaluator();
    let obs = ObservationV1::zeroed();
    let legal = legal3();

    std::thread::scope(|s| {
        for _ in 0..8 {
            let ev = &ev;
            let obs = &obs;
            let legal = &legal;
            s.spawn(move || {
                for _ in 0..50 {
                    let r = ev
                        .evaluate(EvalRequest {
                            observation: obs,
                            legal,
                            side_to_move: recur64_core::Color::White,
                        })
                        .unwrap();
                    assert_eq!(r.policy.len(), 3);
                }
            });
        }
    });

    let m = owner.metrics().snapshot();
    assert_eq!(m.submitted, 400);
    assert_eq!(m.completed, 400);
    assert_eq!(m.errors, 0);
    assert!(m.batches >= 1);
    owner.shutdown();
}

#[test]
fn batching_coalesces_requests() {
    let (model, _calls, _sizes) = fake();
    let owner = InferenceOwner::spawn(model, config());
    let ev = owner.evaluator();
    let obs = ObservationV1::zeroed();
    let legal = legal3();

    std::thread::scope(|s| {
        for _ in 0..8 {
            let ev = &ev;
            let obs = &obs;
            let legal = &legal;
            s.spawn(move || {
                for _ in 0..100 {
                    ev.evaluate(EvalRequest {
                        observation: obs,
                        legal,
                        side_to_move: recur64_core::Color::White,
                    })
                    .unwrap();
                }
            });
        }
    });

    let m = owner.metrics().snapshot();
    assert_eq!(m.submitted, 800);
    assert!(
        m.batch_size_mean > 1.0,
        "expected batching, mean={}",
        m.batch_size_mean
    );
    assert!(m.batch_size_max <= 8);
    owner.shutdown();
}

#[test]
fn errors_propagate_to_every_request() {
    let calls = Arc::new(AtomicU64::new(0));
    let sizes = Arc::new(Mutex::new(Vec::new()));
    let model = FakeModel {
        calls,
        batch_sizes: sizes,
        fail: true,
    };
    let owner = InferenceOwner::spawn(model, config());
    let ev = owner.evaluator();
    let obs = ObservationV1::zeroed();
    let legal = legal3();
    let r = ev.evaluate(EvalRequest {
        observation: &obs,
        legal: &legal,
        side_to_move: recur64_core::Color::White,
    });
    assert!(r.is_err());
    let m = owner.metrics().snapshot();
    assert_eq!(m.submitted, 1);
    assert_eq!(m.completed, 0);
    assert_eq!(m.errors, 1);
    owner.shutdown();
}

#[test]
fn shutdown_makes_further_requests_fail_visibly() {
    let (model, _calls, _sizes) = fake();
    let owner = InferenceOwner::spawn(model, config());
    let ev = owner.evaluator();
    owner.shutdown();
    let obs = ObservationV1::zeroed();
    let legal = legal3();
    let r = ev.evaluate(EvalRequest {
        observation: &obs,
        legal: &legal,
        side_to_move: recur64_core::Color::White,
    });
    assert!(matches!(r, Err(EvalError::Shutdown)));
}

#[test]
fn metrics_are_recorded() {
    let (model, _calls, _sizes) = fake();
    let owner = InferenceOwner::spawn(model, config());
    let ev = owner.evaluator();
    let obs = ObservationV1::zeroed();
    let legal = legal3();
    for _ in 0..10 {
        ev.evaluate(EvalRequest {
            observation: &obs,
            legal: &legal,
            side_to_move: recur64_core::Color::White,
        })
        .unwrap();
    }
    let m = owner.metrics().snapshot();
    assert_eq!(m.submitted, 10);
    assert_eq!(m.completed, 10);
    assert!(m.forward_us_mean >= 0.0);
    owner.shutdown();
}
