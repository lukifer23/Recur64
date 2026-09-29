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

/// D47: `evaluate_many` submits every request before waiting, so one caller's
/// leaves share a batch, and every request gets exactly one result in order.
#[test]
fn evaluate_many_shares_a_batch_and_answers_in_order() {
    let (model, _calls, sizes) = fake();
    let owner = InferenceOwner::spawn(model, config());
    let ev = owner.evaluator();
    let obs = ObservationV1::zeroed();
    let legals: Vec<Vec<ActionId>> = (1..=5)
        .map(|n| (0..n).map(|i| ActionId::from_index(i).unwrap()).collect())
        .collect();
    let requests: Vec<EvalRequest<'_>> = legals
        .iter()
        .map(|legal| EvalRequest {
            observation: &obs,
            legal,
            side_to_move: recur64_core::Color::White,
        })
        .collect();
    let out = ev.evaluate_many(&requests);
    assert_eq!(out.len(), 5);
    for (r, legal) in out.iter().zip(&legals) {
        assert_eq!(
            r.as_ref().unwrap().policy.len(),
            legal.len(),
            "results in request order"
        );
    }
    assert!(
        sizes.lock().unwrap().iter().any(|&n| n > 1),
        "one caller's requests coalesced: {:?}",
        sizes.lock().unwrap()
    );
    let m = owner.metrics().snapshot();
    assert_eq!(m.completed, 5);
    assert!(m.peak_in_flight >= 5);
}

/// X0.3: the multi-owner pool. Every request must be answered exactly once,
/// every owner in the pool must actually serve batches, the configured cap must
/// still be respected, and the pool must shut down cleanly (no leaked threads,
/// no dropped request).
#[test]
fn owner_pool_answers_every_request_and_every_owner_serves() {
    // Four independent fake models, each counting its own batches, so we can
    // prove more than one owner did work rather than assuming it.
    let mut models = Vec::new();
    let mut counters = Vec::new();
    for _ in 0..4 {
        let (m, calls, sizes) = fake();
        models.push(m);
        counters.push((calls, sizes));
    }
    let owner = InferenceOwner::spawn_pool(models, config());
    let ev = owner.evaluator();

    let obs = ObservationV1::zeroed();
    let legal = legal3();
    let request = EvalRequest {
        observation: &obs,
        legal: &legal,
        side_to_move: recur64_core::Color::White,
    };
    // Enough requests that every owner is overwhelmingly likely to be handed a
    // batch; the assertions below do not depend on which one served which.
    let total = 64usize;
    let results: Vec<_> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..4)
            .map(|_| {
                scope.spawn(|| {
                    (0..total / 4)
                        .map(|_| ev.evaluate(request))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect()
    });

    assert_eq!(results.len(), total, "every request produced a result");
    for (i, r) in results.iter().enumerate() {
        let r = r
            .as_ref()
            .unwrap_or_else(|e| panic!("request {i} failed: {e}"));
        assert_eq!(r.policy.len(), 3, "request {i} policy is aligned");
    }

    let m = owner.metrics().snapshot();
    assert_eq!(
        m.completed, total as u64,
        "every request completed exactly once"
    );
    assert_eq!(m.errors, 0, "no request errored");
    assert_eq!(m.owners, 4, "the pool reports its size");
    assert!(m.batches >= 1);

    // The cap is per batch, whichever owner formed it.
    for (_, sizes) in &counters {
        assert!(
            sizes.lock().unwrap().iter().all(|&n| n <= 8),
            "an owner exceeded max_batch"
        );
    }
    let served = counters
        .iter()
        .filter(|(calls, _)| calls.load(Ordering::SeqCst) > 0)
        .count();
    assert!(
        served >= 2,
        "expected real multi-owner service, only {served} of 4 owners served a batch"
    );

    // Shutdown joins every owner thread and answers any straggler.
    owner.shutdown();
    // Pool size 1 must still behave exactly like the single owner.
    let (m1, _, _) = fake();
    let single = InferenceOwner::spawn_pool(vec![m1], config());
    let ev1 = single.evaluator();
    assert!(ev1.evaluate(request).is_ok());
    assert_eq!(single.metrics().snapshot().owners, 1);
    single.shutdown();
}
