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
                            facts: None,
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
                        facts: None,
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
        facts: None,
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
        facts: None,
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
            facts: None,
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
            facts: None,
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

/// T2: an owner pool answers every request exactly once, every owner serves
/// batches, batches respect the cap, and shutdown joins all owners.
#[test]
fn owner_pool_serves_one_queue_with_every_owner() {
    let (a, calls_a, sizes_a) = fake();
    let (b, calls_b, sizes_b) = fake();
    // Slow forwards so that one owner's batch overlaps the other's.
    struct Slow(FakeModel);
    impl BatchEvaluator for Slow {
        fn evaluate_batch(
            &self,
            observations: &[ObservationV1],
            legal: &[Vec<ActionId>],
        ) -> Result<Vec<EvalResult>, EvalError> {
            std::thread::sleep(Duration::from_millis(3));
            self.0.evaluate_batch(observations, legal)
        }
    }
    let owner = InferenceOwner::spawn_pool(vec![Slow(a), Slow(b)], config());
    let ev = owner.evaluator();
    let obs = ObservationV1::zeroed();
    let legal = legal3();
    std::thread::scope(|s| {
        for _ in 0..16 {
            let (ev, obs, legal) = (&ev, &obs, &legal);
            s.spawn(move || {
                for _ in 0..25 {
                    let r = ev
                        .evaluate(EvalRequest {
                            observation: obs,
                            legal,
                            side_to_move: recur64_core::Color::White,
                            facts: None,
                        })
                        .unwrap();
                    assert_eq!(r.policy.len(), 3);
                }
            });
        }
    });
    let m = owner.metrics().snapshot();
    assert_eq!(m.owners, 2);
    assert_eq!((m.submitted, m.completed, m.errors), (400, 400, 0));
    assert!(calls_a.load(Ordering::SeqCst) > 0, "owner 0 served batches");
    assert!(calls_b.load(Ordering::SeqCst) > 0, "owner 1 served batches");
    let served: usize = sizes_a
        .lock()
        .unwrap()
        .iter()
        .chain(sizes_b.lock().unwrap().iter())
        .sum();
    assert_eq!(served, 400, "every request evaluated exactly once");
    assert!(
        sizes_a
            .lock()
            .unwrap()
            .iter()
            .chain(sizes_b.lock().unwrap().iter())
            .all(|&n| n <= 8)
    );
    owner.shutdown();
}
