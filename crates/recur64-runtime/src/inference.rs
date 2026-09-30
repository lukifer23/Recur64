//! Single-owner batched neural inference.
//!
//! Exactly one thread owns the Burn backend. Game/search workers submit
//! single-position requests through a bounded channel and block on a private
//! response channel; the owner coalesces them into batches. Every request
//! receives exactly one response (a result or an error), so no caller can block
//! forever — including during shutdown.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, TrySendError, sync_channel};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use burn::prelude::*;
use burn::tensor::TensorData;

use recur64_core::{ActionId, CandidateFactsV1, ObservationV1};
use recur64_model::action::CandidateBatch;
use recur64_model::candidate::facts_tensor;
use recur64_model::model::{CandidateTensors, ProbeModel};
use recur64_model::net::NeuralModel;
use recur64_search::{EvalError, EvalRequest, EvalResult, Evaluator};

/// Batcher configuration. All values are pilot hypotheses, not tuned constants.
#[derive(Debug, Clone, Copy)]
pub struct InferenceConfig {
    pub max_batch: usize,
    /// Maximum time to wait for additional requests once the first arrives.
    pub batch_timeout: Duration,
    /// Bounded channel capacity (backpressure).
    pub channel_bound: usize,
    /// Idle poll interval, used to notice cancellation promptly.
    pub idle_timeout: Duration,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            max_batch: 32,
            batch_timeout: Duration::from_micros(500),
            channel_bound: 512,
            idle_timeout: Duration::from_millis(50),
        }
    }
}

/// A model that can evaluate a batch of positions in one call.
pub trait BatchEvaluator: Send + 'static {
    fn evaluate_batch(
        &self,
        observations: &[ObservationV1],
        legal: &[Vec<ActionId>],
    ) -> Result<Vec<EvalResult>, EvalError>;

    /// Whether this model consumes `CandidateFactsV1` (default: no).
    fn needs_candidate_facts(&self) -> bool {
        false
    }

    /// Evaluate with per-position candidate facts (aligned to `legal`). The
    /// default ignores them and calls [`Self::evaluate_batch`], which is right
    /// for models that do not consume facts.
    fn evaluate_batch_with_facts(
        &self,
        observations: &[ObservationV1],
        legal: &[Vec<ActionId>],
        facts: &[Option<Vec<CandidateFactsV1>>],
    ) -> Result<Vec<EvalResult>, EvalError> {
        let _ = facts;
        self.evaluate_batch(observations, legal)
    }
}

/// Softmax over three logits.
fn softmax3(logits: [f32; 3]) -> [f32; 3] {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps = logits.map(|v| (v - max).exp());
    let sum = exps.iter().sum::<f32>();
    // NaN/Inf logits propagate as NaN so the caller's finiteness check
    // refuses them; a uniform fallback would hide them as value 0 (ported
    // from the HP branch's core review).
    exps.map(|v| v / sum)
}

/// The production batch evaluator over a Burn model.
pub struct BatchedModel<B: Backend, M: NeuralModel<B> = ProbeModel<B>> {
    model: M,
    recurrence: usize,
    device: B::Device,
    /// Round the candidate width up to fixed buckets (T6, execution only).
    bucket_candidates: bool,
}

impl<B: Backend, M: NeuralModel<B>> BatchedModel<B, M> {
    pub fn new(model: M, recurrence: usize, device: B::Device) -> Self {
        Self {
            model,
            recurrence,
            device,
            bucket_candidates: false,
        }
    }

    /// Enable candidate-width bucketing (fewer distinct tensor shapes, T6).
    pub fn with_candidate_buckets(mut self, on: bool) -> Self {
        self.bucket_candidates = on;
        self
    }
}

/// Release the device memory pool of the thread that ran this model.
///
/// CubeCL keys device streams (and their memory pools) by OS thread. An
/// `InferenceOwner` runs its model on a dedicated thread that exits at
/// shutdown, so without this its stream's pool is never reused or released,
/// and VRAM grows by hundreds of MiB per owner lifecycle (measured, P4.4L).
/// `Drop` runs on the thread that owns the model, so the cleanup targets that
/// thread's stream. Parameter buffers belong to the loading thread's stream
/// and are unaffected.
impl<B: Backend, M: NeuralModel<B>> Drop for BatchedModel<B, M> {
    fn drop(&mut self) {
        B::memory_cleanup(&self.device);
    }
}

impl<B: Backend, M: NeuralModel<B>> BatchEvaluator for BatchedModel<B, M> {
    fn evaluate_batch(
        &self,
        observations: &[ObservationV1],
        legal: &[Vec<ActionId>],
    ) -> Result<Vec<EvalResult>, EvalError> {
        let none = vec![None; observations.len()];
        self.evaluate_batch_with_facts(observations, legal, &none)
    }

    fn needs_candidate_facts(&self) -> bool {
        self.model.needs_candidate_facts()
    }

    fn evaluate_batch_with_facts(
        &self,
        observations: &[ObservationV1],
        legal: &[Vec<ActionId>],
        facts: &[Option<Vec<CandidateFactsV1>>],
    ) -> Result<Vec<EvalResult>, EvalError> {
        let b = observations.len();
        if facts.len() != b {
            return Err(EvalError::Invalid(format!(
                "facts batch {} != observation batch {b}",
                facts.len()
            )));
        }
        if b == 0 {
            return Ok(Vec::new());
        }
        if legal.iter().any(|l| l.is_empty()) {
            return Err(EvalError::Invalid(
                "batch contains a position with no legal actions".into(),
            ));
        }

        let mut data = Vec::with_capacity(b * 64 * 119);
        for o in observations {
            data.extend_from_slice(o.as_slice());
        }
        let board = Tensor::<B, 3>::from_data(TensorData::new(data, [b, 64, 119]), &self.device);

        let lists: Vec<Vec<(u32, u32, u8)>> = legal
            .iter()
            .map(|l| {
                l.iter()
                    .map(|id| {
                        let (from, to, promo) = id.decode();
                        (from as u32, to as u32, promo.code())
                    })
                    .collect()
            })
            .collect();
        let cb = if self.bucket_candidates {
            CandidateBatch::from_lists_bucketed(&lists)
        } else {
            CandidateBatch::from_lists(&lists)
        };
        if cb.width == 0 {
            return Err(EvalError::Invalid("batch has no legal candidates".into()));
        }
        let cands = CandidateTensors::from_batch(&cb, &self.device);

        // Facts are supplied only to models that consume them; a model that
        // needs them and does not get them is an error, never a zero fallback.
        let facts_t = if self.model.needs_candidate_facts() {
            let mut rows: Vec<&[CandidateFactsV1]> = Vec::with_capacity(b);
            for (i, (f, l)) in facts.iter().zip(legal).enumerate() {
                let f = f.as_deref().ok_or_else(|| {
                    EvalError::Invalid(format!(
                        "position {i}: this model requires CandidateFacts and none were supplied"
                    ))
                })?;
                if f.len() != l.len() {
                    return Err(EvalError::Invalid(format!(
                        "position {i}: {} fact rows for {} legal actions",
                        f.len(),
                        l.len()
                    )));
                }
                rows.push(f);
            }
            Some(
                facts_tensor::<B>(&rows, cb.width, &self.device)
                    .map_err(|e| EvalError::Invalid(format!("facts tensor: {e}")))?,
            )
        } else {
            None
        };

        let out = self
            .model
            .forward_inputs(board, &cands, facts_t, self.recurrence, false);
        let readout = out
            .readouts
            .first()
            .ok_or_else(|| EvalError::Backend("model produced no readout".into()))?;

        let log_probs = readout
            .policy
            .log_probs
            .clone()
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| EvalError::Backend(format!("policy read failed: {e}")))?;
        let wdl_logits = readout
            .wdl_logits
            .clone()
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| EvalError::Backend(format!("wdl read failed: {e}")))?;

        let width = cb.width;
        let mut results = Vec::with_capacity(b);
        for (i, l) in legal.iter().enumerate() {
            let n = l.len();
            let mut policy: Vec<f32> = (0..n).map(|k| log_probs[i * width + k].exp()).collect();
            let sum: f32 = policy.iter().sum();
            // A NaN/Inf or all-zero legal policy is a model failure, never a
            // uniform prior (no silent fallback).
            if n > 0 && !(sum.is_finite() && sum > 0.0) {
                return Err(EvalError::Backend(format!(
                    "non-finite or zero legal policy mass ({sum})"
                )));
            }
            for p in policy.iter_mut() {
                *p /= sum;
            }
            let wdl = softmax3([
                wdl_logits[i * 3],
                wdl_logits[i * 3 + 1],
                wdl_logits[i * 3 + 2],
            ]);
            let value = wdl[0] - wdl[2];
            if !value.is_finite() || wdl.iter().any(|w| !w.is_finite()) {
                return Err(EvalError::Backend("non-finite value output".into()));
            }
            results.push(EvalResult { policy, value, wdl });
        }
        Ok(results)
    }
}

struct Request {
    observation: ObservationV1,
    legal: Vec<ActionId>,
    /// Owned copy of the candidate facts (None for models that do not use them).
    facts: Option<Vec<CandidateFactsV1>>,
    submitted_at: Instant,
    respond: SyncSender<Result<EvalResult, EvalError>>,
}

/// Batcher metrics. Counters are atomics; batch-size and queue-wait samples are
/// retained (bounded) for percentiles.
#[derive(Default)]
pub struct InferenceMetrics {
    pub submitted: AtomicU64,
    pub completed: AtomicU64,
    pub errors: AtomicU64,
    pub batches: AtomicU64,
    pub batch_size_sum: AtomicU64,
    pub batch_size_max: AtomicU64,
    pub timeout_flushes: AtomicU64,
    pub queue_wait_us_sum: AtomicU64,
    pub forward_us_sum: AtomicU64,
    /// Simultaneous in-flight evaluator calls. This is the direct measure of
    /// *real* concurrency: a configured worker count only matters if this gauge
    /// actually rises above 1 (and batch p50 rises with it).
    pub in_flight: AtomicUsize,
    pub peak_in_flight: AtomicUsize,
    /// Owner threads serving the queue (T2 owner pool).
    pub owners: AtomicUsize,
    samples: Mutex<SampleBuffer>,
}

#[derive(Default)]
struct SampleBuffer {
    batch_sizes: Vec<u32>,
    queue_waits_us: Vec<u64>,
}

const MAX_SAMPLES: usize = 1_000_000;

impl InferenceMetrics {
    fn record_batch(&self, size: usize, queue_wait_us: u64, forward_us: u64) {
        self.batches.fetch_add(1, Ordering::Relaxed);
        self.batch_size_sum
            .fetch_add(size as u64, Ordering::Relaxed);
        self.batch_size_max
            .fetch_max(size as u64, Ordering::Relaxed);
        self.queue_wait_us_sum
            .fetch_add(queue_wait_us, Ordering::Relaxed);
        self.forward_us_sum.fetch_add(forward_us, Ordering::Relaxed);
        let mut s = self.samples.lock().unwrap();
        if s.batch_sizes.len() < MAX_SAMPLES {
            s.batch_sizes.push(size as u32);
            s.queue_waits_us.push(queue_wait_us);
        }
    }

    pub fn snapshot(&self) -> MetricsSnapshot {
        let s = self.samples.lock().unwrap();
        MetricsSnapshot {
            submitted: self.submitted.load(Ordering::Relaxed),
            completed: self.completed.load(Ordering::Relaxed),
            errors: self.errors.load(Ordering::Relaxed),
            batches: self.batches.load(Ordering::Relaxed),
            batch_size_mean: mean_u64(
                self.batch_size_sum.load(Ordering::Relaxed),
                self.batches.load(Ordering::Relaxed),
            ),
            batch_size_max: self.batch_size_max.load(Ordering::Relaxed),
            batch_size_p50: percentile_u32(&s.batch_sizes, 0.50),
            batch_size_p95: percentile_u32(&s.batch_sizes, 0.95),
            queue_wait_us_mean: mean_u64(
                self.queue_wait_us_sum.load(Ordering::Relaxed),
                self.batches.load(Ordering::Relaxed),
            ),
            queue_wait_us_p50: percentile_u64(&s.queue_waits_us, 0.50),
            queue_wait_us_p95: percentile_u64(&s.queue_waits_us, 0.95),
            forward_us_mean: mean_u64(
                self.forward_us_sum.load(Ordering::Relaxed),
                self.batches.load(Ordering::Relaxed),
            ),
            peak_in_flight: self.peak_in_flight.load(Ordering::Relaxed),
            owners: self.owners.load(Ordering::Relaxed),
        }
    }
}

fn mean_u64(sum: u64, count: u64) -> f64 {
    if count == 0 {
        0.0
    } else {
        sum as f64 / count as f64
    }
}
fn percentile_u32(v: &[u32], q: f64) -> u32 {
    if v.is_empty() {
        return 0;
    }
    let mut c = v.to_vec();
    c.sort_unstable();
    c[((c.len() as f64 - 1.0) * q).round() as usize]
}
fn percentile_u64(v: &[u64], q: f64) -> u64 {
    if v.is_empty() {
        return 0;
    }
    let mut c = v.to_vec();
    c.sort_unstable();
    c[((c.len() as f64 - 1.0) * q).round() as usize]
}

/// A snapshot of batcher metrics.
#[derive(Debug, Clone, serde::Serialize)]
pub struct MetricsSnapshot {
    pub submitted: u64,
    pub completed: u64,
    pub errors: u64,
    pub batches: u64,
    pub batch_size_mean: f64,
    pub batch_size_max: u64,
    pub batch_size_p50: u32,
    pub batch_size_p95: u32,
    pub queue_wait_us_mean: f64,
    pub queue_wait_us_p50: u64,
    pub queue_wait_us_p95: u64,
    pub forward_us_mean: f64,
    /// Peak simultaneous in-flight evaluator calls (real concurrency gauge).
    pub peak_in_flight: usize,
    /// Owner threads that served the queue (1 = the original single owner).
    pub owners: usize,
}

/// Owns the inference thread(s) and their shared request channel.
pub struct InferenceOwner {
    tx: SyncSender<Request>,
    cancel: Arc<AtomicBool>,
    metrics: Arc<InferenceMetrics>,
    handles: Vec<JoinHandle<()>>,
    needs_facts: bool,
}

impl InferenceOwner {
    /// Spawn the owner thread with the given model.
    pub fn spawn<M: BatchEvaluator>(model: M, cfg: InferenceConfig) -> Self {
        Self::spawn_pool(vec![model], cfg)
    }

    /// Spawn one owner thread per model, all serving one request queue (T2).
    ///
    /// Each owner forms a batch while holding the queue, then releases it and
    /// runs its forward, so one owner's batch assembly and kernel submission
    /// overlap another's GPU execution. Every model must be the same network;
    /// each owner's thread has its own device stream. Batching rules
    /// (`max_batch`, `batch_timeout`) apply per batch, as for one owner.
    pub fn spawn_pool<M: BatchEvaluator>(models: Vec<M>, cfg: InferenceConfig) -> Self {
        assert!(
            !models.is_empty(),
            "an inference pool needs at least one model"
        );
        let needs_facts = models[0].needs_candidate_facts();
        assert!(
            models
                .iter()
                .all(|m| m.needs_candidate_facts() == needs_facts),
            "every model in an inference pool must be the same network"
        );
        let (tx, rx) = sync_channel::<Request>(cfg.channel_bound.max(1));
        let rx = Arc::new(Mutex::new(rx));
        let cancel = Arc::new(AtomicBool::new(false));
        let metrics = Arc::new(InferenceMetrics::default());
        metrics.owners.store(models.len(), Ordering::Relaxed);
        let handles = models
            .into_iter()
            .enumerate()
            .map(|(i, model)| {
                let (rx, thread_tx) = (rx.clone(), tx.clone());
                let (thread_cancel, thread_metrics) = (cancel.clone(), metrics.clone());
                std::thread::Builder::new()
                    .name(format!("recur64-inference-{i}"))
                    .spawn(move || {
                        owner_loop(model, rx, thread_tx, thread_cancel, thread_metrics, cfg);
                    })
                    .expect("spawn inference owner")
            })
            .collect();
        Self {
            tx,
            cancel,
            metrics,
            handles,
            needs_facts,
        }
    }

    /// Create a blocking evaluator handle.
    pub fn evaluator(&self) -> BatchedEvaluator {
        BatchedEvaluator {
            tx: self.tx.clone(),
            metrics: self.metrics.clone(),
            needs_facts: self.needs_facts,
        }
    }

    pub fn metrics(&self) -> Arc<InferenceMetrics> {
        self.metrics.clone()
    }

    /// Signal cancellation, drain pending requests with `Shutdown`, and join.
    pub fn shutdown(mut self) {
        self.stop();
    }

    fn stop(&mut self) {
        self.cancel.store(true, Ordering::SeqCst);
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

impl Drop for InferenceOwner {
    fn drop(&mut self) {
        self.stop();
    }
}

fn owner_loop<M: BatchEvaluator>(
    model: M,
    shared_rx: Arc<Mutex<Receiver<Request>>>,
    _keepalive: SyncSender<Request>,
    cancel: Arc<AtomicBool>,
    metrics: Arc<InferenceMetrics>,
    cfg: InferenceConfig,
) {
    loop {
        // Form one batch while holding the queue; the forward runs after the
        // lock is released, so other owners can form the next batch meanwhile.
        let rx = shared_rx.lock().unwrap_or_else(|p| p.into_inner());
        // Wait for the first request of a batch, checking cancellation.
        let first = match rx.recv_timeout(cfg.idle_timeout) {
            Ok(r) => r,
            Err(RecvTimeoutError::Timeout) => {
                if cancel.load(Ordering::SeqCst) {
                    drain(&rx, &metrics);
                    return;
                }
                continue;
            }
            Err(RecvTimeoutError::Disconnected) => {
                drain(&rx, &metrics);
                return;
            }
        };
        if cancel.load(Ordering::SeqCst) {
            let _ = first.respond.send(Err(EvalError::Shutdown));
            metrics.errors.fetch_add(1, Ordering::Relaxed);
            drain(&rx, &metrics);
            return;
        }

        let mut batch = vec![first];
        let deadline = Instant::now() + cfg.batch_timeout;
        let mut timed_out = false;
        while batch.len() < cfg.max_batch {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                timed_out = true;
                break;
            }
            match rx.recv_timeout(remaining) {
                Ok(r) => batch.push(r),
                Err(RecvTimeoutError::Timeout) => {
                    timed_out = true;
                    break;
                }
                Err(RecvTimeoutError::Disconnected) => break,
            }
        }
        if timed_out && batch.len() < cfg.max_batch {
            metrics.timeout_flushes.fetch_add(1, Ordering::Relaxed);
        }
        drop(rx);

        let size = batch.len();
        let queue_wait = batch
            .iter()
            .map(|r| r.submitted_at.elapsed().as_micros() as u64)
            .max()
            .unwrap_or(0);
        let observations: Vec<ObservationV1> =
            batch.iter().map(|r| r.observation.clone()).collect();
        let legal: Vec<Vec<ActionId>> = batch.iter().map(|r| r.legal.clone()).collect();
        let facts: Vec<Option<Vec<CandidateFactsV1>>> =
            batch.iter().map(|r| r.facts.clone()).collect();

        let t0 = Instant::now();
        let results = model.evaluate_batch_with_facts(&observations, &legal, &facts);
        let forward_us = t0.elapsed().as_micros() as u64;

        metrics.record_batch(size, queue_wait, forward_us);

        match results {
            Ok(v) if v.len() == size => {
                // Count before replying, so a requester never reads stale
                // metrics after receiving its result.
                for (r, res) in batch.into_iter().zip(v) {
                    metrics.completed.fetch_add(1, Ordering::Relaxed);
                    let _ = r.respond.send(Ok(res));
                }
            }
            Ok(_) => {
                for r in batch {
                    metrics.errors.fetch_add(1, Ordering::Relaxed);
                    let _ = r.respond.send(Err(EvalError::Backend(
                        "batch result length mismatch".into(),
                    )));
                }
            }
            Err(e) => {
                for r in batch {
                    metrics.errors.fetch_add(1, Ordering::Relaxed);
                    let _ = r.respond.send(Err(e.clone()));
                }
            }
        }
    }
}

/// Answer any requests still queued with `Shutdown`.
fn drain(rx: &Receiver<Request>, metrics: &InferenceMetrics) {
    while let Ok(r) = rx.try_recv() {
        let _ = r.respond.send(Err(EvalError::Shutdown));
        metrics.errors.fetch_add(1, Ordering::Relaxed);
    }
}

/// Blocking evaluator that submits to the owner.
pub struct BatchedEvaluator {
    tx: SyncSender<Request>,
    metrics: Arc<InferenceMetrics>,
    needs_facts: bool,
}

impl Evaluator for BatchedEvaluator {
    fn needs_candidate_facts(&self) -> bool {
        self.needs_facts
    }

    fn evaluate(&self, request: EvalRequest<'_>) -> Result<EvalResult, EvalError> {
        // Real-concurrency gauge: count simultaneous evaluator calls so a
        // configured `cpu_workers`/`active_games` can be checked against actual
        // concurrent execution (see MetricsSnapshot::peak_in_flight).
        let now = self.metrics.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        self.metrics.peak_in_flight.fetch_max(now, Ordering::SeqCst);
        let result = self.evaluate_inner(request);
        self.metrics.in_flight.fetch_sub(1, Ordering::SeqCst);
        result
    }

    /// Submit every request before waiting on any, so a multi-leaf round
    /// (D47) can share one inference batch. Results are in request order.
    fn evaluate_many(&self, requests: &[EvalRequest<'_>]) -> Vec<Result<EvalResult, EvalError>> {
        let n = requests.len();
        let now = self.metrics.in_flight.fetch_add(n, Ordering::SeqCst) + n;
        self.metrics.peak_in_flight.fetch_max(now, Ordering::SeqCst);
        let pending: Vec<Result<Receiver<Result<EvalResult, EvalError>>, EvalError>> =
            requests.iter().map(|r| self.submit(r)).collect();
        let out = pending
            .into_iter()
            .map(|p| p.and_then(|rx| rx.recv().unwrap_or(Err(EvalError::Shutdown))))
            .collect();
        self.metrics.in_flight.fetch_sub(n, Ordering::SeqCst);
        out
    }
}

impl BatchedEvaluator {
    fn evaluate_inner(&self, request: EvalRequest<'_>) -> Result<EvalResult, EvalError> {
        self.submit(&request)?
            .recv()
            .unwrap_or(Err(EvalError::Shutdown))
    }

    /// Queue one request for the owner; returns its response channel.
    fn submit(
        &self,
        request: &EvalRequest<'_>,
    ) -> Result<Receiver<Result<EvalResult, EvalError>>, EvalError> {
        if self.needs_facts {
            match request.facts {
                Some(f) if f.len() == request.legal.len() => {}
                Some(f) => {
                    return Err(EvalError::Invalid(format!(
                        "{} fact rows for {} legal actions",
                        f.len(),
                        request.legal.len()
                    )));
                }
                None => {
                    return Err(EvalError::Invalid(
                        "this evaluator requires CandidateFacts and the request carries none"
                            .into(),
                    ));
                }
            }
        }
        let (respond, response_rx) = sync_channel(1);
        let msg = Request {
            observation: request.observation.clone(),
            legal: request.legal.to_vec(),
            facts: request
                .facts
                .filter(|_| self.needs_facts)
                .map(<[CandidateFactsV1]>::to_vec),
            submitted_at: Instant::now(),
            respond,
        };
        match self.tx.try_send(msg) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {
                return Err(EvalError::Backend(
                    "inference queue is full (backpressure)".into(),
                ));
            }
            Err(TrySendError::Disconnected(_)) => {
                return Err(EvalError::Shutdown);
            }
        }
        self.metrics.submitted.fetch_add(1, Ordering::Relaxed);
        Ok(response_rx)
    }
}
