//! Sharded, resumable storage for `ProofTraceV1`, its independent audit, and the
//! frozen P4 feasibility table.
//!
//! * Shard assignment is a pure function of the source dataset order
//!   (`shard i` = positions `[i * size, (i + 1) * size)`), so output is independent
//!   of thread count and of whether a run was interrupted.
//! * A shard file is reused only after it is re-validated (schema, source digest,
//!   index, range, ids, content digest). A corrupted or mismatched shard is a hard
//!   error, never silently regenerated or skipped.
//! * The audit is sharded and resumable in the same way, and the feasibility table
//!   refuses to run unless a complete, failure-free audit covers every shard.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::mate::MateSolver;
use super::targets::{ProofTargets, Split};
use super::trace::{PositionTrace, TRACE_DEFINITION, TRACE_SCHEMA, build_trace};
use super::trace_audit::{TraceAuditMemo, audit_trace};

pub const SHARD_SIZE: usize = 500;
pub const SHARD_SCHEMA: &str = "proof_trace_shard_v1";
pub const MANIFEST_SCHEMA: &str = "proof_trace_manifest_v1";
pub const AUDIT_SHARD_SCHEMA: &str = "proof_trace_audit_shard_v1";
pub const AUDIT_MANIFEST_SCHEMA: &str = "proof_trace_audit_manifest_v1";
pub const FEASIBILITY_SCHEMA: &str = "v3_p4_feasibility_v1";
/// Version of the generator code path. The scientific output does not depend on it
/// beyond the contract; it is recorded for provenance.
pub const GENERATOR_VERSION: &str = "proof_trace_v1/gen1";

/// Frozen P4 rule (docs/V3_RESEARCH_PLAN.md, V3-D3): primary cell and threshold.
pub const PRIMARY_FAMILY: &str = "KQRvK";
pub const PRIMARY_DEPTH: u8 = 3;
pub const PRIMARY_BUDGET: u64 = 8;
/// `C_8 >= 1/4`, evaluated exactly as `4 * count >= n`.
pub const THRESHOLD_NUM: u64 = 1;
pub const THRESHOLD_DEN: u64 = 4;
pub const CLASS_QUALIFIED: &str = "SCIENTIFICALLY QUALIFIED FOR THE B8 PRIMARY EXPERIMENT";
pub const CLASS_NOT_QUALIFIED: &str = "NOT SCIENTIFICALLY QUALIFIED / BUDGET MIS-SPECIFIED";

fn hex(h: Sha256) -> String {
    format!("{:x}", h.finalize())
}

/// Write via a temporary file and rename, so an interrupted write never leaves a
/// half-written shard that could later pass for a complete one.
fn write_atomic(path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceShard {
    pub schema: String,
    pub trace_schema: String,
    pub trace_definition: String,
    pub generator_version: String,
    pub source_dataset_digest: String,
    pub source_split: Split,
    pub source_positions: usize,
    pub shard_index: usize,
    pub first_position: usize,
    pub count: usize,
    pub positions: Vec<PositionTrace>,
    pub digest: String,
}

impl TraceShard {
    pub fn compute_digest(&self) -> String {
        let mut h = Sha256::new();
        for s in [
            &self.schema,
            &self.trace_schema,
            &self.trace_definition,
            &self.generator_version,
            &self.source_dataset_digest,
        ] {
            h.update((s.len() as u64).to_le_bytes());
            h.update(s.as_bytes());
        }
        h.update(self.source_split.label().as_bytes());
        for v in [
            self.source_positions,
            self.shard_index,
            self.first_position,
            self.count,
        ] {
            h.update((v as u64).to_le_bytes());
        }
        for p in &self.positions {
            h.update(serde_json::to_vec(p).unwrap_or_default());
        }
        hex(h)
    }

    /// Validate against the source dataset: refuses anything that is not exactly the
    /// shard this source and index would produce structurally.
    pub fn validate(&self, src: &ProofTargets, index: usize, size: usize) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.schema == SHARD_SCHEMA,
            "unknown shard schema '{}'",
            self.schema
        );
        anyhow::ensure!(
            self.trace_schema == TRACE_SCHEMA && self.trace_definition == TRACE_DEFINITION,
            "shard {index}: trace contract differs from the current proof_trace_v1"
        );
        anyhow::ensure!(
            self.source_dataset_digest == src.digest && self.source_split == src.split,
            "shard {index}: source dataset digest/split differs ({} vs {})",
            self.source_dataset_digest,
            src.digest
        );
        let first = index * size;
        let count = size.min(src.positions.len().saturating_sub(first));
        anyhow::ensure!(
            self.shard_index == index
                && self.first_position == first
                && self.count == count
                && self.positions.len() == count
                && self.source_positions == src.positions.len(),
            "shard {index}: range/count mismatch"
        );
        for (t, p) in self
            .positions
            .iter()
            .zip(&src.positions[first..first + count])
        {
            anyhow::ensure!(
                t.id == p.id && t.fen == p.fen,
                "shard {index}: trace {} does not match source position {}",
                t.id,
                p.id
            );
        }
        anyhow::ensure!(
            self.digest == self.compute_digest(),
            "shard {index}: content digest mismatch (the shard file was modified or corrupted)"
        );
        Ok(())
    }
}

/// The datasets P4 may trace. HOLDOUT_C (and every other holdout or confirmation
/// set) is never traced.
pub fn ensure_traceable(src: &ProofTargets) -> anyhow::Result<()> {
    anyhow::ensure!(
        matches!(src.split, Split::Train | Split::Tune),
        "refusing to trace a {} dataset: only TRAIN and V3 TUNE are traced (confirmation and \
         holdout sets stay sealed)",
        src.split.label()
    );
    anyhow::ensure!(
        src.digest != super::custody::HOLDOUT_C_DIGEST,
        "refusing to trace HOLDOUT_C"
    );
    Ok(())
}

pub fn shard_count(n: usize, size: usize) -> usize {
    n.div_ceil(size)
}

pub fn shard_path(dir: &Path, index: usize) -> PathBuf {
    dir.join(format!("trace-shard-{index:05}.json"))
}

fn audit_shard_path(dir: &Path, index: usize) -> PathBuf {
    dir.join(format!("audit-shard-{index:05}.json"))
}

/// Trace the positions of one shard on `threads` threads. The result depends only
/// on the positions: contiguous chunks are processed independently and re-joined in
/// order, and each trace is a deterministic function of its position.
pub fn generate_shard(
    src: &ProofTargets,
    index: usize,
    size: usize,
    threads: usize,
) -> anyhow::Result<TraceShard> {
    let first = index * size;
    anyhow::ensure!(first < src.positions.len(), "shard {index} is out of range");
    let count = size.min(src.positions.len() - first);
    let slice = &src.positions[first..first + count];
    let threads = threads.clamp(1, count);
    let chunk = count.div_ceil(threads);
    let parts: Vec<anyhow::Result<Vec<PositionTrace>>> = std::thread::scope(|scope| {
        let hs: Vec<_> = slice
            .chunks(chunk)
            .map(|c| {
                scope.spawn(move || {
                    let mut solver = MateSolver::new();
                    let mut out = Vec::with_capacity(c.len());
                    for p in c {
                        if solver.table_size() > 3_000_000 {
                            solver = MateSolver::new();
                        }
                        out.push(build_trace(&mut solver, p)?);
                    }
                    Ok(out)
                })
            })
            .collect();
        hs.into_iter()
            .map(|h| h.join().expect("trace thread"))
            .collect()
    });
    let mut positions = Vec::with_capacity(count);
    for p in parts {
        positions.extend(p?);
    }
    let mut s = TraceShard {
        schema: SHARD_SCHEMA.into(),
        trace_schema: TRACE_SCHEMA.into(),
        trace_definition: TRACE_DEFINITION.into(),
        generator_version: GENERATOR_VERSION.into(),
        source_dataset_digest: src.digest.clone(),
        source_split: src.split,
        source_positions: src.positions.len(),
        shard_index: index,
        first_position: first,
        count,
        positions,
        digest: String::new(),
    };
    s.digest = s.compute_digest();
    Ok(s)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardRef {
    pub index: usize,
    pub first_position: usize,
    pub count: usize,
    pub digest: String,
    pub file: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceManifest {
    pub schema: String,
    pub trace_schema: String,
    pub trace_definition: String,
    pub generator_version: String,
    pub source_dataset_digest: String,
    pub source_split: Split,
    pub source_positions: usize,
    pub shard_size: usize,
    pub shards: Vec<ShardRef>,
    pub manifest_digest: String,
}

impl TraceManifest {
    pub fn compute_digest(&self) -> String {
        let mut h = Sha256::new();
        for s in [
            &self.schema,
            &self.trace_schema,
            &self.trace_definition,
            &self.generator_version,
            &self.source_dataset_digest,
        ] {
            h.update((s.len() as u64).to_le_bytes());
            h.update(s.as_bytes());
        }
        h.update(self.source_split.label().as_bytes());
        h.update((self.source_positions as u64).to_le_bytes());
        h.update((self.shard_size as u64).to_le_bytes());
        for s in &self.shards {
            h.update((s.index as u64).to_le_bytes());
            h.update((s.first_position as u64).to_le_bytes());
            h.update((s.count as u64).to_le_bytes());
            h.update(s.digest.as_bytes());
        }
        hex(h)
    }

    pub fn load(dir: &Path) -> anyhow::Result<Self> {
        let path = dir.join("trace-manifest.json");
        let m: TraceManifest = serde_json::from_slice(&std::fs::read(&path)?)
            .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
        anyhow::ensure!(m.schema == MANIFEST_SCHEMA, "unknown manifest schema");
        anyhow::ensure!(
            m.manifest_digest == m.compute_digest(),
            "{}: manifest digest mismatch",
            path.display()
        );
        Ok(m)
    }
}

/// Progress callback arguments: (shard index, shard count, reused).
pub type Progress<'a> = &'a dyn Fn(usize, usize, bool);

/// Generate (or resume) every shard of `src` into `dir` and write the manifest.
pub fn generate_traces(
    src: &ProofTargets,
    dir: &Path,
    size: usize,
    threads: usize,
    progress: Progress<'_>,
) -> anyhow::Result<TraceManifest> {
    ensure_traceable(src)?;
    anyhow::ensure!(size > 0, "shard size must be positive");
    std::fs::create_dir_all(dir)?;
    let total = shard_count(src.positions.len(), size);
    let mut shards = Vec::with_capacity(total);
    for i in 0..total {
        let path = shard_path(dir, i);
        let (shard, reused) = if path.exists() {
            let s: TraceShard = serde_json::from_slice(&std::fs::read(&path)?)
                .map_err(|e| anyhow::anyhow!("{}: unreadable shard: {e}", path.display()))?;
            s.validate(src, i, size).map_err(|e| {
                anyhow::anyhow!(
                    "{}: existing shard is invalid ({e}); delete it to regenerate — it is never \
                     silently reused or overwritten",
                    path.display()
                )
            })?;
            (s, true)
        } else {
            let s = generate_shard(src, i, size, threads)?;
            s.validate(src, i, size)?;
            write_atomic(&path, &serde_json::to_vec(&s)?)?;
            (s, false)
        };
        progress(i, total, reused);
        shards.push(ShardRef {
            index: i,
            first_position: shard.first_position,
            count: shard.count,
            digest: shard.digest.clone(),
            file: path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default(),
        });
    }
    let mut m = TraceManifest {
        schema: MANIFEST_SCHEMA.into(),
        trace_schema: TRACE_SCHEMA.into(),
        trace_definition: TRACE_DEFINITION.into(),
        generator_version: GENERATOR_VERSION.into(),
        source_dataset_digest: src.digest.clone(),
        source_split: src.split,
        source_positions: src.positions.len(),
        shard_size: size,
        shards,
        manifest_digest: String::new(),
    };
    m.manifest_digest = m.compute_digest();
    write_atomic(
        &dir.join("trace-manifest.json"),
        &serde_json::to_vec_pretty(&m)?,
    )?;
    Ok(m)
}

fn load_shard(
    dir: &Path,
    r: &ShardRef,
    src: &ProofTargets,
    size: usize,
) -> anyhow::Result<TraceShard> {
    let path = dir.join(&r.file);
    let s: TraceShard = serde_json::from_slice(&std::fs::read(&path)?)
        .map_err(|e| anyhow::anyhow!("{}: unreadable shard: {e}", path.display()))?;
    s.validate(src, r.index, size)?;
    anyhow::ensure!(
        s.digest == r.digest,
        "{}: shard digest differs from the manifest",
        path.display()
    );
    Ok(s)
}

// ---------------------------------------------------------------------------
// Audit
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditShard {
    pub schema: String,
    pub shard_index: usize,
    pub shard_digest: String,
    pub checked: usize,
    pub failures: Vec<String>,
    pub digest: String,
}

impl AuditShard {
    fn compute_digest(&self) -> String {
        let mut h = Sha256::new();
        h.update(self.schema.as_bytes());
        h.update((self.shard_index as u64).to_le_bytes());
        h.update(self.shard_digest.as_bytes());
        h.update((self.checked as u64).to_le_bytes());
        for f in &self.failures {
            h.update(f.as_bytes());
            h.update(b"\n");
        }
        hex(h)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditShardRef {
    pub index: usize,
    pub shard_digest: String,
    pub checked: usize,
    pub failures: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditManifest {
    pub schema: String,
    pub trace_manifest_digest: String,
    pub source_dataset_digest: String,
    pub shards: Vec<AuditShardRef>,
    pub total_checked: usize,
    pub total_failures: usize,
    /// First few failure messages (the full list is in the audit shard files).
    pub first_failures: Vec<String>,
    pub manifest_digest: String,
}

impl AuditManifest {
    pub fn compute_digest(&self) -> String {
        let mut h = Sha256::new();
        h.update(self.schema.as_bytes());
        h.update(self.trace_manifest_digest.as_bytes());
        h.update(self.source_dataset_digest.as_bytes());
        for s in &self.shards {
            h.update((s.index as u64).to_le_bytes());
            h.update(s.shard_digest.as_bytes());
            h.update((s.checked as u64).to_le_bytes());
            h.update((s.failures as u64).to_le_bytes());
        }
        hex(h)
    }

    pub fn ok(&self) -> bool {
        self.total_failures == 0
    }

    pub fn load(dir: &Path) -> anyhow::Result<Self> {
        let path = dir.join("audit-manifest.json");
        let m: AuditManifest = serde_json::from_slice(&std::fs::read(&path)?)
            .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
        anyhow::ensure!(
            m.schema == AUDIT_MANIFEST_SCHEMA,
            "unknown audit manifest schema"
        );
        anyhow::ensure!(
            m.manifest_digest == m.compute_digest(),
            "{}: audit manifest digest mismatch",
            path.display()
        );
        Ok(m)
    }
}

/// Audit one loaded shard on `threads` threads (independent memo per thread).
fn audit_shard(shard: &TraceShard, src: &ProofTargets, threads: usize) -> AuditShard {
    let slice = &src.positions[shard.first_position..shard.first_position + shard.count];
    let threads = threads.clamp(1, shard.count.max(1));
    let chunk = shard.count.div_ceil(threads).max(1);
    let parts: Vec<Vec<String>> = std::thread::scope(|scope| {
        let hs: Vec<_> = shard
            .positions
            .chunks(chunk)
            .zip(slice.chunks(chunk))
            .map(|(ts, ps)| {
                scope.spawn(move || {
                    let mut memo = TraceAuditMemo::default();
                    ts.iter()
                        .zip(ps)
                        .filter_map(|(t, p)| audit_trace(t, p, &mut memo).err())
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        hs.into_iter()
            .map(|h| h.join().expect("audit thread"))
            .collect()
    });
    let mut a = AuditShard {
        schema: AUDIT_SHARD_SCHEMA.into(),
        shard_index: shard.shard_index,
        shard_digest: shard.digest.clone(),
        checked: shard.count,
        failures: parts.into_iter().flatten().collect(),
        digest: String::new(),
    };
    a.digest = a.compute_digest();
    a
}

/// Audit every shard listed in the trace manifest of `dir` (resumable). Every
/// trace is audited; the result is a manifest with exact totals.
pub fn audit_traces(
    src: &ProofTargets,
    dir: &Path,
    threads: usize,
    progress: Progress<'_>,
) -> anyhow::Result<AuditManifest> {
    ensure_traceable(src)?;
    let tm = TraceManifest::load(dir)?;
    anyhow::ensure!(
        tm.source_dataset_digest == src.digest && tm.source_positions == src.positions.len(),
        "the trace manifest does not belong to this source dataset"
    );
    let total = tm.shards.len();
    let mut refs = Vec::with_capacity(total);
    let mut first_failures = Vec::new();
    for r in &tm.shards {
        let path = audit_shard_path(dir, r.index);
        let (a, reused) = if path.exists() {
            let a: AuditShard = serde_json::from_slice(&std::fs::read(&path)?)
                .map_err(|e| anyhow::anyhow!("{}: unreadable audit shard: {e}", path.display()))?;
            anyhow::ensure!(
                a.schema == AUDIT_SHARD_SCHEMA
                    && a.shard_index == r.index
                    && a.shard_digest == r.digest
                    && a.checked == r.count
                    && a.digest == a.compute_digest(),
                "{}: existing audit shard is invalid or belongs to a different trace shard; delete it \
                 to re-audit",
                path.display()
            );
            (a, true)
        } else {
            let shard = load_shard(dir, r, src, tm.shard_size)?;
            let a = audit_shard(&shard, src, threads);
            write_atomic(&path, &serde_json::to_vec(&a)?)?;
            (a, false)
        };
        progress(r.index, total, reused);
        for f in a.failures.iter().take(5) {
            if first_failures.len() < 20 {
                first_failures.push(f.clone());
            }
        }
        refs.push(AuditShardRef {
            index: r.index,
            shard_digest: a.shard_digest.clone(),
            checked: a.checked,
            failures: a.failures.len(),
        });
    }
    let mut m = AuditManifest {
        schema: AUDIT_MANIFEST_SCHEMA.into(),
        trace_manifest_digest: tm.manifest_digest.clone(),
        source_dataset_digest: src.digest.clone(),
        total_checked: refs.iter().map(|s| s.checked).sum(),
        total_failures: refs.iter().map(|s| s.failures).sum(),
        shards: refs,
        first_failures,
        manifest_digest: String::new(),
    };
    m.manifest_digest = m.compute_digest();
    write_atomic(
        &dir.join("audit-manifest.json"),
        &serde_json::to_vec_pretty(&m)?,
    )?;
    Ok(m)
}

// ---------------------------------------------------------------------------
// Feasibility
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CellStats {
    pub family: String,
    pub mate_depth: u8,
    pub n: u64,
    pub q_min: u64,
    pub q_median: u64,
    pub q_p90: u64,
    pub q_p95: u64,
    pub q_max: u64,
    pub count_le_2: u64,
    pub count_le_4: u64,
    pub count_le_8: u64,
    pub count_le_16: u64,
    pub c2: f64,
    pub c4: f64,
    pub c8: f64,
    pub c16: f64,
}

/// Nearest-rank quantile of a sorted slice.
fn quantile(sorted: &[u64], p: f64) -> u64 {
    let rank = ((p * sorted.len() as f64).ceil() as usize).clamp(1, sorted.len());
    sorted[rank - 1]
}

fn stats_of(family: &str, depth: u8, qs: &mut [u64]) -> CellStats {
    qs.sort_unstable();
    let n = qs.len() as u64;
    let le = |k: u64| qs.iter().filter(|&&q| q <= k).count() as u64;
    let (a, b, c, d) = (le(2), le(4), le(8), le(16));
    let f = |x: u64| x as f64 / n as f64;
    CellStats {
        family: family.to_string(),
        mate_depth: depth,
        n,
        q_min: qs[0],
        q_median: quantile(qs, 0.5),
        q_p90: quantile(qs, 0.9),
        q_p95: quantile(qs, 0.95),
        q_max: qs[qs.len() - 1],
        count_le_2: a,
        count_le_4: b,
        count_le_8: c,
        count_le_16: d,
        c2: f(a),
        c4: f(b),
        c8: f(c),
        c16: f(d),
    }
}

/// The frozen rule `C_8 >= 1/4`, evaluated in exact integer arithmetic so that no
/// float rounding can flip it.
pub fn threshold_pass(n: u64, count_le_8: u64) -> bool {
    n > 0 && count_le_8 * THRESHOLD_DEN >= n * THRESHOLD_NUM
}

#[derive(Debug, Clone, Serialize)]
pub struct PrimaryResult {
    pub primary_cell: String,
    pub primary_metric: &'static str,
    pub threshold: f64,
    pub threshold_rule: String,
    pub n: u64,
    pub count_le_8: u64,
    /// Unrounded.
    pub measured_value: f64,
    pub pass: bool,
    pub classification: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct Feasibility {
    pub schema: &'static str,
    /// "primary" (gating, P25_DATA_V1 TRAIN) or "diagnostic" (never gating).
    pub role: String,
    pub source_dataset_digest: String,
    pub source_split: String,
    pub trace_manifest_digest: String,
    pub audit_manifest_digest: String,
    pub positions: u64,
    pub cells: Vec<CellStats>,
    /// Pooled over families, per mate depth.
    pub pooled_by_depth: Vec<CellStats>,
    /// Pooled over everything.
    pub pooled_all: CellStats,
    /// Unweighted mean of the per-cell C_k over cells.
    pub macro_c: [f64; 4],
    /// Present only for the primary role.
    pub primary: Option<PrimaryResult>,
}

/// Compute the feasibility table from complete, audited traces. Refuses unless the
/// trace manifest is intact, the audit manifest covers every shard with zero
/// failures, and both belong to `src`.
pub fn feasibility(src: &ProofTargets, dir: &Path, role: &str) -> anyhow::Result<Feasibility> {
    ensure_traceable(src)?;
    anyhow::ensure!(
        role == "primary" || role == "diagnostic",
        "role must be primary or diagnostic"
    );
    if role == "primary" {
        anyhow::ensure!(
            src.split == Split::Train,
            "the gating feasibility measurement uses the P25_DATA_V1 TRAIN split only"
        );
    }
    let tm = TraceManifest::load(dir)?;
    let am = AuditManifest::load(dir)?;
    anyhow::ensure!(
        tm.source_dataset_digest == src.digest && tm.source_positions == src.positions.len(),
        "trace manifest does not belong to this source dataset"
    );
    anyhow::ensure!(
        am.trace_manifest_digest == tm.manifest_digest && am.source_dataset_digest == src.digest,
        "the audit does not belong to this trace manifest"
    );
    anyhow::ensure!(
        am.shards.len() == tm.shards.len() && am.total_checked == src.positions.len(),
        "the audit does not cover every shard and position ({} of {})",
        am.total_checked,
        src.positions.len()
    );
    anyhow::ensure!(
        am.ok(),
        "the independent audit reported {} failures; the feasibility measurement is refused. \
         First: {:?}",
        am.total_failures,
        am.first_failures.first()
    );
    let mut by_cell: std::collections::BTreeMap<(String, u8), Vec<u64>> = Default::default();
    let mut by_depth: std::collections::BTreeMap<u8, Vec<u64>> = Default::default();
    let mut all: Vec<u64> = Vec::new();
    for r in &tm.shards {
        let shard = load_shard(dir, r, src, tm.shard_size)?;
        for t in shard.positions {
            by_cell
                .entry((t.family.clone(), t.mate_depth))
                .or_default()
                .push(t.q_star);
            by_depth.entry(t.mate_depth).or_default().push(t.q_star);
            all.push(t.q_star);
        }
    }
    anyhow::ensure!(
        all.len() == src.positions.len(),
        "traced position count differs from the source"
    );
    let cells: Vec<CellStats> = by_cell
        .iter_mut()
        .map(|((f, d), qs)| stats_of(f, *d, qs))
        .collect();
    let pooled_by_depth: Vec<CellStats> = by_depth
        .iter_mut()
        .map(|(d, qs)| stats_of("ALL", *d, qs))
        .collect();
    let pooled_all = stats_of("ALL", 0, &mut all);
    let mean = |f: fn(&CellStats) -> f64| cells.iter().map(f).sum::<f64>() / cells.len() as f64;
    let macro_c = [
        mean(|c| c.c2),
        mean(|c| c.c4),
        mean(|c| c.c8),
        mean(|c| c.c16),
    ];
    if role == "primary" {
        anyhow::ensure!(
            cells
                .iter()
                .any(|c| c.family == PRIMARY_FAMILY && c.mate_depth == PRIMARY_DEPTH),
            "the primary cell {PRIMARY_FAMILY} M{PRIMARY_DEPTH} is absent from the traced dataset"
        );
    }
    let primary = (role == "primary").then(|| {
        let cell = cells
            .iter()
            .find(|c| c.family == PRIMARY_FAMILY && c.mate_depth == PRIMARY_DEPTH);
        let (n, le8) = cell.map_or((0, 0), |c| (c.n, c.count_le_8));
        let pass = threshold_pass(n, le8);
        PrimaryResult {
            primary_cell: format!("{PRIMARY_FAMILY} M{PRIMARY_DEPTH}"),
            primary_metric: "C_8",
            threshold: THRESHOLD_NUM as f64 / THRESHOLD_DEN as f64,
            threshold_rule: format!("count_le_8 * {THRESHOLD_DEN} >= n * {THRESHOLD_NUM} (exact)"),
            n,
            count_le_8: le8,
            measured_value: if n > 0 {
                le8 as f64 / n as f64
            } else {
                f64::NAN
            },
            pass,
            classification: if pass {
                CLASS_QUALIFIED
            } else {
                CLASS_NOT_QUALIFIED
            },
        }
    });
    Ok(Feasibility {
        schema: FEASIBILITY_SCHEMA,
        role: role.to_string(),
        source_dataset_digest: src.digest.clone(),
        source_split: src.split.label().to_string(),
        trace_manifest_digest: tm.manifest_digest,
        audit_manifest_digest: am.manifest_digest,
        positions: src.positions.len() as u64,
        cells,
        pooled_by_depth,
        pooled_all,
        macro_c,
        primary,
    })
}
