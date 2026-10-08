//! Extended, source-bound, data-only audit of an accepted V69 dataset.
//!
//! Role: `Role::DataAudit`. Every file open (manifest entries included) goes
//! through `Access`/custody. It may read pool/ and sealed/ solely to verify
//! integrity; no model code shares this module's access role.

use crate::access::Access;
use crate::canon::{canonical_key, key_hex, key_id};
use crate::dataset::*;
use crate::features::{ModelRow, featurize};
use crate::generate::Family;
use crate::oracle::{Oracle, Verdict};
use crate::provenance::{SourceId, sha256_hex, source_id};
use crate::streams::MasterSeed;
use anyhow::{Context, Result};
use cozy_chess::Board;
use serde::Serialize;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Serialize, Debug)]
pub struct AuditV2 {
    pub audit_pass: bool,
    pub failures: Vec<String>,
    pub checks_run: Vec<String>,
    pub input_hashes: BTreeMap<String, String>,
    pub seed_fingerprint: String,
    pub consumer_source: SourceId,
    pub examples: usize,
    pub pool_roots: usize,
    pub rebuilt_groups: usize,
    pub examples_requeried_exact: usize,
}

const FILES: [(&str, &str, Partition); 3] = [
    ("data/fit.jsonl", "meta/fit.meta.jsonl", Partition::Fit),
    ("data/val.jsonl", "meta/val.meta.jsonl", Partition::Val),
    ("sealed/test.jsonl", "sealed/test.meta.jsonl", Partition::Test),
];

fn jsonl<T: serde::de::DeserializeOwned>(text: &str, what: &str) -> Result<Vec<T>> {
    text.lines()
        .enumerate()
        .map(|(i, l)| serde_json::from_str(l).with_context(|| format!("{what} line {i}")))
        .collect()
}

pub fn run_audit_v2(access: &Access, run_rel: &Path, seed_rel: &Path, node_limit: u64, threads: usize) -> Result<AuditV2> {
    let mut failures: Vec<String> = Vec::new();
    let mut checks: Vec<String> = Vec::new();
    macro_rules! check {
        ($name:expr, $ok:expr, $msg:expr) => {
            if !$ok {
                failures.push(format!("[{}] {}", $name, $msg));
            }
        };
    }
    let rd = |rel: &str| -> Result<Vec<u8>> { access.read(&run_rel.join(rel)) };

    // --- manifest: every entry opened via custody; hashes verified ---
    let manifest: BTreeMap<String, String> = serde_json::from_slice(&rd("MANIFEST.sha256.json")?)?;
    let mut input_hashes = BTreeMap::new();
    for (f, h) in &manifest {
        let actual = sha256_hex(&rd(f).with_context(|| format!("manifest entry {f}"))?);
        check!("manifest", actual == *h, format!("hash mismatch {f}"));
        input_hashes.insert(f.clone(), actual);
    }
    checks.push("manifest hashes (custody-routed opens)".into());
    if !failures.is_empty() {
        // Unverified inputs: stop here, never audit files whose hashes disagree.
        return Ok(AuditV2 {
            audit_pass: false,
            failures,
            checks_run: checks,
            input_hashes,
            seed_fingerprint: String::new(),
            consumer_source: source_id(),
            examples: 0,
            pool_roots: 0,
            rebuilt_groups: 0,
            examples_requeried_exact: 0,
        });
    }

    let seed_hex = String::from_utf8(access.read(seed_rel)?)?;
    let seed = MasterSeed::from_hex(seed_hex.trim())?;
    let pool: Vec<RootRec> = jsonl(&String::from_utf8(rd("pool/roots.jsonl")?)?, "pool")?;

    // --- rows vs metadata, by unique id ---
    let mut metas: Vec<Example> = Vec::new();
    let mut seen_ids: HashSet<String> = HashSet::new();
    for (data, meta, part) in FILES {
        let rows: Vec<ModelRow> = jsonl(&String::from_utf8(rd(data)?)?, data)?;
        let ms: Vec<Example> = jsonl(&String::from_utf8(rd(meta)?)?, meta)?;
        let mut by_id: HashMap<&str, &Example> = HashMap::new();
        for m in &ms {
            check!("rows-vs-meta", by_id.insert(&m.id, m).is_none(), format!("duplicate meta id {} in {meta}", m.id));
            check!("rows-vs-meta", m.partition == part, format!("meta {} partition {:?} in {meta}", m.id, m.partition));
            check!("rows-vs-meta", seen_ids.insert(m.id.clone()), format!("id {} appears in two partitions/files", m.id));
        }
        let mut row_ids = HashSet::new();
        for r in &rows {
            check!("rows-vs-meta", row_ids.insert(r.id.as_str()), format!("duplicate row id {} in {data}", r.id));
            match by_id.get(r.id.as_str()) {
                None => failures.push(format!("[rows-vs-meta] row {} in {data} has no metadata", r.id)),
                Some(m) => {
                    check!("rows-vs-meta", m.fen == r.fen, format!("FEN differs for {}", r.id));
                    check!("rows-vs-meta", m.budget == r.budget, format!("budget differs for {}", r.id));
                    check!("rows-vs-meta", m.label == r.label, format!("label differs for {}", r.id));
                    // model-input domain validation (fails visibly on out-of-domain input)
                    match featurize(&r.fen, r.budget) {
                        Ok((_, fam)) => check!("rows-vs-meta", fam.name() == m.family, format!("family differs for {}", r.id)),
                        Err(e) => failures.push(format!("[featurize] {}: {e}", r.id)),
                    }
                }
            }
        }
        for m in &ms {
            check!("rows-vs-meta", row_ids.contains(m.id.as_str()), format!("meta {} has no model row in {data}", m.id));
        }
        check!("rows-vs-meta", rows.len() == ms.len(), format!("{data}: {} rows vs {} meta", rows.len(), ms.len()));
        metas.extend(ms);
    }
    checks.push("model rows == metadata by id (fen,budget,label,partition,family; no missing/extra/duplicate)".into());

    // --- recompute canonical identities ---
    let mut pool_by_id: HashMap<&str, &RootRec> = HashMap::new();
    for r in &pool {
        let rb: Board = r.fen.parse().context("root fen")?;
        let att = rb.side_to_move();
        let k = key_hex(&canonical_key(&rb, att));
        check!("canon", k == r.key, format!("root key mismatch {}", r.id));
        check!("canon", key_id(&r.key) == r.id, format!("root id not derived from key {}", r.id));
        check!("canon", pool_by_id.insert(&r.id, r).is_none(), format!("duplicate root id {}", r.id));
        for c in &r.children {
            let cb: Board = c.fen.parse().context("child fen")?;
            check!("canon", key_hex(&canonical_key(&cb, att)) == c.key, format!("child key mismatch root {} move {}", r.id, c.mv));
        }
    }
    for m in &metas {
        let cb: Board = m.fen.parse()?;
        let rb: Board = m.root_fen.parse()?;
        check!("canon", key_hex(&canonical_key(&cb, rb.side_to_move())) == m.key, format!("example key mismatch {}", m.id));
    }
    checks.push("root/child/example canonical identities recomputed from FEN".into());

    // --- rebuild groups + deterministic partitions + selection from pool and seed ---
    let groups = build_groups(&seed, &pool);
    let mut root_group: HashMap<&str, usize> = HashMap::new();
    for (gi, g) in groups.iter().enumerate() {
        for &ri in &g.roots {
            root_group.insert(&pool[ri].id, gi);
        }
    }
    for m in &metas {
        match root_group.get(m.root_id.as_str()) {
            None => failures.push(format!("[groups] example {} root {} not in pool", m.id, m.root_id)),
            Some(&gi) => {
                check!("groups", groups[gi].id == m.group_id, format!("group id differs for {}", m.id));
                check!("groups", groups[gi].partition == m.partition, format!("partition differs from rebuilt assignment for {}", m.id));
            }
        }
    }
    let sel = select_examples(&seed, &pool, &groups);
    let rebuilt: HashMap<&str, &Example> = sel.examples.iter().map(|e| (e.id.as_str(), e)).collect();
    check!("selection", rebuilt.len() == metas.len(), format!("rebuilt selection has {} examples, stored {}", rebuilt.len(), metas.len()));
    for m in &metas {
        match rebuilt.get(m.id.as_str()) {
            None => failures.push(format!("[selection] stored example {} not reproduced", m.id)),
            Some(e) => check!(
                "selection",
                e.fen == m.fen && e.label == m.label && e.root_id == m.root_id && e.group_id == m.group_id && e.partition == m.partition && e.mv == m.mv && e.budget == m.budget,
                format!("selection differs for {}", m.id)
            ),
        }
    }
    // groups never span partitions (all roots in a group share the assignment by construction; verify stored)
    let mut gp: HashMap<&str, HashSet<Partition>> = HashMap::new();
    for m in &metas {
        gp.entry(&m.group_id).or_default().insert(m.partition);
    }
    for (g, ps) in &gp {
        check!("groups", ps.len() == 1, format!("group {g} spans partitions"));
    }
    checks.push("groups, partitions and selection rebuilt from pool+seed and compared".into());

    // --- each example is the claimed root's child with the exact target ---
    let mut to_query: Vec<&Example> = Vec::new();
    for m in &metas {
        let Some(root) = pool_by_id.get(m.root_id.as_str()) else { continue };
        check!("child", root.depth == m.root_depth, format!("root depth differs {}", m.id));
        check!("child", m.budget + 1 == root.depth, format!("budget != depth-1 for {}", m.id));
        check!("child", root.fen == m.root_fen, format!("root fen differs {}", m.id));
        match root.children.iter().find(|c| c.mv == m.mv) {
            None => failures.push(format!("[child] {} move {} not a legal root move", m.id, m.mv)),
            Some(c) => {
                check!("child", c.fen == m.fen && c.key == m.key, format!("child identity differs {}", m.id));
                let expect = match c.status.as_str() {
                    "pos" => Some(true),
                    "neg" => Some(false),
                    _ => None,
                };
                check!("child", expect == Some(m.label), format!("stored child status {} vs label {} for {}", c.status, m.label, m.id));
            }
        }
        to_query.push(m);
    }
    // fresh exact re-query of every selected example's target (empty cache per query)
    let next = AtomicUsize::new(0);
    let bad = std::sync::Mutex::new(Vec::<String>::new());
    std::thread::scope(|sc| {
        for _ in 0..threads.max(1) {
            sc.spawn(|| {
                let mut o = Oracle::new(node_limit);
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= to_query.len() {
                        break;
                    }
                    let m = to_query[i];
                    o.clear_cache();
                    let cb: Board = m.fen.parse().unwrap();
                    let att = !cb.side_to_move();
                    let v = o.mate_within_child(&cb, att, m.budget);
                    let ok = match v {
                        Verdict::Yes => m.label,
                        Verdict::No => !m.label,
                        Verdict::Unknown => false,
                    };
                    if !ok {
                        bad.lock().unwrap().push(format!("[exact-target] {} label {} but oracle {:?}", m.id, m.label, v));
                    }
                }
            });
        }
    });
    failures.extend(bad.into_inner().unwrap());
    // roots: re-derive minimal depth for contributing roots with a fresh oracle
    let used: HashSet<&str> = metas.iter().map(|m| m.root_id.as_str()).collect();
    let used_roots: Vec<&RootRec> = pool.iter().filter(|r| used.contains(r.id.as_str())).collect();
    let next = AtomicUsize::new(0);
    let bad = std::sync::Mutex::new(Vec::<String>::new());
    std::thread::scope(|sc| {
        for _ in 0..threads.max(1) {
            sc.spawn(|| {
                let mut o = Oracle::new(node_limit);
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= used_roots.len() {
                        break;
                    }
                    let r = used_roots[i];
                    let b: Board = r.fen.parse().unwrap();
                    let fam = Family::ALL.iter().copied().find(|f| f.name() == r.family).unwrap();
                    match analyze_root(&mut o, fam, &b, b.side_to_move()) {
                        RootOutcome::Accepted(r2) => {
                            if r2.depth != r.depth || r2.children.len() != r.children.len() || r2.children.iter().zip(&r.children).any(|(a, b)| a.mv != b.mv || a.status != b.status) {
                                bad.lock().unwrap().push(format!("[root-reanalysis] {} differs", r.id));
                            }
                        }
                        _ => bad.lock().unwrap().push(format!("[root-reanalysis] {} not accepted on re-analysis", r.id)),
                    }
                }
            });
        }
    });
    failures.extend(bad.into_inner().unwrap());
    checks.push("every example: claimed root/child membership, stored status, fresh exact target re-query; every contributing root re-analysed".into());

    // --- quotas ---
    let mut counts: BTreeMap<(Partition, String, u8, bool), usize> = BTreeMap::new();
    for m in &metas {
        *counts.entry((m.partition, m.family.clone(), m.budget, m.label)).or_default() += 1;
    }
    for p in Partition::ALL {
        for f in Family::ALL {
            for b in [1u8, 2] {
                for c in [false, true] {
                    let n = counts.get(&(p, f.name().to_string(), b, c)).copied().unwrap_or(0);
                    check!("quota", n == p.quota(), format!("{}/{}/n{b}/{c}: {n} != {}", p.name(), f.name(), p.quota()));
                }
            }
        }
    }
    checks.push("quotas".into());

    Ok(AuditV2 {
        audit_pass: failures.is_empty(),
        failures,
        checks_run: checks,
        input_hashes,
        seed_fingerprint: seed.fingerprint(),
        consumer_source: source_id(),
        examples: metas.len(),
        pool_roots: pool.len(),
        rebuilt_groups: groups.len(),
        examples_requeried_exact: to_query.len(),
    })
}

/// Convenience used by the CLI: run and write `audit_receipt_v2.json` beside the dataset.
pub fn run_and_write(access: &Access, run_rel: &Path, seed_rel: &Path, node_limit: u64, threads: usize) -> Result<AuditV2> {
    let r = run_audit_v2(access, run_rel, seed_rel, node_limit, threads)?;
    let name = run_rel.file_name().and_then(|s| s.to_str()).unwrap_or("run");
    let out: PathBuf = Path::new("audit").join(format!("{name}_audit_receipt_v2.json"));
    access.write(&out, serde_json::to_string_pretty(&r)?.as_bytes())?;
    Ok(r)
}
