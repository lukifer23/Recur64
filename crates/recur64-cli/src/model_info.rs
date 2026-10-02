//! `recur64 model-info` — exact parameter accounting and executed-block counts.

use std::path::Path;

use burn::backend::Flex;
use recur64_model::candidate::CandidateV25Model;
use recur64_model::config::{Architecture, ProbeConfig};
use recur64_model::model::ProbeModel;

/// Executed transformer blocks for the feed-forward R=1 control. Both the
/// feed-forward family and its matched recurrent family execute 8 blocks at
/// R=1, so this is the neural-compute baseline for the multiplier column.
const CONTROL_R1_BLOCKS: f64 = 8.0;

/// Historical F10 parameter count (descriptive context, not a matched control).
fn f10_params() -> usize {
    let cfg = recur64_model::config::ModelConfig {
        width: 384,
        heads: 12,
        ffn: 768,
        input_blocks: 0,
        core_blocks: 8,
        output_blocks: 0,
        squares: 64,
        in_features: 119,
        policy_dim: 128,
        wdl_classes: 3,
        promo_codes: 5,
        rms_eps: 1e-5,
        architecture: Default::default(),
        candidate: None,
        legacy_facts: None,
        active: None,
        all_info: None,
        evidence: None,
    };
    ProbeModel::<Flex>::new(cfg, &Default::default()).num_params()
}

pub fn run_model_info(path: &Path, json: Option<&Path>) -> anyhow::Result<()> {
    let text = std::fs::read_to_string(path)?;
    let cfg = ProbeConfig::from_toml_str(&text)?;
    cfg.model.validate()?;
    for r in &cfg.recurrence {
        cfg.model.check_recurrence(*r)?;
    }
    // Exhaustive on purpose: a new architecture must pick its own report; it can
    // never fall through to the Probe report below.
    match cfg.model.architecture {
        Architecture::CandidateV25 => return run_candidate_info(&cfg, json),
        Architecture::LegacyFactsV25 => return run_legacy_facts_info(&cfg, json),
        Architecture::ActiveSearchV3 => return run_active_info(&cfg, json),
        Architecture::AllInfoV1 => return run_all_info_info(&cfg, json),
        Architecture::EvidenceBeliefV4 => anyhow::bail!(
            "evidence_belief_v4 is described by `recur64 v4 model-info`, not by the historical `model-info`"
        ),
        Architecture::ProbeV1 => {}
    }
    let device = Default::default();
    let model = ProbeModel::<Flex>::new(cfg.model.clone(), &device);

    println!("config          : {}", cfg.name);
    println!("device          : {:?}", cfg.device);
    println!("precision       : {}", cfg.precision.label());
    println!(
        "geometry        : width={} heads={} ffn={} head_dim={}",
        cfg.model.width,
        cfg.model.heads,
        cfg.model.ffn,
        cfg.model.head_dim()
    );
    println!(
        "blocks          : input={} core={} (shared) output={} unique={}",
        cfg.model.input_blocks,
        cfg.model.core_blocks,
        cfg.model.output_blocks,
        cfg.model.unique_blocks()
    );

    println!("\nparameter breakdown (unique; shared counted once):");
    let mut total = 0usize;
    for (name, n) in model.param_breakdown() {
        println!("  {:<22} {:>12}", name, n);
        total += n;
    }
    println!("  {:<22} {:>12}", "TOTAL UNIQUE", total);
    assert_eq!(
        total,
        model.num_params(),
        "breakdown must sum to num_params"
    );

    println!("\nexecuted transformer blocks and compute multiplier:");
    println!(
        "  {:<10} {:>16} {:>18} {:>12}",
        "R", "final blocks", "deep-sup blocks", "mult vs R=1"
    );
    for r in &cfg.recurrence {
        let final_blocks = cfg.model.executed_blocks_final(*r);
        let deep = cfg.model.executed_blocks_deep_supervision(*r);
        let mult = final_blocks as f64 / CONTROL_R1_BLOCKS;
        println!(
            "  R={:<8} {:>16} {:>18} {:>12.2}x",
            r, final_blocks, deep, mult
        );
    }
    println!(
        "\nnote: 'mult vs R=1' = executed_blocks / {:.0} (the 8-block R=1 control).",
        CONTROL_R1_BLOCKS
    );
    Ok(())
}

/// `model-info` for `active_search_v3`: geometry, contracts, exact parameters and the
/// budget semantics.
fn run_active_info(cfg: &ProbeConfig, json: Option<&Path>) -> anyhow::Result<()> {
    use recur64_model::active::ActiveSearchModel;
    use recur64_model::config::ACTIVE_MAX_BUDGET;
    let a = cfg.model.active.clone().expect("validated active geometry");
    let device = Default::default();
    let model = ActiveSearchModel::<Flex>::new(cfg.model.clone(), &device);
    let scientific_budgets = [0usize, 2, 4, 8, 16];
    let training_budgets = [0usize, 2, 4, 8];
    println!("config          : {}", cfg.name);
    println!("architecture    : {}", cfg.model.architecture.id());
    println!("device          : {:?}", cfg.device);
    println!("precision       : {}", cfg.precision.label());
    println!(
        "root board      : width={} heads={} ffn={} head_dim={} unique blocks={} (executed exactly once per decision)",
        cfg.model.width,
        cfg.model.heads,
        cfg.model.ffn,
        cfg.model.head_dim(),
        cfg.model.core_blocks
    );
    println!(
        "root candidates : dim={} heads={} ffn={} blocks={} facts_hidden={} policy_hidden={} facts_enabled={} (CandidateFactsV1 at the root only)",
        a.candidate.dim,
        a.candidate.heads,
        a.candidate.ffn,
        a.candidate.blocks,
        a.candidate.facts_hidden,
        a.candidate.policy_hidden,
        a.candidate.facts_enabled
    );
    println!(
        "query encoder   : width={} heads={} ffn={} blocks={} (shared by every queried node and step)",
        a.query_dim, a.query_heads, a.query_ffn, a.query_blocks
    );
    println!(
        "workspace       : K={} tokens x {}; branch memory: one {}-wide token per root candidate",
        a.workspace_tokens, a.query_dim, a.query_dim
    );
    println!(
        "planner         : heads={} ffn={} (one shared gated RMS-normalised update per exact state)",
        a.planner_heads, a.planner_ffn
    );
    println!(
        "selector        : hidden={} (STOP logit present and masked in the primary experiment)",
        a.selector_hidden
    );
    println!(
        "readout         : hidden={} (sparse legal-candidate softmax)",
        a.readout_hidden
    );
    println!(
        "rms ceiling     : {} (health guard on workspace / branch RMS)",
        a.rms_ceiling
    );
    println!("\ncontracts:");
    let c = &a.contracts;
    for (k, v) in [
        ("root_encoder", &c.root_encoder),
        ("root_candidate_tokens", &c.root_candidate_tokens),
        ("state_query", &c.state_query),
        ("query_state_encoder", &c.query_state_encoder),
        ("frontier", &c.frontier),
        ("search_memory", &c.search_memory),
        ("selector", &c.selector),
        ("planner", &c.planner),
        ("proof_trace", &c.proof_trace),
        ("budget_training", &c.budget_training),
        ("root_policy", &c.root_policy),
    ] {
        println!("  {k:<24} {v}");
    }
    println!("\nparameter breakdown (unique; independent of the query budget):");
    let mut total = 0usize;
    let mut groups = Vec::new();
    for (name, n) in model.param_breakdown() {
        println!("  {:<28} {:>12}", name, n);
        total += n;
        groups.push(serde_json::json!({ "group": name, "params": n }));
    }
    println!("  {:<28} {:>12}", "TOTAL UNIQUE", total);
    anyhow::ensure!(
        total == model.num_params(),
        "breakdown {total} does not sum to num_params {}",
        model.num_params()
    );
    let bytes = total * 4;
    println!(
        "parameter bytes : {} ({:.1} MiB, fp32)",
        bytes,
        bytes as f64 / (1024.0 * 1024.0)
    );
    println!(
        "STOP head       : {} parameters receive zero gradient while STOP is masked",
        model.stop_head_params()
    );
    println!(
        "\nsupported scientific query budgets : {scientific_budgets:?} (maximum {ACTIVE_MAX_BUDGET})"
    );
    println!(
        "primary training budgets           : {training_budgets:?} (budget_0_2_4_8_v1; B16 is the extrapolation diagnostic)"
    );
    println!(
        "recurrence                         : 1 (there is no recurrent re-reading of the board; the test-time-compute dimension is the exact state-query budget)"
    );
    if let Some(out) = json {
        let doc = serde_json::json!({
            "config": cfg.name,
            "architecture": cfg.model.architecture.id(),
            "model": cfg.model,
            "groups": groups,
            "total_params": total,
            "param_bytes_fp32": bytes,
            "stop_head_params": model.stop_head_params(),
            "scientific_budgets": scientific_budgets,
            "training_budgets": training_budgets,
            "recurrence": 1,
            "test_time_compute_dimension": "exact state-query budget",
        });
        if let Some(dir) = out.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(out, serde_json::to_vec_pretty(&doc)?)?;
        println!("wrote {}", out.display());
    }
    Ok(())
}

/// `model-info` for `all_info_v1`: exact subsystem accounting and the parameter match to
/// `active_search_v3` required by the P6 plan (within 0.5%).
fn run_all_info_info(cfg: &ProbeConfig, json: Option<&Path>) -> anyhow::Result<()> {
    use recur64_model::all_info::AllInfoModel;
    use recur64_model::config::ACTIVE_MAX_DEPTH;
    /// `active_search_v3` parameter count (V3-D9).
    const ACTIVE_PARAMS: usize = 30_853_790;
    let a = cfg
        .model
        .all_info
        .clone()
        .expect("validated all_info geometry");
    let device = Default::default();
    let model = AllInfoModel::<Flex>::new(cfg.model.clone(), &device);
    println!("config          : {}", cfg.name);
    println!("architecture    : {}", cfg.model.architecture.id());
    println!("device          : {:?}", cfg.device);
    println!("precision       : {}", cfg.precision.label());
    println!(
        "root board      : width={} heads={} ffn={} head_dim={} unique blocks={} (executed exactly once per decision)",
        cfg.model.width,
        cfg.model.heads,
        cfg.model.ffn,
        cfg.model.head_dim(),
        cfg.model.core_blocks
    );
    println!(
        "root candidates : dim={} heads={} ffn={} blocks={} facts_hidden={} facts_enabled={} (CandidateFactsV1 at the root only)",
        a.candidate.dim,
        a.candidate.heads,
        a.candidate.ffn,
        a.candidate.blocks,
        a.candidate.facts_hidden,
        a.candidate.facts_enabled
    );
    println!(
        "query encoder   : width={} heads={} ffn={} blocks={} (ONE shared instance encodes every supplied future state)",
        a.query_dim, a.query_heads, a.query_ffn, a.query_blocks
    );
    println!(
        "set integrator  : heads={} ffn={} (per-branch set block, root-token attention pool, cross-branch set block; no positional encoding)",
        a.set_heads, a.set_ffn
    );
    println!(
        "readout         : hidden={} (sparse legal-candidate softmax; the same function as active_search_v3)",
        a.readout_hidden
    );
    println!(
        "input           : the exhaustive raw depth-2 tree (all root successors, all opponent replies); no pruning, no truncation; maximum depth {ACTIVE_MAX_DEPTH} is not used"
    );
    println!("\ncontracts:");
    let c = &a.contracts;
    for (k, v) in [
        ("root_encoder", &c.root_encoder),
        ("root_candidate_tokens", &c.root_candidate_tokens),
        ("state_query", &c.state_query),
        ("query_state_encoder", &c.query_state_encoder),
        ("input", &c.input),
        ("integrator", &c.integrator),
        ("root_policy", &c.root_policy),
    ] {
        println!("  {k:<24} {v}");
    }
    println!("\nparameter breakdown (unique):");
    let mut total = 0usize;
    let mut groups = Vec::new();
    for (name, n) in model.param_breakdown() {
        println!("  {:<28} {:>12}", name, n);
        total += n;
        groups.push(serde_json::json!({ "group": name, "params": n }));
    }
    println!("  {:<28} {:>12}", "TOTAL UNIQUE", total);
    anyhow::ensure!(
        total == model.num_params(),
        "breakdown {total} does not sum to num_params {}",
        model.num_params()
    );
    let diff = total as i64 - ACTIVE_PARAMS as i64;
    let rel = diff.unsigned_abs() as f64 / ACTIVE_PARAMS as f64;
    println!(
        "vs active_search_v3 : {ACTIVE_PARAMS} -> difference {diff} ({:.4}%); within 0.5%: {}",
        rel * 100.0,
        rel <= 0.005
    );
    let bytes = total * 4;
    println!(
        "parameter bytes : {} ({:.1} MiB, fp32)",
        bytes,
        bytes as f64 / (1024.0 * 1024.0)
    );
    if let Some(out) = json {
        let doc = serde_json::json!({
            "config": cfg.name,
            "architecture": cfg.model.architecture.id(),
            "model": cfg.model,
            "groups": groups,
            "total_params": total,
            "param_bytes_fp32": bytes,
            "active_search_v3_params": ACTIVE_PARAMS,
            "difference_vs_active": diff,
            "relative_difference_vs_active": rel,
            "within_half_percent": rel <= 0.005,
            "recurrence": 1,
        });
        if let Some(dir) = out.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(out, serde_json::to_vec_pretty(&doc)?)?;
        println!("wrote {}", out.display());
    }
    Ok(())
}

/// `model-info` for `candidate_v25`: exact subsystem accounting.
fn run_candidate_info(cfg: &ProbeConfig, json: Option<&Path>) -> anyhow::Result<()> {
    let device = Default::default();
    let model = CandidateV25Model::<Flex>::new(cfg.model.clone(), &device);
    let cand = cfg
        .model
        .candidate
        .clone()
        .expect("validated candidate geometry");
    println!("config          : {}", cfg.name);
    println!("architecture    : {}", cfg.model.architecture.id());
    println!("device          : {:?}", cfg.device);
    println!("precision       : {}", cfg.precision.label());
    println!(
        "board geometry  : width={} heads={} ffn={} head_dim={} unique blocks={} (one pass, no recurrence)",
        cfg.model.width,
        cfg.model.heads,
        cfg.model.ffn,
        cfg.model.head_dim(),
        cfg.model.core_blocks
    );
    println!(
        "candidate       : dim={} heads={} ffn={} blocks={} facts_hidden={} policy_hidden={} facts_enabled={}",
        cand.dim,
        cand.heads,
        cand.ffn,
        cand.blocks,
        cand.facts_hidden,
        cand.policy_hidden,
        cand.facts_enabled
    );
    println!("\nparameter breakdown:");
    let mut total = 0usize;
    let mut groups = Vec::new();
    for (name, n) in model.param_breakdown() {
        println!("  {:<22} {:>12}", name, n);
        total += n;
        groups.push(serde_json::json!({ "group": name, "params": n }));
    }
    println!("  {:<22} {:>12}", "TOTAL", total);
    assert_eq!(
        total,
        model.num_params(),
        "breakdown must sum to num_params"
    );
    let bytes = total * 4;
    println!(
        "parameter bytes : {} ({:.1} MiB, fp32)",
        bytes,
        bytes as f64 / (1024.0 * 1024.0)
    );
    let f10 = f10_params();
    println!(
        "\nfor context     : historical F10 = {f10} parameters ({:.2}x smaller; descriptive, not a matched control)",
        total as f64 / f10 as f64
    );
    if let Some(out) = json {
        let doc = serde_json::json!({
            "config": cfg.name,
            "architecture": cfg.model.architecture.id(),
            "model": cfg.model,
            "groups": groups,
            "total_params": total,
            "param_bytes_fp32": bytes,
            "f10_params": f10,
        });
        if let Some(dir) = out.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(out, serde_json::to_vec_pretty(&doc)?)?;
        println!("wrote {}", out.display());
    }
    Ok(())
}

/// `model-info` for `legacy_facts_v25`: the legacy head plus the facts-delta MLP.
fn run_legacy_facts_info(cfg: &ProbeConfig, json: Option<&Path>) -> anyhow::Result<()> {
    use recur64_model::legacy_facts::LegacyFactsModel;
    let device = Default::default();
    let model = LegacyFactsModel::<Flex>::new(cfg.model.clone(), &device);
    println!("config          : {}", cfg.name);
    println!("architecture    : {}", cfg.model.architecture.id());
    println!(
        "board geometry  : width={} heads={} ffn={} head_dim={} unique blocks={} (one pass)",
        cfg.model.width,
        cfg.model.heads,
        cfg.model.ffn,
        cfg.model.head_dim(),
        cfg.model.core_blocks
    );
    println!("\nparameter breakdown:");
    let mut total = 0usize;
    let mut groups = Vec::new();
    for (name, n) in model.param_breakdown() {
        println!("  {:<22} {:>12}", name, n);
        total += n;
        groups.push(serde_json::json!({ "group": name, "params": n }));
    }
    println!("  {:<22} {:>12}", "TOTAL", total);
    assert_eq!(
        total,
        model.num_params(),
        "breakdown must sum to num_params"
    );
    if let Some(out) = json {
        let doc = serde_json::json!({
            "config": cfg.name, "architecture": cfg.model.architecture.id(),
            "model": cfg.model, "groups": groups, "total_params": total,
        });
        if let Some(dir) = out.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(out, serde_json::to_vec_pretty(&doc)?)?;
    }
    Ok(())
}
