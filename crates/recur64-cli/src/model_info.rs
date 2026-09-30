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
    if cfg.model.architecture == Architecture::CandidateV25 {
        return run_candidate_info(&cfg, json);
    }
    if cfg.model.architecture == Architecture::LegacyFactsV25 {
        return run_legacy_facts_info(&cfg, json);
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
