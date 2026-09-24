//! `recur64 model-info` — exact parameter accounting and executed-block counts.

use std::path::Path;

use burn::backend::Flex;
use recur64_model::config::ProbeConfig;
use recur64_model::model::ProbeModel;

/// Executed transformer blocks for the feed-forward R=1 control. Both the
/// feed-forward family and its matched recurrent family execute 8 blocks at
/// R=1, so this is the neural-compute baseline for the multiplier column.
const CONTROL_R1_BLOCKS: f64 = 8.0;

pub fn run_model_info(path: &Path) -> anyhow::Result<()> {
    let text = std::fs::read_to_string(path)?;
    let cfg = ProbeConfig::from_toml_str(&text)?;
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
