//! v69-aggregate: independent metric/gate aggregation from serialized predictions.
//!   v69-aggregate --artifacts DIR --run gen-001 --seed-file seed/master_seed.hex --arm A
//! Role: MetricAggregator (fit/val metadata + prediction files only).

use anyhow::{Context, Result};
use recur64_v69::access::{Access, Role};
use recur64_v69::custody::Custody;
use recur64_v69::dataset::Example;
use recur64_v69::metrics::{PredRow, aggregate};
use recur64_v69::streams::MasterSeed;
use std::path::Path;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn jsonl<T: serde::de::DeserializeOwned>(text: &str) -> Result<Vec<T>> {
    text.lines().map(|l| Ok(serde_json::from_str(l)?)).collect()
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let custody = Custody::new(Path::new(&arg(&args, "--artifacts").context("--artifacts")?))?;
    let run = arg(&args, "--run").context("--run")?;
    let seed_rel = arg(&args, "--seed-file").context("--seed-file")?;
    let arm = arg(&args, "--arm").context("--arm")?;
    let pred_dir = arg(&args, "--pred-dir").unwrap_or_else(|| format!("eval/{arm}"));
    let out = arg(&args, "--out").unwrap_or_else(|| format!("report/{arm}.json"));
    let access = Access::new(&custody, Role::MetricAggregator, Path::new(&run), Path::new(&seed_rel))?;
    let seed = MasterSeed::from_hex(String::from_utf8(access.read(Path::new(&seed_rel))?)?.trim())?;
    let run = Path::new(&run);
    let fit_meta: Vec<Example> = jsonl(&access.read_to_string(&run.join("meta/fit.meta.jsonl"))?)?;
    let val_meta: Vec<Example> = jsonl(&access.read_to_string(&run.join("meta/val.meta.jsonl"))?)?;
    let fit_preds: Vec<PredRow> = jsonl(&access.read_to_string(&Path::new(&pred_dir).join("predictions_fit.jsonl"))?)?;
    let val_preds: Vec<PredRow> = jsonl(&access.read_to_string(&Path::new(&pred_dir).join("predictions_val.jsonl"))?)?;
    let report = aggregate(&arm, &fit_preds, &val_preds, &fit_meta, &val_meta, &seed)?;
    let text = serde_json::to_string_pretty(&report)?;
    access.write(Path::new(&out), text.as_bytes())?;
    println!("{text}");
    Ok(())
}
