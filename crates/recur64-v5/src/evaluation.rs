//! Per-position V5 reader metrics and the pre-registered pilot classifier.

use std::collections::{BTreeMap, HashMap};

use serde::{Deserialize, Serialize};

use crate::graph::Schedule;
use crate::splitmix64;

pub const SHUFFLE_CONTRACT: &str = "v5_payload_shuffle_widening_v1";
pub const EVAL_SCHEMA: &str = "v5_reader_evaluation_v3";
pub const REPORT_SCHEMA: &str = "v5_reader_pilot_report_v3";
pub const EVAL_SEED: u64 = 0x7A50_E001;
pub const SHUFFLE_SEED: u64 = 0x7A50_E002;
pub const COMPOSITION_SEED: u64 = 0x7A50_E003;
pub const BOOT_LOOP: u64 = 0x7A50_0101;
pub const BOOT_B0: u64 = 0x7A50_0102;
pub const BOOT_SHUFFLE_LOSS: u64 = 0x7A50_0103;
pub const BOOT_SHUFFLE_INTERACTION: u64 = 0x7A50_0104;
pub const BOOTSTRAPS: usize = 20_000;
pub const BOOT_LO: usize = 499;
pub const BOOT_HI: usize = 19_499;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TreatmentLabel {
    Baseline,
    Normal,
    PayloadShuffle,
    NoRelationBias,
    NoHypothesisFeedback,
    AllPayloadNull,
    CompositionNeither,
    CompositionA,
    CompositionB,
    CompositionBoth,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicyMetrics {
    pub top1: f64,
    pub correct_mass: f64,
    pub set_loss: f64,
    pub uniform_correct_target_ce: f64,
    pub entropy: f64,
    pub raw_delta_l2: f64,
    pub centered_delta_l2: f64,
    pub candidate_relative_delta_range: f64,
    pub kl_vs_b0: f64,
    pub action_index: usize,
    pub b0_action_index: usize,
    pub action_changed: bool,
}

fn logsumexp(values: &[f64]) -> f64 {
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    max + values
        .iter()
        .map(|value| (value - max).exp())
        .sum::<f64>()
        .ln()
}

fn argmax(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
        .map(|x| x.0)
        .expect("validated nonempty logits")
}

pub fn policy_metrics(
    logits: &[f32],
    b0: &[f32],
    raw_delta: &[f32],
    centered_delta: &[f32],
    correct: &[usize],
) -> anyhow::Result<PolicyMetrics> {
    anyhow::ensure!(
        !logits.is_empty()
            && logits.len() == b0.len()
            && logits.len() == raw_delta.len()
            && logits.len() == centered_delta.len(),
        "metric vectors are empty or misaligned"
    );
    anyhow::ensure!(
        logits.iter().chain(b0).all(|x| x.is_finite()),
        "non-finite policy logits"
    );
    anyhow::ensure!(
        !correct.is_empty() && correct.iter().all(|&i| i < logits.len()),
        "correct set is empty or outside legal logits"
    );
    let z: Vec<f64> = logits.iter().map(|&x| f64::from(x)).collect();
    let z0: Vec<f64> = b0.iter().map(|&x| f64::from(x)).collect();
    let correct_z: Vec<f64> = correct.iter().map(|&i| z[i]).collect();
    let all_lse = logsumexp(&z);
    let correct_lse = logsumexp(&correct_z);
    let p: Vec<f64> = z.iter().map(|x| (x - all_lse).exp()).collect();
    let b0_lse = logsumexp(&z0);
    let p0: Vec<f64> = z0.iter().map(|x| (x - b0_lse).exp()).collect();
    let action = argmax(&z);
    let b0_action = argmax(&z0);
    let raw_delta_l2 = raw_delta
        .iter()
        .map(|&x| f64::from(x).powi(2))
        .sum::<f64>()
        .sqrt();
    let centered_delta_l2 = centered_delta
        .iter()
        .map(|&x| f64::from(x).powi(2))
        .sum::<f64>()
        .sqrt();
    let delta_min = centered_delta.iter().copied().fold(f32::INFINITY, f32::min);
    let delta_max = centered_delta
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    Ok(PolicyMetrics {
        top1: f64::from(u8::from(correct.contains(&action))),
        correct_mass: (correct_lse - all_lse).exp(),
        set_loss: all_lse - correct_lse,
        uniform_correct_target_ce: correct.iter().map(|&i| all_lse - z[i]).sum::<f64>()
            / correct.len() as f64,
        entropy: p
            .iter()
            .zip(&z)
            .map(|(&probability, &logit)| probability * (all_lse - logit))
            .sum(),
        raw_delta_l2,
        centered_delta_l2,
        candidate_relative_delta_range: f64::from(delta_max - delta_min),
        kl_vs_b0: p
            .iter()
            .zip(&p0)
            .map(|(&a, &b)| if a == 0.0 { 0.0 } else { a * (a / b).ln() })
            .sum(),
        action_index: action,
        b0_action_index: b0_action,
        action_changed: action != b0_action,
    })
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoopHealth {
    pub iteration: usize,
    pub stream: String,
    pub evidence_state_rms: f64,
    pub evidence_update_rms: f64,
    pub hypothesis_state_rms: f64,
    pub hypothesis_update_rms: f64,
    pub evidence_attention_entropy: f64,
    pub hypothesis_attention_entropy: f64,
    pub evidence_attention_max: f64,
    pub hypothesis_attention_max: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalRecord {
    pub position_id: String,
    pub family: String,
    pub mate_depth: u8,
    pub schedule: Schedule,
    pub q: usize,
    pub r: usize,
    pub treatment: TreatmentLabel,
    pub graph_digest: Option<String>,
    pub graph_structure_digest: Option<String>,
    pub actual_q: usize,
    pub exhausted_frontier: bool,
    pub maximum_depth: u8,
    pub branches_covered: usize,
    pub legal_generations: u64,
    pub legal_moves_generated: u64,
    pub root_candidates: usize,
    pub root_candidate_facts_evaluated: usize,
    pub root_candidate_facts_probe_wall_seconds: f64,
    /// Wall for the one Q8 acquisition whose nested prefix this record reads.
    pub q8_acquisition_wall_seconds: f64,
    pub reader_wall_seconds: f64,
    pub returned_encoder_examples: usize,
    pub physical_rows: usize,
    pub padded_rows: usize,
    pub shared_core_applications_per_example: usize,
    pub metrics: PolicyMetrics,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub loop_health: Vec<LoopHealth>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluationBundle {
    pub schema: String,
    pub shuffle_contract: String,
    pub architecture: String,
    pub source_sha: String,
    pub config_digest: String,
    pub train_digest: String,
    pub dev_digest: String,
    pub model_hash: String,
    pub graph_manifest_hash: String,
    pub scope: String,
    pub split: String,
    pub seed: u64,
    pub final_update: u64,
    pub microbatch: usize,
    pub precision: String,
    pub acquisition_seed: u64,
    pub intervention_seed: u64,
    pub composition_seed: u64,
    pub normal_replay_exact: bool,
    pub shuffle_mappings: Vec<ShuffleMapping>,
    pub composition_partitions: Vec<CompositionPartition>,
    pub records: Vec<EvalRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShuffleMapping {
    pub schedule: Schedule,
    pub recipient_position_id: String,
    pub recipient_path: Vec<u16>,
    pub recipient_depth: u8,
    pub recipient_root_to_move: bool,
    pub widening_tier: String,
    pub candidate_pool_size: usize,
    pub absolute_depth_delta: u8,
    pub donor_position_id: String,
    pub donor_path: Vec<u16>,
    pub donor_depth: u8,
    pub donor_root_to_move: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompositionPartition {
    pub schedule: Schedule,
    pub position_id: String,
    pub group_a_paths: Vec<Vec<u16>>,
    pub group_b_paths: Vec<Vec<u16>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SummaryMetrics {
    pub n: usize,
    pub top1: f64,
    pub correct_mass: f64,
    pub set_loss: f64,
    pub uniform_correct_target_ce: f64,
    pub entropy: f64,
    pub raw_delta_l2: f64,
    pub centered_delta_l2: f64,
    pub candidate_relative_delta_range: f64,
    pub kl_vs_b0: f64,
    pub action_change_rate: f64,
}

pub fn summarize<'a>(records: impl IntoIterator<Item = &'a EvalRecord>) -> SummaryMetrics {
    let records: Vec<&EvalRecord> = records.into_iter().collect();
    let n = records.len();
    let mean = |f: fn(&PolicyMetrics) -> f64| {
        records.iter().map(|record| f(&record.metrics)).sum::<f64>() / n.max(1) as f64
    };
    SummaryMetrics {
        n,
        top1: mean(|x| x.top1),
        correct_mass: mean(|x| x.correct_mass),
        set_loss: mean(|x| x.set_loss),
        uniform_correct_target_ce: mean(|x| x.uniform_correct_target_ce),
        entropy: mean(|x| x.entropy),
        raw_delta_l2: mean(|x| x.raw_delta_l2),
        centered_delta_l2: mean(|x| x.centered_delta_l2),
        candidate_relative_delta_range: mean(|x| x.candidate_relative_delta_range),
        kl_vs_b0: mean(|x| x.kl_vs_b0),
        action_change_rate: records
            .iter()
            .map(|record| f64::from(u8::from(record.metrics.action_changed)))
            .sum::<f64>()
            / n.max(1) as f64,
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BootstrapCi {
    pub mean: f64,
    pub lo: f64,
    pub hi: f64,
    pub resamples: usize,
    pub lo_rank: usize,
    pub hi_rank: usize,
}

pub fn paired_bootstrap(values: &[f64], seed: u64) -> anyhow::Result<BootstrapCi> {
    anyhow::ensure!(
        !values.is_empty() && values.iter().all(|x| x.is_finite()),
        "bootstrap values must be finite and nonempty"
    );
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let mut state = seed;
    let mut samples = Vec::with_capacity(BOOTSTRAPS);
    for _ in 0..BOOTSTRAPS {
        let mut total = 0.0;
        for _ in 0..values.len() {
            total += values[(splitmix64(&mut state) % values.len() as u64) as usize];
        }
        samples.push(total / values.len() as f64);
    }
    samples.sort_by(f64::total_cmp);
    Ok(BootstrapCi {
        mean,
        lo: samples[BOOT_LO],
        hi: samples[BOOT_HI],
        resamples: BOOTSTRAPS,
        lo_rank: BOOT_LO,
        hi_rank: BOOT_HI,
    })
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PilotContrast {
    pub name: String,
    pub estimate: BootstrapCi,
    pub threshold: f64,
    pub per_schedule: BTreeMap<String, f64>,
    pub pass: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransitionCounts {
    pub wrong_to_right: usize,
    pub right_to_wrong: usize,
    pub unchanged_right: usize,
    pub unchanged_wrong: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PilotClassification {
    pub schema: String,
    pub shuffle_contract: String,
    pub classification: String,
    pub loop_benefit: PilotContrast,
    pub b0_benefit: PilotContrast,
    pub payload_shuffle_loss: PilotContrast,
    pub payload_loop_interaction: PilotContrast,
    pub engineering_integrity_accounting_pass: bool,
    pub wrong_to_right_vs_right_to_wrong: BTreeMap<String, TransitionCounts>,
    pub exact_unrounded_comparisons: bool,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct Key {
    position: String,
    schedule: Schedule,
    q: usize,
    r: usize,
    treatment: TreatmentLabel,
}

fn find<'a>(map: &'a HashMap<Key, &'a EvalRecord>, key: &Key) -> anyhow::Result<&'a EvalRecord> {
    map.get(key).copied().ok_or_else(|| {
        anyhow::anyhow!(
            "missing {} {:?} Q{} R{} {:?}",
            key.position,
            key.schedule,
            key.q,
            key.r,
            key.treatment
        )
    })
}

pub fn validate_shuffle_mappings(mappings: &[ShuffleMapping]) -> anyhow::Result<()> {
    for m in mappings {
        anyhow::ensure!(
            m.recipient_position_id != m.donor_position_id
                && m.recipient_root_to_move == m.donor_root_to_move
                && m.candidate_pool_size > 0
                && m.absolute_depth_delta == m.recipient_depth.abs_diff(m.donor_depth)
                && m.widening_tier
                    == if m.absolute_depth_delta == 0 {
                        "exact_depth"
                    } else {
                        "nearest_same_turn_depth"
                    },
            "shuffle mapping audit mismatch"
        );
    }
    Ok(())
}

pub fn classify_pilot(
    bundle: &EvaluationBundle,
    engineering_integrity_accounting_pass: bool,
    require_full_dev: bool,
) -> anyhow::Result<PilotClassification> {
    anyhow::ensure!(
        bundle.schema == EVAL_SCHEMA && bundle.shuffle_contract == SHUFFLE_CONTRACT,
        "evaluation schema mismatch"
    );
    validate_shuffle_mappings(&bundle.shuffle_mappings)?;
    let mut map = HashMap::new();
    for record in &bundle.records {
        let key = Key {
            position: record.position_id.clone(),
            schedule: record.schedule,
            q: record.q,
            r: record.r,
            treatment: record.treatment,
        };
        anyhow::ensure!(
            map.insert(key, record).is_none(),
            "duplicate evaluation record"
        );
    }
    let mut positions: Vec<&str> = bundle
        .records
        .iter()
        .map(|x| x.position_id.as_str())
        .collect();
    positions.sort_unstable();
    positions.dedup();
    if require_full_dev {
        anyhow::ensure!(
            positions.len() == 4_500,
            "pilot report requires all 4,500 DEV positions"
        );
    }
    let schedules = [Schedule::UniformFrontierV1, Schedule::BaseRankedDepthV1];
    let baseline = |position: &str, schedule| {
        find(
            &map,
            &Key {
                position: position.to_owned(),
                schedule,
                q: 0,
                r: 0,
                treatment: TreatmentLabel::Baseline,
            },
        )
    };
    let normal = |position: &str, schedule, r| {
        find(
            &map,
            &Key {
                position: position.to_owned(),
                schedule,
                q: 8,
                r,
                treatment: TreatmentLabel::Normal,
            },
        )
    };
    let shuffled = |position: &str, schedule, r| {
        find(
            &map,
            &Key {
                position: position.to_owned(),
                schedule,
                q: 8,
                r,
                treatment: TreatmentLabel::PayloadShuffle,
            },
        )
    };
    let mut loop_values = Vec::with_capacity(positions.len());
    let mut b0_values = Vec::with_capacity(positions.len());
    let mut shuffle_loss_values = Vec::with_capacity(positions.len());
    let mut interaction_values = Vec::with_capacity(positions.len());
    let mut per_loop: HashMap<Schedule, Vec<f64>> = HashMap::new();
    let mut per_b0: HashMap<Schedule, Vec<f64>> = HashMap::new();
    let mut per_shuffle: HashMap<Schedule, Vec<f64>> = HashMap::new();
    let mut transitions: BTreeMap<String, TransitionCounts> = BTreeMap::new();
    for &position in &positions {
        let mut loop_sum = 0.0;
        let mut b0_sum = 0.0;
        let mut shuffle_sum = 0.0;
        let mut interaction_sum = 0.0;
        for schedule in schedules {
            let r1 = normal(position, schedule, 1)?;
            let r4 = normal(position, schedule, 4)?;
            let sr1 = shuffled(position, schedule, 1)?;
            let sr4 = shuffled(position, schedule, 4)?;
            let b0 = baseline(position, schedule)?;
            let loop_delta = r4.metrics.top1 - r1.metrics.top1;
            let b0_delta = r4.metrics.top1 - b0.metrics.top1;
            let shuffle_delta = sr4.metrics.set_loss - r4.metrics.set_loss;
            let interaction = loop_delta - (sr4.metrics.top1 - sr1.metrics.top1);
            loop_sum += loop_delta;
            b0_sum += b0_delta;
            shuffle_sum += shuffle_delta;
            interaction_sum += interaction;
            per_loop.entry(schedule).or_default().push(loop_delta);
            per_b0.entry(schedule).or_default().push(b0_delta);
            per_shuffle.entry(schedule).or_default().push(shuffle_delta);
            let counts = transitions
                .entry(schedule.id().into())
                .or_insert(TransitionCounts {
                    wrong_to_right: 0,
                    right_to_wrong: 0,
                    unchanged_right: 0,
                    unchanged_wrong: 0,
                });
            match (r1.metrics.top1 == 1.0, r4.metrics.top1 == 1.0) {
                (false, true) => counts.wrong_to_right += 1,
                (true, false) => counts.right_to_wrong += 1,
                (true, true) => counts.unchanged_right += 1,
                (false, false) => counts.unchanged_wrong += 1,
            }
        }
        loop_values.push(loop_sum / 2.0);
        b0_values.push(b0_sum / 2.0);
        shuffle_loss_values.push(shuffle_sum / 2.0);
        interaction_values.push(interaction_sum / 2.0);
    }
    let per_schedule = |source: &HashMap<Schedule, Vec<f64>>| {
        schedules
            .into_iter()
            .map(|schedule| {
                let values = &source[&schedule];
                (
                    schedule.id().to_owned(),
                    values.iter().sum::<f64>() / values.len() as f64,
                )
            })
            .collect::<BTreeMap<_, _>>()
    };
    let loop_ci = paired_bootstrap(&loop_values, BOOT_LOOP)?;
    let b0_ci = paired_bootstrap(&b0_values, BOOT_B0)?;
    let shuffle_ci = paired_bootstrap(&shuffle_loss_values, BOOT_SHUFFLE_LOSS)?;
    let interaction_ci = paired_bootstrap(&interaction_values, BOOT_SHUFFLE_INTERACTION)?;
    let loop_schedule = per_schedule(&per_loop);
    let b0_schedule = per_schedule(&per_b0);
    let shuffle_schedule = per_schedule(&per_shuffle);
    let gate1 = loop_ci.mean >= 0.03 && loop_ci.lo > 0.0;
    let gate2 = b0_ci.mean >= 0.03 && b0_ci.lo > 0.0;
    let gate3 = shuffle_ci.mean >= 0.01 && shuffle_ci.lo > 0.0;
    let gate4 = interaction_ci.lo > 0.0;
    let gate5 = loop_schedule.values().all(|&x| x >= 0.0)
        && b0_schedule.values().all(|&x| x >= 0.0)
        && shuffle_schedule.values().all(|&x| x >= 0.0);
    let classification =
        if gate1 && gate2 && gate3 && gate4 && gate5 && engineering_integrity_accounting_pass {
            "PILOT_CANDIDATE"
        } else if gate3 && !gate1 {
            "CONTENT_HELPS_LOOPS_DO_NOT"
        } else if gate1 && !gate3 {
            "LOOPS_HELP_PAYLOAD_SPECIFICITY_FAILS"
        } else if gate1 && !gate2 {
            "LOOPS_IMPROVE_BUT_REMAIN_BELOW_B0"
        } else if !engineering_integrity_accounting_pass {
            "ENGINEERING_FAILURE"
        } else {
            "NO_SIGNAL"
        };
    Ok(PilotClassification {
        schema: REPORT_SCHEMA.into(),
        shuffle_contract: SHUFFLE_CONTRACT.into(),
        classification: classification.into(),
        loop_benefit: PilotContrast {
            name: "top1_q8_r4_minus_q8_r1".into(),
            estimate: loop_ci,
            threshold: 0.03,
            per_schedule: loop_schedule,
            pass: gate1,
        },
        b0_benefit: PilotContrast {
            name: "top1_q8_r4_minus_b0".into(),
            estimate: b0_ci,
            threshold: 0.03,
            per_schedule: b0_schedule,
            pass: gate2,
        },
        payload_shuffle_loss: PilotContrast {
            name: "set_loss_shuffle_minus_real_q8_r4".into(),
            estimate: shuffle_ci,
            threshold: 0.01,
            per_schedule: shuffle_schedule,
            pass: gate3,
        },
        payload_loop_interaction: PilotContrast {
            name: "real_loop_benefit_minus_shuffled_loop_benefit".into(),
            estimate: interaction_ci,
            threshold: 0.0,
            per_schedule: BTreeMap::new(),
            pass: gate4,
        },
        engineering_integrity_accounting_pass,
        wrong_to_right_vs_right_to_wrong: transitions,
        exact_unrounded_comparisons: true,
    })
}

pub fn merge_cell_bundles(mut bundles: Vec<EvaluationBundle>) -> anyhow::Result<EvaluationBundle> {
    anyhow::ensure!(
        bundles.len() == 6,
        "merge requires exactly six family/depth cell bundles"
    );
    let first = bundles.first().expect("nonempty bundles").clone();
    let mut positions = std::collections::HashSet::new();
    let mut records = Vec::new();
    let mut shuffle_mappings = Vec::new();
    let mut composition_partitions = Vec::new();
    let mut shard_hash = sha2::Sha256::new();
    use sha2::Digest;
    for bundle in bundles.drain(..) {
        validate_shuffle_mappings(&bundle.shuffle_mappings)?;
        anyhow::ensure!(
            bundle.schema == EVAL_SCHEMA
                && bundle.shuffle_contract == SHUFFLE_CONTRACT
                && bundle.architecture == first.architecture
                && bundle.source_sha == first.source_sha
                && bundle.config_digest == first.config_digest
                && bundle.train_digest == first.train_digest
                && bundle.dev_digest == first.dev_digest
                && bundle.model_hash == first.model_hash
                && bundle.split == first.split
                && bundle.seed == first.seed
                && bundle.final_update == first.final_update
                && bundle.microbatch == first.microbatch
                && bundle.precision == first.precision
                && bundle.acquisition_seed == first.acquisition_seed
                && bundle.intervention_seed == first.intervention_seed
                && bundle.composition_seed == first.composition_seed,
            "cell evaluation identities differ"
        );
        anyhow::ensure!(
            bundle.normal_replay_exact,
            "a cell failed normal replay parity"
        );
        shard_hash.update(bundle.graph_manifest_hash.as_bytes());
        shard_hash.update(b"\n");
        let shard_positions: std::collections::HashSet<&str> = bundle
            .records
            .iter()
            .map(|x| x.position_id.as_str())
            .collect();
        for id in shard_positions {
            anyhow::ensure!(
                positions.insert(id.to_owned()),
                "DEV cell shards overlap at {id}"
            );
        }
        records.extend(bundle.records);
        shuffle_mappings.extend(bundle.shuffle_mappings);
        composition_partitions.extend(bundle.composition_partitions);
    }
    anyhow::ensure!(
        positions.len() == 4_500,
        "merged evaluation is not the 4,500-position DEV set"
    );
    let primary: std::collections::HashSet<&str> = records
        .iter()
        .filter(|x| x.family == "KQRvK" && x.mate_depth == 3)
        .map(|x| x.position_id.as_str())
        .collect();
    anyhow::ensure!(
        primary.len() == 750,
        "KQRvK M3 count is not the expected 750"
    );
    let mut cells: BTreeMap<(String, u8), std::collections::BTreeSet<String>> = BTreeMap::new();
    for row in &records {
        cells
            .entry((row.family.clone(), row.mate_depth))
            .or_default()
            .insert(row.position_id.clone());
    }
    let expected: std::collections::BTreeSet<_> = ["KQRvK", "KRRvK"]
        .into_iter()
        .flat_map(|f| (1..=3).map(move |d| (f.to_string(), d)))
        .collect();
    anyhow::ensure!(
        cells
            .keys()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>()
            == expected
            && cells.values().all(|s| s.len() == 750),
        "DEV merge requires all six exact 750-record cells"
    );
    Ok(EvaluationBundle {
        schema: EVAL_SCHEMA.into(),
        shuffle_contract: SHUFFLE_CONTRACT.into(),
        architecture: first.architecture,
        source_sha: first.source_sha,
        config_digest: first.config_digest,
        train_digest: first.train_digest,
        dev_digest: first.dev_digest,
        model_hash: first.model_hash,
        graph_manifest_hash: format!("{:x}", shard_hash.finalize()),
        scope: "all_dev_4500".into(),
        split: first.split,
        seed: first.seed,
        final_update: first.final_update,
        microbatch: first.microbatch,
        precision: first.precision,
        acquisition_seed: first.acquisition_seed,
        intervention_seed: first.intervention_seed,
        composition_seed: first.composition_seed,
        normal_replay_exact: true,
        shuffle_mappings,
        composition_partitions,
        records,
    })
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompositionInteractionSummary {
    pub schedule: Schedule,
    pub r: usize,
    pub n: usize,
    pub mean_set_loss_interaction: f64,
    pub mean_correct_mass_interaction: f64,
    pub action_changes_neither_to_a: usize,
    pub action_changes_neither_to_b: usize,
    pub action_changes_neither_to_both: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AblationReport {
    pub schema: String,
    pub evaluation_model_hash: String,
    pub graph_manifest_hash: String,
    pub scope: String,
    pub summaries: BTreeMap<String, SummaryMetrics>,
    pub composition: Vec<CompositionInteractionSummary>,
    pub normal_replay_exact: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct R8ScheduleResult {
    pub r4: SummaryMetrics,
    pub r8: SummaryMetrics,
    pub top1_improvement: f64,
    pub set_loss_improvement: f64,
    pub wrong_to_right: usize,
    pub right_to_wrong: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct R8Report {
    pub schema: String,
    pub model_hash: String,
    pub main_graph_manifest_hash: String,
    pub r8_graph_manifest_hash: String,
    pub by_schedule: BTreeMap<String, R8ScheduleResult>,
    pub mean_top1_improvement: f64,
    pub mean_set_loss_improvement: f64,
    pub positive_task_improvement: bool,
    pub corruption_observed: bool,
    pub trained_at_r8: bool,
}

pub fn r8_report(main: &EvaluationBundle, r8: &EvaluationBundle) -> anyhow::Result<R8Report> {
    anyhow::ensure!(
        main.schema == EVAL_SCHEMA
            && r8.schema == EVAL_SCHEMA
            && main.scope == "all_dev_4500"
            && r8.scope == "KQRvK_M3_R8_conditional"
            && main.final_update == 800
            && r8.final_update == 800
            && main.model_hash == r8.model_hash
            && main.source_sha == r8.source_sha
            && main.config_digest == r8.config_digest,
        "R4/R8 evaluation identity mismatch"
    );
    let mut by_schedule = BTreeMap::new();
    let mut top1_sum = 0.0;
    let mut loss_sum = 0.0;
    let mut corruption = false;
    for schedule in [Schedule::UniformFrontierV1, Schedule::BaseRankedDepthV1] {
        let r4: Vec<&EvalRecord> = main
            .records
            .iter()
            .filter(|record| {
                record.family == "KQRvK"
                    && record.mate_depth == 3
                    && record.schedule == schedule
                    && record.q == 8
                    && record.r == 4
                    && record.treatment == TreatmentLabel::Normal
            })
            .collect();
        let r8_records: Vec<&EvalRecord> = r8
            .records
            .iter()
            .filter(|record| record.schedule == schedule && record.q == 8 && record.r == 8)
            .collect();
        anyhow::ensure!(
            r4.len() == 750 && r8_records.len() == 750,
            "incomplete R4/R8 schedule"
        );
        let r4_map: HashMap<&str, &EvalRecord> = r4
            .iter()
            .map(|record| (record.position_id.as_str(), *record))
            .collect();
        let mut wrong_to_right = 0;
        let mut right_to_wrong = 0;
        for record in &r8_records {
            let prior = r4_map
                .get(record.position_id.as_str())
                .ok_or_else(|| anyhow::anyhow!("R8 position absent from R4"))?;
            anyhow::ensure!(
                record.graph_digest == prior.graph_digest,
                "R8 graph identity differs from R4"
            );
            match (prior.metrics.top1 == 1.0, record.metrics.top1 == 1.0) {
                (false, true) => wrong_to_right += 1,
                (true, false) => right_to_wrong += 1,
                _ => {}
            }
        }
        corruption |= right_to_wrong > 0;
        let r4_summary = summarize(r4);
        let r8_summary = summarize(r8_records);
        let top1 = r8_summary.top1 - r4_summary.top1;
        let loss = r4_summary.set_loss - r8_summary.set_loss;
        top1_sum += top1;
        loss_sum += loss;
        by_schedule.insert(
            schedule.id().into(),
            R8ScheduleResult {
                r4: r4_summary,
                r8: r8_summary,
                top1_improvement: top1,
                set_loss_improvement: loss,
                wrong_to_right,
                right_to_wrong,
            },
        );
    }
    Ok(R8Report {
        schema: "v5_r8_diagnostic_v1".into(),
        model_hash: main.model_hash.clone(),
        main_graph_manifest_hash: main.graph_manifest_hash.clone(),
        r8_graph_manifest_hash: r8.graph_manifest_hash.clone(),
        by_schedule,
        mean_top1_improvement: top1_sum / 2.0,
        mean_set_loss_improvement: loss_sum / 2.0,
        positive_task_improvement: top1_sum / 2.0 > 0.0,
        corruption_observed: corruption,
        trained_at_r8: false,
    })
}

pub fn ablation_report(bundle: &EvaluationBundle) -> anyhow::Result<AblationReport> {
    anyhow::ensure!(
        bundle.schema == EVAL_SCHEMA && bundle.shuffle_contract == SHUFFLE_CONTRACT,
        "evaluation schema mismatch"
    );
    anyhow::ensure!(bundle.normal_replay_exact, "normal replay parity failed");
    let mut groups: BTreeMap<String, Vec<&EvalRecord>> = BTreeMap::new();
    for record in &bundle.records {
        groups
            .entry(format!(
                "{}_M{}|{}|Q{}|R{}|{:?}",
                record.family,
                record.mate_depth,
                record.schedule.id(),
                record.q,
                record.r,
                record.treatment
            ))
            .or_default()
            .push(record);
    }
    let summaries = groups
        .into_iter()
        .map(|(key, records)| (key, summarize(records)))
        .collect();
    let composition_labels = [
        TreatmentLabel::CompositionNeither,
        TreatmentLabel::CompositionA,
        TreatmentLabel::CompositionB,
        TreatmentLabel::CompositionBoth,
    ];
    let mut composition = Vec::new();
    for schedule in [Schedule::UniformFrontierV1, Schedule::BaseRankedDepthV1] {
        for r in [1, 4] {
            let mut by_position: BTreeMap<String, BTreeMap<TreatmentLabel, &EvalRecord>> =
                BTreeMap::new();
            for record in bundle.records.iter().filter(|record| {
                record.family == "KQRvK"
                    && record.mate_depth == 3
                    && record.schedule == schedule
                    && record.q == 8
                    && record.r == r
                    && composition_labels.contains(&record.treatment)
            }) {
                by_position
                    .entry(record.position_id.clone())
                    .or_default()
                    .insert(record.treatment, record);
            }
            anyhow::ensure!(!by_position.is_empty(), "composition records are absent");
            let mut set_interaction = 0.0;
            let mut mass_interaction = 0.0;
            let mut changes = [0usize; 3];
            for records in by_position.values() {
                anyhow::ensure!(
                    composition_labels
                        .iter()
                        .all(|label| records.contains_key(label)),
                    "incomplete composition quartet"
                );
                let n = records[&TreatmentLabel::CompositionNeither];
                let a = records[&TreatmentLabel::CompositionA];
                let b = records[&TreatmentLabel::CompositionB];
                let both = records[&TreatmentLabel::CompositionBoth];
                set_interaction += both.metrics.set_loss - a.metrics.set_loss - b.metrics.set_loss
                    + n.metrics.set_loss;
                mass_interaction +=
                    both.metrics.correct_mass - a.metrics.correct_mass - b.metrics.correct_mass
                        + n.metrics.correct_mass;
                changes[0] += usize::from(n.metrics.action_index != a.metrics.action_index);
                changes[1] += usize::from(n.metrics.action_index != b.metrics.action_index);
                changes[2] += usize::from(n.metrics.action_index != both.metrics.action_index);
            }
            let count = by_position.len();
            composition.push(CompositionInteractionSummary {
                schedule,
                r,
                n: count,
                mean_set_loss_interaction: set_interaction / count as f64,
                mean_correct_mass_interaction: mass_interaction / count as f64,
                action_changes_neither_to_a: changes[0],
                action_changes_neither_to_b: changes[1],
                action_changes_neither_to_both: changes[2],
            });
        }
    }
    Ok(AblationReport {
        schema: "v5_ablation_report_v1".into(),
        evaluation_model_hash: bundle.model_hash.clone(),
        graph_manifest_hash: bundle.graph_manifest_hash.clone(),
        scope: bundle.scope.clone(),
        summaries,
        composition,
        normal_replay_exact: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_use_set_logsumexp_and_candidate_only_centering_inputs() {
        let metrics = policy_metrics(
            &[1.0, 0.0, -1.0],
            &[0.0, 1.0, 0.0],
            &[0.6, -0.3, -0.3],
            &[0.6, -0.3, -0.3],
            &[0, 2],
        )
        .unwrap();
        let expected = logsumexp(&[1.0, 0.0, -1.0]) - logsumexp(&[1.0, -1.0]);
        assert!((metrics.set_loss - expected).abs() < 1.0e-12);
        assert_eq!(metrics.top1, 1.0);
        assert!(metrics.action_changed);
        assert!((metrics.candidate_relative_delta_range - 0.9).abs() < 1.0e-6);
    }

    #[test]
    fn bootstrap_is_splitmix_deterministic_and_uses_frozen_ranks() {
        let values: Vec<f64> = (0..200).map(|i| f64::from(i % 7) - 2.0).collect();
        let a = paired_bootstrap(&values, BOOT_LOOP).unwrap();
        let b = paired_bootstrap(&values, BOOT_LOOP).unwrap();
        assert_eq!(a, b);
        assert_eq!((a.resamples, a.lo_rank, a.hi_rank), (20_000, 499, 19_499));
        assert!(a.lo < a.mean && a.mean < a.hi);
    }

    fn fixture_record(
        position: &str,
        schedule: Schedule,
        q: usize,
        r: usize,
        treatment: TreatmentLabel,
        top1: f64,
        set_loss: f64,
    ) -> EvalRecord {
        EvalRecord {
            position_id: position.into(),
            family: "KQRvK".into(),
            mate_depth: 3,
            schedule,
            q,
            r,
            treatment,
            graph_digest: (q > 0).then(|| "graph".into()),
            graph_structure_digest: (q > 0).then(|| "structure".into()),
            actual_q: q,
            exhausted_frontier: false,
            maximum_depth: usize::from(q > 0) as u8,
            branches_covered: usize::from(q > 0),
            legal_generations: q as u64,
            legal_moves_generated: q as u64,
            root_candidates: 2,
            root_candidate_facts_evaluated: 2,
            root_candidate_facts_probe_wall_seconds: 0.0,
            q8_acquisition_wall_seconds: 0.0,
            reader_wall_seconds: 0.0,
            returned_encoder_examples: q,
            physical_rows: 1,
            padded_rows: 0,
            shared_core_applications_per_example: 4 * r,
            metrics: PolicyMetrics {
                top1,
                correct_mass: top1,
                set_loss,
                uniform_correct_target_ce: set_loss,
                entropy: 0.0,
                raw_delta_l2: 0.0,
                centered_delta_l2: 0.0,
                candidate_relative_delta_range: 0.0,
                kl_vs_b0: 0.0,
                action_index: usize::from(top1 == 0.0),
                b0_action_index: 1,
                action_changed: top1 == 1.0,
            },
            loop_health: Vec::new(),
        }
    }

    #[test]
    fn pilot_classifier_averages_schedules_per_position_and_applies_all_gates() {
        let mut records = Vec::new();
        for position in ["a", "b"] {
            for schedule in [Schedule::UniformFrontierV1, Schedule::BaseRankedDepthV1] {
                records.push(fixture_record(
                    position,
                    schedule,
                    0,
                    0,
                    TreatmentLabel::Baseline,
                    0.0,
                    1.0,
                ));
                records.push(fixture_record(
                    position,
                    schedule,
                    8,
                    1,
                    TreatmentLabel::Normal,
                    0.0,
                    1.0,
                ));
                records.push(fixture_record(
                    position,
                    schedule,
                    8,
                    4,
                    TreatmentLabel::Normal,
                    1.0,
                    0.2,
                ));
                records.push(fixture_record(
                    position,
                    schedule,
                    8,
                    1,
                    TreatmentLabel::PayloadShuffle,
                    0.0,
                    1.0,
                ));
                records.push(fixture_record(
                    position,
                    schedule,
                    8,
                    4,
                    TreatmentLabel::PayloadShuffle,
                    0.0,
                    0.4,
                ));
            }
        }
        let bundle = EvaluationBundle {
            schema: EVAL_SCHEMA.into(),
            shuffle_contract: SHUFFLE_CONTRACT.into(),
            architecture: "counterfactual_relational_loop_v1".into(),
            source_sha: "source".into(),
            config_digest: "config".into(),
            train_digest: "train".into(),
            dev_digest: "dev".into(),
            model_hash: "model".into(),
            graph_manifest_hash: "graphs".into(),
            scope: "test".into(),
            split: "test".into(),
            seed: 5301,
            final_update: 800,
            microbatch: 2,
            precision: "fp32".into(),
            acquisition_seed: EVAL_SEED,
            intervention_seed: SHUFFLE_SEED,
            composition_seed: COMPOSITION_SEED,
            normal_replay_exact: true,
            shuffle_mappings: Vec::new(),
            composition_partitions: Vec::new(),
            records,
        };
        let result = classify_pilot(&bundle, true, false).unwrap();
        assert_eq!(result.classification, "PILOT_CANDIDATE");
        assert_eq!(result.loop_benefit.estimate.mean, 1.0);
        assert!((result.payload_shuffle_loss.estimate.mean - 0.2).abs() < 1.0e-12);
        assert_eq!(result.payload_loop_interaction.estimate.mean, 1.0);
    }
}
