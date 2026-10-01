//! Qualification verdict for `recur64 v3-qual` (P3.1).
//!
//! The verdict is a pure function of the report JSON, so it can be tested with
//! synthetic reports and applied unchanged to historical evidence.
//!
//! * **Gates** are engineering conditions whose failure means the graph is not
//!   qualified. Every gate is evaluated independently and reported.
//! * **Diagnostics** (utilization, first-seen shape timing, throughput, repeated
//!   model build/drop allocator slope) are reported but never gate: they are not
//!   correctness conditions. A diagnostic that did not run at all is recorded as
//!   `diagnostics_complete = false`.
//!
//! Report schema versions: a report without `report_schema` is the original P3
//! report. It did not record FIXED-output finiteness, so that single sub-condition
//! is recorded as a limitation (structural success of the FIXED run is still
//! required). Reports with `report_schema == "v3_qual_report_v2"` must record it.

use serde::Serialize;
use serde_json::Value;

pub const REPORT_SCHEMA_V2: &str = "v3_qual_report_v2";

#[derive(Debug, Clone, Serialize)]
pub struct Gate {
    pub name: &'static str,
    pub ok: bool,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct Finding {
    pub name: &'static str,
    pub gating: bool,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct Verdict {
    pub qualification_gates_ok: bool,
    pub qualification_gate_details: Vec<Gate>,
    pub diagnostics_complete: bool,
    pub diagnostic_findings: Vec<Finding>,
    /// Sub-conditions the evaluated report did not record (legacy reports only).
    pub recorded_limitations: Vec<String>,
}

fn gate(name: &'static str, ok: bool, detail: impl Into<String>) -> Gate {
    Gate {
        name,
        ok,
        detail: detail.into(),
    }
}

fn is_true(v: &Value) -> bool {
    v.as_bool() == Some(true)
}

fn finite_number(v: &Value) -> bool {
    v.as_f64().is_some_and(f64::is_finite)
}

fn rows(report: &Value) -> &[Value] {
    report["inference"]["rows"]
        .as_array()
        .map_or(&[], Vec::as_slice)
}

fn training(report: &Value) -> &[Value] {
    report["training"].as_array().map_or(&[], Vec::as_slice)
}

/// Evaluate a `v3-qual` report.
pub fn evaluate(report: &Value) -> Verdict {
    let v2 = report["report_schema"] == REPORT_SCHEMA_V2;
    let mut limitations = Vec::new();
    let mut gates = Vec::new();

    // Device.
    let guard = report["device_known_answer_guard"].as_str().unwrap_or("");
    gates.push(gate(
        "device_known_answer_guard",
        guard == "passed",
        if guard.is_empty() {
            "absent".to_string()
        } else {
            guard.to_string()
        },
    ));

    // Scientific budget range: an engineering-stress run does not qualify the
    // V3.0 science envelope.
    gates.push(gate(
        "scientific_budget_range",
        report["engineering_only"] == false,
        format!("engineering_only = {}", report["engineering_only"]),
    ));

    // ACTIVE inference: every cell.
    let inf = rows(report);
    let bad: Vec<String> = inf
        .iter()
        .filter(|r| !(is_true(&r["ok"]) && is_true(&r["finite"]) && r["invariants"] == "pass"))
        .map(|r| {
            format!(
                "batch {} B{}: ok={} finite={} invariants={}",
                r["batch"], r["budget"], r["ok"], r["finite"], r["invariants"]
            )
        })
        .collect();
    gates.push(gate(
        "active_inference",
        !inf.is_empty() && bad.is_empty(),
        if inf.is_empty() {
            format!("no inference rows {}", report["inference"]["section_error"])
        } else if bad.is_empty() {
            format!("{} cells ok", inf.len())
        } else {
            bad.join("; ")
        },
    ));

    // FIXED inference: the same cells.
    let mut fixed_bad = Vec::new();
    let mut fixed_finite_recorded = true;
    for r in inf {
        let f = &r["fixed_selection"];
        let mut ok = is_true(&f["ok"]);
        if v2 {
            ok &= is_true(&f["finite"]);
        } else if f.get("finite").is_none() {
            fixed_finite_recorded = false;
        } else {
            ok &= is_true(&f["finite"]);
        }
        if !ok {
            fixed_bad.push(format!(
                "batch {} B{}: fixed_selection={}",
                r["batch"], r["budget"], f
            ));
        }
    }
    if !fixed_finite_recorded {
        limitations.push(
            "FIXED-selection output finiteness was not recorded by the original report; only \
             structural success (the run completed with its accounting invariants) is evidenced"
                .to_string(),
        );
    }
    gates.push(gate(
        "fixed_inference",
        !inf.is_empty() && fixed_bad.is_empty(),
        if inf.is_empty() {
            "no inference rows".to_string()
        } else if fixed_bad.is_empty() {
            format!("{} cells ok", inf.len())
        } else {
            fixed_bad.join("; ")
        },
    ));

    // Training path.
    let tr = training(report);
    let train_bad: Vec<String> = tr
        .iter()
        .filter(|t| {
            let losses_ok = t["losses"]
                .as_array()
                .is_some_and(|l| !l.is_empty() && l.iter().all(finite_number));
            !(is_true(&t["ok"]) && is_true(&t["finite"]) && losses_ok)
        })
        .map(|t| {
            format!(
                "B{}: ok={} finite={} losses={}",
                t["budget"], t["ok"], t["finite"], t["losses"]
            )
        })
        .collect();
    gates.push(gate(
        "training_forward_loss_backward_update",
        !tr.is_empty() && train_bad.is_empty(),
        if tr.is_empty() {
            "no training rows".to_string()
        } else if train_bad.is_empty() {
            format!("{} budgets ok", tr.len())
        } else {
            train_bad.join("; ")
        },
    ));

    let cov_bad: Vec<String> = tr
        .iter()
        .filter(|t| {
            let c = &t["gradient_coverage_update_one"];
            let tensors = c["parameter_tensors"].as_u64().unwrap_or(0);
            let missing_empty = c["without_finite_nonzero_gradient_excluding_stop"]
                .as_array()
                .is_some_and(Vec::is_empty);
            !(tensors > 0 && missing_empty)
        })
        .map(|t| format!("B{}: {}", t["budget"], t["gradient_coverage_update_one"]))
        .collect();
    gates.push(gate(
        "gradient_coverage_non_stop",
        !tr.is_empty() && cov_bad.is_empty(),
        if tr.is_empty() {
            "no training rows".to_string()
        } else if cov_bad.is_empty() {
            "every non-STOP parameter tensor has a finite non-zero gradient".to_string()
        } else {
            cov_bad.join("; ")
        },
    ));

    let nonfinite_bad: Vec<String> = tr
        .iter()
        .filter(|t| t["gradient_coverage_update_one"]["any_nonfinite_gradient"] != false)
        .map(|t| format!("B{}", t["budget"]))
        .collect();
    gates.push(gate(
        "no_nonfinite_gradient",
        !tr.is_empty() && nonfinite_bad.is_empty(),
        if tr.is_empty() {
            "no training rows".to_string()
        } else if nonfinite_bad.is_empty() {
            "no non-finite gradient".to_string()
        } else {
            format!("non-finite or unrecorded at {}", nonfinite_bad.join(", "))
        },
    ));

    let stop_bad: Vec<String> = tr
        .iter()
        .filter(|t| t["gradient_coverage_update_one"]["stop_head_gradient_nonzero"] != false)
        .map(|t| format!("B{}", t["budget"]))
        .collect();
    gates.push(gate(
        "stop_head_gradient_exactly_zero",
        !tr.is_empty() && stop_bad.is_empty(),
        if tr.is_empty() {
            "no training rows".to_string()
        } else if stop_bad.is_empty() {
            "STOP head gradient exactly zero while masked".to_string()
        } else {
            format!(
                "non-zero or unrecorded STOP gradient at {}",
                stop_bad.join(", ")
            )
        },
    ));

    // Checkpoint.
    let c = &report["checkpoint"];
    gates.push(gate(
        "checkpoint_round_trip",
        is_true(&c["ok"])
            && c["loaded_architecture"] == "active_search_v3"
            && is_true(&c["loaded_model_id_present"]),
        format!(
            "ok={} architecture={} max_abs_diff={}",
            c["ok"], c["loaded_architecture"], c["max_abs_policy_diff_saved_vs_loaded"]
        ),
    ));

    // Resident-model VRAM (what a science run uses).
    let ri = &report["resident_model_vram"]["inference_b8_batch16"]["vram"];
    gates.push(gate(
        "resident_inference_vram_plateau",
        is_true(&ri["plateau"]),
        format!(
            "growth_from_second_sample_mb = {}",
            ri["growth_from_second_sample_mb"]
        ),
    ));
    let rt = &report["resident_model_vram"]["training_b4_batch8"]["vram"];
    gates.push(gate(
        "resident_training_vram_plateau",
        is_true(&rt["plateau"]),
        format!(
            "growth_from_second_sample_mb = {}",
            rt["growth_from_second_sample_mb"]
        ),
    ));

    let qualification_gates_ok = gates.iter().all(|g| g.ok);

    // Diagnostics: reported, never gating.
    let mut findings = Vec::new();
    let present = |v: &Value| !v.is_null() && v.get("section_error").is_none();
    let diagnostics_complete = present(&report["dynamic_query_widths"])
        && present(&report["sustained_load"])
        && present(&report["lifecycle"]);

    if let Some(modes) = report["lifecycle"]["modes"].as_object() {
        let growing: Vec<String> = modes
            .iter()
            .filter(|(_, m)| m["plateau"] == false)
            .map(|(k, m)| {
                let slope = m
                    .get("slope_mb_per_cycle_from_second_reading")
                    .or_else(|| m.get("slope_mb_per_cycle"));
                format!(
                    "{k}: {} MiB/cycle",
                    slope.map_or("?".to_string(), |s| s.to_string())
                )
            })
            .collect();
        findings.push(Finding {
            name: "repeated_model_build_drop_vram_slope",
            gating: false,
            detail: if growing.is_empty() {
                "all build/drop modes plateau".to_string()
            } else {
                format!(
                    "{} (reproduced on the V2.5 model; absent for a resident model; \
                     non-gating)",
                    growing.join("; ")
                )
            },
        });
    }
    let dw = &report["dynamic_query_widths"];
    if present(dw) {
        findings.push(Finding {
            name: "first_seen_dynamic_shape_timing",
            gating: false,
            detail: format!(
                "first/second pass mean {}, worst first-pass over second-pass median {} \
                 (one-off JIT or autotune cost; non-gating)",
                dw["first_over_second_mean"], dw["first_max_over_second_p50"]
            ),
        });
    }
    let su = &report["sustained_load"];
    if present(su) {
        findings.push(Finding {
            name: "gpu_utilization_and_throughput",
            gating: false,
            detail: format!(
                "positions/s {}, utilization busy mean {} max {} (non-gating)",
                su["positions_per_second"], su["gpu"]["util_busy_mean"], su["gpu"]["util_max"]
            ),
        });
    }

    Verdict {
        qualification_gates_ok,
        qualification_gate_details: gates,
        diagnostics_complete,
        diagnostic_findings: findings,
        recorded_limitations: limitations,
    }
}

impl Verdict {
    /// Names of the failed gates.
    pub fn failed(&self) -> Vec<&'static str> {
        self.qualification_gate_details
            .iter()
            .filter(|g| !g.ok)
            .map(|g| g.name)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A synthetic, fully passing report (test-only fixture).
    fn good() -> Value {
        let row = |batch: u32, budget: u32| {
            json!({
                "batch": batch, "budget": budget, "ok": true, "finite": true,
                "invariants": "pass",
                "fixed_selection": {"ok": true, "finite": true},
            })
        };
        let cov = json!({
            "parameter_tensors": 234,
            "without_finite_nonzero_gradient_excluding_stop": [],
            "stop_head_gradient_nonzero": false,
            "any_nonfinite_gradient": false,
        });
        let train = |b: u32| {
            json!({"budget": b, "ok": true, "finite": true, "losses": [8.4, 8.3, 8.2],
                   "gradient_coverage_update_one": cov})
        };
        json!({
            "report_schema": REPORT_SCHEMA_V2,
            "device_known_answer_guard": "passed",
            "engineering_only": false,
            "inference": {"rows": [row(1, 0), row(1, 8), row(16, 16)]},
            "training": [train(2), train(4), train(8)],
            "checkpoint": {"ok": true, "loaded_architecture": "active_search_v3",
                           "loaded_model_id_present": true,
                           "max_abs_policy_diff_saved_vs_loaded": 0.0},
            "resident_model_vram": {
                "inference_b8_batch16": {"vram": {"plateau": true, "growth_from_second_sample_mb": 0}},
                "training_b4_batch8": {"vram": {"plateau": true, "growth_from_second_sample_mb": 0}},
            },
            "dynamic_query_widths": {"first_over_second_mean": 1.2, "first_max_over_second_p50": 5.6},
            "sustained_load": {"positions_per_second": 224.0,
                               "gpu": {"util_busy_mean": 39.0, "util_max": 47}},
            "lifecycle": {"modes": {"build_then_drop": {"plateau": false,
                          "slope_mb_per_cycle_from_second_reading": 12.8}}},
        })
    }

    fn flips(mutate: impl FnOnce(&mut Value), gate_name: &str) {
        let mut r = good();
        mutate(&mut r);
        let v = evaluate(&r);
        assert!(
            !v.qualification_gates_ok,
            "{gate_name}: must fail qualification"
        );
        assert!(
            v.failed().contains(&gate_name),
            "{gate_name} must be among the failed gates, got {:?}",
            v.failed()
        );
        // Only the intended gate fails: each condition is independent.
        assert_eq!(v.failed(), vec![gate_name], "unrelated gates flipped");
    }

    #[test]
    fn the_baseline_passes_every_gate() {
        let v = evaluate(&good());
        assert!(v.qualification_gates_ok, "{:?}", v.failed());
        assert_eq!(v.qualification_gate_details.len(), 11);
        assert!(v.diagnostics_complete);
        assert!(v.recorded_limitations.is_empty());
    }

    #[test]
    fn an_active_row_failure_fails_qualification() {
        flips(
            |r| r["inference"]["rows"][1]["ok"] = json!(false),
            "active_inference",
        );
        flips(
            |r| r["inference"]["rows"][1]["finite"] = json!(false),
            "active_inference",
        );
    }

    #[test]
    fn a_fixed_row_failure_fails_qualification() {
        flips(
            |r| r["inference"]["rows"][2]["fixed_selection"]["ok"] = json!(false),
            "fixed_inference",
        );
        flips(
            |r| r["inference"]["rows"][0]["fixed_selection"]["finite"] = json!(false),
            "fixed_inference",
        );
    }

    #[test]
    fn an_accounting_invariant_failure_fails_qualification() {
        flips(
            |r| r["inference"]["rows"][0]["invariants"] = json!("root encoder ran 2 times"),
            "active_inference",
        );
    }

    #[test]
    fn a_non_finite_training_loss_fails_qualification() {
        // serde_json writes NaN as null.
        flips(
            |r| r["training"][1]["losses"] = json!([8.4, null, 8.2]),
            "training_forward_loss_backward_update",
        );
        flips(
            |r| r["training"][0]["finite"] = json!(false),
            "training_forward_loss_backward_update",
        );
    }

    #[test]
    fn a_missing_non_stop_gradient_fails_qualification() {
        flips(
            |r| {
                r["training"][2]["gradient_coverage_update_one"]["without_finite_nonzero_gradient_excluding_stop"] =
                    json!(["planner.gate.weight"])
            },
            "gradient_coverage_non_stop",
        );
        flips(
            |r| r["training"][0]["gradient_coverage_update_one"]["parameter_tensors"] = json!(0),
            "gradient_coverage_non_stop",
        );
    }

    #[test]
    fn a_non_finite_gradient_fails_qualification() {
        flips(
            |r| {
                r["training"][1]["gradient_coverage_update_one"]["any_nonfinite_gradient"] =
                    json!(true)
            },
            "no_nonfinite_gradient",
        );
    }

    #[test]
    fn a_non_zero_stop_gradient_fails_qualification() {
        flips(
            |r| {
                r["training"][1]["gradient_coverage_update_one"]["stop_head_gradient_nonzero"] =
                    json!(true)
            },
            "stop_head_gradient_exactly_zero",
        );
    }

    #[test]
    fn a_checkpoint_failure_fails_qualification() {
        flips(
            |r| r["checkpoint"]["ok"] = json!(false),
            "checkpoint_round_trip",
        );
        flips(
            |r| r["checkpoint"]["loaded_architecture"] = json!("candidate_v25"),
            "checkpoint_round_trip",
        );
    }

    #[test]
    fn resident_vram_failures_fail_qualification() {
        flips(
            |r| r["resident_model_vram"]["inference_b8_batch16"]["vram"]["plateau"] = json!(false),
            "resident_inference_vram_plateau",
        );
        flips(
            |r| r["resident_model_vram"]["training_b4_batch8"]["vram"]["plateau"] = json!(false),
            "resident_training_vram_plateau",
        );
    }

    #[test]
    fn a_failed_device_guard_or_engineering_run_fails_qualification() {
        flips(
            |r| r["device_known_answer_guard"] = json!("failed: nvrtc"),
            "device_known_answer_guard",
        );
        flips(
            |r| r["engineering_only"] = json!(true),
            "scientific_budget_range",
        );
    }

    #[test]
    fn non_gating_diagnostics_never_fail_qualification() {
        // Repeated build/drop slope: plateau=false is already in the baseline;
        // make every mode grow and add more modes.
        let mut r = good();
        r["lifecycle"]["modes"]["build_b8_then_drop"] =
            json!({"plateau": false, "slope_mb_per_cycle_from_second_reading": 16.0});
        // A huge dynamic-shape outlier.
        r["dynamic_query_widths"]["first_max_over_second_p50"] = json!(500.0);
        r["dynamic_query_widths"]["first_over_second_mean"] = json!(40.0);
        // Very low utilization and throughput.
        r["sustained_load"]["gpu"]["util_busy_mean"] = json!(1.0);
        r["sustained_load"]["gpu"]["util_max"] = json!(2);
        r["sustained_load"]["positions_per_second"] = json!(0.5);
        let v = evaluate(&r);
        assert!(v.qualification_gates_ok, "{:?}", v.failed());
        assert!(v.diagnostics_complete);
        assert!(v.diagnostic_findings.iter().all(|f| !f.gating));
        assert!(
            v.diagnostic_findings
                .iter()
                .any(|f| f.name == "repeated_model_build_drop_vram_slope"
                    && f.detail.contains("non-gating"))
        );
    }

    #[test]
    fn a_missing_diagnostic_marks_incompleteness_without_failing_gates() {
        let mut r = good();
        r["sustained_load"] = json!({"section_error": "sustained: boom"});
        let v = evaluate(&r);
        assert!(v.qualification_gates_ok);
        assert!(!v.diagnostics_complete);
    }

    #[test]
    fn a_legacy_report_records_the_unrecorded_fixed_finiteness_limitation() {
        let mut r = good();
        r.as_object_mut().unwrap().remove("report_schema");
        for row in r["inference"]["rows"].as_array_mut().unwrap() {
            row["fixed_selection"]
                .as_object_mut()
                .unwrap()
                .remove("finite");
        }
        let v = evaluate(&r);
        assert!(v.qualification_gates_ok, "{:?}", v.failed());
        assert_eq!(v.recorded_limitations.len(), 1);
        // The same omission in a v2 report is a failure.
        let mut r2 = good();
        r2["inference"]["rows"][0]["fixed_selection"]
            .as_object_mut()
            .unwrap()
            .remove("finite");
        assert!(!evaluate(&r2).qualification_gates_ok);
    }

    #[test]
    fn an_empty_or_garbage_report_fails_closed() {
        let v = evaluate(&json!({}));
        assert!(!v.qualification_gates_ok);
        assert!(v.failed().len() >= 8, "{:?}", v.failed());
    }
}
