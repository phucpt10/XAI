"""Run the approved, CPU-only official statistical analysis."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from plantxai_stability import __version__
from plantxai_stability.artifacts import atomic_json
from plantxai_stability.config import load_protocol
from plantxai_stability.official_analysis import (
    RQ1_PRIMARY,
    RQ2_PRIMARY,
    load_analysis_decision,
    load_analysis_support_decision,
    load_and_validate_merges,
    paired_comparisons,
    prediction_summaries,
    records_are_finite,
    rq3_associations,
    validate_analysis_plan,
    validate_analysis_support_authorization,
    xai_summaries,
)
from plantxai_stability.provenance import sha256_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--analysis-decision-record", type=Path, required=True)
    parser.add_argument("--analysis-support-decision-record", type=Path)
    parser.add_argument("--analysis-support-audit-dir", type=Path, required=True)
    parser.add_argument("--resnet50-merge-dir", type=Path, required=True)
    parser.add_argument("--efficientnet-b0-merge-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit("Analysis output exists; use a new immutable directory")

    resolved = load_protocol(args.protocol)
    decision = load_analysis_decision(args.analysis_decision_record)
    validate_analysis_plan(decision, resolved)
    if args.analysis_support_decision_record is not None:
        support_decision = load_analysis_support_decision(
            args.analysis_support_decision_record
        )
        non_estimable_authorization = validate_analysis_support_authorization(
            support_decision=support_decision,
            support_decision_path=args.analysis_support_decision_record,
            support_audit_dir=args.analysis_support_audit_dir,
            analysis_decision_path=args.analysis_decision_record,
        )
        support_gate_mode = "adjudicated_non_estimable_exception"
        support_decision_id: str | None = support_decision["decision_id"]
        support_decision_sha256: str | None = sha256_file(
            args.analysis_support_decision_record
        )
        support_audit_sha256 = support_decision["source_support_audit"]["report_sha256"]
        holm_non_estimable_policy: dict[str, Any] = support_decision[
            "holm_family_slot_policy"
        ]
    else:
        support_audit = _validate_direct_support_audit(
            support_audit_dir=args.analysis_support_audit_dir,
            analysis_decision_path=args.analysis_decision_record,
            decision=decision,
        )
        non_estimable_authorization = None
        support_gate_mode = "all_contrasts_estimable"
        support_decision_id = None
        support_decision_sha256 = None
        support_audit_sha256 = sha256_file(
            args.analysis_support_audit_dir / "analysis_support_audit_report.json"
        )
        holm_non_estimable_policy = {
            "mode": "all_contrasts_estimable",
            "reserved_non_estimable_slots": 0,
            "support_audit_planned_contrast_count": support_audit[
                "planned_contrast_count"
            ],
        }
    predictions, joint, reports = load_and_validate_merges(
        merge_dirs={
            "resnet50": args.resnet50_merge_dir,
            "efficientnet_b0": args.efficientnet_b0_merge_dir,
        },
        decision=decision,
        resolved=resolved,
    )
    policy = decision["statistical_policy"]
    iterations = int(policy["bootstrap_iterations"])
    confidence_level = float(policy["confidence_level"])
    seed = int(policy["base_seed"])

    prediction_summary, prediction_class_summary = prediction_summaries(
        predictions,
        iterations=iterations,
        confidence_level=confidence_level,
        seed=seed,
    )
    xai_summary, exclusion_audit = xai_summaries(
        joint,
        iterations=iterations,
        confidence_level=confidence_level,
        seed=seed,
    )
    comparisons = paired_comparisons(
        predictions,
        joint,
        methods=resolved.values["xai"]["methods"],
        alpha=float(policy["alpha"]),
        minimum_leaf_count=int(policy["minimum_paired_leaf_count"]),
        non_estimable_authorization=non_estimable_authorization,
    )
    associations = rq3_associations(
        predictions,
        joint,
        iterations=iterations,
        confidence_level=confidence_level,
        seed=seed,
    )

    expected_rows = decision["outputs"]["expected_fixed_row_counts"]
    estimable_comparisons = [row for row in comparisons if row["estimable"] is True]
    non_estimable_comparisons = [row for row in comparisons if row["estimable"] is False]
    criteria = {
        "analysis_decision_approved": True,
        "analysis_support_gate_passed": True,
        "analysis_support_audit_hashes_match": True,
        "source_merge_report_hashes_match": True,
        "source_child_artifact_hashes_match": True,
        "cross_model_lineage_matches": True,
        "joint_prediction_identity_matches": True,
        "prediction_factorial_coverage_exact": len(predictions) == 2 * 1693 * 12,
        "joint_factorial_coverage_exact": len(joint) == 2 * 1693 * 12 * 3,
        "prediction_summary_coverage_exact": (
            len(prediction_summary) == expected_rows["prediction_summary.csv"]
            and len(prediction_class_summary) == expected_rows["prediction_class_summary.csv"]
        ),
        "xai_summary_coverage_exact": (len(xai_summary) == expected_rows["xai_summary.csv"]),
        "paired_family_coverage_exact": (
            len(comparisons) == expected_rows["paired_comparisons.csv"]
        ),
        "paired_estimable_count_exact": (
            len(estimable_comparisons)
            == (576 if non_estimable_authorization is None else 573)
        ),
        "paired_non_estimable_count_exact": (
            len(non_estimable_comparisons)
            == (0 if non_estimable_authorization is None else 3)
        ),
        "non_estimable_rows_match_approved_exception": (
            not non_estimable_comparisons
            if non_estimable_authorization is None
            else all(
                row["contrast_type"]
                == non_estimable_authorization["analysis_contrast_type"]
                and row["contrast"]
                == non_estimable_authorization["analysis_contrast"]
                and row["endpoint"] in non_estimable_authorization["endpoints"]
                and row["support_status"]
                == non_estimable_authorization["support_status"]
                and row["n_common_sample_keys"]
                == non_estimable_authorization["expected_common_samples"]
                and row["n_leaf_pairs"]
                == non_estimable_authorization["expected_common_leaves"]
                and row["p_value_raw"] == ""
                and row["p_value_holm"] == ""
                and row["rank_biserial"] == ""
                for row in non_estimable_comparisons
            )
        ),
        "holm_predeclared_family_slots_preserved": all(
            int(row["holm_estimable_count"])
            + int(row["holm_reserved_non_estimable_slots"])
            == int(row["holm_family_size"])
            for row in comparisons
        ),
        "non_estimable_holm_slots_are_conservative": (
            not non_estimable_comparisons
            if non_estimable_authorization is None
            else all(
                row["holm_family_size"] == 12
                and row["holm_estimable_count"] == 11
                and row["holm_reserved_non_estimable_slots"] == 1
                and row["holm_reserved_slot_value_for_adjustment_only"] == 1.0
                and row["reject_h0_holm"] == ""
                for row in non_estimable_comparisons
            )
        ),
        "rq3_coverage_exact": (len(associations) == expected_rows["rq3_association_summary.csv"]),
        "exclusion_audit_reconciles": (
            sum(int(row["row_count"]) for row in exclusion_audit) == len(joint)
        ),
        "official_test_pixels_not_accessed": True,
        "bootstrap_leaf_unit_enforced": True,
        "paired_common_key_and_leaf_unit_enforced": True,
        "holm_families_predeclared": True,
        "summary_metrics_finite": records_are_finite(
            prediction_summary, ("estimate", "lower", "upper")
        )
        and records_are_finite(xai_summary, ("estimate", "lower", "upper"))
        and records_are_finite(associations, ("estimate", "lower", "upper")),
        "comparison_metrics_finite": records_are_finite(
            estimable_comparisons,
            (
                "left_mean",
                "right_mean",
                "mean_difference_left_minus_right",
                "wilcoxon_statistic",
                "p_value_raw",
                "p_value_holm",
                "rank_biserial",
            ),
        ),
        "no_reselection_or_tuning_performed": True,
    }
    if not all(criteria.values()):
        failed = sorted(key for key, value in criteria.items() if not value)
        raise SystemExit(f"Official statistical analysis gate failed: {failed}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    artifacts = {
        "prediction_summary.csv": prediction_summary,
        "prediction_class_summary.csv": prediction_class_summary,
        "xai_summary.csv": xai_summary,
        "xai_exclusion_audit.csv": exclusion_audit,
        "paired_comparisons.csv": comparisons,
        "rq3_association_summary.csv": associations,
    }
    paths: dict[str, Path] = {}
    for name, rows in artifacts.items():
        path = args.output_dir / name
        _write_csv_atomic(path, rows)
        paths[name] = path

    report_path = args.output_dir / "official_analysis_report.json"
    report = {
        "run_type": "authorized_official_test_statistical_analysis",
        "official_test_result": True,
        "official_test_pixels_accessed": False,
        "analysis_decision_id": decision["decision_id"],
        "analysis_decision_record_sha256": sha256_file(args.analysis_decision_record),
        "analysis_support_decision_id": support_decision_id,
        "analysis_support_decision_record_sha256": support_decision_sha256,
        "analysis_support_audit_report_sha256": support_audit_sha256,
        "analysis_support_gate_mode": support_gate_mode,
        "campaign_id": decision["scientific_lineage"]["campaign_id"],
        "governance_protocol_hash": resolved.sha256,
        "source_merge_report_sha256": {
            model: decision["source_merges"][model]["report_sha256"]
            for model in resolved.values["models"]
        },
        "source_merge_run_id": {
            model: reports[model]["run_id"] for model in resolved.values["models"]
        },
        "statistical_policy": policy,
        "analysis_scope": decision["analysis_scope"],
        "endpoint_policy": decision["endpoints"],
        "holm_family_policy": decision["holm_families"],
        "holm_non_estimable_family_slot_policy": holm_non_estimable_policy,
        "row_counts": {
            "source_prediction_rows": int(len(predictions)),
            "source_joint_rows": int(len(joint)),
            "prediction_summary_rows": len(prediction_summary),
            "prediction_class_summary_rows": len(prediction_class_summary),
            "xai_summary_rows": len(xai_summary),
            "xai_exclusion_audit_rows": len(exclusion_audit),
            "paired_comparison_rows": len(comparisons),
            "paired_estimable_rows": len(estimable_comparisons),
            "paired_non_estimable_rows": len(non_estimable_comparisons),
            "rq3_association_rows": len(associations),
        },
        "primary_endpoints": {
            "rq1": list(RQ1_PRIMARY),
            "rq2": list(RQ2_PRIMARY),
        },
        "artifact_sha256": {name: sha256_file(path) for name, path in paths.items()},
        "acceptance_criteria": criteria,
        "scientific_interpretation_constraints": {
            "severity_is_ordinal_within_transformation_only": True,
            "rotation_prediction_claim_is_zero_fill_operator_specific": True,
            "xai_uses_forward_alignment_and_valid_region": True,
            "rq3_is_exploratory_without_hypothesis_testing": True,
            "non_estimable_rows_are_not_interpreted_as_null_results": True,
            "minimum_paired_leaf_threshold_remains_20": True,
            "test_results_must_not_drive_reselection_or_tuning": True,
        },
        "runtime": {
            "python_platform": platform.platform(),
            "software_version": __version__,
            "git_commit": _git_revision(),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "device": "cpu",
        },
    }
    atomic_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Official statistical analysis: PASS\nReport: {report_path}")
    print(f"Report SHA-256: {sha256_file(report_path)}")
    return 0


def _validate_direct_support_audit(
    *,
    support_audit_dir: Path,
    analysis_decision_path: Path,
    decision: dict[str, Any],
) -> dict[str, Any]:
    """Accept a complete metadata-only support audit without an exception DR."""
    report_path = support_audit_dir / "analysis_support_audit_report.json"
    if not report_path.is_file():
        raise ValueError(f"Analysis support audit report is missing: {report_path}")
    with report_path.open(encoding="utf-8") as handle:
        report = json.load(handle)

    expected_merge_hashes = {
        model: values["report_sha256"]
        for model, values in decision["source_merges"].items()
    }
    expected = {
        "run_type": "metadata_only_official_analysis_support_audit",
        "analysis_decision_id": decision["decision_id"],
        "analysis_decision_record_sha256": sha256_file(analysis_decision_path),
        "source_merge_report_sha256": expected_merge_hashes,
        "minimum_common_leaf_count": 20,
        "planned_contrast_count": 192,
        "insufficient_contrast_count": 0,
        "all_contrasts_meet_minimum": True,
        "analysis_execution_allowed_without_adjudication": True,
        "official_test_pixels_accessed": False,
        "endpoint_metric_values_read": False,
        "hypothesis_tests_computed": False,
    }
    mismatches = [
        key for key, value in expected.items() if report.get(key) != value
    ]
    criteria = report.get("acceptance_criteria")
    if not isinstance(criteria, dict) or not all(criteria.values()):
        mismatches.append("acceptance_criteria")

    artifact_hashes = report.get("artifact_sha256")
    if not isinstance(artifact_hashes, dict) or not artifact_hashes:
        mismatches.append("artifact_sha256")
    else:
        for name, expected_hash in artifact_hashes.items():
            artifact_path = support_audit_dir / name
            if not artifact_path.is_file() or sha256_file(artifact_path) != expected_hash:
                mismatches.append(f"artifact_sha256.{name}")
    if mismatches:
        raise ValueError(
            "Direct analysis support audit mismatch: "
            f"{sorted(set(mismatches))}"
        )
    return report


def _write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _git_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
