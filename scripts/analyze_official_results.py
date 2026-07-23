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
    load_and_validate_merges,
    paired_comparisons,
    prediction_summaries,
    records_are_finite,
    rq3_associations,
    validate_analysis_plan,
    xai_summaries,
)
from plantxai_stability.provenance import sha256_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--analysis-decision-record", type=Path, required=True)
    parser.add_argument("--resnet50-merge-dir", type=Path, required=True)
    parser.add_argument("--efficientnet-b0-merge-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit("Analysis output exists; use a new immutable directory")

    resolved = load_protocol(args.protocol)
    decision = load_analysis_decision(args.analysis_decision_record)
    validate_analysis_plan(decision, resolved)
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
    )
    associations = rq3_associations(
        predictions,
        joint,
        iterations=iterations,
        confidence_level=confidence_level,
        seed=seed,
    )

    expected_rows = decision["outputs"]["expected_fixed_row_counts"]
    criteria = {
        "analysis_decision_approved": True,
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
        "rq3_coverage_exact": (len(associations) == expected_rows["rq3_association_summary.csv"]),
        "exclusion_audit_reconciles": (
            sum(int(row["row_count"]) for row in exclusion_audit) == len(joint)
        ),
        "official_test_pixels_accessed": False,
        "bootstrap_leaf_unit_enforced": True,
        "paired_common_key_and_leaf_unit_enforced": True,
        "holm_families_predeclared": True,
        "summary_metrics_finite": records_are_finite(
            prediction_summary, ("estimate", "lower", "upper")
        )
        and records_are_finite(xai_summary, ("estimate", "lower", "upper"))
        and records_are_finite(associations, ("estimate", "lower", "upper")),
        "comparison_metrics_finite": records_are_finite(
            comparisons,
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
        "row_counts": {
            "source_prediction_rows": int(len(predictions)),
            "source_joint_rows": int(len(joint)),
            "prediction_summary_rows": len(prediction_summary),
            "prediction_class_summary_rows": len(prediction_class_summary),
            "xai_summary_rows": len(xai_summary),
            "xai_exclusion_audit_rows": len(exclusion_audit),
            "paired_comparison_rows": len(comparisons),
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
