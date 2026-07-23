"""Audit common sample/leaf support without reading endpoint metric columns."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from plantxai_stability import __version__
from plantxai_stability.analysis_support import (
    EXPECTED_SUPPORT_COUNTS,
    EXPECTED_SUPPORT_TOTAL,
    JOINT_SUPPORT_COLUMNS,
    PREDICTION_SUPPORT_COLUMNS,
    build_support_audit,
    exclusion_reason_audit,
    load_support_metadata,
)
from plantxai_stability.artifacts import atomic_json
from plantxai_stability.config import load_protocol
from plantxai_stability.official_analysis import (
    load_analysis_decision,
    validate_analysis_plan,
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
        raise SystemExit("Support-audit output exists; use a new immutable directory")

    resolved = load_protocol(args.protocol)
    decision = load_analysis_decision(args.analysis_decision_record)
    validate_analysis_plan(decision, resolved)
    predictions, joint, reports = load_support_metadata(
        merge_dirs={
            "resnet50": args.resnet50_merge_dir,
            "efficientnet_b0": args.efficientnet_b0_merge_dir,
        },
        decision=decision,
        resolved=resolved,
    )
    minimum = int(decision["statistical_policy"]["minimum_paired_leaf_count"])
    support_rows = build_support_audit(
        predictions,
        joint,
        methods=resolved.values["xai"]["methods"],
        minimum_leaf_count=minimum,
    )
    insufficient = [row for row in support_rows if not row["passes_minimum_common_leaves"]]
    exclusions = exclusion_reason_audit(joint)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    support_path = args.output_dir / "planned_contrast_support_audit.csv"
    insufficient_path = args.output_dir / "insufficient_common_leaf_contrasts.csv"
    exclusion_path = args.output_dir / "exclusion_reason_audit.csv"
    _write_csv_atomic(support_path, support_rows)
    _write_csv_atomic(
        insufficient_path,
        insufficient,
        fieldnames=list(support_rows[0]),
    )
    _write_csv_atomic(exclusion_path, exclusions)

    counts_by_scope = {
        scope: sum(row["contrast_scope"] == scope for row in support_rows)
        for scope in EXPECTED_SUPPORT_COUNTS
    }
    insufficient_by_scope = {
        scope: sum(row["contrast_scope"] == scope for row in insufficient)
        for scope in EXPECTED_SUPPORT_COUNTS
    }
    criteria = {
        "analysis_decision_and_merge_hashes_match": True,
        "child_artifact_hashes_match": True,
        "metadata_factorial_coverage_exact": True,
        "cross_model_and_joint_leaf_identity_match": True,
        "only_predeclared_support_columns_read": True,
        "planned_support_contrast_coverage_exact": (
            len(support_rows) == EXPECTED_SUPPORT_TOTAL
            and counts_by_scope == EXPECTED_SUPPORT_COUNTS
        ),
        "exclusion_audit_reconciles": (
            sum(int(row["row_count"]) for row in exclusions) == len(joint)
        ),
        "official_test_pixels_accessed": False,
        "endpoint_metric_values_read": False,
        "hypothesis_tests_computed": False,
    }
    if not all(criteria.values()):
        failed = sorted(key for key, value in criteria.items() if not value)
        raise SystemExit(f"Analysis support audit failed: {failed}")

    report_path = args.output_dir / "analysis_support_audit_report.json"
    report = {
        "run_type": "metadata_only_official_analysis_support_audit",
        "analysis_decision_id": decision["decision_id"],
        "analysis_decision_record_sha256": sha256_file(args.analysis_decision_record),
        "source_merge_report_sha256": {
            model_id: decision["source_merges"][model_id]["report_sha256"]
            for model_id in resolved.values["models"]
        },
        "source_merge_run_id": {
            model_id: reports[model_id]["run_id"] for model_id in resolved.values["models"]
        },
        "minimum_common_leaf_count": minimum,
        "planned_contrast_count": len(support_rows),
        "contrast_counts_by_scope": counts_by_scope,
        "insufficient_contrast_count": len(insufficient),
        "insufficient_counts_by_scope": insufficient_by_scope,
        "all_contrasts_meet_minimum": not insufficient,
        "analysis_execution_allowed_without_adjudication": not insufficient,
        "columns_read": {
            "prediction_results.csv": list(PREDICTION_SUPPORT_COLUMNS),
            "joint_results.csv": list(JOINT_SUPPORT_COLUMNS),
        },
        "official_test_pixels_accessed": False,
        "endpoint_metric_values_read": False,
        "hypothesis_tests_computed": False,
        "artifact_sha256": {
            support_path.name: sha256_file(support_path),
            insufficient_path.name: sha256_file(insufficient_path),
            exclusion_path.name: sha256_file(exclusion_path),
        },
        "acceptance_criteria": criteria,
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
    print(f"Analysis support audit: PASS\nReport: {report_path}")
    print(f"Report SHA-256: {sha256_file(report_path)}")
    print(
        "Statistical support gate: "
        + ("PASS" if not insufficient else "BLOCKED_PENDING_ADJUDICATION")
    )
    return 0


def _write_csv_atomic(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> None:
    resolved_fields = list(fieldnames or (list(rows[0]) if rows else []))
    if not resolved_fields:
        raise ValueError(f"CSV schema is empty: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=resolved_fields)
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
