"""Metadata-only checks used before any official-test pixel access."""

from __future__ import annotations

import math
from typing import Any


def validate_g1_audit_evidence(
    report: dict[str, Any],
    checkpoint_decision: dict[str, Any],
    *,
    model_id: str,
) -> dict[str, Any]:
    """Validate one G1 audit against the approved checkpoint registry."""
    lineage = checkpoint_decision.get("training_lineage", {})
    approved = checkpoint_decision.get("approved_checkpoints", {}).get(model_id)
    mismatches: list[str] = []
    if not isinstance(approved, dict):
        raise ValueError(f"Checkpoint Decision Record does not approve {model_id}")
    expected = {
        "run_type": "official_validation_checkpoint_audit",
        "official_checkpoint_selection_evidence": True,
        "official_test_result": False,
        "source_split": "validation",
        "test_split_accessed": False,
        "model_id": model_id,
        "protocol_hash": lineage.get("protocol_hash"),
        "manifest_sha256": lineage.get("manifest_sha256"),
        "freeze_record_sha256": lineage.get("freeze_record_sha256"),
        "checkpoint_sha256": approved.get("checkpoint_sha256"),
        "checkpoint_evidence_sha256": approved.get("checkpoint_evidence_sha256"),
        "sample_ids_sha256": lineage.get("validation_sample_ids_sha256"),
    }
    mismatches.extend(
        key for key, expected_value in expected.items() if report.get(key) != expected_value
    )
    criteria = report.get("acceptance_criteria", {})
    if not criteria or not all(value is True for value in criteria.values()):
        mismatches.append("acceptance_criteria")
    if report.get("artifact_sha256") != approved.get("validation_artifact_sha256"):
        mismatches.append("validation_artifact_sha256")
    metrics = report.get("metrics", {})
    if int(metrics.get("sample_count", -1)) != int(
        lineage.get("validation_sample_count", -2)
    ):
        mismatches.append("validation_sample_count")
    for report_key, decision_key in (
        ("macro_f1", "validation_macro_f1"),
        ("accuracy", "validation_accuracy"),
    ):
        observed = metrics.get(report_key)
        expected_value = approved.get(decision_key)
        if not isinstance(observed, (int, float)) or not isinstance(
            expected_value, (int, float)
        ) or not math.isclose(
            float(observed), float(expected_value), rel_tol=0.0, abs_tol=1e-12
        ):
            mismatches.append(report_key)
    if int(metrics.get("error_count", -1)) != int(
        approved.get("validation_error_count", -2)
    ):
        mismatches.append("validation_error_count")
    if mismatches:
        raise ValueError(f"G1 validation audit mismatch: {sorted(set(mismatches))}")
    return {
        "model_id": model_id,
        "checkpoint_sha256": approved["checkpoint_sha256"],
        "validation_macro_f1": float(metrics["macro_f1"]),
        "validation_accuracy": float(metrics["accuracy"]),
        "validation_error_count": int(metrics["error_count"]),
        "validation_sample_count": int(metrics["sample_count"]),
    }
