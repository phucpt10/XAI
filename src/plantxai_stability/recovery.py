"""Fail-closed physical-freeze recovery and historical-lineage bridging."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from plantxai_stability.provenance import sha256_file


RECOVERY_SCHEMA_VERSION = "physical_freeze_recovery_v1"
REQUIRED_RECOVERY_POLICY = (
    "infrastructure_only",
    "physical_freeze_hash_must_be_distinct_from_historical_hash",
    "logical_historical_freeze_lineage_preserved",
    "verify_all_manifest_image_hashes_before_reauthorization",
    "no_dataset_change",
    "no_split_change",
    "no_checkpoint_change",
    "no_transformation_change",
    "no_xai_change",
    "no_scientific_protocol_change",
    "no_result_change",
    "no_baseline_rerun",
    "no_completed_resnet50_grad_cam_rerun",
    "no_test_based_selection_or_tuning",
    "fail_closed_on_any_mismatch",
)


def load_recovery_decision(path: str | Path) -> dict[str, Any]:
    """Load and validate the project-owner recovery authorization."""
    decision_path = Path(path)
    values = yaml.safe_load(decision_path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("Recovery Decision Record must be a mapping")
    mismatches: list[str] = []
    if values.get("decision_id") != "DR-RECOVERY-001":
        mismatches.append("decision_id")
    if values.get("status") != "approved":
        mismatches.append("status")
    if values.get("decision_type") != "infrastructure_only_freeze_recovery":
        mismatches.append("decision_type")
    evidence = values.get("recovery_evidence", {})
    required_hashes = (
        "archive_sha256",
        "recovery_audit_sha256",
        "manifest_sha256",
        "historical_final_freeze_record_sha256",
        "recovered_source_freeze_record_sha256",
        "checkpoint_training_protocol_hash",
        "governance_protocol_hash",
    )
    for key in required_hashes:
        value = evidence.get(key)
        if not isinstance(value, str) or len(value) != 64:
            mismatches.append(f"recovery_evidence.{key}")
    if evidence.get("verified_sample_count") != 8384:
        mismatches.append("recovery_evidence.verified_sample_count")
    if evidence.get("split_counts") != {
        "train": 5328,
        "validation": 1363,
        "test": 1693,
    }:
        mismatches.append("recovery_evidence.split_counts")
    policy = values.get("recovery_policy", {})
    for key in REQUIRED_RECOVERY_POLICY:
        if policy.get(key) is not True:
            mismatches.append(f"recovery_policy.{key}")
    remaining = values.get("authorized_remaining_joint_parts")
    expected_remaining = [
        {"model_id": "resnet50", "xai_method": "grad_cam_plus_plus"},
        {"model_id": "resnet50", "xai_method": "score_cam"},
        {"model_id": "efficientnet_b0", "xai_method": "grad_cam"},
        {"model_id": "efficientnet_b0", "xai_method": "grad_cam_plus_plus"},
        {"model_id": "efficientnet_b0", "xai_method": "score_cam"},
    ]
    if remaining != expected_remaining:
        mismatches.append("authorized_remaining_joint_parts")
    if mismatches:
        raise ValueError(f"Recovery decision mismatch: {sorted(mismatches)}")
    return values


def validate_recovery_binding(
    *,
    manifest_path: str | Path,
    recovery_decision_path: str | Path,
    recovery_binding_report_path: str | Path,
) -> dict[str, Any]:
    """Validate a recovered physical freeze without opening image pixels."""
    manifest = Path(manifest_path)
    freeze_path = manifest.parent / "freeze_record.json"
    leakage_path = manifest.parent / "split_leakage_report.json"
    decision_path = Path(recovery_decision_path)
    report_path = Path(recovery_binding_report_path)
    decision = load_recovery_decision(decision_path)
    evidence = decision["recovery_evidence"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    leakage = json.loads(leakage_path.read_text(encoding="utf-8"))
    physical_freeze_sha256 = sha256_file(freeze_path)
    decision_sha256 = sha256_file(decision_path)
    report_sha256 = sha256_file(report_path)
    manifest_sha256 = sha256_file(manifest)

    mismatches: list[str] = []
    if report.get("schema_version") != RECOVERY_SCHEMA_VERSION:
        mismatches.append("report.schema_version")
    if report.get("run_type") != "infrastructure_only_physical_freeze_recovery":
        mismatches.append("report.run_type")
    if report.get("recovery_gate_passed") is not True:
        mismatches.append("report.recovery_gate_passed")
    if not all(report.get("acceptance_criteria", {}).values()):
        mismatches.append("report.acceptance_criteria")
    expected_report = {
        "recovery_decision_id": decision["decision_id"],
        "recovery_decision_record_sha256": decision_sha256,
        "archive_sha256": evidence["archive_sha256"],
        "recovery_audit_sha256": evidence["recovery_audit_sha256"],
        "manifest_sha256": evidence["manifest_sha256"],
        "verified_sample_count": evidence["verified_sample_count"],
        "split_counts": evidence["split_counts"],
        "historical_final_freeze_record_sha256": evidence[
            "historical_final_freeze_record_sha256"
        ],
        "source_freeze_record_sha256": evidence[
            "recovered_source_freeze_record_sha256"
        ],
        "physical_freeze_record_sha256": physical_freeze_sha256,
        "official_test_evaluation_computed": False,
    }
    for key, expected in expected_report.items():
        if report.get(key) != expected:
            mismatches.append(f"report.{key}")
    if manifest_sha256 != evidence["manifest_sha256"]:
        mismatches.append("manifest_sha256")
    expected_freeze = {
        "schema_version": RECOVERY_SCHEMA_VERSION,
        "protocol_hash": evidence["checkpoint_training_protocol_hash"],
        "manifest_sha256": evidence["manifest_sha256"],
        "historical_final_freeze_record_sha256": evidence[
            "historical_final_freeze_record_sha256"
        ],
        "source_freeze_record_sha256": evidence[
            "recovered_source_freeze_record_sha256"
        ],
        "recovery_decision_id": decision["decision_id"],
        "recovery_decision_record_sha256": decision_sha256,
        "recovery_audit_sha256": evidence["recovery_audit_sha256"],
        "archive_sha256": evidence["archive_sha256"],
    }
    for key, expected in expected_freeze.items():
        if freeze.get(key) != expected:
            mismatches.append(f"freeze_record.{key}")
    if physical_freeze_sha256 == evidence["historical_final_freeze_record_sha256"]:
        mismatches.append("physical_freeze_not_distinct")
    artifact_hashes = freeze.get("artifact_sha256", {})
    for name, expected_hash in artifact_hashes.items():
        artifact = manifest.parent / str(name)
        if not artifact.is_file() or sha256_file(artifact) != expected_hash:
            mismatches.append(f"artifact_sha256.{name}")
    if artifact_hashes.get(manifest.name) != manifest_sha256:
        mismatches.append("artifact_sha256.manifest")
    if leakage.get("passed") is not True:
        mismatches.append("split_leakage_report")
    if mismatches:
        raise ValueError(f"Recovery binding mismatch: {sorted(set(mismatches))}")
    return {
        "recovery_decision_id": decision["decision_id"],
        "recovery_decision_record_sha256": decision_sha256,
        "recovery_binding_report_sha256": report_sha256,
        "historical_final_freeze_record_sha256": evidence[
            "historical_final_freeze_record_sha256"
        ],
        "physical_freeze_record_sha256": physical_freeze_sha256,
        "archive_sha256": evidence["archive_sha256"],
        "recovery_audit_sha256": evidence["recovery_audit_sha256"],
        "manifest_sha256": manifest_sha256,
    }


def validate_preserved_joint_part(
    *,
    part_dir: str | Path,
    recovery_decision_path: str | Path,
) -> dict[str, Any]:
    """Allow exactly the completed legacy part named by DR-RECOVERY-001."""
    path = Path(part_dir)
    report_path = path / "joint_run_report.json"
    state_path = path / "run_state.json"
    decision = load_recovery_decision(recovery_decision_path)
    expected = decision["preserved_official_results"]["completed_joint_part"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    identity = report.get("run_identity", {})
    mismatches: list[str] = []
    if sha256_file(report_path) != expected["joint_run_report_sha256"]:
        mismatches.append("joint_run_report_sha256")
    for key in ("model_id", "xai_method", "run_id"):
        if identity.get(key) != expected[key]:
            mismatches.append(key)
    if identity.get("run_identity_sha256") != expected["run_identity_sha256"]:
        mismatches.append("run_identity_sha256")
    if identity.get("git_commit") != expected["source_git_commit"]:
        mismatches.append("source_git_commit")
    if report.get("official_test_result") is not True:
        mismatches.append("official_test_result")
    if state.get("status") != "complete" or state.get("run_identity") != identity:
        mismatches.append("run_state")
    for name, expected_hash in report.get("artifact_sha256", {}).items():
        artifact = path / name
        if not artifact.is_file() or sha256_file(artifact) != expected_hash:
            mismatches.append(f"artifact_sha256.{name}")
    if mismatches:
        raise ValueError(f"Preserved joint part mismatch: {sorted(set(mismatches))}")
    return {
        "preserved_part": True,
        "joint_run_report_sha256": expected["joint_run_report_sha256"],
        "run_identity_sha256": expected["run_identity_sha256"],
    }


def authorize_recovery_joint_part(
    *,
    model_id: str,
    xai_method: str,
    recovery_decision_path: str | Path,
) -> None:
    """Permit only one of the five unfinished model-method cells."""
    decision = load_recovery_decision(recovery_decision_path)
    requested = {"model_id": model_id, "xai_method": xai_method}
    if requested not in decision["authorized_remaining_joint_parts"]:
        raise ValueError(
            "DR-RECOVERY-001 does not authorize this model-method part; "
            "completed results must not be rerun"
        )


def validate_recovered_completed_joint_part(
    *,
    part_dir: str | Path,
    model_id: str,
    xai_method: str,
    recovery_lineage: dict[str, Any],
) -> None:
    """Verify a completed post-recovery part before the orchestrator skips it."""
    path = Path(part_dir)
    report = json.loads(
        (path / "joint_run_report.json").read_text(encoding="utf-8")
    )
    state = json.loads((path / "run_state.json").read_text(encoding="utf-8"))
    identity = report.get("run_identity", {})
    mismatches: list[str] = []
    if identity.get("model_id") != model_id:
        mismatches.append("model_id")
    if identity.get("xai_method") != xai_method:
        mismatches.append("xai_method")
    if identity.get("recovery_lineage") != recovery_lineage:
        mismatches.append("recovery_lineage")
    if report.get("official_test_result") is not True:
        mismatches.append("official_test_result")
    if state.get("status") != "complete" or state.get("run_identity") != identity:
        mismatches.append("run_state")
    for name, expected_hash in report.get("artifact_sha256", {}).items():
        artifact = path / name
        if not artifact.is_file() or sha256_file(artifact) != expected_hash:
            mismatches.append(f"artifact_sha256.{name}")
    if mismatches:
        raise ValueError(
            f"Recovered completed joint part mismatch: {sorted(set(mismatches))}"
        )
