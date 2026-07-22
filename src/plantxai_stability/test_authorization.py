"""Fail-closed authorization checks for the registered official-test campaign."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import yaml


def validate_g2_authorization(
    *,
    governance: dict[str, Any],
    governance_protocol_hash: str,
    declared_models: Sequence[str],
    declared_scenarios: Sequence[str],
    declared_xai_methods: Sequence[str],
    test_decision: dict[str, Any],
    readiness_report: dict[str, Any],
    readiness_report_sha256: str,
    checkpoint_decision: dict[str, Any],
    checkpoint_decision_sha256: str,
    manifest_sha256: str,
    freeze_record_sha256: str,
) -> dict[str, Any]:
    """Validate G2 governance and its full evidence chain without image access."""
    mismatches: list[str] = []
    evidence_records = governance.get("evidence_records", {})
    if governance.get("G1_CHECKPOINT_SELECTION") != "pass":
        mismatches.append("G1_CHECKPOINT_SELECTION")
    if governance.get("G2_TEST_EVALUATION_READY") != "pass":
        mismatches.append("G2_TEST_EVALUATION_READY")
    if governance.get("test_evaluation_blockers") != []:
        mismatches.append("test_evaluation_blockers")
    if governance.get("official_experiment_allowed") is not True:
        mismatches.append("official_experiment_allowed")
    if governance.get("official_test_evaluation_allowed") is not True:
        mismatches.append("official_test_evaluation_allowed")
    if test_decision.get("status") != "approved":
        mismatches.append("test_decision_status")
    if test_decision.get("decision_id") != evidence_records.get("test_evaluation"):
        mismatches.append("test_decision_id")
    governance_effect = test_decision.get("governance_effect", {})
    expected_governance = {
        "governance_protocol_hash": governance_protocol_hash,
        "G2_TEST_EVALUATION_READY": "pass",
        "test_evaluation_blockers": [],
        "official_experiment_allowed": True,
        "official_test_evaluation_allowed": True,
        "operational_state": "approved_pending_runtime_authorization_gate_verification",
    }
    mismatches.extend(
        f"governance_effect.{key}"
        for key, expected in expected_governance.items()
        if governance_effect.get(key) != expected
    )
    readiness_evidence = test_decision.get("readiness_evidence", {})
    expected_readiness = {
        "report_sha256": readiness_report_sha256,
        "run_type": "metadata_only_g2_readiness",
        "technical_gate_passed": True,
        "approval_status_before_decision": "pending_g2_human_review",
        "g1_governance_protocol_hash": readiness_report.get(
            "governance_protocol_hash"
        ),
        "checkpoint_training_protocol_hash": readiness_report.get(
            "checkpoint_training_protocol_hash"
        ),
        "checkpoint_decision_record_id": checkpoint_decision.get("decision_id"),
        "checkpoint_decision_record_sha256": checkpoint_decision_sha256,
        "manifest_sha256": manifest_sha256,
        "freeze_record_sha256": freeze_record_sha256,
        "split_summary_sha256": readiness_report.get("split_summary_sha256"),
        "official_test_pixels_accessed": False,
        "official_test_result_computed": False,
    }
    mismatches.extend(
        f"readiness_evidence.{key}"
        for key, expected in expected_readiness.items()
        if readiness_evidence.get(key) != expected
    )
    if readiness_report.get("run_type") != "metadata_only_g2_readiness":
        mismatches.append("readiness_report.run_type")
    if readiness_report.get("approval_status") != "pending_g2_human_review":
        mismatches.append("readiness_report.approval_status")
    if readiness_report.get("technical_gate_passed") is not True:
        mismatches.append("readiness_report.technical_gate_passed")
    readiness_criteria = readiness_report.get("acceptance_criteria", {})
    for key, value in readiness_criteria.items():
        expected = False if key == "official_test_pixels_accessed" else True
        if value is not expected:
            mismatches.append(f"readiness_report.acceptance_criteria.{key}")
    official_test = readiness_report.get("official_test", {})
    if official_test.get("pixels_accessed") is not False:
        mismatches.append("readiness_report.official_test.pixels_accessed")
    if official_test.get("result_computed") is not False:
        mismatches.append("readiness_report.official_test.result_computed")
    identity = test_decision.get("official_test_identity", {})
    for key in ("sample_count", "leaf_count", "sample_ids_sha256", "leaf_ids_sha256"):
        if identity.get(key) != official_test.get(
            "metadata_" + key if key in {"sample_count", "leaf_count"} else key
        ):
            mismatches.append(f"official_test_identity.{key}")
    if identity.get("source_membership_preserved_exactly") is not True:
        mismatches.append("official_test_identity.source_membership_preserved_exactly")
    campaign = test_decision.get("registered_campaign", {})
    expected_campaign = {
        "campaign_id": "plantxai-official-test-v1",
        "models": list(declared_models),
        "scenario_ids": list(declared_scenarios),
        "xai_methods": list(declared_xai_methods),
    }
    mismatches.extend(
        f"registered_campaign.{key}"
        for key, expected in expected_campaign.items()
        if campaign.get(key) != expected
    )
    if test_decision.get("approved_checkpoints") != {
        model_id: checkpoint_decision.get("approved_checkpoints", {})
        .get(model_id, {})
        .get("checkpoint_sha256")
        for model_id in declared_models
    }:
        mismatches.append("approved_checkpoints")
    policy = test_decision.get("execution_policy", {})
    required_true = (
        "one_registered_campaign",
        "immutable_versioned_outputs",
        "no_overwrite",
        "no_checkpoint_reselection_after_test",
        "no_model_or_hyperparameter_tuning_after_test",
        "no_transformation_or_xai_tuning_after_test",
    )
    mismatches.extend(
        f"execution_policy.{key}" for key in required_true if policy.get(key) is not True
    )
    if mismatches:
        raise ValueError(f"G2 authorization mismatch: {sorted(set(mismatches))}")
    return {
        "authorization_decision_id": test_decision["decision_id"],
        "campaign_id": campaign["campaign_id"],
        "checkpoint_training_protocol_hash": readiness_report[
            "checkpoint_training_protocol_hash"
        ],
        "official_test": official_test,
    }


def validate_official_test_metadata(
    records: Sequence[Any], test_decision: dict[str, Any]
) -> dict[str, Any]:
    """Match manifest-only official-test identities to the authorized campaign."""
    test_records = sorted(
        (record for record in records if record.split == "test"),
        key=lambda record: record.sample_id,
    )
    sample_ids = [str(record.sample_id) for record in test_records]
    leaf_ids = sorted({str(record.leaf_id) for record in test_records})
    identity = test_decision.get("official_test_identity", {})
    mismatches: list[str] = []
    if len(sample_ids) != identity.get("sample_count"):
        mismatches.append("sample_count")
    if len(leaf_ids) != identity.get("leaf_count"):
        mismatches.append("leaf_count")
    if len(sample_ids) != len(set(sample_ids)):
        mismatches.append("unique_sample_ids")
    if any(record.source_split != "test" for record in test_records):
        mismatches.append("test_source_membership")
    if sum(record.source_split == "test" for record in records) != len(test_records):
        mismatches.append("official_test_outside_test_split")
    if _identity_hash(sample_ids) != identity.get("sample_ids_sha256"):
        mismatches.append("sample_ids_sha256")
    if _identity_hash(leaf_ids) != identity.get("leaf_ids_sha256"):
        mismatches.append("leaf_ids_sha256")
    if mismatches:
        raise ValueError(f"Official-test metadata mismatch: {sorted(mismatches)}")
    return {
        "sample_count": len(sample_ids),
        "leaf_count": len(leaf_ids),
        "sample_ids_sha256": identity["sample_ids_sha256"],
        "leaf_ids_sha256": identity["leaf_ids_sha256"],
    }


def authorize_official_test_run(
    resolved: Any,
    *,
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    model_id: str,
    checkpoint_decision_path: str | Path,
    test_decision_path: str | Path,
    readiness_report_path: str | Path,
) -> dict[str, Any]:
    """Validate every authorization artifact before a runner opens test pixels."""
    from plantxai_stability.data.freeze import require_frozen_artifacts
    from plantxai_stability.data.manifest import read_manifest_csv
    from plantxai_stability.data.splits import validate_frozen_splits
    from plantxai_stability.governance import approved_checkpoint_lineage
    from plantxai_stability.provenance import sha256_file

    manifest = Path(manifest_path)
    checkpoint = Path(checkpoint_path)
    checkpoint_decision_file = Path(checkpoint_decision_path)
    test_decision_file = Path(test_decision_path)
    readiness_file = Path(readiness_report_path)
    manifest_sha256 = sha256_file(manifest)
    require_frozen_artifacts(manifest)
    freeze_record_sha256 = sha256_file(manifest.parent / "freeze_record.json")
    checkpoint_decision = _load_yaml_mapping(checkpoint_decision_file)
    test_decision = _load_yaml_mapping(test_decision_file)
    readiness_report = json.loads(readiness_file.read_text(encoding="utf-8"))
    scenarios = [
        f"{name}_{severity}"
        for name in resolved.values["transformations"]["names"]
        for severity in resolved.values["transformations"]["severities"]
    ]
    authorization = validate_g2_authorization(
        governance=resolved.values["governance"],
        governance_protocol_hash=resolved.sha256,
        declared_models=resolved.values["models"],
        declared_scenarios=scenarios,
        declared_xai_methods=resolved.values["xai"]["methods"],
        test_decision=test_decision,
        readiness_report=readiness_report,
        readiness_report_sha256=sha256_file(readiness_file),
        checkpoint_decision=checkpoint_decision,
        checkpoint_decision_sha256=sha256_file(checkpoint_decision_file),
        manifest_sha256=manifest_sha256,
        freeze_record_sha256=freeze_record_sha256,
    )
    records = read_manifest_csv(manifest)
    validate_frozen_splits(records)
    test_identity = validate_official_test_metadata(records, test_decision)
    checkpoint_sha256 = sha256_file(checkpoint)
    lineage = approved_checkpoint_lineage(
        checkpoint_decision,
        resolved.values["governance"],
        model_id=model_id,
        declared_models=resolved.values["models"],
        checkpoint_sha256=checkpoint_sha256,
        manifest_sha256=manifest_sha256,
        freeze_record_sha256=freeze_record_sha256,
    )
    if lineage["training_protocol_hash"] != authorization[
        "checkpoint_training_protocol_hash"
    ]:
        raise ValueError("G2 authorization and checkpoint training lineage diverge")
    return {
        "authorization_decision_id": authorization["authorization_decision_id"],
        "campaign_id": authorization["campaign_id"],
        "checkpoint_training_protocol_hash": lineage["training_protocol_hash"],
        "checkpoint_sha256": checkpoint_sha256,
        "manifest_sha256": manifest_sha256,
        "freeze_record_sha256": freeze_record_sha256,
        "test_decision_record_sha256": sha256_file(test_decision_file),
        "checkpoint_decision_record_sha256": sha256_file(checkpoint_decision_file),
        "g2_readiness_report_sha256": sha256_file(readiness_file),
        "test_identity": test_identity,
        "test_records": [record for record in records if record.split == "test"],
    }


def _identity_hash(identities: list[str]) -> str:
    payload = json.dumps(identities, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError(f"Decision Record must be a mapping: {path}")
    return values
