"""Decision-record validation for immutable approved checkpoint lineage."""

from __future__ import annotations

from typing import Any, Sequence


def approved_checkpoint_lineage(
    decision: dict[str, Any],
    governance: dict[str, Any],
    *,
    model_id: str,
    declared_models: Sequence[str],
    checkpoint_sha256: str,
    manifest_sha256: str,
    freeze_record_sha256: str,
) -> dict[str, Any]:
    """Return approved training lineage or fail closed on any G1 mismatch."""
    expected_decision_id = governance.get("evidence_records", {}).get(
        "checkpoint_selection"
    )
    mismatches: list[str] = []
    if governance.get("G1_CHECKPOINT_SELECTION") != "pass":
        mismatches.append("G1_CHECKPOINT_SELECTION")
    if decision.get("status") != "approved":
        mismatches.append("decision_status")
    if decision.get("decision_id") != expected_decision_id:
        mismatches.append("decision_id")
    policy = decision.get("selection_policy", {})
    if policy.get("declared_models") != list(declared_models):
        mismatches.append("declared_models")
    if policy.get("official_test_used_for_selection") is not False:
        mismatches.append("official_test_used_for_selection")
    audit = decision.get("validation_audit", {})
    if audit.get("source_split") != "validation":
        mismatches.append("validation_source_split")
    if audit.get("test_split_accessed") is not False:
        mismatches.append("validation_test_split_accessed")
    lineage = decision.get("training_lineage", {})
    if lineage.get("manifest_sha256") != manifest_sha256:
        mismatches.append("manifest_sha256")
    if lineage.get("freeze_record_sha256") != freeze_record_sha256:
        mismatches.append("freeze_record_sha256")
    training_protocol_hash = lineage.get("protocol_hash")
    if not isinstance(training_protocol_hash, str) or len(training_protocol_hash) != 64:
        mismatches.append("training_protocol_hash")
    checkpoint = decision.get("approved_checkpoints", {}).get(model_id)
    if not isinstance(checkpoint, dict):
        mismatches.append("approved_checkpoint_model")
    elif checkpoint.get("checkpoint_sha256") != checkpoint_sha256:
        mismatches.append("checkpoint_sha256")
    if mismatches:
        raise ValueError(f"Approved checkpoint lineage mismatch: {sorted(mismatches)}")
    return {
        "decision_id": decision["decision_id"],
        "training_protocol_hash": training_protocol_hash,
        "manifest_sha256": lineage["manifest_sha256"],
        "freeze_record_sha256": lineage["freeze_record_sha256"],
        "checkpoint": checkpoint,
    }
