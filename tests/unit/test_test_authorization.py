from dataclasses import dataclass

import pytest

from plantxai_stability.test_authorization import (
    validate_g2_authorization,
    validate_official_test_metadata,
)


def _checkpoint_decision() -> dict[str, object]:
    return {
        "decision_id": "DR-CHECKPOINT-001",
        "approved_checkpoints": {
            "resnet50": {"checkpoint_sha256": "checkpoint"}
        },
    }


def _readiness() -> dict[str, object]:
    return {
        "run_type": "metadata_only_g2_readiness",
        "approval_status": "pending_g2_human_review",
        "technical_gate_passed": True,
        "governance_protocol_hash": "g1-hash",
        "checkpoint_training_protocol_hash": "training-hash",
        "split_summary_sha256": "split-summary",
        "acceptance_criteria": {
            "technical_check": True,
            "official_test_pixels_accessed": False,
        },
        "official_test": {
            "metadata_sample_count": 2,
            "metadata_leaf_count": 2,
            "sample_ids_sha256": "sample-hash",
            "leaf_ids_sha256": "leaf-hash",
            "pixels_accessed": False,
            "result_computed": False,
        },
    }


def _test_decision() -> dict[str, object]:
    return {
        "decision_id": "DR-TEST-001",
        "status": "approved",
        "readiness_evidence": {
            "report_sha256": "readiness-hash",
            "run_type": "metadata_only_g2_readiness",
            "technical_gate_passed": True,
            "approval_status_before_decision": "pending_g2_human_review",
            "g1_governance_protocol_hash": "g1-hash",
            "checkpoint_training_protocol_hash": "training-hash",
            "checkpoint_decision_record_id": "DR-CHECKPOINT-001",
            "checkpoint_decision_record_sha256": "checkpoint-dr-hash",
            "manifest_sha256": "manifest-hash",
            "freeze_record_sha256": "freeze-hash",
            "split_summary_sha256": "split-summary",
            "official_test_pixels_accessed": False,
            "official_test_result_computed": False,
        },
        "official_test_identity": {
            "sample_count": 2,
            "leaf_count": 2,
            "sample_ids_sha256": "sample-hash",
            "leaf_ids_sha256": "leaf-hash",
            "source_membership_preserved_exactly": True,
        },
        "registered_campaign": {
            "campaign_id": "plantxai-official-test-v1",
            "models": ["resnet50"],
            "scenario_ids": ["rotation_mild"],
            "xai_methods": ["grad_cam"],
        },
        "approved_checkpoints": {"resnet50": "checkpoint"},
        "execution_policy": {
            "one_registered_campaign": True,
            "immutable_versioned_outputs": True,
            "no_overwrite": True,
            "no_checkpoint_reselection_after_test": True,
            "no_model_or_hyperparameter_tuning_after_test": True,
            "no_transformation_or_xai_tuning_after_test": True,
        },
        "governance_effect": {
            "governance_protocol_hash": "g2-hash",
            "G2_TEST_EVALUATION_READY": "pass",
            "test_evaluation_blockers": [],
            "official_experiment_allowed": True,
            "official_test_evaluation_allowed": True,
            "operational_state": "approved_pending_runtime_authorization_gate_verification",
        },
    }


def _governance() -> dict[str, object]:
    return {
        "G1_CHECKPOINT_SELECTION": "pass",
        "G2_TEST_EVALUATION_READY": "pass",
        "test_evaluation_blockers": [],
        "official_experiment_allowed": True,
        "official_test_evaluation_allowed": True,
        "evidence_records": {"test_evaluation": "DR-TEST-001"},
    }


def test_g2_authorization_accepts_exact_registered_campaign() -> None:
    result = validate_g2_authorization(
        governance=_governance(),
        governance_protocol_hash="g2-hash",
        declared_models=["resnet50"],
        declared_scenarios=["rotation_mild"],
        declared_xai_methods=["grad_cam"],
        test_decision=_test_decision(),
        readiness_report=_readiness(),
        readiness_report_sha256="readiness-hash",
        checkpoint_decision=_checkpoint_decision(),
        checkpoint_decision_sha256="checkpoint-dr-hash",
        manifest_sha256="manifest-hash",
        freeze_record_sha256="freeze-hash",
    )
    assert result["campaign_id"] == "plantxai-official-test-v1"
    assert result["checkpoint_training_protocol_hash"] == "training-hash"


def test_g2_authorization_rejects_test_pixel_access_and_campaign_change() -> None:
    readiness = _readiness()
    readiness["official_test"]["pixels_accessed"] = True  # type: ignore[index]
    decision = _test_decision()
    decision["registered_campaign"]["scenario_ids"] = []  # type: ignore[index]
    with pytest.raises(ValueError, match="official_test.pixels_accessed.*scenario_ids"):
        validate_g2_authorization(
            governance=_governance(),
            governance_protocol_hash="g2-hash",
            declared_models=["resnet50"],
            declared_scenarios=["rotation_mild"],
            declared_xai_methods=["grad_cam"],
            test_decision=decision,
            readiness_report=readiness,
            readiness_report_sha256="readiness-hash",
            checkpoint_decision=_checkpoint_decision(),
            checkpoint_decision_sha256="checkpoint-dr-hash",
            manifest_sha256="manifest-hash",
            freeze_record_sha256="freeze-hash",
        )


@dataclass
class _Record:
    sample_id: str
    leaf_id: str
    split: str
    source_split: str


def test_official_test_metadata_rejects_identity_mismatch() -> None:
    records = [_Record("a", "leaf-a", "test", "test")]
    with pytest.raises(ValueError, match="leaf_count.*sample_count"):
        validate_official_test_metadata(records, _test_decision())
