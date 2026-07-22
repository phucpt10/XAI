import pytest

from plantxai_stability.g2_readiness import validate_g1_audit_evidence


def _decision() -> dict[str, object]:
    return {
        "training_lineage": {
            "protocol_hash": "protocol",
            "manifest_sha256": "manifest",
            "freeze_record_sha256": "freeze",
            "validation_sample_count": 10,
            "validation_sample_ids_sha256": "samples",
        },
        "approved_checkpoints": {
            "resnet50": {
                "checkpoint_sha256": "checkpoint",
                "checkpoint_evidence_sha256": "evidence",
                "validation_macro_f1": 0.9,
                "validation_accuracy": 0.8,
                "validation_error_count": 2,
                "validation_artifact_sha256": {"predictions.csv": "artifact"},
            }
        },
    }


def _report() -> dict[str, object]:
    return {
        "run_type": "official_validation_checkpoint_audit",
        "official_checkpoint_selection_evidence": True,
        "official_test_result": False,
        "source_split": "validation",
        "test_split_accessed": False,
        "model_id": "resnet50",
        "protocol_hash": "protocol",
        "manifest_sha256": "manifest",
        "freeze_record_sha256": "freeze",
        "checkpoint_sha256": "checkpoint",
        "checkpoint_evidence_sha256": "evidence",
        "sample_ids_sha256": "samples",
        "acceptance_criteria": {"validation_only": True},
        "artifact_sha256": {"predictions.csv": "artifact"},
        "metrics": {
            "sample_count": 10,
            "macro_f1": 0.9,
            "accuracy": 0.8,
            "error_count": 2,
        },
    }


def test_validate_g1_audit_evidence_accepts_exact_registry_binding() -> None:
    result = validate_g1_audit_evidence(
        _report(), _decision(), model_id="resnet50"
    )
    assert result["checkpoint_sha256"] == "checkpoint"
    assert result["validation_error_count"] == 2


def test_validate_g1_audit_evidence_rejects_test_access_and_artifact_change() -> None:
    report = _report()
    report["test_split_accessed"] = True
    report["artifact_sha256"] = {"predictions.csv": "changed"}
    with pytest.raises(ValueError, match="test_split_accessed.*validation_artifact"):
        validate_g1_audit_evidence(report, _decision(), model_id="resnet50")
