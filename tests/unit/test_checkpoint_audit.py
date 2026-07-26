import numpy as np
import pytest

from plantxai_stability.checkpoint_audit import (
    classification_audit,
    validate_training_evidence,
)


def test_classification_audit_reconciles_metrics_and_classes() -> None:
    truth = [0, 0, 1, 1, 2, 2]
    probabilities = np.asarray(
        [
            [0.9, 0.1, 0.0],
            [0.6, 0.3, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.4, 0.5],
            [0.1, 0.2, 0.7],
            [0.2, 0.1, 0.7],
        ],
        dtype=np.float64,
    )
    predicted = probabilities.argmax(axis=1).tolist()
    result = classification_audit(
        truth, predicted, probabilities, ["class-a", "class-b", "class-c"]
    )
    assert result["sample_count"] == 6
    assert result["error_count"] == 1
    assert sum(sum(row) for row in result["confusion_matrix"]) == 6
    assert [item["support"] for item in result["per_class"]] == [2, 2, 2]
    assert 0.0 < result["macro_f1"] < 1.0
    assert np.isfinite(result["negative_log_likelihood"])


def test_classification_audit_rejects_probability_argmax_mismatch() -> None:
    probabilities = np.asarray([[0.8, 0.2], [0.1, 0.9]], dtype=np.float64)
    with pytest.raises(ValueError, match="probability argmax"):
        classification_audit([0, 1], [1, 1], probabilities, ["a", "b"])


def _checkpoint_payload() -> dict[str, object]:
    return {
        "checkpoint_role": "validation_selected_best",
        "model_id": "resnet50",
        "protocol_hash": "protocol-hash",
        "manifest_sha256": "manifest-hash",
        "class_names": ["a", "b"],
        "seed": 42,
        "test_split_accessed": False,
        "validation_macro_f1": 0.9,
        "best_epoch": 7,
    }


def _training_evidence() -> dict[str, object]:
    return {
        "model_id": "resnet50",
        "checkpoint_sha256": "checkpoint-hash",
        "freeze_record_sha256": "freeze-hash",
        "validation_metric": 0.9,
        "config_hash": "protocol-hash",
        "manifest_sha256": "manifest-hash",
        "run_type": "official_checkpoint_selection",
        "official": True,
        "protocol_hash": "protocol-hash",
        "freeze_record_protocol_hash": "protocol-hash",
        "train_sample_count": 10,
        "validation_sample_count": 5,
        "test_split_accessed": False,
        "seed": 42,
        "best_epoch": 7,
    }


def test_training_evidence_accepts_exact_lineage() -> None:
    validate_training_evidence(
        _training_evidence(),
        _checkpoint_payload(),
        model_id="resnet50",
        protocol_hash="protocol-hash",
        manifest_sha256="manifest-hash",
        checkpoint_sha256="checkpoint-hash",
        freeze_record_sha256="freeze-hash",
        seed=42,
        class_names=["a", "b"],
        expected_train_count=10,
        expected_validation_count=5,
    )


def test_training_evidence_rejects_test_access_or_hash_change() -> None:
    evidence = _training_evidence()
    evidence["test_split_accessed"] = True
    evidence["checkpoint_sha256"] = "changed"
    with pytest.raises(ValueError, match="checkpoint_sha256.*test_split_accessed"):
        validate_training_evidence(
            evidence,
            _checkpoint_payload(),
            model_id="resnet50",
            protocol_hash="protocol-hash",
            manifest_sha256="manifest-hash",
            checkpoint_sha256="checkpoint-hash",
            freeze_record_sha256="freeze-hash",
            seed=42,
            class_names=["a", "b"],
            expected_train_count=10,
            expected_validation_count=5,
        )
