"""Pure validation-metric and checkpoint-evidence audit helpers."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def classification_audit(
    truth: Sequence[int],
    predicted: Sequence[int],
    probabilities: np.ndarray,
    class_names: Sequence[str],
) -> dict[str, Any]:
    """Compute deterministic multiclass validation metrics with strict checks."""
    y_true = np.asarray(truth, dtype=np.int64)
    y_pred = np.asarray(predicted, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    names = list(class_names)
    sample_count = len(y_true)
    class_count = len(names)
    if sample_count == 0 or class_count < 2:
        raise ValueError("Classification audit requires samples and at least two classes")
    if y_pred.shape != y_true.shape or scores.shape != (sample_count, class_count):
        raise ValueError("Prediction arrays do not match sample and class counts")
    if not np.isfinite(scores).all():
        raise ValueError("Class probabilities contain NaN or Inf")
    if np.any(scores < 0.0) or np.any(scores > 1.0):
        raise ValueError("Class probabilities must be in [0, 1]")
    if not np.allclose(scores.sum(axis=1), 1.0, atol=1e-5, rtol=0.0):
        raise ValueError("Class probabilities do not sum to one")
    if np.any((y_true < 0) | (y_true >= class_count)) or np.any(
        (y_pred < 0) | (y_pred >= class_count)
    ):
        raise ValueError("Class IDs are outside the declared class range")
    if not np.array_equal(scores.argmax(axis=1), y_pred):
        raise ValueError("Predicted classes do not match probability argmax")
    labels = list(range(class_count))
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    true_positive = np.diag(matrix).astype(np.float64)
    support = matrix.sum(axis=1)
    predicted_support = matrix.sum(axis=0)
    precision = np.divide(
        true_positive,
        predicted_support,
        out=np.zeros(class_count, dtype=np.float64),
        where=predicted_support != 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros(class_count, dtype=np.float64),
        where=support != 0,
    )
    per_class_f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(class_count, dtype=np.float64),
        where=(precision + recall) != 0,
    )
    correct = y_true == y_pred
    confidence = scores.max(axis=1)
    true_probability = scores[np.arange(sample_count), y_true]
    epsilon = np.finfo(np.float64).eps
    one_hot = np.eye(class_count, dtype=np.float64)[y_true]
    per_class = [
        {
            "class_id": class_id,
            "class_name": names[class_id],
            "support": int(support[class_id]),
            "precision": float(precision[class_id]),
            "recall": float(recall[class_id]),
            "f1": float(per_class_f1[class_id]),
        }
        for class_id in labels
    ]
    metrics = {
        "sample_count": sample_count,
        "class_count": class_count,
        "accuracy": float(correct.mean()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(per_class_f1.mean()),
        "weighted_f1": float(np.average(per_class_f1, weights=support)),
        "negative_log_likelihood": float(-np.log(np.clip(true_probability, epsilon, 1.0)).mean()),
        "multiclass_brier_score": float(np.square(scores - one_hot).sum(axis=1).mean()),
        "mean_confidence": float(confidence.mean()),
        "mean_confidence_correct": (
            float(confidence[correct].mean()) if correct.any() else None
        ),
        "mean_confidence_incorrect": (
            float(confidence[~correct].mean()) if (~correct).any() else None
        ),
        "error_count": int(np.count_nonzero(~correct)),
        "per_class": per_class,
        "confusion_matrix": matrix.astype(int).tolist(),
        "confusion_matrix_axis": "rows=true_class; columns=predicted_class",
    }
    finite_values = [
        value
        for key, value in metrics.items()
        if key
        not in {
            "per_class",
            "confusion_matrix",
            "confusion_matrix_axis",
            "mean_confidence_correct",
            "mean_confidence_incorrect",
        }
        and isinstance(value, (int, float))
    ]
    if not np.isfinite(np.asarray(finite_values, dtype=np.float64)).all():
        raise ValueError("Classification audit produced non-finite metrics")
    if int(matrix.sum()) != sample_count or any(item["support"] == 0 for item in per_class):
        raise ValueError("Confusion matrix or class coverage is incomplete")
    return metrics


def validate_training_evidence(
    evidence: dict[str, Any],
    checkpoint_payload: dict[str, Any],
    *,
    model_id: str,
    protocol_hash: str,
    manifest_sha256: str,
    checkpoint_sha256: str,
    freeze_record_sha256: str,
    seed: int,
    class_names: Sequence[str],
    expected_train_count: int,
    expected_validation_count: int,
) -> None:
    """Fail closed when training evidence and selected checkpoint diverge."""
    expected = {
        "model_id": model_id,
        "checkpoint_sha256": checkpoint_sha256,
        "freeze_record_sha256": freeze_record_sha256,
        "seed": seed,
        "config_hash": protocol_hash,
        "manifest_sha256": manifest_sha256,
        "run_type": "official_checkpoint_selection",
        "official": True,
        "protocol_hash": protocol_hash,
        "freeze_record_protocol_hash": protocol_hash,
        "train_sample_count": expected_train_count,
        "validation_sample_count": expected_validation_count,
        "test_split_accessed": False,
    }
    mismatches = [
        key for key, value in expected.items() if evidence.get(key) != value
    ]
    observed_metric = evidence.get("validation_metric")
    if not isinstance(observed_metric, (int, float)) or not np.isclose(
        float(observed_metric),
        float(checkpoint_payload["validation_macro_f1"]),
        rtol=0.0,
        atol=1e-12,
    ):
        mismatches.append("validation_metric")
    if checkpoint_payload.get("checkpoint_role") != "validation_selected_best":
        mismatches.append("checkpoint_role")
    if checkpoint_payload.get("model_id") != model_id:
        mismatches.append("checkpoint_model_id")
    if checkpoint_payload.get("protocol_hash") != protocol_hash:
        mismatches.append("checkpoint_protocol_hash")
    if checkpoint_payload.get("manifest_sha256") != manifest_sha256:
        mismatches.append("checkpoint_manifest_sha256")
    if checkpoint_payload.get("seed") != seed:
        mismatches.append("checkpoint_seed")
    if checkpoint_payload.get("class_names") != list(class_names):
        mismatches.append("checkpoint_class_names")
    if checkpoint_payload.get("test_split_accessed") is not False:
        mismatches.append("checkpoint_test_split_accessed")
    if int(evidence.get("best_epoch", -1)) != int(checkpoint_payload.get("best_epoch", -2)):
        mismatches.append("best_epoch")
    if mismatches:
        raise ValueError(f"Checkpoint training evidence mismatch: {sorted(set(mismatches))}")
