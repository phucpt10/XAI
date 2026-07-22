from __future__ import annotations

import csv
import json

import pytest

from plantxai_stability.provenance import sha256_file
from scripts.merge_joint_runs import (
    _load_part,
    _prediction_payload,
    _validate_baseline_binding,
)


def _write_csv(path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_load_part_verifies_state_identity_and_artifact_hashes(tmp_path) -> None:
    identity = {
        "run_id": "part-1",
        "model_id": "resnet50",
        "xai_method": "grad_cam",
    }
    predictions = tmp_path / "prediction_results.csv"
    joint = tmp_path / "joint_results.csv"
    _write_csv(predictions, [{"run_id": "part-1", "sample_id": "sample_a"}])
    _write_csv(
        joint,
        [
            {
                "run_id": "part-1",
                "sample_id": "sample_a",
                "xai_method": "grad_cam",
            }
        ],
    )
    report = {
        "run_type": "authorized_official_test_joint_part",
        "official_test_result": True,
        "run_identity": identity,
        "artifact_sha256": {
            predictions.name: sha256_file(predictions),
            joint.name: sha256_file(joint),
        },
    }
    (tmp_path / "joint_run_report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    (tmp_path / "run_state.json").write_text(
        json.dumps({"status": "complete", "run_identity": identity}),
        encoding="utf-8",
    )
    loaded = _load_part(tmp_path)
    assert loaded["report"]["run_identity"] == identity
    predictions.write_text("corrupted", encoding="utf-8")
    with pytest.raises(SystemExit, match="hash mismatch"):
        _load_part(tmp_path)


def test_prediction_comparison_ignores_only_part_run_id() -> None:
    left = [{"run_id": "grad", "sample_id": "sample_a", "confidence": "0.9"}]
    right = [{"run_id": "score", "sample_id": "sample_a", "confidence": "0.9"}]
    assert _prediction_payload(left) == _prediction_payload(right)
    right[0]["confidence"] = "0.8"
    assert _prediction_payload(left) != _prediction_payload(right)


def test_baseline_binding_rejects_checkpoint_mismatch() -> None:
    identity = {
        "model_id": "resnet50",
        "campaign_id": "campaign",
        "governance_protocol_hash": "governance",
        "checkpoint_training_protocol_hash": "training",
        "checkpoint_sha256": "checkpoint",
        "manifest_sha256": "manifest",
        "freeze_record_sha256": "freeze",
    }
    baseline = dict(identity)
    _validate_baseline_binding(baseline, identity)
    baseline["checkpoint_sha256"] = "different"
    with pytest.raises(SystemExit, match="checkpoint_sha256"):
        _validate_baseline_binding(baseline, identity)
