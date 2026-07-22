from __future__ import annotations

import json

import pytest

from plantxai_stability.joint_execution import (
    JointProgressStore,
    build_run_identity,
    validate_completed_coverage,
    write_run_state,
)


def _identity() -> dict[str, object]:
    return build_run_identity(
        run_id="official-resnet-gradcam-v1",
        model_id="resnet50",
        xai_method="grad_cam",
        scenario_ids=["rotation_mild", "brightness_mild"],
        sample_ids=["sample_a", "sample_b"],
        seed=42,
        governance_protocol_hash="governance",
        checkpoint_training_protocol_hash="training",
        checkpoint_sha256="checkpoint",
        manifest_sha256="manifest",
        freeze_record_sha256="freeze",
        checkpoint_decision_record_sha256="checkpoint_dr",
        test_decision_record_sha256="test_dr",
        g2_readiness_report_sha256="readiness",
        campaign_id="plantxai-official-test-v1",
        authorization_decision_id="DR-TEST-001",
        transformation_algorithm_version="zero_fill_v6",
        xai_policy={"alignment_policy": "forward_align_original_cam"},
        software_version="0.1.0",
        git_commit="abc123",
        runtime_identity={"torch_version": "2.11.0", "gpu": "A100"},
    )


def _rows(sample_id: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    scenarios = ["rotation_mild", "brightness_mild"]
    predictions = [
        {"sample_id": sample_id, "scenario_id": scenario} for scenario in scenarios
    ]
    joint = [
        {
            "sample_id": sample_id,
            "scenario_id": scenario,
            "xai_method": "grad_cam",
        }
        for scenario in scenarios
    ]
    return predictions, joint


def test_progress_store_commits_complete_sample_and_resumes(tmp_path) -> None:
    identity = _identity()
    database = tmp_path / "progress.sqlite3"
    predictions, joint = _rows("sample_a")
    with JointProgressStore(database, identity) as store:
        store.write_sample(
            sample_id="sample_a",
            leaf_id="leaf_a",
            prediction_rows=predictions,
            joint_rows=joint,
            expected_scenario_ids=["rotation_mild", "brightness_mild"],
            expected_xai_method="grad_cam",
        )
        assert store.completed_sample_ids() == {"sample_a"}
    with JointProgressStore(database, identity) as resumed:
        assert resumed.completed_count() == 1
        assert list(resumed.iter_rows("prediction_rows_json")) == predictions
        assert list(resumed.iter_rows("joint_rows_json")) == joint


def test_progress_store_rejects_changed_resume_identity(tmp_path) -> None:
    identity = _identity()
    database = tmp_path / "progress.sqlite3"
    with JointProgressStore(database, identity):
        pass
    changed = dict(identity)
    changed["checkpoint_sha256"] = "different"
    with pytest.raises(ValueError, match="run identity"):
        JointProgressStore(database, changed)


def test_progress_store_rejects_partial_factorial_sample(tmp_path) -> None:
    predictions, joint = _rows("sample_a")
    with JointProgressStore(tmp_path / "progress.sqlite3", _identity()) as store:
        with pytest.raises(ValueError, match="Prediction coverage mismatch"):
            store.write_sample(
                sample_id="sample_a",
                leaf_id="leaf_a",
                prediction_rows=predictions[:1],
                joint_rows=joint,
                expected_scenario_ids=["rotation_mild", "brightness_mild"],
                expected_xai_method="grad_cam",
            )
        assert store.completed_count() == 0


def test_completed_coverage_requires_full_cross_product() -> None:
    predictions: list[dict[str, object]] = []
    joint: list[dict[str, object]] = []
    for sample_id in ("sample_a", "sample_b"):
        sample_predictions, sample_joint = _rows(sample_id)
        predictions.extend(sample_predictions)
        joint.extend(sample_joint)
    criteria = validate_completed_coverage(
        completed_sample_ids={"sample_a", "sample_b"},
        expected_sample_ids=["sample_a", "sample_b"],
        prediction_rows=predictions,
        joint_rows=joint,
        scenario_ids=["rotation_mild", "brightness_mild"],
        xai_method="grad_cam",
    )
    assert all(criteria.values())
    with pytest.raises(ValueError, match="coverage failed"):
        validate_completed_coverage(
            completed_sample_ids={"sample_a", "sample_b"},
            expected_sample_ids=["sample_a", "sample_b"],
            prediction_rows=predictions[:-1],
            joint_rows=joint,
            scenario_ids=["rotation_mild", "brightness_mild"],
            xai_method="grad_cam",
        )


def test_run_state_is_atomic_and_records_retry_count(tmp_path) -> None:
    path = tmp_path / "run_state.json"
    write_run_state(
        path,
        identity=_identity(),
        status="in_progress",
        completed_sample_count=1,
        retry_count=2,
    )
    state = json.loads(path.read_text(encoding="utf-8"))
    assert state["status"] == "in_progress"
    assert state["completed_sample_count"] == 1
    assert state["retry_count"] == 2
    assert not path.with_suffix(".json.tmp").exists()


def test_run_identity_rejects_unsorted_samples() -> None:
    with pytest.raises(ValueError, match="sorted"):
        build_run_identity(
            run_id="run",
            model_id="resnet50",
            xai_method="grad_cam",
            scenario_ids=["rotation_mild"],
            sample_ids=["sample_b", "sample_a"],
            seed=42,
            governance_protocol_hash="governance",
            checkpoint_training_protocol_hash="training",
            checkpoint_sha256="checkpoint",
            manifest_sha256="manifest",
            freeze_record_sha256="freeze",
            checkpoint_decision_record_sha256="checkpoint_dr",
            test_decision_record_sha256="test_dr",
            g2_readiness_report_sha256="readiness",
            campaign_id="campaign",
            authorization_decision_id="decision",
            transformation_algorithm_version="v6",
            xai_policy={},
            software_version="0.1.0",
            git_commit="abc123",
            runtime_identity={"torch_version": "2.11.0", "gpu": "A100"},
        )
