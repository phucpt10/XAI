from pathlib import Path

import pytest
import yaml

from plantxai_stability.config import load_protocol


def test_protocol_is_frozen_and_g2_approved_with_runtime_gate_required() -> None:
    path = Path("configs/protocol/v0.9/protocol.yaml")
    config = load_protocol(path)
    assert config.values["protocol_version"] == "v0.9"
    assert config.values["status"] == "frozen"
    assert config.values["frozen"] is True
    assert config.values["governance"]["G0B_PROTOCOL_FREEZE_READY"] == "pass"
    assert config.values["governance"]["blockers"] == []
    assert config.values["governance"]["G1_CHECKPOINT_SELECTION"] == "pass"
    assert config.values["governance"]["checkpoint_blockers"] == []
    assert config.values["governance"]["G2_TEST_EVALUATION_READY"] == "pass"
    assert config.values["governance"]["official_training_allowed"] is True
    assert config.values["governance"]["official_test_evaluation_allowed"] is True
    assert config.values["governance"]["official_experiment_allowed"] is True
    assert (
        config.values["governance"]["evidence_records"]["transformation_severity"]
        == "DR-SEVERITY-006"
    )
    assert (
        config.values["governance"]["evidence_records"]["checkpoint_selection"]
        == "DR-CHECKPOINT-001"
    )
    assert (
        config.values["governance"]["evidence_records"]["test_evaluation"]
        == "DR-TEST-001"
    )
    assert config.values["xai"]["target_layers"] == {
        "resnet50": "layer4[-1]",
        "efficientnet_b0": "features[-1]",
    }
    assert config.values["xai"]["alignment_policy"] == "forward_align_original_cam"
    assert config.values["xai"]["valid_region_policy"] == "geometric_support_mask"
    assert config.values["xai"]["topk_iou_sensitivity"] == [0.1, 0.2, 0.3]
    assert (
        config.values["xai"]["rotation_prediction_claim_scope"]
        == "zero_filled_operator_specific"
    )
    assert len(config.sha256) == 64
    test_decision = yaml.safe_load(
        Path(
            "configs/protocol/v0.9/decision_records/DR-TEST-001.yaml"
        ).read_text(encoding="utf-8")
    )
    assert test_decision["governance_effect"]["governance_protocol_hash"] == (
        config.sha256
    )


def test_protocol_rejects_pending_xai_target_layer(tmp_path: Path) -> None:
    source = Path("configs/protocol/v0.9/protocol.yaml")
    values = yaml.safe_load(source.read_text(encoding="utf-8"))
    values["xai"]["target_layers"]["resnet50"] = "PENDING_RUNTIME_VALIDATION"
    candidate = tmp_path / "protocol.yaml"
    candidate.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="runtime-approved XAI target layer"):
        load_protocol(candidate)


def test_frozen_protocol_requires_severity_evidence(tmp_path: Path) -> None:
    source = Path("configs/protocol/v0.9/protocol.yaml")
    values = yaml.safe_load(source.read_text(encoding="utf-8"))
    del values["governance"]["evidence_records"]["transformation_severity"]
    candidate = tmp_path / "protocol.yaml"
    candidate.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="transformation_severity is required"):
        load_protocol(candidate)


def test_passing_g0b_rejects_retained_blockers(tmp_path: Path) -> None:
    source = Path("configs/protocol/v0.9/protocol.yaml")
    values = yaml.safe_load(source.read_text(encoding="utf-8"))
    values["governance"]["blockers"] = ["stale_blocker"]
    candidate = tmp_path / "protocol.yaml"
    candidate.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="cannot retain G0B blockers"):
        load_protocol(candidate)


def test_passing_g1_requires_checkpoint_evidence(tmp_path: Path) -> None:
    source = Path("configs/protocol/v0.9/protocol.yaml")
    values = yaml.safe_load(source.read_text(encoding="utf-8"))
    del values["governance"]["evidence_records"]["checkpoint_selection"]
    candidate = tmp_path / "protocol.yaml"
    candidate.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="checkpoint_selection is required"):
        load_protocol(candidate)


def test_passing_g1_rejects_checkpoint_blockers(tmp_path: Path) -> None:
    source = Path("configs/protocol/v0.9/protocol.yaml")
    values = yaml.safe_load(source.read_text(encoding="utf-8"))
    values["governance"]["checkpoint_blockers"] = ["stale_checkpoint_blocker"]
    candidate = tmp_path / "protocol.yaml"
    candidate.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="cannot retain checkpoint blockers"):
        load_protocol(candidate)


def test_passing_g2_requires_test_authorization_evidence(tmp_path: Path) -> None:
    source = Path("configs/protocol/v0.9/protocol.yaml")
    values = yaml.safe_load(source.read_text(encoding="utf-8"))
    del values["governance"]["evidence_records"]["test_evaluation"]
    candidate = tmp_path / "protocol.yaml"
    candidate.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="test_evaluation is required"):
        load_protocol(candidate)


def test_passing_g2_rejects_test_evaluation_blockers(tmp_path: Path) -> None:
    source = Path("configs/protocol/v0.9/protocol.yaml")
    values = yaml.safe_load(source.read_text(encoding="utf-8"))
    values["governance"]["test_evaluation_blockers"] = ["stale_g2_blocker"]
    candidate = tmp_path / "protocol.yaml"
    candidate.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="cannot retain test-evaluation blockers"):
        load_protocol(candidate)
