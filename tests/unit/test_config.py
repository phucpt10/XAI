from pathlib import Path

import pytest
import yaml

from plantxai_stability.config import load_protocol


def test_protocol_is_frozen_for_g0b_but_test_remains_closed() -> None:
    path = Path("configs/protocol/v0.9/protocol.yaml")
    config = load_protocol(path)
    assert config.values["protocol_version"] == "v0.9"
    assert config.values["status"] == "frozen"
    assert config.values["frozen"] is True
    assert config.values["governance"]["G0B_PROTOCOL_FREEZE_READY"] == "pass"
    assert config.values["governance"]["blockers"] == []
    assert config.values["governance"]["G1_CHECKPOINT_SELECTION"] == "blocked"
    assert config.values["governance"]["G2_TEST_EVALUATION_READY"] == "blocked"
    assert config.values["governance"]["official_training_allowed"] is True
    assert config.values["governance"]["official_test_evaluation_allowed"] is False
    assert (
        config.values["governance"]["evidence_records"]["transformation_severity"]
        == "DR-SEVERITY-006"
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
