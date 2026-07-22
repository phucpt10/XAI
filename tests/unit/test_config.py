from pathlib import Path

import pytest
import yaml

from plantxai_stability.config import load_protocol


def test_protocol_is_draft_and_fail_closed() -> None:
    path = Path("configs/protocol/v0.9/protocol.yaml")
    config = load_protocol(path)
    assert config.values["protocol_version"] == "v0.9"
    assert config.values["status"] == "draft"
    assert config.values["frozen"] is False
    assert config.values["governance"]["G0B_PROTOCOL_FREEZE_READY"] == "blocked"
    assert config.values["governance"]["blockers"] == [
        "transformation_severity_pilot_not_executed",
        "checkpoints_not_selected",
    ]
    assert config.values["xai"]["target_layers"] == {
        "resnet50": "layer4[-1]",
        "efficientnet_b0": "features[-1]",
    }
    assert len(config.sha256) == 64


def test_protocol_rejects_pending_xai_target_layer(tmp_path: Path) -> None:
    source = Path("configs/protocol/v0.9/protocol.yaml")
    values = yaml.safe_load(source.read_text(encoding="utf-8"))
    values["xai"]["target_layers"]["resnet50"] = "PENDING_RUNTIME_VALIDATION"
    candidate = tmp_path / "protocol.yaml"
    candidate.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="runtime-approved XAI target layer"):
        load_protocol(candidate)
