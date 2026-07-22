from pathlib import Path

from plantxai_stability.config import load_protocol


def test_protocol_is_draft_and_fail_closed() -> None:
    path = Path("configs/protocol/v0.9/protocol.yaml")
    config = load_protocol(path)
    assert config.values["protocol_version"] == "v0.9"
    assert config.values["status"] == "draft"
    assert config.values["frozen"] is False
    assert config.values["governance"]["G0B_PROTOCOL_FREEZE_READY"] == "blocked"
    assert len(config.sha256) == 64
