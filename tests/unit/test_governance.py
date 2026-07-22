from pathlib import Path

import pytest
import yaml

from plantxai_stability.config import load_protocol
from plantxai_stability.governance import approved_checkpoint_lineage


def _decision() -> dict[str, object]:
    return {
        "decision_id": "DR-CHECKPOINT-001",
        "status": "approved",
        "selection_policy": {
            "declared_models": ["resnet50", "efficientnet_b0"],
            "official_test_used_for_selection": False,
        },
        "training_lineage": {
            "protocol_hash": "a" * 64,
            "manifest_sha256": "manifest",
            "freeze_record_sha256": "freeze",
        },
        "validation_audit": {
            "source_split": "validation",
            "test_split_accessed": False,
        },
        "approved_checkpoints": {
            "resnet50": {"checkpoint_sha256": "checkpoint"},
            "efficientnet_b0": {"checkpoint_sha256": "other"},
        },
    }


def _governance() -> dict[str, object]:
    return {
        "G1_CHECKPOINT_SELECTION": "pass",
        "evidence_records": {"checkpoint_selection": "DR-CHECKPOINT-001"},
    }


def test_approved_checkpoint_lineage_returns_frozen_training_hash() -> None:
    result = approved_checkpoint_lineage(
        _decision(),
        _governance(),
        model_id="resnet50",
        declared_models=["resnet50", "efficientnet_b0"],
        checkpoint_sha256="checkpoint",
        manifest_sha256="manifest",
        freeze_record_sha256="freeze",
    )
    assert result["training_protocol_hash"] == "a" * 64
    assert result["decision_id"] == "DR-CHECKPOINT-001"


def test_approved_checkpoint_lineage_rejects_test_access_and_hash_change() -> None:
    decision = _decision()
    decision["validation_audit"]["test_split_accessed"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="checkpoint_sha256.*validation_test_split_accessed"):
        approved_checkpoint_lineage(
            decision,
            _governance(),
            model_id="resnet50",
            declared_models=["resnet50", "efficientnet_b0"],
            checkpoint_sha256="changed",
            manifest_sha256="manifest",
            freeze_record_sha256="freeze",
        )


def test_repository_g1_decision_binds_both_approved_checkpoints() -> None:
    protocol = load_protocol("configs/protocol/v0.9/protocol.yaml")
    decision = yaml.safe_load(
        Path(
            "configs/protocol/v0.9/decision_records/DR-CHECKPOINT-001.yaml"
        ).read_text(encoding="utf-8")
    )
    assert decision["governance_effect"]["governance_protocol_hash"] == protocol.sha256
    expected_hashes = {
        "resnet50": "b508abd2851c5f576131db0e47447624cd78f1e3204c2931f7928c266f0c7bfc",
        "efficientnet_b0": "05b592f1ff7f4f2b4a757ae2564a088e3742555e20110ee33d19e563ff2fe60b",
    }
    for model_id, checkpoint_hash in expected_hashes.items():
        result = approved_checkpoint_lineage(
            decision,
            protocol.values["governance"],
            model_id=model_id,
            declared_models=protocol.values["models"],
            checkpoint_sha256=checkpoint_hash,
            manifest_sha256=(
                "323b48e3564708d566e0e9f5c346a07ef728828b2879fc1975e21ca32e024894"
            ),
            freeze_record_sha256=(
                "aed2e96afd2749250d4151780bb4002d198eb96d7433d2bef5b03d4a6ac9212d"
            ),
        )
        assert result["training_protocol_hash"] == (
            "7eb0814be8ffc1a19f54e2bec2d2ca0c84d7f4d869d99e28b69e6c9e0e84523b"
        )
