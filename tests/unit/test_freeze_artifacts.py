from pathlib import Path

import pytest

from plantxai_stability.data.freeze import require_frozen_artifacts, write_frozen_dataset_artifacts
from plantxai_stability.data.manifest import build_manifest


def _row(leaf: str, path: str, source_split: str = "train"):
    return {
        "leaf_id": leaf,
        "class_id": 0,
        "class_name": "Tomato___healthy",
        "source_split": source_split,
        "split": source_split,
        "canonical_relative_path": path,
        "canonical_rgb_sha256": (path[0] * 64),
        "width": 8,
        "height": 8,
    }


def test_freeze_artifacts_are_hashed_and_immutable(tmp_path: Path):
    records = build_manifest(
        [_row("leaf-a", "a.png"), _row("leaf-b", "b.png"), _row("leaf-test", "t.png", "test")]
    )
    hashes = write_frozen_dataset_artifacts(
        records,
        tmp_path,
        protocol_hash="protocol",
        audit_identity="audit",
        class_selection_decision_record="DR-CLASS-001.yaml",
        split_policy="test-preserved",
        seed=42,
    )
    assert "dataset_manifest.parquet" in hashes
    require_frozen_artifacts(tmp_path / "dataset_manifest.csv")
    with pytest.raises(FileExistsError):
        write_frozen_dataset_artifacts(
            records,
            tmp_path,
            protocol_hash="protocol",
            audit_identity="audit",
            class_selection_decision_record="DR-CLASS-001.yaml",
            split_policy="test-preserved",
            seed=42,
        )

