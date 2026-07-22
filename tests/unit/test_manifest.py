import pytest

from plantxai_stability.data.manifest import build_manifest, make_sample_id
from plantxai_stability.data.splits import validate_frozen_splits


def _row(leaf: str = "leaf-1", path: str = "a.png") -> dict[str, object]:
    return {"leaf_id": leaf, "class_id": 0, "class_name": "Tomato___healthy", "source_split": "train", "split": "train", "canonical_relative_path": path, "canonical_rgb_sha256": "a" * 64, "width": 8, "height": 8}


def test_sample_id_is_stable() -> None:
    assert make_sample_id("a.png", "a" * 64) == make_sample_id("a.png", "a" * 64)
    assert build_manifest([_row()])[0].sample_id.startswith("pv_")


def test_leaf_leakage_is_rejected() -> None:
    records = build_manifest([_row(path="a.png"), _row(path="b.png")])
    leaked = records[1].__class__(**{**records[1].__dict__, "split": "test"})
    with pytest.raises(ValueError, match="leaf_id appears"):
        validate_frozen_splits([records[0], leaked])
