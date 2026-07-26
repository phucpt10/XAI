from pathlib import Path

import pytest
from PIL import Image

from plantxai_stability.data.loader import PlantDataset, load_verified_record
from plantxai_stability.data.manifest import build_manifest, canonical_rgb_sha256


def test_loader_rejects_changed_pixels(tmp_path: Path):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (8, 6), (10, 20, 30)).save(image_path)
    digest, width, height = canonical_rgb_sha256(image_path)
    record = build_manifest(
        [
            {
                "leaf_id": "leaf-a",
                "class_id": 0,
                "class_name": "Tomato___healthy",
                "source_split": "test",
                "split": "test",
                "canonical_relative_path": "image.png",
                "canonical_rgb_sha256": digest,
                "width": width,
                "height": height,
            }
        ]
    )[0]
    Image.new("RGB", (8, 6), (1, 2, 3)).save(image_path)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_verified_record(record, tmp_path)


def test_dataset_keeps_identity_metadata(tmp_path: Path):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (8, 6), (10, 20, 30)).save(image_path)
    digest, width, height = canonical_rgb_sha256(image_path)
    record = build_manifest(
        [
            {
                "leaf_id": "leaf-a",
                "class_id": 0,
                "class_name": "Tomato___healthy",
                "source_split": "test",
                "split": "test",
                "canonical_relative_path": "image.png",
                "canonical_rgb_sha256": digest,
                "width": width,
                "height": height,
            }
        ]
    )[0]
    sample = PlantDataset([record], tmp_path, expected_split="test")[0]
    assert sample["sample_id"] == record.sample_id
    assert sample["canonical_rgb_sha256"] == digest
    assert sample["image_pixel_tensor"].shape == (height, width, 3)

