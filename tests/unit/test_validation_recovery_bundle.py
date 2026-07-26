from __future__ import annotations

import csv
import json
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

from plantxai_stability.data.manifest import canonical_rgb_sha256
from scripts.build_validation_recovery_bundle import build_bundle


def test_validation_bundle_extracts_only_verified_manifest_paths(tmp_path: Path) -> None:
    image = np.full((4, 5, 3), 127, dtype=np.uint8)
    source = tmp_path / "source.png"
    Image.fromarray(image).save(source)
    digest, width, height = canonical_rgb_sha256(source)
    relative = "images/train/Tomato___healthy/example.png"
    manifest = tmp_path / "validation_split.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id", "leaf_id", "class_id", "class_name", "source_split", "split",
                "canonical_relative_path", "canonical_rgb_sha256", "width", "height", "source_row_index",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "sample_id": "sample", "leaf_id": "leaf", "class_id": 0,
            "class_name": "Tomato___healthy", "source_split": "train", "split": "validation",
            "canonical_relative_path": relative, "canonical_rgb_sha256": digest,
            "width": width, "height": height, "source_row_index": 0,
        })
    archive = tmp_path / "source.tar"
    with tarfile.open(archive, "w") as handle:
        handle.add(source, arcname="archive-root/" + relative)
        handle.add(source, arcname="archive-root/images/test/must_not_materialize.png")
    freeze = tmp_path / "freeze_record.json"
    freeze.write_text(json.dumps({"freeze": "fixture"}), encoding="utf-8")

    output = tmp_path / "validation-only"
    report = build_bundle(
        archive=archive, validation_manifest=manifest, freeze_record=freeze, output_dir=output
    )

    assert report["validation_image_count"] == 1
    assert (output / relative).is_file()
    assert not (output / "images/test/must_not_materialize.png").exists()
