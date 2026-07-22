"""Render deterministic validation examples for human severity review."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from plantxai_stability.config import load_protocol
from plantxai_stability.data.loader import load_verified_record
from plantxai_stability.data.manifest import read_manifest_csv
from plantxai_stability.provenance import sha256_file
from plantxai_stability.transformations import TransformationPipeline, scenario_grid


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--pilot-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit("Visual review output exists; use a new versioned directory")
    resolved = load_protocol(args.protocol)
    summary = json.loads(args.pilot_summary.read_text(encoding="utf-8"))
    if not summary.get("technical_gate_passed", False):
        raise SystemExit("Visual review requires a passing severity pilot")
    if summary.get("test_split_accessed") is not False:
        raise SystemExit("Severity pilot does not prove validation-only access")
    pilot_root = args.pilot_summary.parent
    selection_path = pilot_root / "severity_pilot_selection.csv"
    records_path = pilot_root / "severity_pilot_records.parquet"
    expected = summary.get("artifact_sha256", {})
    for path in (selection_path, records_path):
        if not path.is_file() or sha256_file(path) != expected.get(path.name):
            raise SystemExit(f"Severity pilot artifact mismatch: {path}")
    with selection_path.open("r", newline="", encoding="utf-8") as handle:
        selected_ids = {row["sample_id"] for row in csv.DictReader(handle)}
    manifest_records = read_manifest_csv(args.manifest)
    by_id = {record.sample_id: record for record in manifest_records}
    if not selected_ids.issubset(by_id):
        raise SystemExit("Pilot selection contains IDs absent from the frozen manifest")
    selected = [by_id[sample_id] for sample_id in selected_ids]
    if any(record.split != "validation" for record in selected):
        raise SystemExit("Visual review selection is not validation-only")
    representatives = []
    for class_name in resolved.values["dataset"]["classes"]:
        candidates = [record for record in selected if record.class_name == class_name]
        if not candidates:
            raise SystemExit(f"No pilot representative for class {class_name}")
        representatives.append(
            min(candidates, key=lambda item: _stable_key(resolved.seed, item.sample_id))
        )
    pipeline = TransformationPipeline(
        resolved.seed, resolved.values["transformations"]["parameters"]
    )
    scenarios = scenario_grid(resolved.values["transformations"]["parameters"])
    by_transformation = {
        name: [item for item in scenarios if item.transformation == name]
        for name in resolved.values["transformations"]["names"]
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sheet_paths = []
    for transformation, transformation_scenarios in by_transformation.items():
        path = args.output_dir / f"severity_review_{transformation}.png"
        _render_sheet(
            path,
            representatives,
            transformation_scenarios,
            pipeline,
            args.image_root,
        )
        sheet_paths.append(path)
    report = {
        "run_type": "validation_only_severity_visual_review",
        "approval_status": "pending_project_owner_review",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_hash": resolved.sha256,
        "pilot_summary_sha256": sha256_file(args.pilot_summary),
        "pilot_selection_sha256": sha256_file(selection_path),
        "source_split": "validation",
        "test_split_accessed": False,
        "representative_policy": (
            "one deterministic, non-performance-selected pilot sample per class"
        ),
        "representative_sample_ids": [item.sample_id for item in representatives],
        "review_questions": [
            "Does each severity preserve recognizable leaf and disease evidence?",
            "Are mild, moderate and severe visually ordered within each transformation?",
            "Does rotation fill introduce an unacceptable artificial shortcut?",
            "Should severity labels be restricted to within-transformation comparisons?",
        ],
        "artifact_sha256": {path.name: sha256_file(path) for path in sheet_paths},
    }
    report_path = args.output_dir / "severity_visual_review_manifest.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Visual severity review rendered: {args.output_dir}")
    return 0


def _render_sheet(
    path: Path,
    representatives: list[Any],
    scenarios: list[Any],
    pipeline: TransformationPipeline,
    image_root: Path,
) -> None:
    tile = 224
    label_width = 190
    header_height = 54
    row_height = tile + 28
    columns = ["original", *[scenario.severity for scenario in scenarios]]
    canvas = Image.new(
        "RGB",
        (label_width + tile * len(columns), header_height + row_height * len(representatives)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for column_index, title in enumerate(columns):
        draw.text((label_width + column_index * tile + 8, 18), title, fill="black")
    for row_index, record in enumerate(representatives):
        top = header_height + row_index * row_height
        disease = record.class_name.split("___", maxsplit=1)[-1]
        draw.text((8, top + 8), disease[:27], fill="black")
        draw.text((8, top + 30), record.sample_id, fill="gray")
        original = load_verified_record(record, image_root)
        images = [original]
        for scenario in scenarios:
            transformed, _ = pipeline.apply(original, record.sample_id, scenario)
            images.append(transformed)
        for column_index, pixels in enumerate(images):
            image = Image.fromarray(
                np.uint8(np.clip(pixels, 0.0, 1.0) * 255.0), mode="RGB"
            ).resize((tile, tile), Image.Resampling.BILINEAR)
            canvas.paste(image, (label_width + column_index * tile, top))
    canvas.save(path, format="PNG")


def _stable_key(seed: int, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
