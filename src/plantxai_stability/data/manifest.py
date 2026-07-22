"""Dataset inspection and stable manifest construction."""

from __future__ import annotations

import csv
import hashlib
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

from plantxai_stability.contracts import SampleRecord


def canonical_rgb_sha256(path: str | Path) -> tuple[str, int, int]:
    """Hash canonical RGB pixels, not compressed file bytes."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Pillow is required for image manifest construction") from exc
    with Image.open(path) as image:
        return canonical_rgb_sha256_image(image)


def canonical_rgb_sha256_image(image: Any) -> tuple[str, int, int]:
    """Hash an in-memory image (including Hugging Face ``Image`` values)."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Pillow is required for image manifest construction") from exc
    if isinstance(image, dict):
        if image.get("bytes") is not None:
            image = Image.open(BytesIO(image["bytes"]))
        elif image.get("path") is not None:
            image = Image.open(image["path"])
        else:
            raise ValueError("Image mapping must contain either 'bytes' or 'path'")
    if not hasattr(image, "convert"):
        raise TypeError(f"Unsupported image value: {type(image)!r}")
    rgb = image.convert("RGB")
    payload = rgb.tobytes()
    return hashlib.sha256(payload).hexdigest(), rgb.width, rgb.height


def make_sample_id(canonical_relative_path: str, canonical_rgb_sha256: str) -> str:
    payload = f"{canonical_relative_path}\0{canonical_rgb_sha256}".encode("utf-8")
    return "pv_" + hashlib.sha256(payload).hexdigest()[:16]


def build_manifest(rows: Iterable[dict[str, Any]]) -> list[SampleRecord]:
    """Build deterministic records and reject duplicate stable IDs."""
    records: list[SampleRecord] = []
    seen: set[str] = set()
    for row in rows:
        required = {"leaf_id", "class_id", "class_name", "source_split", "split", "canonical_relative_path", "canonical_rgb_sha256", "width", "height"}
        missing = required.difference(row)
        if missing:
            raise ValueError(f"Manifest row missing fields: {sorted(missing)}")
        sample_id = make_sample_id(str(row["canonical_relative_path"]), str(row["canonical_rgb_sha256"]))
        if sample_id in seen:
            raise ValueError(f"Duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        records.append(SampleRecord(
            sample_id=sample_id,
            leaf_id=str(row["leaf_id"]),
            class_id=int(row["class_id"]),
            class_name=str(row["class_name"]),
            source_split=str(row["source_split"]),
            split=str(row["split"]),
            canonical_relative_path=str(row["canonical_relative_path"]),
            canonical_rgb_sha256=str(row["canonical_rgb_sha256"]),
            width=int(row["width"]),
            height=int(row["height"]),
            source_row_index=(int(row["source_row_index"]) if row.get("source_row_index") not in (None, "") else None),
        ))
    return sorted(records, key=lambda record: record.sample_id)


def write_manifest(records: Iterable[SampleRecord], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(SampleRecord.__dataclass_fields__)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({name: getattr(record, name) for name in fieldnames})


def read_manifest_csv(path: str | Path) -> list[SampleRecord]:
    output: list[SampleRecord] = []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            output.append(SampleRecord(
                sample_id=str(row["sample_id"]),
                leaf_id=str(row["leaf_id"]),
                class_id=int(row["class_id"]),
                class_name=str(row["class_name"]),
                source_split=str(row["source_split"]),
                split=str(row["split"]),
                canonical_relative_path=str(row["canonical_relative_path"]),
                canonical_rgb_sha256=str(row["canonical_rgb_sha256"]),
                width=int(row["width"]),
                height=int(row["height"]),
                source_row_index=(int(row["source_row_index"]) if row.get("source_row_index") not in (None, "") else None),
            ))
    if not output:
        raise ValueError(f"Manifest is empty: {path}")
    return output


def inspect_manifest(records: Iterable[SampleRecord]) -> dict[str, object]:
    materialized = list(records)
    by_split: dict[str, int] = {}
    leaf_splits: dict[str, set[str]] = {}
    for record in materialized:
        by_split[record.split] = by_split.get(record.split, 0) + 1
        leaf_splits.setdefault(record.leaf_id, set()).add(record.split)
    leakage = sorted(leaf for leaf, splits in leaf_splits.items() if len(splits) > 1)
    return {
        "sample_count": len(materialized),
        "leaf_count": len(leaf_splits),
        "counts_by_split": by_split,
        "leaf_split_leakage": leakage,
        "passed": not leakage,
    }
