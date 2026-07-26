"""Image-level integrity audit and immutable dataset evidence helpers."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from plantxai_stability.contracts import SampleRecord


def audit_manifest_records(records: Iterable[SampleRecord]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Audit duplicates and leaf constraints without silently dropping rows."""
    materialized = list(records)
    by_hash: dict[str, list[SampleRecord]] = defaultdict(list)
    by_leaf: dict[str, list[SampleRecord]] = defaultdict(list)
    for record in materialized:
        by_hash[record.canonical_rgb_sha256].append(record)
        by_leaf[record.leaf_id].append(record)

    rows: list[dict[str, Any]] = []
    invalid = 0
    for record in materialized:
        reasons: list[str] = []
        same_hash = by_hash[record.canonical_rgb_sha256]
        same_leaf = by_leaf[record.leaf_id]
        if len(same_hash) > 1:
            reasons.append("EXACT_PIXEL_DUPLICATE")
            if len({item.split for item in same_hash}) > 1:
                reasons.append("CROSS_SPLIT_DUPLICATE")
            if len({item.class_name for item in same_hash}) > 1:
                reasons.append("CONFLICTING_LABEL_DUPLICATE")
        if len({item.split for item in same_leaf}) > 1:
            reasons.append("LEAF_SPLIT_LEAKAGE")
        if len({item.class_name for item in same_leaf}) > 1:
            reasons.append("LEAF_CLASS_CONFLICT")
        valid = not reasons
        if not valid:
            invalid += 1
        rows.append(
            {
                "source_sample_identity": record.sample_id,
                "source_split": record.source_split,
                "source_row_index": record.source_row_index,
                "canonical_relative_path": record.canonical_relative_path,
                "label_id": record.class_id,
                "canonical_class_name": record.class_name,
                "leaf_id": record.leaf_id,
                "width": record.width,
                "height": record.height,
                "channels": 3,
                "canonical_rgb_sha256": record.canonical_rgb_sha256,
                "valid": valid,
                "reason_code": "OK" if valid else ";".join(reasons),
                "error_detail": None,
            }
        )
    summary = {
        "sample_count": len(materialized),
        "invalid_count": invalid,
        "exact_duplicate_hash_count": sum(1 for values in by_hash.values() if len(values) > 1),
        "cross_split_duplicate_count": sum(
            1 for values in by_hash.values() if len({item.split for item in values}) > 1
        ),
        "conflicting_label_duplicate_count": sum(
            1 for values in by_hash.values() if len({item.class_name for item in values}) > 1
        ),
        "leaf_split_leakage_count": sum(
            1 for values in by_leaf.values() if len({item.split for item in values}) > 1
        ),
        "leaf_class_conflict_count": sum(
            1 for values in by_leaf.values() if len({item.class_name for item in values}) > 1
        ),
        "passed": invalid == 0,
    }
    return rows, summary


def write_image_audit_parquet(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    """Write the required image audit artifact; fail clearly if Parquet support is absent."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("pandas is required to write image_audit.parquet") from exc
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        pd.DataFrame(list(rows)).to_parquet(output, index=False)
    except (ImportError, ValueError) as exc:  # pragma: no cover - optional engine
        raise RuntimeError(
            "Writing image_audit.parquet requires a Parquet engine; install pyarrow"
        ) from exc


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
