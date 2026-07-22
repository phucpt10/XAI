"""Creation of immutable manifest, split and freeze evidence artifacts."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from plantxai_stability.contracts import SampleRecord
from plantxai_stability.data.splits import validate_frozen_splits
from plantxai_stability.provenance import sha256_file


def require_frozen_artifacts(manifest_path: str | Path) -> dict[str, Any]:
    """Require a manifest sibling directory to contain a reviewed freeze record."""
    root = Path(manifest_path).parent
    freeze_path = root / "freeze_record.json"
    leakage_path = root / "split_leakage_report.json"
    if not freeze_path.is_file() or not leakage_path.is_file():
        raise ValueError("Manifest is not accompanied by immutable freeze artifacts")
    record = json.loads(freeze_path.read_text(encoding="utf-8"))
    leakage = json.loads(leakage_path.read_text(encoding="utf-8"))
    if not leakage.get("passed", False):
        raise ValueError("Frozen split leakage report does not pass")
    expected_hash = record.get("artifact_sha256", {}).get(Path(manifest_path).name)
    if expected_hash is None:
        raise ValueError("Manifest path is not one of the frozen split artifacts")
    if expected_hash != sha256_file(manifest_path):
        raise ValueError("Frozen split artifact hash mismatch")
    return record


def _refuse_overwrite(paths: Iterable[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Frozen dataset artifact already exists; create a new version: " + ", ".join(existing)
        )


def _write_csv(records: list[SampleRecord], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(SampleRecord.__dataclass_fields__)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def _write_parquet(records: list[SampleRecord], path: Path) -> None:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pandas is required to write dataset_manifest.parquet") from exc
    try:
        pd.DataFrame([asdict(record) for record in records]).to_parquet(path, index=False)
    except (ImportError, ValueError) as exc:  # pragma: no cover
        raise RuntimeError("Writing dataset_manifest.parquet requires pyarrow") from exc


def _leakage_report(records: list[SampleRecord]) -> dict[str, Any]:
    by_split: dict[str, set[str]] = defaultdict(set)
    by_leaf: dict[str, set[str]] = defaultdict(set)
    for record in records:
        by_split[record.split].add(record.sample_id)
        by_leaf[record.split].add(record.leaf_id)
    split_names = sorted(by_split)
    sample_overlap: dict[str, list[str]] = {}
    leaf_overlap: dict[str, list[str]] = {}
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            sample_overlap[f"{left}∩{right}"] = sorted(by_split[left] & by_split[right])
            leaf_overlap[f"{left}∩{right}"] = sorted(by_leaf[left] & by_leaf[right])
    return {
        "sample_overlap": sample_overlap,
        "leaf_overlap": leaf_overlap,
        "passed": all(not values for values in sample_overlap.values())
        and all(not values for values in leaf_overlap.values()),
    }


def write_frozen_dataset_artifacts(
    records: Iterable[SampleRecord],
    output_dir: str | Path,
    *,
    protocol_hash: str,
    audit_identity: str,
    class_selection_decision_record: str,
    split_policy: str,
    seed: int,
) -> dict[str, str]:
    """Write all freeze artifacts once and return their SHA-256 hashes."""
    materialized = sorted(list(records), key=lambda item: item.sample_id)
    if not materialized:
        raise ValueError("Cannot freeze an empty manifest")
    validate_frozen_splits(materialized)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    expected = [
        root / "dataset_manifest.parquet",
        root / "dataset_manifest.csv",
        root / "train_split.csv",
        root / "validation_split.csv",
        root / "test_split.csv",
        root / "split_summary.json",
        root / "split_leakage_report.json",
        root / "freeze_record.json",
    ]
    _refuse_overwrite(expected)
    _write_parquet(materialized, expected[0])
    _write_csv(materialized, expected[1])
    _write_csv([item for item in materialized if item.split == "train"], expected[2])
    _write_csv([item for item in materialized if item.split == "validation"], expected[3])
    _write_csv([item for item in materialized if item.split == "test"], expected[4])
    summary = {
        "sample_count": len(materialized),
        "counts_by_split": {
            split: sum(item.split == split for item in materialized)
            for split in ("train", "validation", "test")
        },
        "counts_by_class": {
            name: sum(item.class_name == name for item in materialized)
            for name in sorted({item.class_name for item in materialized})
        },
        "leaf_count": len({item.leaf_id for item in materialized}),
    }
    expected[5].write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    leakage = _leakage_report(materialized)
    expected[6].write_text(json.dumps(leakage, indent=2, sort_keys=True), encoding="utf-8")
    if not leakage["passed"]:
        raise ValueError("Split leakage detected; freeze artifacts are invalid")
    hashes = {path.name: sha256_file(path) for path in expected[:7]}
    freeze_record = {
        "protocol_hash": protocol_hash,
        "audit_identity": audit_identity,
        "class_selection_decision_record": class_selection_decision_record,
        "split_policy": split_policy,
        "seed": seed,
        "manifest_sha256": hashes["dataset_manifest.parquet"],
        "artifact_sha256": hashes,
        "creation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    expected[7].write_text(json.dumps(freeze_record, indent=2, sort_keys=True), encoding="utf-8")
    return {path.name: sha256_file(path) for path in expected}
