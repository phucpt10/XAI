"""Leaf identity reconstruction audit for PlantVillage lineage."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from plantxai_stability.provenance import sha256_file


def audit_leaf_identity(
    dataset: Mapping[str, Iterable[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Resolve filename candidates and detect ambiguity, conflicts and leakage."""
    source_rows: list[dict[str, Any]] = []
    for split, rows in dataset.items():
        for fallback_index, row in enumerate(rows):
            mapped_leaf_id = str(row.get("mapped_leaf_id") or row.get("leaf_id") or "")
            reconstructed = str(row.get("reconstructed_leaf_id") or "")
            resolved = mapped_leaf_id or reconstructed
            source_rows.append(
                {
                    "source_split": str(split),
                    "source_row_index": int(row.get("_source_row_index", fallback_index)),
                    "image_path": str(row.get("image_path", "")),
                    "class_name": str(row.get("label", "")),
                    "source_leaf_identity": str(row.get("source_leaf_identity", "")),
                    "mapped_leaf_id": mapped_leaf_id,
                    "reconstructed_leaf_id": reconstructed,
                    "resolved_leaf_id": resolved,
                    "leaf_id_source": "leaf_map" if mapped_leaf_id else "filename_reconstructed",
                    "leaf_map_status": str(row.get("leaf_map_status", "unknown")),
                    "leaf_map_suggestions": json.dumps(
                        list(row.get("leaf_map_suggestions", ())), sort_keys=True
                    ),
                }
            )

    candidate_mapped_ids: dict[str, set[str]] = defaultdict(set)
    candidate_classes: dict[str, set[str]] = defaultdict(set)
    resolved_classes: dict[str, set[str]] = defaultdict(set)
    resolved_splits: dict[str, set[str]] = defaultdict(set)
    for row in source_rows:
        candidate = row["reconstructed_leaf_id"]
        resolved = row["resolved_leaf_id"]
        if candidate:
            candidate_classes[candidate].add(row["class_name"])
            if row["mapped_leaf_id"]:
                candidate_mapped_ids[candidate].add(row["mapped_leaf_id"])
        if resolved:
            resolved_classes[resolved].add(row["class_name"])
            resolved_splits[resolved].add(row["source_split"])

    collision_candidates = {
        candidate
        for candidate in candidate_classes
        if len(candidate_classes[candidate]) > 1 or len(candidate_mapped_ids[candidate]) > 1
    }
    class_conflict_leaves = {
        leaf_id for leaf_id, classes in resolved_classes.items() if len(classes) > 1
    }
    split_overlap_leaves = {
        leaf_id for leaf_id, splits in resolved_splits.items() if len(splits) > 1
    }

    report_rows: list[dict[str, Any]] = []
    for row in source_rows:
        candidate = row["reconstructed_leaf_id"]
        resolved = row["resolved_leaf_id"]
        references = sorted(candidate_mapped_ids.get(candidate, set()))
        reasons: list[str] = []
        if not resolved:
            reasons.append("MISSING_RECONSTRUCTED_IDENTITY")
        if row["leaf_map_status"] == "ambiguous_leaf_map_match":
            reasons.append("AMBIGUOUS_LEAF_MAP_MATCH")
        if candidate in collision_candidates:
            reasons.append("RECONSTRUCTED_IDENTITY_COLLISION")
        if resolved in class_conflict_leaves:
            reasons.append("LEAF_CLASS_CONFLICT")
        if resolved in split_overlap_leaves:
            reasons.append("LEAF_SPLIT_OVERLAP")
        comparison_status = "leaf_map_reference"
        if row["leaf_id_source"] == "filename_reconstructed":
            comparison_status = (
                "one_leaf_map_reference"
                if len(references) == 1
                else "multiple_leaf_map_references"
                if len(references) > 1
                else "no_leaf_map_reference"
            )
        report_rows.append(
            {
                **row,
                "candidate_leaf_map_references": json.dumps(references),
                "leaf_map_comparison_status": comparison_status,
                "candidate_class_count": len(candidate_classes.get(candidate, set())),
                "resolved_leaf_class_count": len(resolved_classes.get(resolved, set())),
                "resolved_leaf_split_count": len(resolved_splits.get(resolved, set())),
                "valid": not reasons,
                "reason_code": "OK" if not reasons else ";".join(reasons),
            }
        )

    total = len(report_rows)
    resolved_count = sum(bool(row["resolved_leaf_id"]) for row in report_rows)
    ambiguous_count = sum(
        row["leaf_map_status"] == "ambiguous_leaf_map_match" for row in report_rows
    )
    invalid_count = sum(not row["valid"] for row in report_rows)
    summary = {
        "sample_count": total,
        "mapped_leaf_id_count": sum(row["leaf_id_source"] == "leaf_map" for row in report_rows),
        "filename_reconstructed_count": sum(
            row["leaf_id_source"] == "filename_reconstructed" for row in report_rows
        ),
        "leaf_map_status_counts": dict(
            sorted(Counter(row["leaf_map_status"] for row in report_rows).items())
        ),
        "leaf_map_comparison_status_counts": dict(
            sorted(
                Counter(
                    row["leaf_map_comparison_status"] for row in report_rows
                ).items()
            )
        ),
        "resolved_identity_count": resolved_count,
        "coverage": (resolved_count / total) if total else 0.0,
        "ambiguous_sample_count": ambiguous_count,
        "collision_candidate_count": len(collision_candidates),
        "collision_candidate_ids": sorted(collision_candidates),
        "leaf_class_conflict_count": len(class_conflict_leaves),
        "leaf_class_conflict_ids": sorted(class_conflict_leaves),
        "leaf_split_overlap_count": len(split_overlap_leaves),
        "leaf_split_overlap_ids": sorted(split_overlap_leaves),
        "invalid_sample_count": invalid_count,
        "acceptance_criteria": {
            "coverage_equals_1": resolved_count == total and total > 0,
            "no_ambiguous_identity": ambiguous_count == 0,
            "no_reconstructed_collision": not collision_candidates,
            "no_leaf_class_conflict": not class_conflict_leaves,
            "no_leaf_split_overlap": not split_overlap_leaves,
        },
    }
    summary["passed"] = all(summary["acceptance_criteria"].values())
    return report_rows, summary


def write_leaf_identity_artifacts(
    rows: Iterable[dict[str, Any]],
    summary: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    """Write immutable Parquet and JSON evidence and return their hashes."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "leaf_identity_resolution_report.parquet"
    summary_path = root / "leaf_identity_resolution_summary.json"
    existing = [str(path) for path in (report_path, summary_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "Leaf identity evidence already exists; create a new version: " + ", ".join(existing)
        )
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pandas and pyarrow are required for leaf identity evidence") from exc
    pd.DataFrame(list(rows)).to_parquet(report_path, index=False)
    resolved_summary = {**summary, "report_sha256": sha256_file(report_path)}
    summary_path.write_text(
        json.dumps(resolved_summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {
        report_path.name: sha256_file(report_path),
        summary_path.name: sha256_file(summary_path),
    }
