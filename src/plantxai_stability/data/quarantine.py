"""Governed quarantine adjudication and modeling-universe construction."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from plantxai_stability.contracts import SampleRecord
from plantxai_stability.provenance import sha256_file


TRAIN_OVERLAP_REASON = "TRAIN_SAMPLE_QUARANTINED_TEST_LEAF_OVERLAP"
TRAIN_DUPLICATE_REASON = "TRAIN_EXACT_PIXEL_DUPLICATE_REDUNDANT"


def _source_key(row: Mapping[str, Any]) -> tuple[str, int, str, str]:
    index = row.get("source_row_index")
    if index in (None, ""):
        raise ValueError("Quarantine identity requires source_row_index")
    return (
        str(row.get("source_split", "")),
        int(index),
        str(row.get("resolved_leaf_id") or row.get("leaf_id") or ""),
        str(row.get("class_name", "")),
    )


def adjudicate_train_test_leaf_overlap(
    report_rows: Iterable[Mapping[str, Any]],
    *,
    approved_overlap_ids: Iterable[str],
    decision_record_id: str,
    expected_quarantined_train_count: int,
    expected_official_test_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Preserve test rows and select every train row in an approved overlap leaf."""
    rows = [dict(row) for row in report_rows]
    approved = set(approved_overlap_ids)
    detected = {
        str(row.get("resolved_leaf_id", ""))
        for row in rows
        if "LEAF_SPLIT_OVERLAP" in str(row.get("reason_code", ""))
    }
    if detected != approved:
        raise ValueError(
            "Decision Record overlap IDs do not exactly match the audit: "
            f"approved={sorted(approved)}, detected={sorted(detected)}"
        )

    by_leaf: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        leaf_id = str(row.get("resolved_leaf_id", ""))
        if leaf_id in approved:
            by_leaf[leaf_id].append(row)
    candidates: list[dict[str, Any]] = []
    registry: list[dict[str, Any]] = []
    for leaf_id in sorted(approved):
        leaf_rows = by_leaf[leaf_id]
        splits = {str(row.get("source_split", "")) for row in leaf_rows}
        classes = {str(row.get("class_name", "")) for row in leaf_rows}
        if splits != {"train", "test"}:
            raise ValueError(f"Approved overlap leaf does not occur in train and test: {leaf_id}")
        if len(classes) != 1:
            raise ValueError(f"Approved overlap leaf maps to multiple classes: {leaf_id}")
        for row in leaf_rows:
            source_split = str(row.get("source_split", ""))
            action = "quarantine" if source_split == "train" else "preserve_official_test"
            candidate = {
                "source_split": source_split,
                "source_row_index": int(row["source_row_index"]),
                "image_path": str(row.get("image_path", "")),
                "class_name": str(row.get("class_name", "")),
                "resolved_leaf_id": leaf_id,
                "leaf_id_source": str(row.get("leaf_id_source", "")),
                "audit_reason_code": str(row.get("reason_code", "")),
                "adjudication_action": action,
                "decision_record_id": decision_record_id,
            }
            candidates.append(candidate)
            if action == "quarantine":
                registry.append(
                    {
                        **candidate,
                        "eligibility_status": "quarantined",
                        "quarantine_reason_code": TRAIN_OVERLAP_REASON,
                        "violation_scope": "leaf_group_cross_split",
                    }
                )

    registry_keys = [_source_key(row) for row in registry]
    if len(registry_keys) != len(set(registry_keys)):
        raise ValueError("Quarantine registry contains duplicate source identities")
    official_test_count = sum(str(row.get("source_split", "")) == "test" for row in rows)
    criteria = {
        "detected_overlap_ids_match_decision_record": detected == approved,
        "quarantines_train_only": all(row["source_split"] == "train" for row in registry),
        "quarantined_train_count_matches": len(registry) == expected_quarantined_train_count,
        "official_test_count_matches": official_test_count == expected_official_test_count,
        "each_overlap_has_train_and_test": all(
            {str(row.get("source_split", "")) for row in by_leaf[leaf_id]}
            == {"train", "test"}
            for leaf_id in approved
        ),
    }
    summary: dict[str, Any] = {
        "decision_record_id": decision_record_id,
        "policy": "preserve_official_test_and_quarantine_overlapping_train_samples",
        "audited_sample_count": len(rows),
        "overlap_leaf_count": len(approved),
        "overlap_sample_count": len(candidates),
        "quarantined_train_count": len(registry),
        "official_test_count": official_test_count,
        "approved_overlap_ids": sorted(approved),
        "acceptance_criteria": criteria,
        "passed": all(criteria.values()),
    }
    return candidates, registry, summary


def apply_quarantine_registry(
    records: Iterable[SampleRecord],
    registry_rows: Iterable[Mapping[str, Any]],
    *,
    decision_record_id: str,
    expected_audited_count: int,
    expected_quarantined_count: int,
    expected_official_test_count: int,
    expected_eligible_count: int,
    expected_eligible_source_train_count: int,
) -> tuple[list[SampleRecord], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Join approved source identities to pixel-audited records and exclude them."""
    materialized = sorted(list(records), key=lambda item: item.sample_id)
    registry = [dict(row) for row in registry_rows]
    approved_keys = {_source_key(row) for row in registry}
    if len(approved_keys) != len(registry):
        raise ValueError("Quarantine registry contains duplicate source identities")
    if any(key[0] != "train" for key in approved_keys):
        raise ValueError("Official test samples cannot be quarantined by this policy")

    eligible: list[SampleRecord] = []
    finalized_registry: list[dict[str, Any]] = []
    lineage: list[dict[str, Any]] = []
    matched: set[tuple[str, int, str, str]] = set()
    for record in materialized:
        record_row = {
            "source_split": record.source_split,
            "source_row_index": record.source_row_index,
            "leaf_id": record.leaf_id,
            "class_name": record.class_name,
        }
        key = _source_key(record_row)
        quarantined = key in approved_keys
        status = "quarantined" if quarantined else "eligible"
        lineage.append(
            {
                **asdict(record),
                "eligibility_status": status,
                "quarantine_reason_code": TRAIN_OVERLAP_REASON if quarantined else None,
                "decision_record_id": decision_record_id if quarantined else None,
            }
        )
        if quarantined:
            matched.add(key)
            finalized_registry.append(
                {
                    **asdict(record),
                    "eligibility_status": "quarantined",
                    "quarantine_reason_code": TRAIN_OVERLAP_REASON,
                    "violation_scope": "leaf_group_cross_split",
                    "decision_record_id": decision_record_id,
                }
            )
        else:
            eligible.append(record)
    unmatched = sorted(approved_keys - matched)
    if unmatched:
        raise ValueError(f"Quarantine source identities were not found in the manifest: {unmatched}")

    full_test_ids = {item.sample_id for item in materialized if item.source_split == "test"}
    eligible_test_ids = {item.sample_id for item in eligible if item.source_split == "test"}
    leaf_splits: dict[str, set[str]] = defaultdict(set)
    for record in eligible:
        leaf_splits[record.leaf_id].add(record.source_split)
    leakage = sorted(leaf_id for leaf_id, splits in leaf_splits.items() if len(splits) > 1)
    criteria = {
        "audited_count_matches": len(materialized) == expected_audited_count,
        "quarantined_count_matches": len(finalized_registry) == expected_quarantined_count,
        "sample_reconciliation": len(materialized)
        == len(eligible) + len(finalized_registry),
        "official_test_preserved_exactly": full_test_ids == eligible_test_ids,
        "official_test_count_matches": len(eligible_test_ids) == expected_official_test_count,
        "eligible_count_matches": len(eligible) == expected_eligible_count,
        "eligible_source_train_count_matches": sum(
            item.source_split == "train" for item in eligible
        )
        == expected_eligible_source_train_count,
        "eligible_leaf_split_overlap_zero": not leakage,
    }
    summary: dict[str, Any] = {
        "decision_record_id": decision_record_id,
        "audited_sample_count": len(materialized),
        "eligible_sample_count": len(eligible),
        "quarantined_sample_count": len(finalized_registry),
        "official_test_count": len(eligible_test_ids),
        "eligible_source_train_count": sum(item.source_split == "train" for item in eligible),
        "eligible_leaf_split_overlap_count": len(leakage),
        "eligible_leaf_split_overlap_ids": leakage,
        "counts_by_eligibility_status": {
            "eligible": len(eligible),
            "quarantined": len(finalized_registry),
        },
        "eligible_counts_by_class": dict(
            sorted(Counter(item.class_name for item in eligible).items())
        ),
        "quarantined_counts_by_class": dict(
            sorted(Counter(str(item["class_name"]) for item in finalized_registry).items())
        ),
        "acceptance_criteria": criteria,
        "passed": all(criteria.values()),
    }
    return eligible, finalized_registry, lineage, summary


def adjudicate_redundant_train_duplicates(
    records: Iterable[SampleRecord],
    *,
    approved_quarantined_sample_ids: Iterable[str],
    decision_record_id: str,
    expected_group_count: int,
) -> tuple[list[dict[str, Any]], list[SampleRecord], dict[str, Any]]:
    """Keep one deterministic representative from benign train duplicate pairs."""
    materialized = list(records)
    by_hash: dict[str, list[SampleRecord]] = defaultdict(list)
    for record in materialized:
        by_hash[record.canonical_rgb_sha256].append(record)
    duplicate_groups = {
        digest: sorted(items, key=lambda item: item.sample_id)
        for digest, items in by_hash.items()
        if len(items) > 1
    }
    candidates: list[dict[str, Any]] = []
    selected: list[SampleRecord] = []
    group_checks: dict[str, bool] = {}
    for digest, items in sorted(duplicate_groups.items()):
        valid_group = (
            len(items) == 2
            and {item.source_split for item in items} == {"train"}
            and len({item.class_name for item in items}) == 1
            and len({item.leaf_id for item in items}) == 1
        )
        group_checks[digest] = valid_group
        if not valid_group:
            raise ValueError(
                "Automatic redundant-copy quarantine requires a two-sample, "
                f"train-only, same-class, same-leaf group: {digest}"
            )
        retained = items[0]
        redundant = items[1]
        selected.append(redundant)
        for record in items:
            candidates.append(
                {
                    **asdict(record),
                    "duplicate_group_sha256": digest,
                    "adjudication_action": (
                        "retain_deterministic_representative"
                        if record.sample_id == retained.sample_id
                        else "quarantine_redundant_copy"
                    ),
                    "representative_sample_id": retained.sample_id,
                    "decision_record_id": decision_record_id,
                }
            )
    approved_ids = set(approved_quarantined_sample_ids)
    detected_ids = {record.sample_id for record in selected}
    criteria = {
        "duplicate_group_count_matches": len(duplicate_groups) == expected_group_count,
        "all_groups_are_benign_train_pairs": all(group_checks.values()),
        "deterministic_selection_matches_decision_record": detected_ids == approved_ids,
        "official_test_untouched": all(record.source_split == "train" for record in selected),
    }
    summary: dict[str, Any] = {
        "decision_record_id": decision_record_id,
        "policy": "retain_minimum_sample_id_and_quarantine_redundant_train_copy",
        "input_eligible_sample_count": len(materialized),
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_sample_count": sum(len(items) for items in duplicate_groups.values()),
        "newly_quarantined_sample_count": len(selected),
        "quarantined_sample_ids": sorted(detected_ids),
        "acceptance_criteria": criteria,
        "passed": all(criteria.values()),
    }
    return candidates, selected, summary


def _write_parquet(rows: Iterable[Mapping[str, Any]], path: Path) -> None:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pandas and pyarrow are required for quarantine evidence") from exc
    pd.DataFrame(list(rows)).to_parquet(path, index=False)


def read_parquet_rows(path: str | Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pandas and pyarrow are required for quarantine evidence") from exc
    return pd.read_parquet(path).to_dict(orient="records")


def write_quarantine_adjudication_artifacts(
    candidates: Iterable[Mapping[str, Any]],
    registry: Iterable[Mapping[str, Any]],
    summary: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    """Write immutable pre-manifest quarantine decision artifacts."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    candidate_path = root / "quarantine_candidates.parquet"
    registry_path = root / "quarantine_decision_registry.parquet"
    summary_path = root / "quarantine_adjudication_summary.json"
    paths = (candidate_path, registry_path, summary_path)
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Quarantine adjudication artifacts already exist; create a new version: "
            + ", ".join(existing)
        )
    _write_parquet(candidates, candidate_path)
    _write_parquet(registry, registry_path)
    resolved_summary = {
        **summary,
        "quarantine_candidates_sha256": sha256_file(candidate_path),
        "quarantine_decision_registry_sha256": sha256_file(registry_path),
    }
    summary_path.write_text(
        json.dumps(resolved_summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {path.name: sha256_file(path) for path in paths}


def write_duplicate_adjudication_artifact(
    rows: Iterable[Mapping[str, Any]], path: str | Path
) -> str:
    """Write the immutable two-row-per-group duplicate decision table."""
    output = Path(path)
    if output.exists():
        raise FileExistsError(
            f"Duplicate adjudication artifact already exists; create a new version: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_parquet(rows, output)
    return sha256_file(output)


def write_quarantine_manifest_artifacts(
    lineage: Iterable[Mapping[str, Any]],
    finalized_registry: Iterable[Mapping[str, Any]],
    summary: Mapping[str, Any],
    output_dir: str | Path,
    *,
    eligible_manifest_path: str | Path,
) -> dict[str, str]:
    """Write all-sample lineage and the finalized pixel-identified registry."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    lineage_path = root / "dataset_lineage_manifest.parquet"
    registry_path = root / "quarantine_registry.parquet"
    summary_path = root / "quarantine_summary.json"
    paths = (lineage_path, registry_path, summary_path)
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Quarantine manifest artifacts already exist; create a new version: "
            + ", ".join(existing)
        )
    _write_parquet(lineage, lineage_path)
    _write_parquet(finalized_registry, registry_path)
    resolved_summary = {
        **summary,
        "eligible_manifest_sha256": sha256_file(eligible_manifest_path),
        "dataset_lineage_manifest_sha256": sha256_file(lineage_path),
        "quarantine_registry_sha256": sha256_file(registry_path),
    }
    summary_path.write_text(
        json.dumps(resolved_summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {path.name: sha256_file(path) for path in paths}
