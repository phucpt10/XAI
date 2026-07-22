"""Apply an approved deterministic quarantine to benign train duplicate pairs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from plantxai_stability.data.audit import audit_manifest_records, write_image_audit_parquet
from plantxai_stability.data.manifest import read_manifest_csv, write_manifest
from plantxai_stability.data.quarantine import (
    TRAIN_DUPLICATE_REASON,
    adjudicate_redundant_train_duplicates,
    read_parquet_rows,
    write_duplicate_adjudication_artifact,
    write_quarantine_manifest_artifacts,
)
from plantxai_stability.provenance import sha256_file


def _validate_prior_lineage(
    decision: dict[str, Any],
    manifest_path: Path,
    lineage_path: Path,
    registry_path: Path,
    summary_path: Path,
    prior_summary: dict[str, Any],
) -> None:
    evidence = decision.get("evidence", {})
    actual = {
        "prior_quarantine_summary_sha256": sha256_file(summary_path),
        "prior_dataset_lineage_manifest_sha256": sha256_file(lineage_path),
        "prior_quarantine_registry_sha256": sha256_file(registry_path),
    }
    mismatches = {
        key: {"expected": evidence.get(key), "actual": value}
        for key, value in actual.items()
        if evidence.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Prior quarantine evidence hash mismatch: {mismatches}")
    if not prior_summary.get("passed", False):
        raise ValueError("Prior quarantine summary did not pass")
    if prior_summary.get("eligible_manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("Prior summary/eligible manifest hash mismatch")
    if prior_summary.get("dataset_lineage_manifest_sha256") != sha256_file(lineage_path):
        raise ValueError("Prior summary/lineage manifest hash mismatch")
    if prior_summary.get("quarantine_registry_sha256") != sha256_file(registry_path):
        raise ValueError("Prior summary/quarantine registry hash mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lineage-manifest", type=Path, required=True)
    parser.add_argument("--quarantine-registry", type=Path, required=True)
    parser.add_argument("--quarantine-summary", type=Path, required=True)
    parser.add_argument(
        "--decision-record",
        type=Path,
        default=Path("configs/protocol/v0.9/decision_records/DR-DUP-001.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit("Duplicate adjudication output directory must be empty")

    decision = yaml.safe_load(args.decision_record.read_text(encoding="utf-8"))
    if decision.get("status") != "approved":
        raise SystemExit("Duplicate quarantine blocked: Decision Record is not approved")
    if decision.get("decision_type") != "iterative_quarantine_adjudication":
        raise SystemExit("Duplicate quarantine blocked: unsupported Decision Record type")
    prior_summary = json.loads(args.quarantine_summary.read_text(encoding="utf-8"))
    try:
        _validate_prior_lineage(
            decision,
            args.manifest,
            args.lineage_manifest,
            args.quarantine_registry,
            args.quarantine_summary,
            prior_summary,
        )
        records = read_manifest_csv(args.manifest)
        expected = decision["expected_counts"]
        approved_ids = [
            str(group["quarantined_sample_id"]) for group in decision["approved_groups"]
        ]
        candidates, newly_quarantined, adjudication = (
            adjudicate_redundant_train_duplicates(
                records,
                approved_quarantined_sample_ids=approved_ids,
                decision_record_id=str(decision["decision_id"]),
                expected_group_count=int(expected["duplicate_groups"]),
            )
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"Duplicate quarantine blocked: {exc}") from exc

    new_ids = {record.sample_id for record in newly_quarantined}
    eligible = [record for record in records if record.sample_id not in new_ids]
    prior_lineage = read_parquet_rows(args.lineage_manifest)
    prior_registry = read_parquet_rows(args.quarantine_registry)
    lineage_ids = {str(row["sample_id"]) for row in prior_lineage}
    if not new_ids.issubset(lineage_ids):
        raise SystemExit("Duplicate quarantine blocked: samples are absent from lineage")
    updated_lineage: list[dict[str, Any]] = []
    for row in prior_lineage:
        resolved = dict(row)
        if str(row["sample_id"]) in new_ids:
            if str(row.get("eligibility_status")) != "eligible":
                raise SystemExit("Duplicate quarantine blocked: sample was already ineligible")
            resolved.update(
                {
                    "eligibility_status": "quarantined",
                    "quarantine_reason_code": TRAIN_DUPLICATE_REASON,
                    "decision_record_id": decision["decision_id"],
                }
            )
        updated_lineage.append(resolved)
    new_registry = [
        {
            **asdict(record),
            "eligibility_status": "quarantined",
            "quarantine_reason_code": TRAIN_DUPLICATE_REASON,
            "violation_scope": "exact_pixel_duplicate_within_leaf",
            "decision_record_id": decision["decision_id"],
        }
        for record in newly_quarantined
    ]
    combined_registry = [*prior_registry, *new_registry]

    audit_rows, audit = audit_manifest_records(eligible)
    eligible_ids = {record.sample_id for record in eligible}
    quarantined_ids = {str(row["sample_id"]) for row in combined_registry}
    all_ids = {str(row["sample_id"]) for row in updated_lineage}
    test_before = {record.sample_id for record in records if record.source_split == "test"}
    test_after = {record.sample_id for record in eligible if record.source_split == "test"}
    eligible_counts = dict(sorted(Counter(item.class_name for item in eligible).items()))
    criteria = {
        "duplicate_adjudication_passed": adjudication["passed"],
        "eligible_image_audit_passed": audit["passed"],
        "audited_count_matches": len(updated_lineage) == int(expected["audited_samples"]),
        "eligible_count_matches": len(eligible) == int(expected["eligible_modeling_samples"]),
        "quarantined_count_matches": len(combined_registry)
        == int(expected["quarantined_source_train_samples"]),
        "sample_reconciliation": eligible_ids.isdisjoint(quarantined_ids)
        and eligible_ids | quarantined_ids == all_ids,
        "official_test_preserved_exactly": test_before == test_after,
        "official_test_count_matches": len(test_after) == int(expected["official_test_samples"]),
        "eligible_source_train_count_matches": sum(
            record.source_split == "train" for record in eligible
        )
        == int(expected["eligible_source_train_samples"]),
        "eligible_class_counts_match": eligible_counts
        == expected["eligible_counts_by_class"],
    }
    summary: dict[str, Any] = {
        "decision_record_id": decision["decision_id"],
        "incorporated_decision_record_ids": decision.get("incorporates", []),
        "audited_sample_count": len(updated_lineage),
        "eligible_sample_count": len(eligible),
        "quarantined_sample_count": len(combined_registry),
        "newly_quarantined_sample_count": len(newly_quarantined),
        "eligible_source_train_count": sum(
            record.source_split == "train" for record in eligible
        ),
        "official_test_count": len(test_after),
        "eligible_counts_by_class": eligible_counts,
        "quarantined_counts_by_class": dict(
            sorted(Counter(str(row["class_name"]) for row in combined_registry).items())
        ),
        "duplicate_adjudication": adjudication,
        "eligible_manifest_audit": audit,
        "prior_quarantine_summary_sha256": sha256_file(args.quarantine_summary),
        "acceptance_criteria": criteria,
        "passed": all(criteria.values()),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.csv"
    write_manifest(eligible, manifest_path)
    write_image_audit_parquet(audit_rows, args.output_dir / "image_audit.parquet")
    duplicate_hash = write_duplicate_adjudication_artifact(
        candidates, args.output_dir / "duplicate_adjudication.parquet"
    )
    summary["duplicate_adjudication_sha256"] = duplicate_hash
    artifact_hashes = write_quarantine_manifest_artifacts(
        updated_lineage,
        combined_registry,
        summary,
        args.output_dir,
        eligible_manifest_path=manifest_path,
    )
    payload = {**summary, "artifact_sha256": artifact_hashes}
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not summary["passed"]:
        raise SystemExit("Duplicate quarantine quality gate failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
