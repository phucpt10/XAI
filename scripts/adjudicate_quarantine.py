"""Create immutable quarantine evidence from an approved leaf-overlap decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from plantxai_stability.data.quarantine import (
    adjudicate_train_test_leaf_overlap,
    read_parquet_rows,
    write_quarantine_adjudication_artifacts,
)
from plantxai_stability.provenance import sha256_file


def _require_exact_evidence(
    decision: dict[str, Any],
    report_path: Path,
    summary_path: Path,
    receipt_path: Path,
    leaf_summary: dict[str, Any],
) -> None:
    evidence = decision.get("evidence", {})
    actual = {
        "dataset_receipt_sha256": sha256_file(receipt_path),
        "leaf_identity_resolution_report_sha256": sha256_file(report_path),
        "leaf_identity_resolution_summary_sha256": sha256_file(summary_path),
    }
    mismatches = {
        key: {"expected": evidence.get(key), "actual": value}
        for key, value in actual.items()
        if evidence.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Quarantine evidence hash mismatch: {mismatches}")
    if leaf_summary.get("dataset_receipt_sha256") != actual["dataset_receipt_sha256"]:
        raise ValueError("Leaf summary does not reference the supplied dataset receipt")
    if leaf_summary.get("report_sha256") != actual[
        "leaf_identity_resolution_report_sha256"
    ]:
        raise ValueError("Leaf summary does not reference the supplied Parquet report")
    if leaf_summary.get("resolved_revision") != decision.get("dataset_revision"):
        raise ValueError("Leaf summary revision does not match the Decision Record")
    if tuple(leaf_summary.get("selected_classes", [])) != tuple(
        decision.get("selected_classes", [])
    ):
        raise ValueError("Leaf summary class scope does not match the Decision Record")
    if leaf_summary.get("source_file_sha256") != evidence.get("source_file_sha256"):
        raise ValueError("Pinned source-file hashes do not match the Decision Record")
    criteria = leaf_summary.get("acceptance_criteria", {})
    expected = {
        "coverage_equals_1": True,
        "no_ambiguous_identity": True,
        "no_reconstructed_collision": True,
        "no_leaf_class_conflict": True,
        "no_leaf_split_overlap": False,
    }
    if criteria != expected:
        raise ValueError(
            "Quarantine is allowed only when leaf overlap is the sole failed identity gate"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leaf-report", type=Path, required=True)
    parser.add_argument("--leaf-summary", type=Path, required=True)
    parser.add_argument("--dataset-receipt", type=Path, required=True)
    parser.add_argument(
        "--decision-record",
        type=Path,
        default=Path("configs/protocol/v0.9/decision_records/DR-LEAF-002.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    decision = yaml.safe_load(args.decision_record.read_text(encoding="utf-8"))
    if decision.get("status") != "approved":
        raise SystemExit("Quarantine blocked: Decision Record is not approved")
    if decision.get("decision_type") != "quarantine_adjudication":
        raise SystemExit("Quarantine blocked: unsupported Decision Record type")
    leaf_summary = json.loads(args.leaf_summary.read_text(encoding="utf-8"))
    try:
        _require_exact_evidence(
            decision,
            args.leaf_report,
            args.leaf_summary,
            args.dataset_receipt,
            leaf_summary,
        )
        candidates, registry, summary = adjudicate_train_test_leaf_overlap(
            read_parquet_rows(args.leaf_report),
            approved_overlap_ids=decision["approved_overlap_leaf_ids"],
            decision_record_id=str(decision["decision_id"]),
            expected_quarantined_train_count=int(
                decision["expected_counts"]["quarantined_source_train_samples"]
            ),
            expected_official_test_count=int(
                decision["expected_counts"]["official_test_samples"]
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"Quarantine adjudication blocked: {exc}") from exc
    summary.update(
        {
            "dataset_revision": decision["dataset_revision"],
            "selected_classes": decision["selected_classes"],
            "input_dataset_receipt_sha256": sha256_file(args.dataset_receipt),
            "input_leaf_report_sha256": sha256_file(args.leaf_report),
            "input_leaf_summary_sha256": sha256_file(args.leaf_summary),
        }
    )
    hashes = write_quarantine_adjudication_artifacts(
        candidates, registry, summary, args.output_dir
    )
    payload = {**summary, "artifact_sha256": hashes}
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not summary["passed"]:
        raise SystemExit("Quarantine adjudication gate failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
