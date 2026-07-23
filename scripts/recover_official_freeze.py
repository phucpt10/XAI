"""Reconstruct physical freeze artifacts under DR-RECOVERY-001."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from plantxai_stability.artifacts import atomic_json
from plantxai_stability.data.freeze import require_frozen_artifacts
from plantxai_stability.data.loader import load_verified_record
from plantxai_stability.data.manifest import read_manifest_csv
from plantxai_stability.data.splits import validate_frozen_splits
from plantxai_stability.provenance import sha256_file
from plantxai_stability.recovery import (
    RECOVERY_SCHEMA_VERSION,
    load_recovery_decision,
    validate_recovery_binding,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-freeze-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--recovery-audit", type=Path, required=True)
    parser.add_argument("--recovery-decision-record", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit("Recovery output exists; use a new immutable versioned directory")
    if args.progress_every < 1:
        raise SystemExit("--progress-every must be at least 1")

    decision = load_recovery_decision(args.recovery_decision_record)
    evidence = decision["recovery_evidence"]
    recovery_audit = json.loads(args.recovery_audit.read_text(encoding="utf-8"))
    if not isinstance(recovery_audit, dict):
        raise SystemExit("Recovery audit must be a JSON mapping")
    source_manifest = args.source_freeze_dir / "dataset_manifest.csv"
    source_freeze_path = args.source_freeze_dir / "freeze_record.json"
    criteria = {
        "archive_sha256_matches": sha256_file(args.archive)
        == evidence["archive_sha256"],
        "recovery_audit_sha256_matches": sha256_file(args.recovery_audit)
        == evidence["recovery_audit_sha256"],
        "recovery_audit_is_json_mapping": True,
        "source_freeze_sha256_matches": sha256_file(source_freeze_path)
        == evidence["recovered_source_freeze_record_sha256"],
        "manifest_sha256_matches": sha256_file(source_manifest)
        == evidence["manifest_sha256"],
    }
    if not all(criteria.values()):
        failed = sorted(key for key, value in criteria.items() if not value)
        raise SystemExit(f"Recovery source evidence failed: {failed}")

    source_freeze = require_frozen_artifacts(source_manifest)
    records = read_manifest_csv(source_manifest)
    validate_frozen_splits(records)
    split_counts = dict(Counter(record.split for record in records))
    criteria["sample_count_matches"] = len(records) == evidence["verified_sample_count"]
    criteria["split_counts_match"] = split_counts == evidence["split_counts"]
    if not criteria["sample_count_matches"] or not criteria["split_counts_match"]:
        raise SystemExit("Recovered manifest sample or split counts do not match approval")

    for index, record in enumerate(records, start=1):
        load_verified_record(record, args.image_root)
        if index % args.progress_every == 0 or index == len(records):
            print(f"Verified recovered images: {index}/{len(records)}")
    criteria["all_manifest_images_verified"] = True

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{args.output_dir.name}.staging-",
            dir=args.output_dir.parent,
        )
    )
    try:
        artifact_hashes = source_freeze.get("artifact_sha256", {})
        if not isinstance(artifact_hashes, dict) or not artifact_hashes:
            raise ValueError("Source freeze does not declare artifact hashes")
        for name, expected_hash in artifact_hashes.items():
            source = args.source_freeze_dir / str(name)
            if not source.is_file() or sha256_file(source) != expected_hash:
                raise ValueError(f"Source freeze artifact mismatch: {name}")
            shutil.copy2(source, staging / str(name))
        copied_hashes = {
            str(name): sha256_file(staging / str(name)) for name in artifact_hashes
        }
        recovered_freeze = {
            "schema_version": RECOVERY_SCHEMA_VERSION,
            "protocol_hash": evidence["checkpoint_training_protocol_hash"],
            "manifest_sha256": evidence["manifest_sha256"],
            "artifact_sha256": copied_hashes,
            "historical_final_freeze_record_sha256": evidence[
                "historical_final_freeze_record_sha256"
            ],
            "source_freeze_record_sha256": evidence[
                "recovered_source_freeze_record_sha256"
            ],
            "recovery_decision_id": decision["decision_id"],
            "recovery_decision_record_sha256": sha256_file(
                args.recovery_decision_record
            ),
            "recovery_audit_sha256": evidence["recovery_audit_sha256"],
            "archive_sha256": evidence["archive_sha256"],
            "creation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "recovery_scope": "infrastructure_only_no_scientific_change",
        }
        atomic_json(staging / "freeze_record.json", recovered_freeze)
        physical_freeze_sha256 = sha256_file(staging / "freeze_record.json")
        criteria["physical_freeze_distinct_from_historical"] = (
            physical_freeze_sha256
            != evidence["historical_final_freeze_record_sha256"]
        )
        report = {
            "schema_version": RECOVERY_SCHEMA_VERSION,
            "run_type": "infrastructure_only_physical_freeze_recovery",
            "recovery_gate_passed": all(criteria.values()),
            "recovery_decision_id": decision["decision_id"],
            "recovery_decision_record_sha256": sha256_file(
                args.recovery_decision_record
            ),
            "archive_sha256": evidence["archive_sha256"],
            "recovery_audit_sha256": evidence["recovery_audit_sha256"],
            "manifest_sha256": evidence["manifest_sha256"],
            "verified_sample_count": len(records),
            "split_counts": split_counts,
            "historical_final_freeze_record_sha256": evidence[
                "historical_final_freeze_record_sha256"
            ],
            "source_freeze_record_sha256": evidence[
                "recovered_source_freeze_record_sha256"
            ],
            "physical_freeze_record_sha256": physical_freeze_sha256,
            "image_bytes_accessed_for_integrity_recovery": True,
            "official_test_evaluation_computed": False,
            "scientific_configuration_changed": False,
            "acceptance_criteria": criteria,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if not report["recovery_gate_passed"]:
            raise ValueError("Recovery acceptance criteria failed")
        atomic_json(staging / "recovery_binding_report.json", report)
        os.replace(staging, args.output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    lineage = validate_recovery_binding(
        manifest_path=args.output_dir / "dataset_manifest.csv",
        recovery_decision_path=args.recovery_decision_record,
        recovery_binding_report_path=args.output_dir / "recovery_binding_report.json",
    )
    print(json.dumps({**report, "validated_lineage": lineage}, indent=2, sort_keys=True))
    print(
        "Physical freeze recovery gate: PASS\n"
        f"Manifest: {args.output_dir / 'dataset_manifest.csv'}\n"
        f"Report: {args.output_dir / 'recovery_binding_report.json'}\n"
        f"Report SHA-256: {sha256_file(args.output_dir / 'recovery_binding_report.json')}\n"
        "New official execution authorized through recovery bridge: TRUE"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
