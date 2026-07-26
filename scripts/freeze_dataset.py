"""Freeze an audited manifest into leaf-safe immutable split artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from plantxai_stability.config import load_protocol
from plantxai_stability.data.audit import audit_manifest_records, write_image_audit_parquet
from plantxai_stability.data.freeze import write_frozen_dataset_artifacts
from plantxai_stability.data.manifest import read_manifest_csv
from plantxai_stability.data.splits import group_train_validation
from plantxai_stability.provenance import sha256_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--class-selection-dr", type=Path, required=True)
    parser.add_argument("--quarantine-dr", type=Path, required=True)
    parser.add_argument("--quarantine-registry", type=Path, required=True)
    parser.add_argument("--quarantine-summary", type=Path, required=True)
    parser.add_argument("--audit-identity", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit("Freeze output already exists; use a new versioned directory")
    resolved = load_protocol(args.protocol)
    decision_record = yaml.safe_load(args.class_selection_dr.read_text(encoding="utf-8"))
    if decision_record.get("status") != "approved":
        raise SystemExit("Freeze blocked: class-selection Decision Record is not approved")
    audit_identity_sha256 = sha256_file(args.audit_identity)
    if decision_record.get("audit_identity") != audit_identity_sha256:
        raise SystemExit("Freeze blocked: class-selection audit identity mismatch")
    records = read_manifest_csv(args.manifest)
    quarantine_decision = yaml.safe_load(args.quarantine_dr.read_text(encoding="utf-8"))
    quarantine_summary = json.loads(args.quarantine_summary.read_text(encoding="utf-8"))
    if quarantine_decision.get("status") != "approved":
        raise SystemExit("Freeze blocked: quarantine Decision Record is not approved")
    if quarantine_decision.get("evidence", {}).get(
        "dataset_receipt_sha256"
    ) != audit_identity_sha256:
        raise SystemExit("Freeze blocked: quarantine dataset receipt mismatch")
    if quarantine_decision.get("dataset_revision") != resolved.values["dataset"]["revision"]:
        raise SystemExit("Freeze blocked: quarantine Decision Record revision mismatch")
    if tuple(quarantine_decision.get("selected_classes", [])) != tuple(
        resolved.values["dataset"]["classes"]
    ):
        raise SystemExit("Freeze blocked: quarantine Decision Record class scope mismatch")
    if quarantine_summary.get("decision_record_id") != quarantine_decision.get("decision_id"):
        raise SystemExit("Freeze blocked: quarantine Decision Record mismatch")
    if not quarantine_summary.get("passed", False):
        raise SystemExit("Freeze blocked: quarantine quality gate did not pass")
    if quarantine_summary.get("eligible_manifest_sha256") != sha256_file(args.manifest):
        raise SystemExit("Freeze blocked: eligible manifest hash mismatch")
    if quarantine_summary.get("quarantine_registry_sha256") != sha256_file(
        args.quarantine_registry
    ):
        raise SystemExit("Freeze blocked: quarantine registry hash mismatch")
    expected_counts = quarantine_decision.get("expected_counts", {})
    if quarantine_summary.get("audited_sample_count") != expected_counts.get(
        "audited_samples"
    ):
        raise SystemExit("Freeze blocked: audited sample reconciliation mismatch")
    if quarantine_summary.get("eligible_sample_count") != expected_counts.get(
        "eligible_modeling_samples"
    ):
        raise SystemExit("Freeze blocked: eligible sample count mismatch")
    if quarantine_summary.get("quarantined_sample_count") != expected_counts.get(
        "quarantined_source_train_samples"
    ):
        raise SystemExit("Freeze blocked: quarantined sample count mismatch")
    if quarantine_summary.get("eligible_source_train_count") != expected_counts.get(
        "eligible_source_train_samples"
    ):
        raise SystemExit("Freeze blocked: eligible source-train count mismatch")
    if quarantine_summary.get("official_test_count") != expected_counts.get(
        "official_test_samples"
    ):
        raise SystemExit("Freeze blocked: official test was not preserved")
    expected_classes = tuple(resolved.values["dataset"]["classes"])
    if tuple(decision_record.get("selected_classes", [])) != expected_classes:
        raise SystemExit("Decision Record classes do not match the frozen protocol")
    audit_rows, audit = audit_manifest_records(records)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    audit_path = args.output_dir / "image_audit.parquet"
    if audit_path.exists():
        raise SystemExit("Freeze blocked: output directory already contains image_audit.parquet")
    if not audit["passed"]:
        write_image_audit_parquet(audit_rows, audit_path)
        raise SystemExit(f"Dataset audit failed: {audit}")
    frozen_records = group_train_validation(records, args.validation_fraction, resolved.seed)
    write_image_audit_parquet(audit_rows, audit_path)
    hashes = write_frozen_dataset_artifacts(
        frozen_records,
        args.output_dir,
        protocol_hash=resolved.sha256,
        audit_identity=audit_identity_sha256,
        class_selection_decision_record=str(args.class_selection_dr),
        quarantine_decision_record=str(args.quarantine_dr),
        quarantine_registry_sha256=sha256_file(args.quarantine_registry),
        quarantine_summary_sha256=sha256_file(args.quarantine_summary),
        split_policy=(
            "official_test_preserved; approved_train_quarantine; "
            "leaf-grouped; class-stratified; deterministic"
        ),
        seed=resolved.seed,
    )
    print({"protocol_hash": resolved.sha256, "artifact_sha256": hashes})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
