"""Freeze an audited manifest into leaf-safe immutable split artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from plantxai_stability.config import load_protocol
from plantxai_stability.data.audit import audit_manifest_records, write_image_audit_parquet
from plantxai_stability.data.freeze import write_frozen_dataset_artifacts
from plantxai_stability.data.manifest import read_manifest_csv
from plantxai_stability.data.splits import group_train_validation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--class-selection-dr", type=Path, required=True)
    parser.add_argument("--audit-identity", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()
    resolved = load_protocol(args.protocol)
    decision_record = yaml.safe_load(args.class_selection_dr.read_text(encoding="utf-8"))
    if decision_record.get("status") != "approved":
        raise SystemExit("Freeze blocked: class-selection Decision Record is not approved")
    records = read_manifest_csv(args.manifest)
    expected_classes = tuple(resolved.values["dataset"]["classes"])
    if tuple(decision_record.get("selected_classes", [])) != expected_classes:
        raise SystemExit("Decision Record classes do not match the frozen protocol")
    audit_rows, audit = audit_manifest_records(records)
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
        audit_identity=args.audit_identity,
        class_selection_decision_record=str(args.class_selection_dr),
        split_policy="official_test_preserved; leaf-grouped; class-stratified; deterministic",
        seed=resolved.seed,
    )
    print({"protocol_hash": resolved.sha256, "artifact_sha256": hashes})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
