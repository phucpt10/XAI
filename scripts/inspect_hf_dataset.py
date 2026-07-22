"""Inspect and optionally materialise a canonical manifest from Hugging Face.

Examples (Colab):
  python scripts/inspect_hf_dataset.py --output-dir artifacts/dataset_inspection
  python scripts/inspect_hf_dataset.py --revision <commit-sha> --classes Tomato___healthy ...
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from plantxai_stability.data.audit import audit_manifest_records, write_image_audit_parquet
from plantxai_stability.data.huggingface import (
    build_hf_manifest,
    inspect_hf_schema,
    load_hf_dataset,
    split_counts,
)
from plantxai_stability.data.manifest import write_manifest
from plantxai_stability.provenance import sha256_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", default="mohanty/PlantVillage")
    parser.add_argument("--configuration", default="color")
    parser.add_argument("--revision", help="Immutable HF commit SHA; required for official runs")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("/content/plantxai-hf-cache")
    )
    parser.add_argument(
        "--classes", nargs="*", help="Optional class subset; defaults to all classes"
    )
    parser.add_argument(
        "--manifest", action="store_true", help="Decode images and write manifest.csv"
    )
    parser.add_argument(
        "--class-selection-dr",
        type=Path,
        default=Path("configs/protocol/v0.9/decision_records/DR-CLASS-001.yaml"),
    )
    parser.add_argument(
        "--leaf-identity-dr",
        type=Path,
        default=Path("configs/protocol/v0.9/decision_records/DR-LEAF-001.yaml"),
    )
    parser.add_argument("--leaf-identity-summary", type=Path)
    args = parser.parse_args()

    decision_record = None
    leaf_decision_record = None
    if args.manifest:
        if not args.classes:
            raise SystemExit("Official manifest requires an explicit --classes list")
        try:
            import yaml
        except ImportError as exc:
            raise SystemExit(
                "PyYAML is required to validate the class-selection Decision Record"
            ) from exc
        try:
            decision_record = yaml.safe_load(
                args.class_selection_dr.read_text(encoding="utf-8")
            )
        except (FileNotFoundError, yaml.YAMLError) as exc:
            raise SystemExit(f"Cannot load class-selection Decision Record: {exc}") from exc
        if decision_record.get("status") != "approved":
            raise SystemExit(
                "Official manifest blocked: class-selection Decision Record is not approved"
            )
        if tuple(args.classes) != tuple(decision_record.get("selected_classes", [])):
            raise SystemExit("Requested classes do not match the approved Decision Record")
        if args.leaf_identity_summary is None:
            raise SystemExit("Official manifest requires --leaf-identity-summary")
        try:
            leaf_decision_record = yaml.safe_load(
                args.leaf_identity_dr.read_text(encoding="utf-8")
            )
            leaf_summary = json.loads(args.leaf_identity_summary.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, yaml.YAMLError) as exc:
            raise SystemExit(f"Cannot load leaf identity governance evidence: {exc}") from exc
        if leaf_decision_record.get("status") != "approved":
            raise SystemExit("Official manifest blocked: leaf identity Decision Record is not approved")
        if leaf_decision_record.get("dataset_revision") != args.revision:
            raise SystemExit("Leaf identity Decision Record revision mismatch")
        if tuple(leaf_decision_record.get("selected_classes", [])) != tuple(args.classes):
            raise SystemExit("Leaf identity Decision Record class scope mismatch")
        if not leaf_summary.get("passed", False):
            raise SystemExit("Official manifest blocked: leaf identity audit did not pass")
        evidence = leaf_decision_record.get("evidence", {})
        if evidence.get("resolution_report_sha256") != leaf_summary.get("report_sha256"):
            raise SystemExit("Leaf identity report hash does not match its Decision Record")
        if evidence.get("resolution_summary_sha256") != sha256_file(args.leaf_identity_summary):
            raise SystemExit("Leaf identity summary hash does not match its Decision Record")
        if evidence.get("dataset_receipt_sha256") != leaf_summary.get(
            "dataset_receipt_sha256"
        ):
            raise SystemExit("Dataset receipt hash does not match the leaf Decision Record")

    dataset = load_hf_dataset(
        args.dataset_id,
        args.configuration,
        args.revision,
        selected_classes=args.classes or None,
        prepare_images=args.manifest,
        allow_filename_reconstruction=args.manifest,
        cache_dir=args.cache_dir,
        token=os.getenv("HF_TOKEN"),
    )
    resolved_revision = getattr(dataset, "resolved_revision", args.revision)
    schema = inspect_hf_schema(
        dataset,
        dataset_id=args.dataset_id,
        configuration=args.configuration,
        revision=resolved_revision,
    )
    report: dict[str, object] = {
        "dataset_id": schema.dataset_id,
        "configuration": schema.configuration,
        "revision": schema.revision,
        "requested_revision": args.revision,
        "resolved_revision": schema.revision,
        "splits": list(schema.splits),
        "split_counts": split_counts(dataset),
        "features": list(schema.features),
        "label_names": list(schema.label_names),
        "leaf_id_present": schema.has_leaf_id,
        "selected_classes": args.classes or list(schema.label_names),
    }
    if not schema.has_leaf_id:
        raise SystemExit("Dataset inspection failed: leaf_id is absent; unsafe to continue")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fingerprints = {
        str(name): getattr(split, "_fingerprint", None) for name, split in dataset.items()
    }
    counts = split_counts(dataset)
    class_counts_by_split = {
        str(name): dict(sorted(Counter(str(row["label"]) for row in split).items()))
        for name, split in dataset.items()
    }
    unresolved_leaf_id_count_by_split = {
        str(name): sum(row.get("leaf_id") in (None, "") for row in split)
        for name, split in dataset.items()
    }
    unresolved_leaf_id_count_by_class_and_split = {
        str(name): dict(
            sorted(
                Counter(
                    str(row["label"])
                    for row in split
                    if row.get("leaf_id") in (None, "")
                ).items()
            )
        )
        for name, split in dataset.items()
    }
    unique_leaf_count_by_split = {
        str(name): len(
            {str(row["leaf_id"]) for row in split if row.get("leaf_id") not in (None, "")}
        )
        for name, split in dataset.items()
    }
    report["class_counts_by_split"] = class_counts_by_split
    report["unresolved_leaf_id_count_by_split"] = unresolved_leaf_id_count_by_split
    report["unresolved_leaf_id_count_by_class_and_split"] = (
        unresolved_leaf_id_count_by_class_and_split
    )
    report["unique_leaf_count_by_split"] = unique_leaf_count_by_split
    receipt = {
        "repository": args.dataset_id,
        "dataset_configuration": args.configuration,
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
        "dataset_fingerprints": fingerprints,
        "feature_schema": list(schema.features),
        "source_split_names": list(schema.splits),
        "sample_count": sum(counts.values()),
        "source_split_counts": counts,
        "class_counts_by_split": class_counts_by_split,
        "unresolved_leaf_id_count_by_split": unresolved_leaf_id_count_by_split,
        "unresolved_leaf_id_count_by_class_and_split": (
            unresolved_leaf_id_count_by_class_and_split
        ),
        "unique_leaf_count_by_split": unique_leaf_count_by_split,
        "cache_or_storage_location": getattr(dataset, "cache_location", str(args.cache_dir)),
        "source_file_sha256": getattr(dataset, "source_file_sha256", {}),
        "retrieval_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.output_dir / "dataset_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8"
    )
    if args.manifest:
        records = build_hf_manifest(
            dataset,
            schema,
            selected_classes=args.classes,
            materialize_root=args.output_dir,
        )
        write_manifest(records, args.output_dir / "manifest.csv")
        audit_rows, audit = audit_manifest_records(records)
        report["manifest_audit"] = audit
        write_image_audit_parquet(audit_rows, args.output_dir / "image_audit.parquet")
        report["image_root"] = str(args.output_dir)
    (args.output_dir / "hf_schema.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
