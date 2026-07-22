"""Build a canonical manifest and audit report from a metadata CSV.

The metadata CSV must contain: relative_path, leaf_id, class_name and
source_split. This script intentionally refuses to fabricate leaf IDs.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from plantxai_stability.data.audit import audit_manifest_records, write_image_audit_parquet
from plantxai_stability.data.manifest import build_manifest, canonical_rgb_sha256, inspect_manifest, write_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path, required=True)
    parser.add_argument("--repository", default="local")
    parser.add_argument("--configuration", default="filesystem")
    parser.add_argument("--requested-revision", default=None)
    parser.add_argument("--resolved-revision", default=None)
    args = parser.parse_args()
    rows = []
    invalid_audit_rows = []
    with args.metadata_csv.open("r", encoding="utf-8", newline="") as handle:
        for source_row_index, row in enumerate(csv.DictReader(handle)):
            try:
                for required in ("relative_path", "leaf_id", "class_name", "source_split"):
                    if not row.get(required):
                        raise ValueError(f"MISSING_{required.upper()}")
                image_path = args.root / row["relative_path"]
                if not image_path.is_file():
                    raise ValueError("IMAGE_NOT_FOUND")
                digest, width, height = canonical_rgb_sha256(image_path)
                rows.append({
                    "leaf_id": row["leaf_id"],
                    "class_id": int(row.get("class_id", 0)),
                    "class_name": row["class_name"],
                    "source_split": row["source_split"],
                    "split": row["source_split"],
                    "canonical_relative_path": row["relative_path"].replace("\\", "/"),
                    "canonical_rgb_sha256": digest,
                    "width": width,
                    "height": height,
                    "source_row_index": source_row_index,
                })
            except (OSError, TypeError, ValueError) as exc:
                invalid_audit_rows.append(
                    {
                        "source_sample_identity": row.get("relative_path", f"row-{source_row_index}"),
                        "source_split": row.get("source_split"),
                        "source_row_index": source_row_index,
                        "canonical_relative_path": row.get("relative_path"),
                        "label_id": row.get("class_id"),
                        "canonical_class_name": row.get("class_name"),
                        "leaf_id": row.get("leaf_id"),
                        "width": None,
                        "height": None,
                        "channels": None,
                        "canonical_rgb_sha256": None,
                        "valid": False,
                        "reason_code": "INVALID_SOURCE_SAMPLE",
                        "error_detail": str(exc),
                    }
                )
    if not rows:
        raise SystemExit("Dataset audit failed: no valid image rows")
    records = build_manifest(rows)
    audit_rows, audit = audit_manifest_records(records)
    audit_rows.extend(invalid_audit_rows)
    audit["invalid_source_sample_count"] = len(invalid_audit_rows)
    audit["passed"] = bool(audit["passed"]) and not invalid_audit_rows
    if not invalid_audit_rows:
        write_manifest(records, args.manifest_out)
    audit.update(inspect_manifest(records))
    write_image_audit_parquet(audit_rows, args.audit_out.with_name("image_audit.parquet"))
    receipt = {
        "repository": args.repository,
        "dataset_configuration": args.configuration,
        "requested_revision": args.requested_revision,
        "resolved_revision": args.resolved_revision,
        "dataset_fingerprints": {"manifest_sha256": None},
        "feature_schema": sorted(records[0].__dataclass_fields__),
        "source_split_names": sorted({record.source_split for record in records}),
        "sample_count": len(records) + len(invalid_audit_rows),
        "source_split_counts": {
            split: sum(record.source_split == split for record in records)
            for split in sorted({record.source_split for record in records})
        },
        "cache_or_storage_location": str(args.root),
        "retrieval_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.audit_out.with_name("dataset_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8"
    )
    args.audit_out.parent.mkdir(parents=True, exist_ok=True)
    args.audit_out.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    if not audit["passed"]:
        raise SystemExit("Dataset audit failed; inspect image_audit.parquet and dataset_audit.json")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
