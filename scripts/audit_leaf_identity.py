"""Audit and reconstruct PlantVillage leaf identity before manifest creation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from plantxai_stability.data.huggingface import inspect_hf_schema, load_hf_dataset
from plantxai_stability.data.leaf_identity import (
    audit_leaf_identity,
    write_leaf_identity_artifacts,
)
from plantxai_stability.provenance import sha256_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", default="mohanty/PlantVillage")
    parser.add_argument("--configuration", default="color")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--classes", nargs="+", required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("/content/plantxai-hf-cache"))
    parser.add_argument("--dataset-receipt", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    dataset = load_hf_dataset(
        args.dataset_id,
        args.configuration,
        args.revision,
        selected_classes=args.classes,
        prepare_images=False,
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
    if not schema.has_leaf_id:
        raise SystemExit("Leaf identity audit blocked: source schema has no leaf_id")
    rows, summary = audit_leaf_identity(dataset)
    summary.update(
        {
            "dataset_id": args.dataset_id,
            "configuration": args.configuration,
            "requested_revision": args.revision,
            "resolved_revision": resolved_revision,
            "selected_classes": args.classes,
            "source_file_sha256": getattr(dataset, "source_file_sha256", {}),
            "dataset_receipt_sha256": (
                sha256_file(args.dataset_receipt) if args.dataset_receipt else None
            ),
            "resolution_policy": {
                "mapped": "leaf-map.json unique or class-disambiguated match",
                "unmapped": "source-loader-compatible fallback_<filename-identity>",
                "unmapped_source_label": "filename_reconstructed",
            },
        }
    )
    hashes = write_leaf_identity_artifacts(rows, summary, args.output_dir)
    payload = {**summary, "artifact_sha256": hashes}
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not summary["passed"]:
        raise SystemExit("Leaf identity resolution gate failed; review the Parquet report")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

