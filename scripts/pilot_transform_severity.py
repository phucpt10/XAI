"""Run a validation-only image-space transformation severity pilot."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from plantxai_stability.config import load_protocol
from plantxai_stability.contracts import SampleRecord
from plantxai_stability.data.freeze import require_frozen_artifacts
from plantxai_stability.data.loader import load_verified_record
from plantxai_stability.data.manifest import read_manifest_csv
from plantxai_stability.provenance import sha256_bytes, sha256_file
from plantxai_stability.severity import (
    image_change_metrics,
    select_leaf_balanced_pilot_records,
    summarize_pilot_rows,
)
from plantxai_stability.transformations import TransformationPipeline, scenario_grid
from plantxai_stability.transformations import TRANSFORMATION_ALGORITHM_VERSION


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-leaves-per-class", type=int, default=50)
    parser.add_argument("--minimum-leaves-per-class", type=int, default=20)
    args = parser.parse_args()
    if args.minimum_leaves_per_class < 1:
        raise SystemExit("minimum-leaves-per-class must be positive")
    if args.output_dir.exists():
        raise SystemExit("Severity pilot output already exists; use a new versioned directory")
    resolved = load_protocol(args.protocol)
    validation_bundle_report = args.manifest.parent / "validation_bundle_manifest.json"
    if validation_bundle_report.is_file():
        bundle = json.loads(validation_bundle_report.read_text(encoding="utf-8"))
        required_bundle_state = {
            "run_type": "validation_only_recovery_bundle",
            "validation_only_manifest": True,
            "non_validation_entries_materialized": False,
            "all_validation_images_canonical_hash_verified": True,
        }
        if any(bundle.get(key) != value for key, value in required_bundle_state.items()):
            raise SystemExit("Validation bundle provenance is not acceptable")
        if bundle.get("validation_manifest_sha256") != sha256_file(args.manifest):
            raise SystemExit("Validation bundle manifest hash mismatch")
        freeze_path = args.manifest.parent / "freeze_record.json"
        if not freeze_path.is_file() or bundle.get("freeze_record_sha256") != sha256_file(freeze_path):
            raise SystemExit("Validation bundle freeze record hash mismatch")
        freeze_record = json.loads(freeze_path.read_text(encoding="utf-8"))
        all_records = read_manifest_csv(args.manifest)
        if any(record.split != "validation" for record in all_records):
            raise SystemExit("Validation bundle manifest contains a non-validation record")
        bundle_mode = True
    else:
        freeze_record = require_frozen_artifacts(args.manifest)
        all_records = read_manifest_csv(args.manifest)
        bundle_mode = False
    validation_records = [record for record in all_records if record.split == "validation"]
    if not validation_records:
        raise SystemExit("Frozen manifest contains no validation records")
    selected = select_leaf_balanced_pilot_records(
        validation_records,
        seed=resolved.seed,
        max_leaves_per_class=args.max_leaves_per_class,
    )
    class_counts = Counter(record.class_name for record in selected)
    declared_classes = list(resolved.values["dataset"]["classes"])
    insufficient = {
        name: class_counts.get(name, 0)
        for name in declared_classes
        if class_counts.get(name, 0) < args.minimum_leaves_per_class
    }
    if insufficient:
        raise SystemExit(f"Insufficient validation leaves for severity pilot: {insufficient}")
    if len({record.leaf_id for record in selected}) != len(selected):
        raise SystemExit("Pilot selection contains more than one sample from a leaf")
    pipeline = TransformationPipeline(
        resolved.seed, resolved.values["transformations"]["parameters"]
    )
    scenarios = scenario_grid(resolved.values["transformations"]["parameters"])
    if len(scenarios) != 12:
        raise SystemExit(f"Expected 12 protocol scenarios, found {len(scenarios)}")
    original_cache = {
        record.sample_id: load_verified_record(record, args.image_root)
        for record in selected
    }
    rows: list[dict[str, Any]] = []
    deterministic_recheck_passed = True
    for scenario in scenarios:
        print(f"Running {scenario.scenario_id}: {len(selected)} validation leaves")
        for index, record in enumerate(selected):
            original = original_cache[record.sample_id]
            transformed, transformation_record = pipeline.apply(
                original, record.sample_id, scenario
            )
            if index == 0:
                repeated, repeated_record = pipeline.apply(
                    original, record.sample_id, scenario
                )
                deterministic_recheck_passed = deterministic_recheck_passed and (
                    repeated_record == transformation_record
                    and sha256_bytes(repeated.tobytes())
                    == sha256_bytes(transformed.tobytes())
                )
            rows.append(
                {
                    "sample_id": record.sample_id,
                    "leaf_id": record.leaf_id,
                    "class_id": record.class_id,
                    "class_name": record.class_name,
                    "split": record.split,
                    "scenario_id": scenario.scenario_id,
                    "transformation": scenario.transformation,
                    "severity": scenario.severity,
                    "derived_seed": transformation_record.seed,
                    "valid_mask_sha256": transformation_record.valid_mask_sha256,
                    "exact_parameters_json": json.dumps(
                        transformation_record.parameters, sort_keys=True
                    ),
                    **image_change_metrics(original, transformed),
                }
            )
    summary = summarize_pilot_rows(rows)
    selected_ids = [record.sample_id for record in selected]
    randomization_seeds: dict[tuple[str, str], set[int]] = {}
    for row in rows:
        key = (str(row["sample_id"]), str(row["transformation"]))
        randomization_seeds.setdefault(key, set()).add(int(row["derived_seed"]))
    shared_randomization_seed_consistent = all(
        len(seeds) == 1 for seeds in randomization_seeds.values()
    )
    rotation_rows = [row for row in rows if row["transformation"] == "rotation"]
    rotation_parameters = [
        json.loads(str(row["exact_parameters_json"])) for row in rotation_rows
    ]
    opencv_runtime_versions = sorted(
        {str(parameters.get("opencv_version")) for parameters in rotation_parameters}
    )
    opencv_distribution_versions = sorted(
        {
            str(parameters.get("opencv_distribution_version"))
            for parameters in rotation_parameters
        }
    )
    rotation_zero_fill_valid_mask_passed = bool(rotation_rows) and all(
        parameters.get("fill_policy") == "constant_zero"
        and parameters.get("rotation_fill_policy") == "constant_zero"
        and parameters.get("valid_region_policy") == "geometric_support_mask"
        and int(parameters.get("valid_pixel_count", 0)) > 0
        and int(parameters.get("invalid_pixel_count", 0)) > 0
        and 0.0 < float(parameters.get("valid_pixel_fraction", 0.0)) < 1.0
        and 0.0 < float(parameters.get("invalid_pixel_fraction", 0.0)) < 1.0
        and len(str(parameters.get("valid_mask_sha256", ""))) == 64
        and len(str(parameters.get("rotated_output_rgb_sha256", ""))) == 64
        and int(parameters.get("opencv_num_threads", -1)) == 1
        and parameters.get("opencv_opencl_enabled") is False
        and len(str(row.get("valid_mask_sha256") or "")) == 64
        for row, parameters in zip(rotation_rows, rotation_parameters)
    )
    invalid_fractions_by_sample: dict[str, dict[str, float]] = {}
    for row, parameters in zip(rotation_rows, rotation_parameters):
        invalid_fractions_by_sample.setdefault(str(row["sample_id"]), {})[
            str(row["severity"])
        ] = float(parameters["invalid_pixel_fraction"])
    rotation_invalid_fraction_strictly_increases = all(
        set(values) == {"mild", "moderate", "severe"}
        and values["mild"] < values["moderate"] < values["severe"]
        for values in invalid_fractions_by_sample.values()
    )
    rotation_invalid_fraction_summary = {}
    for severity in ("mild", "moderate", "severe"):
        values = np.asarray(
            [
                float(parameters["invalid_pixel_fraction"])
                for row, parameters in zip(rotation_rows, rotation_parameters)
                if row["severity"] == severity
            ],
            dtype=np.float64,
        )
        rotation_invalid_fraction_summary[severity] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "q95": float(np.quantile(values, 0.95)),
        }
    technical_gate_passed = bool(
        len(selected) == len(set(selected_ids))
        and all(record.split == "validation" for record in selected)
        and summary["scenario_count"] == 12
        and summary["all_metrics_finite"]
        and summary["ordinal_gate_passed"]
        and deterministic_recheck_passed
        and shared_randomization_seed_consistent
        and rotation_zero_fill_valid_mask_passed
        and rotation_invalid_fraction_strictly_increases
        and len(opencv_runtime_versions) == 1
        and len(opencv_distribution_versions) == 1
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    records_path = args.output_dir / "severity_pilot_records.parquet"
    selection_path = args.output_dir / "severity_pilot_selection.csv"
    summary_path = args.output_dir / "severity_pilot_summary.json"
    pd.DataFrame(rows).to_parquet(records_path, index=False)
    _write_selection(selection_path, selected)
    report = {
        "run_type": "validation_only_image_space_severity_pilot",
        "official_result": False,
        "approval_status": "pending_human_review",
        "human_approval_required": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": resolved.values["protocol_version"],
        "protocol_hash": resolved.sha256,
        "transformation_algorithm_version": TRANSFORMATION_ALGORITHM_VERSION,
        "frozen_manifest_sha256": sha256_file(args.manifest),
        "validation_bundle_mode": bundle_mode,
        "validation_bundle_report_sha256": (
            sha256_file(validation_bundle_report) if bundle_mode else ""
        ),
        "freeze_record_protocol_hash": freeze_record.get("protocol_hash"),
        "freeze_protocol_hash_matches_current": (
            freeze_record.get("protocol_hash") == resolved.sha256
        ),
        "seed": resolved.seed,
        "source_split": "validation",
        "test_split_accessed": False,
        "selection_policy": (
            "one deterministic sample per leaf; balanced cap per declared class"
        ),
        "max_leaves_per_class": args.max_leaves_per_class,
        "minimum_leaves_per_class": args.minimum_leaves_per_class,
        "selected_sample_count": len(selected),
        "selected_leaf_count": len({record.leaf_id for record in selected}),
        "selected_counts_by_class": dict(sorted(class_counts.items())),
        "selection_sample_ids_sha256": sha256_bytes(
            json.dumps(selected_ids, separators=(",", ":")).encode("utf-8")
        ),
        "deterministic_recheck_passed": deterministic_recheck_passed,
        "shared_randomization_seed_consistent": shared_randomization_seed_consistent,
        "rotation_zero_fill_valid_mask_passed": rotation_zero_fill_valid_mask_passed,
        "rotation_invalid_fraction_strictly_increases": (
            rotation_invalid_fraction_strictly_increases
        ),
        "rotation_invalid_fraction_summary": rotation_invalid_fraction_summary,
        "rotation_prediction_claim_scope": (
            resolved.values["xai"]["rotation_prediction_claim_scope"]
        ),
        "xai_alignment_policy": resolved.values["xai"]["alignment_policy"],
        "opencv_runtime_versions": opencv_runtime_versions,
        "opencv_distribution_versions": opencv_distribution_versions,
        "summary": summary,
        "acceptance_criteria": {
            "validation_only": True,
            "one_sample_per_leaf": True,
            "minimum_class_leaf_coverage": not insufficient,
            "all_12_scenarios_present": summary["scenario_count"] == 12,
            "all_metrics_finite": summary["all_metrics_finite"],
            "median_rmse_strictly_increases_by_severity": summary[
                "ordinal_gate_passed"
            ],
            "deterministic_recheck_passed": deterministic_recheck_passed,
            "shared_randomization_seed_consistent": (
                shared_randomization_seed_consistent
            ),
            "rotation_zero_fill_valid_mask_passed": (
                rotation_zero_fill_valid_mask_passed
            ),
            "rotation_invalid_fraction_strictly_increases": (
                rotation_invalid_fraction_strictly_increases
            ),
            "single_opencv_runtime_version": len(opencv_runtime_versions) == 1,
            "single_opencv_distribution_version": (
                len(opencv_distribution_versions) == 1
            ),
        },
        "technical_gate_passed": technical_gate_passed,
        "decision": (
            "No automatic approval. Review distributions and representative images, "
            "then record a Decision Record before changing the protocol blocker."
        ),
        "artifact_sha256": {
            records_path.name: sha256_file(records_path),
            selection_path.name: sha256_file(selection_path),
        },
    }
    summary_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not technical_gate_passed:
        raise SystemExit("Severity pilot technical gate failed; review the report")
    print(f"Severity pilot technical gate: PASS\nReport: {summary_path}")
    return 0


def _write_selection(path: Path, records: list[SampleRecord]) -> None:
    fields = [
        "sample_id",
        "leaf_id",
        "class_id",
        "class_name",
        "split",
        "canonical_relative_path",
        "canonical_rgb_sha256",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            values = asdict(record)
            writer.writerow({field: values[field] for field in fields})


if __name__ == "__main__":
    raise SystemExit(main())
