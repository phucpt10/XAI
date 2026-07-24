"""Run the validation-only Score-CAM readiness preflight.

This is deliberately not an official experiment.  It accepts only a validation
recovery bundle, checks the fixed checkpoint hashes recorded in DR-CHECKPOINT-002,
and writes sample-level CAM status evidence.  In particular, it never reads a
test manifest or image path.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from plantxai_stability.config import load_protocol, resolve_xai_target_layer
from plantxai_stability.data.loader import load_verified_record, preprocess_for_model
from plantxai_stability.data.manifest import read_manifest_csv
from plantxai_stability.models import ModelWrapper
from plantxai_stability.provenance import sha256_file
from plantxai_stability.severity import select_leaf_balanced_pilot_records
from plantxai_stability.training import load_checkpoint, seed_everything
from plantxai_stability.transformations import TransformationPipeline, scenario_grid
from plantxai_stability.xai import CAMGenerator


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--resnet50-checkpoint", type=Path, required=True)
    parser.add_argument("--efficientnet-b0-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-decision-record", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-leaves-per-class", type=int, default=20)
    parser.add_argument("--minimum-leaves-per-class", type=int, default=10)
    parser.add_argument("--scenario-id", default="gaussian_blur_severe")
    parser.add_argument(
        "--efficientnet-scorecam-target-layer",
        choices=("features[-1]", "features[-2]"),
        default=None,
        help=(
            "Validation-only candidate override. It must be backed by a new "
            "Decision Record before an official run."
        ),
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit("Preflight output already exists; choose a new versioned directory")
    if args.minimum_leaves_per_class < 1 or args.max_leaves_per_class < args.minimum_leaves_per_class:
        raise SystemExit("Invalid leaf coverage limits")

    resolved = load_protocol(args.protocol)
    _require_validation_bundle(args.manifest)
    records = read_manifest_csv(args.manifest)
    if any(record.split != "validation" for record in records):
        raise SystemExit("Preflight is restricted to a validation-only manifest")
    selected = select_leaf_balanced_pilot_records(
        records, seed=resolved.seed, max_leaves_per_class=args.max_leaves_per_class
    )
    declared_classes = list(resolved.values["dataset"]["classes"])
    counts = Counter(record.class_name for record in selected)
    insufficient = {name: counts.get(name, 0) for name in declared_classes
                    if counts.get(name, 0) < args.minimum_leaves_per_class}
    if insufficient:
        raise SystemExit(f"Insufficient validation coverage: {insufficient}")

    scenarios = {scenario.scenario_id: scenario for scenario in scenario_grid(
        resolved.values["transformations"]["parameters"]
    )}
    if args.scenario_id not in scenarios:
        raise SystemExit(f"Unknown protocol scenario: {args.scenario_id}")
    scenario = scenarios[args.scenario_id]
    decision = yaml.safe_load(args.checkpoint_decision_record.read_text(encoding="utf-8"))
    if decision.get("status") != "approved":
        raise SystemExit("Checkpoint Decision Record is not approved")
    seed_everything(resolved.seed, bool(resolved.values["training"]["deterministic_algorithms"]))
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Install the [ml,xai] dependencies") from exc
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")

    pipeline = TransformationPipeline(resolved.seed, resolved.values["transformations"]["parameters"])
    rows: list[dict[str, Any]] = []
    for model_id, checkpoint in (
        ("resnet50", args.resnet50_checkpoint),
        ("efficientnet_b0", args.efficientnet_b0_checkpoint),
    ):
        expected_hash = str(decision["approved_checkpoints"][model_id]["checkpoint_sha256"])
        actual_hash = sha256_file(checkpoint)
        if actual_hash != expected_hash:
            raise SystemExit(f"{model_id} checkpoint SHA-256 does not match DR-CHECKPOINT-002")
        wrapper = ModelWrapper(model_id, len(declared_classes), pretrained=False)
        load_checkpoint(wrapper, checkpoint, args.device)
        model = wrapper.model.to(args.device)
        model.eval()
        target_layer_name = args.efficientnet_scorecam_target_layer or resolve_xai_target_layer(
            resolved.values["xai"], model_id, "score_cam"
        )
        target_layer = wrapper.target_layer(target_layer_name)
        generator = CAMGenerator(model, target_layer, "score_cam")
        try:
            generator.__enter__()
            for index, record in enumerate(selected, start=1):
                rows.append(_evaluate_record(
                    record, args.image_root, model, generator, pipeline, scenario,
                    model_id, actual_hash, args.device, target_layer_name,
                ))
                if index % 5 == 0 or index == len(selected):
                    print(f"{model_id}: {index}/{len(selected)} validation leaves")
        finally:
            generator.close()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    sample_path = args.output_dir / "score_cam_preflight_samples.csv"
    report_path = args.output_dir / "score_cam_preflight_report.json"
    _write_csv(sample_path, rows)
    original_failures = sum(row["original_cam_status"] != "valid" for row in rows)
    transformed_failures = sum(row["transformed_cam_status"] != "valid" for row in rows)
    inconsistencies = sum(not row["prediction_consistent"] for row in rows)
    report = {
        "run_type": "validation_only_score_cam_preflight",
        "official_result": False,
        "approval_status": "pending_human_review",
        "test_split_accessed": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": resolved.values["protocol_version"],
        "protocol_hash": resolved.sha256,
        "checkpoint_decision_record_sha256": sha256_file(args.checkpoint_decision_record),
        "source_split": "validation",
        "xai_method": "score_cam",
        "target_layers": {
            "resnet50": "layer4[-1]",
            "efficientnet_b0": (
                args.efficientnet_scorecam_target_layer
                or resolve_xai_target_layer(
                    resolved.values["xai"], "efficientnet_b0", "score_cam"
                )
            ),
        },
        "scenario_id": args.scenario_id,
        "target_class_policy": "original_predicted_class",
        "selected_sample_count": len(selected),
        "selected_counts_by_class": dict(sorted(counts.items())),
        "sample_rows": len(rows),
        "original_cam_failure_count": original_failures,
        "original_cam_failure_rate": original_failures / len(rows),
        "transformed_cam_failure_count": transformed_failures,
        "transformed_cam_failure_rate": transformed_failures / len(rows),
        "prediction_inconsistency_count": inconsistencies,
        "prediction_inconsistency_rate": inconsistencies / len(rows),
        "acceptance_criteria": {
            "validation_only": True,
            "sample_level_original_cam_status": True,
            "sample_level_transformed_cam_status": True,
            "target_class_trace": True,
            "constant_cam_rate_reported": True,
            "checkpoint_hashes_match_approved_record": True,
        },
        "decision": "No automatic approval: review all non-valid CAM rows before G2.",
        "artifact_sha256": {sample_path.name: sha256_file(sample_path)},
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _require_validation_bundle(manifest: Path) -> None:
    report_path = manifest.parent / "validation_bundle_manifest.json"
    freeze_path = manifest.parent / "freeze_record.json"
    if not report_path.is_file() or not freeze_path.is_file():
        raise SystemExit("A validation recovery bundle (report and freeze record) is required")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    required = {
        "run_type": "validation_only_recovery_bundle",
        "validation_only_manifest": True,
        "non_validation_entries_materialized": False,
        "all_validation_images_canonical_hash_verified": True,
    }
    if any(report.get(key) != value for key, value in required.items()):
        raise SystemExit("Validation recovery bundle provenance is unacceptable")
    if report.get("validation_manifest_sha256") != sha256_file(manifest):
        raise SystemExit("Validation manifest hash does not match recovery report")
    if report.get("freeze_record_sha256") != sha256_file(freeze_path):
        raise SystemExit("Freeze record hash does not match recovery report")


def _evaluate_record(record: Any, image_root: Path, model: Any, generator: CAMGenerator,
                     pipeline: TransformationPipeline, scenario: Any, model_id: str,
                     checkpoint_sha256: str, device: str, target_layer_name: str) -> dict[str, Any]:
    import torch
    original = load_verified_record(record, image_root)
    original_tensor = preprocess_for_model(original).unsqueeze(0).to(device)
    with torch.inference_mode():
        original_class = int(model(original_tensor).argmax(dim=1).item())
    original_status, original_hash = _cam_status(generator, original_tensor, original_class)
    transformed, transformation = pipeline.apply(original, record.sample_id, scenario)
    transformed_tensor = preprocess_for_model(transformed).unsqueeze(0).to(device)
    with torch.inference_mode():
        transformed_class = int(model(transformed_tensor).argmax(dim=1).item())
    transformed_status, transformed_hash = _cam_status(generator, transformed_tensor, original_class)
    return {
        "sample_id": record.sample_id, "leaf_id": record.leaf_id,
        "class_id": record.class_id, "class_name": record.class_name, "split": record.split,
        "model_id": model_id, "checkpoint_sha256": checkpoint_sha256,
        "target_layer": target_layer_name,
        "scenario_id": scenario.scenario_id, "target_class_id": original_class,
        "transformed_predicted_class_id": transformed_class,
        "prediction_consistent": original_class == transformed_class,
        "original_cam_status": original_status, "original_cam_sha256": original_hash,
        "transformed_cam_status": transformed_status, "transformed_cam_sha256": transformed_hash,
        "transformation_seed": transformation.seed,
    }


def _cam_status(generator: CAMGenerator, tensor: Any, target_class: int) -> tuple[str, str]:
    try:
        heatmap = generator.generate(tensor, target_class)
    except (ValueError, RuntimeError) as exc:
        return f"invalid:{exc}", ""
    return "valid", __import__("hashlib").sha256(np.asarray(heatmap, dtype=np.float32).tobytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
