"""Run one frozen-model joint robustness/XAI evaluation.

This is intentionally a single-model runner so Colab jobs can be resumed and
audited independently. Run once per model and merge only by explicit keys.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from plantxai_stability.config import load_protocol
from plantxai_stability.data.freeze import require_frozen_artifacts
from plantxai_stability.data.loader import load_verified_record, preprocess_for_model
from plantxai_stability.data.manifest import read_manifest_csv
from plantxai_stability.inference import infer_one
from plantxai_stability.models import ModelWrapper
from plantxai_stability.provenance import RunContext, sha256_bytes, sha256_file
from plantxai_stability.statistics import heatmap_metrics
from plantxai_stability.training import load_checkpoint
from plantxai_stability.transformations import TransformationPipeline, scenario_grid
from plantxai_stability.xai import CAMGenerator, forward_align_heatmap


METRIC_FIELDS = (
    "ssim",
    "pearson",
    "cosine",
    "topk_iou_10",
    "topk_iou_20",
    "topk_iou_30",
    "valid_pixel_count",
    "valid_pixel_fraction",
    "ssim_valid_pixel_count",
    "ssim_valid_pixel_fraction",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-id", choices=["resnet50", "efficientnet_b0"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    resolved = load_protocol(args.protocol)
    if not resolved.values.get("governance", {}).get(
        "official_test_evaluation_allowed", False
    ):
        raise SystemExit("Joint evaluation requires G2 official-test approval")
    require_frozen_artifacts(args.manifest)
    records = [record for record in read_manifest_csv(args.manifest) if record.split == "test"]
    if not records:
        raise SystemExit("Manifest has no test records")
    wrapper = ModelWrapper(args.model_id, len(resolved.values["dataset"]["classes"]), pretrained=False)
    manifest_sha256 = sha256_file(args.manifest)
    checkpoint_payload = load_checkpoint(
        wrapper,
        args.checkpoint,
        args.device,
        expected_protocol_hash=resolved.sha256,
        expected_manifest_sha256=manifest_sha256,
    )
    model = wrapper.model.to(args.device)
    target_layer = wrapper.target_layer()
    context = RunContext.create(resolved.values["protocol_version"], resolved.sha256, resolved.sha256, resolved.seed, args.run_id)
    output = args.output_dir / args.run_id
    output.mkdir(parents=True, exist_ok=True)
    pipeline = TransformationPipeline(resolved.seed, resolved.values["transformations"]["parameters"])
    scenarios = scenario_grid(resolved.values["transformations"]["parameters"])
    methods = resolved.values["xai"]["methods"]
    xai_policy = resolved.values["xai"]
    checkpoint_hash = sha256_file(args.checkpoint)
    original_cache = {}
    prediction_rows = []
    joint_rows = []
    for record in records:
        pixels = load_verified_record(record, args.image_root)
        original_cache[record.sample_id] = (pixels, infer_one(model, pixels, record, args.model_id, args.run_id, "original", checkpoint_hash, args.device))
    for scenario in scenarios:
        for record in records:
            original_pixels, original_prediction = original_cache[record.sample_id]
            transformed_pixels, transformation_record = pipeline.apply(original_pixels, record.sample_id, scenario)
            transformed_prediction = infer_one(model, transformed_pixels, record, args.model_id, args.run_id, scenario.scenario_id, checkpoint_hash, args.device)
            consistent = original_prediction.predicted_class == transformed_prediction.predicted_class
            prediction_rows.append({**asdict(original_prediction), "transformed_predicted_class": transformed_prediction.predicted_class, "transformed_confidence": transformed_prediction.confidence, "is_consistent": consistent, "confidence_delta": transformed_prediction.confidence - original_prediction.confidence, "absolute_confidence_delta": abs(transformed_prediction.confidence - original_prediction.confidence), "transformation": scenario.transformation, "severity": scenario.severity, "rotation_prediction_claim_scope": xai_policy["rotation_prediction_claim_scope"] if scenario.transformation == "rotation" else "not_applicable", "valid_mask_sha256": transformation_record.valid_mask_sha256 or ""})
            if not consistent:
                joint_rows.append(_joint_row(args.run_id, args.model_id, record.sample_id, record.leaf_id, scenario.scenario_id, "", original_prediction.predicted_class, False, None, "", "prediction_inconsistent"))
                continue
            original_tensor = preprocess_for_model(original_pixels).unsqueeze(0).to(args.device)
            transformed_tensor = preprocess_for_model(transformed_pixels).unsqueeze(0).to(args.device)
            for method in methods:
                generator = CAMGenerator(model, target_layer, method)
                try:
                    original_cam = generator.generate(
                        original_tensor, original_prediction.predicted_class
                    )
                    transformed_cam = generator.generate(
                        transformed_tensor, original_prediction.predicted_class
                    )
                    aligned_original, mask = forward_align_heatmap(
                        original_cam, transformation_record.forward_metadata
                    )
                    metrics = heatmap_metrics(
                        aligned_original,
                        transformed_cam,
                        mask,
                        ssim_window_size=int(xai_policy["ssim_window_size"]),
                        topk_values=tuple(xai_policy["topk_iou_sensitivity"]),
                    )
                    mask_hash = sha256_bytes(mask.astype("uint8").tobytes())
                except ValueError as exc:
                    joint_rows.append(_joint_row(args.run_id, args.model_id, record.sample_id, record.leaf_id, scenario.scenario_id, method, original_prediction.predicted_class, True, None, "", f"invalid_cam_or_metric:{exc}"))
                    continue
                joint_rows.append(_joint_row(args.run_id, args.model_id, record.sample_id, record.leaf_id, scenario.scenario_id, method, original_prediction.predicted_class, True, metrics, mask_hash, ""))
    _write_csv(output / "prediction_results.csv", prediction_rows)
    _write_csv(output / "joint_results.csv", joint_rows)
    (output / "run_manifest.json").write_text(json.dumps({"context": context.to_dict(), "protocol_hash": resolved.sha256, "manifest": str(args.manifest), "manifest_sha256": sha256_file(args.manifest), "checkpoint": str(args.checkpoint), "checkpoint_sha256": checkpoint_hash, "scenario_count": len(scenarios), "xai_methods": methods, "xai_alignment_policy": {key: xai_policy[key] for key in ("alignment_policy", "valid_region_policy", "validity_threshold", "ssim_window_size", "topk_iou_primary", "topk_iou_sensitivity", "target_class_policy", "prediction_consistency_required", "rotation_prediction_claim_scope")}, "checkpoint_payload": checkpoint_payload}, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"run_id": args.run_id, "prediction_rows": len(prediction_rows), "joint_rows": len(joint_rows), "scenario_count": len(scenarios)}, indent=2))
    return 0


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _joint_row(
    run_id: str,
    model_id: str,
    sample_id: str,
    leaf_id: str,
    scenario_id: str,
    method: str,
    target_class: int,
    is_consistent: bool,
    metrics: dict[str, float] | None,
    valid_mask_sha256: str,
    exclusion_reason: str,
) -> dict[str, object]:
    values: dict[str, object] = {
        "run_id": run_id,
        "model_id": model_id,
        "sample_id": sample_id,
        "leaf_id": leaf_id,
        "scenario_id": scenario_id,
        "xai_method": method,
        "target_class": target_class,
        "is_consistent": is_consistent,
        "alignment_policy": "forward_align_original_cam",
        "valid_mask_sha256": valid_mask_sha256,
        "exclusion_reason": exclusion_reason,
    }
    values.update(
        {field: "" if metrics is None else metrics[field] for field in METRIC_FIELDS}
    )
    return values


if __name__ == "__main__":
    raise SystemExit(main())
