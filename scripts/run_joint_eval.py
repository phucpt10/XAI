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
from plantxai_stability.provenance import RunContext, sha256_file
from plantxai_stability.statistics import heatmap_metrics
from plantxai_stability.training import load_checkpoint
from plantxai_stability.transformations import TransformationPipeline, scenario_grid
from plantxai_stability.xai import CAMGenerator, inverse_align_heatmap, normalize_heatmap


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
    if not resolved.values.get("frozen", False) or resolved.values.get("governance", {}).get("G0B_PROTOCOL_FREEZE_READY") != "pass":
        raise SystemExit("Joint evaluation requires a frozen protocol with G0B PASS")
    require_frozen_artifacts(args.manifest)
    records = [record for record in read_manifest_csv(args.manifest) if record.split == "test"]
    if not records:
        raise SystemExit("Manifest has no test records")
    wrapper = ModelWrapper(args.model_id, len(resolved.values["dataset"]["classes"]), pretrained=False)
    checkpoint_payload = load_checkpoint(wrapper, args.checkpoint, args.device)
    model = wrapper.model.to(args.device)
    target_layer = wrapper.target_layer()
    context = RunContext.create(resolved.values["protocol_version"], resolved.sha256, resolved.sha256, resolved.seed, args.run_id)
    output = args.output_dir / args.run_id
    output.mkdir(parents=True, exist_ok=True)
    pipeline = TransformationPipeline(resolved.seed, resolved.values["transformations"]["parameters"])
    scenarios = scenario_grid(resolved.values["transformations"]["parameters"])
    methods = resolved.values["xai"]["methods"]
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
            prediction_rows.append({**asdict(original_prediction), "transformed_predicted_class": transformed_prediction.predicted_class, "transformed_confidence": transformed_prediction.confidence, "is_consistent": consistent, "confidence_delta": transformed_prediction.confidence - original_prediction.confidence, "absolute_confidence_delta": abs(transformed_prediction.confidence - original_prediction.confidence), "transformation": scenario.transformation, "severity": scenario.severity})
            if not consistent:
                joint_rows.append({"run_id": args.run_id, "model_id": args.model_id, "sample_id": record.sample_id, "leaf_id": record.leaf_id, "scenario_id": scenario.scenario_id, "xai_method": "", "target_class": original_prediction.predicted_class, "is_consistent": False, "ssim": "", "pearson": "", "cosine": "", "exclusion_reason": "prediction_inconsistent"})
                continue
            original_tensor = preprocess_for_model(original_pixels).unsqueeze(0).to(args.device)
            transformed_tensor = preprocess_for_model(transformed_pixels).unsqueeze(0).to(args.device)
            for method in methods:
                generator = CAMGenerator(model, target_layer, method)
                original_cam = generator.generate(original_tensor, original_prediction.predicted_class)
                transformed_cam = generator.generate(transformed_tensor, original_prediction.predicted_class)
                aligned, mask = inverse_align_heatmap(transformed_cam, transformation_record.inverse_metadata)
                normalized_original, original_quality = normalize_heatmap(original_cam)
                normalized_aligned, aligned_quality = normalize_heatmap(aligned)
                if not original_quality.valid or not aligned_quality.valid:
                    joint_rows.append({"run_id": args.run_id, "model_id": args.model_id, "sample_id": record.sample_id, "leaf_id": record.leaf_id, "scenario_id": scenario.scenario_id, "xai_method": method, "target_class": original_prediction.predicted_class, "is_consistent": True, "ssim": "", "pearson": "", "cosine": "", "exclusion_reason": original_quality.reason or aligned_quality.reason})
                    continue
                metrics = heatmap_metrics(normalized_original, normalized_aligned, mask)
                joint_rows.append({"run_id": args.run_id, "model_id": args.model_id, "sample_id": record.sample_id, "leaf_id": record.leaf_id, "scenario_id": scenario.scenario_id, "xai_method": method, "target_class": original_prediction.predicted_class, "is_consistent": True, **metrics, "exclusion_reason": ""})
    _write_csv(output / "prediction_results.csv", prediction_rows)
    _write_csv(output / "joint_results.csv", joint_rows)
    (output / "run_manifest.json").write_text(json.dumps({"context": context.to_dict(), "protocol_hash": resolved.sha256, "manifest": str(args.manifest), "manifest_sha256": sha256_file(args.manifest), "checkpoint": str(args.checkpoint), "checkpoint_sha256": checkpoint_hash, "scenario_count": len(scenarios), "xai_methods": methods, "checkpoint_payload": checkpoint_payload}, indent=2, default=str), encoding="utf-8")
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


if __name__ == "__main__":
    raise SystemExit(main())
