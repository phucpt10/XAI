"""Evaluate a validation-selected checkpoint on the frozen test split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from plantxai_stability.config import load_protocol
from plantxai_stability.data.freeze import require_frozen_artifacts
from plantxai_stability.data.loader import PlantDataset, build_torch_dataloader
from plantxai_stability.data.manifest import read_manifest_csv
from plantxai_stability.models import ModelWrapper
from plantxai_stability.provenance import sha256_file
from plantxai_stability.training import load_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-id", choices=["resnet50", "efficientnet_b0"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    resolved = load_protocol(args.protocol)
    if not resolved.values.get("governance", {}).get(
        "official_test_evaluation_allowed", False
    ):
        raise SystemExit("Baseline test evaluation requires G2 official-test approval")
    require_frozen_artifacts(args.manifest)
    records = [record for record in read_manifest_csv(args.manifest) if record.split == "test"]
    if not records:
        raise SystemExit("Manifest has no test records")
    wrapper = ModelWrapper(args.model_id, len(resolved.values["dataset"]["classes"]), pretrained=False)
    manifest_sha256 = sha256_file(args.manifest)
    load_checkpoint(
        wrapper,
        args.checkpoint,
        args.device,
        expected_protocol_hash=resolved.sha256,
        expected_manifest_sha256=manifest_sha256,
    )
    checkpoint_sha256 = sha256_file(args.checkpoint)
    dataset = PlantDataset(
        records,
        args.image_root,
        expected_split="test",
        num_classes=len(resolved.values["dataset"]["classes"]),
    )
    loader = build_torch_dataloader(
        dataset,
        int(resolved.values["training"]["batch_size"]),
        shuffle=False,
        seed=resolved.seed,
    )
    try:
        import torch
        from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Install the [ml] dependencies before baseline evaluation") from exc
    model = wrapper.model.to(args.device)
    rows = []
    truth: list[int] = []
    predicted: list[int] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["model_tensor"].to(args.device))
            probabilities = torch.softmax(logits, dim=1)
            confidence, classes = probabilities.max(dim=1)
            for sample_id, label, pred, conf in zip(batch["sample_id"], batch["label"].tolist(), classes.tolist(), confidence.tolist()):
                truth.append(int(label))
                predicted.append(int(pred))
                rows.append({"sample_id": sample_id, "model_id": args.model_id, "true_class": int(label), "predicted_class": int(pred), "confidence": float(conf), "checkpoint_sha256": checkpoint_sha256})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "baseline_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics = {"model_id": args.model_id, "accuracy": float(accuracy_score(truth, predicted)), "macro_precision": float(precision_score(truth, predicted, average="macro", zero_division=0)), "macro_recall": float(recall_score(truth, predicted, average="macro", zero_division=0)), "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)), "confusion_matrix": confusion_matrix(truth, predicted).tolist()}
    (args.output_dir / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
