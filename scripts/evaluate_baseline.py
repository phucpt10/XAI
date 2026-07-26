"""Evaluate one approved checkpoint on official test after G2 authorization."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from plantxai_stability import __version__
from plantxai_stability.checkpoint_audit import classification_audit
from plantxai_stability.config import load_protocol
from plantxai_stability.data.loader import PlantDataset, build_torch_dataloader
from plantxai_stability.models import ModelWrapper
from plantxai_stability.provenance import sha256_file
from plantxai_stability.recovery import load_recovery_decision
from plantxai_stability.test_authorization import authorize_official_test_run
from plantxai_stability.training import load_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-decision-record", type=Path, required=True)
    parser.add_argument("--test-decision-record", type=Path, required=True)
    parser.add_argument("--g2-readiness-report", type=Path, required=True)
    parser.add_argument(
        "--model-id", choices=["resnet50", "efficientnet_b0"], required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    recovery_decision_path = (
        args.protocol.parent / "decision_records" / "DR-RECOVERY-001.yaml"
    )
    if recovery_decision_path.is_file():
        recovery_decision = load_recovery_decision(recovery_decision_path)
        if recovery_decision["recovery_policy"]["no_baseline_rerun"]:
            raise SystemExit(
                "Baseline rerun prohibited by DR-RECOVERY-001; preserve the "
                "two existing baseline reports"
            )
    if args.output_dir.exists():
        raise SystemExit("Baseline output exists; use a new immutable run directory")
    resolved = load_protocol(args.protocol)
    authorization = authorize_official_test_run(
        resolved,
        manifest_path=args.manifest,
        checkpoint_path=args.checkpoint,
        model_id=args.model_id,
        checkpoint_decision_path=args.checkpoint_decision_record,
        test_decision_path=args.test_decision_record,
        readiness_report_path=args.g2_readiness_report,
    )
    records = authorization["test_records"]
    class_names = list(resolved.values["dataset"]["classes"])
    class_count = len(class_names)
    wrapper = ModelWrapper(args.model_id, class_count, pretrained=False)
    load_checkpoint(
        wrapper,
        args.checkpoint,
        args.device,
        expected_protocol_hash=authorization["checkpoint_training_protocol_hash"],
        expected_manifest_sha256=authorization["manifest_sha256"],
    )
    dataset = PlantDataset(
        records,
        args.image_root,
        expected_split="test",
        num_classes=class_count,
    )
    loader = build_torch_dataloader(
        dataset,
        int(resolved.values["training"]["batch_size"]),
        shuffle=False,
        num_workers=args.num_workers,
        seed=resolved.seed,
    )
    try:
        import torch
        import torchvision
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Install the [ml] dependencies before baseline evaluation") from exc
    model = wrapper.model.to(args.device)
    model.eval()
    rows: list[dict[str, Any]] = []
    truth: list[int] = []
    predicted: list[int] = []
    probability_rows: list[list[float]] = []
    with torch.inference_mode():
        for batch in loader:
            logits = model(batch["model_tensor"].to(args.device))
            probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()
            predictions = probabilities.argmax(axis=1)
            labels = np.asarray(batch["label"].tolist(), dtype=np.int64)
            for index, sample_id in enumerate(batch["sample_id"]):
                label = int(labels[index])
                prediction = int(predictions[index])
                probability = probabilities[index]
                truth.append(label)
                predicted.append(prediction)
                probability_rows.append(probability.astype(float).tolist())
                row: dict[str, Any] = {
                    "sample_id": str(sample_id),
                    "leaf_id": str(batch["leaf_id"][index]),
                    "split": str(batch["split"][index]),
                    "true_class_id": label,
                    "true_class_name": class_names[label],
                    "predicted_class_id": prediction,
                    "predicted_class_name": class_names[prediction],
                    "confidence": float(probability[prediction]),
                    "correct": bool(label == prediction),
                    "model_id": args.model_id,
                    "checkpoint_sha256": authorization["checkpoint_sha256"],
                }
                row.update(
                    {
                        f"probability_class_{class_id}": float(probability[class_id])
                        for class_id in range(class_count)
                    }
                )
                rows.append(row)
    expected_ids = sorted(record.sample_id for record in records)
    observed_ids = [str(row["sample_id"]) for row in rows]
    if observed_ids != expected_ids or any(row["split"] != "test" for row in rows):
        raise SystemExit("Official baseline blocked: test identity coverage mismatch")
    metrics = classification_audit(
        truth, predicted, np.asarray(probability_rows), class_names
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = args.output_dir / "baseline_predictions.csv"
    per_class_path = args.output_dir / "baseline_per_class_metrics.csv"
    confusion_path = args.output_dir / "baseline_confusion_matrix.csv"
    report_path = args.output_dir / "baseline_metrics.json"
    _write_csv(predictions_path, rows)
    _write_csv(per_class_path, metrics["per_class"])
    _write_confusion(confusion_path, metrics["confusion_matrix"], class_names)
    report = {
        "run_type": "authorized_official_test_baseline",
        "official_test_result": True,
        "model_id": args.model_id,
        "campaign_id": authorization["campaign_id"],
        "authorization_decision_id": authorization["authorization_decision_id"],
        "governance_protocol_hash": resolved.sha256,
        "checkpoint_training_protocol_hash": authorization[
            "checkpoint_training_protocol_hash"
        ],
        "checkpoint_sha256": authorization["checkpoint_sha256"],
        "manifest_sha256": authorization["manifest_sha256"],
        "freeze_record_sha256": authorization["freeze_record_sha256"],
        "test_decision_record_sha256": authorization[
            "test_decision_record_sha256"
        ],
        "g2_readiness_report_sha256": authorization[
            "g2_readiness_report_sha256"
        ],
        "official_test_identity": authorization["test_identity"],
        "metrics": metrics,
        "artifact_sha256": {
            path.name: sha256_file(path)
            for path in (predictions_path, per_class_path, confusion_path)
        },
        "acceptance_criteria": {
            "g2_authorization_passed_before_pixel_access": True,
            "test_identity_coverage_exact": observed_ids == expected_ids,
            "all_classes_have_support": all(
                int(item["support"]) > 0 for item in metrics["per_class"]
            ),
            "metrics_finite": True,
            "immutable_output_directory": True,
        },
        "runtime": {
            "python_platform": platform.platform(),
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "device": args.device,
            "gpu": (
                torch.cuda.get_device_name(0)
                if args.device.startswith("cuda") and torch.cuda.is_available()
                else None
            ),
            "num_workers": args.num_workers,
            "software_version": __version__,
            "git_commit": _git_revision(),
        },
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Official baseline: PASS\nReport: {report_path}")
    print(f"Report SHA-256: {sha256_file(report_path)}")
    return 0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_confusion(
    path: Path, matrix: list[list[int]], class_names: list[str]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_class\\predicted_class", *class_names])
        for class_name, row in zip(class_names, matrix):
            writer.writerow([class_name, *row])


def _git_revision() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


if __name__ == "__main__":
    raise SystemExit(main())
