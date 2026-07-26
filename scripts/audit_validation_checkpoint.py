"""Audit one validation-selected checkpoint without accessing official test pixels."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from plantxai_stability import __version__
from plantxai_stability.checkpoint_audit import (
    classification_audit,
    validate_training_evidence,
)
from plantxai_stability.config import load_protocol
from plantxai_stability.data.freeze import require_frozen_artifacts
from plantxai_stability.data.loader import PlantDataset, build_torch_dataloader
from plantxai_stability.data.manifest import read_manifest_csv
from plantxai_stability.governance import approved_checkpoint_lineage
from plantxai_stability.models import ModelWrapper
from plantxai_stability.provenance import sha256_bytes, sha256_file
from plantxai_stability.training import load_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-evidence", type=Path, required=True)
    parser.add_argument("--checkpoint-decision-record", type=Path)
    parser.add_argument(
        "--model-id", choices=["resnet50", "efficientnet_b0"], required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit("Audit output already exists; use a new versioned directory")
    resolved = load_protocol(args.protocol)
    governance = resolved.values["governance"]
    if not (
        resolved.values["frozen"]
        and governance["G0B_PROTOCOL_FREEZE_READY"] == "pass"
        and governance["official_training_allowed"] is True
    ):
        raise SystemExit("Validation checkpoint audit requires frozen G0B approval")
    freeze_record = require_frozen_artifacts(args.manifest)
    summary_path = args.manifest.parent / "split_summary.json"
    if not summary_path.is_file():
        raise SystemExit("Validation audit blocked: split_summary.json is missing")
    expected_summary_hash = freeze_record.get("artifact_sha256", {}).get(
        summary_path.name
    )
    if expected_summary_hash != sha256_file(summary_path):
        raise SystemExit("Validation audit blocked: split summary hash mismatch")
    split_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    all_records = read_manifest_csv(args.manifest)
    validation_records = [record for record in all_records if record.split == "validation"]
    expected_validation_count = int(split_summary["counts_by_split"]["validation"])
    expected_train_count = int(split_summary["counts_by_split"]["train"])
    if len(validation_records) != expected_validation_count or not validation_records:
        raise SystemExit("Validation audit blocked: validation split count mismatch")
    class_names = list(resolved.values["dataset"]["classes"])
    class_count = len(class_names)
    if any(record.class_name != class_names[record.class_id] for record in validation_records):
        raise SystemExit("Validation audit blocked: manifest class mapping mismatch")
    manifest_sha256 = sha256_file(args.manifest)
    checkpoint_sha256 = sha256_file(args.checkpoint)
    freeze_record_sha256 = sha256_file(
        args.manifest.parent / "freeze_record.json"
    )
    checkpoint_protocol_hash = resolved.sha256
    checkpoint_decision_id = None
    if freeze_record.get("protocol_hash") != resolved.sha256:
        if args.checkpoint_decision_record is None:
            raise SystemExit(
                "Validation audit blocked: post-G1 audit requires the approved "
                "checkpoint Decision Record"
            )
        checkpoint_decision = yaml.safe_load(
            args.checkpoint_decision_record.read_text(encoding="utf-8")
        )
        lineage = approved_checkpoint_lineage(
            checkpoint_decision,
            governance,
            model_id=args.model_id,
            declared_models=resolved.values["models"],
            checkpoint_sha256=checkpoint_sha256,
            manifest_sha256=manifest_sha256,
            freeze_record_sha256=freeze_record_sha256,
        )
        checkpoint_protocol_hash = str(lineage["training_protocol_hash"])
        checkpoint_decision_id = str(lineage["decision_id"])
        if freeze_record.get("protocol_hash") != checkpoint_protocol_hash:
            raise SystemExit(
                "Validation audit blocked: freeze record does not match approved "
                "checkpoint training lineage"
            )
    wrapper = ModelWrapper(args.model_id, class_count, pretrained=False)
    checkpoint_payload = load_checkpoint(
        wrapper,
        args.checkpoint,
        args.device,
        expected_protocol_hash=checkpoint_protocol_hash,
        expected_manifest_sha256=manifest_sha256,
    )
    training_evidence = json.loads(args.checkpoint_evidence.read_text(encoding="utf-8"))
    validate_training_evidence(
        training_evidence,
        checkpoint_payload,
        model_id=args.model_id,
        protocol_hash=checkpoint_protocol_hash,
        manifest_sha256=manifest_sha256,
        checkpoint_sha256=checkpoint_sha256,
        freeze_record_sha256=freeze_record_sha256,
        seed=resolved.seed,
        class_names=class_names,
        expected_train_count=expected_train_count,
        expected_validation_count=expected_validation_count,
    )
    dataset = PlantDataset(
        validation_records,
        args.image_root,
        expected_split="validation",
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
        raise SystemExit("Install the [ml] dependencies before checkpoint audit") from exc
    model = wrapper.model.to(args.device)
    model.eval()
    rows: list[dict[str, Any]] = []
    truth: list[int] = []
    predicted: list[int] = []
    probability_rows: list[list[float]] = []
    with torch.inference_mode():
        for batch in loader:
            logits = model(batch["model_tensor"].to(args.device))
            if logits.ndim != 2 or logits.shape[1] != class_count:
                raise SystemExit("Validation audit blocked: invalid model output shape")
            probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()
            if not np.isfinite(probabilities).all():
                raise SystemExit("Validation audit blocked: non-finite probabilities")
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
                    "checkpoint_sha256": checkpoint_sha256,
                }
                row.update(
                    {
                        f"probability_class_{class_id}": float(probability[class_id])
                        for class_id in range(class_count)
                    }
                )
                rows.append(row)
    expected_ids = sorted(record.sample_id for record in validation_records)
    observed_ids = [str(row["sample_id"]) for row in rows]
    identity_order_reproducible = observed_ids == expected_ids
    sample_coverage_exact = len(rows) == expected_validation_count and set(
        observed_ids
    ) == set(expected_ids)
    if not sample_coverage_exact or len(observed_ids) != len(set(observed_ids)):
        raise SystemExit("Validation audit blocked: sample identity coverage mismatch")
    if any(row["split"] != "validation" for row in rows):
        raise SystemExit("Validation audit blocked: non-validation sample was loaded")
    metrics = classification_audit(
        truth, predicted, np.asarray(probability_rows), class_names
    )
    selected_metric_matches_training = bool(
        np.isclose(
            metrics["macro_f1"],
            float(checkpoint_payload["validation_macro_f1"]),
            atol=1e-12,
            rtol=0.0,
        )
    )
    if not selected_metric_matches_training:
        raise SystemExit("Validation audit blocked: selected macro-F1 is not reproducible")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    predictions_csv = args.output_dir / "validation_predictions.csv"
    predictions_parquet = args.output_dir / "validation_predictions.parquet"
    per_class_csv = args.output_dir / "validation_per_class_metrics.csv"
    confusion_csv = args.output_dir / "validation_confusion_matrix.csv"
    report_path = args.output_dir / "validation_checkpoint_audit.json"
    _write_csv(predictions_csv, rows)
    pd.DataFrame(rows).to_parquet(predictions_parquet, index=False)
    _write_csv(per_class_csv, metrics["per_class"])
    _write_confusion_matrix(confusion_csv, metrics["confusion_matrix"], class_names)
    artifact_sha256 = {
        path.name: sha256_file(path)
        for path in (
            predictions_csv,
            predictions_parquet,
            per_class_csv,
            confusion_csv,
        )
    }
    report = {
        "run_type": "official_validation_checkpoint_audit",
        "approval_status": "pending_g1_human_review",
        "official_checkpoint_selection_evidence": True,
        "official_test_result": False,
        "source_split": "validation",
        "test_split_accessed": False,
        "model_id": args.model_id,
        "protocol_hash": checkpoint_protocol_hash,
        "governance_protocol_hash": resolved.sha256,
        "checkpoint_decision_record": checkpoint_decision_id,
        "manifest_sha256": manifest_sha256,
        "freeze_record_sha256": freeze_record_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_evidence_sha256": sha256_file(args.checkpoint_evidence),
        "checkpoint_source_git_commit": training_evidence.get("git_commit"),
        "audit_git_commit": _git_revision(),
        "best_epoch": int(checkpoint_payload["best_epoch"]),
        "selected_validation_macro_f1": float(
            checkpoint_payload["validation_macro_f1"]
        ),
        "sample_ids_sha256": sha256_bytes(
            json.dumps(observed_ids, separators=(",", ":")).encode("utf-8")
        ),
        "class_names": class_names,
        "metrics": metrics,
        "acceptance_criteria": {
            "validation_only": True,
            "test_split_not_accessed": True,
            "checkpoint_lineage_matches": True,
            "freeze_protocol_hash_matches": True,
            "sample_coverage_exact": sample_coverage_exact,
            "sample_identity_order_reproducible": identity_order_reproducible,
            "all_classes_have_support": all(
                int(item["support"]) > 0 for item in metrics["per_class"]
            ),
            "confusion_matrix_reconciles": sum(
                sum(int(value) for value in row)
                for row in metrics["confusion_matrix"]
            )
            == expected_validation_count,
            "selected_macro_f1_reproduced": selected_metric_matches_training,
            "metrics_finite": True,
        },
        "artifact_sha256": artifact_sha256,
        "runtime": {
            "python_platform": platform.platform(),
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "device": str(args.device),
            "gpu": (
                torch.cuda.get_device_name(0)
                if str(args.device).startswith("cuda") and torch.cuda.is_available()
                else None
            ),
            "num_workers": args.num_workers,
            "software_version": __version__,
        },
    }
    if not all(report["acceptance_criteria"].values()):
        raise SystemExit("Validation checkpoint audit quality gate failed")
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Validation checkpoint audit: PASS\nReport: {report_path}")
    return 0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_confusion_matrix(
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
