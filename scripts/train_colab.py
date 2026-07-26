"""Train one backbone from a canonical manifest in local or Colab storage."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path

from plantxai_stability import __version__
from plantxai_stability.config import load_protocol
from plantxai_stability.data.freeze import require_frozen_artifacts
from plantxai_stability.data.loader import PlantDataset, build_torch_dataloader
from plantxai_stability.data.manifest import read_manifest_csv
from plantxai_stability.provenance import sha256_file
from plantxai_stability.training import train_model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", choices=["resnet50", "efficientnet_b0"], required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-draft-training", action="store_true", help="Debug-only run; labels outputs non-official")
    args = parser.parse_args()
    resolved = load_protocol(args.protocol)
    governance = resolved.values.get("governance", {})
    official = bool(
        resolved.values.get("frozen")
        and governance.get("G0B_PROTOCOL_FREEZE_READY") == "pass"
        and governance.get("official_training_allowed") is True
    )
    if not official and not args.allow_draft_training:
        raise SystemExit(
            "Training blocked: official training requires frozen G0B approval. "
            "Use --allow-draft-training only for non-official debugging."
        )
    freeze_record = require_frozen_artifacts(args.manifest)
    if official and freeze_record.get("protocol_hash") != resolved.sha256:
        raise SystemExit("Official training blocked: freeze record protocol hash mismatch")
    records = read_manifest_csv(args.manifest)
    train_records = [record for record in records if record.split == "train"]
    validation_records = [record for record in records if record.split == "validation"]
    if not train_records or not validation_records:
        raise SystemExit("Manifest must contain non-empty train and validation splits")
    batch_size = int(resolved.values["training"]["batch_size"])
    num_classes = len(resolved.values["dataset"]["classes"])
    train_loader = build_torch_dataloader(
        PlantDataset(train_records, args.image_root, expected_split="train", num_classes=num_classes),
        batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        seed=resolved.seed,
    )
    validation_loader = build_torch_dataloader(
        PlantDataset(
            validation_records,
            args.image_root,
            expected_split="validation",
            num_classes=num_classes,
        ),
        batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=resolved.seed,
    )
    manifest_sha256 = sha256_file(args.manifest)
    config = {
        **resolved.values["training"],
        "seed": resolved.seed,
        "config_hash": resolved.sha256,
        "manifest_sha256": manifest_sha256,
        "num_classes": len(resolved.values["dataset"]["classes"]),
        "class_names": list(resolved.values["dataset"]["classes"]),
        "software_version": __version__,
        "git_commit": _git_revision(),
    }
    if args.device:
        config["device"] = args.device
    evidence = train_model(
        args.model_id,
        train_loader,
        validation_loader,
        config,
        args.output_dir,
        resume=args.resume,
    )
    try:
        import torch
        import torchvision

        torch_version = torch.__version__
        torchvision_version = torchvision.__version__
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:  # pragma: no cover
        torch_version = None
        torchvision_version = None
        gpu = None
    payload = {
        **evidence.__dict__,
        "run_type": "official_checkpoint_selection" if official else "draft_training",
        "official": official,
        "protocol_hash": resolved.sha256,
        "manifest_sha256": manifest_sha256,
        "freeze_record_protocol_hash": freeze_record.get("protocol_hash"),
        "freeze_record_sha256": sha256_file(args.manifest.parent / "freeze_record.json"),
        "train_sample_count": len(train_records),
        "validation_sample_count": len(validation_records),
        "test_split_accessed": False,
        "num_workers": args.num_workers,
        "resumed": args.resume,
        "python_platform": platform.platform(),
        "torch_version": torch_version,
        "torchvision_version": torchvision_version,
        "gpu": gpu,
        "software_version": __version__,
        "git_commit": config["git_commit"],
    }
    (args.output_dir / f"{args.model_id}_checkpoint_evidence.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


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
