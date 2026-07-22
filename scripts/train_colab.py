"""Train one backbone from a canonical manifest in local or Colab storage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plantxai_stability.config import load_protocol
from plantxai_stability.data.freeze import require_frozen_artifacts
from plantxai_stability.data.loader import PlantDataset, build_torch_dataloader
from plantxai_stability.data.manifest import read_manifest_csv
from plantxai_stability.training import train_model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", choices=["resnet50", "efficientnet_b0"], required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--allow-draft-training", action="store_true", help="Debug-only run; labels outputs non-official")
    args = parser.parse_args()
    resolved = load_protocol(args.protocol)
    governance = resolved.values.get("governance", {})
    official = bool(resolved.values.get("frozen")) and governance.get("G0B_PROTOCOL_FREEZE_READY") == "pass"
    if not official and not args.allow_draft_training:
        raise SystemExit("Training blocked: protocol is not frozen. Use --allow-draft-training only for non-official debugging.")
    records = read_manifest_csv(args.manifest)
    if official:
        require_frozen_artifacts(args.manifest)
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
        seed=resolved.seed,
    )
    config = {**resolved.values["training"], "seed": resolved.seed, "config_hash": resolved.sha256, "num_classes": len(resolved.values["dataset"]["classes"])}
    if args.device:
        config["device"] = args.device
    evidence = train_model(args.model_id, train_loader, validation_loader, config, args.output_dir)
    payload = {**evidence.__dict__, "official": official, "protocol_hash": resolved.sha256}
    (args.output_dir / f"{args.model_id}_checkpoint_evidence.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
