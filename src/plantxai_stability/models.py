"""PyTorch model wrappers and training contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CheckpointEvidence:
    model_id: str
    checkpoint_path: str
    checkpoint_sha256: str
    validation_metric: float
    config_hash: str
    seed: int
    best_epoch: int = 0
    epochs_completed: int = 0
    history_path: str = ""
    latest_checkpoint_path: str = ""
    manifest_sha256: str = ""


class ModelWrapper:
    def __init__(self, model_id: str, num_classes: int = 5, pretrained: bool = True) -> None:
        self.model_id = model_id
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.model: Any = self._build()

    def _build(self) -> Any:
        try:
            import torch.nn as nn
            from torchvision import models
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("torch and torchvision are required for model construction") from exc
        if self.model_id == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if self.pretrained else None
            model = models.resnet50(weights=weights)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        elif self.model_id == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if self.pretrained else None
            model = models.efficientnet_b0(weights=weights)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, self.num_classes)
        else:
            raise ValueError(f"Unsupported model_id: {self.model_id}")
        return model

    def target_layer(self, specification: str | None = None) -> Any:
        default = (
            "layer4[-1]" if self.model_id == "resnet50" else "features[-1]"
        )
        requested = specification or default
        if self.model_id == "resnet50" and requested == "layer4[-1]":
            return self.model.layer4[-1]
        if self.model_id == "efficientnet_b0" and requested == "features[-1]":
            return self.model.features[-1]
        if self.model_id == "efficientnet_b0" and requested == "features[-2]":
            return self.model.features[-2]
        raise ValueError(
            f"Unsupported target layer {requested!r} for {self.model_id}"
        )

    def eval(self) -> None:
        self.model.eval()
