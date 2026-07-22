"""PyTorch model wrappers and training contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CheckpointEvidence:
    model_id: str
    checkpoint_path: str
    checkpoint_sha256: str
    validation_metric: float
    config_hash: str
    seed: int


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
            weights = models.ResNet50_Weights.DEFAULT if self.pretrained else None
            model = models.resnet50(weights=weights)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        elif self.model_id == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if self.pretrained else None
            model = models.efficientnet_b0(weights=weights)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, self.num_classes)
        else:
            raise ValueError(f"Unsupported model_id: {self.model_id}")
        return model

    def target_layer(self) -> Any:
        if self.model_id == "resnet50":
            return self.model.layer4[-1]
        if self.model_id == "efficientnet_b0":
            return self.model.features[-1]
        raise ValueError(f"Unsupported model_id: {self.model_id}")

    def eval(self) -> None:
        self.model.eval()


def train_one_model(*, model_id: str, train_loader: Any, validation_loader: Any, config: dict[str, Any], output_dir: str | Path) -> CheckpointEvidence:
    """Train a model with validation-only checkpoint selection.

    This function intentionally requires PyTorch and a real dataset/checkpoint
    configuration. It never invents metrics when dependencies or data are absent.
    """
    try:
        import torch
        import torch.nn as nn
        from sklearn.metrics import f1_score
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch and scikit-learn are required for training") from exc
    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)
    wrapper = ModelWrapper(model_id, num_classes=int(config.get("num_classes", 5)), pretrained=bool(config.get("pretrained", True)))
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = wrapper.model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("learning_rate", 1e-3)), weight_decay=float(config.get("weight_decay", 1e-4)))
    criterion = nn.CrossEntropyLoss()
    best_f1 = float("-inf")
    best_state: dict[str, Any] | None = None
    patience = int(config.get("early_stopping_patience", 8))
    stale = 0
    max_epochs = int(config.get("max_epochs", 50))
    for _epoch in range(max_epochs):
        model.train()
        for batch in train_loader:
            inputs = batch["model_tensor"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
        model.eval()
        truth: list[int] = []
        pred: list[int] = []
        with torch.no_grad():
            for batch in validation_loader:
                logits = model(batch["model_tensor"].to(device))
                pred.extend(logits.argmax(dim=1).cpu().tolist())
                truth.extend(batch["label"].tolist())
        score = float(f1_score(truth, pred, average="macro", zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("Training produced no validation checkpoint")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / f"{model_id}_best.pt"
    torch.save(best_state, checkpoint)
    from plantxai_stability.provenance import sha256_file
    return CheckpointEvidence(model_id, str(checkpoint), sha256_file(checkpoint), best_f1, str(config.get("config_hash", "")), seed)
