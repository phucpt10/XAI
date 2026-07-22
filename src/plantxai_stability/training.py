"""Reproducible training and inference loops for Colab or local GPU runs."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from plantxai_stability.models import CheckpointEvidence, ModelWrapper
from plantxai_stability.provenance import sha256_file


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        return


def train_model(model_id: str, train_loader: Any, validation_loader: Any, config: dict[str, Any], output_dir: str | Path) -> CheckpointEvidence:
    """Train and select by validation macro-F1; no test loader is accepted."""
    try:
        import torch
        from sklearn.metrics import f1_score
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the [ml] dependencies before training") from exc
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    wrapper = ModelWrapper(model_id, int(config.get("num_classes", 5)), bool(config.get("pretrained", True)))
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = wrapper.model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("learning_rate", 1e-3)), weight_decay=float(config.get("weight_decay", 1e-4)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config.get("max_epochs", 50)))
    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config.get("mixed_precision", False) and device.type == "cuda"))
    best_score = float("-inf")
    best_state: dict[str, Any] | None = None
    stale = 0
    history: list[dict[str, float]] = []
    patience = int(config.get("early_stopping_patience", 8))
    for epoch in range(int(config.get("max_epochs", 50))):
        model.train()
        train_loss = 0.0
        train_count = 0
        for batch in train_loader:
            inputs = batch["model_tensor"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                loss = criterion(model(inputs), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.detach().cpu()) * len(labels)
            train_count += len(labels)
        scheduler.step()
        validation_loss, validation_f1 = _evaluate(model, validation_loader, device, criterion, f1_score)
        history.append({"epoch": float(epoch + 1), "train_loss": train_loss / max(train_count, 1), "validation_loss": validation_loss, "validation_macro_f1": validation_f1})
        if validation_f1 > best_score:
            best_score = validation_f1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("No validation checkpoint was produced")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / f"{model_id}_best.pt"
    torch.save({"model_id": model_id, "state_dict": best_state, "validation_macro_f1": best_score, "seed": seed}, checkpoint)
    (output / f"{model_id}_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return CheckpointEvidence(model_id, str(checkpoint), sha256_file(checkpoint), best_score, str(config.get("config_hash", "")), seed)


def _evaluate(model: Any, loader: Any, device: Any, criterion: Any, f1_score: Any) -> tuple[float, float]:
    import torch
    model.eval()
    losses: list[float] = []
    truth: list[int] = []
    predictions: list[int] = []
    with torch.no_grad():
        for batch in loader:
            labels = batch["label"].to(device)
            logits = model(batch["model_tensor"].to(device))
            losses.append(float(criterion(logits, labels).cpu()))
            truth.extend(labels.cpu().tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
    return float(np.mean(losses)) if losses else float("nan"), float(f1_score(truth, predictions, average="macro", zero_division=0))


def load_checkpoint(model_wrapper: ModelWrapper, checkpoint_path: str | Path, device: str = "cpu") -> Any:
    """Load a checkpoint and verify that its model identity matches the wrapper."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required to load checkpoints") from exc
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if payload.get("model_id") != model_wrapper.model_id:
        raise ValueError("Checkpoint model_id does not match the requested wrapper")
    model_wrapper.model.load_state_dict(payload["state_dict"])
    model_wrapper.model.to(device)
    model_wrapper.model.eval()
    return payload
