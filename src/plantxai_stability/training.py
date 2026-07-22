"""Reproducible training and inference loops for Colab or local GPU runs."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np

from plantxai_stability.models import CheckpointEvidence, ModelWrapper
from plantxai_stability.provenance import sha256_file


def seed_everything(seed: int, deterministic_algorithms: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(deterministic_algorithms)
    except ImportError:
        return


def train_model(
    model_id: str,
    train_loader: Any,
    validation_loader: Any,
    config: dict[str, Any],
    output_dir: str | Path,
    *,
    resume: bool = False,
) -> CheckpointEvidence:
    """Train and select by validation macro-F1; no test loader is accepted."""
    try:
        import torch
        from sklearn.metrics import f1_score
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the [ml] dependencies before training") from exc
    seed = int(config.get("seed", 42))
    seed_everything(seed, bool(config.get("deterministic_algorithms", True)))
    output = Path(output_dir)
    latest_checkpoint = output / f"{model_id}_latest.pt"
    best_checkpoint = output / f"{model_id}_best.pt"
    history_path = output / f"{model_id}_history.json"
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError("Training output exists; use --resume or a new versioned directory")
    output.mkdir(parents=True, exist_ok=True)
    if resume and not latest_checkpoint.is_file():
        raise FileNotFoundError(f"Resume checkpoint is missing: {latest_checkpoint}")
    wrapper = ModelWrapper(
        model_id,
        int(config.get("num_classes", 5)),
        bool(config.get("pretrained", True)) and not resume,
    )
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = wrapper.model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("learning_rate", 1e-3)), weight_decay=float(config.get("weight_decay", 1e-4)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config.get("max_epochs", 50)))
    criterion = torch.nn.CrossEntropyLoss()
    amp_enabled = bool(config.get("mixed_precision", False) and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    best_score = float("-inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []
    patience = int(config.get("early_stopping_patience", 8))
    start_epoch = 0
    if resume:
        resume_payload = torch.load(latest_checkpoint, map_location=device, weights_only=False)
        _validate_resume_payload(resume_payload, model_id, config)
        model.load_state_dict(resume_payload["state_dict"])
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
        scaler.load_state_dict(resume_payload["scaler_state_dict"])
        start_epoch = int(resume_payload["next_epoch"])
        best_score = float(resume_payload["best_validation_macro_f1"])
        best_epoch = int(resume_payload["best_epoch"])
        stale = int(resume_payload["stale_epochs"])
        history = list(resume_payload["history"])
        _restore_rng_state(resume_payload["rng_state"], torch)
        generator_state = resume_payload.get("train_loader_generator_state")
        if generator_state is not None and getattr(train_loader, "generator", None) is not None:
            train_loader.generator.set_state(generator_state)
        print(f"Resuming {model_id} at epoch {start_epoch + 1}")
    max_epochs = int(config.get("max_epochs", 50))
    for epoch in range(start_epoch, max_epochs):
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
        validation_loss, validation_f1 = _evaluate(model, validation_loader, device, criterion, f1_score)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        history.append({"epoch": float(epoch + 1), "train_loss": train_loss / max(train_count, 1), "validation_loss": validation_loss, "validation_macro_f1": validation_f1, "learning_rate": learning_rate})
        if validation_f1 > best_score:
            best_score = validation_f1
            best_epoch = epoch + 1
            stale = 0
            _atomic_torch_save(
                _best_checkpoint_payload(model_id, model, config, best_score, best_epoch, seed),
                best_checkpoint,
                torch,
            )
        else:
            stale += 1
        scheduler.step()
        _atomic_write_json(history_path, history)
        latest_payload = {
            "format_version": 2,
            "checkpoint_role": "latest_training_state",
            "model_id": model_id,
            "num_classes": int(config.get("num_classes", 5)),
            "protocol_hash": str(config.get("config_hash", "")),
            "manifest_sha256": str(config.get("manifest_sha256", "")),
            "training_config": _training_config_snapshot(config),
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "next_epoch": epoch + 1,
            "best_epoch": best_epoch,
            "best_validation_macro_f1": best_score,
            "stale_epochs": stale,
            "history": history,
            "seed": seed,
            "rng_state": _capture_rng_state(torch),
            "train_loader_generator_state": (
                train_loader.generator.get_state()
                if getattr(train_loader, "generator", None) is not None
                else None
            ),
        }
        _atomic_torch_save(latest_payload, latest_checkpoint, torch)
        print(
            f"Epoch {epoch + 1}/{max_epochs}: train_loss={history[-1]['train_loss']:.6f}, "
            f"validation_loss={validation_loss:.6f}, validation_macro_f1={validation_f1:.6f}, "
            f"best={best_score:.6f} (epoch {best_epoch})"
        )
        if stale >= patience:
            print(f"Early stopping after {stale} stale epochs")
            break
    if not best_checkpoint.is_file():
        raise RuntimeError("No validation checkpoint was produced")
    return CheckpointEvidence(
        model_id=model_id,
        checkpoint_path=str(best_checkpoint),
        checkpoint_sha256=sha256_file(best_checkpoint),
        validation_metric=best_score,
        config_hash=str(config.get("config_hash", "")),
        seed=seed,
        best_epoch=best_epoch,
        epochs_completed=len(history),
        history_path=str(history_path),
        latest_checkpoint_path=str(latest_checkpoint),
        manifest_sha256=str(config.get("manifest_sha256", "")),
    )


def _evaluate(model: Any, loader: Any, device: Any, criterion: Any, f1_score: Any) -> tuple[float, float]:
    import torch
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    truth: list[int] = []
    predictions: list[int] = []
    with torch.no_grad():
        for batch in loader:
            labels = batch["label"].to(device)
            logits = model(batch["model_tensor"].to(device))
            loss_sum += float(criterion(logits, labels).cpu()) * len(labels)
            sample_count += len(labels)
            truth.extend(labels.cpu().tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
    return loss_sum / sample_count if sample_count else float("nan"), float(f1_score(truth, predictions, average="macro", zero_division=0))


def load_checkpoint(
    model_wrapper: ModelWrapper,
    checkpoint_path: str | Path,
    device: str = "cpu",
    *,
    expected_protocol_hash: str | None = None,
    expected_manifest_sha256: str | None = None,
) -> Any:
    """Load a checkpoint and verify that its model identity matches the wrapper."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required to load checkpoints") from exc
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if payload.get("model_id") != model_wrapper.model_id:
        raise ValueError("Checkpoint model_id does not match the requested wrapper")
    if int(payload.get("num_classes", -1)) != model_wrapper.num_classes:
        raise ValueError("Checkpoint num_classes does not match the requested wrapper")
    if expected_protocol_hash is not None and payload.get("protocol_hash") != expected_protocol_hash:
        raise ValueError("Checkpoint protocol hash mismatch")
    if expected_manifest_sha256 is not None and payload.get("manifest_sha256") != expected_manifest_sha256:
        raise ValueError("Checkpoint manifest hash mismatch")
    model_wrapper.model.load_state_dict(payload["state_dict"])
    model_wrapper.model.to(device)
    model_wrapper.model.eval()
    return payload


def _best_checkpoint_payload(
    model_id: str,
    model: Any,
    config: dict[str, Any],
    best_score: float,
    best_epoch: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "format_version": 2,
        "checkpoint_role": "validation_selected_best",
        "model_id": model_id,
        "num_classes": int(config.get("num_classes", 5)),
        "class_names": list(config.get("class_names", [])),
        "state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        "validation_macro_f1": best_score,
        "best_epoch": best_epoch,
        "seed": seed,
        "protocol_hash": str(config.get("config_hash", "")),
        "manifest_sha256": str(config.get("manifest_sha256", "")),
        "training_config": _training_config_snapshot(config),
        "test_split_accessed": False,
    }


def _training_config_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "pretrained",
        "pretrained_weights",
        "fine_tuning",
        "loss",
        "class_weighting",
        "max_epochs",
        "batch_size",
        "learning_rate",
        "optimizer",
        "weight_decay",
        "scheduler",
        "early_stopping_patience",
        "selection_metric",
        "mixed_precision",
        "deterministic_algorithms",
        "seed",
        "software_version",
        "git_commit",
    )
    return {key: config.get(key) for key in keys}


def _validate_resume_payload(
    payload: dict[str, Any], model_id: str, config: dict[str, Any]
) -> None:
    expected = {
        "format_version": 2,
        "model_id": model_id,
        "num_classes": int(config.get("num_classes", 5)),
        "protocol_hash": str(config.get("config_hash", "")),
        "manifest_sha256": str(config.get("manifest_sha256", "")),
        "training_config": _training_config_snapshot(config),
    }
    mismatches = [key for key, value in expected.items() if payload.get(key) != value]
    if mismatches:
        raise ValueError(f"Resume checkpoint lineage mismatch: {mismatches}")


def _capture_rng_state(torch: Any) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state: dict[str, Any], torch: Any) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _atomic_torch_save(payload: dict[str, Any], path: Path, torch: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)
