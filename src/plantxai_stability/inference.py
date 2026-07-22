"""Per-sample inference records for baseline and transformed images."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable

import numpy as np

from plantxai_stability.contracts import PredictionRecord, SampleRecord
from plantxai_stability.data.loader import preprocess_for_model


def infer_one(model: Any, pixels: np.ndarray, record: SampleRecord, model_id: str, run_id: str, scenario_id: str, checkpoint_sha256: str, device: Any = None) -> PredictionRecord:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for inference") from exc
    tensor = preprocess_for_model(pixels).unsqueeze(0)
    if device is not None:
        tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted = probabilities.max(dim=1)
    return PredictionRecord(run_id, model_id, record.sample_id, scenario_id, int(predicted.item()), float(confidence.item()), int(predicted.item()) == record.class_id, checkpoint_sha256)


def records_to_rows(records: Iterable[PredictionRecord]) -> list[dict[str, Any]]:
    return [asdict(record) for record in records]
