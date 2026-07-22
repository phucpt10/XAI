"""CAM adapters with explicit dependency and target-class contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class HeatmapQuality:
    valid: bool
    reason: str | None


def normalize_heatmap(heatmap: np.ndarray, epsilon: float = 1e-8) -> tuple[np.ndarray, HeatmapQuality]:
    array = np.asarray(heatmap, dtype=np.float32)
    if array.ndim != 2:
        return array, HeatmapQuality(False, "invalid_shape")
    if not np.isfinite(array).all():
        return array, HeatmapQuality(False, "non_finite_heatmap")
    minimum = float(array.min())
    maximum = float(array.max())
    if np.isclose(maximum, minimum):
        return array, HeatmapQuality(False, "constant_heatmap")
    normalized = (array - minimum) / (maximum - minimum + epsilon)
    return normalized.astype(np.float32), HeatmapQuality(True, None)


def inverse_align_heatmap(heatmap: np.ndarray, inverse_metadata: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Align a transformed heatmap and return a valid-overlap mask."""
    kind = inverse_metadata.get("kind", "identity")
    if kind == "identity":
        return np.asarray(heatmap, dtype=np.float32), np.ones(np.asarray(heatmap).shape, dtype=bool)
    if kind != "rotation":
        raise ValueError(f"Unsupported inverse alignment: {kind}")
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required for heatmap alignment") from exc
    angle = float(inverse_metadata["angle_degrees"])
    array = np.asarray(heatmap, dtype=np.float32)
    image = Image.fromarray(array, mode="F")
    aligned = np.asarray(image.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=0.0), dtype=np.float32)
    valid_source = Image.fromarray(np.ones(array.shape, dtype=np.float32), mode="F")
    mask = np.asarray(valid_source.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=0.0), dtype=np.float32) > 0.999
    if not mask.any():
        raise ValueError("Inverse alignment produced an empty valid-overlap mask")
    return aligned, mask


class CAMGenerator:
    def __init__(self, model: Any, target_layer: Any, method: str) -> None:
        self.model = model
        self.target_layer = target_layer
        self.method = method

    def generate(self, input_tensor: Any, target_class: int) -> np.ndarray:
        if target_class < 0:
            raise ValueError("target_class must be non-negative")
        try:
            from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install the optional 'xai' dependencies to generate CAMs") from exc
        algorithms = {"grad_cam": GradCAM, "grad_cam_plus_plus": GradCAMPlusPlus, "score_cam": ScoreCAM}
        if self.method not in algorithms:
            raise ValueError(f"Unsupported XAI method: {self.method}")
        algorithm = algorithms[self.method](model=self.model, target_layers=[self.target_layer])
        try:
            values = algorithm(input_tensor=input_tensor, targets=[ClassifierOutputTarget(target_class)])[0]
        finally:
            algorithm.__exit__(None, None, None) if hasattr(algorithm, "__exit__") else None
        normalized, quality = normalize_heatmap(values)
        if not quality.valid:
            raise ValueError(f"Invalid heatmap: {quality.reason}")
        return normalized
