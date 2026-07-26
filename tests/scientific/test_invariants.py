import numpy as np

from plantxai_stability.xai import normalize_heatmap


def test_constant_heatmap_is_invalid() -> None:
    _, quality = normalize_heatmap(np.ones((4, 4), dtype=np.float32))
    assert quality.valid is False
    assert quality.reason == "constant_heatmap"


def test_non_finite_heatmap_is_invalid() -> None:
    _, quality = normalize_heatmap(np.array([[0.0, np.nan]], dtype=np.float32))
    assert quality.valid is False
    assert quality.reason == "non_finite_heatmap"
