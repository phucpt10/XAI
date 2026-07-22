import numpy as np

from plantxai_stability.statistics import heatmap_metrics
from plantxai_stability.xai import forward_align_heatmap


def _heatmap(size: int = 64) -> np.ndarray:
    rows, cols = np.mgrid[:size, :size]
    values = np.exp(-((rows - 25.0) ** 2 + (cols - 39.0) ** 2) / 180.0)
    return values.astype(np.float32)


def test_forward_alignment_returns_geometric_valid_region() -> None:
    aligned, mask = forward_align_heatmap(
        _heatmap(),
        {"kind": "rotation", "angle_degrees": 25.0, "validity_threshold": 0.999999},
    )
    assert aligned.shape == mask.shape == (64, 64)
    assert 0 < int(mask.sum()) < mask.size


def test_masked_metrics_ignore_everything_outside_valid_region() -> None:
    aligned, mask = forward_align_heatmap(
        _heatmap(),
        {"kind": "rotation", "angle_degrees": 25.0, "validity_threshold": 0.999999},
    )
    changed_outside = aligned.copy()
    changed_outside[~mask] = 1.0
    metrics = heatmap_metrics(aligned, changed_outside, mask)
    assert np.isclose(metrics["pearson"], 1.0)
    assert np.isclose(metrics["ssim"], 1.0)
    assert np.isclose(metrics["topk_iou_10"], 1.0)
    assert np.isclose(metrics["topk_iou_20"], 1.0)
    assert np.isclose(metrics["topk_iou_30"], 1.0)


def test_masked_metrics_respond_to_changes_inside_valid_region() -> None:
    aligned, mask = forward_align_heatmap(
        _heatmap(),
        {"kind": "rotation", "angle_degrees": 25.0, "validity_threshold": 0.999999},
    )
    changed_inside = aligned.copy()
    changed_inside[mask] = np.flip(aligned[mask])
    metrics = heatmap_metrics(aligned, changed_inside, mask)
    assert metrics["pearson"] < 0.99
    assert metrics["ssim"] < 0.99
    assert metrics["topk_iou_20"] < 0.99
