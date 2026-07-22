"""Stability metrics and predeclared statistical procedures."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def heatmap_metrics(
    original: np.ndarray,
    transformed: np.ndarray,
    valid_mask: np.ndarray | None = None,
    *,
    ssim_window_size: int = 7,
    topk_values: tuple[float, ...] = (0.1, 0.2, 0.3),
) -> dict[str, float]:
    if original.shape != transformed.shape:
        raise ValueError("Heatmaps must have identical shapes")
    mask = np.ones(original.shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if mask.shape != original.shape or not mask.any():
        raise ValueError("Valid-overlap mask is empty or has the wrong shape")
    if ssim_window_size < 3 or ssim_window_size % 2 == 0:
        raise ValueError("SSIM window size must be an odd integer >= 3")
    if any(not 0.0 < value < 1.0 for value in topk_values):
        raise ValueError("Top-k fractions must be in (0, 1)")
    left = np.asarray(original, dtype=np.float64)
    right = np.asarray(transformed, dtype=np.float64)
    x = left[mask]
    y = right[mask]
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Heatmaps contain NaN or Inf")
    if np.isclose(x.std(), 0.0) or np.isclose(y.std(), 0.0):
        raise ValueError("Constant heatmap cannot receive a stability score")
    left = _masked_minmax(left, mask)
    right = _masked_minmax(right, mask)
    x = left[mask]
    y = right[mask]
    pearson = float(np.corrcoef(x, y)[0, 1])
    cosine = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
    try:
        from skimage.metrics import structural_similarity
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the optional 'xai' dependencies for SSIM") from exc
    _, ssim_map = structural_similarity(
        left,
        right,
        data_range=1.0,
        win_size=ssim_window_size,
        full=True,
    )
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for masked SSIM") from exc
    kernel = np.ones((ssim_window_size, ssim_window_size), dtype=np.uint8)
    ssim_mask = cv2.erode(
        mask.astype(np.uint8),
        kernel,
        iterations=1,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    if not ssim_mask.any():
        raise ValueError("SSIM-valid mask is empty after window erosion")
    metrics = {
        "ssim": float(np.asarray(ssim_map)[ssim_mask].mean()),
        "pearson": pearson,
        "cosine": cosine,
        "valid_pixel_count": float(np.count_nonzero(mask)),
        "valid_pixel_fraction": float(np.mean(mask)),
        "ssim_valid_pixel_count": float(np.count_nonzero(ssim_mask)),
        "ssim_valid_pixel_fraction": float(np.mean(ssim_mask)),
    }
    for value in topk_values:
        key = f"topk_iou_{int(round(value * 100)):02d}"
        metrics[key] = _masked_topk_iou(left, right, mask, value)
    return metrics


def _masked_minmax(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    selected = values[mask]
    minimum = float(selected.min())
    maximum = float(selected.max())
    if np.isclose(maximum, minimum):
        raise ValueError("Constant heatmap cannot be normalized on the valid region")
    normalized = (values - minimum) / (maximum - minimum)
    return np.clip(normalized, 0.0, 1.0)


def _masked_topk_iou(
    left: np.ndarray,
    right: np.ndarray,
    mask: np.ndarray,
    fraction: float,
) -> float:
    left_threshold = float(np.quantile(left[mask], 1.0 - fraction))
    right_threshold = float(np.quantile(right[mask], 1.0 - fraction))
    left_top = mask & (left >= left_threshold)
    right_top = mask & (right >= right_threshold)
    union = int(np.count_nonzero(left_top | right_top))
    if union == 0:
        raise ValueError("Top-k IoU union is empty")
    return float(np.count_nonzero(left_top & right_top) / union)


def bootstrap_leaf_means(values: Iterable[float], leaf_ids: Iterable[str], iterations: int, seed: int, confidence_level: float = 0.95) -> dict[str, float]:
    value_list = np.asarray(list(values), dtype=float)
    leaf_list = np.asarray(list(leaf_ids), dtype=str)
    unique = np.unique(leaf_list)
    if value_list.size == 0 or unique.size == 0 or value_list.size != leaf_list.size:
        raise ValueError("Values and leaf_ids must be non-empty and aligned")
    rng = np.random.default_rng(seed)
    by_leaf = {leaf: value_list[leaf_list == leaf] for leaf in unique}
    estimates = np.empty(iterations, dtype=float)
    for index in range(iterations):
        sampled = rng.choice(unique, size=unique.size, replace=True)
        estimates[index] = np.concatenate([by_leaf[leaf] for leaf in sampled]).mean()
    alpha = (1.0 - confidence_level) / 2.0
    return {"estimate": float(value_list.mean()), "lower": float(np.quantile(estimates, alpha)), "upper": float(np.quantile(estimates, 1.0 - alpha)), "n_leaf": int(unique.size), "n_value": int(value_list.size)}


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    values = [float(p) for p in p_values]
    order = sorted(range(len(values)), key=lambda index: values[index])
    adjusted = [0.0] * len(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def paired_wilcoxon(x: Iterable[float], y: Iterable[float]) -> dict[str, float]:
    left = np.asarray(list(x), dtype=float)
    right = np.asarray(list(y), dtype=float)
    if left.size != right.size or left.size == 0:
        raise ValueError("Paired arrays must have equal non-zero length")
    try:
        from scipy.stats import rankdata, wilcoxon
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("SciPy is required for paired Wilcoxon") from exc
    difference = left - right
    result = wilcoxon(left, right, zero_method="pratt", alternative="two-sided")
    nonzero = difference[difference != 0]
    if nonzero.size == 0:
        effect = 0.0
    else:
        ranks = rankdata(np.abs(nonzero))
        effect = float((ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / (ranks.sum()))
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue), "rank_biserial": effect, "n_pairs": int(left.size)}
