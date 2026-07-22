"""Deterministic sampling and image-space metrics for severity calibration."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Any, Iterable, Sequence

import numpy as np

from plantxai_stability.contracts import SampleRecord


SEVERITY_ORDER = ("mild", "moderate", "severe")


def select_leaf_balanced_pilot_records(
    records: Sequence[SampleRecord],
    *,
    seed: int,
    max_leaves_per_class: int,
) -> list[SampleRecord]:
    """Select one stable sample per validation leaf, balanced by class."""
    if max_leaves_per_class < 1:
        raise ValueError("max_leaves_per_class must be positive")
    if any(record.split != "validation" for record in records):
        raise ValueError("Severity pilot selection accepts validation records only")
    by_class_leaf: dict[str, dict[str, list[SampleRecord]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        by_class_leaf[record.class_name][record.leaf_id].append(record)
    selected: list[SampleRecord] = []
    for class_name in sorted(by_class_leaf):
        representatives: list[SampleRecord] = []
        for leaf_id, leaf_records in by_class_leaf[class_name].items():
            representatives.append(
                min(
                    leaf_records,
                    key=lambda item: _stable_key(
                        seed, class_name, leaf_id, item.sample_id
                    ),
                )
            )
        representatives.sort(
            key=lambda item: _stable_key(
                seed, class_name, item.leaf_id, item.sample_id
            )
        )
        selected.extend(representatives[:max_leaves_per_class])
    return sorted(selected, key=lambda item: (item.class_name, item.sample_id))


def image_change_metrics(original: np.ndarray, transformed: np.ndarray) -> dict[str, float]:
    """Measure pixel-space perturbation magnitude on RGB arrays in [0, 1]."""
    left = np.asarray(original, dtype=np.float64)
    right = np.asarray(transformed, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 3 or left.shape[-1] != 3:
        raise ValueError("Severity metrics require aligned HxWx3 RGB arrays")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Severity metric inputs contain NaN or Inf")
    if np.any((left < 0.0) | (left > 1.0)) or np.any((right < 0.0) | (right > 1.0)):
        raise ValueError("Severity metric inputs must be in [0, 1]")
    delta = right - left
    mae = float(np.mean(np.abs(delta)))
    rmse = float(np.sqrt(np.mean(np.square(delta))))
    psnr = float(20.0 * math.log10(1.0 / rmse)) if rmse > 0.0 else math.inf
    try:
        from skimage.metrics import structural_similarity
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install the optional 'xai' dependencies for SSIM") from exc
    ssim = float(
        structural_similarity(left, right, channel_axis=2, data_range=1.0)
    )
    clipped_fraction = float(np.mean((right <= 0.0) | (right >= 1.0)))
    return {
        "mae": mae,
        "rmse": rmse,
        "psnr": psnr,
        "ssim": ssim,
        "clipped_fraction": clipped_fraction,
    }


def summarize_pilot_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarize scenarios and test ordinal severity using median RMSE."""
    materialized = list(rows)
    if not materialized:
        raise ValueError("Severity pilot rows must not be empty")
    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in materialized:
        by_scenario[str(row["scenario_id"])].append(row)
    scenario_summary: dict[str, dict[str, Any]] = {}
    for scenario_id, scenario_rows in sorted(by_scenario.items()):
        item: dict[str, Any] = {
            "transformation": scenario_rows[0]["transformation"],
            "severity": scenario_rows[0]["severity"],
            "sample_count": len(scenario_rows),
        }
        for metric in ("mae", "rmse", "psnr", "ssim", "clipped_fraction"):
            values = np.asarray([float(row[metric]) for row in scenario_rows])
            item[metric] = {
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "q05": float(np.quantile(values, 0.05)),
                "q95": float(np.quantile(values, 0.95)),
            }
        scenario_summary[scenario_id] = item
    transformations = sorted(
        {str(row["transformation"]) for row in materialized}
    )
    ordinal_checks: dict[str, dict[str, Any]] = {}
    for transformation in transformations:
        scenario_ids = [f"{transformation}_{severity}" for severity in SEVERITY_ORDER]
        missing = [item for item in scenario_ids if item not in scenario_summary]
        medians = (
            []
            if missing
            else [
                float(scenario_summary[item]["rmse"]["median"])
                for item in scenario_ids
            ]
        )
        passed = not missing and all(
            left < right for left, right in zip(medians, medians[1:])
        )
        ordinal_checks[transformation] = {
            "scenario_ids": scenario_ids,
            "median_rmse": medians,
            "strictly_increasing": passed,
            "missing_scenarios": missing,
        }
    finite_metrics = all(
        math.isfinite(float(row[metric]))
        for row in materialized
        for metric in ("mae", "rmse", "psnr", "ssim", "clipped_fraction")
    )
    return {
        "row_count": len(materialized),
        "scenario_count": len(by_scenario),
        "scenario_summary": scenario_summary,
        "ordinal_checks": ordinal_checks,
        "all_metrics_finite": finite_metrics,
        "ordinal_gate_passed": all(
            item["strictly_increasing"] for item in ordinal_checks.values()
        ),
    }


def _stable_key(seed: int, *parts: str) -> str:
    payload = ":".join((str(seed), *parts)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
