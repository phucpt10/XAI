"""Prediction pairing and consistency-gated joint evaluation."""

from __future__ import annotations

from dataclasses import asdict
from typing import Iterable, cast

from plantxai_stability.contracts import JointRecord, PredictionRecord


def pair_predictions(original: Iterable[PredictionRecord], transformed: Iterable[PredictionRecord]) -> list[dict[str, object]]:
    left = {(item.run_id, item.model_id, item.sample_id, item.scenario_id): item for item in original}
    right = {(item.run_id, item.model_id, item.sample_id, item.scenario_id): item for item in transformed}
    if set(left) != set(right):
        missing_left = sorted(set(right) - set(left))[:3]
        missing_right = sorted(set(left) - set(right))[:3]
        raise ValueError(f"Prediction keys do not match; missing_original={missing_left}, missing_transformed={missing_right}")
    records: list[dict[str, object]] = []
    for key in sorted(left):
        original_record = left[key]
        transformed_record = right[key]
        delta = transformed_record.confidence - original_record.confidence
        records.append({"key": key, "original": asdict(original_record), "transformed": asdict(transformed_record), "is_consistent": original_record.predicted_class == transformed_record.predicted_class, "confidence_delta": delta, "absolute_confidence_delta": abs(delta)})
    return records


def make_exclusion(pair: dict[str, object], reason: str) -> dict[str, object]:
    return {"key": pair["key"], "is_consistent": pair["is_consistent"], "exclusion_reason": reason}


def make_joint_record(pair: dict[str, object], leaf_id: str, method: str, metrics: dict[str, float] | None, target_class: int, reason: str | None = None) -> JointRecord:
    run_id, model_id, sample_id, scenario_id = cast(tuple[str, str, str, str], pair["key"])
    return JointRecord(
        run_id,
        model_id,
        sample_id,
        leaf_id,
        scenario_id,
        method,
        target_class,
        bool(pair["is_consistent"]),
        None if metrics is None else metrics.get("ssim"),
        None if metrics is None else metrics.get("pearson"),
        None if metrics is None else metrics.get("cosine"),
        None if metrics is None else metrics.get("topk_iou_10"),
        None if metrics is None else metrics.get("topk_iou_20"),
        None if metrics is None else metrics.get("topk_iou_30"),
        None if metrics is None else int(metrics["valid_pixel_count"]),
        None if metrics is None else metrics.get("valid_pixel_fraction"),
        reason,
    )
