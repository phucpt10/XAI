"""Immutable data contracts shared across the research pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    leaf_id: str
    class_id: int
    class_name: str
    source_split: str
    split: str
    canonical_relative_path: str
    canonical_rgb_sha256: str
    width: int
    height: int
    source_row_index: int | None = None


@dataclass(frozen=True)
class PredictionRecord:
    run_id: str
    model_id: str
    sample_id: str
    scenario_id: str
    predicted_class: int
    confidence: float
    is_correct: bool
    checkpoint_sha256: str


@dataclass(frozen=True)
class TransformationRecord:
    sample_id: str
    scenario_id: str
    transformation: str
    severity: str
    seed: int
    parameters: dict[str, Any]
    inverse_metadata: dict[str, Any]
    valid_mask_sha256: Optional[str]


@dataclass(frozen=True)
class JointRecord:
    run_id: str
    model_id: str
    sample_id: str
    leaf_id: str
    scenario_id: str
    xai_method: str
    target_class: int
    is_consistent: bool
    ssim: Optional[float]
    pearson: Optional[float]
    cosine: Optional[float]
    exclusion_reason: Optional[str]
