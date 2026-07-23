from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from plantxai_stability.config import load_protocol
from plantxai_stability.official_analysis import (
    _apply_holm,
    _paired_contrast,
    bootstrap_leaf_correlation,
    cluster_bootstrap_means,
    load_analysis_decision,
    validate_analysis_plan,
)


def test_approved_analysis_plan_matches_frozen_protocol() -> None:
    protocol = load_protocol("configs/protocol/v0.9/protocol.yaml")
    decision = load_analysis_decision(
        Path("configs/protocol/v0.9/decision_records/DR-ANALYSIS-001.yaml")
    )
    validate_analysis_plan(decision, protocol)


def test_analysis_plan_rejects_unapproved_merge_hash() -> None:
    protocol = load_protocol("configs/protocol/v0.9/protocol.yaml")
    decision = load_analysis_decision(
        Path("configs/protocol/v0.9/decision_records/DR-ANALYSIS-001.yaml")
    )
    changed = deepcopy(decision)
    changed["source_merges"]["resnet50"]["report_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source merge SHA-256"):
        validate_analysis_plan(changed, protocol)


def test_cluster_bootstrap_is_deterministic_and_uses_leaf_units() -> None:
    frame = pd.DataFrame(
        {
            "leaf_id": ["leaf_a", "leaf_a", "leaf_b", "leaf_c"],
            "metric_a": [0.0, 1.0, 0.5, 1.0],
            "metric_b": [1.0, 0.0, 0.5, 0.0],
        }
    )
    left = cluster_bootstrap_means(
        frame,
        ("metric_a", "metric_b"),
        iterations=100,
        confidence_level=0.95,
        seed=42,
    )
    right = cluster_bootstrap_means(
        frame,
        ("metric_a", "metric_b"),
        iterations=100,
        confidence_level=0.95,
        seed=42,
    )
    assert left == right
    assert left["metric_a"]["n_leaf"] == 3
    assert left["metric_a"]["n_value"] == 4
    assert left["metric_a"]["estimate"] == 0.625


def test_paired_contrast_uses_common_samples_then_leaf_means() -> None:
    rows = []
    for leaf_index in range(20):
        for sample_index in range(2):
            sample_id = f"sample_{leaf_index}_{sample_index}"
            rows.extend(
                [
                    {
                        "model_id": "resnet50",
                        "sample_id": sample_id,
                        "leaf_id": f"leaf_{leaf_index}",
                        "metric": 0.8,
                    },
                    {
                        "model_id": "efficientnet_b0",
                        "sample_id": sample_id,
                        "leaf_id": f"leaf_{leaf_index}",
                        "metric": 0.7,
                    },
                ]
            )
    result = _paired_contrast(
        pd.DataFrame(rows),
        left_filter={"model_id": "resnet50"},
        right_filter={"model_id": "efficientnet_b0"},
        endpoint="metric",
        family_id="family",
        contrast_type="model",
        contrast="resnet_minus_efficientnet",
        minimum_leaf_count=20,
    )
    assert result["n_common_sample_keys"] == 40
    assert result["n_leaf_pairs"] == 20
    assert np.isclose(result["mean_difference_left_minus_right"], 0.1)
    _apply_holm([result], alpha=0.05)
    assert result["p_value_holm"] == result["p_value_raw"]


def test_leaf_correlation_bootstrap_is_deterministic() -> None:
    left = np.arange(20, dtype=float)
    right = left * 2.0 + 1.0
    first = bootstrap_leaf_correlation(
        left,
        right,
        iterations=100,
        confidence_level=0.95,
        seed=42,
    )
    second = bootstrap_leaf_correlation(
        left,
        right,
        iterations=100,
        confidence_level=0.95,
        seed=42,
    )
    assert first == second
    assert np.isclose(first["estimate"], 1.0)
    assert first["n_leaf"] == 20
