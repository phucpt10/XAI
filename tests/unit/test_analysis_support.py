from __future__ import annotations

import pandas as pd

from plantxai_stability.analysis_support import (
    EXPECTED_SUPPORT_COUNTS,
    EXPECTED_SUPPORT_TOTAL,
    build_support_audit,
    exclusion_reason_audit,
)


SCENARIOS = [
    f"{transformation}_{severity}"
    for transformation in (
        "rotation",
        "brightness",
        "gaussian_noise",
        "gaussian_blur",
    )
    for severity in ("mild", "moderate", "severe")
]
METHODS = ("grad_cam", "grad_cam_plus_plus", "score_cam")
MODELS = ("resnet50", "efficientnet_b0")


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = []
    joint = []
    for leaf_index in range(20):
        sample_id = f"sample_{leaf_index}"
        leaf_id = f"leaf_{leaf_index}"
        for model_id in MODELS:
            for scenario_id in SCENARIOS:
                predictions.append(
                    {
                        "model_id": model_id,
                        "sample_id": sample_id,
                        "leaf_id": leaf_id,
                        "scenario_id": scenario_id,
                        "true_class_name": "class_a",
                    }
                )
                for method in METHODS:
                    joint.append(
                        {
                            "model_id": model_id,
                            "sample_id": sample_id,
                            "leaf_id": leaf_id,
                            "scenario_id": scenario_id,
                            "xai_method": method,
                            "exclusion_reason": "",
                        }
                    )
    return pd.DataFrame(predictions), pd.DataFrame(joint)


def test_support_audit_covers_all_predeclared_contrasts() -> None:
    predictions, joint = _frames()
    rows = build_support_audit(
        predictions,
        joint,
        methods=METHODS,
        minimum_leaf_count=20,
    )
    counts = {
        scope: sum(row["contrast_scope"] == scope for row in rows)
        for scope in EXPECTED_SUPPORT_COUNTS
    }
    assert len(rows) == EXPECTED_SUPPORT_TOTAL
    assert counts == EXPECTED_SUPPORT_COUNTS
    assert all(row["passes_minimum_common_leaves"] for row in rows)


def test_support_audit_retains_insufficient_contrast_without_metrics() -> None:
    predictions, joint = _frames()
    excluded_ids = {f"sample_{index}" for index in range(10)}
    mask = (
        joint["model_id"].eq("efficientnet_b0")
        & joint["xai_method"].eq("score_cam")
        & joint["scenario_id"].eq("gaussian_blur_severe")
        & joint["sample_id"].isin(excluded_ids)
    )
    joint.loc[mask, "exclusion_reason"] = "prediction_inconsistent"
    rows = build_support_audit(
        predictions,
        joint,
        methods=METHODS,
        minimum_leaf_count=20,
    )
    target = next(
        row for row in rows if row["contrast_id"] == "rq2_model::score_cam::gaussian_blur_severe"
    )
    assert target["common_leaves"] == 10
    assert not target["passes_minimum_common_leaves"]
    assert target["support_status"] == "NOT_ESTIMABLE_INSUFFICIENT_COMMON_LEAVES"


def test_exclusion_reason_audit_reconciles_rows() -> None:
    _, joint = _frames()
    joint.loc[joint.index[0], "exclusion_reason"] = "prediction_inconsistent"
    rows = exclusion_reason_audit(joint)
    assert sum(row["row_count"] for row in rows) == len(joint)
    assert {row["reason"] for row in rows} == {
        "included",
        "prediction_inconsistent",
    }
