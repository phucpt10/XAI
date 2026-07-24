from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import plantxai_stability.official_analysis as official_analysis
from plantxai_stability.config import load_protocol
from plantxai_stability.official_analysis import (
    _apply_holm,
    _paired_contrast,
    bootstrap_leaf_correlation,
    cluster_bootstrap_means,
    load_analysis_decision,
    load_analysis_support_decision,
    paired_comparisons,
    validate_analysis_plan,
    validate_analysis_support_authorization,
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


def test_approved_analysis_support_decision_loads() -> None:
    decision = load_analysis_support_decision(
        Path(
            "configs/protocol/v0.9/decision_records/"
            "DR-ANALYSIS-SUPPORT-001.yaml"
        )
    )
    assert decision["source_support_audit"]["report_sha256"] == (
        "f370b3c7ace79cd5523242831593522671844f6459d2fa344f21f829613c13ac"
    )
    assert decision["output_contract"]["estimable_paired_rows"] == 573
    assert decision["output_contract"]["non_estimable_paired_rows"] == 3


def test_analysis_support_authorization_binds_audit_and_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis_decision_path = Path(
        "configs/protocol/v0.9/decision_records/DR-ANALYSIS-001.yaml"
    )
    support_decision_path = Path(
        "configs/protocol/v0.9/decision_records/DR-ANALYSIS-SUPPORT-001.yaml"
    )
    support_decision = load_analysis_support_decision(support_decision_path)
    insufficient = pd.DataFrame(
        [
            {
                "contrast_scope": "rq2_model",
                "contrast_id": "rq2_model::score_cam::gaussian_blur_severe",
                "common_samples": 14,
                "common_leaves": 12,
                "minimum_common_leaves": 20,
                "support_status": "NOT_ESTIMABLE_INSUFFICIENT_COMMON_LEAVES",
            }
        ]
    )
    insufficient.to_csv(
        tmp_path / "insufficient_common_leaf_contrasts.csv",
        index=False,
    )
    (tmp_path / "exclusion_reason_audit.csv").write_text(
        "model_id,row_count\nresnet50,1\n",
        encoding="utf-8",
    )
    (tmp_path / "planned_contrast_support_audit.csv").write_text(
        "contrast_id\nexample\n",
        encoding="utf-8",
    )
    artifact_hashes = {
        name: official_analysis.sha256_file(tmp_path / name)
        for name in (
            "exclusion_reason_audit.csv",
            "insufficient_common_leaf_contrasts.csv",
            "planned_contrast_support_audit.csv",
        )
    }
    report = {
        "run_type": "metadata_only_official_analysis_support_audit",
        "analysis_decision_id": "DR-ANALYSIS-001",
        "analysis_decision_record_sha256": official_analysis.sha256_file(
            analysis_decision_path
        ),
        "minimum_common_leaf_count": 20,
        "planned_contrast_count": 192,
        "insufficient_contrast_count": 1,
        "all_contrasts_meet_minimum": False,
        "analysis_execution_allowed_without_adjudication": False,
        "official_test_pixels_accessed": False,
        "endpoint_metric_values_read": False,
        "hypothesis_tests_computed": False,
        "artifact_sha256": artifact_hashes,
        "acceptance_criteria": {"metadata_only": True},
    }
    report_path = tmp_path / "analysis_support_audit_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    report_hash = official_analysis.sha256_file(report_path)
    monkeypatch.setattr(
        official_analysis,
        "EXPECTED_SUPPORT_AUDIT_REPORT_SHA256",
        report_hash,
    )
    monkeypatch.setattr(
        official_analysis,
        "EXPECTED_SUPPORT_AUDIT_ARTIFACT_SHA256",
        artifact_hashes,
    )
    support_decision = deepcopy(support_decision)
    support_decision["source_support_audit"]["report_sha256"] = report_hash
    support_decision["source_support_audit"]["artifact_sha256"] = artifact_hashes

    exception = validate_analysis_support_authorization(
        support_decision=support_decision,
        support_decision_path=support_decision_path,
        support_audit_dir=tmp_path,
        analysis_decision_path=analysis_decision_path,
    )

    assert exception["expected_common_samples"] == 14
    assert exception["expected_common_leaves"] == 12


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
    assert left["metric_a"]["estimate"] == pytest.approx(2.0 / 3.0)


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


def test_non_estimable_contrast_retains_conservative_holm_family_slot() -> None:
    rows = []
    for leaf_index in range(12):
        sample_id = f"sample_{leaf_index}"
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
    authorization = {
        "analysis_contrast_type": "rq2_model",
        "analysis_contrast": "approved_contrast",
        "endpoints": ["metric"],
        "expected_common_samples": 12,
        "expected_common_leaves": 12,
        "minimum_common_leaves": 20,
        "support_status": "NOT_ESTIMABLE_INSUFFICIENT_COMMON_LEAVES",
    }
    placeholder = _paired_contrast(
        pd.DataFrame(rows),
        left_filter={"model_id": "resnet50"},
        right_filter={"model_id": "efficientnet_b0"},
        endpoint="metric",
        family_id="family",
        contrast_type="rq2_model",
        contrast="approved_contrast",
        minimum_leaf_count=20,
        non_estimable_authorization=authorization,
    )
    family = [
        {
            "family_id": "family",
            "estimable": True,
            "p_value_raw": 0.001 if index == 0 else 0.5,
        }
        for index in range(11)
    ]
    family.append(placeholder)
    _apply_holm(family, alpha=0.05)

    assert placeholder["estimable"] is False
    assert placeholder["p_value_raw"] == ""
    assert placeholder["p_value_holm"] == ""
    assert placeholder["reject_h0_holm"] == ""
    assert placeholder["holm_family_size"] == 12
    assert placeholder["holm_estimable_count"] == 11
    assert placeholder["holm_reserved_non_estimable_slots"] == 1
    assert family[0]["p_value_holm"] == pytest.approx(0.012)


def test_full_comparison_plan_retains_three_authorized_non_estimable_rows() -> None:
    scenarios = [
        f"{transformation}_{severity}"
        for transformation in ("rotation", "brightness", "gaussian_noise", "gaussian_blur")
        for severity in ("mild", "moderate", "severe")
    ]
    methods = ("grad_cam", "grad_cam_plus_plus", "score_cam")
    predictions = []
    joint = []
    for leaf_index in range(28):
        sample_id = f"sample_{leaf_index}"
        leaf_id = f"leaf_{leaf_index}"
        for model_id, offset in (("resnet50", 0.0), ("efficientnet_b0", 0.1)):
            for scenario_id in scenarios:
                predictions.append(
                    {
                        "model_id": model_id,
                        "sample_id": sample_id,
                        "leaf_id": leaf_id,
                        "scenario_id": scenario_id,
                        "is_consistent": 1.0,
                        "transformed_is_correct": 1.0,
                        "confidence_drop": 0.2 + offset,
                    }
                )
                for method_index, method in enumerate(methods):
                    excluded = (
                        method == "score_cam"
                        and scenario_id == "gaussian_blur_severe"
                        and (
                            (model_id == "resnet50" and leaf_index >= 20)
                            or (model_id == "efficientnet_b0" and leaf_index < 8)
                        )
                    )
                    joint.append(
                        {
                            "model_id": model_id,
                            "sample_id": sample_id,
                            "leaf_id": leaf_id,
                            "scenario_id": scenario_id,
                            "xai_method": method,
                            "exclusion_reason": "excluded" if excluded else "",
                            "pearson": 0.8 - offset - method_index * 0.01,
                            "ssim": 0.7 - offset - method_index * 0.01,
                            "topk_iou_20": 0.6 - offset - method_index * 0.01,
                        }
                    )
    authorization = {
        "analysis_contrast_type": "rq2_model",
        "analysis_contrast": (
            "resnet50_minus_efficientnet_b0::score_cam::gaussian_blur_severe"
        ),
        "endpoints": ["pearson", "ssim", "topk_iou_20"],
        "expected_common_samples": 12,
        "expected_common_leaves": 12,
        "minimum_common_leaves": 20,
        "support_status": "NOT_ESTIMABLE_INSUFFICIENT_COMMON_LEAVES",
    }
    comparisons = paired_comparisons(
        pd.DataFrame(predictions),
        pd.DataFrame(joint),
        methods=methods,
        alpha=0.05,
        minimum_leaf_count=20,
        non_estimable_authorization=authorization,
    )
    placeholders = [row for row in comparisons if row["estimable"] is False]

    assert len(comparisons) == 576
    assert len(placeholders) == 3
    assert {row["endpoint"] for row in placeholders} == {
        "pearson",
        "ssim",
        "topk_iou_20",
    }
    assert all(row["holm_family_size"] == 12 for row in placeholders)
    assert all(row["holm_estimable_count"] == 11 for row in placeholders)
    assert all(row["holm_reserved_non_estimable_slots"] == 1 for row in placeholders)


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
