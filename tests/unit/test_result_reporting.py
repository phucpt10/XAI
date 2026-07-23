from __future__ import annotations

from pathlib import Path

import pandas as pd

from plantxai_stability.result_reporting import (
    NON_ESTIMABLE_CONTRAST,
    NON_ESTIMABLE_STATUS,
    REPORTING_FIGURES,
    REPORTING_SUMMARIES,
    REPORTING_TABLES,
    RQ2_PRIMARY,
    _validate_paired_comparisons,
    build_frozen_results_summary,
    build_reporting_tables,
    load_results_decision,
    render_reporting_figures,
    render_summary_markdown,
)


def _non_estimable_contract() -> dict[str, object]:
    return {
        "contrast": NON_ESTIMABLE_CONTRAST,
        "endpoints": list(RQ2_PRIMARY),
        "support_status": NON_ESTIMABLE_STATUS,
        "common_sample_count": 14,
        "common_leaf_count": 12,
        "minimum_common_leaf_count": 20,
        "inferential_fields_empty": True,
        "interpretation_as_non_significant_prohibited": True,
    }


def _paired_row(*, estimable: bool, endpoint: str = "pearson") -> dict[str, object]:
    if estimable:
        return {
            "family_id": "rq2_model::pearson::grad_cam",
            "contrast_type": "rq2_model",
            "contrast": "estimable_contrast",
            "endpoint": endpoint,
            "estimable": True,
            "support_status": "ESTIMABLE",
            "n_common_sample_keys": 100,
            "n_leaf_pairs": 50,
            "left_mean": 0.8,
            "right_mean": 0.7,
            "mean_difference_left_minus_right": 0.1,
            "wilcoxon_statistic": 10.0,
            "p_value_raw": 0.01,
            "rank_biserial": 0.4,
            "p_value_holm": 0.12,
            "reject_h0_holm": False,
            "alpha": 0.05,
            "holm_family_size": 12,
            "holm_estimable_count": 12,
            "holm_reserved_non_estimable_slots": 0,
            "holm_reserved_slot_value_for_adjustment_only": "",
        }
    return {
        "family_id": f"rq2_model::{endpoint}::score_cam",
        "contrast_type": "rq2_model",
        "contrast": NON_ESTIMABLE_CONTRAST,
        "endpoint": endpoint,
        "estimable": False,
        "support_status": NON_ESTIMABLE_STATUS,
        "n_common_sample_keys": 14,
        "n_leaf_pairs": 12,
        "left_mean": "",
        "right_mean": "",
        "mean_difference_left_minus_right": "",
        "wilcoxon_statistic": "",
        "p_value_raw": "",
        "rank_biserial": "",
        "p_value_holm": "",
        "reject_h0_holm": "",
        "alpha": 0.05,
        "holm_family_size": 12,
        "holm_estimable_count": 11,
        "holm_reserved_non_estimable_slots": 1,
        "holm_reserved_slot_value_for_adjustment_only": 1.0,
    }


def test_results_decision_freezes_exact_analysis() -> None:
    decision = load_results_decision(
        Path("configs/protocol/v0.9/decision_records/DR-RESULTS-001.yaml")
    )
    assert decision["source_analysis"]["report_sha256"] == (
        "68a9b47fddb2f203aa35a78645849f4e15c11379dbba6dfc79c9a188557294de"
    )
    assert decision["frozen_row_counts"]["paired_estimable_rows"] == 573
    assert decision["frozen_row_counts"]["paired_non_estimable_rows"] == 3
    assert decision["authorized_reporting_outputs"] == {
        "tables": list(REPORTING_TABLES),
        "figures": list(REPORTING_FIGURES),
        "summaries": list(REPORTING_SUMMARIES),
    }


def test_frozen_paired_rows_retain_exact_non_estimable_contract() -> None:
    rows = [_paired_row(estimable=True) for _ in range(573)]
    rows.extend(_paired_row(estimable=False, endpoint=endpoint) for endpoint in RQ2_PRIMARY)
    _validate_paired_comparisons(
        pd.DataFrame(rows),
        _non_estimable_contract(),
    )


def test_reporting_tables_and_summary_keep_non_estimable_rows() -> None:
    paired_rows = [
        _paired_row(estimable=True),
        *[_paired_row(estimable=False, endpoint=endpoint) for endpoint in RQ2_PRIMARY],
    ]
    frames = {
        "prediction_summary.csv": pd.DataFrame(
            [
                {
                    "model_id": "resnet50",
                    "scenario_id": "rotation_mild",
                    "transformation": "rotation",
                    "severity": "mild",
                    "endpoint": "is_consistent",
                    "estimate": 0.9,
                    "lower": 0.8,
                    "upper": 1.0,
                    "n_leaf": 10,
                    "n_value": 10,
                }
            ]
        ),
        "prediction_class_summary.csv": pd.DataFrame(
            [
                {
                    "model_id": "resnet50",
                    "scenario_id": "rotation_mild",
                    "transformation": "rotation",
                    "severity": "mild",
                    "class_name": "class_a",
                    "endpoint": "is_consistent",
                    "estimate": 0.9,
                    "n_leaf": 10,
                    "n_value": 10,
                    "interval_status": "descriptive_only",
                }
            ]
        ),
        "xai_summary.csv": pd.DataFrame(
            [
                {
                    "model_id": "resnet50",
                    "xai_method": "grad_cam",
                    "scenario_id": "rotation_mild",
                    "transformation": "rotation",
                    "severity": "mild",
                    "endpoint": "pearson",
                    "endpoint_role": "primary",
                    "estimate": 0.7,
                    "lower": 0.6,
                    "upper": 0.8,
                    "n_leaf": 10,
                    "n_value": 10,
                }
            ]
        ),
        "paired_comparisons.csv": pd.DataFrame(paired_rows),
        "xai_exclusion_audit.csv": pd.DataFrame(
            [
                {
                    "model_id": "resnet50",
                    "xai_method": "grad_cam",
                    "scenario_id": "rotation_mild",
                    "exclusion_reason": "included",
                    "row_count": 10,
                }
            ]
        ),
        "rq3_association_summary.csv": pd.DataFrame(
            [
                {
                    "model_id": "resnet50",
                    "xai_method": "grad_cam",
                    "transformation": "rotation",
                    "predictor": "confidence_drop",
                    "endpoint": "pearson",
                    "analysis_role": "exploratory_descriptive_no_hypothesis_test",
                    "estimate": -0.2,
                    "lower": -0.4,
                    "upper": 0.1,
                    "n_leaf": 10,
                }
            ]
        ),
    }
    tables = build_reporting_tables(frames)
    summary = build_frozen_results_summary(
        report={
            "row_counts": {
                "paired_comparison_rows": 4,
                "paired_estimable_rows": 1,
                "paired_non_estimable_rows": 3,
            }
        },
        tables=tables,
    )
    markdown = render_summary_markdown(summary)

    assert len(tables["table_non_estimable_comparisons.csv"]) == 3
    assert summary["interpretation_constraints"][
        "non_estimable_is_not_equivalent_to_non_significant"
    ]
    assert "RQ1 descriptive estimate ranges" in markdown
    assert "RQ2 descriptive estimate ranges" in markdown
    assert "RQ3 exploratory correlation ranges" in markdown
    assert "non-estimable, not non-significant" in markdown


def test_reporting_figures_cover_all_primary_endpoints(tmp_path: Path) -> None:
    rq1_rows = []
    rq2_rows = []
    for model_id in ("resnet50", "efficientnet_b0"):
        for transformation in (
            "rotation",
            "brightness",
            "gaussian_noise",
            "gaussian_blur",
        ):
            for severity_index, severity in enumerate(("mild", "moderate", "severe")):
                for endpoint in (
                    "is_consistent",
                    "transformed_is_correct",
                    "confidence_drop",
                ):
                    estimate = 0.8 - severity_index * 0.1
                    rq1_rows.append(
                        {
                            "model_id": model_id,
                            "transformation": transformation,
                            "severity": severity,
                            "endpoint": endpoint,
                            "estimate": estimate,
                            "lower": estimate - 0.02,
                            "upper": estimate + 0.02,
                        }
                    )
                for method in ("grad_cam", "grad_cam_plus_plus", "score_cam"):
                    for endpoint in ("pearson", "ssim", "topk_iou_20"):
                        estimate = 0.7 - severity_index * 0.1
                        rq2_rows.append(
                            {
                                "model_id": model_id,
                                "xai_method": method,
                                "transformation": transformation,
                                "severity": severity,
                                "endpoint": endpoint,
                                "estimate": estimate,
                                "lower": estimate - 0.02,
                                "upper": estimate + 0.02,
                            }
                        )
    paths = render_reporting_figures(
        tables={
            "table_rq1_primary_summary.csv": pd.DataFrame(rq1_rows),
            "table_rq2_primary_summary.csv": pd.DataFrame(rq2_rows),
        },
        output_dir=tmp_path,
    )

    assert len(paths) == 6
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
