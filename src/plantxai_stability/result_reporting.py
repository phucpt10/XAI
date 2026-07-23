"""Fail-closed reporting from the frozen official statistical-analysis artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from plantxai_stability.provenance import sha256_file


FROZEN_ANALYSIS_REPORT_SHA256 = "68a9b47fddb2f203aa35a78645849f4e15c11379dbba6dfc79c9a188557294de"
FROZEN_ANALYSIS_ARTIFACT_SHA256 = {
    "paired_comparisons.csv": ("1ef53d4eecb109872b379ae612e4d1cfd9540d7d20a835aac8ceb89b34ca7d9a"),
    "prediction_class_summary.csv": (
        "564f6505827eab074ac2903058d2f5abb33e696ca4742a4e774b252fb9d84f0a"
    ),
    "prediction_summary.csv": ("65201d8003c1ad9d269e6c20522f81bb5e0c01f2f39fa2d7853f9058709121d6"),
    "rq3_association_summary.csv": (
        "f7e354bb6dac7e0ebb7f5d217e47a695ad1501282042ff1dc0fa8dab6c734a84"
    ),
    "xai_exclusion_audit.csv": ("ecb9da6925f033e08e6cf9fec95410084798f93c0ae510f16b908e468c8c2ba9"),
    "xai_summary.csv": ("1a32f6f3b9f1d079a297ff1219c154222353a8bcae719b54af74b9aaf4bced74"),
}
FROZEN_ROW_COUNTS = {
    "source_prediction_rows": 40632,
    "source_joint_rows": 121896,
    "prediction_summary_rows": 96,
    "prediction_class_summary_rows": 480,
    "xai_summary_rows": 432,
    "xai_exclusion_audit_rows": 167,
    "paired_comparison_rows": 576,
    "paired_estimable_rows": 573,
    "paired_non_estimable_rows": 3,
    "rq3_association_rows": 72,
}
RQ1_PRIMARY = ("is_consistent", "transformed_is_correct", "confidence_drop")
RQ2_PRIMARY = ("pearson", "ssim", "topk_iou_20")
SEVERITIES = ("mild", "moderate", "severe")
TRANSFORMATIONS = ("rotation", "brightness", "gaussian_noise", "gaussian_blur")
MODELS = ("resnet50", "efficientnet_b0")
METHODS = ("grad_cam", "grad_cam_plus_plus", "score_cam")
REPORTING_TABLES = (
    "table_rq1_primary_summary.csv",
    "table_rq1_class_summary.csv",
    "table_rq2_primary_summary.csv",
    "table_paired_comparisons.csv",
    "table_inferential_overview.csv",
    "table_non_estimable_comparisons.csv",
    "table_xai_exclusion_audit.csv",
    "table_rq3_exploratory_associations.csv",
)
REPORTING_FIGURES = (
    "figure_rq1_is_consistent.png",
    "figure_rq1_transformed_is_correct.png",
    "figure_rq1_confidence_drop.png",
    "figure_rq2_pearson.png",
    "figure_rq2_ssim.png",
    "figure_rq2_topk_iou_20.png",
)
REPORTING_SUMMARIES = (
    "frozen_results_summary.json",
    "frozen_results_summary.md",
    "results_reporting_report.json",
)
NON_ESTIMABLE_CONTRAST = "resnet50_minus_efficientnet_b0::score_cam::gaussian_blur_severe"
NON_ESTIMABLE_STATUS = "NOT_ESTIMABLE_INSUFFICIENT_COMMON_LEAVES"


def load_results_decision(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Results Decision Record must be a YAML mapping")
    if payload.get("decision_id") != "DR-RESULTS-001":
        raise ValueError("Expected DR-RESULTS-001")
    if payload.get("status") != "approved" or payload.get("approved_by") != "project_owner":
        raise ValueError("DR-RESULTS-001 lacks project-owner approval")
    return payload


def load_and_validate_frozen_results(
    *,
    analysis_dir: Path,
    decision: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    source = decision.get("source_analysis", {})
    report_path = analysis_dir / "official_analysis_report.json"
    if not report_path.is_file():
        raise ValueError(f"Missing frozen analysis report: {report_path}")
    if source.get("report_sha256") != FROZEN_ANALYSIS_REPORT_SHA256:
        raise ValueError("DR-RESULTS-001 contains an unapproved report SHA-256")
    if sha256_file(report_path) != FROZEN_ANALYSIS_REPORT_SHA256:
        raise ValueError("Frozen analysis report SHA-256 mismatch")
    if source.get("artifact_sha256") != FROZEN_ANALYSIS_ARTIFACT_SHA256:
        raise ValueError("DR-RESULTS-001 child artifact registry mismatch")
    for name, expected_hash in FROZEN_ANALYSIS_ARTIFACT_SHA256.items():
        path = analysis_dir / name
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"Frozen analysis artifact SHA-256 mismatch: {path}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_report_fields = {
        "run_type": "authorized_official_test_statistical_analysis",
        "official_test_result": True,
        "official_test_pixels_accessed": False,
        "analysis_decision_id": source.get("analysis_decision_id"),
        "analysis_decision_record_sha256": source.get("analysis_decision_record_sha256"),
        "analysis_support_decision_id": source.get("analysis_support_decision_id"),
        "analysis_support_decision_record_sha256": source.get(
            "analysis_support_decision_record_sha256"
        ),
        "analysis_support_audit_report_sha256": source.get("analysis_support_audit_report_sha256"),
    }
    mismatches = [key for key, value in expected_report_fields.items() if report.get(key) != value]
    if mismatches:
        raise ValueError(f"Frozen analysis report lineage mismatch: {mismatches}")
    if report.get("runtime", {}).get("git_commit") != source.get("git_commit"):
        raise ValueError("Frozen analysis code revision mismatch")
    if report.get("row_counts") != FROZEN_ROW_COUNTS:
        raise ValueError("Frozen analysis report row counts mismatch")
    if decision.get("frozen_row_counts") != FROZEN_ROW_COUNTS:
        raise ValueError("DR-RESULTS-001 row-count contract mismatch")
    output_contract = {
        "tables": list(REPORTING_TABLES),
        "figures": list(REPORTING_FIGURES),
        "summaries": list(REPORTING_SUMMARIES),
    }
    if decision.get("authorized_reporting_outputs") != output_contract:
        raise ValueError("DR-RESULTS-001 reporting output contract mismatch")
    if report.get("artifact_sha256") != FROZEN_ANALYSIS_ARTIFACT_SHA256:
        raise ValueError("Frozen analysis report child artifact registry mismatch")
    if not report.get("acceptance_criteria") or not all(report["acceptance_criteria"].values()):
        raise ValueError("Frozen analysis report contains a failed criterion")

    frames = {
        name: pd.read_csv(analysis_dir / name, keep_default_na=False)
        for name in FROZEN_ANALYSIS_ARTIFACT_SHA256
    }
    observed_rows = {
        "prediction_summary_rows": len(frames["prediction_summary.csv"]),
        "prediction_class_summary_rows": len(frames["prediction_class_summary.csv"]),
        "xai_summary_rows": len(frames["xai_summary.csv"]),
        "xai_exclusion_audit_rows": len(frames["xai_exclusion_audit.csv"]),
        "paired_comparison_rows": len(frames["paired_comparisons.csv"]),
        "rq3_association_rows": len(frames["rq3_association_summary.csv"]),
    }
    for key, count in observed_rows.items():
        if count != FROZEN_ROW_COUNTS[key]:
            raise ValueError(f"Frozen artifact row-count mismatch: {key}")
    _validate_paired_comparisons(
        frames["paired_comparisons.csv"],
        decision.get("non_estimable_contract", {}),
    )
    _validate_reporting_scope(frames)
    constraints = decision.get("reporting_constraints", {})
    expected_constraints = {
        "source_artifacts_are_read_only": True,
        "official_test_pixels_accessed": False,
        "predictions_or_cams_recomputed": False,
        "endpoints_or_holm_families_changed": False,
        "non_estimable_rows_retained": True,
        "non_estimable_rows_interpreted_as_non_significant": False,
        "model_or_method_selection_from_results": False,
        "tuning_from_results": False,
        "rq3_remains_exploratory": True,
        "severity_interpreted_within_transformation_only": True,
        "rotation_claim_remains_zero_fill_operator_specific": True,
    }
    if constraints != expected_constraints:
        raise ValueError("DR-RESULTS-001 reporting constraints mismatch")
    return report, frames


def build_reporting_tables(
    frames: Mapping[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    prediction = frames["prediction_summary.csv"].copy()
    prediction_class = frames["prediction_class_summary.csv"].copy()
    xai = frames["xai_summary.csv"].copy()
    paired = frames["paired_comparisons.csv"].copy()
    exclusions = frames["xai_exclusion_audit.csv"].copy()
    rq3 = frames["rq3_association_summary.csv"].copy()

    rq1_primary = prediction.loc[prediction["endpoint"].isin(RQ1_PRIMARY)].copy()
    rq1_primary = rq1_primary.sort_values(["endpoint", "model_id", "transformation", "severity"])
    rq1_class = prediction_class.sort_values(
        ["endpoint", "model_id", "transformation", "severity", "class_name"]
    )
    rq2_primary = xai.loc[xai["endpoint"].isin(RQ2_PRIMARY)].copy()
    rq2_primary = rq2_primary.sort_values(
        ["endpoint", "model_id", "xai_method", "transformation", "severity"]
    )

    paired["estimable"] = _boolean_series(paired["estimable"], "estimable")
    estimable = paired.loc[paired["estimable"]].copy()
    estimable["reject_h0_holm"] = _boolean_series(
        estimable["reject_h0_holm"],
        "reject_h0_holm",
    )
    overview = (
        paired.groupby(["contrast_type", "endpoint"], sort=True)
        .size()
        .reset_index(name="planned_count")
    )
    estimable_counts = (
        estimable.groupby(["contrast_type", "endpoint"], sort=True)
        .agg(
            estimable_count=("contrast", "size"),
            holm_rejection_count=("reject_h0_holm", "sum"),
        )
        .reset_index()
    )
    overview = overview.merge(
        estimable_counts,
        on=["contrast_type", "endpoint"],
        how="left",
        validate="one_to_one",
    )
    overview["estimable_count"] = overview["estimable_count"].fillna(0).astype(int)
    overview["holm_rejection_count"] = overview["holm_rejection_count"].fillna(0).astype(int)
    overview["non_estimable_count"] = overview["planned_count"] - overview["estimable_count"]
    overview["holm_non_rejection_count"] = (
        overview["estimable_count"] - overview["holm_rejection_count"]
    )
    non_estimable = paired.loc[~paired["estimable"]].copy()

    return {
        "table_rq1_primary_summary.csv": rq1_primary.reset_index(drop=True),
        "table_rq1_class_summary.csv": rq1_class.reset_index(drop=True),
        "table_rq2_primary_summary.csv": rq2_primary.reset_index(drop=True),
        "table_paired_comparisons.csv": paired.reset_index(drop=True),
        "table_inferential_overview.csv": overview.reset_index(drop=True),
        "table_non_estimable_comparisons.csv": non_estimable.reset_index(drop=True),
        "table_xai_exclusion_audit.csv": exclusions.sort_values(
            ["model_id", "xai_method", "scenario_id", "exclusion_reason"]
        ).reset_index(drop=True),
        "table_rq3_exploratory_associations.csv": rq3.sort_values(
            ["model_id", "xai_method", "transformation", "endpoint"]
        ).reset_index(drop=True),
    }


def build_frozen_results_summary(
    *,
    report: Mapping[str, Any],
    tables: Mapping[str, pd.DataFrame],
) -> dict[str, Any]:
    rq1 = tables["table_rq1_primary_summary.csv"]
    rq2 = tables["table_rq2_primary_summary.csv"]
    rq3 = tables["table_rq3_exploratory_associations.csv"]
    overview = tables["table_inferential_overview.csv"]
    non_estimable = tables["table_non_estimable_comparisons.csv"]

    return {
        "summary_role": "read_only_reporting_from_frozen_official_analysis",
        "source_analysis_report_sha256": FROZEN_ANALYSIS_REPORT_SHA256,
        "row_counts": dict(report["row_counts"]),
        "rq1_estimate_ranges_by_model_and_endpoint": _estimate_ranges(
            rq1,
            ["model_id", "endpoint"],
        ),
        "rq2_estimate_ranges_by_model_method_and_endpoint": _estimate_ranges(
            rq2,
            ["model_id", "xai_method", "endpoint"],
        ),
        "rq3_exploratory_correlation_ranges": _estimate_ranges(
            rq3,
            ["model_id", "xai_method", "endpoint"],
        ),
        "inferential_overview": overview.to_dict(orient="records"),
        "non_estimable_comparisons": non_estimable.to_dict(orient="records"),
        "interpretation_constraints": {
            "descriptive_ranges_are_not_selection_rules": True,
            "holm_rejections_are_reported_only_for_estimable_rows": True,
            "non_estimable_is_not_equivalent_to_non_significant": True,
            "rq3_is_exploratory_without_hypothesis_tests": True,
            "severity_is_ordinal_within_each_transformation_only": True,
            "rotation_prediction_claim_is_zero_fill_operator_specific": True,
        },
    }


def render_reporting_figures(
    *,
    tables: Mapping[str, pd.DataFrame],
    output_dir: Path,
) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the optional 'report' dependencies") from exc

    paths: list[Path] = []
    rq1 = tables["table_rq1_primary_summary.csv"]
    for endpoint in RQ1_PRIMARY:
        path = output_dir / f"figure_rq1_{endpoint}.png"
        _plot_rq1(plt, rq1, endpoint, path)
        paths.append(path)
    rq2 = tables["table_rq2_primary_summary.csv"]
    for endpoint in RQ2_PRIMARY:
        path = output_dir / f"figure_rq2_{endpoint}.png"
        _plot_rq2(plt, rq2, endpoint, path)
        paths.append(path)
    return paths


def render_summary_markdown(summary: Mapping[str, Any]) -> str:
    rows = summary["row_counts"]
    overview = summary["inferential_overview"]
    rq1_ranges = summary["rq1_estimate_ranges_by_model_and_endpoint"]
    rq2_ranges = summary["rq2_estimate_ranges_by_model_method_and_endpoint"]
    rq3_ranges = summary["rq3_exploratory_correlation_ranges"]
    lines = [
        "# Frozen Official Results Reporting Summary",
        "",
        f"- Source analysis report SHA-256: `{summary['source_analysis_report_sha256']}`",
        f"- Planned paired rows: {rows['paired_comparison_rows']}",
        f"- Estimable paired rows: {rows['paired_estimable_rows']}",
        f"- Non-estimable paired rows: {rows['paired_non_estimable_rows']}",
        "- Official-test pixels accessed during reporting: no",
        "- Predictions or CAMs recomputed during reporting: no",
        "",
        "## RQ1 descriptive estimate ranges",
        "",
        "| Model | Endpoint | Minimum | Maximum |",
        "|---|---|---:|---:|",
    ]
    for row in rq1_ranges:
        lines.append("| {model_id} | {endpoint} | {minimum:.6f} | {maximum:.6f} |".format(**row))
    lines.extend(
        [
            "",
            "## RQ2 descriptive estimate ranges",
            "",
            "| Model | CAM method | Endpoint | Minimum | Maximum |",
            "|---|---|---|---:|---:|",
        ]
    )
    for row in rq2_ranges:
        lines.append(
            "| {model_id} | {xai_method} | {endpoint} | {minimum:.6f} | {maximum:.6f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## RQ3 exploratory correlation ranges",
            "",
            "| Model | CAM method | Endpoint | Minimum | Maximum |",
            "|---|---|---|---:|---:|",
        ]
    )
    for row in rq3_ranges:
        lines.append(
            "| {model_id} | {xai_method} | {endpoint} | {minimum:.6f} | {maximum:.6f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Inferential coverage",
            "",
            "| Contrast scope | Endpoint | Planned | Estimable | Non-estimable | Holm rejections |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in overview:
        lines.append(
            "| {contrast_type} | {endpoint} | {planned_count} | "
            "{estimable_count} | {non_estimable_count} | "
            "{holm_rejection_count} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Required interpretation constraints",
            "",
            "- The three Score-CAM × Gaussian-blur-severe cross-model endpoint rows "
            "are non-estimable, not non-significant.",
            "- Descriptive ranges and Holm rejection counts are reporting summaries, "
            "not model or XAI selection rules.",
            "- RQ3 remains exploratory and has no hypothesis tests.",
            "- Severity comparisons are meaningful only within a transformation.",
            "- Rotation prediction claims remain specific to the zero-filled operator.",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_paired_comparisons(
    frame: pd.DataFrame,
    contract: Mapping[str, Any],
) -> None:
    estimable = _boolean_series(frame["estimable"], "estimable")
    non_estimable = frame.loc[~estimable]
    if int(estimable.sum()) != 573 or len(non_estimable) != 3:
        raise ValueError("Frozen paired-comparison estimability counts mismatch")
    expected_contract = {
        "contrast": NON_ESTIMABLE_CONTRAST,
        "endpoints": list(RQ2_PRIMARY),
        "support_status": NON_ESTIMABLE_STATUS,
        "common_sample_count": 14,
        "common_leaf_count": 12,
        "minimum_common_leaf_count": 20,
        "inferential_fields_empty": True,
        "interpretation_as_non_significant_prohibited": True,
    }
    if dict(contract) != expected_contract:
        raise ValueError("DR-RESULTS-001 non-estimable contract mismatch")
    if (
        set(non_estimable["contrast"]) != {NON_ESTIMABLE_CONTRAST}
        or set(non_estimable["endpoint"]) != set(RQ2_PRIMARY)
        or set(non_estimable["support_status"]) != {NON_ESTIMABLE_STATUS}
        or set(non_estimable["n_common_sample_keys"].astype(int)) != {14}
        or set(non_estimable["n_leaf_pairs"].astype(int)) != {12}
        or set(non_estimable["holm_family_size"].astype(int)) != {12}
        or set(non_estimable["holm_estimable_count"].astype(int)) != {11}
        or set(non_estimable["holm_reserved_non_estimable_slots"].astype(int)) != {1}
        or set(non_estimable["holm_reserved_slot_value_for_adjustment_only"].astype(float)) != {1.0}
    ):
        raise ValueError("Frozen non-estimable rows differ from their contract")
    empty_fields = (
        "left_mean",
        "right_mean",
        "mean_difference_left_minus_right",
        "wilcoxon_statistic",
        "p_value_raw",
        "rank_biserial",
        "p_value_holm",
        "reject_h0_holm",
    )
    if any(non_estimable[column].ne("").any() for column in empty_fields):
        raise ValueError("A non-estimable row contains an inferential result")
    numeric_fields = (
        "left_mean",
        "right_mean",
        "mean_difference_left_minus_right",
        "wilcoxon_statistic",
        "p_value_raw",
        "rank_biserial",
        "p_value_holm",
    )
    estimable_frame = frame.loc[estimable]
    if any(
        not np.isfinite(pd.to_numeric(estimable_frame[column])).all() for column in numeric_fields
    ):
        raise ValueError("An estimable paired-comparison field is not finite")


def _validate_reporting_scope(frames: Mapping[str, pd.DataFrame]) -> None:
    prediction = frames["prediction_summary.csv"]
    xai = frames["xai_summary.csv"]
    if set(prediction["model_id"]) != set(MODELS):
        raise ValueError("Frozen prediction models differ from the reporting scope")
    if set(prediction["transformation"]) != set(TRANSFORMATIONS):
        raise ValueError("Frozen prediction transformations differ from the reporting scope")
    if set(prediction["severity"]) != set(SEVERITIES):
        raise ValueError("Frozen prediction severities differ from the reporting scope")
    if set(xai["model_id"]) != set(MODELS) or set(xai["xai_method"]) != set(METHODS):
        raise ValueError("Frozen XAI scope differs from the reporting scope")
    if set(xai["transformation"]) != set(TRANSFORMATIONS):
        raise ValueError("Frozen XAI transformations differ from the reporting scope")
    if set(xai["severity"]) != set(SEVERITIES):
        raise ValueError("Frozen XAI severities differ from the reporting scope")


def _boolean_series(values: pd.Series, column: str) -> pd.Series:
    mapping = {
        True: True,
        False: False,
        "True": True,
        "False": False,
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }
    converted = values.map(mapping)
    if converted.isna().any():
        raise ValueError(f"Invalid boolean values in {column}")
    return converted.astype(bool)


def _estimate_ranges(
    frame: pd.DataFrame,
    grouping: list[str],
) -> list[dict[str, Any]]:
    output = (
        frame.assign(estimate=pd.to_numeric(frame["estimate"]))
        .groupby(grouping, sort=True)["estimate"]
        .agg(minimum="min", maximum="max")
        .reset_index()
    )
    return output.to_dict(orient="records")


def _plot_rq1(plt: Any, frame: pd.DataFrame, endpoint: str, path: Path) -> None:
    subset = frame.loc[frame["endpoint"].eq(endpoint)].copy()
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), sharey=True)
    colors = {"resnet50": "#1f77b4", "efficientnet_b0": "#d95f02"}
    markers = {"resnet50": "o", "efficientnet_b0": "s"}
    for axis, transformation in zip(axes, TRANSFORMATIONS):
        for model_id in MODELS:
            rows = subset.loc[
                subset["transformation"].eq(transformation) & subset["model_id"].eq(model_id)
            ].copy()
            rows["severity"] = pd.Categorical(
                rows["severity"],
                categories=SEVERITIES,
                ordered=True,
            )
            rows = rows.sort_values("severity")
            estimate = pd.to_numeric(rows["estimate"]).to_numpy()
            lower = pd.to_numeric(rows["lower"]).to_numpy()
            upper = pd.to_numeric(rows["upper"]).to_numpy()
            axis.errorbar(
                range(len(SEVERITIES)),
                estimate,
                yerr=np.vstack([estimate - lower, upper - estimate]),
                label=model_id,
                color=colors[model_id],
                marker=markers[model_id],
                linewidth=1.5,
                capsize=2,
            )
        axis.set_title(transformation.replace("_", " "))
        axis.set_xticks(range(len(SEVERITIES)), SEVERITIES, rotation=25)
        axis.grid(alpha=0.25)
        if endpoint in {"is_consistent", "transformed_is_correct"}:
            axis.set_ylim(-0.02, 1.02)
        else:
            axis.axhline(0.0, color="#666666", linewidth=0.8)
    axes[0].set_ylabel(endpoint.replace("_", " "))
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle(f"RQ1: {endpoint.replace('_', ' ')}")
    fig.tight_layout()
    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
        metadata={"Software": "PlantXAI-Stability"},
    )
    plt.close(fig)


def _plot_rq2(plt: Any, frame: pd.DataFrame, endpoint: str, path: Path) -> None:
    subset = frame.loc[frame["endpoint"].eq(endpoint)].copy()
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharex=True, sharey=True)
    colors = {
        "grad_cam": "#1b9e77",
        "grad_cam_plus_plus": "#7570b3",
        "score_cam": "#d95f02",
    }
    markers = {"grad_cam": "o", "grad_cam_plus_plus": "^", "score_cam": "s"}
    for row_index, model_id in enumerate(MODELS):
        for column_index, transformation in enumerate(TRANSFORMATIONS):
            axis = axes[row_index, column_index]
            for method in METHODS:
                rows = subset.loc[
                    subset["model_id"].eq(model_id)
                    & subset["transformation"].eq(transformation)
                    & subset["xai_method"].eq(method)
                ].copy()
                rows["severity"] = pd.Categorical(
                    rows["severity"],
                    categories=SEVERITIES,
                    ordered=True,
                )
                rows = rows.sort_values("severity")
                estimate = pd.to_numeric(rows["estimate"]).to_numpy()
                lower = pd.to_numeric(rows["lower"]).to_numpy()
                upper = pd.to_numeric(rows["upper"]).to_numpy()
                axis.errorbar(
                    range(len(SEVERITIES)),
                    estimate,
                    yerr=np.vstack([estimate - lower, upper - estimate]),
                    label=method,
                    color=colors[method],
                    marker=markers[method],
                    linewidth=1.3,
                    capsize=2,
                )
            if row_index == 0:
                axis.set_title(transformation.replace("_", " "))
            if column_index == 0:
                axis.set_ylabel(f"{model_id}\n{endpoint.replace('_', ' ')}")
            axis.set_xticks(range(len(SEVERITIES)), SEVERITIES, rotation=25)
            axis.grid(alpha=0.25)
            axis.set_ylim((-1.02, 1.02) if endpoint == "pearson" else (-0.02, 1.02))
    axes[0, -1].legend(frameon=False, fontsize=8)
    fig.suptitle(f"RQ2: {endpoint.replace('_', ' ')}")
    fig.tight_layout()
    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
        metadata={"Software": "PlantXAI-Stability"},
    )
    plt.close(fig)
