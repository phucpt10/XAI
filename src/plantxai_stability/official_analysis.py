"""Fail-closed analysis of the two merged official-test result trees."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import yaml

from plantxai_stability.config import ResolvedConfig
from plantxai_stability.provenance import sha256_file
from plantxai_stability.statistics import holm_adjust, paired_wilcoxon


RQ1_PRIMARY = ("is_consistent", "transformed_is_correct", "confidence_drop")
RQ1_SECONDARY = ("absolute_confidence_delta",)
RQ2_PRIMARY = ("pearson", "ssim", "topk_iou_20")
RQ2_SECONDARY = ("cosine", "topk_iou_10", "topk_iou_30")
APPROVED_MERGE_REPORT_SHA256 = {
    "resnet50": "32610c640f3f35455bcdd998a3f0bb1a09eac2ff34f40dcdb51c0d86e8ac7c1e",
    "efficientnet_b0": ("0cc81bef79c9bf273eff753eef74fa737d35d6b8abef7951acd3fa2e6d534401"),
}
EXPECTED_FIXED_ROW_COUNTS = {
    "prediction_summary.csv": 96,
    "prediction_class_summary.csv": 480,
    "xai_summary.csv": 432,
    "paired_comparisons.csv": 576,
    "rq3_association_summary.csv": 72,
}
BOOLEAN_COLUMNS = ("is_consistent", "transformed_is_correct", "is_correct")
PREDICTION_NUMERIC_COLUMNS = (
    "confidence",
    "transformed_confidence",
    "confidence_delta",
    "absolute_confidence_delta",
)
JOINT_NUMERIC_COLUMNS = RQ2_PRIMARY + RQ2_SECONDARY


def load_analysis_decision(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Analysis Decision Record must be a YAML mapping")
    if payload.get("decision_id") != "DR-ANALYSIS-001":
        raise ValueError("Expected DR-ANALYSIS-001")
    if payload.get("status") != "approved":
        raise ValueError("DR-ANALYSIS-001 is not approved")
    return payload


def validate_analysis_plan(decision: dict[str, Any], resolved: ResolvedConfig) -> None:
    statistics = decision.get("statistical_policy", {})
    protocol_statistics = resolved.values["statistics"]
    expected = {
        "bootstrap_unit": protocol_statistics["bootstrap_unit"],
        "bootstrap_iterations": protocol_statistics["bootstrap_iterations"],
        "confidence_level": protocol_statistics["confidence_level"],
        "base_seed": protocol_statistics["seed"],
        "zero_method": protocol_statistics["zero_method"],
        "effect_size": protocol_statistics["effect_size"],
        "correction": protocol_statistics["correction"],
        "alpha": protocol_statistics["alpha"],
    }
    mismatches = [key for key, value in expected.items() if statistics.get(key) != value]
    if mismatches:
        raise ValueError(f"Analysis statistics diverge from protocol: {mismatches}")
    lineage = decision.get("scientific_lineage", {})
    if lineage.get("governance_protocol_hash") != resolved.sha256:
        raise ValueError("Analysis Decision Record does not match the protocol hash")
    scope = decision.get("analysis_scope", {})
    if scope.get("models") != list(resolved.values["models"]):
        raise ValueError("Analysis model scope differs from the protocol")
    if scope.get("xai_methods") != list(resolved.values["xai"]["methods"]):
        raise ValueError("Analysis XAI methods differ from the protocol")
    expected_scenarios = _scenario_ids(resolved)
    if scope.get("scenario_ids") != expected_scenarios:
        raise ValueError("Analysis scenarios differ from the protocol")
    endpoints = decision.get("endpoints", {})
    endpoint_checks = {
        "rq1_primary": list(RQ1_PRIMARY),
        "rq1_secondary_descriptive": list(RQ1_SECONDARY),
        "rq2_primary": list(RQ2_PRIMARY),
        "rq2_secondary_descriptive": list(RQ2_SECONDARY),
    }
    if any(endpoints.get(key) != value for key, value in endpoint_checks.items()):
        raise ValueError("Analysis endpoints differ from the implemented fixed plan")
    source_merges = decision.get("source_merges", {})
    if any(
        source_merges.get(model, {}).get("report_sha256") != expected_hash
        for model, expected_hash in APPROVED_MERGE_REPORT_SHA256.items()
    ):
        raise ValueError("Analysis source merge SHA-256 values are not approved")
    expected_rows = decision.get("outputs", {}).get("expected_fixed_row_counts")
    if expected_rows != EXPECTED_FIXED_ROW_COUNTS:
        raise ValueError("Analysis output coverage differs from the fixed plan")


def load_and_validate_merges(
    *,
    merge_dirs: dict[str, Path],
    decision: dict[str, Any],
    resolved: ResolvedConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
    source_merges = decision["source_merges"]
    models = list(resolved.values["models"])
    if sorted(merge_dirs) != sorted(models):
        raise ValueError("Exactly the two declared model merge directories are required")
    predictions: list[pd.DataFrame] = []
    joint: list[pd.DataFrame] = []
    reports: dict[str, dict[str, Any]] = {}
    for model_id in models:
        prediction_frame, joint_frame, report = _load_merge(
            merge_dir=merge_dirs[model_id],
            model_id=model_id,
            expected=source_merges[model_id],
            decision=decision,
            resolved=resolved,
        )
        predictions.append(prediction_frame)
        joint.append(joint_frame)
        reports[model_id] = report
    _validate_shared_lineage(reports, decision)
    combined_predictions = pd.concat(predictions, ignore_index=True)
    combined_joint = pd.concat(joint, ignore_index=True)
    _validate_cross_model_identity(combined_predictions)
    _validate_joint_prediction_identity(combined_predictions, combined_joint)
    return combined_predictions, combined_joint, reports


def prediction_summaries(
    frame: pd.DataFrame,
    *,
    iterations: int,
    confidence_level: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    endpoints = RQ1_PRIMARY + RQ1_SECONDARY
    overall: list[dict[str, Any]] = []
    by_class: list[dict[str, Any]] = []
    grouping = ["model_id", "scenario_id", "transformation", "severity"]
    for keys, group in frame.groupby(grouping, sort=True):
        group_id = "rq1|" + "|".join(map(str, keys))
        intervals = cluster_bootstrap_means(
            group,
            endpoints,
            iterations=iterations,
            confidence_level=confidence_level,
            seed=_derived_seed(seed, group_id),
        )
        for endpoint in endpoints:
            overall.append(
                {
                    **dict(zip(grouping, keys)),
                    "endpoint": endpoint,
                    **intervals[endpoint],
                }
            )
        for class_name, class_group in group.groupby("true_class_name", sort=True):
            for endpoint in endpoints:
                values = class_group[endpoint].to_numpy(dtype=float)
                by_class.append(
                    {
                        **dict(zip(grouping, keys)),
                        "class_name": class_name,
                        "endpoint": endpoint,
                        "estimate": float(values.mean()),
                        "n_leaf": int(class_group["leaf_id"].nunique()),
                        "n_value": int(len(values)),
                        "interval_status": "descriptive_only",
                    }
                )
    return overall, by_class


def xai_summaries(
    frame: pd.DataFrame,
    *,
    iterations: int,
    confidence_level: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid = frame.loc[frame["exclusion_reason"].eq("")].copy()
    grouping = ["model_id", "xai_method", "scenario_id"]
    summaries: list[dict[str, Any]] = []
    for keys, group in valid.groupby(grouping, sort=True):
        group_id = "rq2|" + "|".join(map(str, keys))
        intervals = cluster_bootstrap_means(
            group,
            RQ2_PRIMARY,
            iterations=iterations,
            confidence_level=confidence_level,
            seed=_derived_seed(seed, group_id),
        )
        transformation, severity = _split_scenario(str(keys[2]))
        for endpoint in RQ2_PRIMARY:
            summaries.append(
                {
                    **dict(zip(grouping, keys)),
                    "transformation": transformation,
                    "severity": severity,
                    "endpoint": endpoint,
                    "endpoint_role": "primary",
                    **intervals[endpoint],
                }
            )
        for endpoint in RQ2_SECONDARY:
            values = group[endpoint].to_numpy(dtype=float)
            summaries.append(
                {
                    **dict(zip(grouping, keys)),
                    "transformation": transformation,
                    "severity": severity,
                    "endpoint": endpoint,
                    "endpoint_role": "secondary_descriptive",
                    "estimate": float(values.mean()),
                    "lower": "",
                    "upper": "",
                    "n_leaf": int(group["leaf_id"].nunique()),
                    "n_value": int(len(values)),
                }
            )
    exclusions = (
        frame.assign(exclusion_reason=frame["exclusion_reason"].replace("", "included"))
        .groupby(
            ["model_id", "xai_method", "scenario_id", "exclusion_reason"],
            sort=True,
        )
        .size()
        .reset_index(name="row_count")
    )
    return summaries, exclusions.to_dict(orient="records")


def paired_comparisons(
    predictions: pd.DataFrame,
    joint: pd.DataFrame,
    *,
    methods: Sequence[str],
    alpha: float,
    minimum_leaf_count: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scenarios = sorted(predictions["scenario_id"].unique())
    for endpoint in RQ1_PRIMARY:
        family_id = f"rq1_model::{endpoint}"
        for scenario in scenarios:
            subset = predictions.loc[predictions["scenario_id"].eq(scenario)]
            rows.append(
                _paired_contrast(
                    subset,
                    left_filter={"model_id": "resnet50"},
                    right_filter={"model_id": "efficientnet_b0"},
                    endpoint=endpoint,
                    family_id=family_id,
                    contrast_type="rq1_model",
                    contrast=f"resnet50_minus_efficientnet_b0::{scenario}",
                    minimum_leaf_count=minimum_leaf_count,
                )
            )

    valid = joint.loc[joint["exclusion_reason"].eq("")].copy()
    for endpoint in RQ2_PRIMARY:
        for method in methods:
            family_id = f"rq2_model::{endpoint}::{method}"
            for scenario in scenarios:
                subset = valid.loc[
                    valid["scenario_id"].eq(scenario) & valid["xai_method"].eq(method)
                ]
                rows.append(
                    _paired_contrast(
                        subset,
                        left_filter={"model_id": "resnet50"},
                        right_filter={"model_id": "efficientnet_b0"},
                        endpoint=endpoint,
                        family_id=family_id,
                        contrast_type="rq2_model",
                        contrast=(f"resnet50_minus_efficientnet_b0::{method}::{scenario}"),
                        minimum_leaf_count=minimum_leaf_count,
                    )
                )

    method_pairs = list(itertools.combinations(methods, 2))
    for model_id in sorted(valid["model_id"].unique()):
        for endpoint in RQ2_PRIMARY:
            for scenario in scenarios:
                family_id = f"rq2_method::{model_id}::{endpoint}::{scenario}"
                subset = valid.loc[
                    valid["model_id"].eq(model_id) & valid["scenario_id"].eq(scenario)
                ]
                for left_method, right_method in method_pairs:
                    rows.append(
                        _paired_contrast(
                            subset,
                            left_filter={"xai_method": left_method},
                            right_filter={"xai_method": right_method},
                            endpoint=endpoint,
                            family_id=family_id,
                            contrast_type="rq2_method",
                            contrast=f"{left_method}_minus_{right_method}",
                            minimum_leaf_count=minimum_leaf_count,
                        )
                    )

    severity_pairs = list(itertools.combinations(["mild", "moderate", "severe"], 2))
    transformed = valid.assign(
        transformation=valid["scenario_id"].map(lambda value: _split_scenario(value)[0]),
        severity=valid["scenario_id"].map(lambda value: _split_scenario(value)[1]),
    )
    for model_id in sorted(valid["model_id"].unique()):
        for method in methods:
            for endpoint in RQ2_PRIMARY:
                for transformation in sorted(transformed["transformation"].unique()):
                    family_id = f"rq2_severity::{model_id}::{method}::{endpoint}::{transformation}"
                    subset = transformed.loc[
                        transformed["model_id"].eq(model_id)
                        & transformed["xai_method"].eq(method)
                        & transformed["transformation"].eq(transformation)
                    ]
                    for left_severity, right_severity in severity_pairs:
                        rows.append(
                            _paired_contrast(
                                subset,
                                left_filter={"severity": left_severity},
                                right_filter={"severity": right_severity},
                                endpoint=endpoint,
                                family_id=family_id,
                                contrast_type="rq2_severity",
                                contrast=f"{left_severity}_minus_{right_severity}",
                                minimum_leaf_count=minimum_leaf_count,
                            )
                        )
    _apply_holm(rows, alpha)
    return rows


def rq3_associations(
    predictions: pd.DataFrame,
    joint: pd.DataFrame,
    *,
    iterations: int,
    confidence_level: float,
    seed: int,
) -> list[dict[str, Any]]:
    prediction_values = predictions[["model_id", "sample_id", "scenario_id", "confidence_drop"]]
    valid = joint.loc[joint["exclusion_reason"].eq("")].merge(
        prediction_values,
        on=["model_id", "sample_id", "scenario_id"],
        validate="many_to_one",
    )
    valid["transformation"] = valid["scenario_id"].map(lambda value: _split_scenario(value)[0])
    output: list[dict[str, Any]] = []
    grouping = ["model_id", "xai_method", "transformation"]
    for keys, group in valid.groupby(grouping, sort=True):
        leaf_means = (
            group.groupby("leaf_id", sort=True)[["confidence_drop", *RQ2_PRIMARY]]
            .mean()
            .reset_index()
        )
        for endpoint in RQ2_PRIMARY:
            group_id = "rq3|" + "|".join(map(str, (*keys, endpoint)))
            interval = bootstrap_leaf_correlation(
                leaf_means["confidence_drop"].to_numpy(dtype=float),
                leaf_means[endpoint].to_numpy(dtype=float),
                iterations=iterations,
                confidence_level=confidence_level,
                seed=_derived_seed(seed, group_id),
            )
            output.append(
                {
                    **dict(zip(grouping, keys)),
                    "predictor": "confidence_drop",
                    "endpoint": endpoint,
                    "analysis_role": "exploratory_descriptive_no_hypothesis_test",
                    **interval,
                }
            )
    return output


def cluster_bootstrap_means(
    frame: pd.DataFrame,
    value_columns: Sequence[str],
    *,
    iterations: int,
    confidence_level: float,
    seed: int,
) -> dict[str, dict[str, float | int]]:
    if iterations <= 0 or not 0.0 < confidence_level < 1.0:
        raise ValueError("Invalid bootstrap configuration")
    if frame.empty or frame["leaf_id"].isna().any():
        raise ValueError("Bootstrap data must have resolved leaf identities")
    grouped = frame.groupby("leaf_id", sort=True)[list(value_columns)]
    leaf_sums = grouped.sum().to_numpy(dtype=float)
    leaf_counts = grouped.size().to_numpy(dtype=float)
    values = frame[list(value_columns)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Bootstrap endpoints contain NaN or Inf")
    rng = np.random.default_rng(seed)
    estimates = np.empty((iterations, len(value_columns)), dtype=float)
    n_leaf = len(leaf_counts)
    chunk_size = 256
    for start in range(0, iterations, chunk_size):
        stop = min(start + chunk_size, iterations)
        sampled = rng.integers(0, n_leaf, size=(stop - start, n_leaf))
        sampled_sums = leaf_sums[sampled].sum(axis=1)
        sampled_counts = leaf_counts[sampled].sum(axis=1)
        estimates[start:stop] = sampled_sums / sampled_counts[:, None]
    alpha = (1.0 - confidence_level) / 2.0
    output: dict[str, dict[str, float | int]] = {}
    for index, column in enumerate(value_columns):
        output[column] = {
            "estimate": float(values[:, index].mean()),
            "lower": float(np.quantile(estimates[:, index], alpha)),
            "upper": float(np.quantile(estimates[:, index], 1.0 - alpha)),
            "n_leaf": int(n_leaf),
            "n_value": int(len(frame)),
        }
    return output


def bootstrap_leaf_correlation(
    left: np.ndarray,
    right: np.ndarray,
    *,
    iterations: int,
    confidence_level: float,
    seed: int,
) -> dict[str, float | int]:
    if left.size != right.size or left.size < 3:
        raise ValueError("Correlation arrays must have equal size of at least three")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Correlation arrays contain NaN or Inf")
    estimate = _correlation(left, right)
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=float)
    chunk_size = 256
    for start in range(0, iterations, chunk_size):
        stop = min(start + chunk_size, iterations)
        sampled = rng.integers(0, left.size, size=(stop - start, left.size))
        x = left[sampled]
        y = right[sampled]
        x_centered = x - x.mean(axis=1, keepdims=True)
        y_centered = y - y.mean(axis=1, keepdims=True)
        denominator = np.sqrt(np.sum(x_centered**2, axis=1) * np.sum(y_centered**2, axis=1))
        estimates[start:stop] = np.divide(
            np.sum(x_centered * y_centered, axis=1),
            denominator,
            out=np.zeros(stop - start, dtype=float),
            where=denominator > 0,
        )
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "estimate": estimate,
        "lower": float(np.quantile(estimates, alpha)),
        "upper": float(np.quantile(estimates, 1.0 - alpha)),
        "n_leaf": int(left.size),
    }


def _load_merge(
    *,
    merge_dir: Path,
    model_id: str,
    expected: dict[str, Any],
    decision: dict[str, Any],
    resolved: ResolvedConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    report_path = merge_dir / "joint_merge_report.json"
    predictions_path = merge_dir / "prediction_results.csv"
    joint_path = merge_dir / "joint_results.csv"
    for path in (report_path, predictions_path, joint_path):
        if not path.is_file():
            raise ValueError(f"Missing merged artifact: {path}")
    if sha256_file(report_path) != expected["report_sha256"]:
        raise ValueError(f"{model_id} merge report SHA-256 mismatch")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("run_type") != "authorized_official_test_joint_merged"
        or report.get("official_test_result") is not True
        or report.get("model_id") != model_id
    ):
        raise ValueError(f"{model_id} merge report is not an authorized result")
    if not all(report.get("acceptance_criteria", {}).values()):
        raise ValueError(f"{model_id} merge report contains a failed criterion")
    for path in (predictions_path, joint_path):
        if report.get("artifact_sha256", {}).get(path.name) != sha256_file(path):
            raise ValueError(f"Merged artifact SHA-256 mismatch: {path}")
    if report.get("prediction_row_count") != expected["prediction_row_count"]:
        raise ValueError(f"{model_id} prediction row count mismatch")
    if report.get("joint_row_count") != expected["joint_row_count"]:
        raise ValueError(f"{model_id} joint row count mismatch")
    if report.get("governance_protocol_hash") != resolved.sha256:
        raise ValueError(f"{model_id} merge does not match the protocol")
    lineage = decision["scientific_lineage"]
    identity = report.get("official_test_identity", {})
    if (
        identity.get("sample_count") != lineage["official_test_sample_count"]
        or identity.get("leaf_count") != lineage["official_test_leaf_count"]
        or identity.get("sample_ids_sha256") != lineage["official_test_sample_ids_sha256"]
    ):
        raise ValueError(f"{model_id} official-test identity mismatch")
    predictions = pd.read_csv(predictions_path, keep_default_na=False)
    joint = pd.read_csv(joint_path, keep_default_na=False)
    _prepare_prediction_frame(predictions, model_id)
    _prepare_joint_frame(joint, model_id)
    _validate_factorial_coverage(predictions, joint, report, resolved)
    return predictions, joint, report


def _prepare_prediction_frame(frame: pd.DataFrame, model_id: str) -> None:
    required = {
        "model_id",
        "sample_id",
        "leaf_id",
        "scenario_id",
        "true_class_name",
        "transformation",
        "severity",
        *BOOLEAN_COLUMNS,
        *PREDICTION_NUMERIC_COLUMNS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction columns missing: {missing}")
    if set(frame["model_id"]) != {model_id}:
        raise ValueError("Prediction model identity mismatch")
    if (
        frame["sample_id"].eq("").any()
        or frame["leaf_id"].eq("").any()
        or frame["true_class_name"].eq("").any()
    ):
        raise ValueError("Prediction scientific identity is empty")
    expected_scenario = frame["transformation"] + "_" + frame["severity"]
    if frame["scenario_id"].ne(expected_scenario).any():
        raise ValueError("Prediction scenario metadata is inconsistent")
    for column in BOOLEAN_COLUMNS:
        frame[column] = _parse_boolean(frame[column], column)
    for column in PREDICTION_NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if not np.isfinite(frame[list(PREDICTION_NUMERIC_COLUMNS)].to_numpy()).all():
        raise ValueError("Prediction metrics contain NaN or Inf")
    frame["confidence_drop"] = -frame["confidence_delta"]


def _prepare_joint_frame(frame: pd.DataFrame, model_id: str) -> None:
    required = {
        "model_id",
        "sample_id",
        "leaf_id",
        "scenario_id",
        "xai_method",
        "is_consistent",
        "exclusion_reason",
        *JOINT_NUMERIC_COLUMNS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Joint columns missing: {missing}")
    if set(frame["model_id"]) != {model_id}:
        raise ValueError("Joint model identity mismatch")
    if frame["sample_id"].eq("").any() or frame["leaf_id"].eq("").any():
        raise ValueError("Joint scientific identity is empty")
    frame["is_consistent"] = _parse_boolean(frame["is_consistent"], "is_consistent")
    valid = frame["exclusion_reason"].eq("")
    for column in JOINT_NUMERIC_COLUMNS:
        converted = pd.to_numeric(frame.loc[valid, column], errors="raise")
        frame.loc[valid, column] = converted
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not np.isfinite(frame.loc[valid, list(JOINT_NUMERIC_COLUMNS)].to_numpy()).all():
        raise ValueError("Included XAI metrics contain NaN or Inf")
    if frame.loc[valid, "is_consistent"].ne(True).any():
        raise ValueError("An included XAI row is prediction-inconsistent")


def _validate_factorial_coverage(
    predictions: pd.DataFrame,
    joint: pd.DataFrame,
    report: dict[str, Any],
    resolved: ResolvedConfig,
) -> None:
    scenarios = _scenario_ids(resolved)
    methods = list(resolved.values["xai"]["methods"])
    prediction_keys = predictions[["sample_id", "scenario_id"]]
    joint_keys = joint[["sample_id", "scenario_id", "xai_method"]]
    if prediction_keys.duplicated().any() or joint_keys.duplicated().any():
        raise ValueError("Merged results contain duplicate factorial keys")
    if sorted(predictions["scenario_id"].unique()) != sorted(scenarios):
        raise ValueError("Prediction scenarios are incomplete")
    if sorted(joint["scenario_id"].unique()) != sorted(scenarios):
        raise ValueError("Joint scenarios are incomplete")
    if sorted(joint["xai_method"].unique()) != sorted(methods):
        raise ValueError("Joint methods are incomplete")
    expected_samples = report["official_test_identity"]["sample_count"]
    if predictions["sample_id"].nunique() != expected_samples:
        raise ValueError("Prediction sample coverage is incomplete")
    if len(predictions) != expected_samples * len(scenarios):
        raise ValueError("Prediction factorial row count is incomplete")
    if len(joint) != expected_samples * len(scenarios) * len(methods):
        raise ValueError("Joint factorial row count is incomplete")
    successful = int(joint["exclusion_reason"].eq("").sum())
    excluded = int(joint["exclusion_reason"].ne("").sum())
    if (
        successful != report["successful_joint_metric_count"]
        or excluded != report["excluded_joint_metric_count"]
    ):
        raise ValueError("Joint inclusion/exclusion counts do not reconcile")


def _validate_shared_lineage(reports: dict[str, dict[str, Any]], decision: dict[str, Any]) -> None:
    fields = (
        "campaign_id",
        "governance_protocol_hash",
        "checkpoint_training_protocol_hash",
        "manifest_sha256",
        "freeze_record_sha256",
        "official_test_identity",
        "scenario_ids",
        "xai_methods",
    )
    left = reports["resnet50"]
    right = reports["efficientnet_b0"]
    mismatches = [field for field in fields if left.get(field) != right.get(field)]
    if mismatches:
        raise ValueError(f"Cross-model merge lineage mismatch: {mismatches}")
    lineage = decision["scientific_lineage"]
    expected = {
        "campaign_id": lineage["campaign_id"],
        "governance_protocol_hash": lineage["governance_protocol_hash"],
        "checkpoint_training_protocol_hash": lineage["checkpoint_training_protocol_hash"],
        "manifest_sha256": lineage["manifest_sha256"],
        "freeze_record_sha256": lineage["historical_final_freeze_record_sha256"],
    }
    mismatches = [field for field, value in expected.items() if left.get(field) != value]
    if mismatches:
        raise ValueError(f"Merge lineage differs from DR-ANALYSIS-001: {mismatches}")


def _validate_cross_model_identity(predictions: pd.DataFrame) -> None:
    identity_columns = ["sample_id", "leaf_id", "true_class_name"]
    identities = (
        predictions[identity_columns]
        .drop_duplicates()
        .sort_values("sample_id")
        .reset_index(drop=True)
    )
    counts = identities.groupby("sample_id").size()
    if not counts.eq(1).all():
        raise ValueError("Cross-model sample, leaf or class identity mismatch")
    model_keys = {
        model: set(zip(group["sample_id"], group["scenario_id"]))
        for model, group in predictions.groupby("model_id")
    }
    if model_keys["resnet50"] != model_keys["efficientnet_b0"]:
        raise ValueError("Cross-model prediction paired keys differ")


def _validate_joint_prediction_identity(predictions: pd.DataFrame, joint: pd.DataFrame) -> None:
    reference = predictions[
        ["model_id", "sample_id", "scenario_id", "leaf_id", "is_consistent"]
    ].rename(
        columns={
            "leaf_id": "prediction_leaf_id",
            "is_consistent": "prediction_is_consistent",
        }
    )
    aligned = joint.merge(
        reference,
        on=["model_id", "sample_id", "scenario_id"],
        validate="many_to_one",
    )
    if len(aligned) != len(joint):
        raise ValueError("Joint rows do not map exactly to prediction rows")
    if aligned["leaf_id"].ne(aligned["prediction_leaf_id"]).any():
        raise ValueError("Joint/prediction leaf identity mismatch")
    if aligned["is_consistent"].ne(aligned["prediction_is_consistent"]).any():
        raise ValueError("Joint/prediction consistency mismatch")


def _paired_contrast(
    frame: pd.DataFrame,
    *,
    left_filter: dict[str, str],
    right_filter: dict[str, str],
    endpoint: str,
    family_id: str,
    contrast_type: str,
    contrast: str,
    minimum_leaf_count: int,
) -> dict[str, Any]:
    left = _filter(frame, left_filter)[["sample_id", "leaf_id", endpoint]]
    right = _filter(frame, right_filter)[["sample_id", "leaf_id", endpoint]]
    paired = left.merge(
        right,
        on="sample_id",
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    if paired.empty or paired["leaf_id_left"].ne(paired["leaf_id_right"]).any():
        raise ValueError(f"Invalid common paired keys for {contrast}")
    leaf = paired.groupby("leaf_id_left", sort=True)[
        [f"{endpoint}_left", f"{endpoint}_right"]
    ].mean()
    if len(leaf) < minimum_leaf_count:
        raise ValueError(f"Paired leaf count below {minimum_leaf_count} for {contrast}")
    left_values = leaf[f"{endpoint}_left"].to_numpy(dtype=float)
    right_values = leaf[f"{endpoint}_right"].to_numpy(dtype=float)
    result = paired_wilcoxon(left_values, right_values)
    return {
        "family_id": family_id,
        "contrast_type": contrast_type,
        "contrast": contrast,
        "endpoint": endpoint,
        "n_common_sample_keys": int(len(paired)),
        "n_leaf_pairs": int(len(leaf)),
        "left_mean": float(left_values.mean()),
        "right_mean": float(right_values.mean()),
        "mean_difference_left_minus_right": float((left_values - right_values).mean()),
        "wilcoxon_statistic": result["statistic"],
        "p_value_raw": result["p_value"],
        "rank_biserial": result["rank_biserial"],
    }


def _apply_holm(rows: list[dict[str, Any]], alpha: float) -> None:
    families: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        families.setdefault(str(row["family_id"]), []).append(row)
    for family_rows in families.values():
        adjusted = holm_adjust(row["p_value_raw"] for row in family_rows)
        for row, p_value in zip(family_rows, adjusted):
            row["p_value_holm"] = p_value
            row["reject_h0_holm"] = bool(p_value <= alpha)
            row["alpha"] = alpha


def _filter(frame: pd.DataFrame, filters: dict[str, str]) -> pd.DataFrame:
    selected = frame
    for column, value in filters.items():
        selected = selected.loc[selected[column].eq(value)]
    return selected


def _parse_boolean(values: pd.Series, column: str) -> pd.Series:
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


def _scenario_ids(resolved: ResolvedConfig) -> list[str]:
    transformations = resolved.values["transformations"]
    return [
        f"{name}_{severity}"
        for name in transformations["names"]
        for severity in transformations["severities"]
    ]


def _split_scenario(scenario_id: str) -> tuple[str, str]:
    transformation, severity = scenario_id.rsplit("_", 1)
    if severity not in {"mild", "moderate", "severe"}:
        raise ValueError(f"Invalid scenario ID: {scenario_id}")
    return transformation, severity


def _derived_seed(base_seed: int, group_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}|{group_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.isclose(left.std(), 0.0) or np.isclose(right.std(), 0.0):
        raise ValueError("Correlation is undefined for a constant vector")
    return float(np.corrcoef(left, right)[0, 1])


def records_are_finite(records: Iterable[dict[str, Any]], keys: Sequence[str]) -> bool:
    return all(
        np.isfinite(float(record[key]))
        for record in records
        for key in keys
        if record.get(key, "") != ""
    )
