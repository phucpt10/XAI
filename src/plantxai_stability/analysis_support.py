"""Metadata-only common-support audit for predeclared statistical contrasts."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from plantxai_stability.config import ResolvedConfig
from plantxai_stability.official_analysis import (
    _scenario_ids,
    _split_scenario,
    _validate_shared_lineage,
)
from plantxai_stability.provenance import sha256_file


EXPECTED_SUPPORT_COUNTS = {
    "rq1_model": 12,
    "rq2_model": 36,
    "rq2_method": 72,
    "rq2_severity": 72,
}
EXPECTED_SUPPORT_TOTAL = sum(EXPECTED_SUPPORT_COUNTS.values())
PREDICTION_SUPPORT_COLUMNS = (
    "model_id",
    "sample_id",
    "leaf_id",
    "scenario_id",
    "true_class_name",
)
JOINT_SUPPORT_COLUMNS = (
    "model_id",
    "sample_id",
    "leaf_id",
    "scenario_id",
    "xai_method",
    "exclusion_reason",
)


def load_support_metadata(
    *,
    merge_dirs: dict[str, Path],
    decision: dict[str, Any],
    resolved: ResolvedConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
    models = list(resolved.values["models"])
    if sorted(merge_dirs) != sorted(models):
        raise ValueError("Exactly the two declared merge directories are required")
    prediction_frames: list[pd.DataFrame] = []
    joint_frames: list[pd.DataFrame] = []
    reports: dict[str, dict[str, Any]] = {}
    for model_id in models:
        merge_dir = merge_dirs[model_id]
        expected = decision["source_merges"][model_id]
        report_path = merge_dir / "joint_merge_report.json"
        prediction_path = merge_dir / "prediction_results.csv"
        joint_path = merge_dir / "joint_results.csv"
        for path in (report_path, prediction_path, joint_path):
            if not path.is_file():
                raise ValueError(f"Missing merged artifact: {path}")
        if sha256_file(report_path) != expected["report_sha256"]:
            raise ValueError(f"{model_id} merge report SHA-256 mismatch")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        _validate_support_report(
            report=report,
            model_id=model_id,
            expected=expected,
            resolved=resolved,
        )
        for path in (prediction_path, joint_path):
            if report["artifact_sha256"].get(path.name) != sha256_file(path):
                raise ValueError(f"Merged artifact SHA-256 mismatch: {path}")
        prediction_frame = pd.read_csv(
            prediction_path,
            usecols=list(PREDICTION_SUPPORT_COLUMNS),
            keep_default_na=False,
        )
        joint_frame = pd.read_csv(
            joint_path,
            usecols=list(JOINT_SUPPORT_COLUMNS),
            keep_default_na=False,
        )
        _validate_support_frames(
            predictions=prediction_frame,
            joint=joint_frame,
            model_id=model_id,
            report=report,
            resolved=resolved,
        )
        prediction_frames.append(prediction_frame)
        joint_frames.append(joint_frame)
        reports[model_id] = report
    _validate_shared_lineage(reports, decision)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    joint = pd.concat(joint_frames, ignore_index=True)
    _validate_cross_model_support_identity(predictions)
    _validate_joint_support_identity(predictions, joint)
    return predictions, joint, reports


def build_support_audit(
    predictions: pd.DataFrame,
    joint: pd.DataFrame,
    *,
    methods: Sequence[str],
    minimum_leaf_count: int,
) -> list[dict[str, Any]]:
    scenarios = sorted(predictions["scenario_id"].unique())
    valid = joint.loc[joint["exclusion_reason"].eq("")].copy()
    rows: list[dict[str, Any]] = []

    for scenario in scenarios:
        subset = predictions.loc[predictions["scenario_id"].eq(scenario)]
        transformation, severity = _split_scenario(scenario)
        rows.append(
            _support_contrast(
                subset.loc[subset["model_id"].eq("resnet50")],
                subset.loc[subset["model_id"].eq("efficientnet_b0")],
                contrast_scope="rq1_model",
                contrast_id=f"rq1_model::{scenario}",
                left_label="resnet50",
                right_label="efficientnet_b0",
                model_id="both",
                xai_method="not_applicable",
                transformation=transformation,
                scenario_left=scenario,
                scenario_right=scenario,
                minimum_leaf_count=minimum_leaf_count,
            )
        )

    for method in methods:
        for scenario in scenarios:
            subset = valid.loc[valid["scenario_id"].eq(scenario) & valid["xai_method"].eq(method)]
            transformation, _ = _split_scenario(scenario)
            rows.append(
                _support_contrast(
                    subset.loc[subset["model_id"].eq("resnet50")],
                    subset.loc[subset["model_id"].eq("efficientnet_b0")],
                    contrast_scope="rq2_model",
                    contrast_id=f"rq2_model::{method}::{scenario}",
                    left_label="resnet50",
                    right_label="efficientnet_b0",
                    model_id="both",
                    xai_method=method,
                    transformation=transformation,
                    scenario_left=scenario,
                    scenario_right=scenario,
                    minimum_leaf_count=minimum_leaf_count,
                )
            )

    for model_id in sorted(valid["model_id"].unique()):
        for scenario in scenarios:
            subset = valid.loc[valid["model_id"].eq(model_id) & valid["scenario_id"].eq(scenario)]
            transformation, _ = _split_scenario(scenario)
            for left_method, right_method in itertools.combinations(methods, 2):
                rows.append(
                    _support_contrast(
                        subset.loc[subset["xai_method"].eq(left_method)],
                        subset.loc[subset["xai_method"].eq(right_method)],
                        contrast_scope="rq2_method",
                        contrast_id=(
                            f"rq2_method::{model_id}::{scenario}::{left_method}_vs_{right_method}"
                        ),
                        left_label=left_method,
                        right_label=right_method,
                        model_id=model_id,
                        xai_method="pairwise",
                        transformation=transformation,
                        scenario_left=scenario,
                        scenario_right=scenario,
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
            for transformation in sorted(transformed["transformation"].unique()):
                subset = transformed.loc[
                    transformed["model_id"].eq(model_id)
                    & transformed["xai_method"].eq(method)
                    & transformed["transformation"].eq(transformation)
                ]
                for left_severity, right_severity in severity_pairs:
                    left_scenario = f"{transformation}_{left_severity}"
                    right_scenario = f"{transformation}_{right_severity}"
                    rows.append(
                        _support_contrast(
                            subset.loc[subset["severity"].eq(left_severity)],
                            subset.loc[subset["severity"].eq(right_severity)],
                            contrast_scope="rq2_severity",
                            contrast_id=(
                                f"rq2_severity::{model_id}::{method}::"
                                f"{transformation}::{left_severity}_vs_{right_severity}"
                            ),
                            left_label=left_severity,
                            right_label=right_severity,
                            model_id=model_id,
                            xai_method=method,
                            transformation=transformation,
                            scenario_left=left_scenario,
                            scenario_right=right_scenario,
                            minimum_leaf_count=minimum_leaf_count,
                        )
                    )

    rows.sort(
        key=lambda row: (
            row["passes_minimum_common_leaves"],
            row["common_leaves"],
            row["contrast_scope"],
            row["contrast_id"],
        )
    )
    _validate_support_coverage(rows)
    return rows


def exclusion_reason_audit(joint: pd.DataFrame) -> list[dict[str, Any]]:
    frame = joint.assign(reason=joint["exclusion_reason"].replace("", "included"))
    output = (
        frame.groupby(
            ["model_id", "xai_method", "scenario_id", "reason"],
            sort=True,
        )
        .size()
        .reset_index(name="row_count")
    )
    return output.to_dict(orient="records")


def _support_contrast(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    contrast_scope: str,
    contrast_id: str,
    left_label: str,
    right_label: str,
    model_id: str,
    xai_method: str,
    transformation: str,
    scenario_left: str,
    scenario_right: str,
    minimum_leaf_count: int,
) -> dict[str, Any]:
    left_keys = left[["sample_id", "leaf_id"]]
    right_keys = right[["sample_id", "leaf_id"]]
    if left_keys["sample_id"].duplicated().any():
        raise ValueError(f"Duplicate left paired key for {contrast_id}")
    if right_keys["sample_id"].duplicated().any():
        raise ValueError(f"Duplicate right paired key for {contrast_id}")
    paired = left_keys.merge(
        right_keys,
        on="sample_id",
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    if paired["leaf_id_left"].ne(paired["leaf_id_right"]).any():
        raise ValueError(f"Paired leaf identity mismatch for {contrast_id}")
    common_leaves = int(paired["leaf_id_left"].nunique())
    passed = common_leaves >= minimum_leaf_count
    return {
        "contrast_scope": contrast_scope,
        "contrast_id": contrast_id,
        "model_id": model_id,
        "xai_method": xai_method,
        "transformation": transformation,
        "scenario_left": scenario_left,
        "scenario_right": scenario_right,
        "left_label": left_label,
        "right_label": right_label,
        "left_valid_samples": int(len(left_keys)),
        "right_valid_samples": int(len(right_keys)),
        "common_samples": int(len(paired)),
        "common_leaves": common_leaves,
        "minimum_common_leaves": minimum_leaf_count,
        "passes_minimum_common_leaves": passed,
        "support_status": ("PASS" if passed else "NOT_ESTIMABLE_INSUFFICIENT_COMMON_LEAVES"),
    }


def _validate_support_report(
    *,
    report: dict[str, Any],
    model_id: str,
    expected: dict[str, Any],
    resolved: ResolvedConfig,
) -> None:
    if (
        report.get("run_type") != "authorized_official_test_joint_merged"
        or report.get("official_test_result") is not True
        or report.get("model_id") != model_id
    ):
        raise ValueError(f"{model_id} merge is not an authorized official result")
    if not all(report.get("acceptance_criteria", {}).values()):
        raise ValueError(f"{model_id} merge contains a failed criterion")
    if report.get("governance_protocol_hash") != resolved.sha256:
        raise ValueError(f"{model_id} merge does not match the protocol")
    if report.get("prediction_row_count") != expected["prediction_row_count"]:
        raise ValueError(f"{model_id} prediction row count mismatch")
    if report.get("joint_row_count") != expected["joint_row_count"]:
        raise ValueError(f"{model_id} joint row count mismatch")


def _validate_support_frames(
    *,
    predictions: pd.DataFrame,
    joint: pd.DataFrame,
    model_id: str,
    report: dict[str, Any],
    resolved: ResolvedConfig,
) -> None:
    scenarios = _scenario_ids(resolved)
    methods = list(resolved.values["xai"]["methods"])
    sample_count = int(report["official_test_identity"]["sample_count"])
    if set(predictions["model_id"]) != {model_id} or set(joint["model_id"]) != {model_id}:
        raise ValueError("Support metadata model identity mismatch")
    if (
        predictions["sample_id"].eq("").any()
        or predictions["leaf_id"].eq("").any()
        or joint["sample_id"].eq("").any()
        or joint["leaf_id"].eq("").any()
    ):
        raise ValueError("Support metadata contains an empty scientific identity")
    prediction_keys = predictions[["sample_id", "scenario_id"]]
    joint_keys = joint[["sample_id", "scenario_id", "xai_method"]]
    if prediction_keys.duplicated().any() or joint_keys.duplicated().any():
        raise ValueError("Support metadata contains duplicate factorial keys")
    if len(predictions) != sample_count * len(scenarios):
        raise ValueError("Prediction support coverage is incomplete")
    if len(joint) != sample_count * len(scenarios) * len(methods):
        raise ValueError("Joint support coverage is incomplete")
    if sorted(predictions["scenario_id"].unique()) != sorted(scenarios):
        raise ValueError("Prediction support scenarios are incomplete")
    if sorted(joint["scenario_id"].unique()) != sorted(scenarios):
        raise ValueError("Joint support scenarios are incomplete")
    if sorted(joint["xai_method"].unique()) != sorted(methods):
        raise ValueError("Joint support methods are incomplete")
    successful = int(joint["exclusion_reason"].eq("").sum())
    excluded = int(joint["exclusion_reason"].ne("").sum())
    if (
        successful != report["successful_joint_metric_count"]
        or excluded != report["excluded_joint_metric_count"]
    ):
        raise ValueError("Support inclusion/exclusion counts do not reconcile")


def _validate_cross_model_support_identity(predictions: pd.DataFrame) -> None:
    identity = predictions[["sample_id", "leaf_id", "true_class_name"]].drop_duplicates()
    if not identity.groupby("sample_id").size().eq(1).all():
        raise ValueError("Cross-model support identity mismatch")
    keys = {
        model_id: set(zip(group["sample_id"], group["scenario_id"]))
        for model_id, group in predictions.groupby("model_id")
    }
    if keys["resnet50"] != keys["efficientnet_b0"]:
        raise ValueError("Cross-model support paired keys differ")


def _validate_joint_support_identity(predictions: pd.DataFrame, joint: pd.DataFrame) -> None:
    reference = predictions[["model_id", "sample_id", "scenario_id", "leaf_id"]].rename(
        columns={"leaf_id": "prediction_leaf_id"}
    )
    aligned = joint.merge(
        reference,
        on=["model_id", "sample_id", "scenario_id"],
        validate="many_to_one",
    )
    if len(aligned) != len(joint):
        raise ValueError("Joint support rows do not map to prediction rows")
    if aligned["leaf_id"].ne(aligned["prediction_leaf_id"]).any():
        raise ValueError("Joint/prediction support leaf identity mismatch")


def _validate_support_coverage(rows: list[dict[str, Any]]) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        scope = str(row["contrast_scope"])
        counts[scope] = counts.get(scope, 0) + 1
    if counts != EXPECTED_SUPPORT_COUNTS or len(rows) != EXPECTED_SUPPORT_TOTAL:
        raise ValueError(f"Planned support contrast coverage mismatch: {counts}, total={len(rows)}")
