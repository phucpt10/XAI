"""Generate approved read-only classwise supplementary artifacts."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from plantxai_stability import __version__
from plantxai_stability.artifacts import atomic_json
from plantxai_stability.provenance import sha256_file


RQ1_ENDPOINTS = ("is_consistent", "transformed_is_correct", "confidence_drop")
XAI_ENDPOINTS = ("pearson", "ssim", "topk_iou_20")
ARTIFACTS = (
    "supplement_classwise_rq1.csv",
    "supplement_classwise_xai_valid_coverage.csv",
    "supplement_classwise_xai_descriptive.csv",
    "supplement_leave_one_class_out_rq1.csv",
    "supplement_leave_one_class_out_xai.csv",
    "supplement_classwise_methodology.md",
    "classwise_supplementary_report.json",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-decision-record", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--resnet50-merge-dir", type=Path, required=True)
    parser.add_argument("--efficientnet-b0-merge-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit("Supplementary output exists; use a new immutable directory")
    building = Path(f"{args.output_dir}.building")
    if building.exists():
        raise SystemExit(f"Stale supplementary build directory exists: {building}")

    decision = _load_decision(args.results_decision_record)
    analysis_report, class_rq1 = _load_analysis(args.analysis_dir, decision)
    joint = _load_merges(
        merge_dirs={
            "resnet50": args.resnet50_merge_dir,
            "efficientnet_b0": args.efficientnet_b0_merge_dir,
        },
        decision=decision,
    )

    class_rq1 = class_rq1.loc[class_rq1["endpoint"].isin(RQ1_ENDPOINTS)].copy()
    coverage = _coverage(joint)
    xai = _classwise_xai(joint)
    rq1_loco = _leave_one_class_out_rq1(class_rq1)
    xai_loco = _leave_one_class_out_xai(xai)

    building.mkdir(parents=True, exist_ok=False)
    try:
        tables = {
            "supplement_classwise_rq1.csv": class_rq1.sort_values(
                ["model_id", "scenario_id", "class_name", "endpoint"]
            ),
            "supplement_classwise_xai_valid_coverage.csv": coverage.sort_values(
                ["model_id", "xai_method", "class_name"]
            ),
            "supplement_classwise_xai_descriptive.csv": xai.sort_values(
                ["model_id", "xai_method", "scenario_id", "class_name"]
            ),
            "supplement_leave_one_class_out_rq1.csv": rq1_loco.sort_values(
                ["model_id", "scenario_id", "endpoint", "held_out_class"]
            ),
            "supplement_leave_one_class_out_xai.csv": xai_loco.sort_values(
                ["model_id", "xai_method", "scenario_id", "endpoint", "held_out_class"]
            ),
        }
        for name, table in tables.items():
            table.to_csv(building / name, index=False, lineterminator="\n")

        methodology = _methodology_markdown(decision)
        (building / "supplement_classwise_methodology.md").write_text(
            methodology, encoding="utf-8", newline="\n"
        )
        child_paths = sorted(building.iterdir())
        expected_children = sorted(ARTIFACTS[:-1])
        if [path.name for path in child_paths] != expected_children:
            raise RuntimeError("Supplementary artifact coverage mismatch")

        criteria = {
            "results_decision_approved": True,
            "analysis_report_hash_matches": True,
            "analysis_child_hash_matches": True,
            "merge_report_hashes_match": True,
            "merge_child_hashes_match": True,
            "classwise_rq1_is_descriptive_only": True,
            "classwise_xai_uses_valid_rows_only": True,
            "leave_one_class_out_is_descriptive_only": True,
            "official_test_pixels_not_accessed": True,
            "predictions_and_cams_not_recomputed": True,
            "no_hypothesis_tests_or_confidence_intervals_computed": True,
            "no_selection_or_tuning_performed": True,
        }
        report = {
            "run_type": "authorized_classwise_supplementary_reporting",
            "results_decision_id": decision["decision_id"],
            "results_decision_record_sha256": sha256_file(args.results_decision_record),
            "source_analysis_report_sha256": decision["source_analysis"]["report_sha256"],
            "source_merge_report_sha256": {
                model: values["report_sha256"]
                for model, values in decision["source_merges"].items()
            },
            "official_test_pixels_accessed": False,
            "predictions_or_cams_recomputed": False,
            "hypothesis_tests_computed": False,
            "confidence_intervals_computed": False,
            "artifact_row_counts": {name: len(table) for name, table in tables.items()},
            "artifact_sha256": {path.name: sha256_file(path) for path in child_paths},
            "acceptance_criteria": criteria,
            "runtime": {
                "python_platform": platform.platform(),
                "software_version": __version__,
                "git_commit": _git_revision(),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "device": "cpu",
            },
        }
        report_path = building / "classwise_supplementary_report.json"
        atomic_json(report_path, report)
        building.rename(args.output_dir)
    except BaseException:
        if building.exists():
            shutil.rmtree(building)
        raise

    report_path = args.output_dir / "classwise_supplementary_report.json"
    print(json.dumps(json.loads(report_path.read_text(encoding="utf-8")), indent=2))
    print(f"Classwise supplementary reporting: PASS\nReport: {report_path}")
    print(f"Report SHA-256: {sha256_file(report_path)}")
    return 0


def _load_decision(path: Path) -> dict[str, Any]:
    decision = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(decision, dict):
        raise ValueError("Results Decision Record must be a YAML mapping")
    if decision.get("decision_id") != "DR-RESULTS-003":
        raise ValueError("Expected DR-RESULTS-003")
    if decision.get("status") != "approved" or decision.get("approved_by") != "project_owner":
        raise ValueError("DR-RESULTS-003 lacks project-owner approval")
    if decision.get("authorized_outputs", {}).get("artifacts") != list(ARTIFACTS):
        raise ValueError("DR-RESULTS-003 output contract mismatch")
    if not all(decision.get("constraints", {}).values()):
        raise ValueError("DR-RESULTS-003 constraints are incomplete")
    return decision


def _load_analysis(analysis_dir: Path, decision: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    report_path = analysis_dir / "official_analysis_report.json"
    class_path = analysis_dir / "prediction_class_summary.csv"
    if not report_path.is_file() or not class_path.is_file():
        raise ValueError("Required frozen analysis artifacts are missing")
    source = decision["source_analysis"]
    if sha256_file(report_path) != source["report_sha256"]:
        raise ValueError("Frozen analysis report hash mismatch")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected = {
        "run_type": source["run_type"],
        "analysis_decision_id": source["analysis_decision_id"],
        "analysis_decision_record_sha256": source["analysis_decision_record_sha256"],
    }
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError("Frozen analysis lineage mismatch")
    if report.get("artifact_sha256", {}).get(class_path.name) != source[
        "prediction_class_summary_sha256"
    ] or sha256_file(class_path) != source["prediction_class_summary_sha256"]:
        raise ValueError("Frozen classwise prediction artifact hash mismatch")
    return report, pd.read_csv(class_path, keep_default_na=False)


def _load_merges(*, merge_dirs: dict[str, Path], decision: dict[str, Any]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for model, directory in merge_dirs.items():
        report_path = directory / "joint_merge_report.json"
        joint_path = directory / "joint_results.csv"
        prediction_path = directory / "prediction_results.csv"
        if not all(path.is_file() for path in (report_path, joint_path, prediction_path)):
            raise ValueError(f"Required merge artifacts are missing for {model}")
        source = decision["source_merges"][model]
        if sha256_file(report_path) != source["report_sha256"]:
            raise ValueError(f"Merge report hash mismatch for {model}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("run_id") != source["run_id"]:
            raise ValueError(f"Merge run identity mismatch for {model}")
        for name, expected_hash in report.get("artifact_sha256", {}).items():
            artifact = directory / name
            if not artifact.is_file() or sha256_file(artifact) != expected_hash:
                raise ValueError(f"Merge child artifact hash mismatch: {artifact}")

        joint = pd.read_csv(joint_path, keep_default_na=False)
        if "true_class_name" not in joint.columns:
            prediction = pd.read_csv(
                prediction_path,
                usecols=["model_id", "sample_id", "leaf_id", "scenario_id", "true_class_name"],
                keep_default_na=False,
            )
            joint = joint.merge(
                prediction,
                on=["model_id", "sample_id", "leaf_id", "scenario_id"],
                how="left",
                validate="many_to_one",
            )
        frames.append(joint)
    return pd.concat(frames, ignore_index=True)


def _coverage(joint: pd.DataFrame) -> pd.DataFrame:
    table = joint.assign(valid_xai=joint["exclusion_reason"].fillna("").eq(""))
    table = table.groupby(
        ["model_id", "xai_method", "true_class_name"], as_index=False
    ).agg(total_records=("sample_id", "size"), valid_records=("valid_xai", "sum"))
    table["valid_rate"] = table["valid_records"] / table["total_records"]
    return table.rename(columns={"true_class_name": "class_name"})


def _classwise_xai(joint: pd.DataFrame) -> pd.DataFrame:
    valid = joint.loc[joint["exclusion_reason"].fillna("").eq("")].copy()
    for endpoint in XAI_ENDPOINTS:
        valid[endpoint] = pd.to_numeric(valid[endpoint], errors="raise")
    leaf = valid.groupby(
        ["model_id", "xai_method", "scenario_id", "true_class_name", "leaf_id"],
        as_index=False,
    )[list(XAI_ENDPOINTS)].mean()
    result = leaf.groupby(
        ["model_id", "xai_method", "scenario_id", "true_class_name"], as_index=False
    ).agg(
        pearson=("pearson", "mean"),
        ssim=("ssim", "mean"),
        topk_iou_20=("topk_iou_20", "mean"),
        valid_leaf_count=("leaf_id", "nunique"),
    )
    return result.rename(columns={"true_class_name": "class_name"})


def _leave_one_class_out_rq1(class_rq1: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, part in class_rq1.groupby(["model_id", "scenario_id", "endpoint"], sort=True):
        baseline = (part["estimate"] * part["n_leaf"]).sum() / part["n_leaf"].sum()
        for held_out, removed in part.groupby("class_name", sort=True):
            retained = part.loc[part["class_name"] != held_out]
            value = (retained["estimate"] * retained["n_leaf"]).sum() / retained["n_leaf"].sum()
            rows.append(_loco_row(key, held_out, baseline, value, int(retained["n_leaf"].sum())))
    return pd.DataFrame(
        rows,
        columns=[
            "model_id",
            "scenario_id",
            "endpoint",
            "held_out_class",
            "baseline",
            "leave_one_out",
            "absolute_shift",
            "retained_leaf_count",
        ],
    )


def _leave_one_class_out_xai(xai: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for endpoint in XAI_ENDPOINTS:
        for key, part in xai.groupby(["model_id", "xai_method", "scenario_id"], sort=True):
            baseline = _weighted_mean(part, endpoint)
            for held_out, _ in part.groupby("class_name", sort=True):
                retained = part.loc[part["class_name"] != held_out]
                value = _weighted_mean(retained, endpoint)
                row = _loco_row(key, held_out, baseline, value, int(retained["valid_leaf_count"].sum()))
                row["endpoint"] = endpoint
                rows.append(row)
    return pd.DataFrame(rows, columns=_loco_columns())


def _weighted_mean(frame: pd.DataFrame, endpoint: str) -> float:
    return float((frame[endpoint] * frame["valid_leaf_count"]).sum() / frame["valid_leaf_count"].sum())


def _loco_row(key: tuple[Any, ...], held_out: str, baseline: float, value: float, retained: int) -> dict[str, Any]:
    names = ("model_id", "scenario_id", "endpoint") if len(key) == 3 and key[1].startswith(("rotation", "brightness", "gaussian")) else ("model_id", "xai_method", "scenario_id")
    row = dict(zip(names, key))
    row.update(
        held_out_class=held_out,
        baseline=baseline,
        leave_one_out=value,
        absolute_shift=abs(value - baseline),
        retained_leaf_count=retained,
    )
    return row


def _loco_columns() -> list[str]:
    return [
        "model_id",
        "xai_method",
        "scenario_id",
        "endpoint",
        "held_out_class",
        "baseline",
        "leave_one_out",
        "absolute_shift",
        "retained_leaf_count",
    ]


def _methodology_markdown(decision: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Classwise Supplementary Methodology",
            "",
            "This supplement is descriptive only and derives exclusively from the frozen v1.4 analysis and merged result artifacts.",
            "Source analysis report SHA-256: "
            f"`{decision['source_analysis']['report_sha256']}`",
            "",
            "- No official-test image pixels were accessed.",
            "- No predictions or CAMs were recomputed.",
            "- No new confidence intervals, p-values, or multiplicity corrections were computed.",
            "- Classwise XAI summaries use only prediction-consistent records with valid original and transformed CAM metrics.",
            "- Leave-one-class-out values are descriptive sensitivity diagnostics and are not model-selection rules.",
            "- The outputs do not establish external or field-image generalization.",
            "",
        ]
    )


def _git_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


if __name__ == "__main__":
    raise SystemExit(main())
