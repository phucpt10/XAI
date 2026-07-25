"""Merge the three completed method parts for one official-test model."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

from plantxai_stability.artifacts import atomic_json
from plantxai_stability.config import load_protocol, resolve_xai_target_layer
from plantxai_stability.provenance import sha256_file
from plantxai_stability.recovery import (
    authorize_recovery_joint_part,
    load_recovery_decision,
    validate_preserved_joint_part,
    validate_recovery_binding,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--part-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--recovery-decision-record", type=Path)
    parser.add_argument("--recovery-binding-report", type=Path)
    parser.add_argument("--recovery-supersession-record", type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit("Merged output exists; use a new immutable directory")
    recovery_values = (
        args.manifest,
        args.recovery_decision_record,
        args.recovery_binding_report,
    )
    if any(value is not None for value in recovery_values) and not all(
        value is not None for value in recovery_values
    ):
        raise SystemExit(
            "--manifest, --recovery-decision-record and "
            "--recovery-binding-report must be supplied together"
        )
    governed_recovery = (
        args.protocol.parent / "decision_records" / "DR-RECOVERY-001.yaml"
    )
    if governed_recovery.is_file() and args.recovery_decision_record is None:
        raise SystemExit(
            "DR-RECOVERY-001 is active; recovery evidence is required for merge"
        )
    recovery_lineage = None
    if args.recovery_decision_record is not None:
        recovery_lineage = validate_recovery_binding(
            manifest_path=args.manifest,
            recovery_decision_path=args.recovery_decision_record,
            recovery_binding_report_path=args.recovery_binding_report,
        )

    resolved = load_protocol(args.protocol)
    declared_methods = list(resolved.values["xai"]["methods"])
    if len(args.part_dir) != len(declared_methods):
        raise SystemExit(
            f"Expected {len(declared_methods)} --part-dir values, got {len(args.part_dir)}"
        )
    parts = [_load_part(path) for path in args.part_dir]
    methods = [part["report"]["run_identity"]["xai_method"] for part in parts]
    if sorted(methods) != sorted(declared_methods) or len(set(methods)) != len(methods):
        raise SystemExit("Joint parts do not cover the three declared XAI methods exactly")

    reference_identity = parts[0]["report"]["run_identity"]
    scientific_shared_identity_keys = (
        "schema_version",
        "model_id",
        "campaign_id",
        "authorization_decision_id",
        "seed",
        "scenario_ids",
        "scenario_count",
        "sample_count",
        "sample_ids_sha256",
        "governance_protocol_hash",
        "checkpoint_training_protocol_hash",
        "checkpoint_sha256",
        "manifest_sha256",
        "freeze_record_sha256",
        "checkpoint_decision_record_sha256",
        "test_decision_record_sha256",
        "g2_readiness_report_sha256",
        "transformation_algorithm_version",
        "software_version",
        "runtime_identity",
    )
    for part in parts[1:]:
        identity = part["report"]["run_identity"]
        mismatches = [
            key
            for key in scientific_shared_identity_keys
            if identity.get(key) != reference_identity.get(key)
        ]
        if mismatches:
            raise SystemExit(f"Joint part lineage mismatch: {mismatches}")
        if _common_xai_policy(identity) != _common_xai_policy(reference_identity):
            raise SystemExit("Joint part lineage mismatch: ['xai_policy']")
    for part in parts:
        identity = part["report"]["run_identity"]
        policy = identity.get("xai_policy", {})
        expected_layer = resolve_xai_target_layer(
            resolved.values["xai"],
            identity["model_id"],
            identity["xai_method"],
        )
        if policy.get("target_layer") != expected_layer:
            raise SystemExit(
                "Joint part target layer does not match the frozen method policy"
            )
    if recovery_lineage is None:
        commits = {part["report"]["run_identity"].get("git_commit") for part in parts}
        if len(commits) != 1:
            raise SystemExit("Joint part lineage mismatch: ['git_commit']")
        recovery_transition_passed = False
    else:
        recovery_transition_passed = _validate_recovery_transition(
            parts=parts,
            recovery_lineage=recovery_lineage,
            recovery_decision_path=args.recovery_decision_record,
            recovery_supersession_path=args.recovery_supersession_record,
        )
    if reference_identity["governance_protocol_hash"] != resolved.sha256:
        raise SystemExit("Joint parts do not match the current frozen governance protocol")
    baseline = None
    if args.baseline_report is not None:
        baseline = json.loads(args.baseline_report.read_text(encoding="utf-8"))
        if baseline.get("run_type") != "authorized_official_test_baseline":
            raise SystemExit("Baseline report is not an authorized official result")
        if baseline.get("official_test_result") is not True:
            raise SystemExit("Baseline report does not contain an official result")
        _validate_baseline_binding(baseline, reference_identity)
    elif reference_identity.get("authorization_decision_id") != "DR-TEST-003":
        raise SystemExit(
            "--baseline-report is required unless the run is authorized by DR-TEST-003"
        )

    preserved_baseline_hash_matches = True
    if recovery_lineage is not None and baseline is not None:
        recovery_decision = load_recovery_decision(args.recovery_decision_record)
        baseline_key = (
            "resnet50_report_sha256"
            if reference_identity["model_id"] == "resnet50"
            else "efficientnet_b0_report_sha256"
        )
        expected_baseline_hash = recovery_decision["preserved_official_results"][
            "baselines"
        ][baseline_key]
        preserved_baseline_hash_matches = (
            sha256_file(args.baseline_report) == expected_baseline_hash
        )
        if not preserved_baseline_hash_matches:
            raise SystemExit(
                "Baseline report is not the exact result preserved by DR-RECOVERY-001"
            )

    reference_predictions = _prediction_payload(parts[0]["predictions"])
    for part in parts[1:]:
        if _prediction_payload(part["predictions"]) != reference_predictions:
            raise SystemExit("Prediction results diverge across method-specific parts")

    merged_predictions = [dict(row) for row in parts[0]["predictions"]]
    for row in merged_predictions:
        row["source_part_run_id"] = row["run_id"]
        row["run_id"] = args.run_id
    merged_joint: list[dict[str, str]] = []
    for part in parts:
        for source_row in part["joint"]:
            row = dict(source_row)
            row["source_part_run_id"] = row["run_id"]
            row["run_id"] = args.run_id
            merged_joint.append(row)
    merged_predictions.sort(key=lambda row: (row["sample_id"], row["scenario_id"]))
    merged_joint.sort(
        key=lambda row: (row["sample_id"], row["scenario_id"], row["xai_method"])
    )

    sample_count = int(reference_identity["sample_count"])
    scenario_ids = list(reference_identity["scenario_ids"])
    expected_prediction_count = sample_count * len(scenario_ids)
    expected_joint_count = expected_prediction_count * len(declared_methods)
    prediction_keys = {
        (row["sample_id"], row["scenario_id"]) for row in merged_predictions
    }
    joint_keys = {
        (row["sample_id"], row["scenario_id"], row["xai_method"])
        for row in merged_joint
    }
    criteria = {
        "baseline_binding_matches": True,
        "prediction_baseline_bound_by_method_agreement": True,
        "preserved_baseline_hash_matches": preserved_baseline_hash_matches,
        "all_declared_methods_present": sorted(methods) == sorted(declared_methods),
        "child_artifact_hashes_match": True,
        "child_runs_complete": True,
        "shared_lineage_exact": True,
        "recovery_lineage_bridge_passed": (
            recovery_transition_passed if recovery_lineage is not None else True
        ),
        "predictions_identical_across_method_parts": True,
        "prediction_factorial_coverage_exact": (
            len(merged_predictions) == expected_prediction_count
            and len(prediction_keys) == expected_prediction_count
        ),
        "joint_factorial_coverage_exact": (
            len(merged_joint) == expected_joint_count
            and len(joint_keys) == expected_joint_count
        ),
        "official_test_identity_matches": (
            baseline is None
            or baseline.get("official_test_identity")
            == parts[0]["report"].get("official_test_identity")
        ),
        "immutable_output_directory": True,
    }
    if not all(criteria.values()):
        failed = sorted(key for key, value in criteria.items() if not value)
        raise SystemExit(f"Merged joint coverage gate failed: {failed}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = args.output_dir / "prediction_results.csv"
    joint_path = args.output_dir / "joint_results.csv"
    report_path = args.output_dir / "joint_merge_report.json"
    _write_csv_atomic(predictions_path, merged_predictions)
    _write_csv_atomic(joint_path, merged_joint)
    report = {
        "run_type": "authorized_official_test_joint_merged",
        "official_test_result": True,
        "run_id": args.run_id,
        "model_id": reference_identity["model_id"],
        "campaign_id": reference_identity["campaign_id"],
        "governance_protocol_hash": reference_identity["governance_protocol_hash"],
        "checkpoint_training_protocol_hash": reference_identity[
            "checkpoint_training_protocol_hash"
        ],
        "checkpoint_sha256": reference_identity["checkpoint_sha256"],
        "manifest_sha256": reference_identity["manifest_sha256"],
        "freeze_record_sha256": reference_identity["freeze_record_sha256"],
        "recovery_lineage": recovery_lineage,
        "official_test_identity": parts[0]["report"]["official_test_identity"],
        "baseline_provenance": (
            {
                "source": "separate_authorized_baseline_report",
                "baseline_report_sha256": sha256_file(args.baseline_report),
            }
            if baseline is not None
            else {
                "source": "exact_prediction_agreement_across_three_authorized_method_parts",
                "baseline_report_sha256": None,
            }
        ),
        "scenario_ids": scenario_ids,
        "xai_methods": declared_methods,
        "prediction_row_count": len(merged_predictions),
        "joint_row_count": len(merged_joint),
        "successful_joint_metric_count": sum(
            not row["exclusion_reason"] for row in merged_joint
        ),
        "excluded_joint_metric_count": sum(
            bool(row["exclusion_reason"]) for row in merged_joint
        ),
        "baseline_report_sha256": (
            sha256_file(args.baseline_report) if baseline is not None else None
        ),
        "source_parts": [
            {
                "directory": str(part["path"]),
                "xai_method": part["report"]["run_identity"]["xai_method"],
                "run_id": part["report"]["run_identity"]["run_id"],
                "joint_run_report_sha256": sha256_file(
                    part["path"] / "joint_run_report.json"
                ),
            }
            for part in parts
        ],
        "artifact_sha256": {
            predictions_path.name: sha256_file(predictions_path),
            joint_path.name: sha256_file(joint_path),
        },
        "acceptance_criteria": criteria,
    }
    atomic_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Official joint merge: PASS\nReport: {report_path}")
    print(f"Report SHA-256: {sha256_file(report_path)}")
    return 0


def _load_part(path: Path) -> dict[str, Any]:
    report_path = path / "joint_run_report.json"
    state_path = path / "run_state.json"
    predictions_path = path / "prediction_results.csv"
    joint_path = path / "joint_results.csv"
    required = (report_path, state_path, predictions_path, joint_path)
    missing = [item.name for item in required if not item.is_file()]
    if missing:
        raise SystemExit(f"Joint part {path} is missing artifacts: {missing}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if report.get("run_type") != "authorized_official_test_joint_part":
        raise SystemExit(f"Unexpected joint part type: {path}")
    if report.get("official_test_result") is not True or state.get("status") != "complete":
        raise SystemExit(f"Joint part is not complete: {path}")
    if state.get("run_identity") != report.get("run_identity"):
        raise SystemExit(f"Joint part state/report identity mismatch: {path}")
    for name, expected_hash in report.get("artifact_sha256", {}).items():
        artifact = path / name
        if not artifact.is_file() or sha256_file(artifact) != expected_hash:
            raise SystemExit(f"Joint part artifact hash mismatch: {artifact}")
    return {
        "path": path,
        "report": report,
        "predictions": _read_csv(predictions_path),
        "joint": _read_csv(joint_path),
    }


def _validate_baseline_binding(
    baseline: dict[str, Any], identity: dict[str, Any]
) -> None:
    expected = {
        "model_id": identity["model_id"],
        "campaign_id": identity["campaign_id"],
        "governance_protocol_hash": identity["governance_protocol_hash"],
        "checkpoint_training_protocol_hash": identity[
            "checkpoint_training_protocol_hash"
        ],
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "manifest_sha256": identity["manifest_sha256"],
        "freeze_record_sha256": identity["freeze_record_sha256"],
    }
    mismatches = [
        key for key, value in expected.items() if baseline.get(key) != value
    ]
    if mismatches:
        raise SystemExit(f"Baseline/joint lineage mismatch: {mismatches}")


def _validate_recovery_transition(
    *,
    parts: list[dict[str, Any]],
    recovery_lineage: dict[str, Any],
    recovery_decision_path: Path,
    recovery_supersession_path: Path | None,
) -> bool:
    decision = load_recovery_decision(recovery_decision_path)
    preserved = decision["preserved_official_results"]["completed_joint_part"]
    preserved_hash = preserved["joint_run_report_sha256"]
    model_id = parts[0]["report"]["run_identity"]["model_id"]
    legacy_count = 0
    recovered_commits: set[str] = set()
    for part in parts:
        report_hash = sha256_file(part["path"] / "joint_run_report.json")
        identity = part["report"]["run_identity"]
        if report_hash == preserved_hash:
            validate_preserved_joint_part(
                part_dir=part["path"],
                recovery_decision_path=recovery_decision_path,
            )
            legacy_count += 1
            continue
        if identity.get("recovery_lineage") != recovery_lineage:
            raise SystemExit(
                f"Joint part lacks the approved recovery lineage: {part['path']}"
            )
        commit = identity.get("git_commit")
        if not isinstance(commit, str) or not commit:
            raise SystemExit("Recovered joint part lacks a Git commit")
        recovered_commits.add(commit)
    if recovery_supersession_path is not None:
        for part in parts:
            identity = part["report"]["run_identity"]
            authorize_recovery_joint_part(
                model_id=identity["model_id"],
                xai_method=identity["xai_method"],
                recovery_decision_path=recovery_decision_path,
                recovery_supersession_path=recovery_supersession_path,
            )
    expected_legacy_count = (
        0
        if recovery_supersession_path is not None
        else (1 if model_id == preserved["model_id"] else 0)
    )
    if legacy_count != expected_legacy_count:
        raise SystemExit(
            "Recovery merge does not preserve the exact approved legacy joint part"
        )
    if len(recovered_commits) != 1:
        raise SystemExit(
            "Recovered joint parts must share one infrastructure-recovery commit"
        )
    return True


def _prediction_payload(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [{key: value for key, value in row.items() if key != "run_id"} for row in rows]


def _common_xai_policy(identity: dict[str, Any]) -> dict[str, Any]:
    """Exclude the deliberately method-specific target layer from shared lineage."""
    policy = identity.get("xai_policy")
    if not isinstance(policy, dict):
        raise SystemExit("Joint part is missing its XAI policy")
    return {key: value for key, value in policy.items() if key != "target_layer"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv_atomic(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(main())
