"""Verify G2 authorization without opening official-test image files."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import yaml

from plantxai_stability import __version__
from plantxai_stability.config import load_protocol
from plantxai_stability.data.freeze import require_frozen_artifacts
from plantxai_stability.data.manifest import read_manifest_csv
from plantxai_stability.data.splits import validate_frozen_splits
from plantxai_stability.governance import approved_checkpoint_lineage
from plantxai_stability.provenance import sha256_file
from plantxai_stability.test_authorization import (
    validate_g2_authorization,
    validate_official_test_metadata,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-decision-record", type=Path, required=True)
    parser.add_argument("--test-decision-record", type=Path, required=True)
    parser.add_argument("--g2-readiness-report", type=Path, required=True)
    parser.add_argument("--resnet50-checkpoint", type=Path, required=True)
    parser.add_argument("--efficientnet-b0-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(
            "G2 authorization output exists; use a new versioned directory"
        )
    resolved = load_protocol(args.protocol)
    manifest_sha256 = sha256_file(args.manifest)
    require_frozen_artifacts(args.manifest)
    freeze_path = args.manifest.parent / "freeze_record.json"
    freeze_record_sha256 = sha256_file(freeze_path)
    checkpoint_decision = _load_yaml(args.checkpoint_decision_record)
    test_decision = _load_yaml(args.test_decision_record)
    readiness_report = json.loads(
        args.g2_readiness_report.read_text(encoding="utf-8")
    )
    scenarios = [
        f"{name}_{severity}"
        for name in resolved.values["transformations"]["names"]
        for severity in resolved.values["transformations"]["severities"]
    ]
    authorization = validate_g2_authorization(
        governance=resolved.values["governance"],
        governance_protocol_hash=resolved.sha256,
        declared_models=resolved.values["models"],
        declared_scenarios=scenarios,
        declared_xai_methods=resolved.values["xai"]["methods"],
        test_decision=test_decision,
        readiness_report=readiness_report,
        readiness_report_sha256=sha256_file(args.g2_readiness_report),
        checkpoint_decision=checkpoint_decision,
        checkpoint_decision_sha256=sha256_file(args.checkpoint_decision_record),
        manifest_sha256=manifest_sha256,
        freeze_record_sha256=freeze_record_sha256,
    )
    records = read_manifest_csv(args.manifest)
    validate_frozen_splits(records)
    test_identity = validate_official_test_metadata(records, test_decision)
    checkpoint_paths = {
        "resnet50": args.resnet50_checkpoint,
        "efficientnet_b0": args.efficientnet_b0_checkpoint,
    }
    checkpoint_hashes: dict[str, str] = {}
    for model_id, checkpoint_path in checkpoint_paths.items():
        checkpoint_sha256 = sha256_file(checkpoint_path)
        approved_checkpoint_lineage(
            checkpoint_decision,
            resolved.values["governance"],
            model_id=model_id,
            declared_models=resolved.values["models"],
            checkpoint_sha256=checkpoint_sha256,
            manifest_sha256=manifest_sha256,
            freeze_record_sha256=freeze_record_sha256,
        )
        checkpoint_hashes[model_id] = checkpoint_sha256
    report = {
        "run_type": "metadata_only_g2_authorization_verification",
        "authorization_status": "AUTHORIZED_NOT_EXECUTED",
        "authorization_gate_passed": True,
        "official_test_pixels_accessed": False,
        "official_test_result_computed": False,
        "campaign_id": authorization["campaign_id"],
        "authorization_decision_id": authorization["authorization_decision_id"],
        "test_decision_record_sha256": sha256_file(args.test_decision_record),
        "g2_readiness_report_sha256": sha256_file(args.g2_readiness_report),
        "checkpoint_decision_record_sha256": sha256_file(
            args.checkpoint_decision_record
        ),
        "governance_protocol_hash": resolved.sha256,
        "checkpoint_training_protocol_hash": authorization[
            "checkpoint_training_protocol_hash"
        ],
        "manifest_sha256": manifest_sha256,
        "freeze_record_sha256": freeze_record_sha256,
        "checkpoint_sha256": checkpoint_hashes,
        "official_test_identity": test_identity,
        "registered_scenario_ids": scenarios,
        "registered_xai_methods": resolved.values["xai"]["methods"],
        "acceptance_criteria": {
            "g2_governance_and_decision_match": True,
            "readiness_report_hash_and_content_match": True,
            "checkpoint_registry_and_bytes_match": True,
            "manifest_and_freeze_match": True,
            "official_test_identity_matches": True,
            "registered_campaign_matches_protocol": True,
            "official_test_pixels_accessed": False,
            "official_test_result_computed": False,
        },
        "runtime": {
            "python_platform": platform.platform(),
            "software_version": __version__,
            "git_commit": _git_revision(),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report_path = args.output_dir / "g2_authorization_verification.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"G2 authorization gate: PASS\nReport: {report_path}")
    print(f"Report SHA-256: {sha256_file(report_path)}")
    return 0


def _load_yaml(path: Path) -> dict[str, Any]:
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError(f"Decision Record must be a mapping: {path}")
    return values


def _git_revision() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


if __name__ == "__main__":
    raise SystemExit(main())
