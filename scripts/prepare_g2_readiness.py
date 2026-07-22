"""Create G2 readiness evidence without decoding official-test images."""

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
from plantxai_stability.g2_readiness import validate_g1_audit_evidence
from plantxai_stability.governance import approved_checkpoint_lineage
from plantxai_stability.provenance import sha256_bytes, sha256_file


MODELS = ("resnet50", "efficientnet_b0")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-decision-record", type=Path, required=True)
    for model_id in MODELS:
        option = model_id.replace("_", "-")
        parser.add_argument(f"--{option}-checkpoint", type=Path, required=True)
        parser.add_argument(f"--{option}-evidence", type=Path, required=True)
        parser.add_argument(f"--{option}-audit-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit("G2 readiness output exists; use a new versioned directory")
    resolved = load_protocol(args.protocol)
    governance = resolved.values["governance"]
    if not (
        governance.get("G1_CHECKPOINT_SELECTION") == "pass"
        and governance.get("G2_TEST_EVALUATION_READY") == "blocked"
        and governance.get("official_test_evaluation_allowed") is False
    ):
        raise SystemExit(
            "G2 readiness requires G1 PASS while G2 and official test remain blocked"
        )
    manifest_sha256 = sha256_file(args.manifest)
    freeze_record = require_frozen_artifacts(args.manifest)
    freeze_path = args.manifest.parent / "freeze_record.json"
    freeze_record_sha256 = sha256_file(freeze_path)
    split_summary_path = args.manifest.parent / "split_summary.json"
    if not split_summary_path.is_file():
        raise SystemExit("G2 readiness blocked: split_summary.json is missing")
    expected_summary_hash = freeze_record.get("artifact_sha256", {}).get(
        split_summary_path.name
    )
    if expected_summary_hash != sha256_file(split_summary_path):
        raise SystemExit("G2 readiness blocked: split summary hash mismatch")
    split_summary = json.loads(split_summary_path.read_text(encoding="utf-8"))
    records = read_manifest_csv(args.manifest)
    validate_frozen_splits(records)
    test_records = sorted(
        (record for record in records if record.split == "test"),
        key=lambda record: record.sample_id,
    )
    expected_test_count = int(split_summary["counts_by_split"]["test"])
    if len(test_records) != expected_test_count or not test_records:
        raise SystemExit("G2 readiness blocked: official-test metadata count mismatch")
    if any(record.source_split != "test" for record in test_records):
        raise SystemExit(
            "G2 readiness blocked: final test split contains a non-official-test sample"
        )
    if sum(record.source_split == "test" for record in records) != len(test_records):
        raise SystemExit(
            "G2 readiness blocked: an official-test sample is outside the test split"
        )
    test_sample_ids = [record.sample_id for record in test_records]
    test_leaf_ids = sorted({record.leaf_id for record in test_records})
    if len(test_sample_ids) != len(set(test_sample_ids)):
        raise SystemExit("G2 readiness blocked: duplicate official-test sample identity")
    decision = yaml.safe_load(
        args.checkpoint_decision_record.read_text(encoding="utf-8")
    )
    decision_sha256 = sha256_file(args.checkpoint_decision_record)
    model_results: dict[str, Any] = {}
    for model_id in MODELS:
        option = model_id.replace("_", "-")
        checkpoint_path = getattr(args, f"{model_id}_checkpoint")
        evidence_path = getattr(args, f"{model_id}_evidence")
        audit_path = getattr(args, f"{model_id}_audit_report")
        checkpoint_sha256 = sha256_file(checkpoint_path)
        lineage = approved_checkpoint_lineage(
            decision,
            governance,
            model_id=model_id,
            declared_models=resolved.values["models"],
            checkpoint_sha256=checkpoint_sha256,
            manifest_sha256=manifest_sha256,
            freeze_record_sha256=freeze_record_sha256,
        )
        approved = lineage["checkpoint"]
        evidence_sha256 = sha256_file(evidence_path)
        if evidence_sha256 != approved["checkpoint_evidence_sha256"]:
            raise SystemExit(f"G2 readiness blocked: {option} training evidence hash mismatch")
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        _validate_training_evidence_metadata(
            evidence,
            model_id=model_id,
            checkpoint_sha256=checkpoint_sha256,
            lineage=lineage,
        )
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit_result = validate_g1_audit_evidence(
            audit, decision, model_id=model_id
        )
        for artifact_name, expected_hash in audit["artifact_sha256"].items():
            artifact_path = audit_path.parent / artifact_name
            if not artifact_path.is_file() or sha256_file(artifact_path) != expected_hash:
                raise SystemExit(
                    f"G2 readiness blocked: {option} audit artifact mismatch: "
                    f"{artifact_name}"
                )
        model_results[model_id] = {
            **audit_result,
            "checkpoint_evidence_sha256": evidence_sha256,
            "validation_audit_report_sha256": sha256_file(audit_path),
            "validation_artifact_sha256": audit["artifact_sha256"],
        }
    acceptance = {
        "g1_checkpoint_decision_approved": True,
        "both_declared_models_present": set(model_results) == set(MODELS),
        "checkpoint_and_training_evidence_hashes_match": True,
        "validation_audits_and_child_artifacts_match": True,
        "manifest_and_freeze_lineage_match": True,
        "frozen_split_invariants_pass": True,
        "official_test_metadata_count_matches": len(test_records) == expected_test_count,
        "official_test_source_membership_preserved_exactly": all(
            record.source_split == "test" for record in test_records
        ),
        "official_test_sample_ids_unique": len(test_sample_ids)
        == len(set(test_sample_ids)),
        "official_test_pixels_accessed": False,
        "g2_remains_blocked": governance["G2_TEST_EVALUATION_READY"] == "blocked",
        "official_test_evaluation_remains_disabled": governance[
            "official_test_evaluation_allowed"
        ]
        is False,
    }
    if not all(
        value is True
        for key, value in acceptance.items()
        if key != "official_test_pixels_accessed"
    ) or acceptance["official_test_pixels_accessed"] is not False:
        raise SystemExit("G2 readiness quality gate failed")
    report = {
        "run_type": "metadata_only_g2_readiness",
        "approval_status": "pending_g2_human_review",
        "technical_gate_passed": True,
        "governance_protocol_hash": resolved.sha256,
        "checkpoint_training_protocol_hash": decision["training_lineage"][
            "protocol_hash"
        ],
        "checkpoint_decision_record_id": decision["decision_id"],
        "checkpoint_decision_record_sha256": decision_sha256,
        "manifest_sha256": manifest_sha256,
        "freeze_record_sha256": freeze_record_sha256,
        "freeze_record_protocol_hash": freeze_record.get("protocol_hash"),
        "split_summary_sha256": sha256_file(split_summary_path),
        "official_test": {
            "metadata_sample_count": len(test_records),
            "metadata_leaf_count": len(test_leaf_ids),
            "sample_ids_sha256": _identity_hash(test_sample_ids),
            "leaf_ids_sha256": _identity_hash(test_leaf_ids),
            "pixels_accessed": False,
            "result_computed": False,
        },
        "approved_models": model_results,
        "acceptance_criteria": acceptance,
        "decision": (
            "No automatic G2 approval. Project-owner review and a separate "
            "test-authorization Decision Record are required before pixel access."
        ),
        "runtime": {
            "python_platform": platform.platform(),
            "software_version": __version__,
            "git_commit": _git_revision(),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report_path = args.output_dir / "g2_readiness_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    report_sha256 = sha256_file(report_path)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"G2 readiness technical gate: PASS\nReport: {report_path}")
    print(f"Report SHA-256: {report_sha256}")
    return 0


def _validate_training_evidence_metadata(
    evidence: dict[str, Any],
    *,
    model_id: str,
    checkpoint_sha256: str,
    lineage: dict[str, Any],
) -> None:
    expected = {
        "model_id": model_id,
        "checkpoint_sha256": checkpoint_sha256,
        "freeze_record_sha256": lineage["freeze_record_sha256"],
        "config_hash": lineage["training_protocol_hash"],
        "protocol_hash": lineage["training_protocol_hash"],
        "freeze_record_protocol_hash": lineage["training_protocol_hash"],
        "manifest_sha256": lineage["manifest_sha256"],
        "run_type": "official_checkpoint_selection",
        "official": True,
        "test_split_accessed": False,
    }
    mismatches = [
        key for key, expected_value in expected.items() if evidence.get(key) != expected_value
    ]
    if mismatches:
        raise ValueError(
            f"G2 training evidence mismatch for {model_id}: {sorted(mismatches)}"
        )


def _identity_hash(identities: list[str]) -> str:
    return sha256_bytes(
        json.dumps(identities, separators=(",", ":")).encode("utf-8")
    )


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
