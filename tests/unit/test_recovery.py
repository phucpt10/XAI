from __future__ import annotations

import json
from pathlib import Path

import pytest

import plantxai_stability.recovery as recovery
from plantxai_stability.provenance import sha256_file as real_sha256_file


DECISION = Path(
    "configs/protocol/v0.9/decision_records/DR-RECOVERY-001.yaml"
)


def test_recovery_decision_authorizes_only_five_unfinished_parts() -> None:
    recovery.authorize_recovery_joint_part(
        model_id="resnet50",
        xai_method="score_cam",
        recovery_decision_path=DECISION,
    )
    with pytest.raises(ValueError, match="does not authorize"):
        recovery.authorize_recovery_joint_part(
            model_id="resnet50",
            xai_method="grad_cam",
            recovery_decision_path=DECISION,
        )


def test_recovery_binding_preserves_logical_and_physical_freeze_identities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    decision = recovery.load_recovery_decision(DECISION)
    evidence = decision["recovery_evidence"]
    decision_sha = real_sha256_file(DECISION)
    physical_sha = "f" * 64
    report_sha = "r" * 64
    manifest = tmp_path / "dataset_manifest.csv"
    freeze = tmp_path / "freeze_record.json"
    leakage = tmp_path / "split_leakage_report.json"
    report = tmp_path / "recovery_binding_report.json"
    manifest.write_text("manifest", encoding="utf-8")
    leakage.write_text(json.dumps({"passed": True}), encoding="utf-8")
    freeze.write_text(
        json.dumps(
            {
                "schema_version": recovery.RECOVERY_SCHEMA_VERSION,
                "protocol_hash": evidence["checkpoint_training_protocol_hash"],
                "manifest_sha256": evidence["manifest_sha256"],
                "artifact_sha256": {
                    "dataset_manifest.csv": evidence["manifest_sha256"]
                },
                "historical_final_freeze_record_sha256": evidence[
                    "historical_final_freeze_record_sha256"
                ],
                "source_freeze_record_sha256": evidence[
                    "recovered_source_freeze_record_sha256"
                ],
                "recovery_decision_id": "DR-RECOVERY-001",
                "recovery_decision_record_sha256": decision_sha,
                "recovery_audit_sha256": evidence["recovery_audit_sha256"],
                "archive_sha256": evidence["archive_sha256"],
            }
        ),
        encoding="utf-8",
    )
    report.write_text(
        json.dumps(
            {
                "schema_version": recovery.RECOVERY_SCHEMA_VERSION,
                "run_type": "infrastructure_only_physical_freeze_recovery",
                "recovery_gate_passed": True,
                "acceptance_criteria": {"all_images_verified": True},
                "recovery_decision_id": "DR-RECOVERY-001",
                "recovery_decision_record_sha256": decision_sha,
                "archive_sha256": evidence["archive_sha256"],
                "recovery_audit_sha256": evidence["recovery_audit_sha256"],
                "manifest_sha256": evidence["manifest_sha256"],
                "verified_sample_count": 8384,
                "split_counts": evidence["split_counts"],
                "historical_final_freeze_record_sha256": evidence[
                    "historical_final_freeze_record_sha256"
                ],
                "source_freeze_record_sha256": evidence[
                    "recovered_source_freeze_record_sha256"
                ],
                "physical_freeze_record_sha256": physical_sha,
                "official_test_evaluation_computed": False,
            }
        ),
        encoding="utf-8",
    )

    def fake_hash(path: str | Path) -> str:
        resolved = Path(path)
        if resolved == DECISION:
            return decision_sha
        if resolved == manifest:
            return evidence["manifest_sha256"]
        if resolved == freeze:
            return physical_sha
        if resolved == report:
            return report_sha
        return real_sha256_file(resolved)

    monkeypatch.setattr(recovery, "sha256_file", fake_hash)
    lineage = recovery.validate_recovery_binding(
        manifest_path=manifest,
        recovery_decision_path=DECISION,
        recovery_binding_report_path=report,
    )
    assert (
        lineage["historical_final_freeze_record_sha256"]
        == "aed2e96afd2749250d4151780bb4002d198eb96d7433d2bef5b03d4a6ac9212d"
    )
    assert lineage["physical_freeze_record_sha256"] == physical_sha
    assert lineage["recovery_binding_report_sha256"] == report_sha


def test_run_identity_binds_recovery_lineage() -> None:
    from plantxai_stability.joint_execution import build_run_identity

    kwargs = {
        "run_id": "run",
        "model_id": "resnet50",
        "xai_method": "score_cam",
        "scenario_ids": ["rotation_mild"],
        "sample_ids": ["sample"],
        "seed": 42,
        "governance_protocol_hash": "governance",
        "checkpoint_training_protocol_hash": "training",
        "checkpoint_sha256": "checkpoint",
        "manifest_sha256": "manifest",
        "freeze_record_sha256": "historical",
        "checkpoint_decision_record_sha256": "checkpoint-dr",
        "test_decision_record_sha256": "test-dr",
        "g2_readiness_report_sha256": "readiness",
        "campaign_id": "campaign",
        "authorization_decision_id": "DR-TEST-001",
        "transformation_algorithm_version": "v6",
        "xai_policy": {},
        "software_version": "0.1.0",
        "git_commit": "commit",
        "runtime_identity": {"gpu": "A100"},
    }
    without_recovery = build_run_identity(**kwargs)
    with_recovery = build_run_identity(
        **kwargs,
        recovery_lineage={"physical_freeze_record_sha256": "physical"},
    )
    assert "recovery_lineage" not in without_recovery
    assert with_recovery["run_identity_sha256"] != without_recovery[
        "run_identity_sha256"
    ]
