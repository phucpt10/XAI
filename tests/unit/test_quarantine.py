from pathlib import Path

import pytest

from plantxai_stability.data.manifest import build_manifest
from plantxai_stability.data.quarantine import (
    TRAIN_OVERLAP_REASON,
    adjudicate_redundant_train_duplicates,
    adjudicate_train_test_leaf_overlap,
    apply_quarantine_registry,
    write_quarantine_adjudication_artifacts,
)


def _audit_row(split: str, index: int, leaf_id: str, reason: str = "OK"):
    return {
        "source_split": split,
        "source_row_index": index,
        "image_path": f"{split}-{index}.jpg",
        "class_name": "Tomato___healthy",
        "resolved_leaf_id": leaf_id,
        "leaf_id_source": "filename_reconstructed",
        "reason_code": reason,
    }


def _manifest_row(split: str, index: int, leaf_id: str):
    return {
        "leaf_id": leaf_id,
        "class_id": 0,
        "class_name": "Tomato___healthy",
        "source_split": split,
        "split": split,
        "canonical_relative_path": f"{split}-{index}.png",
        "canonical_rgb_sha256": f"{index + (0 if split == 'train' else 10):064x}",
        "width": 8,
        "height": 8,
        "source_row_index": index,
    }


def test_quarantine_preserves_test_and_reconciles_every_sample():
    audit_rows = [
        _audit_row("train", 1, "overlap", "LEAF_SPLIT_OVERLAP"),
        _audit_row("test", 2, "overlap", "LEAF_SPLIT_OVERLAP"),
        _audit_row("train", 3, "train-only"),
    ]
    candidates, registry, adjudication = adjudicate_train_test_leaf_overlap(
        audit_rows,
        approved_overlap_ids=["overlap"],
        decision_record_id="DR-LEAF-002",
        expected_quarantined_train_count=1,
        expected_official_test_count=1,
    )
    assert adjudication["passed"] is True
    assert len(candidates) == 2
    assert registry[0]["quarantine_reason_code"] == TRAIN_OVERLAP_REASON

    records = build_manifest(
        [
            _manifest_row("train", 1, "overlap"),
            _manifest_row("test", 2, "overlap"),
            _manifest_row("train", 3, "train-only"),
        ]
    )
    eligible, finalized, lineage, summary = apply_quarantine_registry(
        records,
        registry,
        decision_record_id="DR-LEAF-002",
        expected_audited_count=3,
        expected_quarantined_count=1,
        expected_official_test_count=1,
        expected_eligible_count=2,
        expected_eligible_source_train_count=1,
    )
    assert summary["passed"] is True
    assert len(eligible) == 2
    assert len(finalized) == 1
    assert len(lineage) == 3
    assert sum(record.source_split == "test" for record in eligible) == 1


def test_quarantine_rejects_unapproved_overlap_identity():
    rows = [
        _audit_row("train", 1, "detected", "LEAF_SPLIT_OVERLAP"),
        _audit_row("test", 2, "detected", "LEAF_SPLIT_OVERLAP"),
    ]
    with pytest.raises(ValueError, match="do not exactly match"):
        adjudicate_train_test_leaf_overlap(
            rows,
            approved_overlap_ids=["different"],
            decision_record_id="DR-LEAF-002",
            expected_quarantined_train_count=1,
            expected_official_test_count=1,
        )


def test_quarantine_artifacts_are_immutable(tmp_path: Path):
    candidate = _audit_row("train", 1, "overlap", "LEAF_SPLIT_OVERLAP")
    registry = {
        **candidate,
        "eligibility_status": "quarantined",
        "quarantine_reason_code": TRAIN_OVERLAP_REASON,
    }
    summary = {"passed": True}
    hashes = write_quarantine_adjudication_artifacts(
        [candidate], [registry], summary, tmp_path
    )
    assert "quarantine_decision_registry.parquet" in hashes
    with pytest.raises(FileExistsError):
        write_quarantine_adjudication_artifacts([candidate], [registry], summary, tmp_path)


def test_redundant_train_duplicate_keeps_minimum_sample_id():
    first = _manifest_row("train", 1, "same-leaf")
    second = _manifest_row("train", 2, "same-leaf")
    second["canonical_rgb_sha256"] = first["canonical_rgb_sha256"]
    records = build_manifest([first, second])
    expected_quarantine = max(record.sample_id for record in records)
    candidates, quarantined, summary = adjudicate_redundant_train_duplicates(
        records,
        approved_quarantined_sample_ids=[expected_quarantine],
        decision_record_id="DR-DUP-001",
        expected_group_count=1,
    )
    assert summary["passed"] is True
    assert quarantined[0].sample_id == expected_quarantine
    assert len(candidates) == 2


def test_redundant_duplicate_policy_rejects_test_pairs():
    first = _manifest_row("test", 1, "same-leaf")
    second = _manifest_row("test", 2, "same-leaf")
    second["canonical_rgb_sha256"] = first["canonical_rgb_sha256"]
    records = build_manifest([first, second])
    with pytest.raises(ValueError, match="train-only"):
        adjudicate_redundant_train_duplicates(
            records,
            approved_quarantined_sample_ids=[],
            decision_record_id="DR-DUP-001",
            expected_group_count=1,
        )
