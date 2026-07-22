from plantxai_stability.data.audit import audit_manifest_records
from plantxai_stability.data.manifest import build_manifest
from plantxai_stability.data.splits import group_train_validation


def _row(leaf: str, path: str, class_name: str = "Tomato___healthy", split: str = "train"):
    return {
        "leaf_id": leaf,
        "class_id": 0 if class_name.endswith("healthy") else 1,
        "class_name": class_name,
        "source_split": split,
        "split": split,
        "canonical_relative_path": path,
        "canonical_rgb_sha256": ("a" if path == "a.png" else "b") * 64,
        "width": 8,
        "height": 8,
    }


def test_audit_detects_cross_split_and_leaf_conflicts():
    records = build_manifest(
        [
            _row("leaf-a", "a.png"),
            _row("leaf-a", "b.png", split="test"),
        ]
    )
    rows, summary = audit_manifest_records(records)
    assert summary["leaf_split_leakage_count"] == 1
    assert summary["passed"] is False
    assert all(row["valid"] is False for row in rows)


def test_group_split_is_class_stratified_and_keeps_test():
    rows = []
    for class_name, prefix in (("Tomato___healthy", "h"), ("Tomato___Early_blight", "e")):
        for index in range(4):
            rows.append(_row(f"{prefix}-leaf-{index}", f"{prefix}-{index}.png", class_name))
    rows.append(_row("test-leaf", "test.png", split="test"))
    records = build_manifest(rows)
    split_records = group_train_validation(records, validation_fraction=0.25, seed=42)
    assert all(record.split == "test" for record in split_records if record.source_split == "test")
    validation = [record for record in split_records if record.split == "validation"]
    assert {record.class_name for record in validation} == {
        "Tomato___healthy",
        "Tomato___Early_blight",
    }
