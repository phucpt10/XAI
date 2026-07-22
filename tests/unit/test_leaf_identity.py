from plantxai_stability.data.leaf_identity import audit_leaf_identity


def _row(
    image_path: str,
    class_name: str,
    mapped: str,
    reconstructed: str,
    status: str = "missing_leaf_map_match",
):
    return {
        "image_path": image_path,
        "label": class_name,
        "leaf_id": mapped,
        "mapped_leaf_id": mapped,
        "reconstructed_leaf_id": reconstructed,
        "source_leaf_identity": reconstructed.removeprefix("fallback_"),
        "leaf_map_status": status,
        "leaf_map_suggestions": (),
        "_source_row_index": 0,
    }


def test_leaf_reconstruction_passes_without_conflicts():
    dataset = {
        "train": [
            _row("a.jpg", "A", "mapped-a", "fallback_a", "unique_leaf_map_match"),
            _row("b.jpg", "A", "", "fallback_b"),
        ],
        "test": [_row("c.jpg", "A", "mapped-c", "fallback_c", "unique_leaf_map_match")],
    }
    rows, summary = audit_leaf_identity(dataset)
    assert summary["coverage"] == 1.0
    assert summary["passed"] is True
    assert {row["leaf_id_source"] for row in rows} == {"leaf_map", "filename_reconstructed"}


def test_leaf_reconstruction_rejects_collision_and_class_conflict():
    dataset = {
        "train": [
            _row("a.jpg", "A", "", "fallback_same"),
            _row("b.jpg", "B", "", "fallback_same"),
        ]
    }
    _, summary = audit_leaf_identity(dataset)
    assert summary["collision_candidate_count"] == 1
    assert summary["leaf_class_conflict_count"] == 1
    assert summary["passed"] is False


def test_leaf_reconstruction_rejects_cross_split_overlap():
    dataset = {
        "train": [_row("a.jpg", "A", "mapped-a", "fallback_a")],
        "test": [_row("b.jpg", "A", "mapped-a", "fallback_b")],
    }
    _, summary = audit_leaf_identity(dataset)
    assert summary["leaf_split_overlap_count"] == 1
    assert summary["leaf_split_overlap_ids"] == ["mapped-a"]
    assert summary["passed"] is False


def test_leaf_reconstruction_rejects_ambiguous_map_match():
    dataset = {
        "train": [
            _row("a.jpg", "A", "", "fallback_a", "ambiguous_leaf_map_match")
        ]
    }
    _, summary = audit_leaf_identity(dataset)
    assert summary["ambiguous_sample_count"] == 1
    assert summary["passed"] is False
