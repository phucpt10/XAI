from dataclasses import replace

from plantxai_stability.contracts import SampleRecord
from plantxai_stability.severity import (
    select_leaf_balanced_pilot_records,
    summarize_pilot_rows,
)


def _record(sample_id: str, leaf_id: str, class_name: str) -> SampleRecord:
    return SampleRecord(
        sample_id=sample_id,
        leaf_id=leaf_id,
        class_id=0,
        class_name=class_name,
        source_split="train",
        split="validation",
        canonical_relative_path=f"images/{sample_id}.png",
        canonical_rgb_sha256="a" * 64,
        width=256,
        height=256,
        source_row_index=1,
    )


def test_pilot_selection_is_deterministic_and_leaf_unique() -> None:
    records = [
        _record("pv_1", "leaf_1", "class_a"),
        _record("pv_2", "leaf_1", "class_a"),
        _record("pv_3", "leaf_2", "class_a"),
        replace(_record("pv_4", "leaf_3", "class_b"), class_id=1),
    ]
    left = select_leaf_balanced_pilot_records(
        records, seed=42, max_leaves_per_class=2
    )
    right = select_leaf_balanced_pilot_records(
        list(reversed(records)), seed=42, max_leaves_per_class=2
    )
    assert [item.sample_id for item in left] == [item.sample_id for item in right]
    assert len({item.leaf_id for item in left}) == len(left)


def test_summary_requires_increasing_median_rmse() -> None:
    rows = []
    for severity, rmse in zip(("mild", "moderate", "severe"), (0.1, 0.2, 0.3)):
        rows.append(
            {
                "scenario_id": f"brightness_{severity}",
                "transformation": "brightness",
                "severity": severity,
                "mae": rmse,
                "rmse": rmse,
                "psnr": 20.0,
                "ssim": 0.9,
                "clipped_fraction": 0.0,
            }
        )
    summary = summarize_pilot_rows(rows)
    assert summary["all_metrics_finite"] is True
    assert summary["ordinal_gate_passed"] is True
