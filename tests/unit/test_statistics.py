import numpy as np
import pytest

from plantxai_stability.statistics import _masked_topk_iou, holm_adjust, paired_wilcoxon


def test_holm_adjustment_is_bounded_and_monotone_in_sorted_order() -> None:
    adjusted = holm_adjust([0.001, 0.02, 0.5])
    assert all(0.0 <= value <= 1.0 for value in adjusted)
    assert adjusted[0] <= adjusted[1] <= adjusted[2]


def test_empty_bootstrap_inputs_fail() -> None:
    from plantxai_stability.statistics import bootstrap_leaf_means

    with pytest.raises(ValueError):
        bootstrap_leaf_means([], [], iterations=10, seed=42)


def test_identical_paired_values_have_neutral_wilcoxon_result() -> None:
    result = paired_wilcoxon([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert result == {
        "statistic": 0.0,
        "p_value": 1.0,
        "rank_biserial": 0.0,
        "n_pairs": 3,
        "n_zero_differences": 3,
        "n_unique_absolute_nonzero_differences": 0,
        "method": "asymptotic",
        "zero_method": "pratt",
        "continuity_correction": False,
    }


def test_topk_iou_selects_exact_cardinality_under_ties() -> None:
    left = np.zeros((2, 5), dtype=float)
    right = np.zeros((2, 5), dtype=float)
    left[0, 0] = 1.0
    right[0, 0] = 1.0
    mask = np.ones((2, 5), dtype=bool)
    assert _masked_topk_iou(left, right, mask, 0.1) == 1.0
    shifted = np.roll(right, 1, axis=1)
    assert _masked_topk_iou(left, shifted, mask, 0.1) == 0.0
