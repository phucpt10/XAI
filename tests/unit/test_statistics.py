import pytest

from plantxai_stability.statistics import holm_adjust


def test_holm_adjustment_is_bounded_and_monotone_in_sorted_order() -> None:
    adjusted = holm_adjust([0.001, 0.02, 0.5])
    assert all(0.0 <= value <= 1.0 for value in adjusted)
    assert adjusted[0] <= adjusted[1] <= adjusted[2]


def test_empty_bootstrap_inputs_fail() -> None:
    from plantxai_stability.statistics import bootstrap_leaf_means

    with pytest.raises(ValueError):
        bootstrap_leaf_means([], [], iterations=10, seed=42)
