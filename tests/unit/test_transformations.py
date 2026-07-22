import numpy as np

from plantxai_stability.transformations import Scenario, TransformationPipeline, derive_seed


def test_seed_is_stable() -> None:
    assert derive_seed(42, "pv_a", "brightness_mild") == derive_seed(42, "pv_a", "brightness_mild")
    assert derive_seed(42, "pv_a", "brightness_mild") != derive_seed(42, "pv_b", "brightness_mild")


def test_brightness_is_deterministic_and_bounded() -> None:
    pixels = np.full((8, 8, 3), 0.5, dtype=np.float32)
    pipeline = TransformationPipeline(42, {})
    scenario = Scenario("brightness_mild", "brightness", "mild", {"factor": 0.1})
    left, left_record = pipeline.apply(pixels, "pv_a", scenario)
    right, right_record = pipeline.apply(pixels, "pv_a", scenario)
    assert np.array_equal(left, right)
    assert left_record == right_record
    assert float(left.min()) >= 0.0 and float(left.max()) <= 1.0
