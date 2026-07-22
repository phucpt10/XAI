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


def test_brightness_direction_is_shared_across_severity() -> None:
    pixels = np.full((8, 8, 3), 0.5, dtype=np.float32)
    pipeline = TransformationPipeline(42, {})
    scenarios = [
        Scenario(f"brightness_{name}", "brightness", name, {"factor": factor})
        for name, factor in (("mild", 0.1), ("moderate", 0.3), ("severe", 0.5))
    ]
    results = [pipeline.apply(pixels, "pv_a", scenario) for scenario in scenarios]
    directions = [record.parameters["direction"] for _, record in results]
    magnitudes = [float(np.mean(np.abs(output - pixels))) for output, _ in results]
    assert len(set(directions)) == 1
    assert magnitudes[0] < magnitudes[1] < magnitudes[2]
    assert len({record.seed for _, record in results}) == 1


def test_rotation_direction_is_shared_across_severity() -> None:
    pixels = np.full((16, 16, 3), 0.5, dtype=np.float32)
    pipeline = TransformationPipeline(42, {})
    scenarios = [
        Scenario(
            f"rotation_{name}",
            "rotation",
            name,
            {
                "angle_degrees": angle,
                "fill_policy": "border_median",
                "border_fraction": 0.05,
            },
        )
        for name, angle in (("mild", 10.0), ("moderate", 25.0), ("severe", 45.0))
    ]
    records = [pipeline.apply(pixels, "pv_a", scenario)[1] for scenario in scenarios]
    signed_angles = [float(record.parameters["angle_degrees"]) for record in records]
    assert all(angle > 0 for angle in signed_angles) or all(angle < 0 for angle in signed_angles)
    assert [abs(angle) for angle in signed_angles] == [10.0, 25.0, 45.0]


def test_rotation_resolves_border_median_instead_of_black_fill() -> None:
    pixels = np.full((32, 32, 3), 0.6, dtype=np.float32)
    pixels[8:24, 8:24] = 0.2
    scenario = Scenario(
        "rotation_mild",
        "rotation",
        "mild",
        {
            "angle_degrees": 10.0,
            "fill_policy": "border_median",
            "border_fraction": 0.05,
        },
    )
    output, record = TransformationPipeline(42, {}).apply(pixels, "pv_a", scenario)
    assert record.parameters["resolved_fill_rgb_uint8"] == [153, 153, 153]
    assert float(output[0, 0].min()) > 0.5


def test_gaussian_noise_uses_shared_standard_field() -> None:
    pixels = np.full((16, 16, 3), 0.5, dtype=np.float32)
    pipeline = TransformationPipeline(42, {})
    scenarios = [
        Scenario(
            f"gaussian_noise_{name}",
            "gaussian_noise",
            name,
            {"sigma": sigma, "mean": 0.0},
        )
        for name, sigma in (("mild", 0.01), ("moderate", 0.02), ("severe", 0.03))
    ]
    outputs = [pipeline.apply(pixels, "pv_a", scenario)[0] for scenario in scenarios]
    standardized = [(output - pixels) / scenario.parameters["sigma"] for output, scenario in zip(outputs, scenarios)]
    assert np.allclose(standardized[0], standardized[1], atol=5e-6)
    assert np.allclose(standardized[0], standardized[2], atol=5e-6)
