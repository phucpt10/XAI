import numpy as np
import pytest

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
                "fill_policy": "constant_zero",
                "validity_threshold": 0.999999,
                "opencv_distribution_version": "4.13.0.92",
            },
        )
        for name, angle in (("mild", 10.0), ("moderate", 25.0), ("severe", 45.0))
    ]
    records = [pipeline.apply(pixels, "pv_a", scenario)[1] for scenario in scenarios]
    signed_angles = [float(record.parameters["angle_degrees"]) for record in records]
    assert all(angle > 0 for angle in signed_angles) or all(angle < 0 for angle in signed_angles)
    assert [abs(angle) for angle in signed_angles] == [10.0, 25.0, 45.0]
    fractions = [float(record.parameters["invalid_pixel_fraction"]) for record in records]
    assert fractions[0] < fractions[1] < fractions[2]
    assert all(record.forward_metadata["kind"] == "rotation" for record in records)


def test_rotation_zero_fill_uses_a_geometric_valid_region_mask() -> None:
    ramp = np.linspace(0.4, 0.8, 32, dtype=np.float32)
    pixels = np.repeat(ramp[None, :, None], 32, axis=0)
    pixels = np.repeat(pixels, 3, axis=2)
    pixels[8:24, 8:24] = 0.2
    scenario = Scenario(
        "rotation_mild",
        "rotation",
        "mild",
        {
            "angle_degrees": 10.0,
            "fill_policy": "constant_zero",
            "validity_threshold": 0.999999,
            "opencv_distribution_version": "4.13.0.92",
        },
    )
    output, record = TransformationPipeline(42, {}).apply(pixels, "pv_a", scenario)
    repeated, repeated_record = TransformationPipeline(42, {}).apply(
        pixels, "pv_a", scenario
    )
    assert output.shape == pixels.shape
    assert np.array_equal(output, repeated)
    assert record == repeated_record
    assert record.parameters["rotation_fill_policy"] == "constant_zero"
    assert record.parameters["valid_region_policy"] == "geometric_support_mask"
    assert record.parameters["valid_pixel_count"] > 0
    assert record.parameters["invalid_pixel_count"] > 0
    assert len(record.parameters["valid_mask_sha256"]) == 64
    assert len(record.valid_mask_sha256 or "") == 64
    assert float(output.min()) == 0.0


def test_rotation_valid_mask_does_not_depend_on_real_black_pixels() -> None:
    light = np.full((32, 32, 3), 0.6, dtype=np.float32)
    black_center = light.copy()
    black_center[10:22, 10:22] = 0.0
    scenario = Scenario(
        "rotation_mild",
        "rotation",
        "mild",
        {
            "angle_degrees": 10.0,
            "fill_policy": "constant_zero",
            "validity_threshold": 0.999999,
            "opencv_distribution_version": "4.13.0.92",
        },
    )
    _, light_record = TransformationPipeline(42, {}).apply(light, "pv_a", scenario)
    output, black_record = TransformationPipeline(42, {}).apply(
        black_center, "pv_a", scenario
    )
    assert light_record.valid_mask_sha256 == black_record.valid_mask_sha256
    assert int(np.count_nonzero(np.all(output == 0.0, axis=2))) > 0


def test_rotation_zero_fill_fails_on_opencv_version_mismatch() -> None:
    scenario = Scenario(
        "rotation_mild",
        "rotation",
        "mild",
        {
            "angle_degrees": 10.0,
            "fill_policy": "constant_zero",
            "validity_threshold": 0.999999,
            "opencv_distribution_version": "0.0.0",
        },
    )
    with pytest.raises(ValueError, match="OpenCV distribution version mismatch"):
        TransformationPipeline(42, {}).apply(
            np.full((16, 16, 3), 0.5, dtype=np.float32), "pv_a", scenario
        )


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
