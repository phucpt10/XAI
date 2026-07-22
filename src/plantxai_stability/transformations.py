"""Deterministic image transformations and inverse metadata."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from plantxai_stability.contracts import TransformationRecord


TRANSFORMATION_ALGORITHM_VERSION = "shared_randomization_reflect_pad_v4"


def derive_seed(global_seed: int, sample_id: str, scenario_id: str) -> int:
    digest = hashlib.sha256(f"{global_seed}:{sample_id}:{scenario_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    transformation: str
    severity: str
    parameters: dict[str, Any]


class TransformationPipeline:
    def __init__(self, global_seed: int, parameter_config: dict[str, Any]) -> None:
        self.global_seed = global_seed
        self.parameter_config = parameter_config

    def apply(self, pixels: np.ndarray, sample_id: str, scenario: Scenario) -> tuple[np.ndarray, TransformationRecord]:
        if pixels.dtype.kind not in "fc" or pixels.ndim != 3 or pixels.shape[-1] != 3:
            raise ValueError("Expected an HWC floating RGB array")
        # The stochastic nuisance realization is shared across severity levels.
        # This isolates severity magnitude from direction/noise resampling.
        seed = derive_seed(self.global_seed, sample_id, scenario.transformation)
        rng = np.random.default_rng(seed)
        params = dict(scenario.parameters)
        params["randomization_scope"] = "sample_transformation_shared_across_severity"
        inverse: dict[str, Any] = {"kind": "identity"}
        output = np.clip(pixels.astype(np.float32, copy=True), 0.0, 1.0)
        if scenario.transformation == "brightness":
            direction = -1.0 if int(rng.integers(0, 2)) == 0 else 1.0
            output = np.clip(output * (1.0 + direction * float(params["factor"])), 0.0, 1.0)
            params["direction"] = direction
        elif scenario.transformation == "gaussian_noise":
            standard_noise = rng.normal(0.0, 1.0, output.shape)
            noise = float(params.get("mean", 0.0)) + float(params["sigma"]) * standard_noise
            output = np.clip(output + noise, 0.0, 1.0).astype(np.float32)
        elif scenario.transformation == "gaussian_blur":
            output = self._blur(output, int(params["kernel_size"]), float(params.get("sigma", 1.0)))
        elif scenario.transformation == "rotation":
            angle = float(params["angle_degrees"])
            direction = -1.0 if int(rng.integers(0, 2)) == 0 else 1.0
            angle *= direction
            if params.get("padding_policy") != "reflect":
                raise ValueError("Rotation requires padding_policy=reflect")
            maximum_angle = float(params.get("padding_max_angle_degrees", 45.0))
            margin = int(params.get("padding_margin_pixels", 2))
            output, padding, leakage_count = self._rotate_reflect_pad_crop(
                output, angle, maximum_angle, margin
            )
            inverse = {"kind": "rotation", "angle_degrees": -angle}
            params["angle_degrees"] = angle
            params["resolved_padding_tblr"] = list(padding)
            params["outside_canvas_fill_pixel_count"] = leakage_count
        else:
            raise ValueError(f"Unsupported transformation: {scenario.transformation}")
        record = TransformationRecord(sample_id, scenario.scenario_id, scenario.transformation, scenario.severity, seed, params, inverse, None)
        return output, record

    @staticmethod
    def _blur(pixels: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
        try:
            from PIL import Image, ImageFilter
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Pillow is required for Gaussian blur") from exc
        if kernel_size % 2 == 0:
            raise ValueError("Gaussian blur kernel_size must be odd")
        image = Image.fromarray(np.uint8(np.clip(pixels, 0, 1) * 255.0))
        blurred = image.filter(ImageFilter.GaussianBlur(radius=sigma))
        return np.asarray(blurred, dtype=np.float32) / 255.0

    @staticmethod
    def _rotate_reflect_pad_crop(
        pixels: np.ndarray,
        angle: float,
        maximum_angle: float,
        margin: int,
    ) -> tuple[np.ndarray, tuple[int, int, int, int], int]:
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Pillow is required for rotation") from exc
        if maximum_angle < abs(angle) or maximum_angle > 45.0:
            raise ValueError("padding_max_angle_degrees must cover angle and be <= 45")
        if margin < 1:
            raise ValueError("padding_margin_pixels must be positive")
        height, width, _ = pixels.shape
        radians = np.deg2rad(maximum_angle)
        cosine = abs(float(np.cos(radians)))
        sine = abs(float(np.sin(radians)))
        pad_x = max(0, int(np.ceil((cosine * width + sine * height - width) / 2))) + margin
        pad_y = max(0, int(np.ceil((sine * width + cosine * height - height) / 2))) + margin
        padded = np.pad(
            pixels,
            ((pad_y, pad_y), (pad_x, pad_x), (0, 0)),
            mode="reflect",
        )
        image = Image.fromarray(np.uint8(np.clip(padded, 0, 1) * 255.0))
        rotated = image.rotate(
            angle,
            resample=Image.Resampling.BILINEAR,
            expand=False,
            fillcolor=(0, 0, 0),
        )
        validity = Image.fromarray(
            np.full(padded.shape[:2], 255, dtype=np.uint8)
        )
        rotated_validity = validity.rotate(
            angle,
            resample=Image.Resampling.NEAREST,
            expand=False,
            fillcolor=0,
        )
        left, top = pad_x, pad_y
        box = (left, top, left + width, top + height)
        cropped = np.asarray(rotated.crop(box), dtype=np.float32) / 255.0
        cropped_validity = np.asarray(rotated_validity.crop(box), dtype=np.uint8)
        leakage_count = int(np.count_nonzero(cropped_validity != 255))
        if leakage_count:
            raise ValueError("Reflect padding did not cover the final rotation crop")
        return cropped, (pad_y, pad_y, pad_x, pad_x), leakage_count


def scenario_grid(parameter_config: dict[str, Any]) -> list[Scenario]:
    scenarios: list[Scenario] = []
    for transformation, severities in parameter_config.items():
        for severity, parameters in severities.items():
            scenarios.append(Scenario(f"{transformation}_{severity}", transformation, severity, dict(parameters)))
    return scenarios
