"""Deterministic image transformations with alignment metadata."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any

import numpy as np

from plantxai_stability.contracts import TransformationRecord


TRANSFORMATION_ALGORITHM_VERSION = "shared_randomization_zero_fill_valid_mask_v7"


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
        forward: dict[str, Any] = {"kind": "identity"}
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
            output = self._blur(
                output,
                int(params["kernel_size"]),
                float(params.get("sigma", 1.0)),
                str(params["opencv_distribution_version"]),
            )
            params["operator"] = "opencv_gaussian_blur"
            params["border_mode"] = "reflect_101"
        elif scenario.transformation == "rotation":
            angle = float(params["angle_degrees"])
            direction = -1.0 if int(rng.integers(0, 2)) == 0 else 1.0
            angle *= direction
            if params.get("fill_policy") != "constant_zero":
                raise ValueError("Rotation requires fill_policy=constant_zero")
            validity_threshold = float(params.get("validity_threshold", 0.999999))
            expected_opencv = str(params["opencv_distribution_version"])
            output, rotation_metadata, valid_mask_sha256 = self._rotate_zero_fill(
                output, angle, validity_threshold, expected_opencv
            )
            forward = {
                "kind": "rotation",
                "angle_degrees": angle,
                "validity_threshold": validity_threshold,
            }
            inverse = {"kind": "rotation", "angle_degrees": -angle}
            params["angle_degrees"] = angle
            params.update(rotation_metadata)
        else:
            raise ValueError(f"Unsupported transformation: {scenario.transformation}")
        record = TransformationRecord(
            sample_id,
            scenario.scenario_id,
            scenario.transformation,
            scenario.severity,
            seed,
            params,
            forward,
            inverse,
            valid_mask_sha256 if scenario.transformation == "rotation" else None,
        )
        return output, record

    @staticmethod
    def _blur(
        pixels: np.ndarray,
        kernel_size: int,
        sigma: float,
        expected_opencv_distribution_version: str = "4.13.0.92",
    ) -> np.ndarray:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("OpenCV is required for Gaussian blur") from exc
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("Gaussian blur kernel_size must be a positive odd integer")
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("Gaussian blur sigma must be finite and positive")
        distribution_version = version("opencv-python-headless")
        if distribution_version != expected_opencv_distribution_version:
            raise ValueError(
                "OpenCV distribution version mismatch: "
                f"expected {expected_opencv_distribution_version}, "
                f"found {distribution_version}"
            )
        cv2.setNumThreads(1)
        if hasattr(cv2, "ocl"):
            cv2.ocl.setUseOpenCL(False)
        source = np.rint(np.clip(pixels, 0.0, 1.0) * 255.0).astype(np.uint8)
        blurred = cv2.GaussianBlur(
            source,
            (kernel_size, kernel_size),
            sigmaX=sigma,
            sigmaY=sigma,
            borderType=cv2.BORDER_REFLECT_101,
        )
        return np.asarray(blurred, dtype=np.float32) / 255.0

    @staticmethod
    def _rotate_zero_fill(
        pixels: np.ndarray,
        angle: float,
        validity_threshold: float,
        expected_opencv_distribution_version: str,
    ) -> tuple[np.ndarray, dict[str, Any], str]:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("OpenCV is required for zero-fill rotation") from exc
        distribution_version = version("opencv-python-headless")
        if distribution_version != expected_opencv_distribution_version:
            raise ValueError(
                "OpenCV distribution version mismatch: "
                f"expected {expected_opencv_distribution_version}, "
                f"found {distribution_version}"
            )
        if not 0.99 <= validity_threshold <= 1.0:
            raise ValueError("validity_threshold must be in [0.99, 1.0]")
        cv2.setNumThreads(1)
        if hasattr(cv2, "ocl"):
            cv2.ocl.setUseOpenCL(False)
        height, width, _ = pixels.shape
        source = np.rint(np.clip(pixels, 0.0, 1.0) * 255.0).astype(np.uint8)
        center = ((width - 1) / 2.0, (height - 1) / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            source,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0.0, 0.0, 0.0),
        )
        warped_validity = cv2.warpAffine(
            np.ones((height, width), dtype=np.float32),
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        valid_mask = np.asarray(
            warped_validity >= validity_threshold, dtype=np.uint8
        )
        valid_pixel_count = int(np.count_nonzero(valid_mask))
        invalid_pixel_count = int(height * width - valid_pixel_count)
        if invalid_pixel_count == 0:
            raise ValueError("Rotation produced no invalid support region")
        valid_mask_sha256 = hashlib.sha256(valid_mask.tobytes()).hexdigest()
        metadata = {
            "rotation_fill_policy": "constant_zero",
            "valid_region_policy": "geometric_support_mask",
            "validity_threshold": validity_threshold,
            "opencv_version": str(cv2.__version__),
            "opencv_distribution_version": distribution_version,
            "opencv_num_threads": int(cv2.getNumThreads()),
            "opencv_opencl_enabled": bool(
                cv2.ocl.useOpenCL() if hasattr(cv2, "ocl") else False
            ),
            "valid_pixel_count": valid_pixel_count,
            "valid_pixel_fraction": valid_pixel_count / float(height * width),
            "invalid_pixel_count": invalid_pixel_count,
            "invalid_pixel_fraction": invalid_pixel_count / float(height * width),
            "valid_mask_sha256": valid_mask_sha256,
            "rotated_output_rgb_sha256": hashlib.sha256(rotated.tobytes()).hexdigest(),
        }
        return rotated.astype(np.float32) / 255.0, metadata, valid_mask_sha256


def scenario_grid(parameter_config: dict[str, Any]) -> list[Scenario]:
    scenarios: list[Scenario] = []
    for transformation, severities in parameter_config.items():
        for severity, parameters in severities.items():
            scenarios.append(Scenario(f"{transformation}_{severity}", transformation, severity, dict(parameters)))
    return scenarios
