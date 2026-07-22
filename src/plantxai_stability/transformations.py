"""Deterministic image transformations and inverse metadata."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any

import numpy as np

from plantxai_stability.contracts import TransformationRecord


TRANSFORMATION_ALGORITHM_VERSION = "shared_randomization_telea_inpainting_v5"


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
            if params.get("completion_method") != "opencv_telea":
                raise ValueError("Rotation requires completion_method=opencv_telea")
            radius = float(params.get("inpaint_radius", 3.0))
            dilation = int(params.get("mask_dilation_pixels", 1))
            expected_opencv = str(params["opencv_distribution_version"])
            output, completion, valid_mask_sha256 = self._rotate_telea_inpaint(
                output, angle, radius, dilation, expected_opencv
            )
            inverse = {"kind": "rotation", "angle_degrees": -angle}
            params["angle_degrees"] = angle
            params.update(completion)
        else:
            raise ValueError(f"Unsupported transformation: {scenario.transformation}")
        record = TransformationRecord(
            sample_id,
            scenario.scenario_id,
            scenario.transformation,
            scenario.severity,
            seed,
            params,
            inverse,
            valid_mask_sha256 if scenario.transformation == "rotation" else None,
        )
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
    def _rotate_telea_inpaint(
        pixels: np.ndarray,
        angle: float,
        radius: float,
        dilation_pixels: int,
        expected_opencv_distribution_version: str,
    ) -> tuple[np.ndarray, dict[str, Any], str]:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("OpenCV is required for Telea rotation completion") from exc
        distribution_version = version("opencv-python-headless")
        if distribution_version != expected_opencv_distribution_version:
            raise ValueError(
                "OpenCV distribution version mismatch: "
                f"expected {expected_opencv_distribution_version}, "
                f"found {distribution_version}"
            )
        if not 0.0 < radius <= 10.0:
            raise ValueError("inpaint_radius must be in (0, 10]")
        if not 0 <= dilation_pixels <= 5:
            raise ValueError("mask_dilation_pixels must be in [0, 5]")
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
        rotated_validity = cv2.warpAffine(
            np.full((height, width), 255, dtype=np.uint8),
            matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        geometric_mask = np.where(rotated_validity == 0, 255, 0).astype(np.uint8)
        invalid_pixel_count = int(np.count_nonzero(geometric_mask))
        if invalid_pixel_count == 0:
            raise ValueError("Rotation produced no geometric completion region")
        inpaint_mask = np.asarray(geometric_mask, dtype=np.uint8)
        if dilation_pixels:
            size = 2 * dilation_pixels + 1
            kernel = np.ones((size, size), dtype=np.uint8)
            inpaint_mask = np.asarray(
                cv2.dilate(inpaint_mask, kernel, iterations=1), dtype=np.uint8
            )
        completed = cv2.inpaint(
            rotated,
            inpaint_mask,
            inpaintRadius=radius,
            flags=cv2.INPAINT_TELEA,
        )
        known = inpaint_mask == 0
        known_pixel_change_count = int(
            np.count_nonzero(np.any(completed[known] != rotated[known], axis=1))
        )
        if known_pixel_change_count:
            raise ValueError("Telea changed pixels outside the declared inpaint mask")
        inpainted_pixel_count = int(np.count_nonzero(inpaint_mask))
        mask_sha256 = hashlib.sha256(inpaint_mask.tobytes()).hexdigest()
        valid_mask_sha256 = hashlib.sha256((inpaint_mask == 0).tobytes()).hexdigest()
        completion = {
            "rotation_completion_method": "opencv_telea",
            "opencv_version": str(cv2.__version__),
            "opencv_distribution_version": distribution_version,
            "opencv_num_threads": int(cv2.getNumThreads()),
            "opencv_opencl_enabled": bool(
                cv2.ocl.useOpenCL() if hasattr(cv2, "ocl") else False
            ),
            "geometric_invalid_pixel_count": invalid_pixel_count,
            "inpainted_pixel_count": inpainted_pixel_count,
            "inpainted_fraction": inpainted_pixel_count / float(height * width),
            "inpaint_mask_sha256": mask_sha256,
            "known_pixel_change_count": known_pixel_change_count,
            "inpainted_output_rgb_sha256": hashlib.sha256(completed.tobytes()).hexdigest(),
        }
        return completed.astype(np.float32) / 255.0, completion, valid_mask_sha256


def scenario_grid(parameter_config: dict[str, Any]) -> list[Scenario]:
    scenarios: list[Scenario] = []
    for transformation, severities in parameter_config.items():
        for severity, parameters in severities.items():
            scenarios.append(Scenario(f"{transformation}_{severity}", transformation, severity, dict(parameters)))
    return scenarios
