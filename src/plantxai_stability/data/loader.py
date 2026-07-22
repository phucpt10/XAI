"""Dataset adapter and DataLoader factory with identity-preserving batches."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from plantxai_stability.contracts import SampleRecord
from plantxai_stability.data.manifest import canonical_rgb_sha256


def load_rgb_float(path: str | Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required for image loading") from exc
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def load_verified_record(record: SampleRecord, root: str | Path, verify_hash: bool = True) -> np.ndarray:
    """Load one frozen-manifest image and enforce lineage, shape and hash."""
    path = Path(root) / record.canonical_relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Frozen manifest image is missing: {path}")
    digest, width, height = canonical_rgb_sha256(path)
    if (width, height) != (record.width, record.height):
        raise ValueError(f"Image dimensions changed for {record.sample_id}")
    if verify_hash and digest != record.canonical_rgb_sha256:
        raise ValueError(f"Canonical RGB hash mismatch for {record.sample_id}")
    pixels = load_rgb_float(path)
    if pixels.shape != (record.height, record.width, 3):
        raise ValueError(f"Invalid RGB shape for {record.sample_id}: {pixels.shape}")
    if not np.isfinite(pixels).all() or np.any((pixels < 0.0) | (pixels > 1.0)):
        raise ValueError(f"Invalid pixel range for {record.sample_id}")
    return pixels


def preprocess_for_model(
    pixels: np.ndarray,
    image_size: tuple[int, int] = (224, 224),
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> Any:
    """Deterministic resize-shortest-side, center-crop and ImageNet normalize."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required for model preprocessing") from exc
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB pixels, got {pixels.shape}")
    image = Image.fromarray(np.uint8(np.clip(pixels, 0.0, 1.0) * 255.0))
    target_h, target_w = image_size
    scale = max(target_h / image.height, target_w / image.width)
    resized_size = (max(target_w, round(image.width * scale)), max(target_h, round(image.height * scale)))
    resized_image = image.resize(resized_size, Image.Resampling.BILINEAR)
    left = (resized_image.width - target_w) // 2
    top = (resized_image.height - target_h) // 2
    resized = np.asarray(
        resized_image.crop((left, top, left + target_w, top + target_h)), dtype=np.float32
    ) / 255.0
    normalized = (resized - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    chw = np.transpose(normalized, (2, 0, 1)).copy()
    try:
        import torch
        return torch.from_numpy(chw).float()
    except ImportError:
        return chw


class PlantDataset:
    """Torch-compatible dataset without importing torch at module import time."""

    def __init__(
        self,
        records: Sequence[SampleRecord],
        root: str | Path,
        transform: Any = None,
        expected_split: str | None = None,
        verify_hash: bool = True,
        num_classes: int | None = None,
    ) -> None:
        self.records = sorted(list(records), key=lambda item: item.sample_id)
        self.root = Path(root)
        self.transform = transform
        self.expected_split = expected_split
        self.verify_hash = verify_hash
        if len({record.sample_id for record in self.records}) != len(self.records):
            raise ValueError("Frozen manifest contains duplicate sample_id values")
        if expected_split is not None and any(
            record.split != expected_split for record in self.records
        ):
            raise ValueError(f"Dataset contains records outside expected split {expected_split}")
        if num_classes is not None and any(
            record.class_id < 0 or record.class_id >= num_classes for record in self.records
        ):
            raise ValueError("Frozen manifest contains an out-of-range class_id")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        if self.expected_split is not None and record.split != self.expected_split:
            raise ValueError(
                f"Sample {record.sample_id} belongs to {record.split}, "
                f"not expected split {self.expected_split}"
            )
        pixels = load_verified_record(record, self.root, self.verify_hash)
        if self.transform is not None:
            pixels = self.transform(pixels, record)
        if not np.isfinite(pixels).all() or np.any((pixels < 0.0) | (pixels > 1.0)):
            raise ValueError(f"Transformation produced invalid pixels for {record.sample_id}")
        return {
            "pixel_array": pixels,
            "image_pixel_tensor": pixels,
            "model_tensor": preprocess_for_model(pixels),
            "label": record.class_id,
            "sample_id": record.sample_id,
            "leaf_id": record.leaf_id,
            "split": record.split,
            "class_name": record.class_name,
            "canonical_relative_path": record.canonical_relative_path,
            "canonical_rgb_sha256": record.canonical_rgb_sha256,
            "source_split": record.source_split,
            "source_row_index": record.source_row_index,
        }


def build_torch_dataloader(
    dataset: PlantDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    seed: int = 42,
) -> Any:
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required to build a DataLoader") from exc
    if dataset.expected_split == "test" and shuffle:
        raise ValueError("Official test DataLoader must use shuffle=False")
    import random

    generator = torch.Generator()
    generator.manual_seed(seed)

    def seed_worker(worker_id: int) -> None:  # noqa: ARG001
        worker_seed = torch.initial_seed() % (2**32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        generator=generator,
        worker_init_fn=seed_worker,
    )
