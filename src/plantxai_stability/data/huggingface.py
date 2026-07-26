"""Hugging Face Datasets adapter for the PlantVillage source.

The adapter deliberately keeps ``datasets`` optional.  This keeps the core
package and unit tests lightweight while making the Colab/GitHub data path
explicit and auditable.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from plantxai_stability.contracts import SampleRecord
from plantxai_stability.data.manifest import build_manifest, canonical_rgb_sha256_image
from plantxai_stability.data.plantvillage_source import load_pinned_plantvillage


@dataclass(frozen=True)
class HuggingFaceSchema:
    dataset_id: str
    configuration: str
    revision: str | None
    splits: tuple[str, ...]
    features: tuple[str, ...]
    image_column: str
    label_column: str
    leaf_id_column: str
    label_names: tuple[str, ...]

    @property
    def has_leaf_id(self) -> bool:
        return self.leaf_id_column in self.features


def _as_pil(image: Any) -> Any:
    if isinstance(image, dict):
        try:
            from io import BytesIO
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - base dependency guard
            raise RuntimeError("Pillow is required for HF image materialisation") from exc
        if image.get("bytes") is not None:
            return Image.open(BytesIO(image["bytes"]))
        if image.get("path") is not None:
            return Image.open(image["path"])
    return image


def load_hf_dataset(
    dataset_id: str = "mohanty/PlantVillage",
    configuration: str = "color",
    revision: str | None = None,
    *,
    selected_classes: Iterable[str] | None = None,
    prepare_images: bool = False,
    allow_filename_reconstruction: bool = False,
    cache_dir: str | Path | None = None,
    token: str | None = None,
) -> Any:
    """Load a pinned (or explicitly exploratory) Hugging Face dataset."""
    if dataset_id == "mohanty/PlantVillage":
        if revision is None:
            raise ValueError("mohanty/PlantVillage requires a pinned revision")
        return load_pinned_plantvillage(
            dataset_id=dataset_id,
            configuration=configuration,
            revision=revision,
            selected_classes=selected_classes,
            prepare_images=prepare_images,
            allow_filename_reconstruction=allow_filename_reconstruction,
            cache_dir=cache_dir,
            token=token,
        )
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Hugging Face support requires the optional dependency: "
            "pip install -e '.[hf]'."
        ) from exc
    kwargs: dict[str, Any] = {"path": dataset_id, "name": configuration}
    if revision:
        kwargs["revision"] = revision
    return load_dataset(**kwargs)


def _feature_names(dataset: Any, label_column: str) -> tuple[str, ...]:
    feature = dataset.features.get(label_column)
    names = getattr(feature, "names", None)
    if names is None:
        return ()
    return tuple(str(name) for name in names)


def inspect_hf_schema(
    dataset: Any,
    *,
    dataset_id: str = "mohanty/PlantVillage",
    configuration: str = "color",
    revision: str | None = None,
    image_column: str = "image",
    label_column: str = "label",
    leaf_id_column: str = "leaf_id",
) -> HuggingFaceSchema:
    """Extract the schema without decoding or materialising image pixels."""
    first_split = next(iter(dataset.values())) if hasattr(dataset, "values") else dataset
    features = tuple(str(name) for name in first_split.features.keys())
    missing = [
        name for name in (image_column, label_column, leaf_id_column) if name not in features
    ]
    if missing:
        raise ValueError(f"PlantVillage schema missing required column(s): {missing}")
    split_names = tuple(str(name) for name in dataset.keys()) if hasattr(dataset, "keys") else ()
    return HuggingFaceSchema(
        dataset_id=dataset_id,
        configuration=configuration,
        revision=revision,
        splits=split_names,
        features=features,
        image_column=image_column,
        label_column=label_column,
        leaf_id_column=leaf_id_column,
        label_names=_feature_names(first_split, label_column),
    )


def _image_source_key(
    image: Any,
    digest: str,
    dataset_id: str,
    configuration: str,
    split: str,
    declared_source_path: Any = None,
) -> str:
    """Return a stable source key without using a row index."""
    if declared_source_path not in (None, ""):
        return str(declared_source_path).replace("\\", "/")
    if isinstance(image, dict) and image.get("path"):
        return str(image["path"]).replace("\\", "/")
    path = getattr(image, "filename", None) or getattr(image, "path", None)
    if path:
        return str(path).replace("\\", "/")
    # The pixel digest is stable across materialisation and avoids row-order IDs.
    return f"hf://{dataset_id}/{configuration}/{split}/{digest}.png"


def _label_name(value: Any, names: tuple[str, ...]) -> str:
    if isinstance(value, int):
        if not names or value < 0 or value >= len(names):
            raise ValueError(f"Label index {value} is not present in dataset ClassLabel names")
        return names[value]
    return str(value)


def iter_manifest_rows(
    dataset: Any,
    schema: HuggingFaceSchema,
    *,
    selected_classes: Iterable[str] | None = None,
    materialize_root: str | Path | None = None,
) -> Iterable[dict[str, Any]]:
    """Yield canonical manifest rows from all HF splits.

    The upstream ``train``/``test`` assignment is preserved.  No custom split
    is inferred here; leaf-aware regrouping belongs to the governance module.
    """
    selected = tuple(selected_classes) if selected_classes is not None else schema.label_names
    class_filter = set(selected)
    unknown = sorted(class_filter.difference(schema.label_names))
    if unknown:
        raise ValueError(f"Selected class(es) are absent from the HF label vocabulary: {unknown}")
    # Remap the selected protocol classes to contiguous model IDs.  Using the
    # global ClassLabel index would silently create gaps for the five-class
    # tomato protocol.
    class_ids = {name: index for index, name in enumerate(selected)}
    for split_name, split in dataset.items():
        for fallback_row_index, row in enumerate(split):
            source_row_index = int(row.get("_source_row_index", fallback_row_index))
            class_name = _label_name(row[schema.label_column], schema.label_names)
            if class_name not in class_filter:
                continue
            image = row[schema.image_column]
            digest, width, height = canonical_rgb_sha256_image(image)
            leaf_id = row.get(schema.leaf_id_column)
            if leaf_id in (None, ""):
                raise ValueError("PlantVillage row has an empty leaf_id; refusing unsafe splits")
            source_key = _image_source_key(
                image,
                digest,
                schema.dataset_id,
                schema.configuration,
                split_name,
                row.get("image_path"),
            )
            if materialize_root is not None:
                # Export decoded RGB pixels so the existing filesystem-backed
                # PyTorch Dataset can consume the exact audited bytes in Colab.
                # The source-key digest preserves lineage when two distinct
                # source samples have identical canonical RGB pixels.
                source_key_sha256 = hashlib.sha256(source_key.encode("utf-8")).hexdigest()
                relative = (
                    Path("images") / split_name / class_name / f"{source_key_sha256}.png"
                )
                output = Path(materialize_root) / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                if not output.exists():
                    rgb = _as_pil(image).convert("RGB")
                    rgb.save(output, format="PNG")
                source_key = relative.as_posix()
            yield {
                "leaf_id": str(leaf_id),
                "class_id": class_ids.get(class_name, 0),
                "class_name": class_name,
                "source_split": str(split_name),
                "split": str(split_name),
                "canonical_relative_path": source_key,
                "canonical_rgb_sha256": digest,
                "width": width,
                "height": height,
                "source_row_index": source_row_index,
            }


def build_hf_manifest(
    dataset: Any,
    schema: HuggingFaceSchema,
    *,
    selected_classes: Iterable[str] | None = None,
    materialize_root: str | Path | None = None,
) -> list[SampleRecord]:
    """Build the canonical manifest for a loaded Hugging Face dataset."""
    return build_manifest(
        iter_manifest_rows(
            dataset,
            schema,
            selected_classes=selected_classes,
            materialize_root=materialize_root,
        )
    )


def split_counts(dataset: Any) -> dict[str, int]:
    """Return split sizes without decoding images."""
    return {str(name): int(len(split)) for name, split in dataset.items()}
