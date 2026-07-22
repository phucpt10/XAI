"""Pinned PlantVillage source loader independent of HF auto-converted configs.

Hugging Face Datasets 4+ no longer executes legacy dataset scripts.  The Hub's
auto-converted ``default`` config does not preserve the scripted ``color``
contract, so this module reads the immutable repository files directly.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any, Iterable

from plantxai_stability.provenance import sha256_file


class _LabelFeature:
    def __init__(self, names: Iterable[str]) -> None:
        self.names = tuple(names)


class PinnedSplit(list[dict[str, Any]]):
    def __init__(self, rows: Iterable[dict[str, Any]], label_names: tuple[str, ...], fingerprint: str) -> None:
        super().__init__(rows)
        self.features = {
            "image": object(),
            "image_path": object(),
            "label": _LabelFeature(label_names),
            "crop": object(),
            "disease": object(),
            "leaf_id": object(),
        }
        self._fingerprint = fingerprint


class PinnedPlantVillageDataset(dict[str, PinnedSplit]):
    resolved_revision: str
    cache_location: str
    source_file_sha256: dict[str, str]


def _source_identity(filename: str) -> str:
    identity = filename.replace("_final_masked", "")
    if "___" in identity:
        identity = identity.split("___")[-1]
    identity = identity.split("copy")[0]
    for extension in (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"):
        identity = identity.replace(extension, "")
    return identity.strip().lower()


def _leaf_id(filename: str, class_name: str, leaf_map: dict[str, Any]) -> str:
    suggestions = leaf_map.get(_source_identity(filename), [])
    if isinstance(suggestions, str):
        suggestions = [suggestions]
    if len(suggestions) == 1:
        return str(suggestions[0])
    matching = [str(item) for item in suggestions if class_name in str(item)]
    if len(matching) == 1:
        return matching[0]
    # Fail closed later: no filename-derived fallback is accepted as leaf evidence.
    return ""


def _read_split(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _extract_selected(archive_path: Path, relative_paths: set[str], output_root: Path) -> None:
    root = output_root.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        members: dict[str, str] = {}
        for member in archive.namelist():
            normalized = member.replace("\\", "/")
            marker = normalized.find("raw/")
            if marker >= 0:
                members[normalized[marker:]] = member
        missing = sorted(relative_paths.difference(members))
        if missing:
            raise ValueError(f"Pinned PlantVillage archive is missing files: {missing[:5]}")
        for relative in sorted(relative_paths):
            target = (root / relative).resolve()
            if root not in target.parents:
                raise ValueError(f"Unsafe archive path: {relative}")
            if target.is_file():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(members[relative]) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)


def load_pinned_plantvillage(
    *,
    dataset_id: str,
    configuration: str,
    revision: str,
    selected_classes: Iterable[str] | None = None,
    prepare_images: bool = False,
    cache_dir: str | Path | None = None,
    token: str | None = None,
) -> PinnedPlantVillageDataset:
    """Load split metadata and optionally extract selected images at one commit."""
    if not revision:
        raise ValueError("PlantVillage requires an immutable requested revision")
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install the [hf] dependencies before loading PlantVillage") from exc
    cache_root = Path(cache_dir or Path.home() / ".cache" / "plantxai-stability")
    hub_cache = cache_root / "hub"
    info = HfApi(token=token).dataset_info(dataset_id, revision=revision)
    resolved_revision = str(info.sha)

    filenames = {
        "train": f"splits/{configuration}_train.txt",
        "test": f"splits/{configuration}_test.txt",
        "leaf_map": "leaf_grouping/leaf-map.json",
    }
    local_files = {
        name: Path(
            hf_hub_download(
                repo_id=dataset_id,
                repo_type="dataset",
                filename=filename,
                revision=resolved_revision,
                cache_dir=str(hub_cache),
                token=token,
            )
        )
        for name, filename in filenames.items()
    }
    leaf_map = json.loads(local_files["leaf_map"].read_text(encoding="utf-8"))
    source_paths = {
        split: _read_split(local_files[split]) for split in ("train", "test")
    }
    all_classes = tuple(
        sorted(
            {
                parts[2]
                for paths in source_paths.values()
                for relative in paths
                if len(parts := relative.replace("\\", "/").split("/")) >= 4
            }
        )
    )
    selected = set(selected_classes) if selected_classes is not None else set(all_classes)
    unknown = sorted(selected.difference(all_classes))
    if unknown:
        raise ValueError(f"Selected PlantVillage classes are absent: {unknown}")

    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "test": []}
    selected_paths: set[str] = set()
    for split, paths in source_paths.items():
        for source_row_index, relative in enumerate(paths):
            normalized = relative.replace("\\", "/")
            parts = normalized.split("/")
            if len(parts) < 4:
                raise ValueError(f"Invalid PlantVillage split path: {relative}")
            class_name = parts[2]
            if class_name not in selected:
                continue
            filename = parts[-1]
            crop_disease = class_name.split("___", maxsplit=1)
            selected_paths.add(normalized)
            rows_by_split[split].append(
                {
                    "image": {"path": normalized, "bytes": None},
                    "image_path": normalized,
                    "label": class_name,
                    "crop": crop_disease[0],
                    "disease": crop_disease[1] if len(crop_disease) == 2 else "unknown",
                    "leaf_id": _leaf_id(filename, class_name, leaf_map),
                    "_source_row_index": source_row_index,
                }
            )

    if prepare_images:
        archive_path = Path(
            hf_hub_download(
                repo_id=dataset_id,
                repo_type="dataset",
                filename="data.zip",
                revision=resolved_revision,
                cache_dir=str(hub_cache),
                token=token,
            )
        )
        extraction_root = cache_root / "extracted" / resolved_revision / configuration
        _extract_selected(archive_path, selected_paths, extraction_root)
        for rows in rows_by_split.values():
            for row in rows:
                row["image"] = {
                    "path": str(extraction_root / row["image_path"]),
                    "bytes": None,
                }

    source_hashes = {filenames[name]: sha256_file(path) for name, path in local_files.items()}
    dataset = PinnedPlantVillageDataset()
    dataset.resolved_revision = resolved_revision
    dataset.cache_location = str(cache_root)
    dataset.source_file_sha256 = source_hashes
    for split, rows in rows_by_split.items():
        fingerprint_payload = json.dumps(
            {
                "revision": resolved_revision,
                "configuration": configuration,
                "split": split,
                "source_sha256": source_hashes,
                "sample_paths": [row["image_path"] for row in rows],
            },
            sort_keys=True,
        ).encode("utf-8")
        fingerprint = hashlib.sha256(fingerprint_payload).hexdigest()
        dataset[split] = PinnedSplit(rows, all_classes, fingerprint)
    return dataset

