"""Fail-closed loading and validation of the versioned protocol."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ResolvedConfig:
    values: dict[str, Any]
    sha256: str

    @property
    def seed(self) -> int:
        return int(self.values["seed"])


def canonical_json(values: dict[str, Any]) -> str:
    return json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def load_protocol(path: str | Path) -> ResolvedConfig:
    protocol_path = Path(path)
    with protocol_path.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    if not isinstance(values, dict):
        raise ValueError("Protocol must be a mapping")
    _validate_protocol(values)
    digest = hashlib.sha256(canonical_json(values).encode("utf-8")).hexdigest()
    return ResolvedConfig(values=values, sha256=digest)


def _validate_protocol(values: dict[str, Any]) -> None:
    required = {"schema_version", "protocol_version", "status", "frozen", "seed", "dataset", "models", "transformations", "xai", "statistics"}
    missing = required.difference(values)
    if missing:
        raise ValueError(f"Protocol missing required keys: {sorted(missing)}")
    if values["status"] not in {"draft", "frozen", "retired"}:
        raise ValueError("status must be draft, frozen or retired")
    if not isinstance(values["seed"], int) or values["seed"] < 0:
        raise ValueError("seed must be a non-negative integer")
    dataset = values["dataset"]
    if dataset.get("group_key") != "leaf_id":
        raise ValueError("dataset.group_key must be leaf_id")
    if not dataset.get("classes"):
        raise ValueError("dataset.classes must not be empty")
    quarantine = dataset.get("quarantine_policy", {})
    if quarantine.get("enabled") is not True:
        raise ValueError("dataset.quarantine_policy.enabled must be true")
    if quarantine.get("official_test_action") != "preserve_exactly":
        raise ValueError("Quarantine policy must preserve the official test exactly")
    if quarantine.get("train_test_leaf_overlap_action") != "quarantine_source_train":
        raise ValueError("Only source-train quarantine is allowed for train/test leaf overlap")
    if values["frozen"] and values.get("governance", {}).get("G0B_PROTOCOL_FREEZE_READY") != "pass":
        raise ValueError("A frozen protocol requires a passing G0B gate")
    stats = values["statistics"]
    if stats.get("bootstrap_unit") != "leaf_id":
        raise ValueError("statistics.bootstrap_unit must be leaf_id")
    if stats.get("correction") != "holm":
        raise ValueError("Only the declared Holm correction is allowed")
