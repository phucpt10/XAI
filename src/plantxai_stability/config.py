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
    required = {
        "schema_version",
        "protocol_version",
        "status",
        "frozen",
        "seed",
        "governance",
        "dataset",
        "models",
        "training",
        "transformations",
        "xai",
        "statistics",
    }
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
    governance = values["governance"]
    evidence_records = governance.get("evidence_records", {})
    if not evidence_records.get("runtime_readiness"):
        raise ValueError("governance.evidence_records.runtime_readiness is required")
    if not evidence_records.get("xai_target_layers"):
        raise ValueError("governance.evidence_records.xai_target_layers is required")
    if values["frozen"] and governance.get("G0B_PROTOCOL_FREEZE_READY") != "pass":
        raise ValueError("A frozen protocol requires a passing G0B gate")
    if governance.get("official_training_allowed") and (
        not values["frozen"] or governance.get("G0B_PROTOCOL_FREEZE_READY") != "pass"
    ):
        raise ValueError("Official training requires a frozen protocol and G0B PASS")
    if governance.get("official_test_evaluation_allowed") and (
        not governance.get("official_training_allowed")
        or governance.get("G1_CHECKPOINT_SELECTION") != "pass"
        or governance.get("G2_TEST_EVALUATION_READY") != "pass"
    ):
        raise ValueError("Official test evaluation requires training, G1 and G2 approval")
    xai = values["xai"]
    if not xai.get("target_layer_decision_record"):
        raise ValueError("xai.target_layer_decision_record is required")
    target_layers = xai.get("target_layers", {})
    missing_target_layers = set(values["models"]).difference(target_layers)
    if missing_target_layers:
        raise ValueError(f"XAI target layers missing for models: {sorted(missing_target_layers)}")
    if any(
        not isinstance(target_layers[model_id], str)
        or not target_layers[model_id].strip()
        or target_layers[model_id].startswith("PENDING_")
        for model_id in values["models"]
    ):
        raise ValueError("Every model requires a runtime-approved XAI target layer")
    transformations = values["transformations"]
    if transformations.get("algorithm_version") != "shared_randomization_v2":
        raise ValueError("Unsupported transformation algorithm version")
    training = values["training"]
    if training.get("optimizer") != "adamw":
        raise ValueError("Only the declared AdamW optimizer is allowed")
    if training.get("pretrained_weights") != {
        "resnet50": "IMAGENET1K_V2",
        "efficientnet_b0": "IMAGENET1K_V1",
    }:
        raise ValueError("Training pretrained weight identities are not approved")
    if training.get("fine_tuning") != "full_model":
        raise ValueError("Only full-model fine-tuning is approved")
    if training.get("loss") != "cross_entropy" or training.get("class_weighting") != "none":
        raise ValueError("Only unweighted cross-entropy is approved")
    if training.get("scheduler") != "cosine":
        raise ValueError("Only the declared cosine scheduler is allowed")
    if training.get("selection_metric") != "validation_macro_f1":
        raise ValueError("Checkpoint selection must use validation macro-F1")
    if training.get("deterministic_algorithms") is not True:
        raise ValueError("training.deterministic_algorithms must be true")
    stats = values["statistics"]
    if stats.get("bootstrap_unit") != "leaf_id":
        raise ValueError("statistics.bootstrap_unit must be leaf_id")
    if stats.get("correction") != "holm":
        raise ValueError("Only the declared Holm correction is allowed")
