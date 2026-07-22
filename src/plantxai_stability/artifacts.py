"""Atomic run output and provenance artifact helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from plantxai_stability.provenance import RunContext, sha256_file


def run_root(output_root: str | Path, run_id: str) -> Path:
    path = Path(output_root) / "runs" / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, target)


def write_run_manifest(root: str | Path, context: RunContext, extra: dict[str, Any] | None = None) -> Path:
    output = Path(root) / "run_manifest.json"
    payload = context.to_dict()
    if extra:
        payload.update(extra)
    atomic_json(output, payload)
    return output


def index_artifacts(root: str | Path) -> list[dict[str, str]]:
    base = Path(root)
    artifacts: list[dict[str, str]] = []
    for path in sorted(base.rglob("*")):
        if path.is_file() and path.name not in {"artifact_index.json"}:
            artifacts.append({"path": str(path.relative_to(base)), "sha256": sha256_file(path)})
    atomic_json(base / "artifact_index.json", {"artifacts": artifacts})
    return artifacts
