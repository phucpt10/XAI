"""Run identity, hashes and artifact lineage."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class RunContext:
    run_id: str
    protocol_version: str
    protocol_hash: str
    resolved_config_hash: str
    seed: int
    python_version: str
    platform: str
    created_at_utc: str

    @classmethod
    def create(cls, protocol_version: str, protocol_hash: str, config_hash: str, seed: int, run_id: str | None = None) -> "RunContext":
        now = datetime.now(timezone.utc).isoformat()
        stable_id = run_id or sha256_bytes(f"{protocol_hash}:{config_hash}:{now}".encode())[:24]
        return cls(stable_id, protocol_version, protocol_hash, config_hash, seed, sys.version, platform.platform(), now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
