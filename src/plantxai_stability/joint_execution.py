"""Transactional progress and coverage contracts for official joint evaluation."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from plantxai_stability.artifacts import atomic_json
from plantxai_stability.provenance import sha256_bytes


JOINT_EXECUTION_SCHEMA_VERSION = "joint_execution_v2"


def canonical_json_sha256(value: Any) -> str:
    """Hash JSON with a stable encoding suitable for run-identity binding."""
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return sha256_bytes(payload)


def build_run_identity(
    *,
    run_id: str,
    model_id: str,
    xai_method: str,
    scenario_ids: Sequence[str],
    sample_ids: Sequence[str],
    seed: int,
    governance_protocol_hash: str,
    checkpoint_training_protocol_hash: str,
    checkpoint_sha256: str,
    manifest_sha256: str,
    freeze_record_sha256: str,
    checkpoint_decision_record_sha256: str,
    test_decision_record_sha256: str,
    g2_readiness_report_sha256: str,
    campaign_id: str,
    authorization_decision_id: str,
    transformation_algorithm_version: str,
    xai_policy: dict[str, Any],
    software_version: str,
    git_commit: str,
    runtime_identity: dict[str, Any],
) -> dict[str, Any]:
    ordered_samples = list(sample_ids)
    if ordered_samples != sorted(ordered_samples):
        raise ValueError("Official-test sample identities must be sorted")
    if len(ordered_samples) != len(set(ordered_samples)):
        raise ValueError("Official-test sample identities must be unique")
    ordered_scenarios = list(scenario_ids)
    if not ordered_scenarios or len(ordered_scenarios) != len(set(ordered_scenarios)):
        raise ValueError("Scenario identities must be non-empty and unique")
    if not git_commit:
        raise ValueError("Official joint execution requires a Git commit identity")
    identity: dict[str, Any] = {
        "schema_version": JOINT_EXECUTION_SCHEMA_VERSION,
        "run_id": run_id,
        "model_id": model_id,
        "xai_method": xai_method,
        "campaign_id": campaign_id,
        "authorization_decision_id": authorization_decision_id,
        "seed": int(seed),
        "scenario_ids": ordered_scenarios,
        "scenario_count": len(ordered_scenarios),
        "sample_count": len(ordered_samples),
        "sample_ids_sha256": canonical_json_sha256(ordered_samples),
        "governance_protocol_hash": governance_protocol_hash,
        "checkpoint_training_protocol_hash": checkpoint_training_protocol_hash,
        "checkpoint_sha256": checkpoint_sha256,
        "manifest_sha256": manifest_sha256,
        "freeze_record_sha256": freeze_record_sha256,
        "checkpoint_decision_record_sha256": checkpoint_decision_record_sha256,
        "test_decision_record_sha256": test_decision_record_sha256,
        "g2_readiness_report_sha256": g2_readiness_report_sha256,
        "transformation_algorithm_version": transformation_algorithm_version,
        "xai_policy": xai_policy,
        "software_version": software_version,
        "git_commit": git_commit,
        "runtime_identity": runtime_identity,
    }
    identity["run_identity_sha256"] = canonical_json_sha256(identity)
    return identity


class JointProgressStore:
    """SQLite-backed one-transaction-per-sample progress ledger."""

    def __init__(self, path: str | Path, identity: dict[str, Any]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sample_results (
                sample_id TEXT PRIMARY KEY,
                leaf_id TEXT NOT NULL,
                prediction_rows_json TEXT NOT NULL,
                joint_rows_json TEXT NOT NULL,
                completed_at_utc TEXT NOT NULL
            )
            """
        )
        serialized = _canonical_json(identity)
        existing = self.connection.execute(
            "SELECT value FROM metadata WHERE key = 'run_identity'"
        ).fetchone()
        if existing is None:
            with self.connection:
                self.connection.execute(
                    "INSERT INTO metadata(key, value) VALUES('run_identity', ?)",
                    (serialized,),
                )
        elif existing[0] != serialized:
            self.connection.close()
            raise ValueError("Resume blocked: run identity does not match progress database")

    def close(self) -> None:
        self.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.connection.close()

    def __enter__(self) -> "JointProgressStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def completed_sample_ids(self) -> set[str]:
        rows = self.connection.execute("SELECT sample_id FROM sample_results")
        return {str(row[0]) for row in rows}

    def completed_count(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) FROM sample_results").fetchone()
        return int(row[0]) if row is not None else 0

    def write_sample(
        self,
        *,
        sample_id: str,
        leaf_id: str,
        prediction_rows: Sequence[dict[str, Any]],
        joint_rows: Sequence[dict[str, Any]],
        expected_scenario_ids: Sequence[str],
        expected_xai_method: str,
    ) -> None:
        _validate_sample_rows(
            sample_id,
            prediction_rows,
            joint_rows,
            expected_scenario_ids,
            expected_xai_method,
        )
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO sample_results(
                    sample_id, leaf_id, prediction_rows_json, joint_rows_json,
                    completed_at_utc
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    sample_id,
                    leaf_id,
                    _canonical_json(list(prediction_rows)),
                    _canonical_json(list(joint_rows)),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def iter_rows(self, column: str) -> Iterator[dict[str, Any]]:
        if column not in {"prediction_rows_json", "joint_rows_json"}:
            raise ValueError(f"Unsupported result column: {column}")
        query = f"SELECT {column} FROM sample_results ORDER BY sample_id"  # noqa: S608
        for (payload,) in self.connection.execute(query):
            rows = json.loads(payload)
            if not isinstance(rows, list):
                raise ValueError("Progress database contains a non-list result payload")
            yield from rows


def write_run_state(
    path: str | Path,
    *,
    identity: dict[str, Any],
    status: str,
    completed_sample_count: int,
    retry_count: int,
    extra: dict[str, Any] | None = None,
) -> None:
    if status not in {"in_progress", "complete"}:
        raise ValueError(f"Unsupported joint run status: {status}")
    payload: dict[str, Any] = {
        "status": status,
        "run_identity": identity,
        "completed_sample_count": int(completed_sample_count),
        "expected_sample_count": int(identity["sample_count"]),
        "retry_count": int(retry_count),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        payload.update(extra)
    atomic_json(path, payload)


def validate_completed_coverage(
    *,
    completed_sample_ids: Iterable[str],
    expected_sample_ids: Sequence[str],
    prediction_rows: Sequence[dict[str, Any]],
    joint_rows: Sequence[dict[str, Any]],
    scenario_ids: Sequence[str],
    xai_method: str,
) -> dict[str, bool]:
    expected_samples = list(expected_sample_ids)
    completed = sorted(completed_sample_ids)
    expected_prediction_keys = {
        (sample_id, scenario_id)
        for sample_id in expected_samples
        for scenario_id in scenario_ids
    }
    prediction_keys = {
        (str(row["sample_id"]), str(row["scenario_id"])) for row in prediction_rows
    }
    joint_keys = {
        (str(row["sample_id"]), str(row["scenario_id"]), str(row["xai_method"]))
        for row in joint_rows
    }
    expected_joint_keys = {
        (sample_id, scenario_id, xai_method)
        for sample_id in expected_samples
        for scenario_id in scenario_ids
    }
    criteria = {
        "completed_sample_identity_exact": completed == expected_samples,
        "prediction_factorial_coverage_exact": (
            len(prediction_rows) == len(expected_prediction_keys)
            and prediction_keys == expected_prediction_keys
        ),
        "joint_factorial_coverage_exact": (
            len(joint_rows) == len(expected_joint_keys) and joint_keys == expected_joint_keys
        ),
        "one_declared_xai_method": all(
            str(row["xai_method"]) == xai_method for row in joint_rows
        ),
    }
    if not all(criteria.values()):
        failed = sorted(key for key, value in criteria.items() if not value)
        raise ValueError(f"Joint execution coverage failed: {failed}")
    return criteria


def _validate_sample_rows(
    sample_id: str,
    prediction_rows: Sequence[dict[str, Any]],
    joint_rows: Sequence[dict[str, Any]],
    scenario_ids: Sequence[str],
    xai_method: str,
) -> None:
    expected = set(scenario_ids)
    prediction_scenarios = [str(row.get("scenario_id")) for row in prediction_rows]
    joint_scenarios = [str(row.get("scenario_id")) for row in joint_rows]
    if len(prediction_scenarios) != len(expected) or set(prediction_scenarios) != expected:
        raise ValueError(f"Prediction coverage mismatch for {sample_id}")
    if len(joint_scenarios) != len(expected) or set(joint_scenarios) != expected:
        raise ValueError(f"Joint coverage mismatch for {sample_id}")
    if any(str(row.get("sample_id")) != sample_id for row in prediction_rows):
        raise ValueError(f"Prediction sample identity mismatch for {sample_id}")
    if any(str(row.get("sample_id")) != sample_id for row in joint_rows):
        raise ValueError(f"Joint sample identity mismatch for {sample_id}")
    if any(str(row.get("xai_method")) != xai_method for row in joint_rows):
        raise ValueError(f"Joint XAI method mismatch for {sample_id}")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
