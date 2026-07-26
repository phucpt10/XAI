"""Build a validation-only image bundle from an approved source archive.

This command is for the project-owner recovery environment.  It materializes
only paths named by a frozen validation manifest and verifies canonical RGB
hashes before publishing an immutable output directory.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from plantxai_stability.artifacts import atomic_json
from plantxai_stability.data.manifest import canonical_rgb_sha256, read_manifest_csv
from plantxai_stability.provenance import sha256_file


def _safe_relative_path(member_name: str) -> Path:
    normalized = PurePosixPath(member_name.lstrip("./"))
    if normalized.is_absolute() or ".." in normalized.parts:
        raise ValueError(f"Unsafe archive member path: {member_name}")
    return Path(*normalized.parts)


def _manifest_relative_path(member_name: str, expected: dict[Path, object]) -> Path | None:
    """Map an archive entry to one manifest path, allowing one archive prefix."""
    member = PurePosixPath(member_name.lstrip("./"))
    matches = [
        path
        for path in expected
        if member.as_posix() == path.as_posix()
        or member.as_posix().endswith("/" + path.as_posix())
    ]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous archive-to-manifest path mapping: {member_name}")
    return matches[0] if matches else None


def build_bundle(
    *, archive: Path, validation_manifest: Path, freeze_record: Path, output_dir: Path
) -> dict[str, object]:
    if output_dir.exists():
        raise ValueError("Output directory already exists; choose a new immutable version")
    records = read_manifest_csv(validation_manifest)
    if not records or any(record.split != "validation" for record in records):
        raise ValueError("Manifest must contain one or more validation-only records")
    expected = {Path(record.canonical_relative_path): record for record in records}
    if len(expected) != len(records):
        raise ValueError("Validation manifest has duplicate canonical paths")
    if not archive.is_file() or not freeze_record.is_file():
        raise ValueError("Archive and freeze record must be regular files")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    materialized: set[Path] = set()
    try:
        with tarfile.open(archive, "r:*") as source:
            for member in source:
                if not member.isfile():
                    continue
                _safe_relative_path(member.name)
                relative = _manifest_relative_path(member.name, expected)
                if relative is None:
                    continue
                record = expected[relative]
                destination = staging / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                payload = source.extractfile(member)
                if payload is None:
                    raise ValueError(f"Cannot read validation member: {member.name}")
                with destination.open("xb") as target:
                    shutil.copyfileobj(payload, target)
                digest, width, height = canonical_rgb_sha256(destination)
                if (
                    digest != record.canonical_rgb_sha256
                    or width != record.width
                    or height != record.height
                ):
                    raise ValueError(f"Validation image verification failed: {relative}")
                materialized.add(relative)
        missing = sorted(str(path) for path in set(expected) - materialized)
        if missing:
            raise ValueError(f"Archive is missing validation images: {missing[:5]}")
        shutil.copy2(validation_manifest, staging / "validation_split.csv")
        shutil.copy2(freeze_record, staging / "freeze_record.json")
        report = {
            "run_type": "validation_only_recovery_bundle",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "archive_sha256": sha256_file(archive),
            "validation_manifest_sha256": sha256_file(validation_manifest),
            "freeze_record_sha256": sha256_file(freeze_record),
            "validation_image_count": len(materialized),
            "validation_only_manifest": True,
            "non_validation_entries_materialized": False,
            "all_validation_images_canonical_hash_verified": True,
        }
        atomic_json(staging / "validation_bundle_manifest.json", report)
        os.replace(staging, output_dir)
        return report
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--freeze-record", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build_bundle(
        archive=args.archive,
        validation_manifest=args.validation_manifest,
        freeze_record=args.freeze_record,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
