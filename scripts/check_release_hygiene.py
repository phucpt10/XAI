"""Fail when private or generated research material is tracked by Git."""

from __future__ import annotations

import subprocess
from pathlib import Path, PurePosixPath


FORBIDDEN_PARTS = {
    ".augment",
    ".codex",
    ".private-data",
    ".private-paper",
    ".private-review",
    "manuscript",
    "manuscripts",
    "paper",
    "papers",
    "reviewer",
    "reviewers",
    "reviews",
}
FORBIDDEN_SUFFIXES = {".ckpt", ".docx", ".pt", ".pth", ".tex"}
FORBIDDEN_NAME_FRAGMENTS = {
    "architecture",
    "reviewer1",
    "reviewer2",
    "software_specification",
    "style_evidence",
}
MAX_TRACKED_FILE_BYTES = 5 * 1024 * 1024


def tracked_paths(repository: Path) -> list[PurePosixPath]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    return [
        PurePosixPath(value.decode("utf-8"))
        for value in result.stdout.split(b"\0")
        if value
    ]


def release_violations(repository: Path) -> list[str]:
    violations: list[str] = []
    for relative in tracked_paths(repository):
        lowered_parts = {part.casefold() for part in relative.parts}
        lowered_name = relative.name.casefold()
        suffix = relative.suffix.casefold()

        if lowered_parts & FORBIDDEN_PARTS:
            violations.append(f"forbidden path: {relative}")
        if suffix in FORBIDDEN_SUFFIXES:
            violations.append(f"forbidden file type: {relative}")
        if any(fragment in lowered_name for fragment in FORBIDDEN_NAME_FRAGMENTS):
            violations.append(f"forbidden filename: {relative}")

        path = repository.joinpath(*relative.parts)
        if path.is_file() and path.stat().st_size > MAX_TRACKED_FILE_BYTES:
            violations.append(
                f"tracked file exceeds {MAX_TRACKED_FILE_BYTES} bytes: {relative}"
            )
    return sorted(set(violations))


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    violations = release_violations(repository)
    if violations:
        print("Release hygiene: FAIL")
        for violation in violations:
            print(f" - {violation}")
        return 1
    print("Release hygiene: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
