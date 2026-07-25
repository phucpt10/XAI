"""Generate tables, figures and summaries from the frozen official analysis."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from plantxai_stability import __version__
from plantxai_stability.artifacts import atomic_json
from plantxai_stability.provenance import sha256_file
from plantxai_stability.result_reporting import (
    build_frozen_results_summary,
    build_reporting_tables,
    load_and_validate_frozen_results,
    load_results_decision,
    render_reporting_figures,
    render_summary_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-decision-record", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit("Reporting output exists; use a new immutable directory")
    temporary = Path(str(args.output_dir) + ".building")
    if temporary.exists():
        raise SystemExit(f"Stale reporting build directory exists: {temporary}")

    decision = load_results_decision(args.results_decision_record)
    source_report, frames = load_and_validate_frozen_results(
        analysis_dir=args.analysis_dir,
        decision=decision,
    )
    tables = build_reporting_tables(frames)
    summary = build_frozen_results_summary(
        report=source_report,
        tables=tables,
        source_analysis_report_sha256=decision["source_analysis"]["report_sha256"],
    )

    expected_outputs = decision["authorized_reporting_outputs"]
    expected_tables = sorted(expected_outputs["tables"])
    if sorted(tables) != expected_tables:
        raise SystemExit("Reporting table plan differs from the Results Decision Record")
    if sorted(expected_outputs["summaries"]) != sorted(
        [
            "frozen_results_summary.json",
            "frozen_results_summary.md",
            "results_reporting_report.json",
        ]
    ):
        raise SystemExit("Reporting summary plan differs from the Results Decision Record")

    temporary.mkdir(parents=True, exist_ok=False)
    try:
        for name, frame in tables.items():
            _write_csv(temporary / name, frame)
        figure_paths = render_reporting_figures(
            tables=tables,
            output_dir=temporary,
        )
        if sorted(path.name for path in figure_paths) != sorted(expected_outputs["figures"]):
            raise RuntimeError("Reporting figure plan differs from DR-RESULTS-001")
        if any(not path.is_file() or path.stat().st_size == 0 for path in figure_paths):
            raise RuntimeError("A reporting figure is missing or empty")

        summary_json_path = temporary / "frozen_results_summary.json"
        atomic_json(summary_json_path, summary)
        summary_markdown_path = temporary / "frozen_results_summary.md"
        summary_markdown_path.write_text(
            render_summary_markdown(summary),
            encoding="utf-8",
            newline="\n",
        )

        child_paths = sorted(
            path
            for path in temporary.iterdir()
            if path.is_file() and path.name != "results_reporting_report.json"
        )
        expected_child_names = sorted(
            [
                *expected_outputs["tables"],
                *expected_outputs["figures"],
                "frozen_results_summary.json",
                "frozen_results_summary.md",
            ]
        )
        if [path.name for path in child_paths] != expected_child_names:
            raise RuntimeError("Generated reporting artifact coverage mismatch")

        criteria = {
            "results_decision_approved": True,
            "frozen_analysis_report_sha256_matches": True,
            "all_frozen_child_artifact_hashes_match": True,
            "frozen_analysis_acceptance_criteria_pass": True,
            "frozen_row_counts_match": True,
            "paired_rows_reconcile_576": (len(tables["table_paired_comparisons.csv"]) == 576),
            "estimable_rows_reconcile": (
                int(tables["table_paired_comparisons.csv"]["estimable"].sum())
                == source_report["row_counts"]["paired_estimable_rows"]
            ),
            "non_estimable_rows_reconcile": (
                len(tables["table_non_estimable_comparisons.csv"])
                == source_report["row_counts"]["paired_non_estimable_rows"]
            ),
            "authorized_table_coverage_exact": True,
            "authorized_figure_coverage_exact": True,
            "official_test_pixels_not_accessed": True,
            "predictions_and_cams_not_recomputed": True,
            "scientific_plan_not_changed": True,
            "results_not_used_for_selection_or_tuning": True,
        }
        if not all(criteria.values()):
            failed = sorted(key for key, value in criteria.items() if not value)
            raise RuntimeError(f"Frozen results reporting gate failed: {failed}")

        report_path = temporary / "results_reporting_report.json"
        report = {
            "run_type": "frozen_official_results_reporting",
            "results_decision_id": decision["decision_id"],
            "results_decision_record_sha256": sha256_file(args.results_decision_record),
            "source_analysis_report_sha256": decision["source_analysis"]["report_sha256"],
            "source_analysis_run_type": source_report["run_type"],
            "source_analysis_git_commit": source_report["runtime"]["git_commit"],
            "official_test_pixels_accessed": False,
            "predictions_or_cams_recomputed": False,
            "scientific_configuration_changed": False,
            "row_counts": source_report["row_counts"],
            "generated_artifact_sha256": {path.name: sha256_file(path) for path in child_paths},
            "acceptance_criteria": criteria,
            "interpretation_constraints": summary["interpretation_constraints"],
            "runtime": {
                "python_platform": platform.platform(),
                "software_version": __version__,
                "git_commit": _git_revision(),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "device": "cpu",
                "matplotlib_version": _matplotlib_version(),
            },
        }
        atomic_json(report_path, report)
        temporary.rename(args.output_dir)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    final_report_path = args.output_dir / "results_reporting_report.json"
    final_report = json.loads(final_report_path.read_text(encoding="utf-8"))
    print(json.dumps(final_report, indent=2, sort_keys=True))
    print(f"Frozen results reporting: PASS\nReport: {final_report_path}")
    print(f"Report SHA-256: {sha256_file(final_report_path)}")
    return 0


def _write_csv(path: Path, frame: Any) -> None:
    """Write an authorized table, preserving headers even when it has no rows."""
    frame.to_csv(path, index=False, lineterminator="\n")


def _git_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _matplotlib_version() -> str:
    import matplotlib

    return str(matplotlib.__version__)


if __name__ == "__main__":
    raise SystemExit(main())
