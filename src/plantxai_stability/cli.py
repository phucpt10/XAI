"""Command-line entry points for governance and safe smoke execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plantxai_stability.config import load_protocol
from plantxai_stability.transformations import scenario_grid


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="plantxai")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate-protocol")
    validate.add_argument("protocol", type=Path)
    smoke = sub.add_parser("smoke")
    smoke.add_argument("protocol", type=Path)
    smoke.add_argument("--sample-id", default="pv_smoke0000000001")
    run = sub.add_parser("run")
    run.add_argument("protocol", type=Path)
    run.add_argument("--dataset-manifest", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_protocol(args.protocol)
    if args.command == "validate-protocol":
        print(json.dumps({"protocol_version": config.values["protocol_version"], "config_hash": config.sha256, "status": config.values["status"], "frozen": config.values["frozen"], "official_experiment_allowed": config.values.get("governance", {}).get("official_experiment_allowed", False)}, indent=2))
        return 0
    if args.command == "smoke":
        scenarios = scenario_grid(config.values["transformations"]["parameters"])
        print(json.dumps({"scenario_count": len(scenarios), "scenario_ids": [item.scenario_id for item in scenarios], "seed": config.seed}, indent=2))
        return 0
    if args.command == "run":
        if not config.values.get("frozen", False) or config.values.get("governance", {}).get("G0B_PROTOCOL_FREEZE_READY") != "pass":
            raise SystemExit("Official experiment blocked: protocol is not frozen and G0B is not PASS")
        if not args.dataset_manifest.exists():
            raise SystemExit(f"Dataset manifest does not exist: {args.dataset_manifest}")
        raise SystemExit("Official runner is intentionally not enabled until model checkpoints and dataset evidence are approved")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
