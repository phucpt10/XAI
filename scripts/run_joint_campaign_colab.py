"""Sequentially launch/resume all registered official joint-evaluation parts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


METHODS = ("grad_cam", "grad_cam_plus_plus", "score_cam")
MODELS = ("resnet50", "efficientnet_b0")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--checkpoint-decision-record", type=Path, required=True)
    parser.add_argument("--test-decision-record", type=Path, required=True)
    parser.add_argument("--g2-readiness-report", type=Path, required=True)
    parser.add_argument("--resnet50-checkpoint", type=Path, required=True)
    parser.add_argument("--efficientnet-b0-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--only-model", choices=MODELS)
    parser.add_argument("--only-method", choices=METHODS)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    runner = repository / "scripts" / "run_joint_eval.py"
    selected_models = [args.only_model] if args.only_model else list(MODELS)
    selected_methods = [args.only_method] if args.only_method else list(METHODS)
    checkpoints = {
        "resnet50": args.resnet50_checkpoint,
        "efficientnet_b0": args.efficientnet_b0_checkpoint,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)

    for model_id in selected_models:
        for method in selected_methods:
            run_id = f"{model_id}-{method}-v1"
            part_dir = args.output_root / run_id
            state_path = part_dir / "run_state.json"
            resume = False
            if state_path.is_file():
                state = json.loads(state_path.read_text(encoding="utf-8"))
                if state.get("status") == "complete":
                    print(f"SKIP complete | {model_id}/{method} | {part_dir}")
                    continue
                resume = True
            command = [
                sys.executable,
                str(runner),
                "--protocol",
                str(args.protocol),
                "--manifest",
                str(args.manifest),
                "--image-root",
                str(args.image_root),
                "--checkpoint",
                str(checkpoints[model_id]),
                "--checkpoint-decision-record",
                str(args.checkpoint_decision_record),
                "--test-decision-record",
                str(args.test_decision_record),
                "--g2-readiness-report",
                str(args.g2_readiness_report),
                "--model-id",
                model_id,
                "--xai-method",
                method,
                "--output-dir",
                str(args.output_root),
                "--run-id",
                run_id,
                "--device",
                args.device,
            ]
            if resume:
                command.append("--resume")
            print(
                f"START {'resume' if resume else 'new'} | "
                f"{model_id}/{method} | {part_dir}"
            )
            completed = subprocess.run(command, cwd=repository, check=False)
            if completed.returncode != 0:
                raise SystemExit(
                    f"Joint campaign stopped at {model_id}/{method}; "
                    f"return code {completed.returncode}. Rerun this same command to resume."
                )
    print("All selected official joint-evaluation parts are complete: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
