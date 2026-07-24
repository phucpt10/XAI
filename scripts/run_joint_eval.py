"""Run one resumable model-method part of the authorized joint campaign."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from plantxai_stability import __version__
from plantxai_stability.artifacts import atomic_json
from plantxai_stability.config import load_protocol, resolve_xai_target_layer
from plantxai_stability.data.loader import load_verified_record, preprocess_for_model
from plantxai_stability.inference import infer_one
from plantxai_stability.joint_execution import (
    JointProgressStore,
    build_run_identity,
    canonical_json_sha256,
    validate_completed_coverage,
    write_run_state,
)
from plantxai_stability.models import ModelWrapper
from plantxai_stability.provenance import sha256_bytes, sha256_file
from plantxai_stability.recovery import authorize_recovery_joint_part
from plantxai_stability.statistics import heatmap_metrics
from plantxai_stability.test_authorization import authorize_official_test_run
from plantxai_stability.training import load_checkpoint, seed_everything
from plantxai_stability.transformations import (
    TRANSFORMATION_ALGORITHM_VERSION,
    TransformationPipeline,
    scenario_grid,
)
from plantxai_stability.xai import CAMGenerator, forward_align_heatmap


METRIC_FIELDS = (
    "ssim",
    "pearson",
    "cosine",
    "topk_iou_10",
    "topk_iou_20",
    "topk_iou_30",
    "valid_pixel_count",
    "valid_pixel_fraction",
    "ssim_valid_pixel_count",
    "ssim_valid_pixel_fraction",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-decision-record", type=Path, required=True)
    parser.add_argument("--test-decision-record", type=Path, required=True)
    parser.add_argument("--g2-readiness-report", type=Path, required=True)
    parser.add_argument("--recovery-decision-record", type=Path)
    parser.add_argument("--recovery-binding-report", type=Path)
    parser.add_argument(
        "--model-id", choices=["resnet50", "efficientnet_b0"], required=True
    )
    parser.add_argument(
        "--xai-method",
        choices=["grad_cam", "grad_cam_plus_plus", "score_cam"],
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.progress_every < 1:
        raise SystemExit("--progress-every must be at least 1")
    governed_recovery = (
        args.protocol.parent / "decision_records" / "DR-RECOVERY-001.yaml"
    )
    if governed_recovery.is_file() and (
        args.recovery_decision_record is None
        or args.recovery_binding_report is None
    ):
        raise SystemExit(
            "DR-RECOVERY-001 is active; both recovery evidence arguments are required"
        )

    resolved = load_protocol(args.protocol)
    declared_methods = list(resolved.values["xai"]["methods"])
    if args.xai_method not in declared_methods:
        raise SystemExit("Selected XAI method is not registered in the frozen campaign")
    authorization = authorize_official_test_run(
        resolved,
        manifest_path=args.manifest,
        checkpoint_path=args.checkpoint,
        model_id=args.model_id,
        checkpoint_decision_path=args.checkpoint_decision_record,
        test_decision_path=args.test_decision_record,
        readiness_report_path=args.g2_readiness_report,
        recovery_decision_path=args.recovery_decision_record,
        recovery_binding_report_path=args.recovery_binding_report,
    )
    if authorization["recovery_lineage"] is not None:
        if args.recovery_decision_record is None:
            raise ValueError("Recovery lineage is missing its Decision Record")
        authorize_recovery_joint_part(
            model_id=args.model_id,
            xai_method=args.xai_method,
            recovery_decision_path=args.recovery_decision_record,
        )
    records = sorted(authorization["test_records"], key=lambda item: item.sample_id)
    scenarios = scenario_grid(resolved.values["transformations"]["parameters"])
    scenario_ids = [scenario.scenario_id for scenario in scenarios]
    sample_ids = [record.sample_id for record in records]
    git_commit = _git_revision()
    runtime_identity = _runtime_identity(args.device)
    xai_policy = resolved.values["xai"]
    bound_xai_policy = {
        key: xai_policy[key]
        for key in (
            "alignment_policy",
            "valid_region_policy",
            "validity_threshold",
            "ssim_window_size",
            "topk_iou_primary",
            "topk_iou_sensitivity",
            "target_class_policy",
            "prediction_consistency_required",
            "rotation_prediction_claim_scope",
        )
    }
    bound_xai_policy["target_layer"] = resolve_xai_target_layer(
        xai_policy, args.model_id, args.xai_method
    )
    identity = build_run_identity(
        run_id=args.run_id,
        model_id=args.model_id,
        xai_method=args.xai_method,
        scenario_ids=scenario_ids,
        sample_ids=sample_ids,
        seed=resolved.seed,
        governance_protocol_hash=resolved.sha256,
        checkpoint_training_protocol_hash=authorization[
            "checkpoint_training_protocol_hash"
        ],
        checkpoint_sha256=authorization["checkpoint_sha256"],
        manifest_sha256=authorization["manifest_sha256"],
        freeze_record_sha256=authorization["freeze_record_sha256"],
        checkpoint_decision_record_sha256=authorization[
            "checkpoint_decision_record_sha256"
        ],
        test_decision_record_sha256=authorization["test_decision_record_sha256"],
        g2_readiness_report_sha256=authorization["g2_readiness_report_sha256"],
        campaign_id=authorization["campaign_id"],
        authorization_decision_id=authorization["authorization_decision_id"],
        transformation_algorithm_version=TRANSFORMATION_ALGORITHM_VERSION,
        xai_policy=bound_xai_policy,
        software_version=__version__,
        git_commit=git_commit,
        runtime_identity=runtime_identity,
        recovery_lineage=authorization["recovery_lineage"],
    )

    output = args.output_dir / args.run_id
    state_path = output / "run_state.json"
    database_path = output / "joint_progress.sqlite3"
    retry_count = _prepare_output(output, state_path, identity, args.resume)
    seed_everything(
        resolved.seed,
        bool(resolved.values["training"]["deterministic_algorithms"]),
    )

    wrapper = ModelWrapper(
        args.model_id, len(resolved.values["dataset"]["classes"]), pretrained=False
    )
    checkpoint_payload = load_checkpoint(
        wrapper,
        args.checkpoint,
        args.device,
        expected_protocol_hash=authorization["checkpoint_training_protocol_hash"],
        expected_manifest_sha256=authorization["manifest_sha256"],
    )
    model = wrapper.model.to(args.device)
    model.eval()
    target_layer = wrapper.target_layer(bound_xai_policy["target_layer"])
    pipeline = TransformationPipeline(
        resolved.seed, resolved.values["transformations"]["parameters"]
    )
    generator = CAMGenerator(model, target_layer, args.xai_method)

    store = JointProgressStore(database_path, identity)
    try:
        generator.__enter__()
        completed = store.completed_sample_ids()
        unexpected = completed.difference(sample_ids)
        if unexpected:
            raise ValueError(
                f"Resume blocked: progress contains unexpected samples {sorted(unexpected)[:3]}"
            )
        write_run_state(
            state_path,
            identity=identity,
            status="in_progress",
            completed_sample_count=len(completed),
            retry_count=retry_count,
        )
        pending = [record for record in records if record.sample_id not in completed]
        print(
            f"Joint part {args.model_id}/{args.xai_method}: "
            f"{len(completed)}/{len(records)} samples already complete; "
            f"{len(pending)} pending"
        )
        for pending_index, record in enumerate(pending, start=1):
            prediction_rows, joint_rows = _evaluate_sample(
                record=record,
                image_root=args.image_root,
                model=model,
                target_layer=target_layer,
                generator=generator,
                pipeline=pipeline,
                scenarios=scenarios,
                model_id=args.model_id,
                xai_method=args.xai_method,
                run_id=args.run_id,
                checkpoint_sha256=authorization["checkpoint_sha256"],
                device=args.device,
                xai_policy=xai_policy,
            )
            store.write_sample(
                sample_id=record.sample_id,
                leaf_id=record.leaf_id,
                prediction_rows=prediction_rows,
                joint_rows=joint_rows,
                expected_scenario_ids=scenario_ids,
                expected_xai_method=args.xai_method,
            )
            current = len(completed) + pending_index
            if current % args.progress_every == 0 or current == len(records):
                write_run_state(
                    state_path,
                    identity=identity,
                    status="in_progress",
                    completed_sample_count=current,
                    retry_count=retry_count,
                )
                print(
                    f"  {args.model_id}/{args.xai_method}: "
                    f"{current}/{len(records)} samples committed"
                )
        prediction_rows = list(store.iter_rows("prediction_rows_json"))
        joint_rows = list(store.iter_rows("joint_rows_json"))
        completed = store.completed_sample_ids()
        criteria = validate_completed_coverage(
            completed_sample_ids=completed,
            expected_sample_ids=sample_ids,
            prediction_rows=prediction_rows,
            joint_rows=joint_rows,
            scenario_ids=scenario_ids,
            xai_method=args.xai_method,
        )
    except KeyboardInterrupt:
        completed_count = store.completed_count()
        write_run_state(
            state_path,
            identity=identity,
            status="in_progress",
            completed_sample_count=completed_count,
            retry_count=retry_count,
            extra={"last_exit": "keyboard_interrupt"},
        )
        print(
            f"Interrupted safely after {completed_count}/{len(records)} committed samples. "
            "Run the identical command with --resume."
        )
        return 130
    finally:
        generator.close()
        store.close()

    prediction_rows.sort(key=lambda row: (row["sample_id"], row["scenario_id"]))
    joint_rows.sort(
        key=lambda row: (row["sample_id"], row["scenario_id"], row["xai_method"])
    )
    predictions_path = output / "prediction_results.csv"
    joint_path = output / "joint_results.csv"
    manifest_path = output / "run_manifest.json"
    report_path = output / "joint_run_report.json"
    _write_csv_atomic(predictions_path, prediction_rows)
    _write_csv_atomic(joint_path, joint_rows)
    checkpoint_metadata = {
        key: checkpoint_payload.get(key)
        for key in (
            "format_version",
            "checkpoint_role",
            "model_id",
            "num_classes",
            "class_names",
            "validation_macro_f1",
            "best_epoch",
            "seed",
            "protocol_hash",
            "manifest_sha256",
            "test_split_accessed",
        )
    }
    run_manifest = {
        "run_type": "authorized_official_test_joint_part",
        "official_test_result": True,
        "run_identity": identity,
        "official_test_identity": authorization["test_identity"],
        "checkpoint_metadata": checkpoint_metadata,
        "runtime": {
            "python_platform": platform.platform(),
            "software_version": __version__,
            "git_commit": git_commit,
            **runtime_identity,
        },
    }
    atomic_json(manifest_path, run_manifest)
    artifacts = {
        path.name: sha256_file(path)
        for path in (predictions_path, joint_path, manifest_path, database_path)
    }
    successful_joint_count = sum(not row["exclusion_reason"] for row in joint_rows)
    report = {
        **run_manifest,
        "prediction_row_count": len(prediction_rows),
        "joint_row_count": len(joint_rows),
        "successful_joint_metric_count": successful_joint_count,
        "excluded_joint_metric_count": len(joint_rows) - successful_joint_count,
        "prediction_rows_sha256": canonical_json_sha256(prediction_rows),
        "joint_rows_sha256": canonical_json_sha256(joint_rows),
        "retry_count": retry_count,
        "artifact_sha256": artifacts,
        "acceptance_criteria": {
            "g2_authorization_passed_before_pixel_access": True,
            "recovery_binding_requirement_satisfied": True,
            "checkpoint_and_protocol_lineage_match": True,
            "transactional_sample_commits": True,
            "resume_identity_bound": True,
            **criteria,
        },
    }
    atomic_json(report_path, report)
    write_run_state(
        state_path,
        identity=identity,
        status="complete",
        completed_sample_count=len(records),
        retry_count=retry_count,
        extra={
            "joint_run_report_sha256": sha256_file(report_path),
            "artifact_sha256": artifacts,
        },
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Official joint part: PASS\nReport: {report_path}")
    print(f"Report SHA-256: {sha256_file(report_path)}")
    return 0


def _evaluate_sample(
    *,
    record: Any,
    image_root: Path,
    model: Any,
    target_layer: Any,
    generator: CAMGenerator,
    pipeline: TransformationPipeline,
    scenarios: Sequence[Any],
    model_id: str,
    xai_method: str,
    run_id: str,
    checkpoint_sha256: str,
    device: str,
    xai_policy: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    del target_layer  # Bound inside the method-specific generator.
    original_pixels = load_verified_record(record, image_root)
    original_prediction = infer_one(
        model,
        original_pixels,
        record,
        model_id,
        run_id,
        "original",
        checkpoint_sha256,
        device,
    )
    original_tensor = preprocess_for_model(original_pixels).unsqueeze(0).to(device)
    original_cam: np.ndarray | None = None
    original_cam_error = ""
    try:
        original_cam = generator.generate(
            original_tensor, original_prediction.predicted_class
        )
    except ValueError as exc:
        original_cam_error = f"invalid_original_cam:{exc}"
    original_cam_hash = "" if original_cam is None else _heatmap_sha256(original_cam)

    prediction_rows: list[dict[str, Any]] = []
    joint_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        transformed_pixels, transformation_record = pipeline.apply(
            original_pixels, record.sample_id, scenario
        )
        transformed_prediction = infer_one(
            model,
            transformed_pixels,
            record,
            model_id,
            run_id,
            scenario.scenario_id,
            checkpoint_sha256,
            device,
        )
        consistent = (
            original_prediction.predicted_class
            == transformed_prediction.predicted_class
        )
        transformation_sha256 = canonical_json_sha256(asdict(transformation_record))
        prediction_rows.append(
            {
                **asdict(original_prediction),
                "scenario_id": scenario.scenario_id,
                "original_scenario_id": original_prediction.scenario_id,
                "true_class_id": record.class_id,
                "true_class_name": record.class_name,
                "leaf_id": record.leaf_id,
                "transformed_predicted_class": transformed_prediction.predicted_class,
                "transformed_confidence": transformed_prediction.confidence,
                "transformed_is_correct": transformed_prediction.is_correct,
                "is_consistent": consistent,
                "confidence_delta": (
                    transformed_prediction.confidence - original_prediction.confidence
                ),
                "absolute_confidence_delta": abs(
                    transformed_prediction.confidence - original_prediction.confidence
                ),
                "transformation": scenario.transformation,
                "severity": scenario.severity,
                "rotation_prediction_claim_scope": (
                    xai_policy["rotation_prediction_claim_scope"]
                    if scenario.transformation == "rotation"
                    else "not_applicable"
                ),
                "valid_mask_sha256": transformation_record.valid_mask_sha256 or "",
                "transformation_record_sha256": transformation_sha256,
            }
        )
        base_hashes = {
            "original_cam_sha256": original_cam_hash,
            "original_cam_status": "valid" if original_cam is not None else original_cam_error,
            "transformed_cam_sha256": "",
            "transformed_cam_status": "not_evaluated",
            "aligned_original_cam_sha256": "",
            "transformation_record_sha256": transformation_sha256,
        }
        if not consistent:
            joint_rows.append(
                _joint_row(
                    run_id,
                    model_id,
                    record.sample_id,
                    record.leaf_id,
                    scenario.scenario_id,
                    xai_method,
                    original_prediction.predicted_class,
                    False,
                    None,
                    transformation_record.valid_mask_sha256 or "",
                    "prediction_inconsistent",
                    base_hashes,
                )
            )
            continue
        if original_cam is None:
            joint_rows.append(
                _joint_row(
                    run_id,
                    model_id,
                    record.sample_id,
                    record.leaf_id,
                    scenario.scenario_id,
                    xai_method,
                    original_prediction.predicted_class,
                    True,
                    None,
                    transformation_record.valid_mask_sha256 or "",
                    original_cam_error,
                    base_hashes,
                )
            )
            continue
        try:
            transformed_tensor = (
                preprocess_for_model(transformed_pixels).unsqueeze(0).to(device)
            )
            transformed_cam = generator.generate(
                transformed_tensor, original_prediction.predicted_class
            )
            aligned_original, mask = forward_align_heatmap(
                original_cam, transformation_record.forward_metadata
            )
            metrics = heatmap_metrics(
                aligned_original,
                transformed_cam,
                mask,
                ssim_window_size=int(xai_policy["ssim_window_size"]),
                topk_values=tuple(xai_policy["topk_iou_sensitivity"]),
            )
            mask_hash = sha256_bytes(mask.astype("uint8").tobytes())
            hashes = {
                **base_hashes,
                "transformed_cam_sha256": _heatmap_sha256(transformed_cam),
                "transformed_cam_status": "valid",
                "aligned_original_cam_sha256": _heatmap_sha256(aligned_original),
            }
        except ValueError as exc:
            joint_rows.append(
                _joint_row(
                    run_id,
                    model_id,
                    record.sample_id,
                    record.leaf_id,
                    scenario.scenario_id,
                    xai_method,
                    original_prediction.predicted_class,
                    True,
                    None,
                    transformation_record.valid_mask_sha256 or "",
                    f"invalid_transformed_cam_or_metric:{exc}",
                    base_hashes,
                )
            )
            continue
        joint_rows.append(
            _joint_row(
                run_id,
                model_id,
                record.sample_id,
                record.leaf_id,
                scenario.scenario_id,
                xai_method,
                original_prediction.predicted_class,
                True,
                metrics,
                mask_hash,
                "",
                hashes,
            )
        )
    return prediction_rows, joint_rows


def _prepare_output(
    output: Path, state_path: Path, identity: dict[str, Any], resume: bool
) -> int:
    if resume:
        if not output.is_dir() or not state_path.is_file():
            raise SystemExit("Resume requested but the joint run directory is incomplete")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("status") == "complete":
            raise SystemExit("Joint run is already complete and immutable")
        if state.get("run_identity") != identity:
            raise SystemExit("Resume blocked: run_state identity mismatch")
        return int(state.get("retry_count", 0)) + 1
    if output.exists():
        raise SystemExit("Joint output exists; use --resume or a new immutable run ID")
    output.mkdir(parents=True, exist_ok=False)
    write_run_state(
        state_path,
        identity=identity,
        status="in_progress",
        completed_sample_count=0,
        retry_count=0,
    )
    return 0


def _write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _joint_row(
    run_id: str,
    model_id: str,
    sample_id: str,
    leaf_id: str,
    scenario_id: str,
    method: str,
    target_class: int,
    is_consistent: bool,
    metrics: dict[str, float] | None,
    valid_mask_sha256: str,
    exclusion_reason: str,
    hashes: dict[str, str],
) -> dict[str, Any]:
    values: dict[str, Any] = {
        "run_id": run_id,
        "model_id": model_id,
        "sample_id": sample_id,
        "leaf_id": leaf_id,
        "scenario_id": scenario_id,
        "xai_method": method,
        "target_class": target_class,
        "is_consistent": is_consistent,
        "alignment_policy": "forward_align_original_cam",
        "valid_mask_sha256": valid_mask_sha256,
        "exclusion_reason": exclusion_reason,
        **hashes,
    }
    values.update(
        {field: "" if metrics is None else metrics[field] for field in METRIC_FIELDS}
    )
    return values


def _heatmap_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype=np.float32)
    return sha256_bytes(array.tobytes())


def _git_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    revision = completed.stdout.strip() if completed.returncode == 0 else ""
    if not revision:
        raise SystemExit("Official joint execution requires a resolvable Git commit")
    return revision


def _runtime_identity(device: str) -> dict[str, Any]:
    try:
        import cv2
        import torch
        import torchvision
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Install the [ml,xai] dependencies before joint evaluation") from exc
    return {
        "python_version": platform.python_version(),
        "device": device,
        "gpu": (
            torch.cuda.get_device_name(0)
            if device.startswith("cuda") and torch.cuda.is_available()
            else None
        ),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "opencv_runtime_version": cv2.__version__,
        "opencv_distribution_version": version("opencv-python-headless"),
        "grad_cam_distribution_version": version("grad-cam"),
        "pillow_distribution_version": version("Pillow"),
        "scikit_image_distribution_version": version("scikit-image"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
