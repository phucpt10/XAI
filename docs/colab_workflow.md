# Colab execution guide for protocol v1.4

This guide assumes that datasets, checkpoints, and run outputs are stored
outside Git, normally in Google Drive. Replace every placeholder path with the
location supplied by the project owner or replication coordinator.

## 1. Start a clean runtime

Use a GPU runtime for training and CAM generation. Mount Drive and clone the
release:

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
%cd /content
!git clone --branch protocol-v1.4 --depth 1 https://github.com/phucpt10/XAI.git XAI
%cd /content/XAI
!python -m pip install -q -e ".[hf,ml,xai,report,dev]"
```

Confirm the runtime and source revision:

```bash
!nvidia-smi
!git rev-parse HEAD
!python -m plantxai_stability.cli validate-protocol configs/protocol/v1.4/protocol.yaml
!python -m plantxai_stability.cli smoke configs/protocol/v1.4/protocol.yaml
```

## 2. Define external paths

```python
%env PXAI_ARCHIVE=/content/drive/MyDrive/PlantXAI-Stability/dataset-backup-v1/plantxai-dataset-bundle.tar
%env PXAI_FREEZE_RECORD=/content/drive/MyDrive/PlantXAI-Stability/dataset-recovery-v1/plantxai-frozen-recovery-v1/freeze_record.json
%env PXAI_VALIDATION_SPLIT=/content/drive/MyDrive/PlantXAI-Stability/dataset-recovery-v1/plantxai-frozen-recovery-v1/validation_split.csv
%env PXAI_RESNET50_CKPT=/content/drive/MyDrive/PlantXAI-Stability/runs/g0b-7eb0814b/resnet50-v1/resnet50_best.pt
%env PXAI_EFFICIENTNET_CKPT=/content/drive/MyDrive/PlantXAI-Stability/runs/g0b-7eb0814b/efficientnet-b0-v1/efficientnet_b0_best.pt
```

```bash
!test -f "$PXAI_ARCHIVE"
!test -f "$PXAI_FREEZE_RECORD"
!test -f "$PXAI_VALIDATION_SPLIT"
!test -f "$PXAI_RESNET50_CKPT"
!test -f "$PXAI_EFFICIENTNET_CKPT"
```

Do not place these files under `/content/XAI`.

## 3. Recover a validation-only image bundle

Use a new output directory. The script refuses to overwrite an existing
version:

```bash
!python scripts/build_validation_recovery_bundle.py \
  --archive "$PXAI_ARCHIVE" \
  --validation-manifest "$PXAI_VALIDATION_SPLIT" \
  --freeze-record "$PXAI_FREEZE_RECORD" \
  --output-dir "/content/plantxai-validation-v1"
```

## 4. Validate checkpoints and CAM configuration

Before a long run, execute the relevant validation-only audit and CAM preflight.
Inspect the exact required arguments in the checked-out release:

```bash
!python scripts/audit_validation_checkpoint.py --help
!python scripts/preflight_xai_validation.py --help
```

The preflight must use validation images only. Do not proceed when a checkpoint
hash, target layer, manifest identity, or CAM-quality check fails.

## 5. Run model--CAM parts

Protocol v1.4 declares:

- models: `resnet50`, `efficientnet_b0`;
- CAM methods: `grad_cam`, `grad_cam_plus_plus`, `score_cam`;
- transformations: rotation, brightness, Gaussian noise, Gaussian blur;
- three severity levels per transformation.

The campaign runner writes each model--method part to a separate immutable
directory and resumes committed samples:

```bash
!python -u scripts/run_joint_campaign_colab.py --help
```

A complete invocation requires:

- `configs/protocol/v1.4/protocol.yaml`;
- the frozen manifest and external image root;
- checkpoint, test-authorization, recovery, and readiness inputs requested by
  the runner;
- both checkpoint paths;
- an output root outside the repository;
- `--device cuda`.

Keep the same output root when resuming an interrupted campaign. Do not delete a
part database merely to restart a completed or partially completed run.

## 6. Merge and analyze

After all six parts report `PASS`, merge the three CAM parts separately for
each model:

```bash
!python scripts/merge_joint_runs.py --help
```

Then run the support audit and statistical analysis:

```bash
!python scripts/audit_analysis_support.py --help
!python scripts/analyze_official_results.py --help
```

Finally generate aggregate reporting tables if required:

```bash
!python scripts/generate_frozen_results_report.py --help
!python scripts/generate_classwise_supplement.py --help
```

All outputs must use new versioned directories. Never overwrite a completed
output directory.

## 7. Save and stop safely

Confirm that final reports and part databases are on Drive:

```bash
!find "/content/drive/MyDrive/PlantXAI-Stability" -type f \
  \( -name "*.json" -o -name "*.csv" -o -name "*.sqlite" \) | tail -n 50
```

Flush pending filesystem writes:

```bash
!sync
```

After files are visible on Drive and hashes/reports have been recorded, stop the
Colab runtime through **Runtime → Disconnect and delete runtime**. Repository
source can be cloned again; Drive artifacts are the persistent outputs.
