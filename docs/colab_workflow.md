# Colab workflow

The repository is designed so that GitHub contains code, protocol and tests,
while Colab provides the dataset cache, GPU and run output directory.

## 1. Clone and install

```python
%cd /content
!git clone <repository-url> PlantXAI-Stability
%cd /content/PlantXAI-Stability
!python -m pip install -e ".[hf,ml,xai,dev]"
```

## 2. Validate the governed protocol

```python
!python -m plantxai_stability.cli validate-protocol configs/protocol/v0.9/protocol.yaml
!python -m plantxai_stability.cli smoke configs/protocol/v0.9/protocol.yaml
```

G0B is now approved, but official training still requires a newly generated
freeze record with the same final protocol hash. Until that rebind exists, use
only non-official draft smoke runs for pipeline debugging.

## 3. Inspect the Hugging Face source and audit leaf identity

The configured source is `mohanty/PlantVillage`, configuration `color`. The
dataset card documents `train`/`test`, `image`, `label`, `leaf_id`, `crop` and
`disease`. First inspect the schema without decoding all images:

```python
!python scripts/inspect_hf_dataset.py \
  --dataset-id mohanty/PlantVillage \
  --configuration color \
  --revision 9e97599868962bd0079b8db4b7f1efa9185fa1e7 \
  --output-dir /content/plantxai-data-inspection
```

Inspect only the protocol's five tomato classes and retain its receipt:

```python
!python scripts/inspect_hf_dataset.py \
  --dataset-id mohanty/PlantVillage \
  --configuration color \
  --revision 9e97599868962bd0079b8db4b7f1efa9185fa1e7 \
  --classes Tomato___healthy Tomato___Bacterial_spot Tomato___Early_blight Tomato___Late_blight Tomato___Septoria_leaf_spot \
  --output-dir /content/plantxai-tomato-inspection-v1
```

Then apply the pinned source loader's filename rule and run all leaf-identity
gates. The output directory is immutable, so use a new versioned path for every
rerun:

```python
!python scripts/audit_leaf_identity.py \
  --dataset-id mohanty/PlantVillage \
  --configuration color \
  --revision 9e97599868962bd0079b8db4b7f1efa9185fa1e7 \
  --classes Tomato___healthy Tomato___Bacterial_spot Tomato___Early_blight Tomato___Late_blight Tomato___Septoria_leaf_spot \
  --dataset-receipt /content/plantxai-tomato-inspection-v1/dataset_receipt.json \
  --output-dir /content/plantxai-leaf-audit-v1
```

For this pinned revision the command is expected to exit nonzero: coverage is
100%, but five reconstructed leaf identities cross train/test. Inspect the ten
affected records without modifying them:

```python
import pandas as pd

report = pd.read_parquet(
    "/content/plantxai-leaf-audit-v1/leaf_identity_resolution_report.parquet"
)
overlap = report[report["reason_code"].str.contains("LEAF_SPLIT_OVERLAP", na=False)]
display(overlap[[
    "source_split", "source_row_index", "image_path", "class_name",
    "resolved_leaf_id", "leaf_id_source", "reason_code",
]].sort_values(["resolved_leaf_id", "source_split"]))
```

`DR-LEAF-001.yaml` permanently records the failed raw-source decision.
`DR-LEAF-002.yaml` approves preserving every official test sample and
quarantining exactly the five source-train counterparts. Materialize that
decision without decoding pixels:

```python
!python scripts/adjudicate_quarantine.py \
  --leaf-report /content/plantxai-leaf-audit-v1/leaf_identity_resolution_report.parquet \
  --leaf-summary /content/plantxai-leaf-audit-v1/leaf_identity_resolution_summary.json \
  --dataset-receipt /content/plantxai-tomato-inspection-v1/dataset_receipt.json \
  --decision-record configs/protocol/v0.9/decision_records/DR-LEAF-002.yaml \
  --output-dir /content/plantxai-quarantine-v1
```

This gate must report 10 overlap members, 5 quarantined train rows, 1,693
preserved official test rows and `passed: true`.

## 4. Build the quarantined modeling manifest

Both class scope and quarantine policy are approved for the exact evidence
hashes recorded in their Decision Records. Decode every image, retain all rows
in the lineage manifest, and exclude only the five approved train rows from the
modeling manifest:

```python
!python scripts/inspect_hf_dataset.py \
  --dataset-id mohanty/PlantVillage \
  --configuration color \
  --revision 9e97599868962bd0079b8db4b7f1efa9185fa1e7 \
  --classes Tomato___healthy Tomato___Bacterial_spot Tomato___Early_blight Tomato___Late_blight Tomato___Septoria_leaf_spot \
  --manifest \
  --class-selection-dr configs/protocol/v0.9/decision_records/DR-CLASS-001.yaml \
  --leaf-identity-dr configs/protocol/v0.9/decision_records/DR-LEAF-002.yaml \
  --leaf-identity-report /content/plantxai-leaf-audit-v1/leaf_identity_resolution_report.parquet \
  --leaf-identity-summary /content/plantxai-leaf-audit-v1/leaf_identity_resolution_summary.json \
  --governance-dataset-receipt /content/plantxai-tomato-inspection-v1/dataset_receipt.json \
  --quarantine-adjudication-summary /content/plantxai-quarantine-v1/quarantine_adjudication_summary.json \
  --quarantine-decision-registry /content/plantxai-quarantine-v1/quarantine_decision_registry.parquet \
  --output-dir /content/plantxai-manifest-v2
```

The `--manifest` option also exports audited RGB PNGs under
`/content/plantxai-manifest-v2/images`, so the existing filesystem-backed
PyTorch loader can consume exactly the hashed pixels. The manifest preserves
the upstream `train`/`test` assignment and records canonical RGB hashes,
dimensions, class labels and `leaf_id`. It also writes the 8,398-row
`dataset_lineage_manifest.parquet`, five-row `quarantine_registry.parquet` and
8,393-row eligible `manifest.csv`. A source-derived reconstruction is
allowed only when its Decision Record and evidence gate are approved; an
unresolved group key is always a hard audit failure.

The pixel audit for the pinned data finds nine additional train-only duplicate
pairs. Every pair has one class and one leaf. Apply `DR-DUP-001` to retain the
minimum stable sample ID and quarantine the redundant copy:

```python
!python scripts/adjudicate_exact_duplicates.py \
  --manifest /content/plantxai-manifest-v2/manifest.csv \
  --lineage-manifest /content/plantxai-manifest-v2/dataset_lineage_manifest.parquet \
  --quarantine-registry /content/plantxai-manifest-v2/quarantine_registry.parquet \
  --quarantine-summary /content/plantxai-manifest-v2/quarantine_summary.json \
  --decision-record configs/protocol/v0.9/decision_records/DR-DUP-001.yaml \
  --output-dir /content/plantxai-manifest-v3
```

The final reconciliation must contain 8,384 eligible samples, 14 quarantined
train samples, 6,691 eligible source-train samples, all 1,693 official test
samples and a passing eligible image audit. The RGB files remain under
`/content/plantxai-manifest-v2/images`; v3 contains the final governance and
manifest artifacts only.

Do not replace missing identity with a row index, traversal order or arbitrary
synthetic `leaf_id`.

After the manifest quarantine and eligible image audit both pass, freeze the
manifest and create immutable split evidence:

```python
!python scripts/freeze_dataset.py \
  --protocol configs/protocol/v0.9/protocol.yaml \
  --manifest /content/plantxai-manifest-v3/manifest.csv \
  --class-selection-dr configs/protocol/v0.9/decision_records/DR-CLASS-001.yaml \
  --quarantine-dr configs/protocol/v0.9/decision_records/DR-DUP-001.yaml \
  --quarantine-registry /content/plantxai-manifest-v3/quarantine_registry.parquet \
  --quarantine-summary /content/plantxai-manifest-v3/quarantine_summary.json \
  --audit-identity /content/plantxai-tomato-inspection-v1/dataset_receipt.json \
  --output-dir /content/plantxai-frozen-data
```

Training and evaluation must use the frozen split CSVs, never a directory scan.

## 5. Pilot transformation severity on validation only

The severity pilot selects at most one sample per leaf, caps selection equally
per declared class, and computes image-space MAE, RMSE, PSNR and SSIM for all
twelve scenarios. It never loads a model or accesses the official test split.
Use a new immutable output directory for each attempt:

Pilot v1 is permanently rejected by `DR-SEVERITY-001`: it independently drew
brightness/rotation direction and Gaussian noise at every severity. The pinned
v2 algorithm fixed that error, but `DR-SEVERITY-002` rejects its constant-black
rotation fill as a severity-correlated shortcut. `DR-SEVERITY-003` rejects the
v3 border-median fill because it still creates severity-correlated uniform
polygons. `DR-SEVERITY-004` rejects v4 because reflect padding repeats leaf
fragments at the image borders. `DR-SEVERITY-005` rejects the visually smeared
Telea v5 output and approves v6: deterministic zero-filled rotation plus a
geometric valid-region mask for XAI comparison. Prediction claims are specific
to the zero-filled operator. Explanation-stability claims use forward CAM
alignment, masked Pearson/SSIM, primary top-0.2 IoU and sensitivity checks at
0.1 and 0.3. M_T denotes valid image support, not leaf segmentation.

```python
!python scripts/pilot_transform_severity.py \
  --protocol configs/protocol/v0.9/protocol.yaml \
  --manifest /content/plantxai-frozen-v1/dataset_manifest.csv \
  --image-root /content/plantxai-manifest-v2 \
  --output-dir /content/plantxai-severity-pilot-v6 \
  --max-leaves-per-class 50 \
  --minimum-leaves-per-class 20
```

A technical PASS requires validation-only lineage, one sample per leaf, all
twelve scenarios, finite metrics, deterministic repeat checks and strictly
increasing median RMSE from mild to moderate to severe for each transformation.
The report remains `pending_human_review`; it must not change the protocol or
remove the severity blocker automatically.

Black corners are an intentional, declared part of the v6 image operator and
are not by themselves a visual-review failure. They remain in model input, so
prediction robustness is reported only for the zero-filled operator. The CAM
comparison gate separately verifies and excludes their geometric support using
M_T; visual review must not describe M_T as a leaf or lesion mask.

Render one deterministic validation example per class for human review. This
verifies the pilot artifact hashes before producing four contact sheets:

```python
!python scripts/render_severity_review.py \
  --protocol configs/protocol/v0.9/protocol.yaml \
  --manifest /content/plantxai-frozen-v1/dataset_manifest.csv \
  --image-root /content/plantxai-manifest-v2 \
  --pilot-summary /content/plantxai-severity-pilot-v6/severity_pilot_summary.json \
  --output-dir /content/plantxai-severity-review-v6
```

The project owner approved these artifacts through `DR-SEVERITY-006` with
`PASS_WITH_DECLARED_OPERATOR_LIMITATION`. Pull that Decision Record and verify
the final G0B protocol identity:

```python
%cd /content/PlantXAI-Stability
!git pull origin main
!python -m plantxai_stability.cli validate-protocol \
  configs/protocol/v0.9/protocol.yaml
```

The expected final G0B protocol hash is
`7eb0814be8ffc1a19f54e2bec2d2ca0c84d7f4d869d99e28b69e6c9e0e84523b`.
The existing `/content/plantxai-frozen-v1` record has the earlier draft hash
and cannot authorize official training. Recreate only the immutable freeze
artifacts; do not redownload, rematerialize or resplit the dataset manually:

```python
!python scripts/freeze_dataset.py \
  --protocol configs/protocol/v0.9/protocol.yaml \
  --manifest /content/plantxai-manifest-v3/manifest.csv \
  --class-selection-dr configs/protocol/v0.9/decision_records/DR-CLASS-001.yaml \
  --quarantine-dr configs/protocol/v0.9/decision_records/DR-DUP-001.yaml \
  --quarantine-registry /content/plantxai-manifest-v3/quarantine_registry.parquet \
  --quarantine-summary /content/plantxai-manifest-v3/quarantine_summary.json \
  --audit-identity /content/plantxai-tomato-inspection-v1/dataset_receipt.json \
  --output-dir /content/plantxai-frozen-final-v1
```

Verify the new cryptographic binding before training:

```python
import json
from pathlib import Path
from plantxai_stability.config import load_protocol

protocol = load_protocol("configs/protocol/v0.9/protocol.yaml")
freeze = json.loads(
    Path("/content/plantxai-frozen-final-v1/freeze_record.json").read_text()
)
assert freeze["protocol_hash"] == protocol.sha256
assert freeze["protocol_hash"] == (
    "7eb0814be8ffc1a19f54e2bec2d2ca0c84d7f4d869d99e28b69e6c9e0e84523b"
)
print("Final G0B freeze binding: PASS")
```

## 6. Train and preserve checkpoints

The training API in `plantxai_stability.training` selects checkpoints only by
validation macro-F1. Training is not a prerequisite for G0B: first approve
severity, freeze the pre-training protocol and regenerate its frozen-data
record. Checkpoint selection is the subsequent G1 gate. Official test remains
locked until G2.

Store training state on Google Drive so a Colab disconnect does not destroy it:

```python
from google.colab import drive
drive.mount("/content/drive")
```

After the final G0B freeze binding passes, train ResNet50 in a new directory:

```python
!python scripts/train_colab.py \
  --protocol configs/protocol/v0.9/protocol.yaml \
  --manifest /content/plantxai-frozen-final-v1/dataset_manifest.csv \
  --image-root /content/plantxai-manifest-v2 \
  --model-id resnet50 \
  --output-dir /content/drive/MyDrive/PlantXAI-Stability/runs/resnet50-v1 \
  --num-workers 0 \
  --device cuda
```

At every completed epoch the runner atomically writes `resnet50_latest.pt`,
updates history, and immediately preserves a new `resnet50_best.pt` when
validation macro-F1 improves. Resume the same run after interruption:

```python
!python scripts/train_colab.py \
  --protocol configs/protocol/v0.9/protocol.yaml \
  --manifest /content/plantxai-frozen-final-v1/dataset_manifest.csv \
  --image-root /content/plantxai-manifest-v2 \
  --model-id resnet50 \
  --output-dir /content/drive/MyDrive/PlantXAI-Stability/runs/resnet50-v1 \
  --num-workers 0 \
  --device cuda \
  --resume
```

Resume is fail-closed: model, class count, protocol hash, manifest hash and all
training settings must exactly match. Repeat in a distinct directory for
EfficientNet-B0. `--allow-draft-training` is restricted to non-official smoke
debugging and cannot produce a G1-approved checkpoint.

Audit each validation-selected best checkpoint before G1 approval. The audit
loads validation pixels only, reproduces the selected macro-F1, verifies exact
sample identity coverage and writes per-class metrics, a confusion matrix and
per-sample class probabilities. It hard-fails on any checkpoint, protocol,
manifest, freeze-record or training-evidence mismatch.

```python
!python scripts/audit_validation_checkpoint.py \
  --protocol configs/protocol/v0.9/protocol.yaml \
  --manifest /content/plantxai-frozen-final-v1/dataset_manifest.csv \
  --image-root /content/plantxai-manifest-v2 \
  --model-id resnet50 \
  --checkpoint /content/drive/MyDrive/PlantXAI-Stability/runs/g0b-7eb0814b/resnet50-v1/resnet50_best.pt \
  --checkpoint-evidence /content/drive/MyDrive/PlantXAI-Stability/runs/g0b-7eb0814b/resnet50-v1/resnet50_checkpoint_evidence.json \
  --output-dir /content/drive/MyDrive/PlantXAI-Stability/runs/g0b-7eb0814b/resnet50-v1/validation-audit-v1 \
  --num-workers 4 \
  --device cuda
```

```python
!python scripts/audit_validation_checkpoint.py \
  --protocol configs/protocol/v0.9/protocol.yaml \
  --manifest /content/plantxai-frozen-final-v1/dataset_manifest.csv \
  --image-root /content/plantxai-manifest-v2 \
  --model-id efficientnet_b0 \
  --checkpoint /content/drive/MyDrive/PlantXAI-Stability/runs/g0b-7eb0814b/efficientnet-b0-v1/efficientnet_b0_best.pt \
  --checkpoint-evidence /content/drive/MyDrive/PlantXAI-Stability/runs/g0b-7eb0814b/efficientnet-b0-v1/efficientnet_b0_checkpoint_evidence.json \
  --output-dir /content/drive/MyDrive/PlantXAI-Stability/runs/g0b-7eb0814b/efficientnet-b0-v1/validation-audit-v1 \
  --num-workers 4 \
  --device cuda
```

Both reports must state `source_split: validation`,
`test_split_accessed: false`, `sample_coverage_exact: true` and
`selected_macro_f1_reproduced: true`. These are G1 candidate artifacts, not
official test results. Do not compare or select checkpoints using test data.

The project owner approved both audited checkpoints in `DR-CHECKPOINT-001`.
Their immutable training protocol remains the G0B hash
`7eb0814be8ffc1a19f54e2bec2d2ca0c84d7f4d869d99e28b69e6c9e0e84523b`;
the later governance protocol has a different hash. This is an explicit
lineage transition, not a reason to rewrite checkpoint metadata or bytes.
When reproducing the audit after G1, add:

```text
--checkpoint-decision-record configs/protocol/v0.9/decision_records/DR-CHECKPOINT-001.yaml
```

G1 approval does not authorize the next commands. Official test remains
blocked until a separate G2 Decision Record and final lineage gate pass.

## 7. Prepare metadata-only G2 readiness evidence

This preflight hashes the two checkpoints, their training evidence, both
validation audit reports and every child audit artifact. It enumerates frozen
test `sample_id` and `leaf_id` values from the manifest only. It does not build
a test Dataset/DataLoader, decode test images, run inference or produce a test
metric. G2 must still be blocked when this command runs.

```python
!python scripts/prepare_g2_readiness.py \
  --protocol configs/protocol/v0.9/protocol.yaml \
  --manifest /content/plantxai-frozen-final-v1/dataset_manifest.csv \
  --checkpoint-decision-record configs/protocol/v0.9/decision_records/DR-CHECKPOINT-001.yaml \
  --resnet50-checkpoint /content/drive/MyDrive/PlantXAI-Stability/runs/g0b-7eb0814b/resnet50-v1/resnet50_best.pt \
  --resnet50-evidence /content/drive/MyDrive/PlantXAI-Stability/runs/g0b-7eb0814b/resnet50-v1/resnet50_checkpoint_evidence.json \
  --resnet50-audit-report /content/drive/MyDrive/PlantXAI-Stability/runs/g0b-7eb0814b/resnet50-v1/validation-audit-v1/validation_checkpoint_audit.json \
  --efficientnet-b0-checkpoint /content/drive/MyDrive/PlantXAI-Stability/runs/g0b-7eb0814b/efficientnet-b0-v1/efficientnet_b0_best.pt \
  --efficientnet-b0-evidence /content/drive/MyDrive/PlantXAI-Stability/runs/g0b-7eb0814b/efficientnet-b0-v1/efficientnet_b0_checkpoint_evidence.json \
  --efficientnet-b0-audit-report /content/drive/MyDrive/PlantXAI-Stability/runs/g0b-7eb0814b/efficientnet-b0-v1/validation-audit-v1/validation_checkpoint_audit.json \
  --output-dir /content/drive/MyDrive/PlantXAI-Stability/runs/g2-readiness-v1
```

The historical pre-authorization report must end with `G2 readiness technical gate: PASS`, retain
`approval_status: pending_g2_human_review`, state
`official_test.pixels_accessed: false` and leave both G2 and official-test
evaluation disabled. Record the separately printed report SHA-256 for the G2
Decision Record. Do not run baseline or joint evaluation after this preflight.

After project-owner approval creates `DR-TEST-001`, pull the G2 code and verify
the complete authorization chain. This command still reads manifest metadata
only and must report that no test pixel or result was accessed:

```python
!python scripts/verify_g2_authorization.py \
  --protocol configs/protocol/v0.9/protocol.yaml \
  --manifest /content/plantxai-frozen-final-v1/dataset_manifest.csv \
  --checkpoint-decision-record configs/protocol/v0.9/decision_records/DR-CHECKPOINT-001.yaml \
  --test-decision-record configs/protocol/v0.9/decision_records/DR-TEST-001.yaml \
  --g2-readiness-report /content/drive/MyDrive/PlantXAI-Stability/runs/g2-readiness-v1/g2_readiness_report.json \
  --resnet50-checkpoint /content/drive/MyDrive/PlantXAI-Stability/runs/g0b-7eb0814b/resnet50-v1/resnet50_best.pt \
  --efficientnet-b0-checkpoint /content/drive/MyDrive/PlantXAI-Stability/runs/g0b-7eb0814b/efficientnet-b0-v1/efficientnet_b0_best.pt \
  --output-dir /content/drive/MyDrive/PlantXAI-Stability/runs/g2-authorization-v1
```

Stop after `G2 authorization gate: PASS` and preserve its printed report hash.
Do not start baseline or joint evaluation until this verification artifact has
been reviewed.

Baseline test evaluation is a separate step:

```python
!python scripts/evaluate_baseline.py \
  --protocol configs/protocol/v0.9/protocol.yaml \
  --manifest /content/plantxai-frozen-final-v1/dataset_manifest.csv \
  --image-root /content/plantxai-manifest-v2 \
  --model-id resnet50 \
  --checkpoint /content/drive/MyDrive/PlantXAI-Stability/runs/g0b-7eb0814b/resnet50-v1/resnet50_best.pt \
  --checkpoint-decision-record configs/protocol/v0.9/decision_records/DR-CHECKPOINT-001.yaml \
  --test-decision-record configs/protocol/v0.9/decision_records/DR-TEST-001.yaml \
  --g2-readiness-report /content/drive/MyDrive/PlantXAI-Stability/runs/g2-readiness-v1/g2_readiness_report.json \
  --output-dir /content/drive/MyDrive/PlantXAI-Stability/runs/official-test-v1/resnet50-baseline \
  --num-workers 4 \
  --device cuda
```

The baseline and joint runners recompute the full authorization chain before
opening a test image. A G2 flag without matching Decision Records, readiness
hash, checkpoint bytes, manifest/freeze lineage and test identities hard-fails.

Run the registered joint campaign as six model-method parts. The launcher skips
completed parts and resumes an interrupted part from its last transaction, so
the same command is safe after a Colab disconnect:

```python
!python scripts/run_joint_campaign_colab.py \
  --protocol configs/protocol/v0.9/protocol.yaml \
  --manifest /content/plantxai-frozen-final-v1/dataset_manifest.csv \
  --image-root /content/plantxai-manifest-v2 \
  --checkpoint-decision-record configs/protocol/v0.9/decision_records/DR-CHECKPOINT-001.yaml \
  --test-decision-record configs/protocol/v0.9/decision_records/DR-TEST-001.yaml \
  --g2-readiness-report /content/drive/MyDrive/PlantXAI-Stability/runs/g2-readiness-v1/g2_readiness_report.json \
  --resnet50-checkpoint /content/drive/MyDrive/PlantXAI-Stability/runs/g0b-7eb0814b/resnet50-v1/resnet50_best.pt \
  --efficientnet-b0-checkpoint /content/drive/MyDrive/PlantXAI-Stability/runs/g0b-7eb0814b/efficientnet-b0-v1/efficientnet_b0_best.pt \
  --output-root /content/drive/MyDrive/PlantXAI-Stability/runs/official-test-v1/joint-parts \
  --device cuda
```

For a short first operational check, append
`--only-model resnet50 --only-method grad_cam`. Then rerun the command without
the filters; the completed part is skipped. Do not delete `run_state.json` or
`joint_progress.sqlite3`. Every retry is bound to the same Git commit and all
scientific identities; a code pull during an incomplete part intentionally
blocks resume.

After all six parts report `Official joint part: PASS`, merge the three parts
for each model. Example for ResNet50:

```python
!python scripts/merge_joint_runs.py \
  --protocol configs/protocol/v0.9/protocol.yaml \
  --baseline-report /content/drive/MyDrive/PlantXAI-Stability/runs/official-test-v1/resnet50-baseline/baseline_metrics.json \
  --part-dir /content/drive/MyDrive/PlantXAI-Stability/runs/official-test-v1/joint-parts/resnet50-grad_cam-v1 \
  --part-dir /content/drive/MyDrive/PlantXAI-Stability/runs/official-test-v1/joint-parts/resnet50-grad_cam_plus_plus-v1 \
  --part-dir /content/drive/MyDrive/PlantXAI-Stability/runs/official-test-v1/joint-parts/resnet50-score_cam-v1 \
  --output-dir /content/drive/MyDrive/PlantXAI-Stability/runs/official-test-v1/resnet50-joint-merged-v1 \
  --run-id resnet50-joint-merged-v1
```

Replace `resnet50` with `efficientnet_b0` and use the EfficientNet baseline
report for the second merge. A successful merged model contains exactly 20,316
prediction rows (`1,693 x 12`) and 60,948 joint rows (`1,693 x 12 x 3`),
including explicit exclusion rows where prediction consistency or CAM quality
prevents a stability metric.

Keep checkpoints and run outputs outside the source tree or in Git LFS/object
storage; do not commit model weights to the normal Git history.

## 8. Export results

Every run must have a unique `run_id` and write resolved configuration,
predictions, heatmaps, metrics, statistics, tables, figures, logs and a
`run_manifest.json`. Upload the final artifact bundle to the report workspace,
not into the Python source package.
