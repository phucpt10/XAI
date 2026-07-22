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

## 2. Validate the draft protocol

```python
!python -m plantxai_stability.cli validate-protocol configs/protocol/v0.9/protocol.yaml
!python -m plantxai_stability.cli smoke configs/protocol/v0.9/protocol.yaml
```

The official runner must remain blocked until G0B is approved. Use a separate
non-official draft training run only for pipeline debugging, and label its
outputs as `draft_smoke`.

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
`shared_randomization_v2` algorithm holds those nuisance variables fixed while
only perturbation magnitude changes.

```python
!python scripts/pilot_transform_severity.py \
  --protocol configs/protocol/v0.9/protocol.yaml \
  --manifest /content/plantxai-frozen-v1/dataset_manifest.csv \
  --image-root /content/plantxai-manifest-v2 \
  --output-dir /content/plantxai-severity-pilot-v2 \
  --max-leaves-per-class 50 \
  --minimum-leaves-per-class 20
```

A technical PASS requires validation-only lineage, one sample per leaf, all
twelve scenarios, finite metrics, deterministic repeat checks and strictly
increasing median RMSE from mild to moderate to severe for each transformation.
The report remains `pending_human_review`; it must not change the protocol or
remove the severity blocker automatically.

Render one deterministic validation example per class for human review. This
verifies the pilot artifact hashes before producing four contact sheets:

```python
!python scripts/render_severity_review.py \
  --protocol configs/protocol/v0.9/protocol.yaml \
  --manifest /content/plantxai-frozen-v1/dataset_manifest.csv \
  --image-root /content/plantxai-manifest-v2 \
  --pilot-summary /content/plantxai-severity-pilot-v2/severity_pilot_summary.json \
  --output-dir /content/plantxai-severity-review-v2
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

After G0B passes, train ResNet50 in a new directory:

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

Baseline test evaluation is a separate step:

```python
!python scripts/evaluate_baseline.py \
  --protocol configs/protocol/v0.9/protocol.yaml \
  --manifest /content/plantxai-frozen-final-v1/dataset_manifest.csv \
  --image-root /content/plantxai-manifest-v2 \
  --model-id resnet50 \
  --checkpoint /content/plantxai-runs/resnet50/resnet50_best.pt \
  --output-dir /content/plantxai-runs/resnet50/baseline \
  --device cuda
```

The baseline and joint robustness/XAI runners both hard-fail until G2 explicitly
sets `official_test_evaluation_allowed: true` after checkpoint approval.

Keep checkpoints and run outputs outside the source tree or in Git LFS/object
storage; do not commit model weights to the normal Git history.

## 7. Export results

Every run must have a unique `run_id` and write resolved configuration,
predictions, heatmaps, metrics, statistics, tables, figures, logs and a
`run_manifest.json`. Upload the final artifact bundle to the report workspace,
not into the Python source package.
