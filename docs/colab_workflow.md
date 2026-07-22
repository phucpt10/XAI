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

## 3. Inspect the Hugging Face source and build the manifest

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

For the protocol's five tomato classes, materialise the canonical manifest:

The repository ships `DR-CLASS-001.yaml` in `draft` status. A reviewer must
complete its audit identity and approval fields, then set `status: approved`
before the `--manifest` command is allowed to create an official manifest.

```python
!python scripts/inspect_hf_dataset.py \
  --dataset-id mohanty/PlantVillage \
  --configuration color \
  --revision 9e97599868962bd0079b8db4b7f1efa9185fa1e7 \
  --classes Tomato___healthy Tomato___Bacterial_spot Tomato___Early_blight Tomato___Late_blight Tomato___Septoria_leaf_spot \
  --manifest \
  --output-dir /content/plantxai-data-inspection
```

The `--manifest` option also exports audited RGB PNGs under
`/content/plantxai-data-inspection/images`, so the existing filesystem-backed
PyTorch loader can consume exactly the hashed pixels. The manifest preserves
the upstream `train`/`test` assignment and records canonical RGB hashes,
dimensions, class labels and `leaf_id`. Never fabricate `leaf_id`; an absent or
empty group key is a hard audit failure.

Prepare a metadata CSV with `relative_path`, `leaf_id`, `class_name`,
`class_id` and `source_split`, then run:

```python
!python scripts/inspect_dataset.py \
  --root /content/PlantXAI-data \
  --metadata-csv /content/metadata.csv \
  --manifest-out /content/manifest.csv \
  --audit-out /content/dataset_audit.json
```

Do not create a synthetic `leaf_id` when the source does not provide one.

After the class-selection Decision Record is approved, freeze the manifest and
create the immutable split evidence:

```python
!python scripts/freeze_dataset.py \
  --protocol configs/protocol/v0.9/protocol.yaml \
  --manifest /content/plantxai-frozen-data/dataset_manifest.csv \
  --class-selection-dr configs/protocol/v0.9/decision_records/DR-CLASS-001.yaml \
  --audit-identity /content/plantxai-data-inspection/dataset_receipt.json \
  --output-dir /content/plantxai-frozen-data
```

Training and evaluation must use the frozen split CSVs, never a directory scan.

## 4. Train and preserve checkpoints

The training API in `plantxai_stability.training` selects checkpoints only by
validation macro-F1 and writes both the checkpoint hash and training history.
For a non-official debugging run while G0B is blocked:

```python
!python scripts/train_colab.py \
  --protocol configs/protocol/v0.9/protocol.yaml \
  --manifest /content/plantxai-frozen-data/dataset_manifest.csv \
  --image-root /content/plantxai-data-inspection \
  --model-id resnet50 \
  --output-dir /content/plantxai-runs/draft_smoke/resnet50 \
  --allow-draft-training \
  --device cuda
```

After protocol freeze, remove `--allow-draft-training` and train each backbone
in a separate run directory. The script selects checkpoints only from the
validation split and writes a checkpoint evidence JSON file.

Baseline test evaluation is a separate step:

```python
!python scripts/evaluate_baseline.py \
  --protocol configs/protocol/v0.9/protocol.yaml \
  --manifest /content/plantxai-frozen-data/dataset_manifest.csv \
  --image-root /content/plantxai-data-inspection \
  --model-id resnet50 \
  --checkpoint /content/plantxai-runs/resnet50/resnet50_best.pt \
  --output-dir /content/plantxai-runs/resnet50/baseline \
  --device cuda
```

The joint robustness/XAI runner is `scripts/run_joint_eval.py` and requires a
frozen protocol, a validated checkpoint and the XAI dependencies.

Keep checkpoints and run outputs outside the source tree or in Git LFS/object
storage; do not commit model weights to the normal Git history.

## 5. Export results

Every run must have a unique `run_id` and write resolved configuration,
predictions, heatmaps, metrics, statistics, tables, figures, logs and a
`run_manifest.json`. Upload the final artifact bundle to the report workspace,
not into the Python source package.
