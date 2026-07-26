# PlantXAI-Stability

PlantXAI-Stability is research software for measuring prediction robustness and
conditional class-activation-map (CAM) stability under controlled image
transformations.

This repository contains executable code, protocol configurations, tests, and
the minimum instructions required to run the software. It does not contain
datasets, checkpoints, experiment outputs, reviewer material, manuscripts,
AI-assisted working files, or system-design documents.

## Release scope

The `protocol-v1.4` release contains:

- deterministic rotation, brightness, Gaussian-noise, and Gaussian-blur
  transformations at three within-family severity levels;
- ResNet50 and EfficientNet-B0 training and inference adapters;
- Grad-CAM, Grad-CAM++, and Score-CAM generation with model-specific target
  layers;
- forward alignment and geometric support masking for rotated CAMs;
- masked Pearson correlation, SSIM, exact top-k IoU, and explicit invalid-CAM
  handling;
- physical-leaf bootstrap inference, paired Wilcoxon tests, rank-biserial
  effects, and Holm correction;
- scripts for data inspection, split freezing, training, validation,
  joint evaluation, merging, analysis, and result-table generation;
- unit, integration, and scientific-invariant tests.

## Requirements

- Python 3.10 or newer;
- CUDA-capable PyTorch environment for practical training and CAM execution;
- CPU-only execution is sufficient for configuration validation, most tests,
  merging, and statistical analysis.

Install the complete environment:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[hf,ml,xai,report,dev]"
```

## Verify the release

```bash
python -m pytest -q
python -m ruff check src tests scripts
python -m mypy src
python -m plantxai_stability.cli validate-protocol configs/protocol/v1.4/protocol.yaml
python -m plantxai_stability.cli smoke configs/protocol/v1.4/protocol.yaml
```

## Inputs not stored in Git

Execution requires separately supplied artifacts, depending on the selected
stage:

- the pinned PlantVillage source or a prepared image store;
- frozen manifest and split CSV files;
- checkpoint files for evaluation;
- prior-stage reports required by a selected verification command.

Keep all such files outside the repository. Do not copy data, checkpoints, or
run outputs into Git.

## Typical workflow

1. Install dependencies and validate `configs/protocol/v1.4/protocol.yaml`.
2. Prepare or mount the dataset outside the repository.
3. Inspect and freeze the manifest if starting from the source dataset.
4. Train the declared models or provide compatible checkpoints.
5. Run validation and XAI preflight checks.
6. Execute the selected model--CAM parts.
7. Merge completed parts.
8. Run statistical analysis and generate aggregate tables.

See [`docs/colab_workflow.md`](docs/colab_workflow.md) for command templates.
Every script also provides argument-level help:

```bash
python scripts/run_joint_eval.py --help
python scripts/analyze_official_results.py --help
```

## Reproducibility boundaries

- The configured dataset revision and five tomato classes define the released
  experiment population.
- Rotation results are specific to the configured zero-filled operator.
- CAM stability is conditional on prediction consistency and valid clean and
  transformed CAMs.
- Severity values are ordinal only within the same transformation family.
- Stability must not be interpreted as explanation faithfulness or biological
  correctness.
- The release characterizes fixed trained systems unless users explicitly run
  and report multiple training seeds.

## Repository hygiene

Before every commit, verify:

```bash
git status --short
git ls-files
```

The tracked tree must not contain private data, reviewer correspondence,
manuscripts, generated papers, AI-assisted workspaces, model checkpoints, or
experiment outputs.
