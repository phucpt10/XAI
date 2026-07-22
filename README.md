# PlantXAI-Stability

PlantXAI-Stability is research software for evaluating prediction robustness and XAI explanation stability under controlled image transformations.

The implementation follows the English specification in `PlantXAI-Stability_Research_Software_Specification_En.docx` and keeps the protocol in a fail-closed draft state until the dataset audit, leaf identity, frozen splits, severity pilot, checkpoint selection and target-layer validation are approved.

## Current implementation

- Versioned protocol draft at `configs/protocol/v0.9/protocol.yaml`.
- JSON schema and fail-closed protocol loader.
- Immutable data contracts for samples, predictions, transformations and joint records.
- Canonical RGB hashing and deterministic `sample_id` construction.
- Optional Hugging Face Datasets adapter for `mohanty/PlantVillage` with schema
  inspection, `leaf_id` validation and manifest materialisation.
- Dataset receipt, image-level audit, duplicate/conflict detection and immutable
  manifest/split freeze artifacts.
- Governed quarantine adjudication that preserves every official test sample,
  excludes only approved source-train conflicts, reconciles every audited row,
  and carries registry hashes into the freeze record.
- Deterministic leaf-stratified splitting and DataLoader re-validation of image
  shape, RGB hash, pixel range and identity metadata.
- Leaf-safe split validation and train/validation grouping.
- Identity-preserving dataset adapter and optional PyTorch DataLoader.
- Deterministic rotation, brightness, Gaussian noise and Gaussian blur transformations.
- Optional PyTorch model wrappers for ResNet50 and EfficientNet-B0.
- Optional CAM adapter for Grad-CAM, Grad-CAM++ and Score-CAM through `pytorch-grad-cam`.
- Heatmap quality validation, SSIM/Pearson/cosine metrics, leaf-cluster bootstrap, paired Wilcoxon and Holm correction.
- Run provenance and artifact index helpers.
- Unit, integration and scientific invariant tests.

## Safe commands

```powershell
python -m pip install -e ".[hf,dev]"
python -m pytest
python -m plantxai_stability.cli validate-protocol configs/protocol/v0.9/protocol.yaml
python -m plantxai_stability.cli smoke configs/protocol/v0.9/protocol.yaml
```

The official `run` command intentionally refuses to execute while the protocol is draft or G0B is blocked. This prevents accidental scientific claims before the required evidence exists.

## Official-run prerequisites

1. Audit the pinned dataset revision (`9e97599868962bd0079b8db4b7f1efa9185fa1e7`) and verify that `leaf_id` is present and reliable.
2. Build and approve the canonical manifest.
3. Freeze leaf-safe train/validation/test splits.
4. Pilot and approve transformation severities.
5. Select and hash validation-approved model checkpoints.
6. Runtime-validate target layers and CAM adapters.
7. Freeze protocol and set G0B to PASS through a reviewed governance change.

No dataset count, accuracy, confidence interval, p-value or stability result is fabricated by this repository.
