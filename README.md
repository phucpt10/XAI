# PlantXAI-Stability

PlantXAI-Stability is research software for evaluating prediction robustness and XAI explanation stability under controlled image transformations.

The implementation follows the English specification in `PlantXAI-Stability_Research_Software_Specification_En.docx`. G0B is frozen after the approved dataset, runtime, target-layer and severity evidence; checkpoint selection and official test evaluation remain independently fail-closed.

## Current implementation

- Versioned G0B-frozen protocol at `configs/protocol/v0.9/protocol.yaml`.
- JSON schema and fail-closed protocol loader.
- Immutable data contracts for samples, predictions, transformations and joint records.
- Canonical RGB hashing and deterministic `sample_id` construction.
- Source-lineage-preserving materialized paths, including distinct identities
  for different source samples with identical canonical RGB pixels.
- Optional Hugging Face Datasets adapter for `mohanty/PlantVillage` with schema
  inspection, `leaf_id` validation and manifest materialisation.
- Dataset receipt, image-level audit, duplicate/conflict detection and immutable
  manifest/split freeze artifacts.
- Governed quarantine adjudication that preserves every official test sample,
  excludes only approved source-train conflicts, reconciles every audited row,
  and carries registry hashes into the freeze record.
- Iterative exact-duplicate adjudication that deterministically retains one
  train representative per same-class, same-leaf pixel-identical pair.
- Deterministic leaf-stratified splitting and DataLoader re-validation of image
  shape, RGB hash, pixel range and identity metadata.
- Leaf-safe split validation and train/validation grouping.
- Identity-preserving dataset adapter and optional PyTorch DataLoader.
- Deterministic rotation, brightness, Gaussian noise and Gaussian blur transformations.
- Shared-randomization transformation algorithm: each sample keeps the same
  brightness/rotation direction or base noise field across severity levels.
- Rotation v6 uses a declared zero-fill operator and a geometric valid-region
  mask. Prediction claims are operator-specific; CAM claims use forward
  alignment and exclude invalid support from every primary XAI metric.
- Validation-only, leaf-balanced image-space severity pilot with immutable
  per-sample metrics and an explicit human-approval gate.
- Optional PyTorch model wrappers for ResNet50 and EfficientNet-B0.
- Epoch-resumable Colab training with atomic latest/best checkpoints, complete
  optimizer/RNG state and protocol/manifest lineage enforcement.
- Optional CAM adapter for Grad-CAM, Grad-CAM++ and Score-CAM through `pytorch-grad-cam`.
- Runtime-approved CAM targets `layer4[-1]` (ResNet50) and `features[-1]`
  (EfficientNet-B0), pinned by `DR-XAI-001` evidence.
- Heatmap quality validation, masked Pearson/SSIM, top-k IoU at 0.1/0.2/0.3,
  secondary cosine, leaf-cluster bootstrap, paired Wilcoxon and Holm correction.
- Run provenance and artifact index helpers.
- Unit, integration and scientific invariant tests.

## Safe commands

```powershell
python -m pip install -e ".[hf,ml,xai,dev]"
python -m pytest
python -m plantxai_stability.cli validate-protocol configs/protocol/v0.9/protocol.yaml
python -m plantxai_stability.cli smoke configs/protocol/v0.9/protocol.yaml
```

Official training additionally requires a frozen dataset record whose protocol
hash exactly matches the G0B-frozen protocol. Official test evaluation remains
blocked until validation-selected checkpoints receive G1/G2 approval.

## Official-run prerequisites

1. Audit the pinned dataset revision (`9e97599868962bd0079b8db4b7f1efa9185fa1e7`) and verify that `leaf_id` is present and reliable.
2. Build and approve the canonical manifest.
3. Freeze leaf-safe train/validation/test splits.
4. Pilot and approve transformation severities.
5. Freeze the pre-training protocol and rebind the frozen dataset record (G0B).
6. Select and hash validation-approved model checkpoints (G1).
7. Approve checkpoints and explicitly unlock official test evaluation (G2).

No dataset count, accuracy, confidence interval, p-value or stability result is fabricated by this repository.
