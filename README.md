# PlantXAI-Stability

PlantXAI-Stability is research software for evaluating prediction robustness and XAI explanation stability under controlled image transformations.

The implementation follows the English specification in `PlantXAI-Stability_Research_Software_Specification_En.docx`. G2 authorizes one registered official-test campaign after validation-only checkpoint selection and metadata-only readiness review. Every runner remains fail-closed until the runtime authorization chain passes.

## Current implementation

- Versioned G2-approved protocol at `configs/protocol/v0.9/protocol.yaml`.
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
- Validation-only checkpoint audit with exact identity coverage, per-class
  precision/recall/F1, confusion matrices, probability records and immutable
  artifact hashes; official test pixels remain inaccessible before G2.
- Approved `DR-CHECKPOINT-001` registry for both validation-selected checkpoints,
  preserving their original G0B training lineage while governance advances.
- Metadata-only G2 readiness gate that reconciles both checkpoint registries,
  validation audits and all child artifact hashes while enumerating official
  test identities without decoding test images or computing test results.
- Approved `DR-TEST-001` registered-campaign authorization plus a metadata-only
  runtime verification command required before any official-test runner.
- Transactional official joint runner split by model and CAM method. It commits
  a complete 12-scenario result per sample to SQLite, cryptographically binds
  resume attempts to the identical protocol/checkpoint/manifest/code identity,
  caches each original CAM once per method and preserves explicit exclusions.
- Fail-closed joint merger that requires exact prediction agreement across the
  three method parts and complete sample x scenario x method coverage.
- Approved `DR-ANALYSIS-001` statistical plan and a CPU-only fail-closed
  analysis runner. It binds the exact two merged reports, resamples
  `leaf_id`, intersects paired sample identities before leaf aggregation,
  applies the predeclared Wilcoxon/Holm families and retains every exclusion
  in a separate audit table.
- Metadata-only analysis-support preflight covering all 192 planned model,
  CAM-method and within-transformation severity contrasts without reading
  endpoint metric columns or computing hypothesis tests.
- Infrastructure-only physical-freeze recovery governed by `DR-RECOVERY-001`.
  It keeps the historical freeze hash as logical lineage, records the recovered
  physical hash separately, re-verifies every manifest image and prevents
  recomputation of completed official results.
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
hash exactly matches its training protocol. G2 is approved; baseline and joint
runners still require the exact readiness report and both Decision Records. A
boolean protocol flag alone cannot authorize pixel access.

## Official-run prerequisites

1. Audit the pinned dataset revision (`9e97599868962bd0079b8db4b7f1efa9185fa1e7`) and verify that `leaf_id` is present and reliable.
2. Build and approve the canonical manifest.
3. Freeze leaf-safe train/validation/test splits.
4. Pilot and approve transformation severities.
5. Freeze the pre-training protocol and rebind the frozen dataset record (G0B).
6. Select, audit and approve both model checkpoints using validation only (G1; complete).
7. Reconcile final lineage and explicitly unlock official test evaluation (G2).
8. Merge both complete model result trees and run the predeclared
   `DR-ANALYSIS-001` statistical analysis without reopening image pixels.

No dataset count, accuracy, confidence interval, p-value or stability result is fabricated by this repository.
