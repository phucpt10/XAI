# Implementation and experiment readiness

Date: 2026-07-22

## Implemented

- Protocol v0.9 draft and fail-closed validation.
- Stable sample identity and canonical RGB hashing.
- Hugging Face adapter for `mohanty/PlantVillage` (`color`) with schema
  inspection, explicit `leaf_id` checks and optional manifest materialisation.
- Source-compatible filename reconstruction with explicit
  `filename_reconstructed` provenance, comparison against `leaf-map.json`, and
  ambiguity/collision/class-conflict/train-test-overlap gates.
- Immutable leaf-identity evidence (`leaf_identity_resolution_report.parquet`
  and its hashed JSON summary) plus a manifest governance check.
- Image-level audit artifacts (`dataset_receipt.json`, `image_audit.parquet`),
  duplicate/label/leaf conflict detection and immutable freeze artifacts.
- Deterministic class-stratified leaf split construction and DataLoader hash,
  shape, range and identity checks.
- Manifest audit and leaf-safe split validation.
- Identity-preserving dataset adapter and optional PyTorch DataLoader.
- Deterministic four-transformation pipeline with twelve scenario labels.
- Optional ResNet50/EfficientNet-B0 wrappers and validation-only training API.
- Optional Grad-CAM, Grad-CAM++ and Score-CAM adapter.
- Colab-ready training, baseline evaluation and single-model joint evaluation scripts.
- Heatmap quality gate, metrics, leaf-cluster bootstrap, paired Wilcoxon and Holm correction.
- Joint prediction/explanation contracts, run provenance and artifact indexing.
- Unit, integration and scientific invariant tests.

## Validation performed

```text
ruff check src tests scripts    PASS
mypy src                        PASS
pytest                          PASS (20 tests)
compileall                      PASS
protocol validation             PASS
scenario smoke                 PASS (12 scenarios)
constant heatmap gate          PASS
```

## Official experiment status

The official runner is intentionally blocked. The protocol currently has:

```text
status: draft
frozen: false
G0A_BOOTSTRAP_READY: PASS
G0B_PROTOCOL_FREEZE_READY: BLOCKED
official_experiment_allowed: false
```

The pinned metadata audit covers all 8,398 selected samples, but it found five
reconstructed leaf identities shared by upstream train and test (10 affected
samples). `DR-LEAF-001` is therefore rejected and manifest creation remains
blocked. No model accuracy, confidence interval, p-value or XAI stability
result has been produced. The following evidence is still required before a
scientific run:

1. Audit the pinned dataset revision (`9e97599868962bd0079b8db4b7f1efa9185fa1e7`).
2. Resolve the five upstream train/test leaf overlaps through a reviewed
   protocol change or stronger identity evidence.
3. Build and approve the canonical manifest.
4. Freeze leaf-safe train/validation/test splits.
5. Pilot and approve transformation severity.
6. Train/select and hash validation-approved checkpoints.
7. Runtime-validate target layers and CAM dependencies.
8. Freeze the protocol through a reviewed governance change.
