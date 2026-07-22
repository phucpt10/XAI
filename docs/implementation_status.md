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
- Approved `DR-LEAF-002` policy that preserves all 1,693 official test samples
  and quarantines exactly five approved source-train overlap samples.
- Two-stage quarantine evidence: metadata adjudication followed by a finalized
  pixel-identified registry, all-sample lineage manifest and reconciliation gate.
- Approved `DR-DUP-001` policy for nine train-only, same-class, same-leaf exact
  duplicate pairs; it retains the minimum stable sample ID and quarantines the
  other nine samples without modifying official test.
- Image-level audit artifacts (`dataset_receipt.json`, `image_audit.parquet`),
  duplicate/label/leaf conflict detection and immutable freeze artifacts.
- Deterministic class-stratified leaf split construction and DataLoader hash,
  shape, range and identity checks.
- Manifest audit and leaf-safe split validation.
- Identity-preserving dataset adapter and optional PyTorch DataLoader.
- Deterministic four-transformation pipeline with twelve scenario labels.
- Optional ResNet50/EfficientNet-B0 wrappers and validation-only training API.
- Optional Grad-CAM, Grad-CAM++ and Score-CAM adapter.
- Approved runtime target layers: ResNet50 `layer4[-1]` with activation shape
  `1x2048x7x7`, and EfficientNet-B0 `features[-1]` with `1x1280x7x7`.
- Six passing CAM runtime checks across Grad-CAM, Grad-CAM++ and Score-CAM,
  recorded by `DR-XAI-001` without accessing official test.
- Passing deterministic DataLoader and two-backbone mixed-precision smoke
  evidence recorded by `DR-RUNTIME-001`.
- Colab-ready resumable training with epoch-boundary state, validation-selected
  best checkpoints, lineage validation and test-gated evaluation scripts.
- Heatmap quality gate, metrics, leaf-cluster bootstrap, paired Wilcoxon and Holm correction.
- Joint prediction/explanation contracts, run provenance and artifact indexing.
- Unit, integration and scientific invariant tests.

## Validation performed

```text
ruff check src tests scripts    PASS
mypy src                        PASS
pytest                          PASS (35 tests; torch integration skipped when unavailable)
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
G1_CHECKPOINT_SELECTION: BLOCKED
G2_TEST_EVALUATION_READY: BLOCKED
official_training_allowed: false
official_test_evaluation_allowed: false
official_experiment_allowed: false
```

The pinned metadata audit covers all 8,398 selected samples and found five
reconstructed leaf identities shared by upstream train and test (10 affected
samples). `DR-LEAF-001` records the failed raw-source gate. The project owner
approved `DR-LEAF-002`: preserve all 1,693 official test samples and quarantine
the five source-train counterparts, leaving 8,393 eligible modeling samples.
Colab completed the cumulative quarantine, yielding 8,384 eligible samples,
14 quarantined train samples and all 1,693 official test samples preserved.
Leaf-safe freeze, deterministic loading, both backbone smoke runs and all six
target-layer/CAM checks pass. `DR-RUNTIME-001` and `DR-XAI-001` bind the
reported evidence hashes. No model accuracy, confidence interval, p-value or
official XAI stability result has been produced.

The remaining work is staged to avoid circular governance:

1. G0B: visually approve transformation severity, freeze the protocol and
   regenerate the frozen data bundle against that final pre-training hash.
2. G1: train both backbones and select/hash checkpoints using validation only.
3. G2: approve both checkpoint records and only then unlock official test.

Severity pilot v1 passed its original numerical gate but failed human visual
review because brightness and rotation direction, and the Gaussian base-noise
field, were independently resampled at each severity. `DR-SEVERITY-001`
permanently rejects that run. The protocol now pins `shared_randomization_v2`;
pilot and visual-review evidence must be regenerated in new v2 directories.
