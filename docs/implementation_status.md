# Implementation and experiment readiness

Date: 2026-07-23

## Implemented

- Protocol v0.9 advanced to approved G1 with fail-closed G2 validation.
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
- Validation-only G1 checkpoint audit with checkpoint/training/freeze lineage,
  exact sample coverage, per-class metrics, confusion matrices, NLL, Brier
  score, deterministic prediction records and immutable hashes.
- Approved `DR-CHECKPOINT-001` registry for both predeclared backbones. It binds
  checkpoint, training evidence, validation prediction and metric artifact
  hashes while preserving the immutable G0B training-protocol lineage.
- Heatmap quality gate, metrics, leaf-cluster bootstrap, paired Wilcoxon and Holm correction.
- Joint prediction/explanation contracts, run provenance and artifact indexing.
- Unit, integration and scientific invariant tests.

## Validation performed

```text
ruff check src tests scripts    PASS
mypy src                        PASS
pytest                          PASS (53 tests; torch integration skipped when unavailable)
compileall                      PASS
protocol validation             PASS
scenario smoke                 PASS (12 scenarios)
constant heatmap gate          PASS
```

## Official experiment status

Official test evaluation is intentionally blocked. Official training is
authorized only with a freeze record matching the final G0B hash. The protocol
currently has:

```text
status: frozen
frozen: true
G0A_BOOTSTRAP_READY: PASS
G0B_PROTOCOL_FREEZE_READY: PASS
G1_CHECKPOINT_SELECTION: PASS
G2_TEST_EVALUATION_READY: BLOCKED
official_training_allowed: true
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
reported evidence hashes. No official-test accuracy, confidence interval,
p-value or XAI stability result has been produced.

`DR-SEVERITY-006` binds the approved v6 pilot and four visual-review artifacts.
Its outcome is `PASS_WITH_DECLARED_OPERATOR_LIMITATION`: severity is ordinal
only within each transformation; rotation prediction claims are specific to
the zero-filled operator; CAM stability requires forward alignment and M_T.
The two G1 checkpoints retain the final G0B training hash
`7eb0814be8ffc1a19f54e2bec2d2ca0c84d7f4d869d99e28b69e6c9e0e84523b`.
`DR-CHECKPOINT-001` is the explicit lineage bridge from that immutable training
configuration to the later G1 governance state; checkpoint files are not
rewritten or re-signed. The resulting G1 governance protocol hash is
`88440f4e740a707e128cb80d0680dca4f38c104388f51d9493b5b1adb76affe9`.

The remaining work is staged as follows:

1. Bind the final test-evaluation freeze/governance lineage without changing
   the selected checkpoint bytes or scientific configuration.
2. G2: approve the one-time official-test evaluation and explicitly unlock it.
3. Run baseline and joint prediction/XAI evaluation only after every G2 gate passes.

Severity pilot v1 passed its original numerical gate but failed human visual
review because brightness and rotation direction, and the Gaussian base-noise
field, were independently resampled at each severity. `DR-SEVERITY-001`
permanently rejects that run. The protocol now pins `shared_randomization_v2`;
pilot v2 corrected shared randomization, but its rotation contact sheet exposed
severity-correlated black corners. `DR-SEVERITY-002` rejects the complete v2
set. The v3 border-median replacement still produced uniform corner polygons;
`DR-SEVERITY-003` rejects that run. V4 removed uniform fill but repeated leaf
fragments through reflection, so `DR-SEVERITY-004` rejects it. Telea v5 passed
technical checks but produced severity-correlated radial smearing.
`DR-SEVERITY-005` replaces it with
`shared_randomization_zero_fill_valid_mask_v6`. Rotation prediction results are
operator-specific; explanation stability uses forward alignment and geometric
M_T with masked Pearson/SSIM and top-k IoU. The v6 technical and visual evidence
passed and is approved by `DR-SEVERITY-006`.
