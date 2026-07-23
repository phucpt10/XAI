# Implementation and experiment readiness

Date: 2026-07-24

## Implemented

- Protocol v0.9 advanced to approved G2 with fail-closed runtime authorization.
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
- Metadata-only G2 readiness report generator. It verifies both checkpoint and
  validation-audit trees, frozen split invariants and official-test identity
  counts while guaranteeing that no official-test image is decoded.
- Approved `DR-TEST-001` for one registered baseline and joint robustness/XAI
  campaign, with a separate metadata-only authorization verification required
  before official-test pixel access.
- Authorized official baseline evaluation completed for both checkpoints:
  ResNet50 accuracy 0.995275 / macro-F1 0.993991 (8 errors), and EfficientNet-B0
  accuracy 0.995865 / macro-F1 0.995467 (7 errors), over all 1,693 test samples.
- Model-method joint execution is transactionally resumable after Colab
  interruption. Every committed sample contains the full 12-scenario cross
  product; method parts are merged only after exact lineage, artifact hash,
  prediction equality and factorial-coverage checks.
- Approved `DR-RECOVERY-001` infrastructure-only recovery policy. It preserves
  the historical final-freeze SHA as logical lineage, records a distinct
  recovered physical-freeze SHA, independently verifies all 8,384 image hashes,
  prohibits baseline and completed ResNet50 Grad-CAM reruns, and authorizes
  only the five unfinished model-method parts through a fail-closed bridge.
- Heatmap quality gate, metrics, leaf-cluster bootstrap, paired Wilcoxon and Holm correction.
- Approved `DR-ANALYSIS-001` binds the exact ResNet50 and EfficientNet-B0
  merged-report hashes before final metric analysis. Its CPU-only runner
  enforces common paired sample keys, leaf-level inference, 10,000 bootstrap
  replicates, explicit Holm families, immutable outputs and a complete
  exclusion audit.
- Approved `DR-ANALYSIS-SUPPORT-001` binds metadata-only support audit
  `f370b3c...13ac`. Of 192 planned support contrasts, 191 meet the 20-leaf
  threshold. The single `Score-CAM x Gaussian blur severe` cross-model
  contrast has 14 common samples from 12 leaves and is retained as three
  non-estimable endpoint rows. The output remains 576 rows (573 estimable,
  three non-estimable), with conservative reserved Holm family slots.
- Official statistical analysis completed with all acceptance criteria passing.
  Report SHA `68a9b47f...294de` binds 576 paired rows: 573 estimable and the
  three approved non-estimable Score-CAM by Gaussian-blur-severe rows.
- Approved `DR-RESULTS-001` freezes that report plus its six child CSV hashes
  and authorizes a CPU-only, read-only reporting stage. The runner emits an
  exact allowlist of eight tables, six figures and two result summaries, then
  hashes every child artifact into an immutable reporting report.
- Joint prediction/explanation contracts, run provenance and artifact indexing.
- Unit, integration and scientific invariant tests.

## Validation performed

```text
ruff check src tests scripts    PASS
mypy src                        PASS
pytest                          PASS (90 tests; torch integration skipped when unavailable)
compileall                      PASS
protocol validation             PASS
scenario smoke                 PASS (12 scenarios)
constant heatmap gate          PASS
```

## Official experiment status

G2 governance and the runtime authorization gate both pass for the single
registered campaign. Official baseline evaluation, all six model-method joint
parts and both model merges are complete. Official training stays bound to the
final G0B freeze. A Colab storage loss removed the physical final freeze, but
the approved recovery path does not alter its logical scientific lineage. The
protocol has:

```text
status: frozen
frozen: true
G0A_BOOTSTRAP_READY: PASS
G0B_PROTOCOL_FREEZE_READY: PASS
G1_CHECKPOINT_SELECTION: PASS
G2_TEST_EVALUATION_READY: PASS
official_training_allowed: true
official_test_evaluation_allowed: true
official_experiment_allowed: true
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
reported evidence hashes. Official baseline predictions and all joint records
have been produced and merged. The predeclared statistical analysis completed
successfully, including 10,000-replicate leaf-cluster intervals, paired
Wilcoxon tests and fixed Holm families. Its verified report is frozen by
`DR-RESULTS-001`; publication tables, figures and summaries must now be
derived only from those frozen artifacts.

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

The recovery, all six model-method parts, both model merges, the metadata-only
192-contrast support audit and the final statistical analysis are complete.
The remaining work is staged as follows:

1. Run the `DR-RESULTS-001` CPU-only reporting generator against the exact
   `statistical-analysis-v2` directory.
2. Verify and preserve the generated reporting-report SHA-256.
3. Integrate the frozen tables and figures into the manuscript without
   reselection, tuning or interpreting non-estimable rows as non-significant.

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
