# Audit correction and re-execution plan (v1.0)

## Baseline preservation

`baseline-results-v1.0` remains the immutable reference for the original
results.  No result produced with protocol v0.9 may be overwritten, relabelled,
or merged with v1.0 artifacts.

## Corrected components

| Component | Correction | Affected evidence |
| --- | --- | --- |
| Gaussian blur | OpenCV Gaussian convolution with declared odd kernel, sigma, `BORDER_REFLECT_101`, and pinned OpenCV distribution. | All Gaussian-blur predictions and XAI metrics. |
| Top-k IoU | Exact `ceil(k*n_valid)` selection with deterministic row-major tie breaking. | All top-k IoU values, summaries, contrasts, and RQ3 rows. |
| Cluster summaries | Equal-leaf point estimate and leaf bootstrap. | Prediction and XAI summaries and every reported confidence interval. |
| Wilcoxon provenance | Explicit two-sided Pratt asymptotic test without continuity correction, with zero/tie diagnostics. | All paired-comparison tables. |
| CAM exclusion evidence | Original/transformed CAM status is emitted independently of prediction consistency. | XAI exclusion audit and Score-CAM quality claims. |

## Required sequence

1. Freeze a new v1.0 protocol only after new decision records approve the
   corrected operator, metric, estimand, target layers, and reporting schema.
2. Re-run the severity pilot for Gaussian blur.  Confirm monotonic severity
   under the v1.0 operator before any test-pixel access.
3. Run the Score-CAM/target-layer preflight on the predeclared validation set;
   publish sample-level CAM-status diagnostics.  Resolve any adapter defect
   before the official campaign.
4. Re-run the full joint evaluation for both fixed checkpoints, all twelve
   scenarios, and all three XAI methods.  No training or checkpoint selection
   is required solely by these evaluation/metric corrections.
5. Re-merge only v1.0 child artifacts and run the v1.0 statistical analysis.
   Recompute every summary, all paired contrasts, and RQ3 associations; do not
   splice v0.9 and v1.0 rows.
6. Review the new exclusion audit, top-k tie/cardinality diagnostics, Wilcoxon
   diagnostics, and equal-leaf versus image-weighted sensitivity table before
   making scientific claims.
7. Generate a separately versioned frozen-results report.  Revise claims only
   from v1.0 output and disclose the baseline supersession explicitly.

## Stop conditions

- Any failure of validation preflight, severity monotonicity, CAM quality, or
  protocol gate stops the campaign.
- A change to training, checkpoints, data split, or hyperparameters requires a
  separate decision record; it is not authorized by this correction plan.
- Reviewer files and manuscript drafts remain local-only and are never added
  to Git artifacts.
