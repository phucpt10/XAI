# Scientific claim scope

## Rotation prediction robustness

Rotation uses a deterministic, constant-zero fill operator. Prediction results
therefore estimate robustness to the declared zero-filled rotation operator;
they are not evidence of pure physical rotation invariance. The invalid-support
fraction is reported for every severity.

## Rotation explanation stability

For prediction-consistent pairs, the original CAM is forward-aligned into the
transformed frame using the same angle. Pearson, SSIM and top-k IoU are computed
only on geometric valid support M_T. SSIM uses an eroded M_T so every local
window remains within valid support. Primary top-k IoU uses k=0.2; k=0.1 and
k=0.3 are sensitivity analyses. Both CAMs use the original predicted class.

M_T is an image-support mask. It is not a leaf-segmentation mask and does not
restrict metrics to biological foreground alone. Masking prevents invalid
padding pixels from directly inflating similarity metrics, but it does not
prevent zero-filled corners from influencing model predictions or internal
activations.
