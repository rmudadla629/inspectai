# Spatial patch experiment

This experiment keeps spatial ResNet18 layer2 features instead of globally pooling the image. It is a simple position-specific diagonal-Gaussian baseline, not a reproduction of full PaDiM or PatchCore. The existing upload app continues to use the original global-feature method.

## Fixed procedure

Use the same 168 normal references and 41 normal calibration images from the original fit manifest, verifying their hashes. The frozen backbone produces 128-channel, 28×28 spatial features after the existing ImageNet resize and center crop. At each spatial position, estimate channel means and population variances from normal references. Add a fixed 0.01 variance regularizer, then compute the square root of the mean standardized squared channel deviation. The image score is the maximum of its 784 spatial scores.

Layer, regularizer, maximum aggregation, and calibration rule were fixed before running this experiment, without a parameter sweep. The threshold is the higher-order 99th percentile of the 41 normal calibration scores: **2.8813557625**. No defect labels or masks fit the method or choose its threshold. This small calibration set provides no guaranteed false-alarm rate.

## Measured results

| Method | AUROC | Defects detected | Defects missed | False alarms |
| --- | --- | --- | --- | --- |
| Original global ResNet18 | 0.9873 | 61/63 | 2/63 | 1/20 |
| Spatial diagonal Gaussian | 0.9913 | 58/63 | 5/63 | 0/20 |

[Raw patch evaluation](reports/patch-evaluation-20261023T045431Z.json). Both methods use the same already-inspected 83 test images. This is development evidence, not a fresh independent evaluation. A higher AUROC does not imply better recall at the independently calibrated operating threshold. The patch method has fewer false alarms here but misses more defects. We retain the original app method and do not retune either threshold using these test results.

For the original errors, the patch method flags `broken_small/007.png` (2.9385) and accepts `good/006.png` (2.7990), but still misses `contamination/003.png` (2.3343). Fixing two individual examples does not offset the additional misses across the full set.

## Local heatmap review

```sh
.venv/bin/python -m inspectai.patches --dataset data/mvtec/bottle
open artifacts/patch-review.html
```

The command fits the separate patch artifact, records calibration scores, data hashes and artifact digest, evaluates all test images, and creates a local HTML review. It displays all patch-method errors plus fixed examples including the three original errors. Generated artifacts and embedded dataset images are excluded from Git.

Each pair shows the actual resized, center-cropped input and its 28×28 model score map enlarged to 224×224. Blue is lower and red higher **within each individual image**. Independent min–max color scaling makes contrast visible even in normal images; color intensity cannot be compared between images or interpreted as defect probability. Maps are coarse feature deviations, not segmentation masks. Cropping discards outer image content and spatial alignment assumptions can be sensitive to object movement.

Visual inspection shows high patch scores near the right rim in the small-break example, but this is qualitative evidence only. No pixel AUROC, overlap metric, or localization threshold has been evaluated. A visually plausible map does not establish reliable localization.

## Limits and next decision

No task-specific neural fine-tuning occurred. Channel independence, fixed position matching, generic pretrained features, and max-score sensitivity are limitations. Do not select the regularizer or layer against this test set and then call the result independently validated. Any subsequent model selection needs a defined development split and untouched evaluation data. No inference-speed claim is made for this experiment.
