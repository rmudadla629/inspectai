# First bottle baseline — October 22, 2026

Reports use UTC timestamps, so this local late-evening run is named October 23 in report files.

## Method and data

Frozen torchvision ResNet18 ImageNet features, global 512-dimensional pooled representation, L2 normalization, and Euclidean distance to the nearest normal reference. A high distance flags possible anomaly. This fits a reference bank and threshold; no neural-network fine-tuning occurred.

The 209 normal MVTec bottle training images were partitioned deterministically (seed 42): 168 references and 41 normal calibration images. A fixed threshold of **0.1366028786** was selected as the 99th-percentile calibration score using the higher-order statistic. With 41 calibration images this is the largest observed calibration score. It does not guarantee a 1% real-world false-positive rate.

The first test evaluation used all 83 official test images from the documented mirror: 20 good and 63 anomalous. Neither test labels nor test scores selected the threshold. Exact duplicate checks found no overlap between fitting/calibration and test images. Near-duplicate absence was not established. [Dataset provenance](reports/dataset-provenance.json), [fit manifest](reports/fit-manifest.json), [backbone provenance](reports/backbone-provenance.json).

## First test results

| Metric | Measured result |
| --- | --- |
| Image AUROC | 0.9873 |
| Defects correctly flagged | 61/63 |
| Defects missed | 2/63 |
| Good images correctly accepted | 19/20 |
| False alarms | 1/20 (5%) |
| Defect recall at fixed threshold | 96.83% |

[Raw image scores and metrics](reports/evaluation-20261023T042247Z.json). These numbers describe one category of a small public dataset, not a factory deployment or arbitrary-product accuracy. AUROC measures ranking across thresholds; operational recall and false alarms above use the already fixed calibration threshold.

## Observed mistakes

| Image | Error | Score |
| --- | --- | --- |
| test/broken_small/007.png | Defective image below threshold | 0.134995 |
| test/contamination/003.png | Defective image below threshold | 0.124177 |
| test/good/006.png | Good image above threshold | 0.186343 |

The scores are recorded evidence, not calibrated probabilities. The app should say “Within baseline range,” not certify an item as defect-free. A high score requests review; it does not establish a particular defect type. No localization heatmap is implemented.

## Reproducibility and next experiments

The fit artifact and pretrained weights remain local and excluded from Git. Manifests record content hashes and model provenance so another user can reproduce fitting with the documented dataset copy. MPS floating-point/device differences may alter exact scores; bit-for-bit reproduction across hardware is not guaranteed.

Keep this first result and its fixed threshold as the baseline. Future parameter/model selection needs a development-validation plan; repeatedly tuning to these observed test errors would contaminate claims of independent test performance. Planned comparisons should report both quality and inference cost, include a simpler baseline, and retain false-positive/false-negative analysis. A separate unseen dataset or reserved split is needed for stronger generalization claims.

Dataset rights remain CC BY-NC-SA 4.0. No dataset pixels are distributed by this repository and no commercial suitability is claimed.

## Subsequent comparison

The original artifact and threshold were retained for a [grayscale-feature comparison](COMPARISON.md). On the same already-inspected test split, pixel features detected 55/63 defects with two false alarms; ResNet18 reproduced 61/63 with one false alarm. This is additional comparative development evidence, not a second independent test set.

## Visual error review

[Error analysis](ERROR_ANALYSIS.md) reviews all three mistakes, separates visual observations from untested hypotheses, and provides a generator for a local image-and-mask report. The original model and threshold remain unchanged.
