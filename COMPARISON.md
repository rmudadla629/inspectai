# Frozen features versus grayscale pixels

## Experiment

The simple method was specified before this comparison run: center-fit an image to 32×32 grayscale pixels, flatten to 1,024 dimensions, and L2-normalize. Both methods use nearest-normal Euclidean distance. ResNet18 uses its original frozen 512-dimensional features and unchanged fitted reference bank/threshold.

The grayscale method uses the **same 168 reference images and 41 normal calibration images** recorded in the original fit manifest. Its threshold is the 99th percentile of its own normal calibration scores using the higher order statistic. No test labels or scores select either threshold. The report verifies original artifact and training/calibration image hashes before scoring.

Both methods were evaluated on the same 83 bottle test images (63 defects, 20 good). This split was already inspected in the first baseline run, so this is a comparative development benchmark, not a newly independent test. No threshold was adjusted after viewing this comparison.

## Results

| Measurement | Grayscale pixels | Frozen ResNet18 |
| --- | --- | --- |
| Image AUROC | 0.9516 | 0.9873 |
| Defects flagged | 55/63 | 61/63 |
| Missed defects | 8/63 | 2/63 |
| False alarms | 2/20 (10%) | 1/20 (5%) |
| Defect recall | 87.30% | 96.83% |
| Warm mean processing time | 0.51 ms | 12.14 ms |

[Raw report, scores, timings, and provenance](reports/comparison-20261023T044311573325Z.json).

For these fixed methods and thresholds, pretrained features detected six additional defective images and produced one fewer false alarm. The grayscale baseline remains useful as a fast, transparent reference. This comparison supports retaining ResNet18 as the app's baseline for now; it does not establish statistical significance, general superiority across products, or factory suitability.

## Timing scope

On the local M3 Mac, each method processed the same 10 deterministically sampled, predecoded test images three times, after one warm-up. Pixel features used CPU; ResNet18 used MPS with outputs transferred to CPU before scoring. Durations include preprocessing, feature extraction, and reference-distance scoring; they exclude image decoding, model initialization, web requests, and network delivery. Each method was measured sequentially, not in interleaved controlled trials. The small sample and different execution devices limit speed conclusions. These are not end-to-end application latencies or memory/energy measurements.

## Reproduce

After acquiring the dataset and fitting the original baseline:

```sh
.venv/bin/python -m inspectai.compare --dataset data/mvtec/bottle
```

The command writes a separate local pixel artifact and timestamped comparison report. It does not overwrite the original ResNet18 artifact or change the app's default model. Fitted artifacts and dataset images remain excluded from GitHub. Future model/parameter selection needs a reserved validation plan and fresh data for stronger final evaluation claims.
