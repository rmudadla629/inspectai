# InspectAI

Visual anomaly inspection for a single industrial product category. The first baseline uses frozen ResNet18 ImageNet features and distance to normal bottle-image references.

[Screenshot walkthrough](WALKTHROUGH.md) · [Measured results](RESULTS.md) · [Docker setup](DOCKER.md) · [Failure analysis](ERROR_ANALYSIS.md)

## Results and scope

First bottle baseline fitted and evaluated locally on the M3 GPU. The frozen ResNet18 feature model uses 168 normal reference images and 41 separate normal calibration images. On the first 83-image test evaluation it detected 61/63 defects, missed 2, and flagged 1/20 good images. Image AUROC was 0.9873. See [results and limitations](RESULTS.md) and the [raw report](reports/evaluation-20261023T042247Z.json).

The upload endpoint was implemented with explicit missing-model and invalid-file handling. A [comparison with grayscale pixel features](COMPARISON.md) found 55/63 versus 61/63 defects detected at separately calibrated thresholds; the simpler method was faster in a small local timing sample. An [experimental spatial method](PATCH_EXPERIMENT.md) adds coarse local heatmaps but has not established localization accuracy; Docker packaging is available for local use; public deployment remains future work. This baseline fits references and a threshold; it does not fine-tune the pretrained neural network.

## Request flow

Upload PNG/JPEG → validate size and decoding → frozen ResNet18 features → nearest normal-reference distance → compare with the normal-calibrated threshold → return a review message.

The separate patch experiment produces coarse spatial maps in a local report; it does not replace this upload flow.

## Setup

Python 3.12 on Apple Silicon. From the project directory:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m inspectai
```

Open http://127.0.0.1:8100/. PNG/JPEG uploads are processed in memory, limited to 8 MiB and decoded image-size checks. The app stays on localhost.

## Docker

See [Docker setup](DOCKER.md) for a CPU container on localhost port 8101. Prepare the local model and cached weights first; Compose mounts them read-only.

## Dataset

Use the bottle category from [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad). The official page requires its download form; its bottle link returned 404 during setup. A pinned third-party mirror was used, with per-file integrity verification and source attribution. The dataset is CC BY-NC-SA 4.0, restricted to non-commercial use under its stated terms. Keep the dataset local; this repository does not distribute the images or grant dataset rights. See [dataset notes](DATASET.md).

Place the extracted category so that `data/mvtec/bottle/train/good/` and `data/mvtec/bottle/test/` exist. Inspect archives before extraction; keep their contents inside the intended data directory.

## Fit and evaluate

```sh
.venv/bin/python -m inspectai.train --dataset data/mvtec/bottle
.venv/bin/python -m inspectai.evaluate --dataset data/mvtec/bottle
```

Fitting downloads official ResNet18 pretrained weights into the local project cache (unless TORCH_HOME is already configured). The backbone stays frozen: this fits a reference bank and threshold, not a fine-tuned neural network. The first run may need internet for those weights.

The normal training split is deterministically partitioned into 80% reference images and 20% normal calibration images (seed 42). A threshold is chosen from the 99th percentile of calibration distances. This is not a probability or a promised false-positive rate. Exact duplicates are rejected before partitioning. The official test set is not used to choose the threshold. Evaluation checks exact train/test overlap and reports image AUROC, defect recall, false-positive rate, and confusion counts.

Do not repeatedly tune using test results and continue calling that split held out. Near duplicates and real-world dataset shifts need further investigation. The initial model scores whole images and cannot localize defects or reliably inspect arbitrary objects. A below-threshold score does not certify a product as defect-free.

## Validation

```sh
.venv/bin/python -m unittest discover -s tests -v
```

Tests exercise score behavior, feature validation, upload errors, missing-model behavior, and disjoint deterministic splits. Synthetic images in software tests are not a defect detection benchmark.

## Continuous integration

The `Tests` GitHub Actions workflow runs on pushes and pull requests, with an optional manual trigger. It installs CPU-only PyTorch and pinned dependencies under Python 3.12 on Ubuntu, checks package compatibility, and runs thirteen tests. No dataset, fitted artifact, GPU, or pretrained-weight download is required by these tests. The missing-model test also verifies that the backbone is not loaded.

This validates software behavior, not defect-detection quality or MPS execution. The separate local dataset evaluation and saved reports provide the measured baseline results. Both software and container checks passed on GitHub Actions during development. The container job builds without model/data files and checks the packaged app and missing-model response. See [Docker details](DOCKER.md).

## Version 1 coverage

1. Completed: acquire the bottle dataset and record mirror provenance and attribution.
2. Completed: fit and evaluate the frozen-feature baseline with explicit data splits.
3. Completed: compare against grayscale pixel features and report missed defects, false alarms, and timing limits. A [visual error analysis](ERROR_ANALYSIS.md) documents both missed defects and the false alarm, with a locally generated image review.
4. Completed experiment: spatial features and local heatmaps; measured tradeoffs retain the original app baseline. Localization accuracy remains unvalidated.
5. Completed: software CI and local Docker packaging with verified CPU uploads. The [reviewed screenshot walkthrough](WALKTHROUGH.md) completes the local demo documentation.

Repository: [rmudadla629/inspectai](https://github.com/rmudadla629/inspectai). Dataset images, pretrained weights, fitted artifacts, and local caches are excluded from Git. Reproduce the model locally using the documented data acquisition and fitting commands.
