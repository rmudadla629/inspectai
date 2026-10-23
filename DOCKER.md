# Run InspectAI with Docker

Docker Desktop must be running. An account is not needed for this local build. The image installs pinned Python dependencies, so the first build needs internet and may take several minutes. Dataset pixels, fitted artifacts, and pretrained weights are not copied into the image.

## Prepare the local model

Use the dataset acquisition and fit steps in [README.md](README.md) first. The container needs these existing local paths:

- `artifacts/bottle-baseline.npz`: the fitted original baseline.
- `.cache/torch/hub/checkpoints/resnet18-f37072fd.pth`: pretrained ResNet18 weights.

The native fit command normally creates them. If you explicitly set `TORCH_HOME` elsewhere, place your verified cached weights at the path above for Compose. The container's cache is read-only; do not rely on an automatic download during an inspection request. No dataset directory is mounted.

## Start and open

From the project directory:

```sh
docker compose up --build -d
docker compose ps
```

Open http://127.0.0.1:8101/. This separate port leaves the native app on 8100 available. The service binds to localhost on your Mac. Upload the same bottle test images used in the native walkthrough.

```sh
docker compose logs --tail=50
docker compose down
```

The last command stops and removes this Compose service. It does not delete your local model, weights, or dataset.

## Behavior and limits

The container runs as an unprivileged user with a read-only root filesystem, temporary `/tmp`, dropped Linux capabilities, and read-only model/cache mounts. Uploads remain in memory. Do not use `down -v` or delete local data to troubleshoot a missing artifact.

Linux containers on a Mac use CPU inference; they do not use the host's PyTorch MPS acceleration. First inference may take longer while loading weights. Native and container floating-point results may differ slightly; inspect decisions near the threshold carefully. No container throughput or latency target is claimed.

The Docker health check tests HTTP responsiveness. `/health` separately reports whether the fitted artifact exists; neither test proves model quality, cache completeness, or successful inference. If a model is missing, inspection returns 503 rather than a fabricated score. A malformed artifact or missing cached weights requires setup repair; it is not a successful inspection.

Compose requires the mount directories to exist. Run native setup/fitting first; do not create an empty model file. This package is a local research demo, with no public hosting, production authentication, or arbitrary-product validation. Dataset terms still apply.

## CI

The container job builds without any dataset or model files, then checks the packaged home page, health response, and missing-model handling using Flask's test client. Local verification with mounted artifacts checks actual server startup and CPU inference separately. CI does not download the MVTec dataset or measure detection quality.

## Local verification

The Linux ARM64 image built successfully on Docker Desktop. With read-only model/cache mounts, actual HTTP uploads returned 200: normal `test/good/000.png` scored 0.1131 and defective `test/broken_large/000.png` scored 0.3091, matching the native decisions. See [saved smoke-check evidence](reports/container-smoke.json). These two samples do not replace the full dataset evaluation.
