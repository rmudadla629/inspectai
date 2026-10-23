"""Compare frozen ResNet18 and grayscale pixels without test-based threshold tuning."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score

from .baseline import ROOT, MODEL_PATH, anomaly_scores
from .simple_features import encode_pixels


def read_image(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def extract(paths, encoder):
    return np.concatenate([encoder([read_image(p) for p in paths[start:start+16]])
                           for start in range(0, len(paths), 16)])


def metrics(labels, scores, threshold):
    tn, fp, fn, tp = confusion_matrix(labels, scores > threshold, labels=[0, 1]).ravel().tolist()
    return {"image_auroc": float(roc_auc_score(labels, scores)),
            "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            "defect_recall": tp / (tp + fn), "false_positive_rate": fp / (fp + tn)}


def timing(images, encoder, references, repeats=3):
    # Predecoded in-memory images; encoders include preprocessing and feature extraction.
    anomaly_scores(encoder([images[0]]), references)
    durations = []
    for _ in range(repeats):
        for image in images:
            start = perf_counter()
            anomaly_scores(encoder([image]), references)
            durations.append((perf_counter() - start) * 1000)
    return {"sample_images": len(images), "repetitions": repeats,
            "mean_ms": float(np.mean(durations)), "median_ms": float(np.median(durations)),
            "p95_ms": float(np.quantile(durations, .95)),
            "scope": "Warm single-image preprocessing, features, CPU transfer where applicable, and reference scoring; excludes disk decode, model loading, network, and HTTP overhead."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    dataset = args.dataset
    fit_path = ROOT / "reports/fit-manifest.json"
    fit = json.loads(fit_path.read_text())
    if hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() != fit["model_sha256"]:
        raise ValueError("Original model artifact does not match the fit manifest.")
    groups = []
    for key in ("references", "normal_calibration"):
        paths = []
        for row in fit[key]:
            path = dataset / row["file"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError("Reference/calibration dataset changed.")
            paths.append(path)
        groups.append(paths)
    reference_paths, calibration_paths = groups
    used = {row["sha256"] for row in fit["references"] + fit["normal_calibration"]}
    test_paths = sorted((dataset / "test").glob("*/*.png"))
    labels = np.array([int(path.parent.name != "good") for path in test_paths])
    if set(labels) != {0, 1}:
        raise ValueError("Both good and anomalous test images are required.")
    hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in test_paths]
    if used.intersection(hashes):
        raise ValueError("Test data duplicates fit/calibration data.")

    # Only normal reference and calibration data choose the simple model threshold.
    pixels = extract(reference_paths, encode_pixels)
    calibration_scores = anomaly_scores(extract(calibration_paths, encode_pixels), pixels)
    pixel_threshold = float(np.quantile(calibration_scores, .99, method="higher"))
    simple_path = ROOT / "artifacts/bottle-pixels.npz"
    np.savez_compressed(simple_path, references=pixels, threshold=pixel_threshold)

    from .features import encode, backbone
    with np.load(MODEL_PATH, allow_pickle=False) as model:
        neural_references = model["references"].copy()
        neural_threshold = float(model["threshold"].item())
    indices = np.random.default_rng(2026).choice(len(test_paths), size=min(10, len(test_paths)), replace=False)
    timing_images = [read_image(test_paths[i]) for i in indices]
    results = {}
    for name, encoder, references, threshold, device in [
        ("grayscale_pixels", encode_pixels, pixels, pixel_threshold, "cpu"),
        ("resnet18", encode, neural_references, neural_threshold, str(backbone()[2])),
    ]:
        scores = anomaly_scores(extract(test_paths, encoder), references)
        results[name] = {**metrics(labels, scores, threshold), "threshold": threshold,
                         "feature_dimensions": int(references.shape[1]), "device": device,
                         "timing": timing(timing_images, encoder, references),
                         "scores": scores.tolist()}
        print(name, json.dumps({k: results[name][k] for k in ("image_auroc", "confusion", "defect_recall", "false_positive_rate")}), flush=True)
    report = {"created_at": datetime.now(timezone.utc).isoformat(),
              "platform": platform.platform(), "processor_architecture": platform.machine(),
              "versions": {p: importlib.metadata.version(p) for p in ("torch", "torchvision", "numpy", "pillow")},
              "fit_manifest_sha256": hashlib.sha256(fit_path.read_bytes()).hexdigest(),
              "original_model_sha256": fit["model_sha256"],
              "simple_model_sha256": hashlib.sha256(simple_path.read_bytes()).hexdigest(),
              "method": "Both nearest-normal reference distance; ResNet18 512-D frozen features vs 32x32 L2-normalized grayscale pixels, 1024-D. Simple design fixed before this comparison run.",
              "reference_count": len(reference_paths), "calibration_count": len(calibration_paths),
              "calibration_rule": "99th percentile of normal calibration scores, higher order statistic, separately for each feature representation",
              "test": [{"file": str(p.relative_to(dataset)), "label": int(y), "sha256": digest} for p, y, digest in zip(test_paths, labels, hashes)],
              "timing_sample_files": [str(test_paths[i].relative_to(dataset)) for i in indices],
              "results": results,
              "limitations": "Same already-inspected bottle test set; not a new independent test. No thresholds tuned on test labels. Timings are local warm measurements, not end-to-end latency or controlled cross-device efficiency. Tiny single-category corpus; no localization or real manufacturing claims."}
    output = ROOT / "reports" / ("comparison-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(output, flush=True)


if __name__ == "__main__":
    main()
