"""Fit normal references and calibrate using only normal training images."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image

from .baseline import ROOT, MODEL_PATH, anomaly_scores
from .features import encode, backbone


def features_for(paths):
    rows = []
    for start in range(0, len(paths), 16):
        images = []
        for path in paths[start:start+16]:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        rows.append(encode(images))
    return np.concatenate(rows)


def split_normal(paths, seed=42):
    if len(paths) < 20:
        raise ValueError("Need at least 20 normal training images.")
    hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    if len(set(hashes)) != len(hashes):
        raise ValueError("Exact duplicate normal images found; resolve duplicates before splitting.")
    indices = np.random.default_rng(seed).permutation(len(paths))
    size = max(4, int(len(paths) * .2))
    return [paths[i] for i in indices[size:]], [paths[i] for i in indices[:size]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="MVTec bottle directory containing train/good and test")
    args = parser.parse_args()
    paths = sorted((args.dataset / "train/good").glob("*.png"))
    references, calibration = split_normal(paths)
    reference_features = features_for(references)
    calibration_scores = anomaly_scores(features_for(calibration), reference_features)
    threshold = float(np.quantile(calibration_scores, .99, method="higher"))
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(MODEL_PATH, references=reference_features, threshold=threshold)
    def manifest(paths):
        return [{"file": str(p.relative_to(args.dataset)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "category": "bottle",
              "method": "Frozen ResNet18 ImageNet features, L2 normalized; nearest-normal Euclidean distance",
              "weights": "ResNet18_Weights.IMAGENET1K_V1", "seed": 42, "threshold": threshold,
              "calibration": "99th percentile of held-out normal training images; not a probability or guarantee",
              "references": manifest(references), "normal_calibration": manifest(calibration),
              "model_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
              "device": str(backbone()[2]), "feature_dimensions": int(reference_features.shape[1]),
              "normal_calibration_scores": calibration_scores.tolist(),
              "test_data_used": False}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/fit-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Fitted {len(references)} reference images; calibrated on {len(calibration)} normals. Test set untouched.")
    print(MODEL_PATH)


if __name__ == "__main__":
    main()
