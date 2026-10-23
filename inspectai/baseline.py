"""Nearest-normal image-feature anomaly scoring."""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "artifacts/bottle-baseline.npz"


def anomaly_scores(features, references):
    features, references = np.asarray(features), np.asarray(references)
    if features.ndim != 2 or references.ndim != 2 or not len(references) or features.shape[1] != references.shape[1]:
        raise ValueError("Expected compatible feature matrices and nonempty references.")
    if not np.isfinite(features).all() or not np.isfinite(references).all():
        raise ValueError("Feature values must be finite.")
    return np.sqrt(np.maximum(((features[:, None, :] - references[None, :, :]) ** 2).sum(axis=2), 0)).min(axis=1)


def predict(image, model_path=None):
    from .features import encode
    path = Path(model_path) if model_path else MODEL_PATH
    if not path.exists():
        raise FileNotFoundError("No fitted bottle baseline. Download the dataset and run the fit command first.")
    with np.load(path, allow_pickle=False) as model:
        threshold = float(model["threshold"].item())
        references = model["references"]
        score = float(anomaly_scores(encode([image]), references)[0])
    return {"score": score, "threshold": threshold,
            "result": "Review for possible anomaly" if score > threshold else "Within baseline range",
            "category": "bottle"}
