"""Evaluate fitted bottle baseline without choosing a threshold on test data."""
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
from .baseline import ROOT, MODEL_PATH, anomaly_scores
from .train import features_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((ROOT / "reports/fit-manifest.json").read_text())
    model_digest = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    if model_digest != manifest["model_sha256"]:
        raise ValueError("Fitted artifact does not match the fit manifest.")
    used = {r["sha256"] for r in manifest["references"] + manifest["normal_calibration"]}
    paths = sorted((args.dataset / "test").glob("*/*.png"))
    labels = [int(p.parent.name != "good") for p in paths]
    if len(set(labels)) != 2:
        raise ValueError("Test set must contain both good and anomalous images.")
    if any(hashlib.sha256(p.read_bytes()).hexdigest() in used for p in paths):
        raise ValueError("Test image duplicates a fitting/calibration image.")
    with np.load(MODEL_PATH, allow_pickle=False) as model:
        scores = anomaly_scores(features_for(paths), model["references"])
        threshold = float(model["threshold"].item())
    predictions = scores > threshold
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel().tolist()
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "category": "bottle",
              "image_auroc": float(roc_auc_score(labels, scores)), "threshold": threshold,
              "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
              "defect_recall": tp / (tp + fn), "false_positive_rate": fp / (fp + tn),
              "model_sha256": model_digest,
              "test_count": len(paths),
              "results": [{"file": str(p.relative_to(args.dataset)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                           "label": y, "score": float(s)} for p, y, s in zip(paths, labels, scores)],
              "limitations": "Image-level score only; no localization, probability calibration, or real factory validation. Test results must not be used to retune this threshold and still called held-out."}
    output = ROOT / "reports" / ("evaluation-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("image_auroc", "confusion", "defect_recall", "false_positive_rate")}, indent=2))
    print(output)


if __name__ == "__main__":
    main()
