"""Deterministic grayscale pixel features, without neural-network weights."""
import numpy as np
from PIL import Image, ImageOps


def encode_pixels(images):
    rows = []
    for image in images:
        thumbnail = ImageOps.fit(image.convert("L"), (32, 32), method=Image.Resampling.BILINEAR)
        vector = np.asarray(thumbnail, dtype=np.float32).reshape(-1) / 255.0
        norm = np.linalg.norm(vector)
        rows.append(vector / norm if norm > 0 else vector)
    if not rows:
        raise ValueError("At least one image is required.")
    return np.stack(rows)
