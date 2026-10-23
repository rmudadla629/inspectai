from io import BytesIO
from html import escape
from flask import Flask, request, Response
from PIL import Image, UnidentifiedImageError
from .baseline import predict, MODEL_PATH

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 20_000_000


def page(message=""):
    ready = "Fitted bottle baseline available" if MODEL_PATH.exists() else "Setup stage: baseline not fitted yet"
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>InspectAI</title><body style="max-width:760px;margin:60px auto;padding:24px;font:18px/1.6 system-ui;color:#16243b"><h1>InspectAI</h1><p>Visual anomaly inspection · bottle baseline</p><p>{ready}</p><form action="/inspect" method="post" enctype="multipart/form-data"><label for="image">Inspection image (PNG or JPEG, up to 8 MiB)</label><br><input id="image" type="file" name="image" accept="image/png,image/jpeg" required><button>Inspect image</button></form><section role="status">{message}</section><p>Research prototype. Scores are not probabilities. This baseline has no defect localization and is not validated for arbitrary products. Uploaded images are processed in memory and not saved.</p></body></html>'''


@app.get("/")
def home():
    return page()


@app.post("/inspect")
def inspect():
    file = request.files.get("image")
    if file is None:
        return page("<p>Please select an image.</p>"), 400
    try:
        raw = file.read()
        with Image.open(BytesIO(raw)) as image:
            if image.format not in ("PNG", "JPEG"):
                raise ValueError("Use PNG or JPEG.")
            if image.width * image.height > 20_000_000:
                raise ValueError("Image exceeds decoded size limit.")
            image.load()
            result = predict(image.convert("RGB"))
    except FileNotFoundError as error:
        return page(f"<p>{escape(str(error))}</p>"), 503
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        return page("<p>The file could not be processed. Use a valid PNG/JPEG within image size limits.</p>"), 400
    return page(f'<h2>{escape(result["result"])}</h2><p>Anomaly score: {result["score"]:.4f} · Threshold: {result["threshold"]:.4f}</p>')


@app.get("/health")
def health():
    return {"status": "ok", "baseline_fitted": MODEL_PATH.exists()}


@app.errorhandler(413)
def too_large(error):
    return page("<p>Image upload exceeds 8 MiB.</p>"), 413


@app.after_request
def headers(response):
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response
