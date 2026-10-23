"""Experimental spatial diagonal-Gaussian features, not a full PaDiM implementation."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from io import BytesIO
import base64
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score
from .features import backbone
from .baseline import ROOT

PATCH_PATH = ROOT / 'artifacts/bottle-patches.npz'


def patch_features(paths):
    model, transform, device = backbone()
    outputs = []
    for start in range(0, len(paths), 8):
        images = []
        for path in paths[start:start+8]:
            with Image.open(path) as image:
                images.append(transform(image.convert('RGB')))
        with torch.inference_mode():
            x = torch.stack(images).to(device)
            x = model.maxpool(model.relu(model.bn1(model.conv1(x))))
            x = model.layer2(model.layer1(x))
        outputs.append(x.cpu().numpy().astype(np.float32))
    return np.concatenate(outputs)


def score_maps(features, mean, variance):
    if features.ndim != 4 or features.shape[1:] != mean.shape or mean.shape != variance.shape:
        raise ValueError('Incompatible spatial features.')
    if not all(np.isfinite(x).all() for x in (features, mean, variance)) or (variance <= 0).any():
        raise ValueError('Finite features and positive variances required.')
    return np.sqrt(np.mean((features - mean) ** 2 / variance, axis=1))


def heatmap_png(values):
    """Per-image display scaling only: these colors cannot be compared across images."""
    low, high = float(values.min()), float(values.max())
    scaled = (values-low) / max(high-low, 1e-12)
    rgb = np.stack([scaled, np.zeros_like(scaled), 1-scaled], axis=-1)
    image = Image.fromarray((rgb*255).astype('uint8')).resize((224,224), Image.Resampling.BILINEAR)
    output = BytesIO(); image.save(output, format='PNG')
    return base64.b64encode(output.getvalue()).decode('ascii')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    args = parser.parse_args()
    manifest_path = ROOT / 'reports/fit-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    def paths_for(rows):
        paths = []
        for row in rows:
            path = (args.dataset / row['file']).resolve()
            if not path.is_relative_to(args.dataset.resolve()):
                raise ValueError('Path escapes dataset.')
            if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError('Dataset content changed.')
            paths.append(path)
        return paths
    reference = patch_features(paths_for(manifest['references']))
    mean = reference.mean(axis=0)
    # Fixed regularization chosen before this experiment; never selected on test scores.
    variance = reference.var(axis=0) + 0.01
    del reference
    calibration = score_maps(patch_features(paths_for(manifest['normal_calibration'])), mean, variance).max(axis=(1,2))
    threshold = float(np.quantile(calibration, .99, method='higher'))
    np.savez_compressed(PATCH_PATH, mean=mean, variance=variance, threshold=threshold)
    paths = sorted((args.dataset / 'test').glob('*/*.png'))
    used = {r['sha256'] for r in manifest['references'] + manifest['normal_calibration']}
    digests = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    if used.intersection(digests):
        raise ValueError('Test image overlaps normal fitting data.')
    maps = score_maps(patch_features(paths), mean, variance)
    scores = maps.max(axis=(1,2))
    labels = [int(p.parent.name != 'good') for p in paths]
    tn, fp, fn, tp = confusion_matrix(labels, scores>threshold, labels=[0,1]).ravel().tolist()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    report = {'created_at': datetime.now(timezone.utc).isoformat(), 'method':'ResNet18 layer2 spatial diagonal Gaussian, variance regularizer 0.01, maximum patch score',
        'feature_shape':list(mean.shape), 'reference_count':len(manifest['references']), 'calibration_count':len(manifest['normal_calibration']),
        'threshold':threshold, 'calibration_scores':calibration.tolist(), 'image_auroc':float(roc_auc_score(labels,scores)),
        'confusion':dict(tn=tn,fp=fp,fn=fn,tp=tp), 'defect_recall':tp/(tp+fn), 'false_positive_rate':fp/(fp+tn),
        'device':str(backbone()[2]), 'fit_manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'artifact_sha256':hashlib.sha256(PATCH_PATH.read_bytes()).hexdigest(),
        'results':[dict(file=str(p.relative_to(args.dataset)),sha256=h,label=y,score=float(s)) for p,h,y,s in zip(paths,digests,labels,scores)],
        'limitations':'Already inspected test set: development evidence, not fresh independent evaluation. No pixel-level localization metric. Heatmaps show coarse feature distance, not verified defect boundaries. No hyperparameters selected on test scores.'}
    report_path = ROOT / f'reports/patch-evaluation-{stamp}.json'
    report_path.write_text(json.dumps(report,indent=2)+'\n')
    # Local pixels only; artifacts directory is ignored by Git.
    from torchvision.transforms import functional as TF
    selected = [i for i,(p,y,s) in enumerate(zip(paths,labels,scores)) if bool(s>threshold)!=bool(y) or str(p.relative_to(args.dataset)) in ('test/broken_small/007.png','test/contamination/003.png','test/good/006.png','test/good/000.png','test/broken_large/000.png')]
    cards = []
    for i in selected:
        with Image.open(paths[i]) as original:
            crop = TF.center_crop(TF.resize(original.convert('RGB'),256),[224,224])
        output=BytesIO(); crop.save(output,format='PNG')
        source=base64.b64encode(output.getvalue()).decode('ascii')
        cards.append(f'<section><h2>{paths[i].relative_to(args.dataset)}</h2><p>Dataset label: {paths[i].parent.name}; score {scores[i]:.4f}; threshold {threshold:.4f}</p><img alt="Model input crop" src="data:image/png;base64,{source}"><img alt="Coarse anomaly score map" src="data:image/png;base64,{heatmap_png(maps[i])}"></section>')
    (ROOT / 'artifacts/patch-review.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>InspectAI patch review</title><style>body{font:17px system-ui;max-width:900px;margin:40px auto}section{border-top:1px solid #ccc;padding:20px}img{width:224px;margin:10px}</style><h1>Experimental patch review</h1><p>Left: resized, center-cropped model input. Right: 28×28 anomaly map enlarged for display. Blue is lower and red higher within each image; colors are independently scaled, not probabilities or comparable across images. No localization accuracy has been established. All errors plus fixed illustrative examples are shown. Dataset pixels stay local.</p>'+''.join(cards)+'</html>')
    print(json.dumps({k:report[k] for k in ('image_auroc','confusion','threshold')},indent=2)); print(report_path)

if __name__ == '__main__':
    main()
