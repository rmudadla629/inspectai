"""Build a local visual review from saved evaluation scores; no model fitting."""
import argparse
import base64
import hashlib
import html
import json
from pathlib import Path
from .baseline import ROOT


def build_report(dataset, report_path, output):
    dataset = dataset.resolve()
    report = json.loads(report_path.read_text())
    threshold = report['threshold']
    cards = []
    for row in report['results']:
        predicted = row['score'] > threshold
        if predicted == bool(row['label']):
            continue
        path = (dataset / row['file']).resolve()
        if not path.is_relative_to(dataset):
            raise ValueError('Image path escapes dataset directory.')
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Image differs from evaluated bytes: ' + row['file'])
        def image_tag(image_path, caption):
            encoded = base64.b64encode(image_path.read_bytes()).decode('ascii')
            return f'<figure><img src="data:image/png;base64,{encoded}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>'
        pictures = image_tag(path, 'Evaluated image')
        mask = dataset / 'ground_truth' / path.parent.name / (path.stem + '_mask.png')
        if row['label'] and mask.exists():
            pictures += image_tag(mask, 'Dataset ground-truth mask — not a model prediction')
        elif row['label']:
            pictures += '<p>Ground-truth mask unavailable.</p>'
        else:
            pictures += '<p>Dataset label: good. No defect mask applies.</p>'
        error = 'Missed defect (false negative)' if row['label'] else 'False alarm (false positive)'
        cards.append(f'<section><h2>{error}: {html.escape(row["file"])}</h2><p>Score {row["score"]:.6f}; threshold {threshold:.6f}; signed margin {row["score"]-threshold:+.6f}. Scores are not probabilities.</p><div>{pictures}</div></section>')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>InspectAI error review</title><style>body{font:17px system-ui;max-width:1000px;margin:40px auto;padding:0 20px;background:#f5f7fa;color:#182334}section{background:white;padding:20px;margin:20px 0;border-radius:12px}div{display:flex;flex-wrap:wrap;gap:20px}figure{margin:0;max-width:420px}img{width:100%;max-width:420px}figcaption{font-size:14px}h2{font-size:20px}</style><h1>InspectAI: baseline error review</h1><p>Saved evaluation: '+html.escape(report_path.name)+'. All '+str(len(cards))+' misclassified images. No model inference or threshold changes. Dataset masks show annotations, not model localization. This local file embeds non-commercial dataset images; keep it outside Git.</p>'+''.join(cards)+'</html>')
    return len(cards)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--report', type=Path, default=ROOT / 'reports/evaluation-20261023T042247Z.json')
    args = parser.parse_args()
    output = ROOT / 'artifacts/error-review.html'
    count = build_report(args.dataset, args.report, output)
    print(f'Created {count}-image error review: {output}')


if __name__ == '__main__':
    main()
