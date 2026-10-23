"""Download only the bottle category from a pinned third-party dataset mirror.

Uses curl's verified TLS and verifies every file against the mirror manifest.
This verifies mirror integrity, not equality to an unavailable official archive.
"""
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent
REPOSITORY = "foersben/mvtec-ad"
REVISION = "c75b39616f84db43677bcc8228caaafaf5096d7f"
BASE = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/"


def fetch(url):
    return subprocess.check_output(["curl", "-fsSL", "--retry", "2", "--max-time", "60", url])


def main():
    manifest_url = f"https://huggingface.co/api/datasets/{REPOSITORY}/tree/{REVISION}/bottle?recursive=true&limit=1000"
    manifest = json.loads(fetch(manifest_url))
    files = [row for row in manifest if row["type"] == "file"]
    if len(files) != 357:
        raise ValueError("Unexpected bottle manifest size; review before downloading.")
    def download(row):
        relative = PurePosixPath(row["path"])
        if relative.is_absolute() or '..' in relative.parts or relative.parts[0] != 'bottle':
            raise ValueError("Unexpected file path")
        path = ROOT / "data/mvtec" / str(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        content = path.read_bytes() if path.exists() else fetch(BASE + str(relative))
        digest = hashlib.sha256(content).hexdigest()
        if "lfs" in row:
            valid = digest == row["lfs"]["oid"]
        else:
            valid = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest() == row["oid"]
        if len(content) != row["size"] or not valid:
            raise ValueError(f"Integrity verification failed: {relative}")
        if not path.exists():
            temporary = path.with_suffix(path.suffix + '.part')
            temporary.write_bytes(content)
            temporary.replace(path)
        return {"path": str(relative), "sha256": digest, "size": len(content)}
    results = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(download, row) for row in files]):
            results.append(future.result())
            if len(results) % 50 == 0:
                print(f"Verified {len(results)}/{len(files)} files", flush=True)
    from PIL import Image
    for row in results:
        if row['path'].endswith('.png'):
            with Image.open(ROOT / 'data/mvtec' / row['path']) as image:
                image.verify()
    report = {"retrieved_at": datetime.now(timezone.utc).isoformat(), "dataset": "MVTec AD bottle",
              "source_type": "Third-party mirror; official MyDrive bottle URL returned HTTP 404",
              "repository": REPOSITORY, "revision": REVISION, "manifest_url": manifest_url,
              "license": "CC BY-NC-SA 4.0", "count": len(results),
              "limitation": "All files match the pinned mirror manifest; equality to the unavailable official archive was not independently established.",
              "files": sorted(results, key=lambda row: row['path'])}
    (ROOT / 'reports').mkdir(exist_ok=True)
    (ROOT / 'reports/dataset-provenance.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f"Verified {len(results)} files and decoded all PNG headers. Images remain local under data/mvtec/bottle.", flush=True)


if __name__ == '__main__':
    main()
