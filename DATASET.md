# Dataset and model provenance

## Intended dataset

- MVTec AD, bottle category: [official source](https://www.mvtec.com/research-teaching/datasets/mvtec-ad).
- MVTec AD provides defect-free training images, test images with and without anomalies, and pixel-level annotations. The baseline evaluation uses image labels. Pixel masks appear only in the local retrospective error review; they do not fit the model or establish localization accuracy.
- The official page states CC BY-NC-SA 4.0 and prohibits commercial use. Review those terms before obtaining or using the dataset. The public portfolio must not imply a commercially licensed product.
- Download status: bottle category obtained from a pinned third-party mirror after both current and older official MyDrive archive links returned HTTP 404. The mirror preserves the MVTec attribution and CC BY-NC-SA 4.0 license files. No official archive was downloaded, so no official archive checksum is claimed.

Citation: Paul Bergmann, Michael Fauser, David Sattlegger, and Carsten Steger, “MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection,” CVPR 2019. [Publisher-provided paper](https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/company/research/datasets/mvtec_ad.pdf).

## Backbone

Torchvision ResNet18 with `ResNet18_Weights.IMAGENET1K_V1`. The classification head is replaced with identity and parameters are used in inference mode. Feature vectors are L2-normalized. [Official model source](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py).

Pretrained weights originate from the Torchvision weight URL, not random initialization. The code and any dataset/pretrained-weight rights are separate. No neural-network fine-tuning is performed by the initial baseline.

## Reproducibility limits

The fit manifest records split image content hashes and seed. The evaluation records its fitted artifact hash and test image hashes. Dependency versions are pinned. Exact duplicate checks do not establish absence of near duplicates. The measured benchmark covers one category and cannot support claims about all industrial defects.

## Acquired copy

Source: [foersben/mvtec-ad on Hugging Face](https://huggingface.co/datasets/foersben/mvtec-ad/tree/c75b39616f84db43677bcc8228caaafaf5096d7f/bottle), revision `c75b39616f84db43677bcc8228caaafaf5096d7f`. This is a third-party mirror, not an MVTec-hosted copy. All 357 files were verified against that revision's LFS SHA-256 or Git blob identifiers, and all PNG headers passed Pillow validation. Equality to the unavailable official archive was not independently established.

The copy contains 209 normal training images, 20 good test images, 63 anomalous test images, 63 masks, and two license/attribution files. Images and notices remain local in `data/mvtec/bottle/`. [Provenance manifest](reports/dataset-provenance.json) records each downloaded file's SHA-256 and source metadata. The acquisition did not fit a model or evaluate its quality.

Reproduce the acquisition with `.venv/bin/python scripts/download_bottle.py`. It downloads only the pinned bottle category and verifies files, using curl with certificate verification. Dataset usage remains subject to MVTec's non-commercial license. Dataset pixels are excluded from GitHub.
