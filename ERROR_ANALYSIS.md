# Bottle baseline error analysis

All three errors from the first 83-image evaluation are reviewed here. The frozen model, reference bank, and threshold **0.1366028786** are unchanged. This is retrospective analysis of an already inspected test set, not new validation.

| Image | Dataset label | Decision | Score | Score minus threshold |
| --- | --- | --- | --- | --- |
| test/broken_small/007.png | broken_small | Within baseline range | 0.134995 | −0.001608 |
| test/contamination/003.png | contamination | Within baseline range | 0.124177 | −0.012426 |
| test/good/006.png | good | Review for possible anomaly | 0.186343 | +0.049740 |

Evidence: [original per-image scores](reports/evaluation-20261023T042247Z.json). A margin measures distance from the decision boundary, not confidence or a probability.

## Visual observations and hypotheses

**Small break:** The dataset mask marks a compact region on the right inner rim. The original image shows a localized bright, irregular area there. Its whole-image score falls only slightly below the threshold. Global pooling may dilute a small local defect, but this explanation has not been established experimentally. The small margin also makes this case useful for future preprocessing/device stability checks; such checks have not yet been performed.

**Contamination:** The mask marks a localized lower-right inner-rim region. A gray patch is visible there in the original image. The score is farther below the threshold than the small-break example. Whole-image similarity can remain high despite a local abnormality; global features may be insufficiently sensitive to this kind of patch. This is a hypothesis, not a localization result from the model.

**False alarm:** The dataset labels this image good. It has visible rim texture and reflections, but visual review does not establish which feature caused its high distance. Possible explanations include normal appearance variation poorly covered by the reference bank, or sensitivity of generic ImageNet features to lighting or texture. Nearest-reference comparisons and controlled experiments would be needed to test these explanations. Do not relabel it defective just because the model flags it.

## Local visual report

Run from the project directory:

```sh
.venv/bin/python -m inspectai.error_analysis --dataset data/mvtec/bottle
open artifacts/error-review.html
```

This produces a standalone local page with the three original images, recorded scores, and dataset ground-truth masks for the two defects. Masks are dataset annotations, **not predicted heatmaps**. The generator verifies each reviewed image against the original evaluation SHA-256. It does not rerun or fit the model. Masks are read from the local dataset; see its provenance manifest for source hashes.

The generated HTML embeds dataset pixels, stays in the ignored `artifacts/` directory, and must remain local. Only code and this text report belong in GitHub. Dataset attribution and terms are documented in [DATASET.md](DATASET.md).

## Engineering implications

A below-threshold result must remain “Within baseline range”; it cannot certify defect-free products. A flagged image needs review rather than an automatic defect declaration.

Lowering the threshold would trade missed defects for more false alarms. We have not changed it after looking at these errors. Choosing a new threshold from these test examples would bias the reported test result.

A justified next experiment is a patch-level feature method that preserves spatial information, compared under a predefined validation plan. Existing test results can be reported as development evidence; stronger claims require untouched evaluation data. This report does not establish that patch features will solve these errors or generalize to new products.
