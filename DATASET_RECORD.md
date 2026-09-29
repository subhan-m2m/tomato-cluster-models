# Dataset record for the truss pilot

## Source

- **Dataset:** Laboro Tomato (`tomato_mixed`)
- **Publisher:** Laboro.AI Inc.
- **Official page:** <https://github.com/laboroai/LaboroTomato>
- **Download:** Official archive linked from that page. The downloaded copy was extracted into `data/laboro_tomato`.
- **Accessed:** 2026-09-28
- **Terms:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0). The publisher asks users to contact Laboro.AI for commercial use. See <https://github.com/laboroai/LaboroTomato#license>.
- **Use in this repository:** Internal research and evaluation only. Do not present a model trained on these images as a commercially cleared AgriTwin model.

## Existing annotations and gap

The local copy has 643 images in Laboro's `train` split and 161 in `test`. Its COCO annotations describe individual tomatoes and ripeness classes. It has **no truss boxes or fruit-to-truss membership labels**. The truss pilot therefore requires manual annotations.

## Pilot split

`prepare_truss_pilot.py` selected 50 training and 10 validation images from Laboro's training split, plus 20 as-yet-unviewed images from its test split, with random seed `42`. The script excludes the ten test images already used for prompt exploration and spaces nearby numbered or timestamped frames across selected splits. This reduces but does not rule out scene similarity. The exact selected files and original image IDs are in the tracked `truss_pilot_manifest.json` and a matching copy at `data/truss_pilot/manifest.json`. Image data is ignored by Git.

Manual annotations are to use one class, `truss`, with one box around the visible fruit attached to one fruit-bearing stem. COCO JSON files are stored as `data/truss_pilot/{train,valid,test}/_annotations.coco.json` after CVAT export and import. The test split must not be used to choose prompts, thresholds, or training settings.

The test image `IMG_20191215_111813.jpg` has zero truss boxes by the annotator's explicit review: no truss is visible under the agreed same-stem definition. It is a confirmed negative image, not an unfinished annotation. Import its CVAT export with `import_cvat_truss.py --allow-empty`.

## Product data path

Ask a greenhouse partner for permission to use representative cart images with row and capture metadata. Preserve their written usage terms, source, capture date, and split assignments. Run a separate held-out evaluation on those images before treating pilot performance as representative of AgriTwin deployment.
