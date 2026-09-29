# AgRobTomato fruit DETR baseline — 2026-09-29

## Checkpoint 1: Data and experiment definition

- **Source:** [AgRobTomato, Zenodo record 5596799](https://zenodo.org/records/5596799), DOI `10.5281/zenodo.5596799`. The record's API reports CC BY 4.0. Retain attribution when sharing results.
- **Downloaded archive:** `Dataset-Greenhouse_Tomato_AgRob.zip`, MD5 `890666716924415720f073b06a9a02a3` (matches Zenodo).
- **Validated annotations:** 449 JPEG/XML pairs; 6,084 Pascal VOC fruit boxes across `unriped` (5,594), `breaking` (276), `reddish` (184), and `riped` (30). No truss boxes or truss IDs are present.
- **Split:** 297 `tomate_barroselas_*` images (4,242 fruit) for pilot checks; 152 `tomates_*` images (1,842 fruit) held for final scoring. This is a reproducible naming-sequence split, not proof of independent scenes. Both filename patterns contain August 6; the Zenodo description says collection occurred August 6 and 8. The archive does not identify a trustworthy day-based split. Adjacent video frames can overlap.
- **Model:** [MohamedKhayat/fruit-detector-detr-50](https://huggingface.co/MohamedKhayat/fruit-detector-detr-50), revision `19d09855a284cf040460c1ac39fe994e400cdf47`. This is a fruit-specific **DETR**, separate from the linked [RT-DETR](https://github.com/lyuwenyu/RT-DETR) project. All 530 loaded tensors were checked against the checkpoint. Inference ran on CPU; no fine-tuning was done.

## Checkpoint 2: Pilot and fixed settings

Twenty pilot images were sampled every 15th sorted image to spread review over the first sequence. Confidence threshold `0.30` and one-to-one box matching at IoU `0.50` were used. The same threshold was fixed before the 152-image test run.

| Pilot view | Mean absolute count error | Fruit matched at IoU 0.50 | Interpretation |
| --- | ---: | ---: | --- |
| `Tomato` class only | 12.60 fruit/image | 0 / 252 | The model assigned no boxes to `Tomato`. |
| All fruit classes | 7.15 fruit/image | 7 / 252 | Some counts were closer, but the locations were generally wrong. |

The all-fruit view is a **diagnostic** for this tomato-only dataset. Its output is not a reliable tomato detector. Raising the threshold above `0.30` on the same pilot predictions increased count error, so `0.30` was retained. The pilot summary files are in `results/agrob/`.

## Checkpoint 3: Held-out sequence result

| Test view, 152 images | Predicted boxes | Mean absolute count error | Mean signed error | Exact-count images | Matched / missed / extra boxes at IoU 0.50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Tomato` class only | 0 | **12.118** fruit/image | -12.118 | 0 | 0 / 1,842 / 0 |
| All fruit classes | 2,740 | **9.289** fruit/image | +5.908 | 8 | 31 / 1,811 / 2,709 |

The all-fruit predictions were labeled `Apple` (2,733) or `Pomegranate` (7), never `Tomato`. Its box recall was 1.7% and precision 1.1% at IoU 0.50. All 8 images with an exact **count** had zero matching fruit boxes, showing that the count was correct by cancellation of misses and false boxes. Results are per image; the same physical fruit may appear in adjacent frames.

## Common misses and visual review

The test split has 1,772 `unriped`, 38 `breaking`, 26 `reddish`, and 6 `riped` labeled fruit. With all fruit classes included, the model missed 1,741 / 1,772 unriped and every fruit in the other three classes at IoU 0.50. It missed all 18 fruit marked `occluded`. These are annotation-based summaries; the rarer ripeness groups are too small for a stable subgroup estimate.

In the saved previews, green boxes are ground truth and red boxes are predictions:

- `tomates_2020-08-06-11-35-15_side_0101.jpg`: 17 labeled fruit, 53 predicted boxes, 1 match. Repeated boxes cluster on leaves and bright gaps; many visible green fruit have no aligned prediction.
- `tomates_2020-08-06-11-35-15_side_0229.jpg`: 18 labeled fruit, 2 predicted boxes, 0 matches. The visible orange and green fruit are missed.
- `tomates_2020-08-06-11-35-15_side_0084.jpg`: 13 labeled fruit and 13 predictions, yet 0 matches. This is an example of a misleading exact count.

The local preview images are under `outputs/agrob-fruit-detr-test-any-t030/previews/`; the dataset and previews are excluded from Git. `results/agrob/` contains the small shareable summaries and per-image count tables.

## Reproduce on this Windows checkout

From this repository, create the environment as described in `README.md`. Download the [Zenodo archive](https://zenodo.org/records/5596799/files/Dataset-Greenhouse_Tomato_AgRob.zip?download=1) to `data/agrob/Dataset-Greenhouse_Tomato_AgRob.zip`, verify its MD5 above, and extract it under `data/agrob/`. Confirm that `data/agrob/Dataset-Greenhouse_Tomato_AgRob/Annotations/` and `JPEGImages/` exist. The archive and extracted images remain ignored by Git.

Run each command with a **new** output directory name. The runner refuses to overwrite an earlier experiment.

```powershell
.\.venv\Scripts\python.exe evaluate_agrob_fruit.py --split pilot --stride 15 --threshold 0.30 --label-filter tomato --output outputs\rerun-pilot-tomato
.\.venv\Scripts\python.exe evaluate_agrob_fruit.py --split pilot --stride 15 --threshold 0.30 --label-filter any-fruit --output outputs\rerun-pilot-any
.\.venv\Scripts\python.exe evaluate_agrob_fruit.py --split test --threshold 0.30 --label-filter tomato --output outputs\rerun-test-tomato
.\.venv\Scripts\python.exe evaluate_agrob_fruit.py --split test --threshold 0.30 --label-filter any-fruit --output outputs\rerun-test-any
```

Each output folder contains `summary.json`, `counts.csv`, `predictions.jsonl`, and green/red preview images. The summaries record the dataset MD5, model revision, package versions, threshold, label filter, split, and run time. The count error is the mean of `abs(predicted boxes - annotated fruit)` per image. Box matching is one-to-one at IoU 0.50, so a correct count can still have poor localization.

## Decision and next experiment

This unadapted checkpoint is a measured baseline, but it is **not usable** for tomato counting on AgRob. The next fruit experiment should fine-tune a tomato detector on labeled greenhouse images, then score an untouched scene-separated test set. RT-DETR is a candidate architecture for that run; its public COCO weights do not provide a tomato-specific class without adaptation. A GPU runtime is preferable because this computer's PyTorch environment reports no CUDA device.

Truss counting remains separate. AgRob's fruit boxes cannot establish same-stem truss membership or a truss count error; use the existing truss labeling pilot and its own held-out annotations for that result.
