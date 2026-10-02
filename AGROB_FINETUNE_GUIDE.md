# Improve the AgRob tomato detector: step-by-step handoff

This is the follow-up to [the measured pretrained baseline](AGROB_FRUIT_BASELINE.md). The fruit-specific checkpoint is **DETR**, not RT-DETR. Its original tomato-only result found no boxes on the 152-image AgRob sequence (12.118 fruit/image mean absolute count error). The work here prepares a **single-class tomato fine-tune** and a way to compare visible fruit counts after training. There is no trained full-data model or improved accuracy claim yet.

## 1. Understand the three kinds of images

Think of the labeled images as practice questions, progress checks, and an exam:

| Set | Images | Fruit boxes | Use |
| --- | ---: | ---: | --- |
| `train` | 237 | 3,289 | Change model weights. Includes 14 annotated zero-fruit images. |
| `valid` | 40 | 645 | Pick the best saved epoch and the confidence threshold. |
| `development` | 152 | 1,842 | Compare with the old baseline, after settings are fixed. This sequence was already inspected during baseline work. |
| `buffer` | 20 | — | Excluded around validation blocks to reduce adjacent-frame overlap. |

The split is recorded in `agrob_finetune_split.json`. Validation uses two blocks of 20 frames from the 297-image pilot sequence, with five neighboring frames excluded on each side. AgRob contains video frames that may be highly similar. This split helps, but does **not** prove independent scenes. A newly captured, labeled final set is required for a trustworthy final score.

Every source box labeled `unriped`, `breaking`, `reddish`, or `riped` is one `tomato` box for the model. The original ripeness name is kept for missed-fruit analysis. The proposed box rules are in [AGROB_LABEL_POLICY.md](AGROB_LABEL_POLICY.md).

## 2. What I prepared and checked

- Verified the 449 original image/XML pairs and Zenodo archive MD5 `890666716924415720f073b06a9a02a3`.
- Converted the source labels into one-class COCO-style `train.json`, `valid.json`, and `development.json` under `data/agrob_finetune_v1/annotations/`.
- Made `data/agrob_finetune_v1/review/review_queue.csv` with 43 suggested examples and five numbered contact sheets. The queue includes crowded, partly hidden, ripening, zero-fruit, random, and old-error images.
- Added a training/evaluation program, a Pascal VOC converter for a future final set, and a GPU transfer bundle.
- Ran an eight-image, two-epoch **CPU rehearsal**. It completed model loading, one-class training, validation, checkpoint saving, and evaluation. The tiny run's losses and counts are a software check only; they are not an accuracy result.

The 43 suggested review images have **not** been manually certified. The prepared set is labeled `provisional; team review pending` in `summary.json`.

## 3. Review the labels with another person

This step needs a human who can decide which shapes are tomatoes. Open `data/agrob_finetune_v1/review/sheet_01.jpg` through `sheet_05.jpg` to see the quick overview, then examine each selected source JPEG at full size. The source images live in `data/agrob/Dataset-Greenhouse_Tomato_AgRob/JPEGImages/`; the matching Pascal VOC XMLs are in `Annotations/`.

For each row of `review_queue.csv`, check the rules in `AGROB_LABEL_POLICY.md`: one tight box per visible distinct fruit, no leaf boxes, no duplicates, and an explicit XML for a true zero-fruit image. Record your name, `yes` or `no` under `needs_correction`, and a short note. Have Andrei or another reviewer settle ambiguous fruit. Review at least the suggested 43; adding more, especially validation frames, is better. The contact sheets are too small for final decisions.

You can use an annotation program that imports and exports **Pascal VOC XML**. For example, CVAT supports Pascal VOC. Import the JPEGs and their existing XMLs into a task, correct the boxes visually, and export Pascal VOC annotations. Check that each corrected XML has the original JPEG basename and size. Put **only** changed XMLs into `data/agrob_label_overrides_v2/`. Do not edit the Zenodo originals. A team member should verify a few exported XMLs by reopening them.

When corrections are ready, create a new prepared version from this repository:

```powershell
.\.venv\Scripts\python.exe prepare_agrob_finetune.py --overrides data/agrob_label_overrides_v2 --review-record data/agrob_finetune_v1/review/review_queue.csv --output data/agrob_finetune_v2
```

The `--overrides` folder is optional. If the team accepts every original box, omit `--overrides` but keep `--review-record` and a new v2 output folder; this preserves who reviewed what. If review is still pending, use `data/agrob_finetune_v1/` and clearly call the results provisional. For v2, use `--prepared data/agrob_finetune_v2` in every later command and create a new GPU bundle using `make_agrob_gpu_bundle.py --prepared data/agrob_finetune_v2`. The v2 folder retains the completed review record, while its newly generated queue is a suggested index.

## 4. Run the full training on a GPU

This Windows machine currently has CPU PyTorch and reports **no CUDA GPU**. A full DETR run needs access to a CUDA GPU computer or a Google Colab GPU runtime. The easiest handoff is the ready-made notebook [AGROB_GPU_COLAB.ipynb](AGROB_GPU_COLAB.ipynb):

1. Make the transfer ZIP on this computer with the command below. It contains the scripts, prepared labels, review pack, and the 429 train/validation/development JPEGs. The 20 buffer frames and original Zenodo ZIP are omitted.
2. Upload `outputs/agrob-finetune-gpu-bundle.zip` to your Google Drive. Open `AGROB_GPU_COLAB.ipynb` in Colab (upload the notebook if needed).
3. In Colab, choose **Runtime → Change runtime type → GPU**, then run cells from top to bottom. The notebook mounts Drive, unpacks the ZIP, installs the training libraries, checks for CUDA, trains, and saves outputs under `MyDrive/agrob_finetune_runs/`.
4. The default full run uses 15 epochs, 480-pixel images, batch size 1, and accumulation 2. If GPU memory runs out, choose a **new run folder** and lower `--image-size` to 320. Record that change when sharing results. If the session stops before completion, keep the saved checkpoint but do not call the run complete; rerun with a new folder because the script does not resume the optimizer.

```powershell
.\.venv\Scripts\python.exe make_agrob_gpu_bundle.py --output outputs/agrob-finetune-gpu-bundle.zip
```

If you used corrected v2 labels, create `outputs/agrob-finetune-gpu-bundle-v2.zip` with `--prepared data/agrob_finetune_v2 --output outputs/agrob-finetune-gpu-bundle-v2.zip`. In the notebook's first code cell, change `bundle` to that filename; in its training cell, set `prepared_name = 'agrob_finetune_v2'`.

From a GPU computer with this checkout and all dependencies installed, the equivalent training command is:

```powershell
.\.venv\Scripts\python.exe finetune_agrob_fruit.py train --prepared data/agrob_finetune_v1 --epochs 15 --image-size 480 --batch-size 1 --grad-accum 2 --output outputs/agrob-full-v1
```

The program saves the best checkpoint by lowest validation **loss** in `best_model/`, plus `training_history.csv`, `run_config.json`, and `best_epoch.json`. A lower loss alone is not the final decision; also inspect count error and box alignment.

## 5. Choose the confidence threshold on validation images

After training, run validation scoring once. The program tries confidence thresholds 0.10, 0.20, 0.30, 0.40, and 0.50; it chooses the lowest mean absolute count error, then the higher box F1 if tied. It also writes the full `thresholds.csv` so the team can inspect the trade-off.

```powershell
.\.venv\Scripts\python.exe finetune_agrob_fruit.py evaluate --prepared data/agrob_finetune_v1 --checkpoint outputs/agrob-full-v1/best_model --split valid --output outputs/agrob-full-v1-valid
```

Read `outputs/agrob-full-v1-valid/summary.json` and copy `chosen_metrics.threshold`. Look at `counts.csv`, especially the largest errors, and inspect `previews/` at full size. In these previews, **green = label** and **red = prediction**. Check whether exact counts come from correctly placed boxes; the original baseline had some exact counts with no aligned fruit boxes. If the validation result is poor, correct labels or change the training recipe, then retrain under a new run name. Do not tune on the development or final set.

## 6. Compare once with the viewed development sequence

After the checkpoint and threshold are frozen, score the 152-image development sequence. Replace `0.3` below with the chosen validation value:

```powershell
.\.venv\Scripts\python.exe finetune_agrob_fruit.py evaluate --prepared data/agrob_finetune_v1 --checkpoint outputs/agrob-full-v1/best_model --split development --thresholds 0.3 --output outputs/agrob-full-v1-development
```

Report mean absolute count error in **fruit per image**, mean signed count error (positive = overcount), exact-count image count, box precision/recall/F1 at IoU 0.50, and common misses by ripeness, crowding, occlusion, or lighting. Compare with the original Tomato-class baseline **12.118 fruit/image MAE**, while saying that development images were previously viewed. Do not call it an untouched test score.

## 7. Make a fresh final test before claiming success

This step needs newly captured images and human labels. Use a different row, camera session, or date from AgRob; keep an entire capture sequence together so similar neighboring frames do not cross sets. A practical starting target is 50 or more varied images including some confirmed zero-fruit scenes. Do not look at predictions while drawing these labels.

Create this folder layout:

```text
data/final_tomato_v1/
  JPEGImages/       photo_001.jpg, photo_002.jpg, ...
  Annotations/      photo_001.xml, photo_002.xml, ...
```

Each JPEG needs one Pascal VOC XML, including an empty-object XML for a confirmed zero-fruit image. Run the converter, which checks names, dimensions, labels, and box coordinates:

```powershell
.\.venv\Scripts\python.exe convert_tomato_voc.py --source data/final_tomato_v1 --output data/agrob_finetune_v1/annotations/final.json
```

Then score the fixed checkpoint at the fixed validation threshold; again replace `0.3` with the actual value:

```powershell
.\.venv\Scripts\python.exe finetune_agrob_fruit.py evaluate --prepared data/agrob_finetune_v1 --images-dir data/final_tomato_v1/JPEGImages --checkpoint outputs/agrob-full-v1/best_model --split final --thresholds 0.3 --output outputs/agrob-full-v1-final
```

Keep the final images, their XMLs, `final.json`, and the output folder together for reproducibility. If the first fresh final score is weak, treat it as a finding and start a **new** training/validation cycle; do not retroactively change the threshold for that score. For truss counting, collect distinct truss labels and make a separate model/evaluation.

## 8. What to share at each checkpoint

| When | Share with the team |
| --- | --- |
| Now | Baseline: 152 viewed images, Tomato-class MAE 12.118; original checkpoint fails to locate tomatoes. Prepared split: 237 train, 40 validation, 152 development, 20 buffer. Labels provisional. |
| After label review | Agreed box policy, number of reviewed/corrected images, examples of ambiguous cases, and the versioned prepared-data name. |
| After GPU training | Run config, best epoch, validation loss trend, validation threshold table, count MAE, box F1, and five representative previews. |
| After development comparison | Fixed-threshold development metrics versus baseline, with common misses and the note that these images were already viewed. |
| After fresh final set | Capture/label protocol, set size, final count error, box metrics, example misses, and whether it is sufficient for the team's use. |

The `outputs/` and `data/` folders are intentionally ignored by Git because they contain large images and checkpoints. Share a small report and selected previews through your team's normal channel; retain the full artifacts for audit. AgRob is attributed to [Zenodo record 5596799](https://zenodo.org/records/5596799) under the license noted in the baseline report.
