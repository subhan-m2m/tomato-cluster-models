# AgRob tomato DETR GPU run — 2026-10-01

## What ran

The fruit-specific DETR checkpoint `MohamedKhayat/fruit-detector-detr-50` (revision `19d09855a284cf040460c1ac39fe994e400cdf47`) was fine-tuned to **one class, `tomato`**, on the AgRob images. Training used the local NVIDIA GeForce RTX 3070 (8 GiB), PyTorch `2.14.1+cu130`, Transformers `4.57.6`, 15 epochs, 480-pixel longest side, batch size 1, gradient accumulation 2, backbone learning rate `1e-5`, other learning rate `5e-5`, and seed 42. The best checkpoint was saved at **epoch 14** by lowest validation loss (`1.2702`). Full settings are in `results/agrob_finetune_v1_480/run_config.json`.

The run used `data/agrob_finetune_v1`: **237 training images / 3,289 boxes** and **40 validation images / 645 boxes**. No corrected XML files or completed review CSV were present in the project at run time, so these are the **original AgRob labels**. The user's visual review may have been completed separately, but it is not represented in this data version. If corrected labels exist, prepare a v2 and rerun before claiming results for the reviewed set.

An initial GPU run used fixed square padding, which shifted the vertical box targets. Its outputs under `outputs/agrob-gpu-full-v1-480/` are **invalid** and must not be used for accuracy claims. The corrected run uses dynamic padding and checks that processed box coordinates match the source labels. The results below are only from `outputs/agrob-gpu-full-unpadded-v1-480/`.

## Results from the corrected checkpoint

The confidence threshold **0.90** was chosen on the 40 validation images from a fixed sweep of `0.10, 0.20, …, 0.90, 0.95, 0.99`. It had the lowest validation mean absolute count error. The same threshold was then applied unchanged to the 152-image development sequence.

| Measure | Validation | Development sequence |
| --- | ---: | ---: |
| Images / annotated fruit | 40 / 645 | 152 / 1,842 |
| Predicted boxes | 634 | 2,435 |
| Mean absolute count error, fruit/image | **2.625** | **4.375** |
| Mean signed count error, fruit/image | -0.275 | +3.901 |
| Images with exact annotated count | 7 | 9 |
| Fruit matched at IoU 0.50 | 382 | 993 |
| Missed annotated fruit | 263 | 849 |
| Extra boxes relative to labels | 252 | 1,442 |
| Box precision / recall / F1 at IoU 0.50 | 0.6025 / 0.5922 / 0.5973 | 0.4078 / 0.5391 / 0.4643 |

The original pretrained checkpoint's `Tomato` class had **12.118 fruit/image count MAE** on this same 152-image sequence and matched none of 1,842 fruit. The corrected fine-tuned run lowers that annotation-based count error to **4.375 fruit/image**. The development sequence was inspected during the earlier baseline and may share scene conditions with training, so this comparison is **not a fresh final test**. The all-fruit original diagnostic had 9.289 count MAE but almost no aligned boxes; it should not be treated as a tomato detector.

## Common misses and label questions

- **Overcount on the development sequence:** 130 of 152 images have more predicted boxes than source fruit labels. The largest errors include `tomates_2020-08-06-11-35-15_side_0080.jpg` (7 labeled, 23 detected), `_0100.jpg` (26, 41), and `_0139.jpg` (11, 25).
- **Dense clusters:** previews show several overlapping predictions around neighboring fruit. A validation-only trial of ordinary nonmaximum suppression improved box F1 slightly but did **not** lower count MAE below 2.625, so it was not added to the count workflow.
- **Small or hidden fruit:** on validation, 179 of 381 source fruit marked `occluded` missed IoU 0.50, versus 84 of 264 not marked occluded. On development, 827 of 1,772 `unriped` fruit missed. These are source-label subgroup counts, not estimates for all greenhouse fruit.
- **Possible incomplete labels:** in the visually inspected worst development previews, several red boxes appear to surround real small tomatoes without matching green source boxes. Some extra-box and overcount penalties may reflect missing annotations. Other red boxes appear duplicated or misplaced. A human should adjudicate these images at full resolution before changing labels or interpreting true visible count error.

The prediction tool reports **detections**, not verified counts. AgRob has no truss membership or truss boxes, so this run does not count trusses. Neither validation nor development contains confirmed zero-fruit frames; the 14 zero-fruit frames are in training. The fresh final set should include zero-fruit scenes.

## Files and next action

- Small shareable metrics and per-image counts: `results/agrob_finetune_v1_480/`.
- Full local checkpoint: `outputs/agrob-gpu-full-unpadded-v1-480/best_model/` (`model.safetensors`, config, processor config).
- Shareable local checkpoint package: `outputs/agrob-tomato-detr-480-model.zip` (checkpoint, counting script, provenance, no source images). Read its `USAGE.md` after extracting.
- Full local evaluation outputs, including green-label/red-prediction previews: `outputs/agrob-gpu-full-unpadded-v1-480-valid/` and `outputs/agrob-gpu-full-unpadded-v1-480-development/`.
- Count a new image with `count_tomatoes.py --images <image-or-folder> --output <new-output-folder>` using the saved 0.90 threshold. Review the red boxes before using its count operationally.

The next required steps are to save the user's completed review record and any corrected Pascal VOC XML files into a new prepared version, retrain if labels changed, and measure the fixed model on a newly captured, fully labeled scene-separated final set. That final set is needed before declaring the baseline ready for the team's counting task.
