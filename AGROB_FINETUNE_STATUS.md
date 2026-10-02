# Tomato detector experiment: team checkpoint

## Completed on this branch

- The pretrained fruit DETR baseline was measured on 152 AgRob frames. Its `Tomato` class returned no boxes, with mean absolute visible-fruit count error **12.118 per image**. The all-fruit diagnostic returned **9.289**, but only 31 of 1,842 labeled fruit were matched by a box at IoU 0.50, so it is not a usable tomato detector.
- The AgRob source archive was checked by MD5, and 449 image/XML pairs with 6,084 fruit boxes were read. All four source ripeness names were mapped to one training class, `tomato`.
- A fixed split is saved: 237 training images (3,289 boxes), 40 validation images (645 boxes), 152 previously viewed development images (1,842 boxes), and 20 unused buffer frames. The 14 annotated zero-fruit images are in training.
- A 43-image label-review queue, five contact sheets, one-class annotations, training/evaluation scripts, final-set converter, Colab notebook, and 126.6 MiB GPU transfer bundle were produced.
- The full corrected 15-epoch GPU run completed on the local RTX 3070. The best checkpoint is from epoch 14. Threshold `0.90` was selected on 40 validation images: **2.625 fruit/image count MAE**, box F1 **0.5973**. At that fixed threshold, the 152-image viewed development sequence reached **4.375 fruit/image count MAE**, box F1 **0.4643**. See [the run report](AGROB_FINETUNE_RUN_2026-10-01.md) for settings and limitations.
- The first GPU attempt had a square-padding coordinate bug and its outputs were discarded. The corrected preprocessing now checks box geometry before training.
- The corrected model package is `outputs/agrob-tomato-detr-480-model.zip` (158.9 MiB). It was extracted and reproduced a known image count. The local counter is `count_tomatoes.py`.

## Decisions and work still needed

1. **Subhan and Andrei:** provide the completed review CSV and any corrected Pascal VOC XML files. The local queue still has blank reviewer/correction columns, so this run used the original labels. If labels changed, prepare v2 and retrain.
2. **For a final performance claim:** collect a new capture session or row, label each visible fruit including confirmed zero-fruit scenes, and run one fixed-threshold evaluation. The old 152-image sequence is a development comparison because it was previously viewed.
3. **For truss counting:** collect truss-specific labels. AgRob fruit boxes do not identify trusses.

The honest current result is an **improved annotation-based development baseline**, with substantial box errors and no fresh final test. See [AGROB_FINETUNE_GUIDE.md](AGROB_FINETUNE_GUIDE.md) for manual steps.
