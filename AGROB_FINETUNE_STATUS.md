# Tomato detector experiment: team checkpoint

## Completed on this branch

- The pretrained fruit DETR baseline was measured on 152 AgRob frames. Its `Tomato` class returned no boxes, with mean absolute visible-fruit count error **12.118 per image**. The all-fruit diagnostic returned **9.289**, but only 31 of 1,842 labeled fruit were matched by a box at IoU 0.50, so it is not a usable tomato detector.
- The AgRob source archive was checked by MD5, and 449 image/XML pairs with 6,084 fruit boxes were read. All four source ripeness names were mapped to one training class, `tomato`.
- A fixed split is saved: 237 training images (3,289 boxes), 40 validation images (645 boxes), 152 previously viewed development images (1,842 boxes), and 20 unused buffer frames. The 14 annotated zero-fruit images are in training.
- A 43-image label-review queue, five contact sheets, one-class annotations, training/evaluation scripts, final-set converter, Colab notebook, and 126.6 MiB GPU transfer bundle were produced.
- A tiny CPU rehearsal passed the software path from model load through checkpoint save and evaluation. **It is not an accuracy result.**

## Decisions and work still needed

1. **Subhan and Andrei:** agree on [the proposed labeling rules](AGROB_LABEL_POLICY.md), check the review queue at full image size, resolve uncertain fruit, and version any corrected XML labels.
2. **Someone with a CUDA GPU:** run the [Colab notebook](AGROB_GPU_COLAB.ipynb) or equivalent full GPU command on the reviewed data. Save `training_history.csv`, `run_config.json`, the best checkpoint, and the validation results.
3. **Before looking at development counts:** choose one confidence threshold from validation `thresholds.csv`. Record count MAE and box precision/recall; inspect previews for false exact counts.
4. **For a final performance claim:** collect a new capture session or row, label each visible fruit, freeze the checkpoint and threshold, then run one final evaluation. The old 152-image sequence is a development comparison because it was previously viewed.
5. **For truss counting:** collect truss-specific labels. AgRob fruit boxes do not identify trusses.

Until steps 1–4 are complete, the honest result is **a failed pretrained baseline plus a ready-to-run fine-tuning workflow**, not an improved model. See [AGROB_FINETUNE_GUIDE.md](AGROB_FINETUNE_GUIDE.md) for commands and manual instructions.
