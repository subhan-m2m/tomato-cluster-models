# AgRob tomato counting baseline

This repository evaluates the pretrained [Fruit Detector DETR-50](https://huggingface.co/MohamedKhayat/fruit-detector-detr-50) checkpoint on the [AgRobTomato dataset](https://zenodo.org/records/5596799). It counts visible fruit boxes in each image and compares them with AgRob's annotations. The checkpoint is a standard DETR model, separate from the RT-DETR project.

The **original pretrained checkpoint** is not accurate enough for tomato counting on AgRob. Read [AGROB_FRUIT_BASELINE.md](AGROB_FRUIT_BASELINE.md) for its fixed experiment, test results, and representative misses. Small shareable summaries and per-image counts are in `results/agrob/`.

The RTX 3070 fine-tuning run and its measured results are in [AGROB_FINETUNE_RUN_2026-10-01.md](AGROB_FINETUNE_RUN_2026-10-01.md). The corrected checkpoint reduces annotation-based count MAE on the previously viewed development sequence to **4.375 fruit/image** at a validation-chosen threshold of **0.90**. It used original AgRob labels; a completed review record or corrected XMLs have not been added to this checkout. A fresh final set remains necessary. [AGROB_FINETUNE_GUIDE.md](AGROB_FINETUNE_GUIDE.md) covers label review, rerunning, and final testing. The GPU notebook is [AGROB_GPU_COLAB.ipynb](AGROB_GPU_COLAB.ipynb).
For a short team handoff, use [AGROB_FINETUNE_STATUS.md](AGROB_FINETUNE_STATUS.md).

## Files used by this experiment

| Path | Purpose |
| --- | --- |
| `evaluate_agrob_fruit.py` | Loads AgRob Pascal VOC labels, runs fruit DETR, saves counts and box previews. |
| `requirements.txt` | Python packages for inference. |
| `data/README.md` | Dataset placement and provenance. |
| `AGROB_FRUIT_BASELINE.md` | Experiment settings, interpretation, and next steps. |
| `results/agrob/` | Tracked summary JSON and per-image test CSV files. |
| `prepare_agrob_finetune.py` | Converts source labels to one-class training data and makes a label-review pack. |
| `finetune_agrob_fruit.py` | Trains and evaluates the one-class tomato detector. |
| `convert_tomato_voc.py` | Converts a newly labeled Pascal VOC final set for scoring. |
| `make_agrob_gpu_bundle.py` | Packs the prepared images and scripts for GPU transfer. |
| `agrob_finetune_split.json` | Fixed image assignments for the next experiment. |
| `AGROB_LABEL_POLICY.md` | Proposed human annotation rules. |
| `count_tomatoes.py` | Counts detections in a new image or folder and saves red-box previews. |
| `summarize_agrob_errors.py` | Summarizes common misses from evaluation predictions. |
| `package_agrob_checkpoint.py` | Packs the saved model and matching result files for sharing. |
| `results/agrob_finetune_v1_480/` | Small tracked metrics from the RTX 3070 run. |

Images, model caches, and full output folders are ignored by Git.

## Set up on Windows

Open PowerShell in this repository. The existing `.venv` can be reused. For a new checkout, install Python 3.12 and [`uv`](https://docs.astral.sh/uv/getting-started/installation/), then run:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

Download `Dataset-Greenhouse_Tomato_AgRob.zip` from [Zenodo](https://zenodo.org/records/5596799) to `data/agrob/`, verify MD5 `890666716924415720f073b06a9a02a3`, and extract it there. These paths must exist:

```text
data/agrob/Dataset-Greenhouse_Tomato_AgRob/Annotations/
data/agrob/Dataset-Greenhouse_Tomato_AgRob/JPEGImages/
```

The model is pinned to checkpoint revision `19d09855a284cf040460c1ac39fe994e400cdf47`. The first run may download its weights from Hugging Face. CPU inference works; a CUDA GPU is used automatically when available.

## Run the baseline

Start with five pilot images:

```powershell
.\.venv\Scripts\python.exe evaluate_agrob_fruit.py --split pilot --limit 5 --threshold 0.30 --label-filter tomato --output outputs\my-pilot-check
```

For the fixed full test, run the Tomato-class result and the diagnostic that counts every fruit class:

```powershell
.\.venv\Scripts\python.exe evaluate_agrob_fruit.py --split test --threshold 0.30 --label-filter tomato --output outputs\my-test-tomato
.\.venv\Scripts\python.exe evaluate_agrob_fruit.py --split test --threshold 0.30 --label-filter any-fruit --output outputs\my-test-any
```

Choose a **new output folder** for each run; the script refuses to overwrite existing results. Each folder contains `summary.json`, `counts.csv`, `predictions.jsonl`, and preview images with green ground-truth boxes and red predictions. `--label-filter any-fruit` is diagnostic only: this checkpoint mislabeled most predicted boxes as Apple in the AgRob test.

The `pilot` split has 297 `tomate_barroselas_*` images; `test` has 152 `tomates_*` images. The groups come from filename patterns and are not a verified independent scene split. Nearby source video frames may overlap. No training is performed here.

AgRob has individual-fruit boxes and ripeness labels, but no truss boxes or same-stem grouping. Truss counting needs its own annotations and evaluation.
