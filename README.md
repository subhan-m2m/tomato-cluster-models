# Tomato model trials

This starter project runs two existing Hugging Face checkpoints on the **same
Laboro Tomato test images**. It saves the detected boxes, an image with boxes
drawn on it, and a count table for each model. You can run it on Windows with
PowerShell; no model training is required for this first trial.

## What each trial tells you

| Trial | What the checkpoint predicts | What Laboro can verify |
| --- | --- | --- |
| [Grounding DINO Tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny) | Boxes requested by a text prompt such as `a cluster of tomatoes.` | **Visual review only for clusters.** Laboro does not label trusses or clusters. |
| [Fruit Detector DETR-50](https://huggingface.co/MohamedKhayat/fruit-detector-detr-50) | Boxes for its `Tomato` class (individual fruit) | Compare its number of fruit boxes with Laboro's annotated fruit count. It cannot directly identify clusters. |

Both model cards list Apache-2.0. The Fruit DETR card says it was trained on a
custom fruit dataset; check those underlying data rights before product use.
Laboro Tomato is [CC BY-NC-SA 4.0](https://github.com/laboroai/LaboroTomato#license)
and asks users to contact Laboro.AI for commercial use. Keep this first trial
internal and exploratory. Do not publish Laboro images or a model trained on
them as a commercial AgriTwin asset without permission.

## Step 1 — Open this folder

Open PowerShell and run:

```powershell
cd 'C:\Users\subha\Projects\Work\Internships\M2M\2026_Q2\agritwin\tomato-cluster-models'
```

Run all later commands from this folder. `OLD` is not used.

## Step 2 — Check the Python environment

This computer already has **Python 3.12 (64-bit)** and `uv`. I created `.venv`
and installed the packages listed in `requirements.txt` there. Check it:

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -c "import torch, transformers, PIL; print(torch.__version__, transformers.__version__)"
```

If you set up the project again on another computer, install Python 3.12 and
[`uv`](https://docs.astral.sh/uv/getting-started/installation/), then create the
environment and install the packages:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

The commands below use the environment's Python directly, so PowerShell
script activation is unnecessary. I downloaded both checkpoints during the
one-image checks. On a fresh computer, the first model run downloads them
from Hugging Face into the user cache and needs internet access.
Grounding DINO is substantially larger and slower than Fruit DETR, especially
on a CPU. The code automatically uses a CUDA GPU when PyTorch can see one.

## Step 3 — Check Laboro Tomato

I downloaded and extracted the full `tomato_mixed` dataset in this workspace.
It has 643 training and 161 test images. Verify the files:

```text
data\laboro_tomato\annotations\test.json
data\laboro_tomato\test\
```

```powershell
Test-Path .\data\laboro_tomato\annotations\test.json
.\.venv\Scripts\python.exe inspect_dataset.py
```

The second command checks every image named in `test.json` and prints the fruit
label counts. The original test split stays held out; this project performs
inference only and does not train on it. The downloaded data and model outputs
are excluded from Git by `.gitignore`.

If the data is missing in a fresh checkout, download it from the
[official LaboroTomato page](https://github.com/laboroai/LaboroTomato), extract
the archive, and put its `laboro_tomato` folder under `data`. It should contain
both `annotations` and `test`. The official link uses an older HTTP address;
the certificate-valid address for the same archive is
`https://s3.ap-northeast-1.amazonaws.com/assets.laboro.ai/laborotomato/laboro_tomato.zip`.

## Step 4 — Run Grounding DINO on ten test images

```powershell
.\.venv\Scripts\python.exe run.py grounding-dino --limit 10 --prompt 'a cluster of tomatoes.'
```

Open `outputs\grounding-dino\previews` in File Explorer. Each red box is a
candidate cluster. Count **actual clusters yourself** in these images and note
missed or duplicate boxes. Try another prompt on the same images if useful:

```powershell
.\.venv\Scripts\python.exe run.py grounding-dino --limit 10 --prompt 'a tomato truss.' --output outputs\grounding-dino-truss-prompt
```

The runner removes nearly identical boxes from different prompt words at a
default intersection-over-union cutoff of `0.7`. This is a duplicate filter,
not a way to determine true truss boundaries. The prompt, confidence threshold,
and image limit are recorded in the output.

In the first image checked here, Grounding DINO returned an encompassing box
and smaller boxes around fruit within the same group. Do **not** read its box
count as the number of trusses.
`--threshold 0.3` is the default; a lower value shows more boxes and more
false positives. Use `--limit 0` only after the small run succeeds.

## Step 5 — Run Fruit DETR on the same ten images

```powershell
.\.venv\Scripts\python.exe run.py fruit-detr --limit 10
```

Open `outputs\fruit-detr\previews` to inspect its tomato boxes. Read
`outputs\fruit-detr\summary.json` for mean absolute **fruit** count error,
and `counts.csv` for each image's predicted and annotated fruit counts. This
metric does not say how well it identifies clusters.

Both commands process the first ten test images sorted by filename, so the
previews are directly comparable. Outputs include `predictions.jsonl` with
box coordinates and scores. A new run to the same output folder replaces its
count files and matching previews; use `--output` to preserve alternatives.

## Step 6 — Initialize Git and create experiment branches

You said you will initialize Git and add the remote yourself. The folder is
ready for that; no `.git` directory or remote has been created here. Once you
have reviewed the files, run:

```powershell
git init -b main
git add .
git commit -m 'Set up tomato model trials'
git remote add origin YOUR_REMOTE_URL
git push -u origin main
```

Replace `YOUR_REMOTE_URL` with the URL from your empty Git hosting repository.
Then make one branch per experiment:

```powershell
git switch -c experiment/grounding-dino
.\.venv\Scripts\python.exe run.py grounding-dino --limit 10
```

Commit any **code or configuration changes** you make on that branch. Results
and images are ignored by Git. When finished, create the second branch from
the unchanged main branch:

```powershell
git switch main
git switch -c experiment/fruit-detr
.\.venv\Scripts\python.exe run.py fruit-detr --limit 10
```

The common runner contains both adapters so each branch starts from the same
images and output format. Keep branch-specific prompt, threshold, or code
changes on the matching branch.

## Interpreting the results

* Laboro labels **individual fruit**, not cluster boundaries or truss IDs.
  Grounding DINO cluster accuracy cannot be measured from its existing labels.
* Fruit DETR's mean absolute count error uses one image at a time. It does not
  account for the same fruit appearing in adjacent cart frames.
* Before making a cluster-count claim, define what one visible truss means,
  annotate a held-out set with cluster boxes or counts, and compare each
  model's cluster predictions against those annotations.
* These checkpoints are unadapted to the team's greenhouse camera. Treat this
  as a quick baseline, then decide whether dedicated truss labels and
  fine-tuning are needed.

## Common problems

* **`.venv` is missing:** run the two `uv` setup commands in Step 2. On a new
  computer, install Python 3.12 first.
* **`test.json` missing:** move the extracted inner dataset folder into
  `data\laboro_tomato`; do not rename `annotations` or `test`.
* **Images missing:** check for an extra nested folder inside
  `data\laboro_tomato\test`.
* **CUDA is unavailable:** the code runs on CPU. If you expected GPU support,
  install a CUDA-enabled PyTorch build using the
  [official PyTorch selector](https://pytorch.org/get-started/locally/).
* **Model download fails:** check access to Hugging Face, then rerun the same
  command. The checkpoint is downloaded only once per user cache.
* **Many PyTorch or Hugging Face warnings:** some checkpoint-loading and
  Windows cache warnings are informational. Confirm that the command finishes
  with `Saved results in ...`; if it ends with `Traceback`, use the last error
  line to diagnose it.
