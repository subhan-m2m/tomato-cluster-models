# Run the pruning prototype from a fresh clone

This guide is included in both repositories. You need both feature branches, a Python environment, Node.js, and the **separately shared trained checkpoint bundle**. A Git clone alone does not include trained weights, photos, datasets, or saved observations. There is no automatic download of a replacement detector.

## 1. Prerequisites and repository access

- Git and permission to clone both repositories below. HTTPS cloning may prompt you to sign in to GitHub. If access is denied, request repository access from the maintainer.
- Python **3.12** with `venv` and pip. The tested Windows interpreter is Python 3.12. On Windows the commands use `py -3.12`; on macOS/Linux use `python3.12`.
- Node.js **22.12 or later** and npm; the implementation was checked with Node 22.14.0 and npm 11.2.0. The installed Vite dependency requires Node `^20.19.0 || >=22.12.0`.
- Internet access for the initial Git/package installation. A GPU is optional: the CPU setup below runs both detectors. CUDA is an optional faster path for compatible NVIDIA systems.
- Obtain `pruning-checkpoints-v1.zip` from the repository maintainer. The approximately **318 MiB** bundle is not checked into Git or published at a download URL. It contains both trained checkpoints and their configurations, no photos or grower notes. The maintainer can regenerate it with `python pruning_checkpoints.py package` in the model repository. Do not substitute the original pretrained Hugging Face detector; it is a different model.

## 2. Clone the exact branches and layout

Run in the parent folder where you want the project. The following commands work in PowerShell and a POSIX shell:

```text
mkdir agritwin
cd agritwin
git clone --branch feature/pruning-association-prototype https://github.com/subhan-m2m/tomato-cluster-models.git tomato-cluster-models
mkdir frontend
git clone --branch feature/pruning-review-prototype https://github.com/m2mtechconnect/agritwin-harvest-dashboard-oct-2026.git frontend/agritwin-harvest-dashboard-oct-2026
```

Keep this nesting: the API resolves the model repository from the frontend's location, even when custom checkpoint paths are supplied.

```text
agritwin/
  tomato-cluster-models/
    pruning_association.py
    pruning_checkpoints.py
    pruning-checkpoints.json
    outputs/
      agrob-gpu-full-unpadded-v1-480/best_model/
      agrob-cluster-detr-v2-480/best_model/
  frontend/
    agritwin-harvest-dashboard-oct-2026/
      inference/server.py
      inference/requirements.txt
      package.json
      package-lock.json
```

If you already have these clones, check `git status` and preserve local changes before switching to the named feature branches. Do not reset existing work to follow this guide.

## 3. Install and verify the trained checkpoints

Each `best_model` directory must contain **all three** files: `config.json`, `preprocessor_config.json`, and `model.safetensors`. The ZIP contains paths starting with `outputs/`; extract it into `tomato-cluster-models`, not into `outputs` or an additional nested bundle folder.

Windows PowerShell, from `agritwin` (replace the archive path):

```powershell
Expand-Archive -LiteralPath 'C:\path\to\pruning-checkpoints-v1.zip' -DestinationPath .\tomato-cluster-models
py -3.12 .\tomato-cluster-models\pruning_checkpoints.py verify
```

macOS/Linux, from `agritwin`:

```sh
unzip /path/to/pruning-checkpoints-v1.zip -d tomato-cluster-models
python3.12 tomato-cluster-models/pruning_checkpoints.py verify
```

The verification script uses only Python's standard library. It checks all six files against the sizes and SHA-256 values in the tracked `pruning-checkpoints.json`. A successful check reports `Verified 6 checkpoint files`. Ask for the correct bundle if a file is missing or mismatched; copying only the weight files is insufficient.

## 4. Create the Python environment

The explicit interpreter paths avoid activation and PowerShell execution-policy changes. Commands in this section start from `agritwin`.

Windows PowerShell, CPU setup:

```powershell
py -3.12 -m venv .\tomato-cluster-models\.venv
& .\tomato-cluster-models\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\tomato-cluster-models\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
& .\tomato-cluster-models\.venv\Scripts\python.exe -m pip install -r .\frontend\agritwin-harvest-dashboard-oct-2026\inference\requirements.txt
```

macOS/Linux, CPU setup:

```sh
python3.12 -m venv tomato-cluster-models/.venv
tomato-cluster-models/.venv/bin/python -m pip install --upgrade pip
tomato-cluster-models/.venv/bin/python -m pip install -r frontend/agritwin-harvest-dashboard-oct-2026/inference/requirements.txt
```

For Linux you may use the same CPU wheel index as the Windows example before installing the requirements. On macOS, install PyTorch from the normal package index as shown. Start the API with `--device cpu` in either case; the prototype does not select Apple's MPS backend. The Windows workflow is smoke-tested; the shell examples provide the corresponding paths for macOS/Linux.

The inference requirements pin Transformers to **4.57.6**. Use those requirements for this prototype, even though the model repository also has broader requirements for older experiments.

For NVIDIA CUDA, install a PyTorch/torchvision pair suitable for your OS and driver using the [official PyTorch installer](https://pytorch.org/get-started/locally/) instead of the CPU-only wheel command, then install the same inference requirements. You can reuse an existing `.venv-gpu` by substituting its interpreter path throughout. Check that the environment sees your GPU:

```powershell
& .\tomato-cluster-models\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

Use `.venv/bin/python` for that command on macOS/Linux. A CUDA result of `False` does not prevent CPU operation.

## 5. Install the frontend

From `agritwin`, enter the frontend directory and install the locked dependencies:

```text
cd frontend/agritwin-harvest-dashboard-oct-2026
npm ci
```

If PowerShell blocks `npm.ps1`, run `npm.cmd ci` and `npm.cmd run dev`; changing machine-wide execution policy is unnecessary. Use `npm ci` for this branch's npm lockfile. The pre-existing `bun.lock` belongs to the original project tooling; do not alternate package managers during this setup.

## 6. Start the two processes

Keep two terminals open in `agritwin/frontend/agritwin-harvest-dashboard-oct-2026`.

**Terminal 1: Python API**, Windows PowerShell:

```powershell
& ..\..\tomato-cluster-models\.venv\Scripts\python.exe inference\server.py --device cpu
```

**Terminal 1: Python API**, macOS/Linux:

```sh
../../tomato-cluster-models/.venv/bin/python inference/server.py --device cpu
```

The API prints `Photo counting ready at http://127.0.0.1:8001 (cpu)`. If using a working CUDA environment, omit `--device cpu` to select CUDA automatically. The service loads the two local checkpoints; no training, dataset download, ripeness service, or cloud account is needed to analyze your own photo.

**Terminal 2: frontend**, either shell:

```text
npm run dev
```

Open the **Local** URL printed by Vite, then visit **Settings** or `/settings`. Use the Vite URL, not port 8001, for the application. Vite proxies `/api/count` and all of its subroutes to port 8001. Keep the API at that port unless you also update `vite.config.ts`.

To check the API directly, open [the health endpoint](http://127.0.0.1:8001/api/count/health): it should return JSON with `"ready": true`. A 404 at the API root `/` is expected; that process does not host the UI. Stop either process with Ctrl+C in its terminal. `npm run build` checks a production build; hosting it requires separately configuring the API and is outside this local setup.

## 7. First-use walkthrough and saved files

1. In **Settings**, enter your crop profile and an integer maximum visible fruit per cluster. There is no default limit. This is a grower-selected review rule, not an agronomic recommendation supplied by the software.
2. In **Crop analysis → Tomato & cluster counts**, optionally choose a row and capture time, then upload a JPEG, PNG, or WebP under 10 MB and 40 megapixels.
3. Select a cluster to focus its evidence boxes. Counts describe detected visible fruit; ambiguous memberships and quality flags remain visible.
4. In **Greenhouse heatmap → Pruning hotspots**, inspect the row's active photo, flagged/assessed counts, and uncertain groups. Open **Location unassigned** to locate a photo later. A manual location colors a whole row; it does not invent positions along it.
5. Try a different crop maximum: saved assessments recalculate without another model run. Exact duplicate uploads reuse the saved run; explicit reanalysis creates history. Acknowledgment records review, not completed trimming.
6. Export an observation as a ZIP containing its record, crop rule history and evidence photo, or delete a selected record after its confirmation. The Ripeness overlay shows an explicit unavailable state.

The service creates `inference/local-data/observations.sqlite3` and evidence JPEGs on first use. They survive refresh and restart and are ignored by Git. Keep the database and evidence together for backups. Pass `--data-dir "another-folder"` to keep a separate test collection. New clones start with no observations. Stop the service before moving its data. Never delete a directory containing records you want to retain.

## 8. Verify the installation

Frontend directory, Windows PowerShell:

```powershell
npm test
node node_modules/typescript/bin/tsc --noEmit
npm run lint
npm run build
& ..\..\tomato-cluster-models\.venv\Scripts\python.exe -m unittest discover -s inference
```

For macOS/Linux, replace the last interpreter with `../../tomato-cluster-models/.venv/bin/python`. The API tests start temporary loopback servers; allow local loopback connections in your test environment.

Model directory, without model weights or third-party Python packages:

```text
python -m unittest test_pruning_association test_pruning_review
```

Use `py -3.12` or `python3.12` if your shell does not provide `python`. The implementation originally passed 8 frontend tests, 13 API/storage tests and 9 association/scoring tests. Lint reports six existing Fast Refresh warnings in shared UI components; there should be zero lint errors.

## 9. Optional: regenerate the manual review pack

Analyzing your own photos requires no dataset. Regenerating the 24-image reference pack additionally requires the team's **V2 cluster COCO export**, shared separately by the maintainer. The original public AgRob download does not contain the team's added V2 cluster labels. The generated pack under `outputs/` is also excluded from Git; a new clone will not contain its HTML or image files.

From `tomato-cluster-models`, with the Python inference environment installed and the V2 archive available:

```powershell
& .venv\Scripts\python.exe prepare_agrob_clusters.py --archive 'C:\path\to\tomato cluster.v2i.coco.zip' --output data/agrob_clusters_v2 --manifest outputs/local-v2-import-manifest.json
& .venv\Scripts\python.exe prepare_pruning_review.py --output outputs/my-pruning-review
```

The importer expects a new destination; do not overwrite an existing prepared dataset. On macOS/Linux use `.venv/bin/python` and a POSIX archive path. Open the generated `index.html` and follow `PRUNING_ASSOCIATION.md` in the model repository. Complete development membership/count labels before scoring the reserved scene. The scorer rejects unreviewed references; numerical recommendation accuracy and grower validation are still pending.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Clone says repository not found | Correct GitHub account and permission to both repos; branch names above. |
| `No module named pruning_association` | Both feature branches and the exact sibling folder layout in section 2. Custom weight paths do not change the module lookup. |
| `Checkpoint is missing` or local config cannot load | Obtain and extract the separate bundle at the correct level, then run `pruning_checkpoints.py verify`. |
| `torch`, `transformers`, or `PIL` import fails | Install inference requirements with the same Python interpreter used to start the API. |
| Frontend says to start the counting service | Check the API terminal, health endpoint, port 8001, and that Vite is running. |
| Port 8001 is already in use | Stop an earlier counting/ripeness service in its own terminal. Both cannot use the same port. |
| Node/Vite engine or startup error | Check `node --version`, use a supported Node version, and run `npm ci` in the frontend directory. |
| No pruning flags after upload | Configure a crop maximum first. Empty cluster results or a failed model remain not evaluated; unknown is not zero risk. |
| CUDA is unavailable or memory is insufficient | Use the CPU environment and start with `--device cpu`; CPU inference can take longer. |
| `index.html` for the review pack is absent | Obtain the separate review pack or regenerate it with the V2 labeled dataset; it is not included in Git. |
