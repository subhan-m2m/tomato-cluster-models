# Truss pilot: beginner walkthrough

This guide starts from the existing `tomato-cluster-models` folder on Subhan's Windows computer. The agreed label is **one truss for fruit attached to the same fruit-bearing stem**. The goal is to count visible trusses in one image. We will use Laboro for an internal research pilot; its license is CC BY-NC-SA 4.0 and commercial use requires contacting Laboro.AI: <https://github.com/laboroai/LaboroTomato#license>.

## 1. Find the prepared images

Open PowerShell and enter:

```powershell
cd 'C:\Users\subha\Projects\Work\Internships\M2M\2026_Q2\agritwin\tomato-cluster-models'
Get-ChildItem .\data\truss_pilot
```

The pilot image selection has already been created. It contains `train` (50 images), `valid` (10 images), `test` (20 images), and `manifest.json`. A matching `truss_pilot_manifest.json` is tracked in the repository. The first ten Laboro test images already viewed in earlier trials were excluded. The selection also spaces nearby numbered or timestamped photos across splits, although you should still review for visually similar plants. **Do not move images between these folders after labeling starts.** Image folders and model outputs are excluded from Git.

If you clone the repository onto a different computer, first download Laboro as described in `README.md`, then run `python prepare_truss_pilot.py` from the project environment. The script checks that its recreated manifest agrees with the tracked copy so every person uses the same image list.

## 2. Label the images in CVAT

CVAT's task creation and rectangle tools are documented at <https://docs.cvat.ai/docs/workspace/tasks-page/> and <https://docs.cvat.ai/docs/annotation/tools/shape-mode-advanced/>. Use a private team workspace appropriate for your data. Nothing in this repository uploads images automatically.

1. Open <https://app.cvat.ai/> in a browser and sign in or create an account.
2. Open **Tasks**, select **+**, then **Create new task**.
3. Name the first task `AgriTwin truss train`.
4. Add one label named exactly `truss`. Set its shape to **Rectangle**.
5. Use **Select files** to upload the 50 `.jpg` files from `data\truss_pilot\train`. Submit the task. Upload the image files, not `manifest.json`.
6. Open the task's annotation job. For each image, select **Rectangle**, select **Shape**, then draw one tight box around the visible fruit belonging to each same-stem truss. Include the visible connecting stem within the box when practical. Two nearby trusses get two boxes, even if their boxes overlap. Several fruit on one stem get **one** truss box. If only one fruit is visible from a clearly identifiable truss, draw one box. Do not draw individual-fruit boxes inside a truss box.
7. If a stem connection cannot be determined, record the filename for Eric and Bao to review. Resolve it before using that image for scoring; do not guess or treat an uncertain truss as absent.
8. Save your work with CVAT's **Save** button. Review a few images together with Eric to check that the same-stem rule is being applied consistently, then finish the task.
9. Repeat steps 2–8 for `AgriTwin truss valid` using `data\truss_pilot\valid`, and `AgriTwin truss test` using `data\truss_pilot\test`. Keep the test labels private from model decisions until the end.

For each task, select **Actions → Export task dataset**. Choose **COCO 1.0** if available; the importer also accepts **CVAT for images 1.1** ZIPs containing `annotations.xml`. You do not need CVAT's **Save images** option because the image files already exist locally. CVAT describes exports at <https://docs.cvat.ai/docs/dataset_management/export-datasets/>. Put the three downloaded ZIP files in a new `data\cvat_exports` folder and rename them `train.zip`, `valid.zip`, and `test.zip`. Make sure you put each export under the matching name.

## 3. Check and import the labels

Back in PowerShell, from the project folder:

```powershell
New-Item -ItemType Directory -Force .\data\cvat_exports
```

Now copy the three downloaded ZIP files into `data\cvat_exports` and rename them `train.zip`, `valid.zip`, and `test.zip`. Then run:

```powershell
Get-ChildItem .\data\cvat_exports
.\.venv\Scripts\python.exe import_cvat_truss.py --allow-empty
```

`Get-ChildItem` should show all three ZIPs. The import command checks that every exported image matches the fixed split, that the only class is `truss`, and that the boxes are valid. It writes `_annotations.coco.json` into each split folder. The `--allow-empty` flag accepts the test image `IMG_20191215_111813.jpg`, which the annotator reviewed and confirmed has no visible truss. If a future export reports other images without boxes, review those images before accepting them. Fix missing images, wrong classes, or invalid boxes in CVAT and rerun the import.

## 4. Score Grounding DINO on validation images

Use **validation** for choosing the prompt and confidence threshold. Each command below saves a separate folder under `outputs`; red preview boxes are Grounding DINO predictions and green boxes are your manual labels. Run these one at a time:

```powershell
.\.venv\Scripts\python.exe evaluate_grounding_dino_truss.py --split valid --prompt 'a tomato truss.' --threshold 0.25 --output outputs\dino-truss-025
.\.venv\Scripts\python.exe evaluate_grounding_dino_truss.py --split valid --prompt 'a tomato truss.' --threshold 0.35 --output outputs\dino-truss-035
.\.venv\Scripts\python.exe evaluate_grounding_dino_truss.py --split valid --prompt 'a cluster of tomatoes.' --threshold 0.25 --output outputs\dino-cluster-025
.\.venv\Scripts\python.exe evaluate_grounding_dino_truss.py --split valid --prompt 'a cluster of tomatoes.' --threshold 0.35 --output outputs\dino-cluster-035
```

For each run, read `summary.json`, `counts.csv`, and several `previews` images. `mean_absolute_truss_count_error` is the average number of trusses over- or under-counted per image; smaller is better. `false_boxes` includes nested individual-fruit boxes. `missed_trusses` counts labeled trusses with no matching prediction. `box_precision` and `box_recall` require a prediction to overlap one manual box by at least 50% and match it one-to-one. Choose **one** prompt and threshold using these validation results. Also agree with Eric what error is acceptable for the project demo; no fixed pass threshold has been set yet.

Write down DINO's winning prompt and threshold. If its validation boxes still alternate between whole trusses and individual fruit, proceed to the trained detector below. Keep the test set unused until all model choices are made.

## 5. Fine-tune RF-DETR Small if DINO is inadequate

RF-DETR accepts the COCO folder layout created above and provides `RFDETRSmall` for fine-tuning: <https://rfdetr.roboflow.com/latest/learn/train/>. The current local PyTorch environment reports `cuda_available=False`, so use a GPU computer or a Google Colab GPU runtime for training. Colab runtime availability may vary. These are pilot settings, not an accuracy guarantee.

**Prepare a training-only ZIP in PowerShell:**

```powershell
Compress-Archive -Path .\data\truss_pilot\train,.\data\truss_pilot\valid -DestinationPath .\data\truss_train_valid.zip -Force
```

Upload `data\truss_train_valid.zip` to your Google Drive. In a new Colab notebook, choose **Runtime → Change runtime type → GPU**. Run the following cells in order:

```python
!pip -q install "rfdetr[train]"
import torch
print("GPU available:", torch.cuda.is_available())
```

Stop if the output says `False`; choose a GPU runtime or another GPU computer.

```python
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/truss_pilot
!unzip -q '/content/drive/MyDrive/truss_train_valid.zip' -d /content/truss_pilot
```

If your ZIP is in a different Drive folder, change that path. Confirm that `/content/truss_pilot/train/_annotations.coco.json` and `/content/truss_pilot/valid/_annotations.coco.json` exist:

```python
from pathlib import Path
for split in ('train', 'valid'):
    print(split, (Path('/content/truss_pilot') / split / '_annotations.coco.json').exists())
```

Both lines must say `True`. Then train:

```python
from rfdetr import RFDETRSmall
model = RFDETRSmall()
model.train(
    dataset_dir='/content/truss_pilot',
    output_dir='/content/drive/MyDrive/rfdetr_truss_pilot',
    epochs=50,
    batch_size=4,
    grad_accum_steps=4,
    early_stopping=True,
    seed=42,
    tensorboard=False,
)
```

If the GPU runs out of memory, retry a new run with `batch_size=2` and `grad_accum_steps=8`. Keep each run's output in a separate Drive folder. The best saved model is expected at `checkpoint_best_total.pth`. Training uses only `train` and `valid` here; the test images are not uploaded yet.

**Choose a counting threshold on validation:**

```python
import json
from collections import Counter
from rfdetr import RFDETRSmall

best = RFDETRSmall(pretrain_weights='/content/drive/MyDrive/rfdetr_truss_pilot/checkpoint_best_total.pth')

def count_errors(split, threshold):
    folder = Path('/content/truss_pilot') / split
    coco = json.loads((folder / '_annotations.coco.json').read_text())
    true_counts = Counter(a['image_id'] for a in coco['annotations'])
    errors = []
    for image in coco['images']:
        predicted = len(best.predict(str(folder / image['file_name']), threshold=threshold))
        actual = true_counts[image['id']]
        errors.append(abs(predicted - actual))
        print(image['file_name'], 'actual:', actual, 'predicted:', predicted)
    print('Mean absolute count error:', sum(errors) / len(errors))

for threshold in (0.3, 0.5, 0.7):
    print('THRESHOLD', threshold)
    count_errors('valid', threshold)
```

Choose one threshold from these validation results and enter it in the next cell. For a visual check of the boxes, run:

```python
from PIL import Image, ImageDraw
chosen_threshold = 0.5  # replace with the validation winner
folder = Path('/content/truss_pilot/valid')
for path in sorted(folder.glob('*.jpg'))[:3]:
    image = Image.open(path).convert('RGB')
    draw = ImageDraw.Draw(image)
    for box in best.predict(str(path), threshold=chosen_threshold).xyxy:
        draw.rectangle(tuple(float(value) for value in box), outline='red', width=5)
    print(path.name)
    display(image)
```

Compare its validation count error and example boxes with DINO. Choose the model and confidence settings using validation only. If you trained RF-DETR, upload the **test** folder only now:

```powershell
Compress-Archive -Path .\data\truss_pilot\test -DestinationPath .\data\truss_test.zip -Force
```

Upload `truss_test.zip` to Drive, then in Colab run:

```python
!unzip -q '/content/drive/MyDrive/truss_test.zip' -d /content/truss_pilot
count_errors('test', chosen_threshold)
```

## 6. Run the final test once

In PowerShell, run DINO on the untouched test split using its previously chosen prompt and threshold. This example is for `a tomato truss.` at `0.25`; replace those values with your validation winner:

```powershell
.\.venv\Scripts\python.exe evaluate_grounding_dino_truss.py --split test --prompt 'a tomato truss.' --threshold 0.25 --output outputs\dino-final-test
```

If you trained RF-DETR, its Colab test call in step 5 gives its count error. Record both test count errors, DINO's missed and extra boxes, and representative errors from each model. Do not change either model's settings after inspecting test results and then describe the same test set as untouched. Twenty test images give a **pilot** comparison, not a final reliability estimate. If both fail, add more labeled training images or obtain representative cart images from a greenhouse partner before trying more architectures.
