# Team presentation: tomato counting findings

Open `tomato_findings_dashboard.html` in Chrome or Edge. It is a standalone file with ten embedded sample images and metrics; you can share just that HTML file. Click **Present** for seven sequential sections and **Speaker notes** for prompts. Use Previous/Next, arrow keys, or the section buttons. Click an image to enlarge it. Source-report links require internet. V2 full-data results are current; V1 subset examples are explicitly marked as history.

## Suggested 8–10 minute walkthrough

1. **Start with the goal — Summary (45 seconds).**
   - “We want visible counts from an image, and a repeatable way to record errors.”
   - “We now have separate runs for individual tomatoes and visual clusters.”
   - State that these are pilots, with remaining accuracy and data limitations.

2. **Define what is being counted — Data & method (1 minute).**
   - Fruit means one box per visible tomato; clusters mean one box per labeled group.
   - The group appears to share a stem, but botanical membership is unverified.
   - Fruit training used 237 images and 40 validation images.
   - The completed cluster export has **449 images and 2,985 boxes**: 360 training, 44 validation, and 45 supplied test images.
   - All 116 earlier cluster images keep the same boxes and split assignments. V2 adds 333 images, including 19 zero-cluster images across the full dataset (15 train, 3 validation, 1 test).
   - Explain validation as practice data for choosing settings, then evaluation as scoring fixed settings.
   - Both fine-tuning runs used standard DETR, 15 epochs, 480-pixel resizing, and the local RTX 3070.

3. **Show the starting point — Original baseline (1 minute).**
   - “At confidence 0.30, the original Tomato class produced zero detections on 152 images.”
   - “Count error was 12.118 fruit per image; greenhouse adaptation was necessary.”
   - Show the frame with five green labels and no predictions.

4. **Show individual-fruit progress — Fruit results (1–2 minutes).**
   - “On that same development sequence, count error fell to 4.375 fruit per image.”
   - Show the same frame after training: five labels, six predictions, five matched boxes.
   - “Validation error was 2.625. Development precision was 40.8%, recall 53.9%.”
   - “It still overcounts by 3.901 fruit/image on average. These development images were already viewed, and fruit label completeness needs review.”

5. **Show full-data cluster results — Cluster results (1–2 minutes).**
   - “We trained a new cluster model on the completed annotations and froze confidence 0.90 on validation before testing.”
   - “On 45 supplied test images, count error is **1.933 clusters/image**, with average undercount **1.356**.”
   - “Box precision is **79.2%**, recall **66.6%**, and eight images have exact counts.”
   - “On the same 44 earlier test frames with unchanged labels, error improved from **2.341 to 1.977** clusters/image. Precision improved from 49.7% to 79.2%, and recall from 41.0% to 66.6%.”
   - The 45th test image is an added empty scene with zero detections. It slightly lowers the full-test average, which is why the dashboard also compares the same 44 frames.
   - These frames were already inspected. The result is a development comparison, not an independent final accuracy claim.
   - Explain that fruit and cluster MAEs have different units and datasets, so they cannot rank these models.

6. **Make the errors visible — Image review (2 minutes).**
   - Select **fine-tuned side 0080**: identify overlapping red boxes and possible unlabeled fruit.
   - Select **V2 full · Barroselas 0083**: compare with its V1 example. V2 recovers 15 labeled groups versus 11, but a near-correct total still hides misses and extra boxes.
   - Select **V2 full · Barroselas 0044**: point to missed small/hidden groups and partial boundaries.
   - Select **V2 full · Barroselas 0119**: show overlapping partial-group boxes and the remaining overcount.
   - Select **V2 full · Empty-scene frame 0150**: zero labels and zero predictions. One empty test image is not enough to measure false alarms broadly.
   - Explain green = supplied labels, red = predictions. A “matched” box overlaps sufficiently with one supplied label.
   - Say that these selected examples show failure types; their frequency has not been manually measured.

7. **Close with decisions — Next decisions (1–2 minutes).**
   - Agree on singleton, occlusion, image-edge, and unclear-stem labeling rules.
   - Review the completed labels around remaining failures; record any corrections in a new version.
   - Keep nearby frames/identical plants together when splitting data. V2 has **275 nearby-frame pairs across splits**, including 136 train/test pairs.
   - Reserve a fresh fully labeled final set, including empty scenes. The inspected set becomes development material.
   - Keep this V2 480-pixel result as the recorded baseline, then try 640-pixel input as one controlled change. Select settings on validation.
   - Agree on an acceptable count-error target before claiming task completion.

## Short closing statement

“The completed cluster annotations are incorporated. V2 improves both count error and box matching on the same development frames, but small hidden groups and extra detections remain. I propose we review these failures and reserve independent scenes before claiming final accuracy.”

## Rebuild the dashboard

From the repository, with the original fruit preview folders and the tracked V1/V2 cluster reviewed previews present:

```powershell
.\.venv-gpu\Scripts\python.exe build_team_dashboard.py
```

The template is `presentation/dashboard_template.html`. The generated dashboard contains all required content; the template alone is not the shareable presentation. `dashboard_sources.json` records source file hashes. Only the corrected dynamic-padding fruit results are used.
