# Team presentation: tomato counting findings

Open `tomato_findings_dashboard.html` in Chrome or Edge. It is a standalone file with six embedded sample images and metrics; you can share just that HTML file. Click **Present** for seven sequential sections and **Speaker notes** for prompts. Use Previous/Next, arrow keys, or the section buttons. Click an image to enlarge it. Source-report links require internet.

## Suggested 8–10 minute walkthrough

1. **Start with the goal — Summary (45 seconds).**
   - “We want visible counts from an image, and a repeatable way to record errors.”
   - “We now have separate runs for individual tomatoes and visual clusters.”
   - State that these are pilots, with remaining accuracy and data limitations.

2. **Define what is being counted — Data & method (1 minute).**
   - Fruit means one box per visible tomato; clusters mean one box per labeled group.
   - The group appears to share a stem, but botanical membership is unverified.
   - Fruit training used 237 images and 40 validation images; cluster training used 54 and 18.
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

5. **Show the cluster result — Cluster results (1 minute).**
   - “We used the relabeled COCO subset and froze confidence 0.90 on validation before testing.”
   - “Across 44 supplied test images, count error was 2.341 clusters/image; average undercount was 1.523.”
   - “Precision was 49.7%, recall 41.0%. Eight exact counts still had unmatched boxes.”
   - Explain that fruit and cluster MAEs have different units and datasets, so they cannot rank these models.

6. **Make the errors visible — Image review (2 minutes).**
   - Select **fine-tuned side 0080**: identify overlapping red boxes and possible unlabeled fruit.
   - Select **cluster Barroselas 0083**: point to missed shaded and partly hidden groups.
   - Select **cluster side 0137**: show a green cluster split into multiple red detections.
   - If time permits, show **Barroselas 0177** for predictions covering only part of a group.
   - Explain green = supplied labels, red = predictions. A “matched” box overlaps sufficiently with one supplied label.
   - Say that these selected examples show failure types; their frequency has not been manually measured.

7. **Close with decisions — Next decisions (1–2 minutes).**
   - Agree on singleton, occlusion, image-edge, and unclear-stem labeling rules.
   - Assign label review and record corrections in a new version.
   - Keep nearby frames/identical plants together when splitting data. The cluster splits contain 42 nearby-frame pairs across splits.
   - Reserve a fresh fully labeled final set, including empty scenes. The inspected set becomes development material.
   - Repeat the 480-pixel run, then try 640-pixel input as one controlled change. Select settings on validation.
   - Agree on an acceptable count-error target before claiming task completion.

## Short closing statement

“The pipeline runs, and the count errors and examples are recorded. Fine-tuning helped individual-fruit detection, while cluster detection still misses and splits groups. I propose we agree on the annotation rules, complete label review, and reserve independent scenes before the next accuracy claim.”

## Rebuild the dashboard

From the repository, with the original evaluation preview folders present:

```powershell
.\.venv-gpu\Scripts\python.exe build_team_dashboard.py
```

The template is `presentation/dashboard_template.html`. The generated dashboard contains all required content; the template alone is not the shareable presentation. `dashboard_sources.json` records source file hashes. Only the corrected dynamic-padding fruit results are used.
