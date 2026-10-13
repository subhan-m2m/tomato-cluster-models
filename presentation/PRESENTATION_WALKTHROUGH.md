# Five-minute team walkthrough

Open `tomato_findings_dashboard.html` in Chrome or Edge and click **Present**. Use **Next** or the arrow keys. **Speaker notes** gives short prompts; click any photo to enlarge it. The file works offline and includes 20 photos: 4 original, 4 trained fruit, 4 first-version clusters, and 8 full-label clusters.

## Walk through these six points

1. **Why we use visible clusters — 45 seconds**
   - “Leaves usually hide the stems, so the images cannot reliably show actual truss connections.”
   - “We label compact fruit groups that appear to grow from one place and use them as inferred truss proxies.”
   - “The intended use is useful visible group information for a grower reviewing fruit or flower thinning.”

2. **The original fruit model — 30 seconds**
   - “The starting fruit model missed the tomatoes in our tested images.”
   - Show the green manual boxes and absence of red model boxes.
   - “That told us tomato-specific training was needed.”

3. **After teaching it tomatoes — 45 seconds**
   - “The model now finds tomatoes. On the same review images, average count error fell from about 12 to about 4 per photo.”
   - Show a closer count and the crowded example with extra boxes.
   - “This could later help count the visible fruit within each cluster.”

4. **The cluster baseline improved — 60 seconds**
   - “Completing the cluster annotations grew the dataset from 116 to 449 photos.”
   - Switch through the four before-and-after comparison photos.
   - “Both sides detect the same kind of visible group. More labeled data improved agreement with our visual annotations.”

5. **What still needs checking — 45 seconds**
   - Show correct groups, missed groups, extra groups, and an empty scene.
   - “The newer cluster count is off by about 2 groups per photo on the supplied test set.”
   - “We also need to check whether the model merges visually separate groups or splits one labeled group.”
   - “These are visible cluster results. Stem membership, hidden fruit, and flower-stage performance are not measured.”

6. **How we make the proxy useful — 75 seconds**
   - “Keep consistent visible-group labels and test detection on new greenhouse scenes.”
   - “Connect individual fruit detections to groups, and flag assignments that are unclear.”
   - “Review the output with a grower using photos from the actual thinning stage.”
   - Close: “We have an improved visible-cluster baseline. Next we check whether group-level observations support thinning review.”

## If someone asks for more detail

- **Have we proven the clusters are actual trusses?** No. They are inferred truss proxies. Hidden connections cannot be established reliably from the current images.
- **Do we need a separate truss annotation pass on these photos?** We can continue with the visible-group target. Better views or field observations would be needed for reliable biological-membership labels.
- **What exactly is a cluster here?** A compact fruit group that appears to grow from one place. The model class stays `tomato_cluster`.
- **What does count error mean?** How far the model’s count is from the manually marked count, averaged across photos. Smaller is better.
- **Why is count accuracy insufficient?** Visually separate groups could be merged or one group split into several detections. A correct total can hide a miss and an extra detection.
- **Does a grower review verify the hidden stems?** It can assess usefulness and plausibility. Actual connection evidence still requires another view or an observation at the plant.
- **Can it tell us what to remove now?** Current outputs are fruit/group boxes and counts. Fruit-to-group association, flower-stage detection, condition, and grower-agreed rules are additional work.

Exact results and report links are under **Optional: supporting reports and exact results** in the final section. You do not need to show them in the main walkthrough. The updated proxy-labeling and evaluation plan is in `TRUSS_CLUSTER_COMPARISON.md`.

## Share and rebuild

Share just **`tomato_findings_dashboard.html`**. All sample photos and counts are embedded; supporting repository links need internet and access.

Rebuild from the repository with the saved local evaluation preview folders available:

```powershell
.\.venv-gpu\Scripts\python.exe build_team_dashboard.py
```

Edit `dashboard_template.html` for wording or layout; `build_team_dashboard.py` selects photos and loads their recorded counts. `dashboard_sources.json` records source hashes, and `dashboard_verification.json` records browser checks. `PROJECT_CONTEXT.md` records the current use case and earlier context.
