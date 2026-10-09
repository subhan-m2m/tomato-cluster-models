# Five-minute team walkthrough

Open `tomato_findings_dashboard.html` in Chrome or Edge and click **Present**. Use **Next** or the arrow keys. **Speaker notes** gives short prompts; click any photo to enlarge it. The file works offline and includes 20 photos: 4 original, 4 trained fruit, 4 first-version clusters, and 8 full-label clusters.

## Walk through these six points

1. **The decision we want to make — 45 seconds**
   - “We want to compare visual clusters with verified trusses, then choose what the model can recognize reliably.”
   - “The purpose is to help a grower review fruit or flower thinning within a truss.”
   - “Nearby fruit can belong to different trusses. The grouping must preserve which fruit or flowers belong together.”

2. **The original fruit model — 30 seconds**
   - “The starting fruit model missed the tomatoes in our tested images.”
   - Show the green manual boxes and absence of red model boxes.
   - “That told us tomato-specific training was needed.”

3. **After teaching it tomatoes — 45 seconds**
   - “The model now finds tomatoes. On the same review images, average count error fell from about 12 to about 4 per photo.”
   - Show a closer count and the crowded example with extra boxes.
   - “This could later help count fruit within a truss, once membership is established.”

4. **The cluster baseline improved — 60 seconds**
   - “Completing the cluster annotations grew the dataset from 116 to 449 photos.”
   - Switch through the four before-and-after comparison photos.
   - “Both sides are cluster models. These results show improvement from more labeled data; they do not yet compare clusters against verified trusses.”

5. **What still needs checking — 45 seconds**
   - Show correct groups, missed groups, extra groups, and an empty scene.
   - “The newer cluster count is off by about 2 groups per photo on the supplied test set.”
   - “For thinning, we must also check whether a group mixes separate trusses or splits one truss into several groups.”
   - “The current labels do not confirm stem membership or cover flowers.”

6. **How we choose between the approaches — 75 seconds**
   - “Use the same new photos from the actual thinning stage, with flowers and young fruit.”
   - “Label visual groups and confirmed trusses separately, and record which fruit or flowers belong to each truss.”
   - “Compare recognition errors and whether the grouping is useful to a grower reviewing thinning.”
   - Close: “We have a cluster baseline. The next step is the truss comparison, so the final choice supports the intended pruning task.”

## If someone asks for more detail

- **Have we proven clusters are better than trusses?** No. We have compared two cluster versions. Verified truss labels and a comparable model run are still needed.
- **What is the distinction?** For this experiment, visual clusters group objects by appearance; trusses require confirmed shared fruiting-stem membership. Growers sometimes use the terms interchangeably, so we must agree on the label rules.
- **What does count error mean?** How far the model’s count is from the manually marked count, averaged across photos. Smaller is better.
- **Why is count accuracy insufficient?** Two neighboring trusses could be merged, or one truss split into two groups. A correct overall count can also hide a miss and an extra detection.
- **Which should we choose for thinning?** Choose the approach that reliably preserves the intended truss and its members. A cluster box may serve as a candidate region, but it must be checked against confirmed membership.
- **Can it tell us what to remove now?** Current outputs are fruit/group boxes and counts. Flower detection, membership, condition, and grower-agreed thinning rules are additional work.

Exact results and report links are under **Optional: supporting reports and exact results** in the final section. You do not need to show them in the main walkthrough. The proposed experiment is in `TRUSS_CLUSTER_COMPARISON.md`.

## Share and rebuild

Share just **`tomato_findings_dashboard.html`**. All sample photos and counts are embedded; supporting repository links need internet and access.

Rebuild from the repository with the saved local evaluation preview folders available:

```powershell
.\.venv-gpu\Scripts\python.exe build_team_dashboard.py
```

Edit `dashboard_template.html` for wording or layout; `build_team_dashboard.py` selects photos and loads their recorded counts. `dashboard_sources.json` records source hashes, and `dashboard_verification.json` records browser checks. `PROJECT_CONTEXT.md` records the current use case and earlier context.
