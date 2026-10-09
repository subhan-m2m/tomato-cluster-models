# Five-minute team walkthrough

Open `tomato_findings_dashboard.html` in Chrome or Edge and click **Present**. Use **Next** or the arrow keys. **Speaker notes** gives short prompts; click any photo to enlarge it. The file works offline and includes 20 photos: 4 original, 4 trained fruit, 4 first-version groups, and 8 full-label groups.

## Walk through these six points

1. **Why this matters to AgriTwin — 40 seconds**
   - “The goal is to help a supervisor decide which greenhouse rows to pick next.”
   - “Counting tells us how much crop is visible. The separate ripeness model tells us how ready it is.”
   - “Later, we will attach those observations to rows so the team can review picking priorities.”

2. **The first try — 30 seconds**
   - “We tried an existing fruit model. It missed the tomatoes in our tested images.”
   - Point to the green manual boxes and the absence of red model boxes in the four photos.
   - “That told us we needed tomato-specific training.”

3. **Teaching it tomatoes — 50 seconds**
   - “Here are the same four photos after training. It now finds tomatoes.”
   - “The average count error dropped from about 12 to about 4 tomatoes per photo.”
   - Show a closer count and the crowded example. “It still misses some tomatoes and adds extra boxes.”

4. **Counting groups — 70 seconds**
   - “We also trained it to count groups, using the labels I created.”
   - “The first group dataset had 116 photos. The completed version has 449.”
   - Switch through the four comparison buttons: left = first group model; right = full-label group model.
   - “The newer version improved on the same review photos, but still makes mistakes.”
   - “These are visual groups that appear to share a stem; we have not confirmed true trusses.”

5. **What works and what still misses — 50 seconds**
   - “With the full labels, it is off by about 2 groups per photo on the supplied test set.”
   - Show the four examples: correct groups, missed groups, extra groups, and an empty scene.
   - “These photos show progress. We need new scenes to check how well it carries over.”

6. **The next project step — 60 seconds**
   - “Agree on the group-label rules and review the remaining mistakes.”
   - “Test new greenhouse photos, then save counts and ripeness with the row and date.”
   - “Connect that to the row map and let a supervisor review the picking priorities.”
   - Close: “We now have the counting baseline and its known weaknesses. The next step is making those observations useful in AgriTwin.”

## If someone asks for more detail

- **What does count error mean?** How far the model’s count is from the manually marked count, averaged across photos. Smaller is better.
- **Does a correct count mean everything was detected correctly?** No. A missed object and an extra box can cancel out. The photos let us check this.
- **Is this ready for use in a greenhouse?** It is a development baseline. Some supplied test photos resemble training photos; an independent new-scene check is still needed.
- **Can we compare fruit and group error directly?** They count different things and use different data. Compare each version with its own earlier version.
- **Is this a yield forecast?** It counts what is visible in each photo. Whole-row totals need to avoid duplicates; harvest-weight forecasting needs harvest records.
- **Is it already connected to the product?** The count-to-row connection is the next integration step, alongside the separate ripeness work.

Exact results and report links are under **Optional: supporting reports and exact results** in the final section. You do not need to show them during the main walkthrough.

## Share and rebuild

Share just **`tomato_findings_dashboard.html`**. All sample photos and counts are embedded; supporting repository links need internet and access.

Rebuild from the repository with the saved local evaluation preview folders available:

```powershell
.\.venv-gpu\Scripts\python.exe build_team_dashboard.py
```

Edit `dashboard_template.html` for presentation wording or layout; `build_team_dashboard.py` selects the photos and loads their actual recorded counts. `dashboard_sources.json` records source hashes, and `dashboard_verification.json` records browser checks. `PROJECT_CONTEXT.md` explains the connection to earlier AgriTwin planning.
