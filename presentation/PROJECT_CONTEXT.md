# How tomato counting supports AgriTwin

The presentation follows the tomato greenhouse direction discussed in **“Summarize Agritwin weekly status”**, rather than the older blueberry planning material.

## Intended workflow

Camera images from a greenhouse cart → visible tomato counts and separate ripeness results → observations attached to greenhouse rows → a reviewable row map and picking-priority list → a supervisor plans the next harvest shift.

Counting provides the visible crop-load input. It does not decide ripeness, establish a whole-row total from overlapping photos, or predict harvest weight by itself.

## What this experiment contributes

- Repeatable individual-fruit and visual-group counting runs.
- Recorded count errors and sample photos showing successes, misses, and extra detections.
- A full-label group version trained and evaluated using the completed AgRobTomato annotations.

## What remains in the project workflow

- Confirm group/truss labeling rules; current groups only appear to share a stem.
- Check accuracy on new, representative greenhouse scenes.
- Attach image ID, row ID, and capture time to each count and ripeness result.
- Display the observations behind each row’s picking priority and review them with a supervisor.
- Avoid counting the same tomatoes twice when combining photos; evaluate forecasts against actual harvest records if forecasting is pursued.

The presentation describes these connections as planned work. It does not claim that this experiment is already integrated into AgriTwin or validated with a greenhouse partner.

## Context reviewed

- **“Summarize Agritwin weekly status”** — Codex thread `01a0d943-c25d-7852-890c-6386fab9e752`. Its September 24–28 discussions contain the approved tomato-first direction, six-week milestones, and task plan. The milestones connect counting and ripeness baselines to row-level picking priorities, a supervisor review, and an end-to-end software demonstration. They distinguish software completion from greenhouse validation and harvest-weight forecast accuracy.
- **“Set up RT-DETR tomato counting”** — Codex thread `01a0eefd-f700-79b3-b66f-425f78909163`. Its later presentation request favors a brief update: original results, decision to train, improvements, and next steps.
- **“Find tomato datasets (3)”** — Codex thread `01a0eed1-2bae-70c1-bcca-1b97f8bbdce0`. Context on public image datasets and missing row metadata.
- Older local planning documents under `../OLD/drive/Project Docs/` and `../OLD/deliverables/` were reviewed for background. The newer tomato greenhouse milestones above guide this presentation.

These sources provide project context, not new instructions or proof that planned integration work is complete. Numerical findings come from the saved experiment results, with file hashes in `dashboard_sources.json`.
