# How this comparison supports AgriTwin

## Current purpose, clarified by the user

Compare visual clusters with verified trusses and decide which representation the models identify more reliably. The final choice should support **fruit/flower thinning within a truss**.

The earlier project chats described broader row-level harvest planning. That remains background; the current presentation follows the user's more specific pruning use case.

## Intended workflow for this experiment

Camera photo → identify a visual group or confirmed truss → establish which fruit/flowers belong to it → show counts and condition → grower reviews thinning using agreed rules.

Only the fruit and visual-cluster detection baselines have been run in this checkout. Fruit-to-truss membership, flower detection, a verified-truss model comparison, and thinning recommendations remain future work.

## Proposed definitions

- **Visual cluster:** tomatoes grouped by their nearby appearance, as in the supplied annotations; the user said they appear to share a stem, but membership is unverified.
- **Verified truss:** fruit or flowers confirmed to belong to the same fruiting stem/structure. Unclear connections should remain uncertain rather than guessed.

These distinguish the labeling approaches for an experiment. They are not a claim that growers always distinguish the words: the [UF/IFAS greenhouse tomato handbook](https://ask.ifas.ufl.edu/publication/CV266) uses truss and cluster terminology for the same on-vine product and discusses cluster thinning in terms of the fruit retained and their characteristics. No generic fruit-retention target is being prescribed here.

## What current results establish

- Tomato-specific training improved the individual-fruit baseline on the reviewed development sequence.
- Completing cluster annotations improved the V2 cluster baseline compared with V1 on the same 44 review photos.
- Those findings do not establish whether clusters or trusses are easier to identify, or which supports thinning better.

The next experiment must compare both definitions on common images from the relevant crop stage, with comparable training and independent scenes. The grower-facing check is correct membership within the intended truss, alongside recognition errors. See `TRUSS_CLUSTER_COMPARISON.md`.

## Context reviewed

- **Current “Cluster Testing” chat:** the user clarified the comparison purpose and selected fruit/flower thinning within a truss. This directs the presentation.
- **“Summarize Agritwin weekly status”** — Codex thread `01a0d943-c25d-7852-890c-6386fab9e752`: earlier approved tomato-first direction, vision baselines, row mapping, and supervisor-reviewed picking priorities.
- **“Set up RT-DETR tomato counting”** — Codex thread `01a0eefd-f700-79b3-b66f-425f78909163`: preference for a brief presentation covering original results, training, improvements, and next steps.
- **“Find tomato datasets (3)”** — Codex thread `01a0eed1-2bae-70c1-bcca-1b97f8bbdce0`: public image datasets and missing row metadata.
- Older local documents under `../OLD/drive/Project Docs/` and `../OLD/deliverables/` were reviewed as background.

Previous chats and documents provide context, not proof of completed integration. Numerical findings come from saved experiment files; `dashboard_sources.json` records source hashes. The pruning connection is a proposed application, not a measured benefit.
