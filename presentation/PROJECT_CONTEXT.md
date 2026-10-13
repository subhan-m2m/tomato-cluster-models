# How visible clusters support AgriTwin

## Current purpose and image constraint

The user wants a useful representation for **fruit/flower thinning within a truss**. They clarified on October 13 that stems are almost never visible because leaves block them. Their cluster annotations enclose tightly packed fruit that appear to originate from one place; they do not enclose all fruit on a plant.

Given that constraint, the practical detection target is a **visible fruit cluster used as an inferred truss proxy**. Hidden stem connections remain unconfirmed. A separate verified-truss dataset cannot be reliably created from these same obstructed images alone, so making it a prerequisite for continuing the cluster baseline is inappropriate.

The broader earlier project chats described row-level harvest planning. The presentation now follows this specific thinning use case and the available image evidence.

## Intended workflow

Photo → detect compact fruit groups → associate visible individual fruit with each group → show visible counts and photo evidence → grower reviews whether that information supports thinning.

The fruit and visual-cluster detection baselines have been run. Fruit-to-group association, flower detection, and validated thinning support remain future work. Hidden fruit and actual truss membership are not measured by the current boxes.

## Working label definition

**One compact visible fruit group whose arrangement suggests a shared fruiting origin.** Stem visibility is not required. Keep the model class `tomato_cluster`; describe its intended application as an inferred truss proxy.

Detection agreement with visual labels and correspondence to biological trusses are separate questions. Extra camera views or field observations could establish the latter later. A grower reviewing the same obstructed photo can assess usefulness or plausibility, but cannot supply reliable hidden-connection ground truth solely from that photo.

## What current results establish

- Tomato-specific training improved the individual-fruit baseline on the reviewed development sequence.
- Full cluster annotations improved V2 compared with V1 on the same 44 review photos.
- Those are improvements in visual detection and count agreement. They do not establish true truss-count accuracy, full fruit load, flower-stage performance, or a benefit from thinning recommendations.

The next steps are a label-consistency review, new-scene detection evaluation, visible fruit-per-group association, and a grower review using images from the intended crop stage. See `TRUSS_CLUSTER_COMPARISON.md` for the updated plan.

## Context reviewed

- **Current “Cluster Testing” chat:** fruit/flower thinning is the intended use case; stems are obscured and compact fruit groups are the proposed substitute. This latest clarification directs the presentation.
- **“Summarize Agritwin weekly status”** — Codex thread `01a0d943-c25d-7852-890c-6386fab9e752`: earlier tomato-first vision baselines, row mapping, and supervisor-reviewed picking priorities.
- **“Set up RT-DETR tomato counting”** — Codex thread `01a0eefd-f700-79b3-b66f-425f78909163`: preference for a brief presentation of original results, training, improvements, and next steps.
- **“Find tomato datasets (3)”** — Codex thread `01a0eed1-2bae-70c1-bcca-1b97f8bbdce0`: public datasets and missing row metadata.
- Older documents under `../OLD/drive/Project Docs/` and `../OLD/deliverables/` were reviewed as background.

The [UF/IFAS greenhouse tomato handbook](https://ask.ifas.ufl.edu/publication/CV266) discusses truss/cluster terminology and fruit thinning. It does not validate our visual proxy or model performance. Numerical results come from saved experiment files; `dashboard_sources.json` records source hashes. The thinning connection remains a proposed application.
