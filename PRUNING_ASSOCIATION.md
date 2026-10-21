# Cluster pruning review prototype

Branch: `feature/pruning-association-prototype`. Pair with frontend branch `feature/pruning-review-prototype` in the sibling frontend checkout.

`pruning_association.py` is the shared, dependency-free association and assessment implementation. It consumes oriented xyxy pixel boxes; it never assigns greenhouse locations. The frontend service imports this module directly. Training scripts, datasets, and weights are unchanged.

## Frozen prototype contract

Version `visible-association-v1`, detection threshold **0.90** for each existing checkpoint. Association parameters are separate from that threshold and the grower's crop limit:

| Parameter | Value | Meaning |
| --- | --- | --- |
| Expansion per box side | 0% | Conservative initial baseline; the implementation bounds experimental margins to 0–10%. |
| Minimum fruit area coverage | 50% | Fruit center must also be inside the candidate region. |
| Winner separation | 0.15 | Require this score difference from the next candidate; a tie remains ambiguous. |
| Fruit duplicate IoU | 0.95 | Suppress only near-identical boxes in descending confidence order; retain ordinary overlapping fruit. Raw service results remain available in each record. |
| Group overlap IoU | 0.55 | Flag overlapping or merged groups; containment of 90% of the smaller group also flags them. |
| Image-edge tolerance | 1 pixel | An edge-clipped cluster needs review. |

Original-box support receives score `1 + original fruit coverage`; an expansion-only candidate receives its expanded coverage. Thus original support outranks expansion-only support. Stable IDs determine display order, never ownership in a genuine tie. Every retained fruit is assigned once, ambiguous with its candidates, or unassigned. Original cluster boxes remain the displayed evidence boxes.

Clearly assigned detections above the configured limit produce **Over limit — review thinning**, including an explicit excess count. An ambiguous candidate that could cross the limit, or a group quality flag, produces **Association uncertain**. A clear lower bound already above the limit remains flagged, with its quality warning shown separately. Missing rules or incomplete inference produce **Not evaluated**. Empty cluster results do not establish that no pruning is needed.

The range is a count of detected candidates, not a confidence interval or bound on hidden fruit. No biological truss verification, cutting points, maturity, unique cross-photo tracking, or whole-plant load is inferred.

## Review pack and remaining validation

The local pack is `outputs/pruning-review-v1/index.html` with annotated previews, source photos, stable IDs, and `reference.json`. Its 24 images come from the existing V2 validation set: 12 Barroselas development examples and 12 side-camera examples reserved for association evaluation. It includes an empty outdoor frame, clipped groups, missed groups, and overlapping predictions. No existing test images were used to choose association settings. These existing video data are a pilot; nearby frames were used in model training, so this is **not** a fresh generalization benchmark.

The reference currently contains **unreviewed placeholders**, not invented memberships or count labels. The tracked `results/pruning_prototype/review_manifest.json` preserves filenames, source hashes, model hashes, partitions, settings, and runtime measurements. The large local pack stays ignored. Visual triage of development previews 01, 05, and 08 found:

- 01: at least one annotated small group lacks a cluster prediction, while some predicted boxes cover only part of the reference group. A close total cannot establish correct membership.
- 05: several small or shaded groups have no individual-fruit detections, an overlapping group pair is present, and an edge group is clipped. Association cannot recover missed fruit.
- 08: the outdoor scene has no detected fruit or clusters; the application must show **No clusters detected / Not evaluated**.

The development-only comparison at margins 0%, 3%, and 5% is saved in `development_margin_comparison.json`. Expansion adds one assignment among 212 detected fruit, with 11 still ambiguous. This is coverage information, **not measured accuracy**; retain the zero-expansion baseline until manual references justify changing it. Any change requires a new association version and evaluation before replacing historical results.

To regenerate the pack from this model directory:

```powershell
& .venv-gpu\Scripts\python.exe prepare_pruning_review.py --output outputs/pruning-review-new
```

Inspect source photos and edit the pack's `reference.json`: correct/add reference cluster boxes, fill every reference cluster's `visible_fruit_count`, label every retained fruit as `assignable`, `ambiguous`, or `unassigned`, provide `reference_cluster_id` for assignable fruit, and record reviewer/notes plus missed/merged/split/duplicate issues in `quality`. Set `reviewed: true` only after completing an image. Do not copy inferred membership into the reference. Freeze settings on development examples before scoring the reserved scene; use fresh scene groups for a subsequent greenhouse acceptance trial.

```powershell
& .venv-gpu\Scripts\python.exe evaluate_pruning_review.py outputs/pruning-review-v1/reference.json --maximum 5 --partition development --output outputs/pruning-development-metrics.json
& .venv-gpu\Scripts\python.exe evaluate_pruning_review.py outputs/pruning-review-v1/reference.json --maximum 5 --partition reserved_evaluation --output outputs/pruning-evaluation-metrics.json
```

The scorer refuses unreviewed references. It reports assignment accuracy on assignable detected fruit alongside coverage/ambiguity, matched-cluster count error, missed/extra clusters, and over-limit precision/recall including missed clusters. Matching uses greedy one-to-one IoU ≥0.50; inspect split/merged failures explicitly. These metrics are not agronomic outcome validation. Agree acceptable miss/false-alert thresholds with the team after the reference review.

## Verification

```powershell
& .venv-gpu\Scripts\python.exe -m unittest test_pruning_association test_pruning_review
```

Tests cover boundaries, competing clusters, expansion precedence, near-identical duplicate handling versus overlapping real fruit, edge clipping, equal-to-limit, missing rule, partial inference, deterministic order, empty results, ambiguity crossing the crop limit, and scoring missed groups in flag recall.
