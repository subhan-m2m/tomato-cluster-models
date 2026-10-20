# Tomato detection findings and branch guide

Updated October 19, 2026. These are saved experiment results; this documentation cleanup did not retrain models.

## Findings

| Experiment | Recorded result | Interpretation |
| --- | --- | --- |
| Original fruit DETR | No Tomato-class detections; count MAE 12.118 fruit/image on 152 frames. | The pretrained checkpoint needed tomato-specific training. |
| Fine-tuned fruit DETR | Count MAE 4.375 fruit/image on the same 152-frame development sequence. | Training improved counting, but missed, duplicate, and misplaced boxes remain. |
| Cluster V1 | Count MAE 2.341 clusters/image on 44 supplied evaluation frames. | First baseline using the relabeled subset. |
| Cluster V2 | Count MAE 1.933 clusters/image on 45 supplied evaluation frames; precision 79.19%, recall 66.58% at IoU 0.50. | Current cluster baseline, trained with the full 449-image / 2,985-box export. |

On the same 44 frames with unchanged labels, cluster MAE improved from **2.341 to 1.977**. The 45-frame V2 result also includes one empty image. MAE means average absolute count error, not detection accuracy. The trained fruit and cluster thresholds were selected on validation data and fixed at 0.90.

**Limits:** the fruit development frames were previously viewed; cluster evaluation frames were reused; 275 nearby-frame pairs cross V2 splits. These are development benchmarks, not independent new-scene performance estimates. Compact visible clusters are inferred truss proxies; their hidden stem connections are unverified. The fruit model uses the original AgRob labels, without a completed correction record in this checkout.

## Branches

| Branch | Purpose |
| --- | --- |
| `main` | Original repository foundation. |
| `experiment/agrob-tomato-count` | Pretrained AgRob fruit-count baseline. |
| `experiment/agrob-finetune` | Corrected tomato-specific GPU training and evaluation. |
| `experiment/agrob-cluster-count` | Current working experiment: fruit workflow, cluster V1/V2, reports, and offline presentation. Start here for continued model work. |
| `archive/truss-pilot-before-agrob-cleanup` | Separate earlier Laboro/Grounding DINO/truss-pilot work, retained as historical research. It is not the current AgRob workflow. |

The experiment stages and the separate archive are intentionally preserved. Raw datasets and model weights remain in ignored local folders; cloning a branch does not download checkpoints.

## Next application

Combine fruit boxes with cluster boxes to estimate **visible fruit per group**, then flag groups above a grower-configured maximum for thinning review. Evaluate assignments against manually reviewed membership examples, allow ambiguity, and count each fruit at most once. This association and alert layer is planned, not implemented by these experiments.

The ripeness detector could supply fruit boxes and maturity labels once its per-fruit response is available. Plant detection can later locate the parent plant; no plant-model result is recorded in these branches. Neither cluster counts nor plant boxes alone establish whole-plant overload or a harvest-quality benefit.

Details: [fruit run](AGROB_FINETUNE_RUN_2026-10-01.md), [cluster V2 run](AGROB_CLUSTER_V2_RUN_2026-10-09.md), [offline presentation](presentation/tomato_findings_dashboard.html).
