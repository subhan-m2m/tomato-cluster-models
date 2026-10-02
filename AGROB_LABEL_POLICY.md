# Tomato label policy for the next experiment

**Status: proposed for team review.** Agree on these rules before correcting labels or collecting the final set. Record any changes here and give the prepared data a new version name. The existing AgRob annotations have been converted automatically and have not been certified by a human reviewer.

## What one box means

- Draw **one tight rectangle around each visible tomato fruit**, regardless of ripeness. Keep a partly hidden fruit when enough of its body is visible to identify it as a distinct tomato. Mark it `occluded` when leaves, stems, or another fruit hide part of it.
- Do not draw a box around a whole cluster, truss, leaf, stem, flower, reflection, or a fruit visible only as an unidentifiable sliver.
- Use the visible outline for the box. If the fruit touches the image edge, keep the visible part and mark it `truncated`.
- Record a confirmed zero-fruit image with its image file and an XML annotation containing **zero `<object>` entries**. A missing XML is an unlabeled image, not a zero count.
- If two fruits overlap, draw two boxes when each fruit can be recognized separately. If the image cannot establish whether there are one or two, flag it for adjudication rather than guessing.

## Class names

The detector learns **one class, `tomato`**. AgRob's four spelling-specific ripeness names (`unriped`, `breaking`, `reddish`, `riped`) all become `tomato` during training; the original names remain in the prepared JSON for analysis. Corrected XML may use those four source names or `tomato`. Do not rename a ripeness class to another fruit class. This first run does not predict ripeness.

## Review method

1. Open each image at full size with its boxes overlaid. The contact sheets are a quick index, not detailed enough to settle small or hidden fruit.
2. Check for **missed fruit**, boxes on leaves, duplicate boxes on one fruit, boxes covering multiple fruit, and misplaced or loose boxes. Check the 14 zero-fruit frames too.
3. Have a second reviewer resolve uncertain cases and write the decision in `review_queue.csv` under `notes`; fill `reviewed_by` and `needs_correction`.
4. Export corrected files as Pascal VOC XML with exactly the same image basename. Keep originals untouched. Put only changed XMLs in `data/agrob_label_overrides_v2/`.
5. Run `prepare_agrob_finetune.py --overrides data/agrob_label_overrides_v2 --output data/agrob_finetune_v2`. The fixed image split stays the same; only corrected labels change. Use the v2 path in later training and evaluation.

## Fresh final evaluation set

Collect a distinct greenhouse row, camera session, or date that is absent from the 449 AgRob frames. Keep all frames from the same capture sequence together. Record the row/date/camera and any duplicate or near-duplicate check. Aim for at least 50 diverse images with green and red fruit, crowding, occlusion, different light, and some confirmed zero-fruit scenes. This is a practical starting target, not a statistical guarantee.

Label those images with the same rules. Put `.jpg` files in `data/final_tomato_v1/JPEGImages/` and one Pascal VOC `.xml` per image in `data/final_tomato_v1/Annotations/`. The `convert_tomato_voc.py` command in the guide checks image/XML pairs and box coordinates before making `final.json`. Do not view the model's predictions on this set until the checkpoint and confidence threshold are fixed.

**Trusses are a separate label project.** AgRob does not say which fruit share a stem or provide truss boxes. A fruit count cannot be converted to a reliable truss count from these labels.
