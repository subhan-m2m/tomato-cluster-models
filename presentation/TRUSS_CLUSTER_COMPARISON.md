# Visible clusters as truss proxies: labeling and evaluation plan

Updated October 13 following the user's clarification about hidden stems. This replaces the earlier plan requiring two fully labeled versions of the same obstructed images.

## Team explanation

“Stems are usually hidden by leaves, so we annotate compact fruit groups that appear to grow from the same place. We use these visible clusters as inferred truss proxies. We can test how well the model finds those groups, then check whether their visible fruit information helps a grower review thinning.”

## What the labels mean

- **Detection target:** a compact visible fruit group whose arrangement suggests a shared fruiting origin.
- **Model class:** keep `tomato_cluster` for consistency with existing data and checkpoints.
- **Intended interpretation:** an inferred truss proxy, with unconfirmed biological membership.
- **Measured result:** visible cluster counts and detections compared with the supplied visual labels.

Seeing the entire plant or its main stem is not required. Existing valid boxes can remain. If label boundaries are unchanged, this clarification does not change the trained model or its recorded scores.

## Practical labeling rules

1. **Box the compact fruit group.** Keep a reasonably tight rectangle around its visible fruit. Stems do not have to be visible or included.
2. **Use apparent shared origin as a visual cue.** Consider the arrangement and spacing together. Avoid joining clearly separate groups merely because their fruit overlap in the image.
3. **Keep group boundaries consistent.** Do not enclose the entire plant or require vertical boxes. Review possible merges or splits using the same visual rule.
4. **Record genuine visual ambiguity.** Flag cases where one versus two groups cannot be decided consistently. Hidden stems alone do not make an otherwise clear visual cluster unsuitable for this detection target.
5. **Keep edge and singleton treatment explicit.** Preserve the current dataset's treatment for the baseline. Agree and document any changes before creating a new label version. Box only visible content; do not invent hidden fruit extent or count.
6. **Keep uncertainty separate from the object name.** Use review notes for ambiguous grouping and partial visibility. A class rename to “truss” cannot establish stem connections.

## Sequential next steps

1. **Check label consistency on a varied sample.** Review roughly 20–30 photos with another annotator or grower. Include crowded scenes, partial groups, and apparent shared origins. Check whether reviewers agree on the visible group boundaries; this is not confirmation of invisible stems. A sample audit identifies problems but does not prove every annotation is correct.

2. **Keep the existing baseline.** Correct only demonstrated labeling inconsistencies in a new dataset version. Retrain when labels or another experimental setting change, while preserving the current recorded results. There is no need to reconstruct every box simply because the stems are hidden.

3. **Evaluate detection on new scenes.** Keep views of the same plants together when splitting data. Freeze settings before the test. Record missed and extra groups, count error, and manually reviewed merge/split examples against the visual-label policy. These scores measure agreement with visual groups rather than actual trusses.

4. **Add visible fruit per group.** Combine individual-fruit and cluster detections. Record or flag ambiguous associations where groups overlap; box containment alone can be insufficient. Evaluate associations against human-visible-group assignments. Hidden fruit remain outside the measured visible count.

5. **Review the thinning use case.** Use photos from the stage when thinning is considered. Ask a grower whether the displayed group, visible members, and their condition are useful for review. Flower-stage use needs flower/young-fruit examples and evaluation; the current fruit-only cluster run does not demonstrate it. Establish the grower's rules before evaluating a recommendation.

6. **Validate biological correspondence if better evidence becomes available.** Additional views, video revealing connections, or observations made at the plant can confirm actual truss membership on a separate reference sample. Record the evidence source. Only then measure how often a visual proxy corresponds to one biological truss. This is additional evidence rather than a label requirement that the current images cannot satisfy.

## What we can compare now

| Question | Evidence |
| --- | --- |
| Did full annotations improve cluster detection? | V1/V2 results on the same 44 review photos. |
| Does the model agree with visible group labels on new scenes? | Next independent detection evaluation. |
| Do annotators agree on the visual grouping? | Label-consistency audit. |
| Can fruit be assigned to the visible groups reliably? | Planned fruit-to-group evaluation. |
| Does the output help a grower review thinning? | Planned review on appropriate crop-stage photos. |
| Does every proxy correspond to one true truss? | Unmeasured; requires evidence of actual connections. |

Fruit counts and cluster counts use different units; their average errors cannot directly rank the approaches. The former describes visible members and the latter identifies candidate group regions. No automated thinning recommendation or cut point is produced by the current experiment.

## Background

The [UF/IFAS greenhouse tomato handbook](https://ask.ifas.ufl.edu/publication/CV266) discusses truss/cluster terminology and fruit thinning in relation to fruit characteristics and cultivar. This supports defining the intended grower decision, but does not establish that our visual groups correspond to true trusses or that their detections are suitable for thinning.
