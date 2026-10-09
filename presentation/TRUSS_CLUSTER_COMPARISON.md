# Trusses versus visual clusters: the next experiment

## Team explanation

“We want to choose what the camera should recognize to support fruit or flower thinning. A visible cluster is a group that looks close together; a verified truss is a group confirmed to belong to one fruiting stem. We will test which approach is detected reliably and keeps the correct fruit or flowers together for a grower to review.”

## Current evidence

| Approach | Current status | What we can conclude |
| --- | --- | --- |
| Individual tomatoes | Original and trained models evaluated | Training helped; some misses and extra boxes remain. |
| Visual clusters | V1 subset and V2 full-label models evaluated | Full labels improved the cluster baseline. |
| Verified trusses | No comparable run or verified membership labels in this checkout | Truss-versus-cluster winner is undetermined. |

Both current group versions use visual-cluster labels. Renaming the category to “truss” would not establish biological membership or create a meaningful comparison.

## Sequential comparison

1. **Choose the operating situation with a grower.** Use photos from the actual fruit/flower thinning stage. Record the crop variety and the information needed to review thinning. Include examples with neighboring trusses and obscured stems, as well as clear views. The current fruit-heavy photos do not establish flower-stage performance.

2. **Write a one-page labeling rule.** Agree what the box covers in each approach. A cluster version surrounds the visible group by appearance. A truss version identifies a confirmed fruiting unit; specify whether its box includes the supporting stem. Use stable IDs to associate each visible fruit/flower with its verified truss. Mark hidden or unclear connections as uncertain. Box containment alone cannot prove membership when trusses overlap.

3. **Label the same images twice.** Keep the existing cluster labels intact. Create a separate verified-truss version on a common set, including individual fruit/flower members where visible. Have a grower or knowledgeable annotator check those relationships. If multiple camera views are needed to confirm a stem, record that confirmation separately from what is visible in the evaluation image.

4. **Keep the comparison fair.** Use the same image identities, training/validation/test assignments, starting model, and comparable training effort. Keep related views of the same plants together. Select settings using validation images and freeze them before scoring genuinely new test scenes. Record labeling effort as another practical consideration.

5. **Score recognition and the intended grouping.** For each representation, record missed objects, extra predictions, and count error against its own labels. Separately compare both approaches against the verified truss-member reference: does a predicted unit mix members from different trusses, split one truss, or omit visible members? Raw count errors across different definitions cannot by themselves select the useful representation. Do not treat total-count agreement as proof of correct membership.

6. **Review usefulness for thinning.** Show the same scenes with each output to a grower. Ask whether they can identify the intended truss, review its visible fruit/flowers, and distinguish neighboring trusses. Record unreviewable cases. Counts alone do not determine removal: member condition and grower-agreed rules are further inputs.

7. **Choose using the evidence.** Favor the approach that achieves reliable recognition and preserves the grouping required by the thinning task. A visual cluster can be a useful candidate region if it maps to the correct truss. If it repeatedly merges different trusses, a better-looking detection score is insufficient. If neither approach supports a reliable review, document that outcome and improve images or labels before choosing.

## Simple scorecard for the team

| Question | Visual-cluster approach | Verified-truss approach |
| --- | --- | --- |
| How often is the intended object missed? | Pending fair comparison | Pending fair comparison |
| How often are different trusses mixed together? | Pending membership review | Pending membership review |
| How often is one truss split into several groups? | Pending membership review | Pending membership review |
| Can visible fruit/flowers be assigned correctly? | Pending | Pending |
| Can a grower review thinning from the evidence? | Pending | Pending |
| How much labeling effort is needed? | To record | To record |

No pruning recommendation or automated cut point is produced by the current experiment.

## Background

The [UF/IFAS greenhouse tomato handbook](https://ask.ifas.ufl.edu/publication/CV266) describes overlapping truss/cluster terminology and discusses fruit thinning within the group, with choices affected by cultivar and fruit characteristics. This motivates checking actual membership and defining the task with a grower; it does not establish which model representation will perform best.
