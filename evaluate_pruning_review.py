"""Score completed human membership/count references, including missed clusters.

Unreviewed examples are rejected instead of producing misleading accuracy numbers.
Reference visible counts include detector misses; membership scoring additionally
reports detection coverage, ambiguity and unassigned rates.
"""
import argparse
import json
from pathlib import Path
from pruning_association import assess, iou


def ratio(numerator, denominator):
    return numerator/denominator if denominator else None


def evaluate(reference, maximum, partition):
    samples = [s for s in reference["samples"] if s["partition"] == partition]
    if not samples or any(not s["reviewed"] for s in samples):
        raise ValueError("All examples in the selected partition must be manually reviewed first.")
    metrics = {"images": len(samples), "matched_clusters": 0, "missed_clusters": 0, "extra_clusters": 0,
               "assignable_fruit": 0, "correct_assignments": 0, "assigned_fruit": 0, "detected_fruit": 0,
               "ambiguous_fruit": 0, "unassigned_fruit": 0, "flag_tp": 0, "flag_fp": 0, "flag_fn": 0}
    count_errors = []
    quality_reviews = []
    for sample in samples:
        refs = sample["reference_clusters"]
        if any(type(r["visible_fruit_count"]) is not int or r["visible_fruit_count"] < 0 for r in refs):
            raise ValueError("Every reference cluster needs a reviewed nonnegative visible-fruit count.")
        association = sample["association"]
        predicted = association["clusters"]
        candidates = sorted([(iou(p["box"], r["box"]), p["id"], r["id"]) for p in predicted for r in refs], reverse=True)
        matches = {}
        used = set()
        for overlap, pid, rid in candidates:
            if overlap >= .5 and pid not in matches and rid not in used:
                matches[pid] = rid
                used.add(rid)
        metrics["matched_clusters"] += len(matches)
        metrics["missed_clusters"] += len(refs)-len(used)
        metrics["extra_clusters"] += len(predicted)-len(matches)
        predicted_memberships = {m["fruit_id"]: m for m in association["memberships"]}
        expected_ids = {f["id"] for f in association["tomatoes"]}
        if {m["fruit_id"] for m in sample["memberships"]} != expected_ids or len(sample["memberships"]) != len(expected_ids):
            raise ValueError("Reference memberships must cover every retained detector fruit exactly once.")
        for membership in sample["memberships"]:
            if membership["status"] not in {"assignable", "ambiguous", "unassigned"}:
                raise ValueError("Every fruit needs a manual assignable/ambiguous/unassigned status.")
            if membership["status"] == "assignable" and membership["reference_cluster_id"] not in {r["id"] for r in refs}:
                raise ValueError("Assignable fruit must identify a reference cluster.")
            prediction = predicted_memberships[membership["fruit_id"]]
            metrics["detected_fruit"] += 1
            metrics["assigned_fruit"] += prediction["status"] == "assigned"
            metrics["ambiguous_fruit"] += prediction["status"] == "ambiguous"
            metrics["unassigned_fruit"] += prediction["status"] == "unassigned"
            if membership["status"] == "assignable":
                metrics["assignable_fruit"] += 1
                metrics["correct_assignments"] += matches.get(prediction["cluster_id"]) == membership["reference_cluster_id"]
        by_ref = {r["id"]: r for r in refs}
        results = assess(association, {"maximum": maximum, "version": "evaluation"})
        flagged_refs = set()
        for result in results:
            rid = matches.get(result["cluster_id"])
            if rid:
                count_errors.append(abs(result["visible_count"]-by_ref[rid]["visible_fruit_count"]))
            if result["state"] == "over_limit":
                if rid and by_ref[rid]["visible_fruit_count"] > maximum:
                    metrics["flag_tp"] += 1
                    flagged_refs.add(rid)
                else:
                    metrics["flag_fp"] += 1
        metrics["flag_fn"] += sum(r["visible_fruit_count"] > maximum and r["id"] not in flagged_refs for r in refs)
        quality_reviews.extend({"file_name": sample["file_name"], "cluster_id": r["id"], "quality": r.get("quality", [])} for r in refs if r.get("quality"))
    return {"partition": partition, "maximum": maximum, "association_version": reference["association_version"],
            "settings": reference["settings"], **metrics,
            "assignment_accuracy_on_assignable_detected_fruit": ratio(metrics["correct_assignments"], metrics["assignable_fruit"]),
            "assignment_coverage": ratio(metrics["assigned_fruit"], metrics["detected_fruit"]),
            "ambiguous_rate": ratio(metrics["ambiguous_fruit"], metrics["detected_fruit"]),
            "unassigned_rate": ratio(metrics["unassigned_fruit"], metrics["detected_fruit"]),
            "matched_cluster_count_mae": sum(count_errors)/len(count_errors) if count_errors else None,
            "over_limit_precision": ratio(metrics["flag_tp"], metrics["flag_tp"]+metrics["flag_fp"]),
            "over_limit_recall_including_missed_clusters": ratio(metrics["flag_tp"], metrics["flag_tp"]+metrics["flag_fn"]),
            "quality_reviews": quality_reviews,
            "note": "Pilot geometric membership; does not verify biological trusses or hidden fruit."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--maximum", type=int, required=True)
    parser.add_argument("--partition", choices=("development", "reserved_evaluation"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.maximum < 1:
        parser.error("Use a positive maximum and a new output file.")
    try:
        result = evaluate(json.loads(args.reference.read_text(encoding="utf-8")), args.maximum, args.partition)
    except ValueError as error:
        parser.error(str(error))
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
