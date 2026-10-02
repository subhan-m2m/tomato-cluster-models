"""Summarize count and box errors from a saved AgRob fine-tune evaluation."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from evaluate_agrob_fruit import match_boxes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    args = parser.parse_args()
    source = args.evaluation / "predictions.jsonl"
    if not source.is_file():
        parser.error(f"Predictions are missing: {source}")
    if (args.evaluation / "error_analysis.json").exists():
        parser.error("error_analysis.json already exists; preserve the original evaluation")
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    if not records:
        parser.error("Predictions file has no images")
    missed_attributes = Counter()
    total_attributes = Counter()
    missed_ripeness = Counter()
    total_ripeness = Counter()
    rows = []
    for record in records:
        truth, detections = record["ground_truth"], record["detections"]
        matches, missed, extra = match_boxes(truth, detections)
        for fruit in truth:
            label = fruit.get("source_ripeness", "unknown")
            total_ripeness[label] += 1
            total_attributes["occluded" if fruit.get("occluded") else "not_occluded"] += 1
        for index in missed:
            fruit = truth[index]
            missed_ripeness[fruit.get("source_ripeness", "unknown")] += 1
            missed_attributes["occluded" if fruit.get("occluded") else "not_occluded"] += 1
        signed = len(detections) - len(truth)
        rows.append({"file_name": record["file_name"], "labeled_fruit": len(truth),
                     "detections": len(detections), "signed_count_error": signed,
                     "absolute_count_error": abs(signed), "matched_iou_50": len(matches),
                     "missed_iou_50": len(missed), "extra_iou_50": len(extra)})
    rows.sort(key=lambda row: (-row["absolute_count_error"], row["file_name"]))
    with (args.evaluation / "worst_images.csv").open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    analysis = {
        "images": len(rows),
        "images_overcounted": sum(row["signed_count_error"] > 0 for row in rows),
        "images_undercounted": sum(row["signed_count_error"] < 0 for row in rows),
        "images_exact_count": sum(row["signed_count_error"] == 0 for row in rows),
        "missed_by_occlusion": {key: {"missed": missed_attributes[key], "total": total_attributes[key]}
                                for key in ("occluded", "not_occluded")},
        "missed_by_source_ripeness": {key: {"missed": missed_ripeness[key], "total": total_ripeness[key]}
                                       for key in sorted(total_ripeness)},
        "note": "Box misses use one-to-one IoU 0.50 matching against source labels; some visible fruit may lack source boxes.",
    }
    (args.evaluation / "error_analysis.json").write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
