"""Evaluate the existing fruit DETR checkpoint on AgRob Pascal VOC annotations."""

import argparse
import csv
import hashlib
import json
import statistics
import warnings
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw

from run import MODEL_IDS, box_iou, detect, load_model


DATA_ROOT = Path("data/agrob/Dataset-Greenhouse_Tomato_AgRob")
ARCHIVE = Path("data/agrob/Dataset-Greenhouse_Tomato_AgRob.zip")
EXPECTED_ARCHIVE_MD5 = "890666716924415720f073b06a9a02a3"
MODEL_REVISION = "19d09855a284cf040460c1ac39fe994e400cdf47"
RIPENESS = {"unriped", "breaking", "reddish", "riped"}


def sequence(stem: str) -> str:
    if stem.startswith("tomate_barroselas_"):
        return "pilot"
    if stem.startswith("tomates_"):
        return "test"
    raise ValueError(f"Unknown AgRob capture sequence: {stem}")


def load_images(root: Path, split: str) -> list[dict]:
    annotations_dir = root / "Annotations"
    image_dir = root / "JPEGImages"
    if not annotations_dir.is_dir() or not image_dir.is_dir():
        raise FileNotFoundError(f"Expected Annotations/ and JPEGImages/ under {root}")
    items = []
    for xml_path in sorted(annotations_dir.glob("*.xml")):
        group = sequence(xml_path.stem)
        if split != "all" and group != split:
            continue
        annotation = ET.parse(xml_path).getroot()
        image_name = Path(annotation.findtext("filename", "")).name
        if image_name != xml_path.with_suffix(".jpg").name:
            raise ValueError(f"Image/annotation name mismatch: {xml_path}")
        image_path = image_dir / image_name
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        width = int(annotation.findtext("size/width"))
        height = int(annotation.findtext("size/height"))
        fruit = []
        for obj in annotation.findall("object"):
            label = obj.findtext("name", "").strip().lower()
            if label not in RIPENESS:
                raise ValueError(f"Unexpected label {label!r} in {xml_path}")
            box = [float(obj.findtext(f"bndbox/{side}")) for side in ("xmin", "ymin", "xmax", "ymax")]
            if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
                raise ValueError(f"Invalid box {box} in {xml_path}")
            fruit.append({
                "label": label,
                "box": box,
                "occluded": obj.findtext("occluded", "0") == "1",
                "truncated": obj.findtext("truncated", "0") == "1",
                "difficult": obj.findtext("difficult", "0") == "1",
            })
        items.append({"image": image_path, "sequence": group, "size": (width, height), "fruit": fruit})
    if not items:
        raise ValueError(f"No AgRob images found for split {split!r}")
    return items


def match_boxes(truth: list[dict], predicted: list[dict], minimum_iou: float = 0.5):
    """One-to-one, descending-confidence matching for object-level review."""
    remaining = set(range(len(truth)))
    matches = []
    extra = []
    for index, detection in sorted(enumerate(predicted), key=lambda pair: pair[1]["score"], reverse=True):
        candidates = [(box_iou(detection["box"], truth[i]["box"]), i) for i in remaining]
        if candidates:
            overlap, truth_index = max(candidates)
            if overlap >= minimum_iou:
                matches.append((truth_index, index, overlap))
                remaining.remove(truth_index)
                continue
        extra.append(index)
    return matches, sorted(remaining), extra


def save_preview(image: Image.Image, truth: list[dict], predicted: list[dict], path: Path) -> None:
    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    for fruit in truth:
        draw.rectangle(fruit["box"], outline="lime", width=3)
    for detection in predicted:
        draw.rectangle(detection["box"], outline="red", width=3)
    preview.save(path, quality=88)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--split", choices=("pilot", "test", "all"), default="pilot")
    parser.add_argument("--limit", type=int, default=0, help="Sorted image count; 0 runs the full split")
    parser.add_argument("--stride", type=int, default=1, help="Take every Nth sorted image for a spread-out pilot")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--label-filter", choices=("tomato", "any-fruit"), default="tomato")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, required=True, help="New output directory for this run")
    args = parser.parse_args()
    if args.limit < 0 or args.stride < 1 or not 0 <= args.threshold <= 1:
        parser.error("limit must be nonnegative, stride positive, and threshold 0–1")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}; choose a fresh directory")
    archive_md5 = hashlib.md5(ARCHIVE.read_bytes()).hexdigest() if ARCHIVE.is_file() else None
    if archive_md5 is not None and archive_md5 != EXPECTED_ARCHIVE_MD5:
        raise ValueError(f"AgRob archive checksum mismatch: {archive_md5}")

    items = load_images(args.data_root, args.split)
    items = items[::args.stride]
    if args.limit:
        items = items[: args.limit]
    import torch
    import transformers

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but PyTorch cannot see a CUDA GPU")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    print(f"Loading {MODEL_IDS['fruit-detr']} on {device} for {len(items)} {args.split} images", flush=True)
    # timm's nested backbone initialization emits a no-op copy warning. The
    # checkpoint tensors were independently checked against the loaded model.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="for .* copying from a non-meta parameter")
        torch, processor, model = load_model("fruit-detr", device, revision=MODEL_REVISION)
    args.output.mkdir(parents=True)
    preview_dir = args.output / "previews"
    preview_dir.mkdir()

    rows = []
    missed_by_label = Counter()
    total_by_label = Counter()
    missed_by_attribute = Counter()
    total_by_attribute = Counter()
    with (args.output / "predictions.jsonl").open("w", encoding="utf-8") as pred_file:
        for number, item in enumerate(items, 1):
            with Image.open(item["image"]) as source:
                image = source.convert("RGB")
            if image.size != item["size"]:
                raise ValueError(f"Image dimensions disagree with annotation: {item['image']}")
            predicted = detect("fruit-detr", image, processor, model, torch, device, args.threshold, 0.0, "",
                               tomato_only=args.label_filter == "tomato")
            truth = item["fruit"]
            matches, missed, extra = match_boxes(truth, predicted)
            for fruit in truth:
                total_by_label[fruit["label"]] += 1
                for name in ("occluded", "truncated", "difficult"):
                    if fruit[name]:
                        total_by_attribute[name] += 1
            for index in missed:
                fruit = truth[index]
                missed_by_label[fruit["label"]] += 1
                for name in ("occluded", "truncated", "difficult"):
                    if fruit[name]:
                        missed_by_attribute[name] += 1
            preview_name = f"{item['image'].stem}.jpg"
            save_preview(image, truth, predicted, preview_dir / preview_name)
            signed_error = len(predicted) - len(truth)
            row = {
                "file_name": item["image"].name,
                "sequence": item["sequence"],
                "ground_truth_fruit_count": len(truth),
                "predicted_fruit_count": len(predicted),
                "signed_count_error": signed_error,
                "absolute_count_error": abs(signed_error),
                "matched_fruit_iou_50": len(matches),
                "missed_fruit_iou_50": len(missed),
                "extra_boxes_iou_50": len(extra),
                "preview": str(preview_dir / preview_name),
            }
            rows.append(row)
            pred_file.write(json.dumps({**row, "ground_truth": truth, "detections": predicted,
                                        "missed_ground_truth_indices": missed,
                                        "extra_detection_indices": extra}) + "\n")
            print(f"[{number}/{len(items)}] {item['image'].name}: {len(predicted)} / {len(truth)}", flush=True)

    with (args.output / "counts.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    misses = sum(row["missed_fruit_iou_50"] for row in rows)
    extras = sum(row["extra_boxes_iou_50"] for row in rows)
    matched = sum(row["matched_fruit_iou_50"] for row in rows)
    summary = {
        "dataset": "AgRobTomato (Zenodo 5596799)",
        "archive_md5": archive_md5,
        "model": MODEL_IDS["fruit-detr"],
        "model_revision": getattr(model.config, "_commit_hash", None),
        "split": args.split,
        "split_rule": "tomate_barroselas_* = pilot; tomates_* = held-out test",
        "images_processed": len(rows),
        "limited_run": args.limit != 0 or args.stride != 1,
        "stride": args.stride,
        "threshold": args.threshold,
        "label_filter": args.label_filter,
        "iou_match_threshold": 0.5,
        "device": device,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "ground_truth_fruit": sum(row["ground_truth_fruit_count"] for row in rows),
        "predicted_fruit": sum(row["predicted_fruit_count"] for row in rows),
        "mean_absolute_count_error": round(statistics.mean(row["absolute_count_error"] for row in rows), 3),
        "median_absolute_count_error": statistics.median(row["absolute_count_error"] for row in rows),
        "mean_signed_count_error": round(statistics.mean(row["signed_count_error"] for row in rows), 3),
        "exact_count_images": sum(row["absolute_count_error"] == 0 for row in rows),
        "matched_fruit_iou_50": matched,
        "missed_fruit_iou_50": misses,
        "extra_boxes_iou_50": extras,
        "box_recall_iou_50": round(matched / (matched + misses), 3) if matched + misses else None,
        "box_precision_iou_50": round(matched / (matched + extras), 3) if matched + extras else None,
        "missed_by_ripeness": {label: {"missed": missed_by_label[label], "total": total_by_label[label]}
                               for label in sorted(total_by_label)},
        "missed_by_attribute": {name: {"missed": missed_by_attribute[name], "total": total_by_attribute[name]}
                                for name in ("occluded", "truncated", "difficult")},
        "worst_count_images": [row["file_name"] for row in sorted(rows, key=lambda r: r["absolute_count_error"], reverse=True)[:10]],
        "notes": "Counts are per image; overlapping video frames may show the same fruit. AgRob has no truss labels.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved results in {args.output.resolve()}", flush=True)
    print(f"Mean absolute count error: {summary['mean_absolute_count_error']}", flush=True)


if __name__ == "__main__":
    main()
