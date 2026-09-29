"""Score Grounding DINO truss boxes against the manually labeled pilot."""

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw

from run import box_iou, detect, load_model, remove_duplicate_boxes


def match_boxes(predictions: list[dict], truth: list[list[float]], iou_cutoff: float) -> tuple[int, int, int]:
    remaining = set(range(len(truth)))
    matched = 0
    for prediction in sorted(predictions, key=lambda item: item["score"], reverse=True):
        if not remaining:
            break
        best = max(remaining, key=lambda index: box_iou(prediction["box"], truth[index]))
        if box_iou(prediction["box"], truth[best]) >= iou_cutoff:
            remaining.remove(best)
            matched += 1
    return matched, len(predictions) - matched, len(truth) - matched


def draw_preview(image: Image.Image, truth: list[list[float]], predictions: list[dict], path: Path) -> None:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for box in truth:
        draw.rectangle(box, outline="lime", width=5)
        draw.text((box[0], max(0, box[1] - 16)), "manual truss", fill="lime")
    for item in predictions:
        box = item["box"]
        draw.rectangle(box, outline="red", width=4)
        draw.text((box[0], min(image.height - 16, box[3] + 2)), f"DINO {item['score']:.2f}", fill="red")
    canvas.save(path, quality=90)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/truss_pilot"))
    parser.add_argument("--split", choices=("train", "valid", "test"), default="valid")
    parser.add_argument("--prompt", default="a tomato truss.")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=Path("outputs/truss-evaluation"))
    args = parser.parse_args()
    if not all(0 <= value <= 1 for value in (args.threshold, args.text_threshold, args.match_iou)) or not 0 < args.nms_iou <= 1:
        parser.error("Thresholds and IoU cutoffs must be between 0 and 1")
    split_dir = args.data_root / args.split
    annotation_file = split_dir / "_annotations.coco.json"
    if not annotation_file.is_file():
        parser.error(f"Missing {annotation_file}; export labels from CVAT and run import_cvat_truss.py first")
    coco = json.loads(annotation_file.read_text(encoding="utf-8"))
    truth_by_image = {item["id"]: [] for item in coco["images"]}
    for annotation in coco["annotations"]:
        x, y, width, height = annotation["bbox"]
        truth_by_image[annotation["image_id"]].append([x, y, x + width, y + height])

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested, but PyTorch cannot see a CUDA GPU")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    torch, processor, model = load_model("grounding-dino", device)
    previews = args.output / "previews"
    previews.mkdir(parents=True, exist_ok=True)
    rows = []
    prediction_records = []
    for image_item in sorted(coco["images"], key=lambda item: item["file_name"]):
        name = Path(image_item["file_name"]).name
        with Image.open(split_dir / name) as source:
            image = source.convert("RGB")
        predictions = detect("grounding-dino", image, processor, model, torch, device, args.threshold, args.text_threshold, args.prompt)
        predictions = remove_duplicate_boxes(predictions, args.nms_iou)
        truth = truth_by_image[image_item["id"]]
        tp, fp, fn = match_boxes(predictions, truth, args.match_iou)
        draw_preview(image, truth, predictions, previews / name)
        rows.append({"file_name": name, "true_trusses": len(truth), "predicted_boxes": len(predictions), "absolute_count_error": abs(len(predictions) - len(truth)), "matched": tp, "false_boxes": fp, "missed_trusses": fn})
        prediction_records.append({"file_name": name, "predictions": predictions})
        print(f"{name}: true={len(truth)}, predicted={len(predictions)}, matched={tp}")
    with (args.output / "counts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "predictions.json").write_text(json.dumps(prediction_records, indent=2) + "\n", encoding="utf-8")
    tp = sum(row["matched"] for row in rows)
    fp = sum(row["false_boxes"] for row in rows)
    fn = sum(row["missed_trusses"] for row in rows)
    summary = {
        "split": args.split,
        "prompt": args.prompt,
        "threshold": args.threshold,
        "text_threshold": args.text_threshold,
        "nms_iou": args.nms_iou,
        "match_iou": args.match_iou,
        "images": len(rows),
        "mean_absolute_truss_count_error": sum(row["absolute_count_error"] for row in rows) / len(rows),
        "exact_count_fraction": sum(row["absolute_count_error"] == 0 for row in rows) / len(rows),
        "box_precision": tp / (tp + fp) if tp + fp else 0,
        "box_recall": tp / (tp + fn) if tp + fn else 0,
        "matched_boxes": tp,
        "false_boxes": fp,
        "missed_trusses": fn,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
