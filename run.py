"""Run a pretrained detector on the Laboro Tomato held-out images."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

from inspect_dataset import load_split


MODEL_IDS = {
    "grounding-dino": "IDEA-Research/grounding-dino-tiny",
    "fruit-detr": "MohamedKhayat/fruit-detector-detr-50",
}


def box_iou(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    overlap = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - overlap
    return overlap / union if union else 0.0


def remove_duplicate_boxes(detections: list[dict], iou_threshold: float) -> list[dict]:
    """Keep the strongest box when prompt words describe the same region."""
    kept = []
    for detection in sorted(detections, key=lambda item: item["score"], reverse=True):
        if all(box_iou(detection["box"], other["box"]) < iou_threshold for other in kept):
            kept.append(detection)
    return kept


def load_model(kind: str, device: str, revision: str | None = None):
    import torch
    from transformers import (
        AutoImageProcessor,
        AutoModelForObjectDetection,
        AutoModelForZeroShotObjectDetection,
        AutoProcessor,
    )

    model_id = MODEL_IDS[kind]
    if kind == "grounding-dino":
        processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, revision=revision)
    else:
        processor = AutoImageProcessor.from_pretrained(model_id, revision=revision)
        model = AutoModelForObjectDetection.from_pretrained(model_id, revision=revision)
    model = model.to(device).eval()
    return torch, processor, model


def detect(kind, image, processor, model, torch, device, threshold, text_threshold, prompt, tomato_only=True):
    if kind == "grounding-dino":
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        result = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=threshold,
            text_threshold=text_threshold,
            target_sizes=[(image.height, image.width)],
        )[0]
        labels = result.get("text_labels", result.get("labels", []))
        return [
            {"label": str(label), "score": round(float(score), 4), "box": [round(float(x), 2) for x in box.tolist()]}
            for box, score, label in zip(result["boxes"], result["scores"], labels)
        ]

    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    result = processor.post_process_object_detection(
        outputs,
        threshold=threshold,
        target_sizes=torch.tensor([[image.height, image.width]]),
    )[0]
    detections = []
    for box, score, label_id in zip(result["boxes"], result["scores"], result["labels"]):
        label = str(model.config.id2label[int(label_id)])
        if tomato_only and label.casefold() != "tomato":
            continue
        detections.append({
            "label": label,
            "score": round(float(score), 4),
            "box": [round(float(x), 2) for x in box.tolist()],
        })
    return detections


def save_preview(image: Image.Image, detections: list[dict], destination: Path) -> None:
    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    for detection in detections:
        box = detection["box"]
        draw.rectangle(box, outline="red", width=5)
        caption = f"{detection['label']} {detection['score']:.2f}"
        draw.text((box[0], max(0, box[1] - 15)), caption, fill="red")
    preview.save(destination, quality=90)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODEL_IDS)
    parser.add_argument("--data-root", type=Path, default=Path("data/laboro_tomato"))
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--limit", type=int, default=10, help="Number of images; use 0 for all")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--text-threshold", type=float, default=0.25, help="Grounding DINO only")
    parser.add_argument("--nms-iou", type=float, default=0.7, help="Grounding DINO duplicate-box overlap cutoff")
    parser.add_argument("--prompt", default="a cluster of tomatoes.", help="Grounding DINO only")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.limit < 0 or any(not 0 <= value <= 1 for value in (args.threshold, args.text_threshold)) or not 0 < args.nms_iou <= 1:
        parser.error("limit must be nonnegative; thresholds must be 0–1 and NMS IoU must be above 0")

    data, image_items = load_split(args.data_root, args.split)
    image_items.sort(key=lambda pair: str(pair[0]["file_name"]))
    if args.limit:
        image_items = image_items[: args.limit]
    if not image_items:
        parser.error("No images found in the selected split")

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but PyTorch cannot see a CUDA GPU")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    print(f"Loading {MODEL_IDS[args.model]} on {device}. First run downloads model weights.")
    torch, processor, model = load_model(args.model, device)

    output = args.output or Path("outputs") / args.model
    previews = output / "previews"
    previews.mkdir(parents=True, exist_ok=True)
    ground_truth = Counter(
        item["image_id"] for item in data["annotations"] if not item.get("iscrowd", 0)
    )
    rows = []
    predictions_path = output / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as predictions_file:
        for index, (item, path) in enumerate(image_items, start=1):
            with Image.open(path) as source:
                image = source.convert("RGB")
            detections = detect(
                args.model, image, processor, model, torch, device,
                args.threshold, args.text_threshold, args.prompt,
            )
            if args.model == "grounding-dino":
                detections = remove_duplicate_boxes(detections, args.nms_iou)
            preview_name = f"{item['id']}_{Path(item['file_name']).stem}.jpg"
            save_preview(image, detections, previews / preview_name)
            truth_count = ground_truth[item["id"]]
            row = {
                "image_id": item["id"],
                "file_name": item["file_name"],
                "ground_truth_fruit_count": truth_count,
                "predicted_box_count": len(detections),
                "absolute_fruit_count_error": abs(len(detections) - truth_count)
                if args.model == "fruit-detr" else "",
                "preview": str(previews / preview_name),
            }
            rows.append(row)
            predictions_file.write(json.dumps({**row, "detections": detections}) + "\n")
            print(f"[{index}/{len(image_items)}] {path.name}: {len(detections)} boxes")

    with (output / "counts.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "model": MODEL_IDS[args.model],
        "split": args.split,
        "images_processed": len(rows),
        "threshold": args.threshold,
        "duplicate_box_iou_cutoff": args.nms_iou if args.model == "grounding-dino" else None,
        "prompt": args.prompt if args.model == "grounding-dino" else None,
        "measurement": "candidate cluster boxes; no cluster ground truth"
        if args.model == "grounding-dino" else "tomato fruit boxes versus Laboro fruit instances",
        "mean_absolute_fruit_count_error": round(
            sum(int(row["absolute_fruit_count_error"]) for row in rows) / len(rows), 3
        ) if args.model == "fruit-detr" else None,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved results in {output.resolve()}")
    if args.model == "grounding-dino":
        print("Laboro has fruit labels only; visually review boxes before treating them as clusters.")
    else:
        print(f"Mean absolute fruit-count error: {summary['mean_absolute_fruit_count_error']}")
        print("Fruit DETR detects individual tomatoes, not clusters.")


if __name__ == "__main__":
    main()
